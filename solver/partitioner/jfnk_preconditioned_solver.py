"""
jfnk_preconditioned_solver.py — JFNK with V11 block-Thomas preconditioner.

The preconditioner M^{-1} is V11's linear solve structure:
  - 2x2 block Thomas for pressure-void (captures acoustic coupling)
  - Diagonal momentum with implicit friction + beta_eff coupling to dp
  - Diagonal energy with CFL correction (1 + CFL)

Cost per GMRES iteration: O(N) arithmetic, NO bridge evaluation.
Expected GMRES iterations: 3-5 (vs 30-50 unpreconditioned).
Total cost per timestep: 1 (V11 warm start) + K_newton * K_gmres (JVPs)
                       ≈ 1 + 3*4 = 13 bridge evaluations.

Design: solver-architect agent, 2026-03-30.
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

from .codegen.equation_bridge import OMEquationBridge
from .jfnk_solver import _pack, _unpack, _SP, _SH


class PreconditionedJFNKSolver:
    """JFNK with V11 block-Thomas preconditioner.

    The preconditioner captures V11's operator structure:
    1. Pressure-void: 2x2 block tridiagonal (A11, A12, A21, A22 + face coupling)
    2. Momentum: diagonal (1+sigma) with beta_eff coupling to dp
    3. Energy: diagonal (1 + CFL_l/v)

    This block lower-triangular structure matches V11's solve order:
    pressure-void first, then momentum uses dp, then energy independent.
    GMRES handles the coupling that V11 misses (energy-pressure feedback).
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 newton_tol=1e-3, newton_maxiter=5,
                 gmres_maxiter=10, gmres_restart=10,
                 fd_epsilon=1e-7,
                 tau_mix=4.5e-4, use_isentropic_a11=True):
        self.bridge = bridge
        self.spec = spec
        self.N = bridge.N

        self.newton_tol = newton_tol
        self.newton_maxiter = newton_maxiter
        self.gmres_maxiter = gmres_maxiter
        self.gmres_restart = gmres_restart
        self.fd_epsilon = fd_epsilon
        self.tau_mix = tau_mix
        self.use_isentropic_a11 = use_isentropic_a11

        # Geometry
        self.dx = spec.dx
        self.A_flow = spec.A_flow
        self.D_h = spec.D_h
        self.f_D = spec.f_D
        self.V_cell = spec.V_cell

        # BCs
        self.inlet_closed = getattr(spec, 'inlet_closed', True)
        self.p_out = getattr(spec, 'p_out', 101325.0) or 101325.0

        # Critical flow
        self.use_critical_flow = getattr(spec, 'use_critical_flow', False)
        if not self.use_critical_flow and bridge.has('mdot_crit'):
            self.use_critical_flow = True

        # Set bridge parameters
        bridge.set_params_from_spec(spec, es=es)

        # V11 for warm start
        from .bridge_5eq_solver_v11_a12mod import BridgeDriftFluxSolver
        self.v11 = BridgeDriftFluxSolver(
            bridge, spec, es=es,
            tau_mix=tau_mix, use_isentropic_a11=use_isentropic_a11)

        # Old-state storage
        self._rho_m_old = None
        self._arv_old = None
        self._m_l_old = None
        self._m_v_old = None

        # Preconditioner coefficients (set by _build_preconditioner)
        self._pc_a = None  # lower 2x2 blocks
        self._pc_b = None  # diagonal 2x2 blocks
        self._pc_c = None  # upper 2x2 blocks
        self._pc_beta_eff = None
        self._pc_sigma = None
        self._pc_diag_hl = None  # energy diagonal for liquid
        self._pc_diag_hv = None  # energy diagonal for vapour

        # Time
        self.time = 0.0

        # Diagnostics
        self.newton_iters = 0
        self.gmres_iters = []
        self.F_norms = []
        self.converged = False
        self._n_bridge_evals = 0

    # ================================================================
    # Bridge evaluation and residual (reuse from jfnk_solver)
    # ================================================================

    def _eval(self, p, alpha, h_l, h_v, mdot, t):
        """Evaluate bridge physics at given state."""
        self.bridge.set_time(t)
        self.bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v, mdot=mdot)
        self.bridge.evaluate()
        self._n_bridge_evals += 1

        N = self.N
        rho_l = self.bridge.get('rho_l').copy()
        rho_v = self.bridge.get('rho_v').copy()

        d = {
            'rho_l': rho_l, 'rho_v': rho_v,
            'rho_m': (1 - alpha) * rho_l + alpha * rho_v,
            'rho_face': self.bridge.get('rho_face').copy(),
            'Gamma': self.bridge.get('Gamma').copy(),
            'q_i_l': self.bridge.get('q_i_l').copy(),
            'q_i_v': self.bridge.get('q_i_v').copy(),
            'h_sat_l': self.bridge.get('h_sat_l').copy(),
            'h_sat_v': self.bridge.get('h_sat_v').copy(),
            'Phi2': (self.bridge.get('Phi2').copy()
                     if self.bridge.has('Phi2') else np.ones(N + 1)),
        }
        if self.bridge.has('mdot_v') and self.bridge.has('mdot_l'):
            d['mdot_v'] = self.bridge.get('mdot_v').copy()
            d['mdot_l'] = self.bridge.get('mdot_l').copy()
            d['has_df'] = True
        else:
            d['has_df'] = False
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            arr = self.bridge.get('mdot_crit')
            d['mdot_crit'] = arr[0] if len(arr) > 0 else 1e10
        else:
            d['mdot_crit'] = 1e10
        return d

    def _residual_phys(self, p, al, hl, hv, m,
                       p0, al0, hl0, hv0, m0, dt):
        """Conservation equation residual (identical to jfnk_solver)."""
        # Import the implementation from jfnk_solver to avoid duplication
        from .jfnk_solver import JFNKSolver
        # Temporarily use the parent's method by rebinding
        tmp = JFNKSolver.__new__(JFNKSolver)
        tmp.bridge = self.bridge
        tmp.N = self.N
        tmp.V_cell = self.V_cell
        tmp.A_flow = self.A_flow
        tmp.dx = self.dx
        tmp.D_h = self.D_h
        tmp.f_D = self.f_D
        tmp._K_geom = self.f_D * self.dx / (2 * self.D_h)
        tmp._beta_over_dt = self.A_flow / self.dx
        tmp.inlet_closed = self.inlet_closed
        tmp.p_out = self.p_out
        tmp.use_critical_flow = self.use_critical_flow
        tmp._rho_m_old = self._rho_m_old
        tmp._arv_old = self._arv_old
        tmp._m_l_old = self._m_l_old
        tmp._m_v_old = self._m_v_old
        tmp.time = self.time
        tmp._n_bridge_evals = self._n_bridge_evals
        result = tmp._residual_phys(p, al, hl, hv, m, p0, al0, hl0, hv0, m0, dt)
        self._n_bridge_evals = tmp._n_bridge_evals
        return result

    def _residual(self, xs, xs0, dt):
        """Scaled residual: physical residual with energy blocks / _SH."""
        N = self.N
        p, al, hl, hv, m = _unpack(xs, N)
        p0, al0, hl0, hv0, m0 = _unpack(xs0, N)
        F = self._residual_phys(p, al, hl, hv, m, p0, al0, hl0, hv0, m0, dt)
        F[2*N:4*N] /= _SH
        if not np.all(np.isfinite(F)):
            F = np.where(np.isfinite(F), F, np.sign(F) * 1e10)
            F[~np.isfinite(F)] = 1e10
        return F

    def _jvp(self, xs_k, xs0, v, F_k, dt):
        """Jacobian-vector product via finite differences."""
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-20:
            return np.zeros_like(F_k)
        eps = self.fd_epsilon * max(np.linalg.norm(xs_k), 1.0) / v_norm
        F_pert = self._residual(xs_k + eps * v, xs0, dt)
        return (F_pert - F_k) / eps

    # ================================================================
    # Preconditioner
    # ================================================================

    def _build_preconditioner(self, p, alpha, h_l, h_v, mdot, dt):
        """Extract V11 block Thomas coefficients for preconditioner.

        Evaluates bridge once at the given state and stores all coefficients
        needed by _apply_preconditioner. Cost: 1 bridge evaluation.
        """
        N = self.N

        # Evaluate bridge
        self.bridge.set_time(self.time)
        self.bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v, mdot=mdot)
        self.bridge.evaluate()
        self._n_bridge_evals += 1

        rho_face = self.bridge.get('rho_face')
        rho_l = self.bridge.get('rho_l')
        rho_v = self.bridge.get('rho_v')
        Phi2 = (self.bridge.get('Phi2')
                if self.bridge.has('Phi2') else np.ones(N + 1))
        T_l = self.bridge.get('T_l')
        T_sat = self.bridge.get('T_sat_cell')
        Gamma = self.bridge.get('Gamma')
        h_sat_l = self.bridge.get('h_sat_l')
        h_sat_v = self.bridge.get('h_sat_v')

        if self.bridge.has('drho_l_dp') and self.bridge.has('drho_v_dp'):
            drho_l_dp = self.bridge.get('drho_l_dp')
            drho_v_dp = self.bridge.get('drho_v_dp')
        else:
            drho_dp = self.bridge.get('drho_dp')
            drho_l_dp = drho_dp * 0.1
            drho_v_dp = drho_dp * 10.0

        if self.use_isentropic_a11 and self.bridge.has('drho_l_dp_s'):
            drho_l_dp_s = self.bridge.get('drho_l_dp_s')
            drho_v_dp_s = self.bridge.get('drho_v_dp_s')
        else:
            drho_l_dp_s = drho_l_dp
            drho_v_dp_s = drho_v_dp

        # Critical flow
        mdot_crit = 1e10
        outlet_choked = False
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            arr = self.bridge.get('mdot_crit')
            mdot_crit = arr[0] if len(arr) > 0 else 1e10
            outlet_choked = mdot[N] > 0

        # Friction coefficients
        K_geom = self.f_D * self.dx / (2 * self.D_h)
        sigma = np.zeros(N + 1)
        for i in range(N + 1):
            phi2_i = Phi2[min(i, len(Phi2) - 1)]
            rf = rho_face[i]
            if rf > 0.01 and np.isfinite(mdot[i]):
                sigma[i] = (2 * dt * phi2_i * K_geom * abs(mdot[i])
                            / (rf * self.A_flow ** 2))

        beta = dt * self.A_flow / self.dx
        beta_eff = beta / (1.0 + sigma)

        # ---- Assemble 2x2 block tridiagonal ----
        a_blocks = [np.zeros((2, 2)) for _ in range(N)]
        b_blocks = [np.zeros((2, 2)) for _ in range(N)]
        c_blocks = [np.zeros((2, 2)) for _ in range(N)]

        for i in range(N):
            al = alpha[i]
            rv_i = max(rho_v[i], 0.01)
            rl_i = max(rho_l[i], 1.0)

            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_eff[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_eff[i + 1]

            drho_mech = (1 - al) * drho_l_dp_s[i] + al * drho_v_dp_s[i]
            A11 = self.V_cell / dt * drho_mech + bL + bR
            A12 = self.V_cell / dt * (rv_i - rl_i) / (1.0 + self.tau_mix / dt)

            A21 = self.V_cell / dt * al * drho_v_dp[i]
            A22 = self.V_cell / dt * rv_i

            # Gamma linearization
            if Gamma[i] > 0 and T_l[i] > T_sat[i]:
                superheat = max(T_l[i] - T_sat[i], 0.1)
                h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
                dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / rl_i) / h_fg
                dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                A21 += self.V_cell * dGamma_dp

            b_blocks[i] = np.array([[A11, A12], [A21, A22]])
            if i > 0:
                a_blocks[i] = np.array([[-bL, 0.0], [0.0, 0.0]])
            if i < N - 1:
                c_blocks[i] = np.array([[-bR, 0.0], [0.0, 0.0]])

        # ---- Energy diagonals: 1 + CFL ----
        has_df = self.bridge.has('mdot_v') and self.bridge.has('mdot_l')
        diag_hl = np.ones(N)
        diag_hv = np.ones(N)
        for i in range(N):
            al = alpha[i]
            m_l = max((1 - al) * rho_l[i] * self.V_cell, 1e-12)
            m_v = max(al * rho_v[i] * self.V_cell, 1e-12)

            if has_df:
                ml_out = abs(self.bridge.get('mdot_l')[i + 1])
                mv_out = abs(self.bridge.get('mdot_v')[i + 1])
            else:
                ml_out = abs(mdot[i + 1]) * (1 - al)
                mv_out = abs(mdot[i + 1]) * al

            diag_hl[i] = 1.0 + dt * ml_out / m_l
            diag_hv[i] = 1.0 + dt * mv_out / m_v

        # Store all coefficients
        self._pc_a = a_blocks
        self._pc_b = b_blocks
        self._pc_c = c_blocks
        self._pc_beta_eff = beta_eff
        self._pc_sigma = sigma
        self._pc_diag_hl = diag_hl
        self._pc_diag_hv = diag_hv

    @staticmethod
    def _block_thomas_solve(a_blocks, b_blocks, c_blocks, d_vecs):
        """Block Thomas for 2x2 tridiagonal system."""
        N = len(d_vecs)
        Z = np.zeros((2, 2))
        cp = [None] * N
        dp = [None] * N

        try:
            b_inv = np.linalg.inv(b_blocks[0])
        except np.linalg.LinAlgError:
            b_inv = np.eye(2) * 1e-10
        cp[0] = b_inv @ c_blocks[0]
        dp[0] = b_inv @ d_vecs[0]

        for i in range(1, N):
            temp = b_blocks[i] - a_blocks[i] @ cp[i - 1]
            try:
                temp_inv = np.linalg.inv(temp)
            except np.linalg.LinAlgError:
                temp_inv = np.eye(2) * 1e-10
            cp[i] = temp_inv @ c_blocks[i] if i < N - 1 else Z.copy()
            dp[i] = temp_inv @ (d_vecs[i] - a_blocks[i] @ dp[i - 1])

        x = [None] * N
        x[N - 1] = dp[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] @ x[i + 1]
        return x

    def _apply_preconditioner(self, r_scaled):
        """Apply M^{-1} to scaled residual vector.

        Cost: O(N) arithmetic, NO bridge evaluation.

        Block lower-triangular solve:
        1. Pressure-void: block Thomas with (r_mass, r_void) as RHS
        2. Momentum: r_mom with beta_eff coupling to dp from step 1
        3. Energy: diagonal solve with (1 + CFL) diagonal
        """
        N = self.N
        result = np.zeros_like(r_scaled)

        # Unpack residual (in scaled units)
        r_mass = r_scaled[0:N].copy()
        r_void = r_scaled[N:2*N].copy()
        r_hl = r_scaled[2*N:3*N].copy()
        r_hv = r_scaled[3*N:4*N].copy()
        r_mom = r_scaled[4*N:5*N+1].copy()

        # 1. Block Thomas for pressure-void
        # RHS is the JFNK residual for mass and void equations
        d_vecs = [np.array([r_mass[i], r_void[i]]) for i in range(N)]
        x = self._block_thomas_solve(self._pc_a, self._pc_b, self._pc_c,
                                     d_vecs)
        dp_s = np.array([x[i][0] for i in range(N)])  # scaled pressure
        dalpha = np.array([x[i][1] for i in range(N)])

        # 2. Momentum: dmdot from r_mom + beta_eff * dp_gradient
        # dp is in scaled units (dp_s = dp / _SP). beta_eff * dp needs
        # physical dp, but since the residual is in physical units for
        # momentum (kg/s) and the Jacobian dF_mom/dp_s = -beta * _SP,
        # we need: dmdot = (r_mom + beta_eff * _SP * (dp_s[j-1] - dp_s[j]))
        #                    / (1 + sigma)
        dmdot = np.zeros(N + 1)
        dmdot[0] = r_mom[0]  # wall BC passthrough
        for j in range(1, N):
            dp_grad = _SP * (dp_s[j - 1] - dp_s[j])
            dmdot[j] = (r_mom[j] + self._pc_beta_eff[j] * dp_grad
                        ) / (1.0 + self._pc_sigma[j])
        # Outlet
        dp_grad_out = _SP * dp_s[N - 1]  # p[N-1] - p_out, only dp[N-1] varies
        dmdot[N] = (r_mom[N] + self._pc_beta_eff[N] * dp_grad_out
                    ) / (1.0 + self._pc_sigma[N])

        # 3. Energy: diagonal solve
        # The scaled energy residual is F_h / _SH. The Jacobian diagonal
        # in scaled coords is dF_h_scaled / dh_scaled = (1 + CFL).
        # So dh_scaled = r_h_scaled / (1 + CFL).
        dhl_s = r_hl / self._pc_diag_hl
        dhv_s = r_hv / self._pc_diag_hv

        # Pack result
        result[0:N] = dp_s
        result[N:2*N] = dalpha
        result[2*N:3*N] = dhl_s
        result[3*N:4*N] = dhv_s
        result[4*N:5*N+1] = dmdot

        return result

    # ================================================================
    # Physical bounds
    # ================================================================

    def _bounds(self, xs):
        """Enforce physical bounds on scaled state vector."""
        N = self.N
        xs = xs.copy()
        xs[0:N] = np.clip(xs[0:N], 1e4 / _SP, 50e6 / _SP)
        xs[N:2*N] = np.clip(xs[N:2*N], 0.0, 1.0)
        xs[2*N:3*N] = np.clip(xs[2*N:3*N], 1e4 / _SH, 4e6 / _SH)
        xs[3*N:4*N] = np.clip(xs[3*N:4*N], 1e5 / _SH, 4e6 / _SH)
        mask = ~np.isfinite(xs[4*N:])
        xs[4*N:][mask] = 0.0
        return xs

    # ================================================================
    # Main timestepping
    # ================================================================

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One fully-implicit timestep with preconditioned JFNK.

        1. Store old-state densities
        2. V11 warm start → baseline
        3. Build preconditioner at warm-start state (1 bridge eval, amortized)
        4. Newton loop with preconditioned GMRES
        5. Use best result (always >= V11 quality)
        """
        N = self.N
        self._n_bridge_evals = 0

        # ---- 1. Store old-state densities ----
        d0 = self._eval(p, alpha, h_l, h_v, mdot, self.time)
        self._rho_m_old = d0['rho_m'].copy()
        self._arv_old = (alpha * d0['rho_v']).copy()
        self._m_l_old = np.maximum((1 - alpha) * d0['rho_l'] * self.V_cell,
                                   1e-12)
        self._m_v_old = np.maximum(alpha * d0['rho_v'] * self.V_cell, 1e-12)

        x_old = _pack(p.copy(), alpha.copy(), h_l.copy(), h_v.copy(),
                       mdot.copy())

        # ---- 2. V11 warm start ----
        pg = p.copy(); ag = alpha.copy()
        hlg = h_l.copy(); hvg = h_v.copy()
        mg = mdot.copy()
        self.v11.time = self.time
        self.v11.step(pg, ag, hlg, hvg, mg, dt)
        x_v11 = _pack(pg, ag, hlg, hvg, mg)

        # ---- 3. Build preconditioner at V11 warm-start state ----
        # Amortized: V11 already evaluated the bridge at the old state.
        # We re-evaluate at the warm-start state for the preconditioner.
        self._build_preconditioner(pg, ag, hlg, hvg, mg, dt)

        # ---- 4. Newton loop with preconditioned GMRES ----
        x_k = x_v11.copy()
        self.F_norms = []
        self.gmres_iters = []
        self.converged = False

        best_x = x_v11.copy()
        F_v11 = self._residual(x_v11, x_old, dt)
        F_v11_norm = np.linalg.norm(F_v11)
        if not np.isfinite(F_v11_norm):
            F_v11_norm = np.inf
        best_norm = F_v11_norm
        self.F_norms.append(F_v11_norm)

        F_norm_0 = max(F_v11_norm, 1e-14)
        if F_v11_norm < 1e-12:
            self.converged = True

        n_sys = 5 * N + 1
        M_op = LinearOperator((n_sys, n_sys),
                              matvec=self._apply_preconditioner)

        for k in range(self.newton_maxiter):
            if self.converged:
                break

            F_k = self._residual(x_k, x_old, dt) if k > 0 else F_v11
            F_norm = np.linalg.norm(F_k)

            if not np.isfinite(F_norm):
                break

            if k > 0:
                self.F_norms.append(F_norm)
                if F_norm < best_norm:
                    best_norm = F_norm
                    best_x = x_k.copy()

            if F_norm < self.newton_tol * F_norm_0 or F_norm < 1e-12:
                self.converged = True
                best_x = x_k.copy()
                break

            # Preconditioned GMRES: solve J * dx = -F
            _xk = x_k.copy()
            _Fk = F_k.copy()

            def matvec(v, xk=_xk, Fk=_Fk):
                return self._jvp(xk, x_old, v, Fk, dt)

            J_op = LinearOperator((n_sys, n_sys), matvec=matvec)

            gi = [0]

            def _cb(rk):
                gi[0] += 1

            dx, info = gmres(
                J_op, -F_k,
                M=M_op,  # RIGHT preconditioner
                maxiter=self.gmres_maxiter,
                restart=min(self.gmres_restart, n_sys),
                rtol=1e-2,  # loose inner solve
                atol=1e-10,
                callback=_cb,
                callback_type='legacy')
            self.gmres_iters.append(gi[0])

            if not np.all(np.isfinite(dx)):
                break

            # Backtracking line search
            accepted = False
            for step_sz in [1.0, 0.5, 0.25, 0.125]:
                x_try = self._bounds(x_k + step_sz * dx)
                F_try = self._residual(x_try, x_old, dt)
                F_try_norm = np.linalg.norm(F_try)
                if np.isfinite(F_try_norm) and F_try_norm < F_norm:
                    x_k = x_try
                    if F_try_norm < best_norm:
                        best_norm = F_try_norm
                        best_x = x_try.copy()
                    accepted = True
                    break

            if not accepted:
                break

        self.newton_iters = len(self.F_norms) - 1

        # ---- 5. Use best result (always >= V11 quality) ----
        if self.converged:
            p[:], alpha[:], h_l[:], h_v[:], mdot[:] = _unpack(best_x, N)
        else:
            # Fall back to V11 baseline for trajectory consistency
            p[:], alpha[:], h_l[:], h_v[:], mdot[:] = _unpack(x_v11, N)
