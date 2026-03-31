"""
jfnk_solver.py — Jacobian-Free Newton-Krylov solver for 5-eq two-phase flow.

Eliminates operator splitting by solving all 5 conservation equations
simultaneously. Newton's method with GMRES for the linear solve.
Jacobian-vector products via finite differences (matrix-free).

The V11 semi-implicit solver provides the warm start (initial guess).
This breaks the Pareto frontier between onset timing and pressure accuracy
by coupling pressure, void, and energy equations implicitly within each
Newton iteration — no operator splitting, no tau_mix.

State vector: x = [p[0..N-1], alpha[0..N-1], h_l[0..N-1], h_v[0..N-1], mdot[0..N]]
Total unknowns: 5*N + 1

Conservation equations (backward Euler, all evaluated at NEW state):
  Eq 1: mixture mass [kg/s]
  Eq 2: vapour mass  [kg/s]
  Eq 3: liquid energy [J/kg] (h-form)
  Eq 4: vapour energy [J/kg] (h-form)
  Eq 5: momentum      [kg/s]

Architecture:
    Newton iteration (outer, 2-5 per timestep)
      └─ GMRES (inner linear solve, scipy)
           └─ Jacobian-vector product: J*v ≈ (F(x+εv) - F(x)) / ε
      └─ Backtracking line search
"""

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.optimize import root as scipy_root

from .codegen.equation_bridge import OMEquationBridge


# --- State vector packing with scaling for FD accuracy ---
# Without scaling, FD epsilon for alpha (~0.1) would be dominated by p (~7e6).
_SP = 1e6   # pressure scale [Pa]
_SH = 1e6   # enthalpy scale [J/kg]
# alpha and mdot are already O(1), no scaling needed.


def _pack(p, alpha, h_l, h_v, mdot):
    """Pack OPAL state arrays into a single scaled vector."""
    return np.concatenate([p / _SP, alpha, h_l / _SH, h_v / _SH, mdot])


def _unpack(xs, N):
    """Unpack scaled vector into physical state arrays."""
    return (xs[0:N] * _SP,
            xs[N:2*N].copy(),
            xs[2*N:3*N] * _SH,
            xs[3*N:4*N] * _SH,
            xs[4*N:5*N+1].copy())


class JFNKSolver:
    """Jacobian-Free Newton-Krylov solver for 5-equation two-phase flow.

    All physics from OM bridge (Modelica). Solver provides only:
    - Newton iteration (nonlinear solve)
    - GMRES (linear solve, scipy.sparse.linalg)
    - Finite-difference Jacobian-vector products (matrix-free)
    - Backtracking line search
    - V11 warm start (initial guess for Newton)

    The key difference from V11: energy equations use NEW enthalpies for
    advection and source evaluation, not frozen OLD values. This eliminates
    the operator splitting error that creates the Pareto frontier.
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 newton_tol=1e-6, newton_maxiter=10,
                 gmres_maxiter=30, gmres_restart=30,
                 fd_epsilon=1e-7,
                 use_v11_warmstart=True,
                 tau_mix=4.5e-4, use_isentropic_a11=True):
        """
        Args:
            bridge: OMEquationBridge with compiled Modelica model.
            spec: Pipe1DGridSpec (geometry, BCs).
            es: EquationSystem from XML — provides closure parameter values.
            newton_tol: Relative tolerance for Newton convergence.
            newton_maxiter: Maximum Newton iterations per timestep.
            gmres_maxiter: Maximum GMRES iterations per Newton step.
            gmres_restart: GMRES restart parameter.
            fd_epsilon: Base epsilon for finite-difference JVP.
            use_v11_warmstart: Use one V11 step as Newton initial guess.
            tau_mix: V11 tau_mix for warm start (only used if use_v11_warmstart).
            use_isentropic_a11: V11 isentropic flag for warm start.
        """
        self.bridge = bridge
        self.spec = spec
        self.N = bridge.N

        self.newton_tol = newton_tol
        self.newton_maxiter = newton_maxiter
        self.gmres_maxiter = gmres_maxiter
        self.gmres_restart = gmres_restart
        self.fd_epsilon = fd_epsilon

        # Geometry from spec
        self.dx = spec.dx
        self.A_flow = spec.A_flow
        self.D_h = spec.D_h
        self.f_D = spec.f_D
        self.V_cell = spec.V_cell

        # Derived constants
        self._beta_over_dt = self.A_flow / self.dx   # beta = dt * A / dx
        self._K_geom = self.f_D * self.dx / (2 * self.D_h)

        # BCs
        self.inlet_closed = getattr(spec, 'inlet_closed', True)
        self.p_out = getattr(spec, 'p_out', 101325.0) or 101325.0

        # Critical flow
        self.use_critical_flow = getattr(spec, 'use_critical_flow', False)
        if not self.use_critical_flow and bridge.has('mdot_crit'):
            self.use_critical_flow = True

        # Set bridge parameters from spec + XML
        bridge.set_params_from_spec(spec, es=es)

        # V11 for warm start
        self.use_v11_warmstart = use_v11_warmstart
        if use_v11_warmstart:
            from .bridge_5eq_solver_v11_a12mod import BridgeDriftFluxSolver
            self.v11 = BridgeDriftFluxSolver(
                bridge, spec, es=es,
                tau_mix=tau_mix, use_isentropic_a11=use_isentropic_a11)
        else:
            self.v11 = None

        # Old-state storage (set at start of each timestep, constant during Newton)
        self._rho_m_old = None    # mixture density at old state [kg/m³]
        self._arv_old = None      # alpha * rho_v at old state [kg/m³]
        self._m_l_old = None      # (1-α) ρ_l V at old state [kg]
        self._m_v_old = None      # α ρ_v V at old state [kg]

        # Time
        self.time = 0.0

        # Per-step diagnostics
        self.newton_iters = 0
        self.gmres_iters = []
        self.F_norms = []
        self._n_bridge_evals = 0  # bridge evaluations per step

    # ================================================================
    # Bridge evaluation
    # ================================================================

    def _eval(self, p, alpha, h_l, h_v, mdot, t):
        """Evaluate bridge physics at given state. Returns property dict."""
        self.bridge.set_time(t)
        self.bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v, mdot=mdot)
        self.bridge.evaluate()
        self._n_bridge_evals += 1

        N = self.N
        rho_l = self.bridge.get('rho_l').copy()
        rho_v = self.bridge.get('rho_v').copy()

        d = {
            'rho_l': rho_l,
            'rho_v': rho_v,
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

        # Drift-flux phasic mass flows from Modelica
        if self.bridge.has('mdot_v') and self.bridge.has('mdot_l'):
            d['mdot_v'] = self.bridge.get('mdot_v').copy()
            d['mdot_l'] = self.bridge.get('mdot_l').copy()
            d['has_df'] = True
        else:
            d['has_df'] = False

        # Critical flow
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            arr = self.bridge.get('mdot_crit')
            d['mdot_crit'] = arr[0] if len(arr) > 0 else 1e10
        else:
            d['mdot_crit'] = 1e10

        return d

    # ================================================================
    # Residual F(x_new) = 0
    # ================================================================

    def _residual_phys(self, p, al, hl, hv, m,
                       p0, al0, hl0, hv0, m0, dt):
        """Compute conservation equation residual in physical units.

        All physics evaluated at the NEW (candidate) state via bridge.
        Time derivatives use stored OLD-state densities.

        Returns:
            F: array of size 5*N+1, zero when conservation is satisfied.
        """
        N = self.N
        V = self.V_cell
        A = self.A_flow
        K = self._K_geom
        beta = dt * self._beta_over_dt

        d = self._eval(p, al, hl, hv, m, self.time + dt)

        rho_l = d['rho_l']; rho_v = d['rho_v']; rho_m = d['rho_m']
        rho_f = d['rho_face']; Gam = d['Gamma']
        qil = d['q_i_l']; qiv = d['q_i_v']
        h_sat_l = d['h_sat_l']; h_sat_v = d['h_sat_v']
        Phi2 = d['Phi2']; mc = d['mdot_crit']
        has_df = d['has_df']

        F = np.zeros(5 * N + 1)

        # ---- Eq 1: Mixture mass conservation [kg/s] ----
        for i in range(N):
            F[i] = V / dt * (rho_m[i] - self._rho_m_old[i]) - (m[i] - m[i + 1])

        # ---- Eq 2: Vapour mass conservation [kg/s] ----
        for i in range(N):
            arv_new = al[i] * rho_v[i]

            if has_df:
                mvi = d['mdot_v'][i]; mvo = d['mdot_v'][i + 1]
            else:
                ali = al[i - 1] if (i > 0 and m[i] >= 0) else al[i]
                alo = al[i] if m[i + 1] >= 0 else (
                    al[i + 1] if i < N - 1 else al[i])
                mvi = m[i] * ali
                mvo = m[i + 1] * alo

            F[N + i] = (V / dt * (arv_new - self._arv_old[i])
                        - (mvi - mvo) - V * Gam[i])

        # ---- Eq 3: Liquid enthalpy [J/kg] (h-form) ----
        # F = (h_new - h_old) - dt/m_old * (advection + pressure_work + q_i - Gamma*h)
        # Using NEW enthalpies for advection — this is what breaks the Pareto frontier.
        for i in range(N):
            a = al[i]
            ml = self._m_l_old[i]

            # Degenerate case: near-pure vapour → constrain h_l to saturation
            if (1 - a) < 1e-4:
                F[2 * N + i] = hl[i] - h_sat_l[i]
                continue

            if has_df:
                mli = d['mdot_l'][i]; mlo = d['mdot_l'][i + 1]
            else:
                ali = al[i - 1] if (i > 0 and m[i] >= 0) else al[i]
                alo = al[i] if m[i + 1] >= 0 else (
                    al[i + 1] if i < N - 1 else al[i])
                mli = m[i] * (1 - ali)
                mlo = m[i + 1] * (1 - alo)

            # Donor-cell face enthalpies using NEW values
            fin = mli if has_df else m[i]
            fout = mlo if has_df else m[i + 1]
            hin = hl[i - 1] if (i > 0 and fin >= 0) else hl[i]
            hout = hl[i] if fout >= 0 else (
                hl[i + 1] if i < N - 1 else hl[i])

            flux = mli * (hin - hl[i]) - mlo * (hout - hl[i])
            pw = (1 - a) * V * (p[i] - p0[i]) / dt
            qi = qil[i] * V
            phase = -Gam[i] * hl[i] * V

            F[2 * N + i] = (hl[i] - hl0[i]) - dt / ml * (flux + pw + qi + phase)

        # ---- Eq 4: Vapour enthalpy [J/kg] (h-form) ----
        for i in range(N):
            a = al[i]
            mv = self._m_v_old[i]

            if a < 1e-4:
                F[3 * N + i] = hv[i] - h_sat_v[i]
                continue

            if has_df:
                mvi = d['mdot_v'][i]; mvo = d['mdot_v'][i + 1]
            else:
                ali = al[i - 1] if (i > 0 and m[i] >= 0) else al[i]
                alo = al[i] if m[i + 1] >= 0 else (
                    al[i + 1] if i < N - 1 else al[i])
                mvi = m[i] * ali
                mvo = m[i + 1] * alo

            finv = mvi if has_df else m[i]
            foutv = mvo if has_df else m[i + 1]
            hvin = hv[i - 1] if (i > 0 and finv >= 0) else hv[i]
            hvout = hv[i] if foutv >= 0 else (
                hv[i + 1] if i < N - 1 else hv[i])

            fluxv = mvi * (hvin - hv[i]) - mvo * (hvout - hv[i])
            pwv = a * V * (p[i] - p0[i]) / dt
            qiv_val = qiv[i] * V
            phasev = Gam[i] * hv[i] * V

            F[3 * N + i] = (hv[i] - hv0[i]) - dt / mv * (
                fluxv + pwv + qiv_val + phasev)

        # ---- Eq 5: Momentum [kg/s] ----
        # Same discretization as V11: (mdot - mdot_old) = beta*Δp - dt*fric
        # Residual: F = (mdot - mdot_old) - beta*Δp + dt*fric
        # Newton handles friction nonlinearity exactly (no sigma approximation).

        # Face 0: closed wall BC
        F[4 * N] = m[0]

        # Interior faces 1..N-1
        for j in range(1, N):
            rf = max(rho_f[j], 0.01)
            phi2_j = Phi2[min(j, len(Phi2) - 1)]
            fric = phi2_j * K * abs(m[j]) * m[j] / (rf * A ** 2)
            F[4 * N + j] = (m[j] - m0[j]) - beta * (p[j - 1] - p[j]) + dt * fric

        # Outlet face N
        rf_N = max(rho_f[N], 0.01)
        phi2_N = Phi2[min(N, len(Phi2) - 1)]
        fric_N = phi2_N * K * abs(m[N]) * m[N] / (rf_N * A ** 2)
        F[4 * N + N] = ((m[N] - m0[N]) - beta * (p[N - 1] - self.p_out)
                        + dt * fric_N)

        # Critical flow: if choked, enforce mdot = mdot_crit
        if self.use_critical_flow and m[N] > mc and mc < 1e9:
            F[4 * N + N] = m[N] - mc

        return F

    def _residual(self, xs, xs0, dt):
        """Residual wrapper: scaled state → physical → scaled residual.

        Energy residuals (blocks 2,3) are divided by _SH to balance the
        Jacobian. Without this, the h-form residual is O(1e3) J/kg while
        mass/momentum are O(1) kg/s, causing GMRES to ignore mass/momentum.
        With scaling, all Jacobian diagonal entries are O(1) in the scaled
        state space, giving well-conditioned GMRES.
        """
        N = self.N
        p, al, hl, hv, m = _unpack(xs, N)
        p0, al0, hl0, hv0, m0 = _unpack(xs0, N)
        F = self._residual_phys(p, al, hl, hv, m,
                                p0, al0, hl0, hv0, m0, dt)
        # Scale energy residuals: [J/kg] → [MJ/kg] to match state scaling
        F[2*N:4*N] /= _SH
        # NaN protection: if bridge returned NaN, signal with large residual
        if not np.all(np.isfinite(F)):
            F = np.where(np.isfinite(F), F, np.sign(F) * 1e10)
            F[~np.isfinite(F)] = 1e10
        return F

    # ================================================================
    # Jacobian-vector product (matrix-free)
    # ================================================================

    def _jvp(self, xs_k, xs0, v, F_k, dt):
        """J*v ≈ (F(x + εv) - F(x)) / ε via forward finite difference.

        The Jacobian is never formed explicitly. Each JVP costs one
        bridge evaluation (the dominant cost).
        """
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-20:
            return np.zeros_like(F_k)
        # Standard epsilon choice (Dennis & Schnabel):
        # ε = sqrt(machine_eps) * (1 + ||x||) / ||v||
        eps = self.fd_epsilon * max(np.linalg.norm(xs_k), 1.0) / v_norm
        F_pert = self._residual(xs_k + eps * v, xs0, dt)
        return (F_pert - F_k) / eps

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
    # Main timestepping interface
    # ================================================================

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One fully-implicit JFNK timestep. Modifies arrays in-place.

        Strategy: V11 warm start is the guaranteed baseline. Scipy's
        Krylov solver attempts to improve upon it. The result is used if
        it improves the residual (even partial convergence helps). If
        scipy produces NaN or increases residual, V11 baseline is kept.

        1. Evaluate bridge at old state, store densities.
        2. V11 warm start → baseline solution.
        3. Scipy Krylov solve for conservation residual.
        4. Use best result (always >= V11 quality).
        """
        N = self.N
        self._n_bridge_evals = 0

        # ---- 1. Store old-state densities (constant during Newton) ----
        d0 = self._eval(p, alpha, h_l, h_v, mdot, self.time)
        self._rho_m_old = d0['rho_m'].copy()
        self._arv_old = (alpha * d0['rho_v']).copy()
        self._m_l_old = np.maximum((1 - alpha) * d0['rho_l'] * self.V_cell,
                                   1e-12)
        self._m_v_old = np.maximum(alpha * d0['rho_v'] * self.V_cell, 1e-12)

        x_old = _pack(p.copy(), alpha.copy(), h_l.copy(), h_v.copy(),
                       mdot.copy())

        # ---- 2. V11 warm start (baseline — always physical) ----
        if self.v11 is not None:
            pg = p.copy(); ag = alpha.copy()
            hlg = h_l.copy(); hvg = h_v.copy()
            mg = mdot.copy()
            self.v11.time = self.time
            self.v11.step(pg, ag, hlg, hvg, mg, dt)
            x_v11 = _pack(pg, ag, hlg, hvg, mg)
        else:
            x_v11 = x_old.copy()

        # Evaluate V11 baseline residual
        F_v11 = self._residual(x_v11, x_old, dt)
        F_v11_norm = np.linalg.norm(F_v11)
        if not np.isfinite(F_v11_norm):
            F_v11_norm = np.inf

        # ---- 3. Scipy Krylov solve ----
        self.F_norms = [F_v11_norm]
        self.gmres_iters = []
        self.converged = False

        # Track best state (initialized to V11, so output >= V11 quality)
        best_x = x_v11.copy()
        best_norm = F_v11_norm

        def _F(xs):
            r = self._residual(xs, x_old, dt)
            if not np.all(np.isfinite(r)):
                return np.full_like(r, 1e10)
            return r

        try:
            result = scipy_root(
                _F, x0=x_v11,
                method='krylov',
                options={
                    'maxiter': self.newton_maxiter,
                    'fatol': max(self.newton_tol * F_v11_norm, 1e-10),
                    'disp': False,
                })

            F_result = result.fun
            F_result_norm = np.linalg.norm(F_result)

            if (np.all(np.isfinite(F_result)) and
                    np.isfinite(F_result_norm) and
                    F_result_norm < best_norm):
                best_x = result.x.copy()
                best_norm = F_result_norm
                self.converged = result.success

            self.F_norms.append(best_norm)
            self.newton_iters = getattr(result, 'nit', 0)

        except Exception:
            # Scipy failed (e.g., singular Jacobian) — use V11
            self.newton_iters = 0

        # ---- 4. Use best result (always >= V11 quality) ----
        p[:], alpha[:], h_l[:], h_v[:], mdot[:] = _unpack(best_x, N)
