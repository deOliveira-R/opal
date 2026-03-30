"""
bridge_5eq_solver_v13_newton_offdiag.py -- Variant 13: Newton iteration + drift-flux
off-diagonal coupling.

Based on V5 (block Thomas). Adds TWO improvements:

1. **Newton iteration** (2-4 sweeps per timestep): After the block Thomas solve,
   re-evaluate the bridge at the updated (p, alpha) state and re-solve. This
   reduces linearization error from the semi-implicit scheme. The first iteration
   uses phasic drho_dp for A11 (onset-friendly), subsequent iterations use h_mix
   drho_dp from the re-evaluated bridge (correct wave speed).

2. **Drift-flux off-diagonals**: The (1,0) entries of the off-diagonal blocks are
   populated with the vapor flux response to neighbor pressure changes. When a
   neighbor's pressure changes, the mixture momentum at the shared face changes
   (via beta_eff), and the drift-flux split allocates a fraction to the vapor
   phase:

       d(mdot_v)/dp_neighbor = alpha_face * rho_v_face * beta_eff / rho_face

   This couples vapor transport to the pressure solve, improving void propagation.

The block system per cell is:
    [A11  A12] [delta_p    ]   [R1]
    [A21  A22] [delta_alpha] = [R2]

Row 1: mixture mass conservation (pressure equation)
Row 2: vapour mass conservation (void equation) -- now with phasic off-diagonals

Deltas are ALWAYS relative to start-of-step (p_old, alpha_old), not the previous
Newton iteration. Friction is frozen at the initial evaluation (no re-evaluation
of sigma/beta_eff). Energy update happens once after the Newton loop completes.

ALL physics evaluation (properties, closures, friction multiplier, interfacial
transfer) from OM-generated C via the equation bridge. The solver provides
ONLY the semi-implicit numerical method.

State variables: p[N], alpha[N], h_l[N], h_v[N], mdot[N+1]
Bridge provides: rho_l, rho_v, rho_m, rho_face, drho_dp, drho_l_dp, drho_v_dp,
    Gamma, q_i_l, q_i_v, Phi2, T_l, T_sat_cell, h_sat_l, h_sat_v, h_mix,
    alpha_face, rho_v_face (for off-diagonals)
"""

import numpy as np

from .codegen.equation_bridge import OMEquationBridge


class BridgeDriftFluxSolver:
    """5-equation drift-flux solver with Newton-iterated block Thomas and
    drift-flux off-diagonal coupling.

    Extends V5 with:
    - Newton iteration: re-evaluate bridge and re-solve 2-4 times per step
    - Drift-flux off-diagonals: vapor flux coupling in (1,0) off-diagonal entries

    Physics (properties, closures, friction multiplier, interfacial transfer)
    from OM-generated C. Numerics (block tridiagonal solve, momentum, phasic
    energy) from Python.
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 reconstruction='donor_cell', use_block_coupling=False,
                 corrector_steps=0, n_newton=3):
        """
        Args:
            bridge: OMEquationBridge with compiled Modelica model
            spec: Pipe1DGridSpec (geometry, BCs)
            es: EquationSystem from XML -- pass this to load ALL parameter values
                from Modelica (H_i, C_0, etc.). Without es, only geometry is set.
            reconstruction: 'donor_cell' (1st order) or 'muscl' (2nd order, minmod TVD)
            use_block_coupling: accepted for API compatibility but ignored --
                block coupling is always active in V13.
            corrector_steps: accepted for API compatibility but ignored --
                Newton iteration replaces the corrector loop.
            n_newton: number of Newton iterations per timestep (default 3).
        """
        self.reconstruction = reconstruction
        # V13: block coupling is always active; Newton replaces corrector
        self.use_block_coupling = True
        self.corrector_steps = 0
        self.n_newton = n_newton
        self.bridge = bridge
        self.N = bridge.N
        self.spec = spec

        # Geometry from spec
        self.dx = spec.dx
        self.A_flow = spec.A_flow
        self.D_h = spec.D_h
        self.f_D = spec.f_D
        self.V_cell = spec.V_cell

        # BC flags
        self.inlet_closed = getattr(spec, 'inlet_closed', True)
        self.p_out = getattr(spec, 'p_out', 101325.0) or 101325.0

        # Critical flow: use if bridge provides mdot_crit (Modelica computes it)
        self.use_critical_flow = getattr(spec, 'use_critical_flow', False)
        if not self.use_critical_flow and bridge.has('mdot_crit'):
            # Bridge has mdot_crit from Modelica -> critical flow is active
            self.use_critical_flow = True

        # Set bridge parameters from spec + XML (es provides closure params)
        bridge.set_params_from_spec(spec, es=es)

    @staticmethod
    def _minmod(a, b):
        """Minmod slope limiter: TVD, most diffusive limiter."""
        if a * b <= 0:
            return 0.0
        return a if abs(a) < abs(b) else b

    @staticmethod
    def _muscl_face(field, i, N, mdot_face, bc_left, bc_right):
        """MUSCL-reconstructed face value with minmod limiter.

        For face i (between cell i-1 and cell i), reconstructs a second-order
        value using the 4-point stencil [i-2, i-1, i, i+1] with donor-cell
        upwinding and minmod slope limiting.
        """
        if mdot_face >= 0:
            # Upwind from left: reconstruct at right edge of cell i-1
            if i <= 0:
                return bc_left
            L = field[i - 1]
            # Slopes
            dL = (L - (field[i - 2] if i >= 2 else bc_left))
            dR = (field[i] if i < N else bc_right) - L
            # Minmod limited slope
            if dL * dR <= 0:
                slope = 0.0
            else:
                slope = dL if abs(dL) < abs(dR) else dR
            return L + 0.5 * slope
        else:
            # Upwind from right: reconstruct at left edge of cell i
            if i >= N:
                return bc_right
            R = field[i]
            dL = R - (field[i - 1] if i > 0 else bc_left)
            dR = (field[i + 1] if i < N - 1 else bc_right) - R
            if dL * dR <= 0:
                slope = 0.0
            else:
                slope = dL if abs(dL) < abs(dR) else dR
            return R - 0.5 * slope

    @staticmethod
    def _thomas_solve(a, b, c, d):
        """Scalar Thomas algorithm (kept for fallback)."""
        N = len(d)
        cp = np.zeros(N); dp = np.zeros(N)
        cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
        for i in range(1, N):
            denom = b[i] - a[i] * cp[i - 1]
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
        x = np.zeros(N)
        x[N - 1] = dp[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    @staticmethod
    def _block_thomas_solve(a_blocks, b_blocks, c_blocks, d_vecs):
        """Block Thomas algorithm for 2x2 block tridiagonal system.

        Solves the system:
            a_blocks[i] * x[i-1] + b_blocks[i] * x[i] + c_blocks[i] * x[i+1] = d_vecs[i]

        where each block is a 2x2 matrix and each x[i], d_vecs[i] is a 2-vector.

        Args:
            a_blocks[i]: 2x2 lower diagonal block (a_blocks[0] unused)
            b_blocks[i]: 2x2 diagonal block
            c_blocks[i]: 2x2 upper diagonal block (c_blocks[N-1] unused)
            d_vecs[i]: 2-vector RHS

        Returns:
            list of 2-vectors x[i]
        """
        N = len(d_vecs)
        Z = np.zeros((2, 2))

        # Forward sweep
        cp = [None] * N  # modified upper diagonal blocks
        dp = [None] * N  # modified RHS vectors

        # i=0: no lower diagonal
        b_inv = np.linalg.inv(b_blocks[0])
        cp[0] = b_inv @ c_blocks[0]
        dp[0] = b_inv @ d_vecs[0]

        for i in range(1, N):
            temp = b_blocks[i] - a_blocks[i] @ cp[i - 1]
            temp_inv = np.linalg.inv(temp)
            if i < N - 1:
                cp[i] = temp_inv @ c_blocks[i]
            else:
                cp[i] = Z.copy()
            dp[i] = temp_inv @ (d_vecs[i] - a_blocks[i] @ dp[i - 1])

        # Backward substitution
        x = [None] * N
        x[N - 1] = dp[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] @ x[i + 1]

        return x

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One semi-implicit timestep with Newton-iterated 2x2 block Thomas solve
        and drift-flux off-diagonal coupling.

        The Newton loop re-evaluates bridge physics at the updated state and
        re-assembles/re-solves the block system. Deltas are always relative to
        start-of-step values. Friction is frozen at the initial evaluation.
        Energy update happens once after all Newton iterations complete.

        Modifies all arrays in-place.
        """
        N = self.N
        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_old = mdot.copy()

        # ====================================================================
        # INITIAL PHYSICS EVALUATION -- ALL from OM-generated C
        # ====================================================================
        # Set simulation time for time-varying BCs (RampedBreak, etc.)
        self.bridge.set_time(getattr(self, 'time', 0.0))
        self.bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v, mdot=mdot)
        self.bridge.evaluate()

        drho_dp = self.bridge.get('drho_dp')
        rho_face = self.bridge.get('rho_face')
        rho_l = self.bridge.get('rho_l')
        rho_v = self.bridge.get('rho_v')
        Gamma = self.bridge.get('Gamma')
        q_i_l = self.bridge.get('q_i_l')
        q_i_v = self.bridge.get('q_i_v')
        h_sat_l = self.bridge.get('h_sat_l')
        h_sat_v = self.bridge.get('h_sat_v')
        T_l = self.bridge.get('T_l')
        T_sat = self.bridge.get('T_sat_cell')

        # Phi2 may have N or N+1 entries depending on model
        Phi2 = self.bridge.get('Phi2') if self.bridge.has('Phi2') else np.ones(N + 1)

        # Phasic compressibility -- required for block-coupled solve
        if self.bridge.has('drho_l_dp') and self.bridge.has('drho_v_dp'):
            drho_l_dp = self.bridge.get('drho_l_dp')
            drho_v_dp = self.bridge.get('drho_v_dp')
        else:
            # Fallback: approximate phasic compressibilities from mixture
            drho_l_dp = drho_dp * 0.1
            drho_v_dp = drho_dp * 10.0

        # ====================================================================
        # NUMERICAL METHOD ONLY BELOW
        # ====================================================================

        # -- Critical flow (from Modelica via bridge) --
        mdot_crit = 1e10
        outlet_choked = False
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            mdot_crit_arr = self.bridge.get('mdot_crit')
            mdot_crit = mdot_crit_arr[0] if len(mdot_crit_arr) > 0 else 1e10
            mdot_crit *= getattr(self, 'C_d_factor', 1.0)
            outlet_choked = mdot_old[N] > 0

        # -- Friction with implicit resistance (semi-implicit friction treatment) --
        # Friction is FROZEN at initial evaluation -- not re-evaluated in Newton loop
        K_geom = self.f_D * self.dx / (2 * self.D_h)
        fric = np.zeros(N + 1)
        sigma = np.zeros(N + 1)

        for i in range(N + 1):
            phi2_i = Phi2[min(i, len(Phi2) - 1)]
            if rho_face[i] > 0.01 and np.isfinite(mdot_old[i]):
                f = (phi2_i * K_geom * abs(mdot_old[i]) * mdot_old[i]
                     / (rho_face[i] * self.A_flow**2))
                fric[i] = f if np.isfinite(f) else 0.0
                sigma[i] = (2 * dt * phi2_i * K_geom * abs(mdot_old[i])
                            / (rho_face[i] * self.A_flow**2))

        beta = dt * self.A_flow / self.dx
        beta_eff = beta / (1.0 + sigma)

        # -- Drift-flux phasic mass flows from bridge (if available) --
        has_drift_flux = self.bridge.has('mdot_v') and self.bridge.has('mdot_l')
        if has_drift_flux:
            mdot_v_face = self.bridge.get('mdot_v')
            mdot_l_face = self.bridge.get('mdot_l')
        else:
            mdot_v_face = None
            mdot_l_face = None

        # -- Face arrays for off-diagonal coupling --
        # alpha_face and rho_v_face are needed for the (1,0) off-diagonal entries.
        # These come from the bridge if available; otherwise approximate from cells.
        if self.bridge.has('alpha_face'):
            alpha_face_arr = self.bridge.get('alpha_face')
        else:
            # Upwind approximation from cell values
            alpha_face_arr = np.zeros(N + 1)
            for i in range(N + 1):
                if i == 0:
                    alpha_face_arr[i] = alpha_old[0]
                elif i == N:
                    alpha_face_arr[i] = alpha_old[N - 1]
                else:
                    alpha_face_arr[i] = 0.5 * (alpha_old[i - 1] + alpha_old[i])

        if self.bridge.has('rho_v_face'):
            rho_v_face_arr = self.bridge.get('rho_v_face')
        else:
            # Approximate from cell values
            rho_v_face_arr = np.zeros(N + 1)
            for i in range(N + 1):
                if i == 0:
                    rho_v_face_arr[i] = rho_v[0]
                elif i == N:
                    rho_v_face_arr[i] = rho_v[N - 1]
                else:
                    rho_v_face_arr[i] = 0.5 * (rho_v[i - 1] + rho_v[i])

        # ====================================================================
        # NEWTON ITERATION LOOP
        # ====================================================================
        # Each iteration: assemble block system, solve, apply deltas relative
        # to start-of-step. Re-evaluate bridge physics for iterations > 0.
        # First iteration uses phasic drho_dp for A11 (onset-friendly).
        # Subsequent iterations use h_mix drho_dp from re-evaluated bridge.

        for newton_iter in range(self.n_newton):

            # ---- Re-evaluate bridge at current state (iterations > 0) ----
            if newton_iter > 0:
                self.bridge.set_state(p, alpha=alpha, h_l=h_l_old,
                                      h_v=h_v_old, mdot=mdot)
                self.bridge.evaluate()

                # Get updated physics from bridge
                drho_dp = self.bridge.get('drho_dp')
                rho_l = self.bridge.get('rho_l')
                rho_v = self.bridge.get('rho_v')
                Gamma = self.bridge.get('Gamma')
                T_l = self.bridge.get('T_l')
                T_sat = self.bridge.get('T_sat_cell')
                h_sat_l = self.bridge.get('h_sat_l')
                h_sat_v = self.bridge.get('h_sat_v')
                rho_face = self.bridge.get('rho_face')

                # Phasic compressibility
                if self.bridge.has('drho_l_dp') and self.bridge.has('drho_v_dp'):
                    drho_l_dp = self.bridge.get('drho_l_dp')
                    drho_v_dp = self.bridge.get('drho_v_dp')

                # Drift-flux phasic flows
                if has_drift_flux:
                    mdot_v_face = self.bridge.get('mdot_v')
                    mdot_l_face = self.bridge.get('mdot_l')

                # Face arrays for off-diagonals
                if self.bridge.has('alpha_face'):
                    alpha_face_arr = self.bridge.get('alpha_face')
                if self.bridge.has('rho_v_face'):
                    rho_v_face_arr = self.bridge.get('rho_v_face')

            # KEY: First iteration uses phasic drho_dp for A11 (onset-friendly).
            # Subsequent iterations use h_mix drho_dp from re-evaluated bridge
            # (correct wave speed, since bridge now sees actual void state).
            use_phasic_A11 = (newton_iter == 0)

            # ================================================================
            # 2x2 BLOCK TRIDIAGONAL ASSEMBLY
            # ================================================================
            a_blocks = [np.zeros((2, 2)) for _ in range(N)]
            b_blocks = [np.zeros((2, 2)) for _ in range(N)]
            c_blocks = [np.zeros((2, 2)) for _ in range(N)]
            d_vecs = [np.zeros(2) for _ in range(N)]

            for i in range(N):
                al = alpha_old[i]  # ALWAYS relative to start-of-step
                rv_i = max(rho_v[i], 0.01)
                rl_i = max(rho_l[i], 1.0)

                # Face coupling coefficients (same logic as scalar solver)
                bL = 0.0 if (self.inlet_closed and i == 0) else (
                    0.0 if i == 0 else beta_eff[i])
                bR = 0.0 if (i == N - 1 and outlet_choked) else beta_eff[i + 1]

                # ---- Row 1: Mixture mass conservation ----
                if use_phasic_A11:
                    # Phasic drho_dp: onset-friendly (small, lets void grow)
                    drho_A11 = (1 - al) * drho_l_dp[i] + al * drho_v_dp[i]
                else:
                    # h_mix drho_dp from re-evaluated bridge: correct wave speed
                    drho_A11 = drho_dp[i]

                A11 = self.V_cell / dt * drho_A11 + bL + bR
                # Void changes mixture density: d(rho_m)/d(alpha) = rho_v - rho_l
                A12 = self.V_cell / dt * (rv_i - rl_i)

                # RHS Row 1: mass residual + implicit friction correction
                R1 = (mdot_old[i] - mdot_old[i + 1])
                R1 -= dt * (fric[i] / (1.0 + sigma[i])
                            - fric[i + 1] / (1.0 + sigma[i + 1]))

                # Old pressure gradient contributions from momentum coupling
                if not (self.inlet_closed and i == 0) and i > 0:
                    R1 -= bL * (p_old[i] - p_old[i - 1])
                if i < N - 1:
                    R1 -= bR * (p_old[i] - p_old[i + 1])

                # Outlet BC contribution to R1
                if i == N - 1:
                    if outlet_choked:
                        R1 += (mdot_old[N] - mdot_crit)
                    else:
                        R1 -= bR * (p_old[i] - self.p_out)

                # ---- Row 2: Vapour mass conservation ----
                A21 = self.V_cell / dt * al * drho_v_dp[i]
                A22 = self.V_cell / dt * rv_i

                # Linearize Gamma response to pressure (dGamma/dp term)
                if Gamma[i] > 0 and T_l[i] > T_sat[i]:
                    superheat = max(T_l[i] - T_sat[i], 0.1)
                    h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
                    dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / rl_i) / h_fg
                    dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                    A21 += self.V_cell * dGamma_dp

                # ---- Drift-flux off-diagonal coupling for Row 2 ----
                # Vapor flux response to pressure: when pressure in a neighbor
                # changes, mixture momentum changes by beta_eff, and the
                # drift-flux split allocates a fraction to vapor:
                #   d(mdot_v)/dp = alpha_face * rho_v_face * beta_eff / rho_face
                beta_v_L = 0.0
                beta_v_R = 0.0
                if bL > 0.0:
                    rho_f_L = max(rho_face[i], 1.0)
                    beta_v_L = (alpha_face_arr[i] * rho_v_face_arr[i]
                                * beta_eff[i] / rho_f_L)
                if bR > 0.0:
                    rho_f_R = max(rho_face[i + 1], 1.0)
                    beta_v_R = (alpha_face_arr[i + 1] * rho_v_face_arr[i + 1]
                                * beta_eff[i + 1] / rho_f_R)

                # Diagonal contribution: vapor leaves this cell when its
                # pressure rises (both faces)
                A21 -= (beta_v_L + beta_v_R)

                # RHS Row 2: vapour flux + phase change source
                if has_drift_flux:
                    flux_v_old = mdot_v_face[i] - mdot_v_face[i + 1]
                else:
                    # Donor-cell vapour flux using old-state values
                    if mdot_old[i] >= 0:
                        alpha_in = alpha_old[i - 1] if i > 0 else al
                    else:
                        alpha_in = al
                    if mdot_old[i + 1] >= 0:
                        alpha_out = al
                    else:
                        alpha_out = alpha_old[i + 1] if i < N - 1 else al
                    flux_v_old = (mdot_old[i] * alpha_in
                                  - mdot_old[i + 1] * alpha_out)

                R2 = flux_v_old + self.V_cell * Gamma[i]

                # ---- Assemble blocks ----
                b_blocks[i] = np.array([[A11, A12],
                                        [A21, A22]])
                d_vecs[i] = np.array([R1, R2])

                # Off-diagonal blocks: pressure coupling in (0,0) and
                # vapor flux coupling in (1,0)
                if i > 0:
                    a_blocks[i] = np.array([[-bL, 0.0],
                                            [beta_v_L, 0.0]])
                if i < N - 1:
                    c_blocks[i] = np.array([[-bR, 0.0],
                                            [beta_v_R, 0.0]])

            # ================================================================
            # BLOCK THOMAS SOLVE
            # ================================================================
            x = self._block_thomas_solve(a_blocks, b_blocks, c_blocks, d_vecs)

            # Apply: deltas ALWAYS relative to start-of-step
            for i in range(N):
                delta_p = x[i][0]
                delta_alpha = x[i][1]

                # Update pressure
                p_new = p_old[i] + delta_p
                if not np.isfinite(p_new):
                    p_new = p_old[i]
                p[i] = max(self.bridge.p_min, min(self.bridge.p_max, p_new))

                # Update void fraction
                alpha_new = alpha_old[i] + delta_alpha
                alpha[i] = max(0.0, min(1.0, alpha_new))

                # Nucleation floor: if phase change is active, ensure minimum void
                if Gamma[i] > 0:
                    alpha[i] = max(alpha[i], 1e-3)

            # Update momentum (use initial beta_eff -- frozen friction)
            mdot[0] = 0.0  # wall BC
            for i in range(1, N):
                mdot[i] = (mdot_old[i] + beta_eff[i] * (p[i - 1] - p[i])
                           - dt * fric[i] / (1.0 + sigma[i]))

            # Outlet with critical flow limiter
            mdot_mom = (mdot_old[N] + beta_eff[N] * (p[N - 1] - self.p_out)
                        - dt * fric[N] / (1.0 + sigma[N]))
            if self.use_critical_flow and mdot_mom > 0:
                mdot[N] = min(mdot_mom, mdot_crit)
            else:
                mdot[N] = mdot_mom

        # ====================================================================
        # ENERGY UPDATE -- after Newton loop completes (once, not per iteration)
        # ====================================================================
        # Re-evaluate bridge at final state for energy source terms
        # (q_i_l, q_i_v, Gamma, h_sat are needed at the converged state)
        if self.n_newton > 1:
            self.bridge.set_state(p, alpha=alpha, h_l=h_l_old,
                                  h_v=h_v_old, mdot=mdot)
            self.bridge.evaluate()
            q_i_l = self.bridge.get('q_i_l')
            q_i_v = self.bridge.get('q_i_v')
            Gamma = self.bridge.get('Gamma')
            h_sat_l = self.bridge.get('h_sat_l')
            h_sat_v = self.bridge.get('h_sat_v')
            rho_l = self.bridge.get('rho_l')
            rho_v = self.bridge.get('rho_v')
            if has_drift_flux:
                mdot_v_face = self.bridge.get('mdot_v')
                mdot_l_face = self.bridge.get('mdot_l')

        # -- Phasic energy (explicit, using _old enthalpy values) --
        dp_dt = (p - p_old) / dt

        for i in range(N):
            al = alpha_old[i]

            # Liquid energy
            m_l = max((1 - al) * rho_l[i] * self.V_cell, 1e-12)
            if (1 - al) > 1e-6:
                if has_drift_flux:
                    ml_in = mdot_l_face[i]
                    ml_out = mdot_l_face[i + 1]
                else:
                    al_in = alpha_old[i - 1] if i > 0 and mdot[i] >= 0 else al
                    al_out = al if mdot[i + 1] >= 0 else (alpha_old[i + 1] if i < N - 1 else al)
                    ml_in = mdot[i] * (1 - al_in)
                    ml_out = mdot[i + 1] * (1 - al_out)

                # Face enthalpy reconstruction using OLD values
                flow_in = ml_in if has_drift_flux else mdot[i]
                flow_out = ml_out if has_drift_flux else mdot[i + 1]
                if self.reconstruction == 'muscl':
                    h_in = self._muscl_face(h_l_old, i, N, flow_in, h_l_old[0], h_l_old[N - 1])
                    h_out = self._muscl_face(h_l_old, i + 1, N, flow_out, h_l_old[0], h_l_old[N - 1])
                else:
                    h_in = h_l_old[i - 1] if (i > 0 and flow_in >= 0) else h_l_old[i]
                    h_out = h_l_old[i] if flow_out >= 0 else (h_l_old[i + 1] if i < N - 1 else h_l_old[i])

                flux = ml_in * (h_in - h_l_old[i]) - ml_out * (h_out - h_l_old[i])
                pw = (1 - al) * self.V_cell * dp_dt[i]
                qi = q_i_l[i] * self.V_cell
                phase = -Gamma[i] * h_l_old[i] * self.V_cell

                h_l[i] = h_l_old[i] + dt / m_l * (flux + pw + qi + phase)
                h_l[i] = max(1e4, min(h_l[i], h_sat_v[i]))
            else:
                h_l[i] = h_sat_l[i]

            # Vapour energy
            m_v = max(al * rho_v[i] * self.V_cell, 1e-12)
            if al > 1e-6:
                if has_drift_flux:
                    mv_in = mdot_v_face[i]
                    mv_out = mdot_v_face[i + 1]
                else:
                    al_in = alpha_old[i - 1] if i > 0 and mdot[i] >= 0 else al
                    al_out = al if mdot[i + 1] >= 0 else (alpha_old[i + 1] if i < N - 1 else al)
                    mv_in = mdot[i] * al_in
                    mv_out = mdot[i + 1] * al_out

                # Face enthalpy reconstruction using OLD values
                flow_in_v = mv_in if has_drift_flux else mdot[i]
                flow_out_v = mv_out if has_drift_flux else mdot[i + 1]
                if self.reconstruction == 'muscl':
                    hv_in = self._muscl_face(h_v_old, i, N, flow_in_v, h_v_old[0], h_v_old[N - 1])
                    hv_out = self._muscl_face(h_v_old, i + 1, N, flow_out_v, h_v_old[0], h_v_old[N - 1])
                else:
                    hv_in = h_v_old[i - 1] if (i > 0 and flow_in_v >= 0) else h_v_old[i]
                    hv_out = h_v_old[i] if flow_out_v >= 0 else (h_v_old[i + 1] if i < N - 1 else h_v_old[i])

                flux_v = mv_in * (hv_in - h_v_old[i]) - mv_out * (hv_out - h_v_old[i])
                pw_v = al * self.V_cell * dp_dt[i]
                qi_v = q_i_v[i] * self.V_cell
                phase_v = Gamma[i] * h_v_old[i] * self.V_cell

                h_v[i] = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(h_sat_v[i], min(h_v[i], 4e6))
            else:
                h_v[i] = h_sat_v[i]
