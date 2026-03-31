"""
bridge_5eq_solver_picard.py — Picard iteration of V11 block Thomas.

Iterates the V11 semi-implicit scheme with re-evaluated physics at each
iteration. This captures the full thermodynamic coupling:
  corrected h → corrected Gamma → corrected α → corrected p

V11's operator splitting evaluates ALL physics at the old state. Picard
re-evaluates at the latest guess, so the pressure solve sees updated
phase change rates and enthalpies. Converges in 2-3 iterations.

Cost: N_iter bridge evaluations per step (vs 1 for V11, ~100 for JFNK).
"""

import numpy as np

from .codegen.equation_bridge import OMEquationBridge
from .bridge_5eq_solver_v11_a12mod import BridgeDriftFluxSolver


class PicardSolver:
    """Picard iteration of V11's semi-implicit scheme.

    Each timestep:
    1. V11 step from old state → initial guess
    2. Re-evaluate bridge at guess state → updated physics
    3. Re-run block Thomas + momentum + energy with updated physics
       but time derivatives from the REAL old state
    4. Check convergence, iterate if needed

    This eliminates operator splitting by making the physics evaluation
    self-consistent with the solution state.
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 tau_mix=4.5e-4, use_isentropic_a11=True,
                 max_iters=3, tol=1e-3):
        """
        Args:
            max_iters: Maximum Picard iterations (1 = same as V11).
            tol: Convergence tolerance (relative pressure change).
        """
        self.bridge = bridge
        self.N = bridge.N
        self.spec = spec
        self.tau_mix = tau_mix
        self.use_isentropic_a11 = use_isentropic_a11
        self.max_iters = max_iters
        self.tol = tol

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

        # Time
        self.time = 0.0

        # Diagnostics
        self.picard_iters = 0
        self.dp_history = []

    @staticmethod
    def _block_thomas_solve(a_blocks, b_blocks, c_blocks, d_vecs):
        """Block Thomas algorithm for 2x2 block tridiagonal system."""
        N = len(d_vecs)
        Z = np.zeros((2, 2))
        cp = [None] * N
        dp = [None] * N
        b_inv = np.linalg.inv(b_blocks[0])
        cp[0] = b_inv @ c_blocks[0]
        dp[0] = b_inv @ d_vecs[0]
        for i in range(1, N):
            temp = b_blocks[i] - a_blocks[i] @ cp[i - 1]
            temp_inv = np.linalg.inv(temp)
            cp[i] = temp_inv @ c_blocks[i] if i < N - 1 else Z.copy()
            dp[i] = temp_inv @ (d_vecs[i] - a_blocks[i] @ dp[i - 1])
        x = [None] * N
        x[N - 1] = dp[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] @ x[i + 1]
        return x

    def _solve_one_step(self, p_old, alpha_old, h_l_old, h_v_old, mdot_old,
                        p_guess, alpha_guess, h_l_guess, h_v_guess, mdot_guess,
                        dt):
        """One semi-implicit step with physics at GUESS, time derivatives from OLD.

        This is V11's numerical scheme but with decoupled physics evaluation:
        - Bridge is evaluated at the guess state (captures current physics)
        - Time derivatives use the real old state (correct temporal discretization)

        Returns: (p_new, alpha_new, h_l_new, h_v_new, mdot_new)
        """
        N = self.N

        # ---- Evaluate bridge at GUESS state ----
        # Use old time for bridge (matches V11 convention for RampedBreak)
        self.bridge.set_time(self.time)
        self.bridge.set_state(p_guess, alpha=alpha_guess,
                              h_l=h_l_guess, h_v=h_v_guess, mdot=mdot_guess)
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
        Phi2 = self.bridge.get('Phi2') if self.bridge.has('Phi2') else np.ones(N + 1)

        if self.bridge.has('drho_l_dp') and self.bridge.has('drho_v_dp'):
            drho_l_dp = self.bridge.get('drho_l_dp')
            drho_v_dp = self.bridge.get('drho_v_dp')
        else:
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
            outlet_choked = mdot_old[N] > 0

        # Drift-flux phasic flows
        has_df = self.bridge.has('mdot_v') and self.bridge.has('mdot_l')
        if has_df:
            mdot_v_face = self.bridge.get('mdot_v')
            mdot_l_face = self.bridge.get('mdot_l')

        # ---- Friction (using OLD mdot for linearization) ----
        K_geom = self.f_D * self.dx / (2 * self.D_h)
        fric = np.zeros(N + 1)
        sigma = np.zeros(N + 1)
        for i in range(N + 1):
            phi2_i = Phi2[min(i, len(Phi2) - 1)]
            rf = rho_face[i]
            if rf > 0.01 and np.isfinite(mdot_old[i]):
                f = phi2_i * K_geom * abs(mdot_old[i]) * mdot_old[i] / (rf * self.A_flow**2)
                fric[i] = f if np.isfinite(f) else 0.0
                sigma[i] = 2 * dt * phi2_i * K_geom * abs(mdot_old[i]) / (rf * self.A_flow**2)

        beta = dt * self.A_flow / self.dx
        beta_eff = beta / (1.0 + sigma)

        # ---- Block Thomas assembly (same as V11, but physics at GUESS) ----
        a_blocks = [np.zeros((2, 2)) for _ in range(N)]
        b_blocks = [np.zeros((2, 2)) for _ in range(N)]
        c_blocks = [np.zeros((2, 2)) for _ in range(N)]
        d_vecs = [np.zeros(2) for _ in range(N)]

        for i in range(N):
            al = alpha_old[i]  # OLD alpha for linearization
            rv_i = max(rho_v[i], 0.01)
            rl_i = max(rho_l[i], 1.0)

            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_eff[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_eff[i + 1]

            # Row 1: mixture mass
            drho_mech = (1 - al) * drho_l_dp_s[i] + al * drho_v_dp_s[i]
            A11 = self.V_cell / dt * drho_mech + bL + bR
            A12 = self.V_cell / dt * (rv_i - rl_i) / (1.0 + self.tau_mix / dt)

            # RHS uses OLD mdot and OLD pressure
            R1 = mdot_old[i] - mdot_old[i + 1]
            R1 -= dt * (fric[i] / (1.0 + sigma[i]) - fric[i + 1] / (1.0 + sigma[i + 1]))
            if not (self.inlet_closed and i == 0) and i > 0:
                R1 -= bL * (p_old[i] - p_old[i - 1])
            if i < N - 1:
                R1 -= bR * (p_old[i] - p_old[i + 1])
            if i == N - 1:
                if outlet_choked:
                    R1 += mdot_old[N] - mdot_crit
                else:
                    R1 -= bR * (p_old[i] - self.p_out)

            # Row 2: vapour mass (Gamma at GUESS state)
            A21 = self.V_cell / dt * al * drho_v_dp[i]
            A22 = self.V_cell / dt * rv_i

            if Gamma[i] > 0 and T_l[i] > T_sat[i]:
                superheat = max(T_l[i] - T_sat[i], 0.1)
                h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
                dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / rl_i) / h_fg
                dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                A21 += self.V_cell * dGamma_dp

            if has_df:
                flux_v_old = mdot_v_face[i] - mdot_v_face[i + 1]
            else:
                if mdot_old[i] >= 0:
                    alpha_in = alpha_old[i - 1] if i > 0 else al
                else:
                    alpha_in = al
                if mdot_old[i + 1] >= 0:
                    alpha_out = al
                else:
                    alpha_out = alpha_old[i + 1] if i < N - 1 else al
                flux_v_old = mdot_old[i] * alpha_in - mdot_old[i + 1] * alpha_out

            R2 = flux_v_old + self.V_cell * Gamma[i]

            b_blocks[i] = np.array([[A11, A12], [A21, A22]])
            d_vecs[i] = np.array([R1, R2])
            if i > 0:
                a_blocks[i] = np.array([[-bL, 0.0], [0.0, 0.0]])
            if i < N - 1:
                c_blocks[i] = np.array([[-bR, 0.0], [0.0, 0.0]])

        # ---- Solve block Thomas ----
        x = self._block_thomas_solve(a_blocks, b_blocks, c_blocks, d_vecs)

        # ---- Apply pressure/void updates ----
        p_new = p_old.copy()
        alpha_new = alpha_old.copy()
        for i in range(N):
            dp = x[i][0]
            da = x[i][1]
            p_new[i] = max(self.bridge.p_min, min(self.bridge.p_max,
                                                   p_old[i] + dp))
            alpha_new[i] = max(0.0, min(1.0, alpha_old[i] + da))
            if Gamma[i] > 0:
                alpha_new[i] = max(alpha_new[i], 1e-3)

        # ---- Momentum (with implicit friction) ----
        mdot_new = mdot_old.copy()
        mdot_new[0] = 0.0
        for i in range(1, N):
            mdot_new[i] = (mdot_old[i] + beta_eff[i] * (p_new[i - 1] - p_new[i])
                           - dt * fric[i] / (1.0 + sigma[i]))
        mdot_mom = (mdot_old[N] + beta_eff[N] * (p_new[N - 1] - self.p_out)
                    - dt * fric[N] / (1.0 + sigma[N]))
        if self.use_critical_flow and mdot_mom > 0:
            mdot_new[N] = min(mdot_mom, mdot_crit)
        else:
            mdot_new[N] = mdot_mom

        # ---- Phasic energy (using GUESS enthalpies for advection) ----
        dp_dt = (p_new - p_old) / dt
        h_l_new = h_l_old.copy()
        h_v_new = h_v_old.copy()

        for i in range(N):
            al = alpha_old[i]

            # Liquid energy
            m_l = max((1 - al) * rho_l[i] * self.V_cell, 1e-12)
            if (1 - al) > 1e-6:
                if has_df:
                    ml_in = mdot_l_face[i]; ml_out = mdot_l_face[i + 1]
                else:
                    al_in = alpha_old[i - 1] if i > 0 and mdot_new[i] >= 0 else al
                    al_out = al if mdot_new[i + 1] >= 0 else (alpha_old[i + 1] if i < N - 1 else al)
                    ml_in = mdot_new[i] * (1 - al_in)
                    ml_out = mdot_new[i + 1] * (1 - al_out)

                flow_in = ml_in if has_df else mdot_new[i]
                flow_out = ml_out if has_df else mdot_new[i + 1]
                # GUESS enthalpies for advection (not old!)
                h_in = h_l_guess[i - 1] if (i > 0 and flow_in >= 0) else h_l_guess[i]
                h_out = h_l_guess[i] if flow_out >= 0 else (h_l_guess[i + 1] if i < N - 1 else h_l_guess[i])

                flux = ml_in * (h_in - h_l_guess[i]) - ml_out * (h_out - h_l_guess[i])
                pw = (1 - al) * self.V_cell * dp_dt[i]
                qi = q_i_l[i] * self.V_cell
                phase = -Gamma[i] * h_l_guess[i] * self.V_cell

                h_l_new[i] = h_l_old[i] + dt / m_l * (flux + pw + qi + phase)
                h_l_new[i] = max(1e4, min(h_l_new[i], h_sat_v[i]))
            else:
                h_l_new[i] = h_sat_l[i]

            # Vapour energy
            m_v = max(al * rho_v[i] * self.V_cell, 1e-12)
            if al > 1e-6:
                if has_df:
                    mv_in = mdot_v_face[i]; mv_out = mdot_v_face[i + 1]
                else:
                    al_in = alpha_old[i - 1] if i > 0 and mdot_new[i] >= 0 else al
                    al_out = al if mdot_new[i + 1] >= 0 else (alpha_old[i + 1] if i < N - 1 else al)
                    mv_in = mdot_new[i] * al_in
                    mv_out = mdot_new[i + 1] * al_out

                flow_in_v = mv_in if has_df else mdot_new[i]
                flow_out_v = mv_out if has_df else mdot_new[i + 1]
                hv_in = h_v_guess[i - 1] if (i > 0 and flow_in_v >= 0) else h_v_guess[i]
                hv_out = h_v_guess[i] if flow_out_v >= 0 else (h_v_guess[i + 1] if i < N - 1 else h_v_guess[i])

                flux_v = mv_in * (hv_in - h_v_guess[i]) - mv_out * (hv_out - h_v_guess[i])
                pw_v = al * self.V_cell * dp_dt[i]
                qi_v = q_i_v[i] * self.V_cell
                phase_v = Gamma[i] * h_v_guess[i] * self.V_cell

                h_v_new[i] = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v_new[i] = max(h_sat_v[i], min(h_v_new[i], 4e6))
            else:
                h_v_new[i] = h_sat_v[i]

        return p_new, alpha_new, h_l_new, h_v_new, mdot_new

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One timestep with Picard iteration. Modifies arrays in-place."""
        N = self.N

        # Save REAL old state
        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_old = mdot.copy()

        # Initial guess = old state (first iteration = standard V11)
        p_g = p_old.copy()
        alpha_g = alpha_old.copy()
        h_l_g = h_l_old.copy()
        h_v_g = h_v_old.copy()
        mdot_g = mdot_old.copy()

        for it in range(self.max_iters):
            p_new, alpha_new, h_l_new, h_v_new, mdot_new = \
                self._solve_one_step(
                    p_old, alpha_old, h_l_old, h_v_old, mdot_old,
                    p_g, alpha_g, h_l_g, h_v_g, mdot_g, dt)

            # Convergence check: relative pressure change from last iterate
            if it > 0:
                dp_rel = np.max(np.abs(p_new - p_g)) / max(np.max(np.abs(p_new)), 1.0)
                if dp_rel < self.tol:
                    self.picard_iters = it + 1
                    break

            # Update guess for next iteration
            p_g = p_new.copy()
            alpha_g = alpha_new.copy()
            h_l_g = h_l_new.copy()
            h_v_g = h_v_new.copy()
            mdot_g = mdot_new.copy()
        else:
            self.picard_iters = self.max_iters

        # Apply final result
        p[:] = p_new
        alpha[:] = alpha_new
        h_l[:] = h_l_new
        h_v[:] = h_v_new
        mdot[:] = mdot_new
