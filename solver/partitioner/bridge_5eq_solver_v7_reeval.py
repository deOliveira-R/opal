"""
bridge_5eq_solver_v7_reeval.py — V7: Bridge re-evaluation for void onset.

Keeps the stable base pressure solve (h_mix drho_dp for correct wave speed)
but re-evaluates the bridge at the NEW pressure/momentum state before the void
update. This gives the void equation access to Gamma computed at the new-state
T_sat, breaking the circular dependency that causes 140ms onset delay.

Cost: 2x bridge evaluations per timestep.
Expected: Correct pressure (same as baseline) + improved void onset.

State variables: p[N], alpha[N], h_l[N], h_v[N], mdot[N+1]
Bridge provides: rho_l, rho_v, rho_m, rho_face, drho_dp,
    Gamma, q_i_l, q_i_v, Phi2, T_l, T_sat_cell, h_sat_l, h_sat_v, h_mix
"""

import numpy as np

from .codegen.equation_bridge import OMEquationBridge


class BridgeDriftFluxSolver:
    """5-equation drift-flux solver driven by the OM equation bridge.

    Physics (properties, closures, friction multiplier, interfacial transfer)
    from OM-generated C. Numerics (pressure solve, momentum, void transport,
    phasic energy) from Python.
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 reconstruction='donor_cell', use_block_coupling=False,
                 corrector_steps=0):
        """
        Args:
            bridge: OMEquationBridge with compiled Modelica model
            spec: Pipe1DGridSpec (geometry, BCs)
            es: EquationSystem from XML — pass this to load ALL parameter values
                from Modelica (H_i, C_0, etc.). Without es, only geometry is set.
            reconstruction: 'donor_cell' (1st order) or 'muscl' (2nd order, minmod TVD)
            use_block_coupling: if True, use Schur complement block-coupled
                pressure-void solve instead of sequential (explicit void transport)
            corrector_steps: number of predictor-corrector iterations (0=none, 1=one
                corrector pass). Each corrector re-evaluates the bridge at the
                predicted state and re-solves pressure, reducing the lag between
                pressure and void fraction.
        """
        self.reconstruction = reconstruction
        self.use_block_coupling = use_block_coupling
        self.corrector_steps = corrector_steps
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
            # Bridge has mdot_crit from Modelica → critical flow is active
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

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One semi-implicit timestep. Modifies all arrays in-place."""
        N = self.N
        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_old = mdot.copy()

        # ══════════════════════════════════════════════════════════
        # PHYSICS EVALUATION — ALL from OM-generated C
        # ══════════════════════════════════════════════════════════
        # Set simulation time for time-varying BCs (RampedBreak, etc.)
        self.bridge.set_time(getattr(self, 'time', 0.0))
        self.bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v, mdot=mdot)
        self.bridge.evaluate()

        drho_dp = self.bridge.get('drho_dp')
        # Note: drho_dh is available from the bridge but unused by the semi-implicit
        # scheme — enthalpies are frozen during the pressure solve, so the drho_dh
        # coupling term is zero by construction. The h_mix-evaluated drho_dp already
        # captures the effective compressibility including thermal effects.
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

        # Phasic compressibility for block-coupled pressure-void solve
        block_active = (self.use_block_coupling
                        and self.bridge.has('drho_l_dp')
                        and self.bridge.has('drho_v_dp'))
        if block_active:
            drho_l_dp = self.bridge.get('drho_l_dp')
            drho_v_dp = self.bridge.get('drho_v_dp')

        # ══════════════════════════════════════════════════════════
        # NUMERICAL METHOD ONLY BELOW
        # ══════════════════════════════════════════════════════════

        # ── Critical flow (from Modelica via bridge) ──
        # mdot_crit already incorporates C_d_eff (time-varying if RampedBreak is used).
        # The C_d_factor attribute is kept for backward compatibility but should be
        # unnecessary when the Modelica model handles the ramp via C_d_eff.
        mdot_crit = 1e10
        outlet_choked = False
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            mdot_crit_arr = self.bridge.get('mdot_crit')
            mdot_crit = mdot_crit_arr[0] if len(mdot_crit_arr) > 0 else 1e10
            mdot_crit *= getattr(self, 'C_d_factor', 1.0)  # Legacy; prefer Modelica C_d_eff
            outlet_choked = mdot_old[N] > 0

        # ── Friction with implicit resistance (semi-implicit friction treatment) ──
        # The friction force is linearized and partially treated implicitly:
        #   fric(mdot_new) ≈ fric(mdot_old) + dfric/dmdot * (mdot_new - mdot_old)
        # This yields an effective coupling coefficient beta_eff = beta / (1 + sigma)
        # where sigma = dt * dfric/dmdot. When friction is large (high flow in
        # single-phase), sigma >> 1 and beta_eff → 0, naturally regularizing the
        # pressure tridiagonal without floors or hacks.
        # Ref: standard semi-implicit friction treatment in TH codes (RELAP5, TRACE).
        K_geom = self.f_D * self.dx / (2 * self.D_h)
        fric = np.zeros(N + 1)
        sigma = np.zeros(N + 1)

        for i in range(N + 1):
            phi2_i = Phi2[min(i, len(Phi2) - 1)]
            if rho_face[i] > 0.01 and np.isfinite(mdot_old[i]):
                f = (phi2_i * K_geom * abs(mdot_old[i]) * mdot_old[i]
                     / (rho_face[i] * self.A_flow**2))
                fric[i] = f if np.isfinite(f) else 0.0
                # Implicit friction resistance: sigma = dt * dfric/dmdot
                sigma[i] = (2 * dt * phi2_i * K_geom * abs(mdot_old[i])
                            / (rho_face[i] * self.A_flow**2))

        # Effective coupling: beta_eff smoothly blends inertial (sigma→0)
        # and algebraic (sigma→∞) momentum treatment at each face.
        beta = dt * self.A_flow / self.dx
        beta_eff = beta / (1.0 + sigma)

        # ── Drift-flux phasic mass flows from bridge (if available) ──
        # Read before pressure assembly (needed for R2 in block-coupled mode).
        has_drift_flux = self.bridge.has('mdot_v') and self.bridge.has('mdot_l')
        if has_drift_flux:
            mdot_v_face = self.bridge.get('mdot_v')
            mdot_l_face = self.bridge.get('mdot_l')
        else:
            mdot_v_face = None
            mdot_l_face = None

        # ── Pressure tridiagonal (with implicit friction resistance) ──
        a_tri = np.zeros(N); b_tri = np.zeros(N)
        c_tri = np.zeros(N); d_tri = np.zeros(N)

        # Block-coupled: store Schur complement coefficients for alpha back-substitution
        if block_active:
            _A21 = np.zeros(N)
            _A22 = np.zeros(N)
            _R2 = np.zeros(N)

        for i in range(N):
            # Face coupling using beta_eff (no floors needed)
            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_eff[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_eff[i + 1]

            a_tri[i] = -bL if i > 0 else 0.0
            c_tri[i] = -bR if i < N - 1 else 0.0

            if block_active:
                # ── Block-coupled: smooth drho_dp transition at saturation ──
                # At small α (onset): use Schur phasic compressibility (no 2400x
                # jump, allows void onset). At moderate α: transition to h_mix
                # compressibility (correct wave propagation speed). The transition
                # mirrors the physical thermal equilibration: freshly nucleated
                # cells behave as mechanical mixtures, while cells with established
                # void are in thermal quasi-equilibrium.
                al = alpha_old[i]
                rv_i = max(rho_v[i], 0.01)
                rl_i = max(rho_l[i], 1.0)

                # Schur amplification: ρ_l/ρ_v boost on vapor compressibility
                drho_schur = ((1 - al) * drho_l_dp[i]
                              + al * (rl_i / rv_i) * drho_v_dp[i])

                # Sigmoid transition: Schur at small α → h_mix at moderate α
                # alpha_mid=0.05: half-Schur, half-hmix at 5% void
                ALPHA_MID = 0.05
                blend = min(al / ALPHA_MID, 1.0)  # linear ramp [0, alpha_mid]
                drho_eff = (1 - blend) * drho_schur + blend * drho_dp[i]
                alpha_coeff = self.V_cell / dt * drho_eff

                b_tri[i] = alpha_coeff + bL + bR
                d_tri[i] = alpha_coeff * p_old[i]

                # RHS: mass residual + friction
                d_tri[i] += (mdot_old[i] - mdot_old[i + 1])
                d_tri[i] -= dt * (fric[i] / (1.0 + sigma[i])
                                - fric[i + 1] / (1.0 + sigma[i + 1]))

                # Semi-implicit void-pressure coupling
                if Gamma[i] > 0 and T_l[i] > T_sat[i]:
                    superheat = max(T_l[i] - T_sat[i], 0.1)
                    h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
                    dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / rl_i) / h_fg
                    dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                    void_diag = -self.V_cell * (rho_l[i] - rho_v[i]) / rv_i * dGamma_dp
                    if void_diag > 0:
                        b_tri[i] += void_diag
                        d_tri[i] += void_diag * p_old[i]

            else:
                # ── Sequential: original h_mix compressibility ──
                alpha_coeff = self.V_cell * drho_dp[i] / dt

                b_tri[i] = alpha_coeff + bL + bR
                d_tri[i] = alpha_coeff * p_old[i]

                # RHS: mass residual + friction (with implicit friction correction)
                d_tri[i] += (mdot_old[i] - mdot_old[i + 1])
                d_tri[i] -= dt * (fric[i] / (1.0 + sigma[i])
                                - fric[i + 1] / (1.0 + sigma[i + 1]))

                # Semi-implicit void-pressure coupling (ad-hoc diagonal term).
                if Gamma[i] > 0 and T_l[i] > T_sat[i]:
                    rv_i = max(rho_v[i], 0.01)
                    superheat = max(T_l[i] - T_sat[i], 0.1)
                    h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
                    dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / max(rho_l[i], 1.0)) / h_fg
                    dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                    void_diag = -self.V_cell * (rho_l[i] - rho_v[i]) / rv_i * dGamma_dp
                    if void_diag > 0:
                        b_tri[i] += void_diag
                        d_tri[i] += void_diag * p_old[i]

            # Outlet BC (same for both modes)
            if i == N - 1:
                if outlet_choked:
                    d_tri[i] += (mdot_old[N] - mdot_crit)
                else:
                    d_tri[i] += bR * self.p_out

        p[:] = self._thomas_solve(a_tri, b_tri, c_tri, d_tri)

        for i in range(N):
            if not np.isfinite(p[i]):
                p[i] = p_old[i]
            p[i] = max(self.bridge.p_min, min(self.bridge.p_max, p[i]))

        # ── Momentum (with implicit friction) ──
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

        # ── V7: Re-evaluate bridge at NEW pressure/momentum for void update ──
        # The base solver evaluates bridge at old state, so Gamma reflects
        # old T_sat(p_old). After depressurization, T_sat(p_new) < T_sat(p_old),
        # so the new-state superheat is larger → Gamma should be larger.
        # Re-evaluation breaks the onset delay circular dependency.
        self.bridge.set_state(p, alpha=alpha_old, h_l=h_l_old, h_v=h_v_old, mdot=mdot)
        self.bridge.evaluate()
        Gamma = self.bridge.get('Gamma')
        q_i_l = self.bridge.get('q_i_l')
        q_i_v = self.bridge.get('q_i_v')
        rho_v_new = self.bridge.get('rho_v')
        h_sat_l = self.bridge.get('h_sat_l')
        h_sat_v = self.bridge.get('h_sat_v')
        if has_drift_flux:
            mdot_v_face = self.bridge.get('mdot_v')
            mdot_l_face = self.bridge.get('mdot_l')

        # ── Void fraction update (explicit conservative form) ──
        if True:
            for i in range(N):
                al = alpha_old[i]
                rv = max(rho_v[i], 0.01)  # old rho_v for alpha*rho_v product

                if has_drift_flux:
                    flux_v = mdot_v_face[i] - mdot_v_face[i + 1]
                else:
                    if mdot[i] >= 0:
                        alpha_in = alpha_old[i - 1] if i > 0 else al
                    else:
                        alpha_in = al
                    if mdot[i + 1] >= 0:
                        alpha_out = al
                    else:
                        alpha_out = alpha_old[i + 1] if i < N - 1 else al
                    flux_v = mdot[i] * alpha_in - mdot[i + 1] * alpha_out

                alpha_rho_v_new = al * rv + dt / self.V_cell * (flux_v + self.V_cell * Gamma[i])
                # V7: use new-state rho_v (from re-evaluated bridge at new pressure)
                rv_new = max(rho_v_new[i], 0.01)
                alpha_new = max(0.0, min(1.0, alpha_rho_v_new / rv_new))

                if Gamma[i] > 0:
                    alpha_new = max(alpha_new, 1e-3)

                alpha[i] = alpha_new

        # ── Phasic energy (explicit, using _old enthalpy values) ──
        # All advection uses h_l_old/h_v_old to prevent directional bias from
        # sequential cell updates. Ref: C++ prototype five_eq_model.cpp:339-340.
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
                    h_in = self._muscl_face(h_l_old, i, N, flow_in, h_l_old[0], h_l_old[N-1])
                    h_out = self._muscl_face(h_l_old, i+1, N, flow_out, h_l_old[0], h_l_old[N-1])
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
                    hv_in = self._muscl_face(h_v_old, i, N, flow_in_v, h_v_old[0], h_v_old[N-1])
                    hv_out = self._muscl_face(h_v_old, i+1, N, flow_out_v, h_v_old[0], h_v_old[N-1])
                else:
                    hv_in = h_v_old[i - 1] if (i > 0 and flow_in_v >= 0) else h_v_old[i]
                    hv_out = h_v_old[i] if flow_out_v >= 0 else (h_v_old[i + 1] if i < N - 1 else h_v_old[i])

                flux_v = mv_in * (hv_in - h_v_old[i]) - mv_out * (hv_out - h_v_old[i])
                pw_v = al * self.V_cell * dp_dt[i]
                qi_v = q_i_v[i] * self.V_cell
                phase_v = Gamma[i] * h_v_old[i] * self.V_cell

                h_v[i] = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(h_sat_v[i], min(h_v[i], 4e6))  # Floor at h_sat_v, not 1e4
            else:
                h_v[i] = h_sat_v[i]

        # ── Predictor-corrector: re-solve pressure with updated state ──
        # After the predictor pass (above), p/alpha/h have been updated.
        # Re-evaluate the bridge at the predicted state to get updated drho_dp,
        # then re-solve pressure and momentum. This reduces the lag between
        # pressure and void, particularly at the saturation crossing where
        # drho_dp(h_mix) jumps 2400x. The corrector sees the post-void h_mix,
        # which better reflects the two-phase state.
        for _corr in range(self.corrector_steps):
            # Use old enthalpies — energy update shifts h_l toward saturation,
            # triggering the drho_dp jump in the corrector (counterproductive).
            # Only p and alpha carry information to the corrector.
            self.bridge.set_state(p, alpha=alpha, h_l=h_l_old, h_v=h_v_old, mdot=mdot)
            self.bridge.evaluate()

            drho_dp_c = self.bridge.get('drho_dp')
            rho_face_c = self.bridge.get('rho_face')
            Gamma_c = self.bridge.get('Gamma')
            T_l_c = self.bridge.get('T_l')
            T_sat_c = self.bridge.get('T_sat_cell')
            rho_l_c = self.bridge.get('rho_l')
            rho_v_c = self.bridge.get('rho_v')
            h_sat_l_c = self.bridge.get('h_sat_l')
            h_sat_v_c = self.bridge.get('h_sat_v')
            Phi2_c = self.bridge.get('Phi2') if self.bridge.has('Phi2') else np.ones(N + 1)

            # Re-compute friction at corrected state
            fric_c = np.zeros(N + 1)
            sigma_c = np.zeros(N + 1)
            for i in range(N + 1):
                phi2_i = Phi2_c[min(i, len(Phi2_c) - 1)]
                if rho_face_c[i] > 0.01 and np.isfinite(mdot[i]):
                    f = (phi2_i * K_geom * abs(mdot[i]) * mdot[i]
                         / (rho_face_c[i] * self.A_flow**2))
                    fric_c[i] = f if np.isfinite(f) else 0.0
                    sigma_c[i] = (2 * dt * phi2_i * K_geom * abs(mdot[i])
                                  / (rho_face_c[i] * self.A_flow**2))
            beta_eff_c = beta / (1.0 + sigma_c)

            # Re-assemble pressure tridiagonal with corrected drho_dp
            a_c = np.zeros(N); b_c = np.zeros(N)
            c_c = np.zeros(N); d_c = np.zeros(N)

            outlet_choked_c = self.use_critical_flow and mdot[N] > 0
            mdot_crit_c = 1e10
            if self.use_critical_flow and self.bridge.has('mdot_crit'):
                arr = self.bridge.get('mdot_crit')
                mdot_crit_c = arr[0] if len(arr) > 0 else 1e10
                mdot_crit_c *= getattr(self, 'C_d_factor', 1.0)

            for i in range(N):
                alpha_coeff_c = self.V_cell * drho_dp_c[i] / dt
                bL = 0.0 if (self.inlet_closed and i == 0) else (
                    0.0 if i == 0 else beta_eff_c[i])
                bR = 0.0 if (i == N - 1 and outlet_choked_c) else beta_eff_c[i + 1]

                a_c[i] = -bL if i > 0 else 0.0
                c_c[i] = -bR if i < N - 1 else 0.0
                b_c[i] = alpha_coeff_c + bL + bR
                d_c[i] = alpha_coeff_c * p_old[i]

                d_c[i] += (mdot_old[i] - mdot_old[i + 1])
                d_c[i] -= dt * (fric_c[i] / (1.0 + sigma_c[i])
                                - fric_c[i + 1] / (1.0 + sigma_c[i + 1]))

                if i == N - 1:
                    if outlet_choked_c:
                        d_c[i] += (mdot_old[N] - mdot_crit_c)
                    else:
                        d_c[i] += bR * self.p_out

                # Semi-implicit void-pressure coupling
                if Gamma_c[i] > 0 and T_l_c[i] > T_sat_c[i]:
                    rv_i = max(rho_v_c[i], 0.01)
                    superheat = max(T_l_c[i] - T_sat_c[i], 0.1)
                    h_fg = max(h_sat_v_c[i] - h_sat_l_c[i], 1.0)
                    dTsat_dp = T_sat_c[i] * (1.0 / rv_i - 1.0 / max(rho_l_c[i], 1.0)) / h_fg
                    dGamma_dp = -Gamma_c[i] * dTsat_dp / superheat
                    void_diag = -self.V_cell * (rho_l_c[i] - rho_v_c[i]) / rv_i * dGamma_dp
                    if void_diag > 0:
                        b_c[i] += void_diag
                        d_c[i] += void_diag * p_old[i]

            p[:] = self._thomas_solve(a_c, b_c, c_c, d_c)
            for i in range(N):
                if not np.isfinite(p[i]):
                    p[i] = p_old[i]
                p[i] = max(self.bridge.p_min, min(self.bridge.p_max, p[i]))

            # Re-do momentum
            mdot[0] = 0.0
            for i in range(1, N):
                mdot[i] = (mdot_old[i] + beta_eff_c[i] * (p[i - 1] - p[i])
                           - dt * fric_c[i] / (1.0 + sigma_c[i]))
            mdot_mom_c = (mdot_old[N] + beta_eff_c[N] * (p[N - 1] - self.p_out)
                          - dt * fric_c[N] / (1.0 + sigma_c[N]))
            if self.use_critical_flow and mdot_mom_c > 0:
                mdot[N] = min(mdot_mom_c, mdot_crit_c)
            else:
                mdot[N] = mdot_mom_c
