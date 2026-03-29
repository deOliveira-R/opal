"""
bridge_5eq_solver.py — 5-equation drift-flux semi-implicit solver using the OM bridge.

ALL physics evaluation (properties, closures, friction multiplier, interfacial
transfer) from OM-generated C via the equation bridge. The solver provides
ONLY the semi-implicit numerical method.

State variables: p[N], alpha[N], h_l[N], h_v[N], mdot[N+1]
Bridge provides: rho_l, rho_v, rho_m, rho_face, drho_dp, drho_dh,
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
                 reconstruction='donor_cell'):
        """
        Args:
            bridge: OMEquationBridge with compiled Modelica model
            spec: Pipe1DGridSpec (geometry, BCs)
            es: EquationSystem from XML — pass this to load ALL parameter values
                from Modelica (H_i, C_0, etc.). Without es, only geometry is set.
            reconstruction: 'donor_cell' (1st order) or 'muscl' (2nd order, minmod TVD)
        """
        self.reconstruction = reconstruction
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
        drho_dh = self.bridge.get('drho_dh')
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

        # ── Pressure tridiagonal (with implicit friction resistance) ──
        a_tri = np.zeros(N); b_tri = np.zeros(N)
        c_tri = np.zeros(N); d_tri = np.zeros(N)

        for i in range(N):
            alpha_coeff = self.V_cell * drho_dp[i] / dt

            # Face coupling using beta_eff (no floors needed)
            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_eff[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_eff[i + 1]

            a_tri[i] = -bL if i > 0 else 0.0
            c_tri[i] = -bR if i < N - 1 else 0.0
            b_tri[i] = alpha_coeff + bL + bR
            d_tri[i] = alpha_coeff * p_old[i]

            # RHS: mass residual + friction (with implicit friction correction)
            d_tri[i] += (mdot_old[i] - mdot_old[i + 1])
            d_tri[i] -= dt * (fric[i] / (1.0 + sigma[i])
                            - fric[i + 1] / (1.0 + sigma[i + 1]))

            if i == N - 1:
                if outlet_choked:
                    d_tri[i] += (mdot_old[N] - mdot_crit)
                else:
                    d_tri[i] += bR * self.p_out

            # Semi-implicit void-pressure coupling.
            # Evaporation (Gamma > 0) creates void → changes mixture density.
            # The explicit void update runs ahead of the implicit pressure solve.
            # Linearize Gamma w.r.t. pressure to add stabilizing diagonal term:
            #   Gamma(p_new) ≈ Gamma + dGamma/dp * dp
            #   dGamma/dp ≈ -Gamma * dT_sat/dp / max(T_l - T_sat, eps)
            #   (higher p → higher T_sat → less superheat → less Gamma)
            # Void source contribution to mass conservation:
            #   S_void = V * (rho_l - rho_v) / rho_v * Gamma  [kg/s]
            # Semi-implicit: move dS/dp to diagonal (stabilizing, positive):
            #   void_diag = -V * (rho_l - rho_v) / rho_v * dGamma/dp
            # Only active during evaporation (Gamma > 0, T_l > T_sat).
            # At baseline (weak Gamma), this term is negligible.
            if Gamma[i] > 0 and T_l[i] > T_sat[i]:
                rv_i = max(rho_v[i], 0.01)
                superheat = max(T_l[i] - T_sat[i], 0.1)
                h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
                # Clausius-Clapeyron: dT_sat/dp ≈ T_sat * (1/rho_v - 1/rho_l) / h_fg
                dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / max(rho_l[i], 1.0)) / h_fg
                # dGamma/dp < 0 (more pressure → less superheat → less evaporation)
                dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                # Diagonal contribution (positive → stabilizing)
                void_diag = -self.V_cell * (rho_l[i] - rho_v[i]) / rv_i * dGamma_dp
                if void_diag > 0:
                    b_tri[i] += void_diag
                    d_tri[i] += void_diag * p_old[i]

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

        # ── Drift-flux phasic mass flows from bridge (if available) ──
        # The Modelica model computes mdot_v/mdot_l via drift-flux algebraic slip.
        # If available, use them; otherwise fall back to HEM-like mdot*alpha.
        has_drift_flux = self.bridge.has('mdot_v') and self.bridge.has('mdot_l')
        if has_drift_flux:
            mdot_v_face = self.bridge.get('mdot_v')
            mdot_l_face = self.bridge.get('mdot_l')
        else:
            mdot_v_face = None
            mdot_l_face = None

        # ── Void fraction transport (conservative form, explicit) ──
        # Update alpha*rho_v as a product, then extract alpha by dividing by new rho_v.
        # This is the conservative form that correctly handles rapid rho_v changes
        # during depressurization. Ref: C++ prototype five_eq_model.cpp:394-399.
        for i in range(N):
            al = alpha_old[i]
            rv = max(rho_v[i], 0.01)

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

            # Conservative: update alpha*rho_v product
            alpha_rho_v_new = al * rv + dt / self.V_cell * (flux_v + self.V_cell * Gamma[i])

            # Extract alpha using rho_v. Using old-pressure rho_v for now;
            # new-pressure evaluation accelerates void growth too aggressively
            # with the current explicit scheme, causing runaway depressurization.
            rv_new = max(rho_v[i], 0.01)
            alpha_new = max(0.0, min(1.0, alpha_rho_v_new / rv_new))

            # Nucleation floor: when flashing is active, maintain minimum void
            # to prevent advective washout of the nucleation seed.
            # Ref: C++ prototype five_eq_model.cpp:415-417.
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
