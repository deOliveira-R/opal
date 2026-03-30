"""
bridge_5eq_solver_v2_twostage.py — Variant 2: Two-stage pressure solve.

Decouples void onset from wave propagation using two sequential pressure solves:

  Stage 1 (onset detection):
    Solve pressure with PHASIC drho_dp = (1-alpha)*drho_l_dp + alpha*drho_v_dp.
    This is small (~5e-7 in single-phase liquid), so depressurization proceeds
    past saturation without the 2400x h_mix compressibility jump that freezes it.
    Update momentum from Stage 1 pressure.

  Intermediate void update:
    Use Stage 1 pressure and momentum to update void fraction via the same
    conservative alpha*rho_v product method as the base solver.

  Stage 2 (wave propagation):
    Re-evaluate bridge at (p_stage1, alpha_intermediate, h_l_old, h_v_old).
    The bridge now sees the actual void state and returns an appropriate drho_dp
    (mixture compressibility reflecting the new void fraction). Re-solve pressure
    with this drho_dp. Update momentum from Stage 2 pressure.

  Energy update:
    Use Stage 2 pressure for dp/dt and Stage 2 momentum for advection.

ALL physics evaluation (properties, closures, friction multiplier, interfacial
transfer) from OM-generated C via the equation bridge. The solver provides
ONLY the semi-implicit numerical method.

State variables: p[N], alpha[N], h_l[N], h_v[N], mdot[N+1]
Bridge provides: rho_l, rho_v, rho_m, rho_face, drho_dp, drho_l_dp, drho_v_dp,
    Gamma, q_i_l, q_i_v, Phi2, T_l, T_sat_cell, h_sat_l, h_sat_v, h_mix
"""

import numpy as np

from .codegen.equation_bridge import OMEquationBridge


class BridgeDriftFluxSolver:
    """5-equation drift-flux solver with two-stage pressure solve.

    Stage 1 uses phasic drho_dp (small, allows void onset).
    Stage 2 re-evaluates at updated void and uses the resulting drho_dp
    (correct wave speed). Physics from OM-generated C; numerics from Python.
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 reconstruction='donor_cell'):
        """
        Args:
            bridge: OMEquationBridge with compiled Modelica model
            spec: Pipe1DGridSpec (geometry, BCs)
            es: EquationSystem from XML -- pass this to load ALL parameter values
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

    def _compute_friction(self, mdot_ref, rho_face, Phi2, dt, K_geom):
        """Compute friction force and implicit resistance coefficient.

        Returns (fric, sigma, beta_eff) arrays.
        """
        N = self.N
        fric = np.zeros(N + 1)
        sigma = np.zeros(N + 1)

        for i in range(N + 1):
            phi2_i = Phi2[min(i, len(Phi2) - 1)]
            if rho_face[i] > 0.01 and np.isfinite(mdot_ref[i]):
                f = (phi2_i * K_geom * abs(mdot_ref[i]) * mdot_ref[i]
                     / (rho_face[i] * self.A_flow**2))
                fric[i] = f if np.isfinite(f) else 0.0
                sigma[i] = (2 * dt * phi2_i * K_geom * abs(mdot_ref[i])
                            / (rho_face[i] * self.A_flow**2))

        beta = dt * self.A_flow / self.dx
        beta_eff = beta / (1.0 + sigma)
        return fric, sigma, beta_eff

    def _assemble_pressure_tridiagonal(self, drho_dp_vals, beta_eff, fric, sigma,
                                       p_old, mdot_old, outlet_choked, mdot_crit,
                                       Gamma, T_l, T_sat, rho_l, rho_v,
                                       h_sat_l, h_sat_v):
        """Assemble the pressure tridiagonal system.

        Args:
            drho_dp_vals: compressibility array (N,) -- phasic for Stage 1, bridge for Stage 2
            beta_eff: effective coupling coefficients (N+1,)
            fric, sigma: friction arrays (N+1,)
            p_old, mdot_old: state at beginning of timestep
            outlet_choked: whether outlet is choked
            mdot_crit: critical mass flow rate
            Gamma, T_l, T_sat, rho_l, rho_v, h_sat_l, h_sat_v: bridge quantities

        Returns:
            (a, b, c, d) tridiagonal arrays
        """
        N = self.N
        dt = self._current_dt  # set by step() before calling

        a_tri = np.zeros(N)
        b_tri = np.zeros(N)
        c_tri = np.zeros(N)
        d_tri = np.zeros(N)

        for i in range(N):
            # Face coupling using beta_eff (no floors needed)
            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_eff[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_eff[i + 1]

            a_tri[i] = -bL if i > 0 else 0.0
            c_tri[i] = -bR if i < N - 1 else 0.0

            alpha_coeff = self.V_cell * drho_dp_vals[i] / dt

            b_tri[i] = alpha_coeff + bL + bR
            d_tri[i] = alpha_coeff * p_old[i]

            # RHS: mass residual + friction (with implicit friction correction)
            d_tri[i] += (mdot_old[i] - mdot_old[i + 1])
            d_tri[i] -= dt * (fric[i] / (1.0 + sigma[i])
                              - fric[i + 1] / (1.0 + sigma[i + 1]))

            # Semi-implicit void-pressure coupling (ad-hoc diagonal term)
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

            # Outlet BC
            if i == N - 1:
                if outlet_choked:
                    d_tri[i] += (mdot_old[N] - mdot_crit)
                else:
                    d_tri[i] += bR * self.p_out

        return a_tri, b_tri, c_tri, d_tri

    def _solve_pressure(self, drho_dp_vals, beta_eff, fric, sigma,
                        p_old, mdot_old, outlet_choked, mdot_crit,
                        Gamma, T_l, T_sat, rho_l, rho_v,
                        h_sat_l, h_sat_v):
        """Assemble and solve the pressure tridiagonal. Returns new pressure array."""
        a, b, c, d = self._assemble_pressure_tridiagonal(
            drho_dp_vals, beta_eff, fric, sigma,
            p_old, mdot_old, outlet_choked, mdot_crit,
            Gamma, T_l, T_sat, rho_l, rho_v, h_sat_l, h_sat_v)

        p_new = self._thomas_solve(a, b, c, d)

        # Sanitize
        for i in range(self.N):
            if not np.isfinite(p_new[i]):
                p_new[i] = p_old[i]
            p_new[i] = max(self.bridge.p_min, min(self.bridge.p_max, p_new[i]))

        return p_new

    def _update_momentum(self, p_new, beta_eff, fric, sigma, mdot_old,
                         outlet_choked, mdot_crit):
        """Update momentum from new pressure. Returns new mdot array."""
        N = self.N
        mdot_new = np.zeros(N + 1)

        # Inlet wall BC
        mdot_new[0] = 0.0

        # Interior faces
        for i in range(1, N):
            mdot_new[i] = (mdot_old[i] + beta_eff[i] * (p_new[i - 1] - p_new[i])
                           - self._current_dt * fric[i] / (1.0 + sigma[i]))

        # Outlet with critical flow limiter
        mdot_mom = (mdot_old[N] + beta_eff[N] * (p_new[N - 1] - self.p_out)
                    - self._current_dt * fric[N] / (1.0 + sigma[N]))
        if self.use_critical_flow and mdot_mom > 0:
            mdot_new[N] = min(mdot_mom, mdot_crit)
        else:
            mdot_new[N] = mdot_mom

        return mdot_new

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One semi-implicit timestep with two-stage pressure solve.

        Stage 1: Phasic drho_dp -> pressure solve -> momentum update
        Intermediate: Void fraction update using Stage 1 results
        Stage 2: Re-evaluate bridge -> new drho_dp -> pressure solve -> momentum update
        Energy: Use Stage 2 pressure and momentum
        """
        N = self.N
        self._current_dt = dt
        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_old = mdot.copy()

        # ==============================================================
        # PHYSICS EVALUATION -- ALL from OM-generated C
        # ==============================================================
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

        # Phasic compressibility for Stage 1
        # These are the individual phase derivatives drho_l/dp and drho_v/dp,
        # which are small (~5e-7) and do not have the 2400x h_mix jump.
        drho_l_dp = self.bridge.get('drho_l_dp')
        drho_v_dp = self.bridge.get('drho_v_dp')

        # ==============================================================
        # NUMERICAL METHOD ONLY BELOW
        # ==============================================================

        # -- Critical flow (from Modelica via bridge) --
        mdot_crit = 1e10
        outlet_choked = False
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            mdot_crit_arr = self.bridge.get('mdot_crit')
            mdot_crit = mdot_crit_arr[0] if len(mdot_crit_arr) > 0 else 1e10
            mdot_crit *= getattr(self, 'C_d_factor', 1.0)  # Legacy; prefer Modelica C_d_eff
            outlet_choked = mdot_old[N] > 0

        # -- Friction with implicit resistance --
        K_geom = self.f_D * self.dx / (2 * self.D_h)
        fric, sigma, beta_eff = self._compute_friction(
            mdot_old, rho_face, Phi2, dt, K_geom)

        # -- Drift-flux phasic mass flows from bridge (if available) --
        has_drift_flux = self.bridge.has('mdot_v') and self.bridge.has('mdot_l')
        if has_drift_flux:
            mdot_v_face = self.bridge.get('mdot_v')
            mdot_l_face = self.bridge.get('mdot_l')
        else:
            mdot_v_face = None
            mdot_l_face = None

        # ==============================================================
        # STAGE 1: Onset detection (phasic drho_dp)
        # ==============================================================
        # Phasic compressibility: weighted sum of individual phase derivatives.
        # In single-phase liquid (alpha~0): drho_phasic ~ drho_l_dp ~ 5e-7.
        # This is 2400x smaller than drho_dp(h_mix) at saturation, allowing
        # depressurization to proceed past saturation without stalling.
        drho_phasic = np.zeros(N)
        for i in range(N):
            drho_phasic[i] = ((1.0 - alpha_old[i]) * drho_l_dp[i]
                              + alpha_old[i] * drho_v_dp[i])

        p_stage1 = self._solve_pressure(
            drho_phasic, beta_eff, fric, sigma,
            p_old, mdot_old, outlet_choked, mdot_crit,
            Gamma, T_l, T_sat, rho_l, rho_v, h_sat_l, h_sat_v)

        mdot_stage1 = self._update_momentum(
            p_stage1, beta_eff, fric, sigma, mdot_old,
            outlet_choked, mdot_crit)

        # ==============================================================
        # INTERMEDIATE: Void fraction update using Stage 1 results
        # ==============================================================
        # Conservative: alpha*rho_v product update with old rho_v (avoids
        # void-density positive feedback). Same method as base solver.
        alpha_inter = np.zeros(N)
        for i in range(N):
            al = alpha_old[i]
            rv = max(rho_v[i], 0.01)

            if has_drift_flux:
                flux_v = mdot_v_face[i] - mdot_v_face[i + 1]
            else:
                if mdot_stage1[i] >= 0:
                    alpha_in = alpha_old[i - 1] if i > 0 else al
                else:
                    alpha_in = al
                if mdot_stage1[i + 1] >= 0:
                    alpha_out = al
                else:
                    alpha_out = alpha_old[i + 1] if i < N - 1 else al
                flux_v = (mdot_stage1[i] * alpha_in
                          - mdot_stage1[i + 1] * alpha_out)

            alpha_rho_v_new = (al * rv
                               + dt / self.V_cell * (flux_v + self.V_cell * Gamma[i]))
            rv_new = max(rho_v[i], 0.01)
            alpha_new = max(0.0, min(1.0, alpha_rho_v_new / rv_new))

            if Gamma[i] > 0:
                alpha_new = max(alpha_new, 1e-3)

            alpha_inter[i] = alpha_new

        # ==============================================================
        # STAGE 2: Wave propagation (bridge-evaluated drho_dp)
        # ==============================================================
        # Re-evaluate bridge at (p_stage1, alpha_intermediate, h_l_old, h_v_old).
        # The bridge now sees the actual void state from Stage 1 and returns
        # drho_dp reflecting the two-phase mixture. This corrects the wave speed.
        self.bridge.set_state(p_stage1, alpha=alpha_inter,
                              h_l=h_l_old, h_v=h_v_old, mdot=mdot_stage1)
        self.bridge.evaluate()

        drho_dp_s2 = self.bridge.get('drho_dp')
        rho_face_s2 = self.bridge.get('rho_face')
        rho_l_s2 = self.bridge.get('rho_l')
        rho_v_s2 = self.bridge.get('rho_v')
        Gamma_s2 = self.bridge.get('Gamma')
        T_l_s2 = self.bridge.get('T_l')
        T_sat_s2 = self.bridge.get('T_sat_cell')
        h_sat_l_s2 = self.bridge.get('h_sat_l')
        h_sat_v_s2 = self.bridge.get('h_sat_v')
        Phi2_s2 = self.bridge.get('Phi2') if self.bridge.has('Phi2') else np.ones(N + 1)

        # Re-read phasic flows for energy update (evaluated at Stage 2 state)
        if has_drift_flux:
            mdot_v_face_s2 = self.bridge.get('mdot_v')
            mdot_l_face_s2 = self.bridge.get('mdot_l')
        else:
            mdot_v_face_s2 = None
            mdot_l_face_s2 = None

        # Re-read critical flow at Stage 2 state
        mdot_crit_s2 = 1e10
        outlet_choked_s2 = False
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            arr = self.bridge.get('mdot_crit')
            mdot_crit_s2 = arr[0] if len(arr) > 0 else 1e10
            mdot_crit_s2 *= getattr(self, 'C_d_factor', 1.0)
            outlet_choked_s2 = mdot_stage1[N] > 0

        # Re-compute friction at Stage 2 state
        fric_s2, sigma_s2, beta_eff_s2 = self._compute_friction(
            mdot_stage1, rho_face_s2, Phi2_s2, dt, K_geom)

        # Stage 2 pressure solve with bridge-evaluated drho_dp
        p_stage2 = self._solve_pressure(
            drho_dp_s2, beta_eff_s2, fric_s2, sigma_s2,
            p_old, mdot_old, outlet_choked_s2, mdot_crit_s2,
            Gamma_s2, T_l_s2, T_sat_s2, rho_l_s2, rho_v_s2,
            h_sat_l_s2, h_sat_v_s2)

        mdot_stage2 = self._update_momentum(
            p_stage2, beta_eff_s2, fric_s2, sigma_s2, mdot_old,
            outlet_choked_s2, mdot_crit_s2)

        # ==============================================================
        # COMMIT: Write final state
        # ==============================================================
        # Pressure and momentum from Stage 2
        p[:] = p_stage2
        mdot[:] = mdot_stage2

        # Void fraction from intermediate step (driven by Stage 1)
        alpha[:] = alpha_inter

        # ==============================================================
        # PHASIC ENERGY (explicit, using _old enthalpy values)
        # ==============================================================
        # All advection uses h_l_old/h_v_old to prevent directional bias from
        # sequential cell updates. dp/dt uses Stage 2 pressure. Momentum uses
        # Stage 2 mdot for advection.
        dp_dt = (p - p_old) / dt

        # Use Stage 2 phasic flows and properties for energy update
        _mdot_v_face = mdot_v_face_s2 if has_drift_flux else None
        _mdot_l_face = mdot_l_face_s2 if has_drift_flux else None
        _rho_l = rho_l_s2
        _rho_v = rho_v_s2
        _q_i_l = self.bridge.get('q_i_l')
        _q_i_v = self.bridge.get('q_i_v')
        _Gamma = Gamma_s2
        _h_sat_l = h_sat_l_s2
        _h_sat_v = h_sat_v_s2

        for i in range(N):
            al = alpha_old[i]

            # Liquid energy
            m_l = max((1 - al) * _rho_l[i] * self.V_cell, 1e-12)
            if (1 - al) > 1e-6:
                if has_drift_flux:
                    ml_in = _mdot_l_face[i]
                    ml_out = _mdot_l_face[i + 1]
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
                qi = _q_i_l[i] * self.V_cell
                phase = -_Gamma[i] * h_l_old[i] * self.V_cell

                h_l[i] = h_l_old[i] + dt / m_l * (flux + pw + qi + phase)
                h_l[i] = max(1e4, min(h_l[i], _h_sat_v[i]))
            else:
                h_l[i] = _h_sat_l[i]

            # Vapour energy
            m_v = max(al * _rho_v[i] * self.V_cell, 1e-12)
            if al > 1e-6:
                if has_drift_flux:
                    mv_in = _mdot_v_face[i]
                    mv_out = _mdot_v_face[i + 1]
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
                qi_v = _q_i_v[i] * self.V_cell
                phase_v = _Gamma[i] * h_v_old[i] * self.V_cell

                h_v[i] = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(_h_sat_v[i], min(h_v[i], 4e6))  # Floor at h_sat_v, not 1e4
            else:
                h_v[i] = _h_sat_v[i]
