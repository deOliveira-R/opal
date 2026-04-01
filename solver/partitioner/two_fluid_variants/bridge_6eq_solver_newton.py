"""
bridge_6eq_solver_newton.py — 6-equation two-fluid solver with Newton iteration
on the pressure equation, with re-evaluation of bridge physics at each iteration.

ALL physics evaluation (properties, closures, interfacial drag, wall friction)
from OM-generated C via the equation bridge. The solver provides ONLY the
semi-implicit numerical method.

State variables: p[N], alpha[N], h_l[N], h_v[N], mdot_l[N+1], mdot_v[N+1]

Three iteration modes:
  mode='pressure': Newton on pressure only, void explicit after convergence
  mode='gauss_seidel': Alternate pressure solve -> void update -> re-evaluate bridge
  mode='coupled': (removed, unstable for this problem)

Two compressibility models:
  compressibility='isenthalpic': phasic drho_l/dp, drho_v/dp from rho_ph FD + Schur
  compressibility='hmix': bridge drho_dp (thermal equilibrium, no Schur RHS)
  compressibility='hmix_schur': bridge drho_dp + Schur RHS (unstable with iteration)

Edwards blowdown results (0.6s, dt=50us, N=24, alpha_max=0.95):
  Baseline single-pass (h_mix+Schur+dGamma):       41.2% MAPE, 750 steps/s
  Newton GS-1 isenthalpic:                          51.2% MAPE, 346 steps/s
  Newton GS-3 isenthalpic:                          52.7% MAPE, 205 steps/s
  Newton P-1 isenthalpic:                           46.2% MAPE, 267 steps/s

NEGATIVE RESULT: Newton iteration does not improve over single-pass for this
problem. Root cause: isenthalpic compressibility gives correct mechanical wave
speed (~1500 m/s) but misses thermal equilibration that reduces the effective
two-phase sound speed (~50-200 m/s). The h_mix compressibility captures this
thermal coupling implicitly. Newton cannot recover the missing physics.
"""

import numpy as np

from ..codegen.equation_bridge import OMEquationBridge


class BridgeTwoFluidSolverNewton:
    """6-equation two-fluid solver with iterative pressure-void coupling."""

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 reconstruction='donor_cell',
                 break_form_loss=False,
                 alpha_max=0.95,
                 max_iterations=3,
                 mode='gauss_seidel',
                 compressibility='isenthalpic'):
        self.reconstruction = reconstruction
        self.bridge = bridge
        self.N = bridge.N
        self.spec = spec
        self.break_form_loss = break_form_loss
        self.alpha_max = alpha_max
        self.max_iterations = max_iterations
        self.mode = mode
        self.compressibility = compressibility  # 'isenthalpic' or 'hmix'

        self.dx = spec.dx
        self.A_flow = spec.A_flow
        self.D_h = spec.D_h
        self.f_D = spec.f_D
        self.V_cell = spec.V_cell

        self.inlet_closed = getattr(spec, 'inlet_closed', True)
        self.p_out = getattr(spec, 'p_out', 101325.0) or 101325.0

        self.use_critical_flow = getattr(spec, 'use_critical_flow', False)
        if not self.use_critical_flow and bridge.has('mdot_crit'):
            self.use_critical_flow = True

        self._cd_eff_idx = None
        cd_name = f'{bridge.prefix}.C_d_eff'
        if cd_name in bridge.info.all_vars:
            self._cd_eff_idx = bridge.info.all_vars[cd_name].index

        self._rho_ph_fn = bridge._rho_ph_fn

        bridge.set_params_from_spec(spec, es=es)

        # Diagnostics
        self.last_iterations = 0
        self.last_dp_max = 0.0

    def _compute_phasic_drho_dp(self, p, h_l, h_v):
        """Compute drho_l/dp and drho_v/dp via finite difference on rho_ph.

        Uses the OM-compiled media function (all physics from Modelica).
        """
        N = self.N
        drho_l_dp = np.zeros(N)
        drho_v_dp = np.zeros(N)
        dp_fd = 100.0  # Pa

        if self._rho_ph_fn is not None:
            for i in range(N):
                rho_l_plus = self._rho_ph_fn(p[i] + dp_fd, h_l[i])
                rho_l_minus = self._rho_ph_fn(p[i] - dp_fd, h_l[i])
                drho_l_dp[i] = max((rho_l_plus - rho_l_minus) / (2 * dp_fd), 1e-10)

                rho_v_plus = self._rho_ph_fn(p[i] + dp_fd, h_v[i])
                rho_v_minus = self._rho_ph_fn(p[i] - dp_fd, h_v[i])
                drho_v_dp[i] = max((rho_v_plus - rho_v_minus) / (2 * dp_fd), 1e-10)
        else:
            for i in range(N):
                drho_l_dp[i] = 4.5e-7 * 800.0
                drho_v_dp[i] = max(p[i] / (461.5 * 600), 0.1) / max(p[i], 1e4)

        return drho_l_dp, drho_v_dp

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

    def _evaluate_bridge_and_get_properties(self, p, alpha, h_l, h_v,
                                             mdot_l, mdot_v):
        """Evaluate bridge at given state and return all needed properties."""
        self.bridge.set_time(getattr(self, 'time', 0.0))
        self.bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v,
                              mdot_l=mdot_l, mdot_v=mdot_v)
        self.bridge.evaluate()

        props = {}
        props['rho_l'] = self.bridge.get('rho_l')
        props['rho_v'] = self.bridge.get('rho_v')
        props['Gamma'] = self.bridge.get('Gamma')
        props['q_i_l'] = self.bridge.get('q_i_l')
        props['q_i_v'] = self.bridge.get('q_i_v')
        props['h_sat_l'] = self.bridge.get('h_sat_l')
        props['h_sat_v'] = self.bridge.get('h_sat_v')
        props['drho_dp'] = self.bridge.get('drho_dp')

        props['alpha_face'] = self.bridge.get('alpha_face')
        props['rho_l_face'] = self.bridge.get('rho_l_face')
        props['rho_v_face'] = self.bridge.get('rho_v_face')
        props['F_drag'] = self.bridge.get('F_drag')
        props['v_l'] = self.bridge.get('v_l')
        props['v_v'] = self.bridge.get('v_v')

        if self.bridge.has('T_l'):
            props['T_l'] = self.bridge.get('T_l')
        if self.bridge.has('T_sat_cell'):
            props['T_sat'] = self.bridge.get('T_sat_cell')

        # Critical flow
        props['mdot_crit'] = 1e10
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            arr = self.bridge.get('mdot_crit')
            props['mdot_crit'] = arr[0] if len(arr) > 0 else 1e10
            props['mdot_crit'] *= getattr(self, 'C_d_factor', 1.0)

        return props

    def _compute_momentum_coefficients(self, mdot_l, mdot_v,
                                       alpha_face, rho_l_face, rho_v_face,
                                       F_drag_face, v_l_face, v_v_face,
                                       dt):
        """Compute per-face 2x2 momentum coupling coefficients."""
        N = self.N
        EPS = 1e-6
        K_geom = self.f_D * self.dx / (2 * self.D_h)
        V_face = self.dx * self.A_flow

        fric_l = np.zeros(N + 1)
        fric_v = np.zeros(N + 1)
        sigma_fric_l = np.zeros(N + 1)
        sigma_fric_v = np.zeros(N + 1)
        sigma_drag_l = np.zeros(N + 1)
        sigma_drag_v = np.zeros(N + 1)
        mat_a_ll = np.zeros(N + 1)
        mat_a_vv = np.zeros(N + 1)
        mat_det = np.zeros(N + 1)
        al_face = np.zeros(N + 1)
        av_face = np.zeros(N + 1)

        for i in range(N + 1):
            af = alpha_face[i]
            rl_f = max(rho_l_face[i], 0.01)
            rv_f = max(rho_v_face[i], 0.01)
            al_f = max(1 - af, EPS)
            av_f = max(af, EPS)
            al_face[i] = al_f
            av_face[i] = av_f

            if rl_f > 0.01 and np.isfinite(mdot_l[i]):
                fric_l[i] = K_geom * abs(mdot_l[i]) * mdot_l[i] / (
                    al_f * rl_f * self.A_flow**2)
                sigma_fric_l[i] = 2 * dt * K_geom * abs(mdot_l[i]) / (
                    al_f * rl_f * self.A_flow**2)

            if rv_f > 0.01 and np.isfinite(mdot_v[i]):
                fric_v[i] = K_geom * abs(mdot_v[i]) * mdot_v[i] / (
                    av_f * rv_f * self.A_flow**2)
                sigma_fric_v[i] = 2 * dt * K_geom * abs(mdot_v[i]) / (
                    av_f * rv_f * self.A_flow**2)

            if self.break_form_loss and i == N:
                C_d_current = 1.0
                if self._cd_eff_idx is not None:
                    C_d_current = self.bridge.lib.opal_bridge_get_var(self._cd_eff_idx)
                if not np.isfinite(C_d_current) or C_d_current < 1e-4:
                    C_d_current = 1e-4
                K_break = max(1.0 / C_d_current**2 - 1.0, 0.0)
                if K_break > 0:
                    mdot_total_out = mdot_l[N] + mdot_v[N]
                    rho_m_f = max(al_f * rl_f + av_f * rv_f, 0.01)
                    if np.isfinite(mdot_total_out):
                        fric_break = K_break * abs(mdot_total_out) * mdot_total_out / (
                            rho_m_f * self.A_flow**2)
                        sigma_break = 2 * dt * K_break * abs(mdot_total_out) / (
                            rho_m_f * self.A_flow**2)
                        if np.isfinite(fric_break):
                            fric_l[N] += al_f * fric_break
                            fric_v[N] += av_f * fric_break
                            sigma_fric_l[N] += al_f * sigma_break
                            sigma_fric_v[N] += av_f * sigma_break

            v_rel = v_v_face[i] - v_l_face[i] if (
                np.isfinite(v_v_face[i]) and np.isfinite(v_l_face[i])) else 0.0
            F_d = F_drag_face[i] if np.isfinite(F_drag_face[i]) else 0.0
            v_rel_abs = max(abs(v_rel), 1e-6)
            K_drag = 2 * abs(F_d) / v_rel_abs

            rho_f = max((1 - af) * rl_f + af * rv_f, 0.01)
            sigma_drag = 2 * dt * K_drag * V_face / (rho_f * self.A_flow**2 * self.dx)
            sigma_drag_l[i] = sigma_drag
            sigma_drag_v[i] = sigma_drag

            a_ll = 1.0 + sigma_fric_l[i] + sigma_drag_l[i]
            a_vv = 1.0 + sigma_fric_v[i] + sigma_drag_v[i]
            if al_f < 0.05:
                a_ll += (0.05 - al_f) / 0.05 * 200.0
            if av_f < 0.05:
                a_vv += (0.05 - av_f) / 0.05 * 200.0

            mat_a_ll[i] = a_ll
            mat_a_vv[i] = a_vv
            mat_det[i] = max(a_ll * a_vv - sigma_drag_l[i] * sigma_drag_v[i],
                             1e-20)

        return (fric_l, fric_v, sigma_fric_l, sigma_fric_v,
                sigma_drag_l, sigma_drag_v, mat_a_ll, mat_a_vv, mat_det,
                al_face, av_face)

    def _solve_pressure(self, p_old, alpha, rho_l, rho_v, drho_l_dp, drho_v_dp,
                        drho_dp_hmix,
                        Gamma, h_sat_l, h_sat_v, T_l, T_sat,
                        mdot_l_old, mdot_v_old, mdot_crit,
                        beta_total, beta_v, corr, corr_v,
                        outlet_choked, dt):
        """Assemble and solve the pressure tridiagonal with Schur augmentation.

        compressibility='isenthalpic': phasic drho/dp + Schur void coupling
        compressibility='hmix': bridge drho_dp (thermal, proven at 40.9%)

        Returns new pressure array.
        """
        N = self.N
        beta = dt * self.A_flow / self.dx

        a_tri = np.zeros(N); b_tri = np.zeros(N)
        c_tri = np.zeros(N); d_tri = np.zeros(N)

        for i in range(N):
            al = alpha[i]
            rv_i = max(rho_v[i], 0.01)
            rl_i = max(rho_l[i], 1.0)

            # Face coupling
            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_total[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_total[i + 1]

            if self.compressibility == 'hmix':
                # h_mix compressibility (thermal): proven stable at 40.9%
                # No Schur coupling — h_mix already captures thermal effects
                drho_mech = drho_dp_hmix[i]
                void_coupling = 0.0
                use_schur_rhs = False
            elif self.compressibility == 'hmix_schur':
                # h_mix diagonal WITH Schur RHS — matches baseline exactly
                drho_mech = drho_dp_hmix[i]
                void_coupling = 0.0
                use_schur_rhs = True
            else:
                # Isenthalpic phasic + Schur void coupling
                # Term 1: mechanical compressibility
                drho_mech = (1 - al) * drho_l_dp[i] + al * drho_v_dp[i]

                # Term 2: Schur void-pressure coupling
                void_coupling = (rl_i - rv_i) / rv_i * al * drho_v_dp[i]
                use_schur_rhs = True

            # Term 3: Gamma-pressure coupling (Clausius-Clapeyron)
            gamma_coupling = 0.0
            if T_l is not None and T_sat is not None:
                if T_l[i] > T_sat[i] and Gamma[i] > 0:
                    superheat = max(T_l[i] - T_sat[i], 0.1)
                    h_fg = max(h_sat_v[i] - h_sat_l[i], 1e3)
                    dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / rl_i) / h_fg
                    dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                    gamma_coupling = (rl_i - rv_i) / rv_i * abs(dGamma_dp) * dt

            drho_eff = drho_mech + void_coupling + gamma_coupling
            alpha_coeff = self.V_cell * drho_eff / dt

            a_tri[i] = -bL if i > 0 else 0.0
            c_tri[i] = -bR if i < N - 1 else 0.0
            b_tri[i] = alpha_coeff + bL + bR
            d_tri[i] = alpha_coeff * p_old[i]

            # RHS: mixture mass residual
            mdot_total_in = mdot_l_old[i] + mdot_v_old[i]
            mdot_total_out = mdot_l_old[i + 1] + mdot_v_old[i + 1]
            d_tri[i] += (mdot_total_in - mdot_total_out)

            # Schur RHS: void coupling contribution
            if use_schur_rhs:
                flux_v_old = mdot_v_old[i] - mdot_v_old[i + 1]
                R2 = flux_v_old + self.V_cell * Gamma[i]
                schur_coeff = (rl_i - rv_i) / rv_i
                d_tri[i] += schur_coeff * R2

            # Explicit friction correction
            corr_in = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else corr[i])
            corr_out = 0.0 if (i == N - 1 and outlet_choked) else corr[i + 1]
            d_tri[i] += (corr_in - corr_out)

            # Outlet BC
            if i == N - 1:
                if outlet_choked:
                    d_tri[i] += (mdot_total_out - mdot_crit)
                else:
                    d_tri[i] += bR * self.p_out

        p_new = self._thomas_solve(a_tri, b_tri, c_tri, d_tri)

        for i in range(N):
            if not np.isfinite(p_new[i]):
                p_new[i] = p_old[i]
            p_new[i] = max(self.bridge.p_min, min(self.bridge.p_max, p_new[i]))

        return p_new

    def _update_void(self, alpha_old, rho_v, Gamma, mdot_v, dt):
        """Explicit conservative void fraction update."""
        N = self.N
        ALPHA_MIN = 1e-4
        ALPHA_MAX = self.alpha_max
        alpha_new = np.zeros(N)

        for i in range(N):
            al = alpha_old[i]
            rv = max(rho_v[i], 0.01)
            flux_v = mdot_v[i] - mdot_v[i + 1]
            alpha_rho_v_new = al * rv + dt / self.V_cell * (flux_v + self.V_cell * Gamma[i])
            rv_new = max(rho_v[i], 0.01)
            alpha_new_i = max(ALPHA_MIN, min(ALPHA_MAX, alpha_rho_v_new / rv_new))
            if Gamma[i] > 0:
                alpha_new_i = max(alpha_new_i, 1e-3)
            alpha_new[i] = alpha_new_i

        return alpha_new

    def step(self, p, alpha, h_l, h_v, mdot_l, mdot_v, dt):
        """One semi-implicit timestep with iterative pressure-void coupling.

        Modifies all arrays in-place.
        """
        N = self.N

        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_l_old = mdot_l.copy()
        mdot_v_old = mdot_v.copy()

        beta = dt * self.A_flow / self.dx

        # ═══════════════════════════════════════════════════════════════
        # ITERATIVE PRESSURE-VOID COUPLING
        # ═══════════════════════════════════════════════════════════════
        # Working copies for iteration
        p_k = p_old.copy()
        alpha_k = alpha_old.copy()
        mdot_l_k = mdot_l_old.copy()
        mdot_v_k = mdot_v_old.copy()

        for k in range(self.max_iterations):
            # ── Evaluate bridge at current iterate ──
            props = self._evaluate_bridge_and_get_properties(
                p_k, alpha_k, h_l, h_v, mdot_l_k, mdot_v_k)

            rho_l = props['rho_l']
            rho_v = props['rho_v']
            Gamma = props['Gamma']
            h_sat_l = props['h_sat_l']
            h_sat_v = props['h_sat_v']
            T_l = props.get('T_l')
            T_sat = props.get('T_sat')
            mdot_crit = props['mdot_crit']

            outlet_choked = (mdot_l_old[N] + mdot_v_old[N]) > 0

            # Phasic compressibility from rho_ph (all physics from Modelica)
            drho_l_dp, drho_v_dp = self._compute_phasic_drho_dp(p_k, h_l, h_v)

            # Momentum coefficients
            mom = self._compute_momentum_coefficients(
                mdot_l_old, mdot_v_old,
                props['alpha_face'], props['rho_l_face'], props['rho_v_face'],
                props['F_drag'], props['v_l'], props['v_v'], dt)
            (fric_l, fric_v, sigma_fric_l, sigma_fric_v,
             sigma_drag_l, sigma_drag_v, mat_a_ll, mat_a_vv, mat_det,
             al_face, av_face) = mom

            # Beta coefficients
            beta_total = np.zeros(N + 1)
            beta_v = np.zeros(N + 1)
            for i in range(N + 1):
                beta_total[i] = beta * (
                    al_face[i] * (mat_a_vv[i] + sigma_drag_l[i])
                    + av_face[i] * (mat_a_ll[i] + sigma_drag_v[i])
                ) / mat_det[i]
                beta_v[i] = beta * (
                    mat_a_ll[i] * av_face[i]
                    + sigma_drag_l[i] * al_face[i]
                ) / mat_det[i]

            # Friction corrections
            corr = np.zeros(N + 1)
            corr_v = np.zeros(N + 1)
            for i in range(N + 1):
                R_l_0 = -dt * fric_l[i]
                R_v_0 = -dt * fric_v[i]
                corr[i] = ((mat_a_vv[i] + sigma_drag_l[i]) * R_l_0
                           + (mat_a_ll[i] + sigma_drag_v[i]) * R_v_0
                           ) / mat_det[i]
                corr_v[i] = (mat_a_ll[i] * R_v_0
                             + sigma_drag_l[i] * R_l_0
                             ) / mat_det[i]

            # ── Solve pressure (Schur-augmented tridiagonal) ──
            drho_dp_hmix = props['drho_dp']
            p_new = self._solve_pressure(
                p_old, alpha_k, rho_l, rho_v, drho_l_dp, drho_v_dp,
                drho_dp_hmix,
                Gamma, h_sat_l, h_sat_v, T_l, T_sat,
                mdot_l_old, mdot_v_old, mdot_crit,
                beta_total, beta_v, corr, corr_v,
                outlet_choked, dt)

            # ── Update momentum at new pressure ──
            mdot_l_new = np.zeros(N + 1)
            mdot_v_new = np.zeros(N + 1)

            for i in range(1, N):
                dp_face = p_new[i - 1] - p_new[i]
                R_l = beta * al_face[i] * dp_face - dt * fric_l[i]
                R_v = beta * av_face[i] * dp_face - dt * fric_v[i]
                Delta_l = (R_l * mat_a_vv[i] + sigma_drag_v[i] * R_v) / mat_det[i]
                Delta_v = (mat_a_ll[i] * R_v + sigma_drag_l[i] * R_l) / mat_det[i]
                mdot_l_new[i] = mdot_l_old[i] + Delta_l
                mdot_v_new[i] = mdot_v_old[i] + Delta_v

            # Outlet
            dp_out = p_new[N - 1] - self.p_out
            R_l_out = beta * al_face[N] * dp_out - dt * fric_l[N]
            R_v_out = beta * av_face[N] * dp_out - dt * fric_v[N]
            Delta_l_out = (R_l_out * mat_a_vv[N] + sigma_drag_v[N] * R_v_out) / mat_det[N]
            Delta_v_out = (mat_a_ll[N] * R_v_out + sigma_drag_l[N] * R_l_out) / mat_det[N]
            mdot_l_mom = mdot_l_old[N] + Delta_l_out
            mdot_v_mom = mdot_v_old[N] + Delta_v_out

            if self.use_critical_flow:
                mdot_total_out = mdot_l_mom + mdot_v_mom
                if mdot_total_out > 0 and mdot_total_out > mdot_crit:
                    ratio = mdot_crit / mdot_total_out
                    mdot_l_new[N] = mdot_l_mom * ratio
                    mdot_v_new[N] = mdot_v_mom * ratio
                else:
                    mdot_l_new[N] = mdot_l_mom
                    mdot_v_new[N] = mdot_v_mom
            else:
                mdot_l_new[N] = mdot_l_mom
                mdot_v_new[N] = mdot_v_mom

            # ── Update void (explicit conservative) ──
            if self.mode in ('gauss_seidel', 'coupled'):
                alpha_new = self._update_void(
                    alpha_old, rho_v, Gamma, mdot_v_new, dt)
            else:
                # pressure-only mode: void stays at alpha_k for pressure iterations
                alpha_new = alpha_k.copy()

            # ── Convergence check ──
            dp_max = np.max(np.abs(p_new - p_k) / np.maximum(np.abs(p_k), 1e5))

            # Update iterates for next round
            p_k = p_new.copy()
            alpha_k = alpha_new.copy()
            mdot_l_k = mdot_l_new.copy()
            mdot_v_k = mdot_v_new.copy()

            self.last_dp_max = dp_max
            if dp_max < 1e-4:
                self.last_iterations = k + 1
                break
        else:
            self.last_iterations = self.max_iterations

        # ═══════════════════════════════════════════════════════════════
        # CONVERGED: Write final state
        # ═══════════════════════════════════════════════════════════════
        p[:] = p_k

        # Final void update (always, even in pressure-only mode)
        if self.mode == 'pressure':
            # Do one final void update with converged pressure
            props = self._evaluate_bridge_and_get_properties(
                p_k, alpha_old, h_l, h_v, mdot_l_k, mdot_v_k)
            alpha[:] = self._update_void(
                alpha_old, props['rho_v'], props['Gamma'], mdot_v_k, dt)
        else:
            alpha[:] = alpha_k

        mdot_l[:] = mdot_l_k
        mdot_v[:] = mdot_v_k

        # ── Re-evaluate bridge at final state for energy update ──
        props_final = self._evaluate_bridge_and_get_properties(
            p, alpha, h_l, h_v, mdot_l, mdot_v)

        rho_l_conv = props_final['rho_l']
        rho_v_conv = props_final['rho_v']
        Gamma_conv = props_final['Gamma']
        q_i_l = props_final['q_i_l']
        q_i_v = props_final['q_i_v']
        h_sat_l_conv = props_final['h_sat_l']
        h_sat_v_conv = props_final['h_sat_v']

        # ── Phasic energy (explicit, using old enthalpies) ──
        dp_dt = (p - p_old) / dt

        for i in range(N):
            al = alpha_old[i]

            m_l = max((1 - al) * rho_l_conv[i] * self.V_cell, 1e-12)
            if (1 - al) > 1e-6:
                ml_in = mdot_l[i]; ml_out = mdot_l[i + 1]
                h_in = h_l_old[i - 1] if (i > 0 and ml_in >= 0) else h_l_old[i]
                h_out = h_l_old[i] if ml_out >= 0 else (
                    h_l_old[i + 1] if i < N - 1 else h_l_old[i])
                flux = ml_in * (h_in - h_l_old[i]) - ml_out * (h_out - h_l_old[i])
                pw = (1 - al) * self.V_cell * dp_dt[i]
                qi = q_i_l[i] * self.V_cell
                phase = -Gamma_conv[i] * h_l_old[i] * self.V_cell
                h_l[i] = h_l_old[i] + dt / m_l * (flux + pw + qi + phase)
                h_l[i] = max(1e4, min(h_l[i], h_sat_v_conv[i]))
            else:
                h_l[i] = h_sat_l_conv[i]

            m_v = max(al * rho_v_conv[i] * self.V_cell, 1e-12)
            if al > 1e-6:
                mv_in = mdot_v[i]; mv_out = mdot_v[i + 1]
                hv_in = h_v_old[i - 1] if (i > 0 and mv_in >= 0) else h_v_old[i]
                hv_out = h_v_old[i] if mv_out >= 0 else (
                    h_v_old[i + 1] if i < N - 1 else h_v_old[i])
                flux_v = mv_in * (hv_in - h_v_old[i]) - mv_out * (hv_out - h_v_old[i])
                pw_v = al * self.V_cell * dp_dt[i]
                qi_v = q_i_v[i] * self.V_cell
                phase_v = Gamma_conv[i] * h_v_old[i] * self.V_cell
                h_v[i] = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(h_sat_v_conv[i], min(h_v[i], 4e6))
            else:
                h_v[i] = h_sat_v_conv[i]
