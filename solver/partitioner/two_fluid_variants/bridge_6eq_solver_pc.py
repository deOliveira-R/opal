"""
bridge_6eq_solver_pc.py -- 6-equation two-fluid solver with inner predictor-corrector.

ALL physics evaluation from OM-generated C via the equation bridge.
The solver provides ONLY the semi-implicit numerical method.

Design:
  Pressure diagonal uses a BLENDED compressibility:
    - At small alpha (onset): Schur complement (isenthalpic + rho_l/rho_v boost)
      allows void nucleation and wave propagation
    - At moderate alpha: h_mix compressibility (correct two-phase wave speed,
      stable for long transients)
  Transition at ALPHA_MID with linear blend, same as 5-eq production solver.

  The predictor-corrector iterates between pressure solve and void update:
    1. Evaluate bridge at current state
    2. Solve pressure with blended diagonal -> p_new
    3. Update momentum from p_new (2x2 Cramer per face)
    4. Update void fraction (conservative alpha*rho_v transport)
    5. Repeat from step 1 with updated (p, alpha, mdot)
    6. After convergence, update enthalpies once

  Key improvement over single-pass: the bridge re-evaluation at the predicted
  state updates Gamma (interfacial mass transfer), rho_v (vapor density), and
  alpha_face (for momentum coefficients). This is most valuable during the
  onset period (0-10ms in Edwards) when the state changes rapidly.

State variables: p[N], alpha[N], h_l[N], h_v[N], mdot_l[N+1], mdot_v[N+1]
"""

import numpy as np

from .codegen.equation_bridge import OMEquationBridge


class BridgeTwoFluidSolverPC:
    """6-equation two-fluid solver with inner predictor-corrector iteration."""

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 reconstruction='donor_cell',
                 n_iterations=2,
                 pc_tol=1e-3,
                 alpha_max=0.95,
                 alpha_mid=0.05,
                 freeze_diagonal=False):
        """
        Args:
            bridge: OMEquationBridge with compiled Modelica model
            spec: Pipe1DGridSpec (geometry, BCs)
            es: EquationSystem from XML
            reconstruction: 'donor_cell' (1st order)
            n_iterations: max predictor-corrector iterations (1=predictor only)
            pc_tol: relative pressure convergence tolerance for early exit
            alpha_max: maximum void fraction
            alpha_mid: blend transition point (Schur -> h_mix)
            freeze_diagonal: if True, freeze compressibility at old-state values
                during iteration (only update RHS through void/momentum)
        """
        self.reconstruction = reconstruction
        self.bridge = bridge
        self.N = bridge.N
        self.spec = spec
        self.n_iterations = max(1, n_iterations)
        self.pc_tol = pc_tol
        self.alpha_max = alpha_max
        self.alpha_mid = alpha_mid
        self.freeze_diagonal = freeze_diagonal

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

        bridge.set_params_from_spec(spec, es=es)

        # Diagnostics (set after each step)
        self.last_iterations = 0
        self.last_dp_residual = 0.0

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

    def _evaluate_bridge(self, p, alpha, h_l, h_v, mdot_l, mdot_v):
        """Evaluate the bridge and return all needed physics quantities."""
        self.bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v,
                              mdot_l=mdot_l, mdot_v=mdot_v)
        self.bridge.evaluate()

        props = {}
        props['drho_dp'] = self.bridge.get('drho_dp')  # h_mix compressibility
        props['rho_l'] = self.bridge.get('rho_l')
        props['rho_v'] = self.bridge.get('rho_v')
        props['Gamma'] = self.bridge.get('Gamma')
        props['q_i_l'] = self.bridge.get('q_i_l')
        props['q_i_v'] = self.bridge.get('q_i_v')
        props['h_sat_l'] = self.bridge.get('h_sat_l')
        props['h_sat_v'] = self.bridge.get('h_sat_v')

        props['alpha_face'] = self.bridge.get('alpha_face')
        props['rho_l_face'] = self.bridge.get('rho_l_face')
        props['rho_v_face'] = self.bridge.get('rho_v_face')
        props['F_drag'] = self.bridge.get('F_drag')
        props['v_l'] = self.bridge.get('v_l')
        props['v_v'] = self.bridge.get('v_v')

        # Isenthalpic phasic compressibility
        if self.bridge.has('drho_l_dp') and self.bridge.has('drho_v_dp'):
            props['drho_l_dp'] = self.bridge.get('drho_l_dp')
            props['drho_v_dp'] = self.bridge.get('drho_v_dp')
        else:
            props['drho_l_dp'] = None
            props['drho_v_dp'] = None

        # T_l / T_sat for dGamma/dp augmentation
        props['T_l'] = self.bridge.get('T_l') if self.bridge.has('T_l') else None
        props['T_sat'] = self.bridge.get('T_sat_cell') if self.bridge.has('T_sat_cell') else None

        # Critical flow
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            mdot_crit_arr = self.bridge.get('mdot_crit')
            props['mdot_crit'] = mdot_crit_arr[0] if len(mdot_crit_arr) > 0 else 1e10
            props['mdot_crit'] *= getattr(self, 'C_d_factor', 1.0)
        else:
            props['mdot_crit'] = 1e10

        return props

    def _compute_face_coefficients(self, props, mdot_l_ref, mdot_v_ref, dt):
        """Compute per-face momentum coefficients (2x2 coupled block)."""
        N = self.N
        EPS = 1e-6

        alpha_face = props['alpha_face']
        rho_l_face = props['rho_l_face']
        rho_v_face = props['rho_v_face']
        F_drag_face = props['F_drag']
        v_l_face = props['v_l']
        v_v_face = props['v_v']

        K_geom = self.f_D * self.dx / (2 * self.D_h)
        V_face = self.dx * self.A_flow
        beta = dt * self.A_flow / self.dx

        fric_l = np.zeros(N + 1)
        fric_v = np.zeros(N + 1)
        sigma_fric_l = np.zeros(N + 1)
        sigma_fric_v = np.zeros(N + 1)
        sigma_drag_l = np.zeros(N + 1)
        sigma_drag_v = np.zeros(N + 1)
        mat_a_ll = np.zeros(N + 1)
        mat_a_vv = np.zeros(N + 1)
        mat_det = np.zeros(N + 1)
        alpha_l_face = np.zeros(N + 1)
        alpha_v_face = np.zeros(N + 1)

        for i in range(N + 1):
            af = alpha_face[i]
            rl_f = max(rho_l_face[i], 0.01)
            rv_f = max(rho_v_face[i], 0.01)
            al_f = max(1 - af, EPS)
            av_f = max(af, EPS)
            alpha_l_face[i] = al_f
            alpha_v_face[i] = av_f

            # Per-phase wall friction
            if rl_f > 0.01 and np.isfinite(mdot_l_ref[i]):
                fric_l[i] = K_geom * abs(mdot_l_ref[i]) * mdot_l_ref[i] / (
                    al_f * rl_f * self.A_flow**2)
                sigma_fric_l[i] = 2 * dt * K_geom * abs(mdot_l_ref[i]) / (
                    al_f * rl_f * self.A_flow**2)

            if rv_f > 0.01 and np.isfinite(mdot_v_ref[i]):
                fric_v[i] = K_geom * abs(mdot_v_ref[i]) * mdot_v_ref[i] / (
                    av_f * rv_f * self.A_flow**2)
                sigma_fric_v[i] = 2 * dt * K_geom * abs(mdot_v_ref[i]) / (
                    av_f * rv_f * self.A_flow**2)

            # Drag linearization
            v_rel = v_v_face[i] - v_l_face[i] if (
                np.isfinite(v_v_face[i]) and np.isfinite(v_l_face[i])) else 0.0
            F_d = F_drag_face[i] if np.isfinite(F_drag_face[i]) else 0.0
            v_rel_abs = max(abs(v_rel), 1e-6)
            K_drag = 2 * abs(F_d) / v_rel_abs

            rho_f = max((1 - af) * rl_f + af * rv_f, 0.01)
            sigma_drag = 2 * dt * K_drag * V_face / (rho_f * self.A_flow**2 * self.dx)
            sigma_drag_l[i] = sigma_drag
            sigma_drag_v[i] = sigma_drag

            # 2x2 matrix diagonal (including phase-absence boost)
            a_ll = 1.0 + sigma_fric_l[i] + sigma_drag_l[i]
            a_vv = 1.0 + sigma_fric_v[i] + sigma_drag_v[i]

            if al_f < 0.05:
                a_ll += (0.05 - al_f) / 0.05 * 200.0
            if av_f < 0.05:
                a_vv += (0.05 - av_f) / 0.05 * 200.0

            mat_a_ll[i] = a_ll
            mat_a_vv[i] = a_vv
            mat_det[i] = max(a_ll * a_vv - sigma_drag_l[i] * sigma_drag_v[i], 1e-20)

        face = {
            'fric_l': fric_l, 'fric_v': fric_v,
            'sigma_fric_l': sigma_fric_l, 'sigma_fric_v': sigma_fric_v,
            'sigma_drag_l': sigma_drag_l, 'sigma_drag_v': sigma_drag_v,
            'mat_a_ll': mat_a_ll, 'mat_a_vv': mat_a_vv, 'mat_det': mat_det,
            'alpha_l_face': alpha_l_face, 'alpha_v_face': alpha_v_face,
            'beta': beta,
        }
        return face

    def _solve_pressure(self, p_old, alpha_work, mdot_l_work, mdot_v_work,
                        props, face, dt, outlet_choked):
        """Assemble and solve the pressure tridiagonal system.

        Uses blended compressibility:
          At small alpha: Schur (isenthalpic + rho_l/rho_v boost on vapor)
          At moderate alpha: h_mix (correct two-phase wave speed)
        Blend at alpha_mid with linear ramp, same as 5-eq production solver.

        Also includes dGamma/dp diagonal augmentation for semi-implicit
        void-pressure coupling through the Clausius-Clapeyron relation.
        """
        N = self.N
        beta = face['beta']
        mat_a_ll = face['mat_a_ll']
        mat_a_vv = face['mat_a_vv']
        mat_det = face['mat_det']
        alpha_l_face = face['alpha_l_face']
        alpha_v_face = face['alpha_v_face']
        sigma_drag_l = face['sigma_drag_l']
        sigma_drag_v = face['sigma_drag_v']
        fric_l = face['fric_l']
        fric_v = face['fric_v']
        mdot_crit = props['mdot_crit']

        drho_dp = props['drho_dp']      # h_mix compressibility
        drho_l_dp = props['drho_l_dp']  # isenthalpic phasic
        drho_v_dp = props['drho_v_dp']
        rho_l = props['rho_l']
        rho_v = props['rho_v']
        Gamma = props['Gamma']
        h_sat_l = props['h_sat_l']
        h_sat_v = props['h_sat_v']
        T_l = props['T_l']
        T_sat = props['T_sat']

        has_phasic = drho_l_dp is not None and drho_v_dp is not None

        # beta_total per face (mixture)
        beta_total = np.zeros(N + 1)
        for i in range(N + 1):
            beta_total[i] = beta * (
                alpha_l_face[i] * (mat_a_vv[i] + sigma_drag_l[i])
                + alpha_v_face[i] * (mat_a_ll[i] + sigma_drag_v[i])
            ) / mat_det[i]

        # Explicit friction correction per face (mixture)
        corr = np.zeros(N + 1)
        for i in range(N + 1):
            R_l_0 = -dt * fric_l[i]
            R_v_0 = -dt * fric_v[i]
            corr[i] = ((mat_a_vv[i] + sigma_drag_l[i]) * R_l_0
                      + (mat_a_ll[i] + sigma_drag_v[i]) * R_v_0
                      ) / mat_det[i]

        # Assemble tridiagonal
        a_tri = np.zeros(N); b_tri = np.zeros(N)
        c_tri = np.zeros(N); d_tri = np.zeros(N)

        for i in range(N):
            al = alpha_work[i]
            rv_i = max(rho_v[i], 0.01)
            rl_i = max(rho_l[i], 1.0)

            # ── Blended compressibility: Schur at onset, h_mix at moderate alpha ──
            if has_phasic:
                # Schur: rho_l/rho_v boost on vapor compressibility (same as 5-eq)
                drho_schur = ((1 - al) * drho_l_dp[i]
                              + al * (rl_i / rv_i) * drho_v_dp[i])

                # Linear blend: Schur at alpha=0 -> h_mix at alpha=alpha_mid
                blend = min(al / self.alpha_mid, 1.0)
                drho_eff = (1 - blend) * drho_schur + blend * drho_dp[i]
            else:
                drho_eff = drho_dp[i]

            drho_eff = max(drho_eff, 1e-10)

            # dGamma/dp augmentation (Clausius-Clapeyron semi-implicit coupling)
            void_diag = 0.0
            if T_l is not None and T_sat is not None:
                if Gamma[i] > 0 and T_l[i] > T_sat[i]:
                    superheat = max(T_l[i] - T_sat[i], 0.1)
                    h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
                    dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / rl_i) / h_fg
                    dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                    void_diag = -self.V_cell * (rl_i - rv_i) / rv_i * dGamma_dp
                    if void_diag < 0:
                        void_diag = 0.0

            alpha_coeff = self.V_cell * drho_eff / dt + void_diag

            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_total[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_total[i + 1]

            a_tri[i] = -bL if i > 0 else 0.0
            c_tri[i] = -bR if i < N - 1 else 0.0
            b_tri[i] = alpha_coeff + bL + bR
            d_tri[i] = alpha_coeff * p_old[i]

            # ── RHS: mixture mass flux ──
            mdot_total_in = mdot_l_work[i] + mdot_v_work[i]
            mdot_total_out = mdot_l_work[i + 1] + mdot_v_work[i + 1]
            d_tri[i] += (mdot_total_in - mdot_total_out)

            # ── Explicit friction correction (mixture) ──
            corr_in = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else corr[i])
            corr_out = 0.0 if (i == N - 1 and outlet_choked) else corr[i + 1]
            d_tri[i] += (corr_in - corr_out)

            if i == N - 1:
                if outlet_choked:
                    d_tri[i] += (mdot_total_out - mdot_crit)
                else:
                    d_tri[i] += bR * self.p_out

        p_new = self._thomas_solve(a_tri, b_tri, c_tri, d_tri)

        # Clamp
        for i in range(N):
            if not np.isfinite(p_new[i]):
                p_new[i] = p_old[i]
            p_new[i] = max(self.bridge.p_min, min(self.bridge.p_max, p_new[i]))

        return p_new

    def _update_momentum(self, p, mdot_l_old, mdot_v_old, face, dt, outlet_choked, mdot_crit):
        """Update phasic momentum from new pressure (2x2 Cramer per face)."""
        N = self.N
        beta = face['beta']
        alpha_l_face = face['alpha_l_face']
        alpha_v_face = face['alpha_v_face']
        mat_a_ll = face['mat_a_ll']
        mat_a_vv = face['mat_a_vv']
        mat_det = face['mat_det']
        sigma_drag_l = face['sigma_drag_l']
        sigma_drag_v = face['sigma_drag_v']
        fric_l = face['fric_l']
        fric_v = face['fric_v']

        mdot_l = np.zeros(N + 1)
        mdot_v = np.zeros(N + 1)

        mdot_l[0] = 0.0
        mdot_v[0] = 0.0

        for i in range(1, N):
            dp_face = p[i - 1] - p[i]
            R_l = beta * alpha_l_face[i] * dp_face - dt * fric_l[i]
            R_v = beta * alpha_v_face[i] * dp_face - dt * fric_v[i]

            Delta_l = (R_l * mat_a_vv[i] + sigma_drag_v[i] * R_v) / mat_det[i]
            Delta_v = (mat_a_ll[i] * R_v + sigma_drag_l[i] * R_l) / mat_det[i]

            mdot_l[i] = mdot_l_old[i] + Delta_l
            mdot_v[i] = mdot_v_old[i] + Delta_v

        # Outlet face
        dp_out = p[N - 1] - self.p_out
        R_l_out = beta * alpha_l_face[N] * dp_out - dt * fric_l[N]
        R_v_out = beta * alpha_v_face[N] * dp_out - dt * fric_v[N]

        Delta_l_out = (R_l_out * mat_a_vv[N] + sigma_drag_v[N] * R_v_out) / mat_det[N]
        Delta_v_out = (mat_a_ll[N] * R_v_out + sigma_drag_l[N] * R_l_out) / mat_det[N]

        mdot_l_mom = mdot_l_old[N] + Delta_l_out
        mdot_v_mom = mdot_v_old[N] + Delta_v_out

        if self.use_critical_flow:
            mdot_total_out = mdot_l_mom + mdot_v_mom
            if mdot_total_out > 0 and mdot_total_out > mdot_crit:
                ratio = mdot_crit / mdot_total_out
                mdot_l[N] = mdot_l_mom * ratio
                mdot_v[N] = mdot_v_mom * ratio
            else:
                mdot_l[N] = mdot_l_mom
                mdot_v[N] = mdot_v_mom
        else:
            mdot_l[N] = mdot_l_mom
            mdot_v[N] = mdot_v_mom

        return mdot_l, mdot_v

    def _update_void(self, alpha_old, rho_v, Gamma, mdot_v, dt):
        """Conservative void fraction transport."""
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
            alpha_val = max(ALPHA_MIN, min(ALPHA_MAX, alpha_rho_v_new / rv_new))
            if Gamma[i] > 0:
                alpha_val = max(alpha_val, 1e-3)
            alpha_new[i] = alpha_val
        return alpha_new

    def step(self, p, alpha, h_l, h_v, mdot_l, mdot_v, dt):
        """One semi-implicit timestep with inner predictor-corrector.

        Modifies all arrays in-place.
        """
        N = self.N

        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_l_old = mdot_l.copy()
        mdot_v_old = mdot_v.copy()

        # Set simulation time for time-dependent BCs
        self.bridge.set_time(getattr(self, 'time', 0.0))

        # ================================================================
        # PREDICTOR-CORRECTOR ITERATION
        #
        # The bridge is re-evaluated at each iteration with the predicted
        # state. This updates:
        #   - Gamma (interfacial mass transfer) at new pressure/void
        #   - rho_v, rho_l at new pressure (for void transport denominator)
        #   - alpha_face, rho_*_face (for momentum coupling)
        #   - drho_*_dp (compressibility at new state)
        #   - mdot_crit (critical flow at new state)
        #
        # Most valuable during onset (0-10ms): single-phase cells suddenly
        # nucleate, and the bridge at old state doesn't know about the new
        # void. Re-evaluation catches this within the timestep.
        # ================================================================

        # Working copies that evolve through iterations
        p_work = p_old.copy()
        alpha_work = alpha_old.copy()
        mdot_l_work = mdot_l_old.copy()
        mdot_v_work = mdot_v_old.copy()

        n_iter_done = 0
        dp_residual = float('inf')
        props_frozen = None  # For freeze_diagonal mode

        for iteration in range(self.n_iterations):
            # ── Evaluate bridge at current working state ──
            props = self._evaluate_bridge(p_work, alpha_work, h_l_old, h_v_old,
                                          mdot_l_work, mdot_v_work)

            # ── Freeze diagonal: use first iteration's compressibility for all ──
            if self.freeze_diagonal:
                if iteration == 0:
                    props_frozen = {
                        'drho_dp': props['drho_dp'].copy(),
                        'drho_l_dp': props['drho_l_dp'].copy() if props['drho_l_dp'] is not None else None,
                        'drho_v_dp': props['drho_v_dp'].copy() if props['drho_v_dp'] is not None else None,
                    }
                else:
                    # Use frozen compressibility but fresh Gamma, rho, etc.
                    props['drho_dp'] = props_frozen['drho_dp']
                    props['drho_l_dp'] = props_frozen['drho_l_dp']
                    props['drho_v_dp'] = props_frozen['drho_v_dp']

            # ── Compute face coefficients ──
            face = self._compute_face_coefficients(props, mdot_l_work, mdot_v_work, dt)

            # ── Determine outlet choking ──
            outlet_choked = False
            if self.use_critical_flow:
                outlet_choked = (mdot_l_work[N] + mdot_v_work[N]) > 0

            # ── Solve pressure ──
            p_new = self._solve_pressure(p_old, alpha_work, mdot_l_work, mdot_v_work,
                                         props, face, dt, outlet_choked)

            # ── Update momentum from new pressure ──
            mdot_l_new, mdot_v_new = self._update_momentum(
                p_new, mdot_l_old, mdot_v_old, face, dt,
                outlet_choked, props['mdot_crit'])

            # ── Update void fraction ──
            alpha_new = self._update_void(alpha_old, props['rho_v'],
                                          props['Gamma'], mdot_v_new, dt)

            # ── Convergence check ──
            dp_max = np.max(np.abs(p_new - p_work))
            p_scale = max(np.max(np.abs(p_new)), 1e3)
            dp_residual = dp_max / p_scale

            n_iter_done = iteration + 1

            # Update working state for next iteration
            p_work = p_new.copy()
            alpha_work = alpha_new.copy()
            mdot_l_work = mdot_l_new.copy()
            mdot_v_work = mdot_v_new.copy()

            if dp_residual < self.pc_tol and iteration > 0:
                break

        # Store diagnostics
        self.last_iterations = n_iter_done
        self.last_dp_residual = dp_residual

        # ── Commit converged pressure, void, and momentum ──
        p[:] = p_work
        alpha[:] = alpha_work
        mdot_l[:] = mdot_l_work
        mdot_v[:] = mdot_v_work

        # ── Phasic energy (done ONCE with converged state) ──
        dp_dt = (p - p_old) / dt
        rho_l = props['rho_l']
        rho_v = props['rho_v']
        q_i_l = props['q_i_l']
        q_i_v = props['q_i_v']
        h_sat_l = props['h_sat_l']
        h_sat_v = props['h_sat_v']
        Gamma = props['Gamma']

        for i in range(N):
            al = alpha_old[i]

            m_l = max((1 - al) * rho_l[i] * self.V_cell, 1e-12)
            if (1 - al) > 1e-6:
                ml_in = mdot_l[i]; ml_out = mdot_l[i + 1]
                h_in = h_l_old[i - 1] if (i > 0 and ml_in >= 0) else h_l_old[i]
                h_out = h_l_old[i] if ml_out >= 0 else (
                    h_l_old[i + 1] if i < N - 1 else h_l_old[i])
                flux = ml_in * (h_in - h_l_old[i]) - ml_out * (h_out - h_l_old[i])
                pw = (1 - al) * self.V_cell * dp_dt[i]
                qi = q_i_l[i] * self.V_cell
                phase = -Gamma[i] * h_l_old[i] * self.V_cell
                h_l[i] = h_l_old[i] + dt / m_l * (flux + pw + qi + phase)
                h_l[i] = max(1e4, min(h_l[i], h_sat_v[i]))
            else:
                h_l[i] = h_sat_l[i]

            m_v = max(al * rho_v[i] * self.V_cell, 1e-12)
            if al > 1e-6:
                mv_in = mdot_v[i]; mv_out = mdot_v[i + 1]
                hv_in = h_v_old[i - 1] if (i > 0 and mv_in >= 0) else h_v_old[i]
                hv_out = h_v_old[i] if mv_out >= 0 else (
                    h_v_old[i + 1] if i < N - 1 else h_v_old[i])
                flux_v = mv_in * (hv_in - h_v_old[i]) - mv_out * (hv_out - h_v_old[i])
                pw_v = al * self.V_cell * dp_dt[i]
                qi_v = q_i_v[i] * self.V_cell
                phase_v = Gamma[i] * h_v_old[i] * self.V_cell
                h_v[i] = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(h_sat_v[i], min(h_v[i], 4e6))
            else:
                h_v[i] = h_sat_v[i]
