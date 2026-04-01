"""
bridge_6eq_solver.py — 6-equation two-fluid semi-implicit solver using the OM bridge.

ALL physics evaluation (properties, closures, interfacial drag, wall friction)
from OM-generated C via the equation bridge. The solver provides ONLY the
semi-implicit numerical method.

State variables: p[N], alpha[N], h_l[N], h_v[N], mdot_l[N+1], mdot_v[N+1]
Bridge provides: rho_l, rho_v, rho_m, rho_face, rho_l_face, rho_v_face,
    alpha_face, drho_dp, drho_dh, Gamma, q_i_l, q_i_v, F_drag, fric_l, fric_v,
    v_l, v_v, T_l, T_sat_cell, h_sat_l, h_sat_v, h_mix, mdot_crit

Derived in: derivations/two_fluid_momentum.py, derivations/interfacial_drag.py,
            derivations/two_fluid_coupled_momentum.py

Semi-implicit design:
  - Pressure coupling: beta = dt*A/dx (geometric, removes acoustic CFL — same as 5-eq)
  - Friction: implicit through sigma (standard semi-implicit treatment)
  - Drag: RELAP5-style 2x2 coupled block solve per face (explicit drag + implicit sigma)
  - Phase absence: sigma boost keeps absent-phase beta_eff → 0 smoothly
"""

import numpy as np

from .codegen.equation_bridge import OMEquationBridge


class BridgeTwoFluidSolver:
    """6-equation two-fluid solver driven by the OM equation bridge."""

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 reconstruction='donor_cell'):
        self.reconstruction = reconstruction
        self.bridge = bridge
        self.N = bridge.N
        self.spec = spec

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

    def step(self, p, alpha, h_l, h_v, mdot_l, mdot_v, dt):
        """One semi-implicit timestep. Modifies all arrays in-place."""
        N = self.N
        EPS = 1e-6
        ALPHA_MIN = 1e-4
        ALPHA_MAX = 0.95

        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_l_old = mdot_l.copy()
        mdot_v_old = mdot_v.copy()

        # ══════════════════════════════════════════════════════════
        # PHYSICS EVALUATION — ALL from OM-generated C
        # ══════════════════════════════════════════════════════════
        self.bridge.set_time(getattr(self, 'time', 0.0))
        self.bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v,
                              mdot_l=mdot_l, mdot_v=mdot_v)
        self.bridge.evaluate()

        drho_dp = self.bridge.get('drho_dp')
        rho_l = self.bridge.get('rho_l')
        rho_v = self.bridge.get('rho_v')
        Gamma = self.bridge.get('Gamma')
        q_i_l = self.bridge.get('q_i_l')
        q_i_v = self.bridge.get('q_i_v')
        h_sat_l = self.bridge.get('h_sat_l')
        h_sat_v = self.bridge.get('h_sat_v')

        alpha_face = self.bridge.get('alpha_face')
        rho_l_face = self.bridge.get('rho_l_face')
        rho_v_face = self.bridge.get('rho_v_face')
        F_drag_face = self.bridge.get('F_drag')
        v_l_face = self.bridge.get('v_l')
        v_v_face = self.bridge.get('v_v')

        # ══════════════════════════════════════════════════════════
        # NUMERICAL METHOD ONLY BELOW
        # ══════════════════════════════════════════════════════════

        # ── Critical flow ──
        mdot_crit = 1e10
        outlet_choked = False
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            mdot_crit_arr = self.bridge.get('mdot_crit')
            mdot_crit = mdot_crit_arr[0] if len(mdot_crit_arr) > 0 else 1e10
            mdot_crit *= getattr(self, 'C_d_factor', 1.0)
            outlet_choked = (mdot_l_old[N] + mdot_v_old[N]) > 0

        # ── Per-face coefficients (2x2 coupled block solve) ──
        # Derived in: derivations/two_fluid_coupled_momentum.py
        K_geom = self.f_D * self.dx / (2 * self.D_h)
        V_face = self.dx * self.A_flow
        beta = dt * self.A_flow / self.dx  # geometric, same as 5-eq

        fric_l = np.zeros(N + 1)
        fric_v = np.zeros(N + 1)
        sigma_fric_l_arr = np.zeros(N + 1)
        sigma_fric_v_arr = np.zeros(N + 1)
        sigma_drag_l_arr = np.zeros(N + 1)
        sigma_drag_v_arr = np.zeros(N + 1)
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

            # Per-phase wall friction (same as before)
            if rl_f > 0.01 and np.isfinite(mdot_l_old[i]):
                fric_l[i] = K_geom * abs(mdot_l_old[i]) * mdot_l_old[i] / (
                    al_f * rl_f * self.A_flow**2)
                sigma_fric_l_arr[i] = 2 * dt * K_geom * abs(mdot_l_old[i]) / (
                    al_f * rl_f * self.A_flow**2)

            if rv_f > 0.01 and np.isfinite(mdot_v_old[i]):
                fric_v[i] = K_geom * abs(mdot_v_old[i]) * mdot_v_old[i] / (
                    av_f * rv_f * self.A_flow**2)
                sigma_fric_v_arr[i] = 2 * dt * K_geom * abs(mdot_v_old[i]) / (
                    av_f * rv_f * self.A_flow**2)

            # Drag linearization — enters ONLY through sigma (implicit coupling)
            # NO explicit drag correction in R_l/R_v (physics reviewer recommendation)
            # Mixture sigma normalization for adequate coupling strength
            # (per-phase normalization gives sigma << 1, too weak)
            v_rel = v_v_face[i] - v_l_face[i] if (
                np.isfinite(v_v_face[i]) and np.isfinite(v_l_face[i])) else 0.0
            F_d = F_drag_face[i] if np.isfinite(F_drag_face[i]) else 0.0
            v_rel_abs = max(abs(v_rel), 1e-6)
            K_drag = 2 * abs(F_d) / v_rel_abs

            # Shared drag sigma (mixture normalization, same for both phases)
            rho_f = max((1 - af) * rl_f + af * rv_f, 0.01)
            sigma_drag = 2 * dt * K_drag * V_face / (rho_f * self.A_flow**2 * self.dx)
            sigma_drag_l_arr[i] = sigma_drag
            sigma_drag_v_arr[i] = sigma_drag

            # 2x2 matrix diagonal (including phase-absence boost)
            a_ll = 1.0 + sigma_fric_l_arr[i] + sigma_drag_l_arr[i]
            a_vv = 1.0 + sigma_fric_v_arr[i] + sigma_drag_v_arr[i]

            if al_f < 0.05:
                a_ll += (0.05 - al_f) / 0.05 * 200.0
            if av_f < 0.05:
                a_vv += (0.05 - av_f) / 0.05 * 200.0

            mat_a_ll[i] = a_ll
            mat_a_vv[i] = a_vv
            # det = a_ll*a_vv - sd_l*sd_v (always positive, see derivation)
            mat_det[i] = max(a_ll * a_vv - sigma_drag_l_arr[i] * sigma_drag_v_arr[i],
                             1e-20)

        # ── Pressure tridiagonal ──
        # beta_total = beta * [(1-alpha)*(a_vv + sd_l) + alpha*(a_ll + sd_v)] / det
        beta_total = np.zeros(N + 1)
        for i in range(N + 1):
            beta_total[i] = beta * (
                alpha_l_face[i] * (mat_a_vv[i] + sigma_drag_l_arr[i])
                + alpha_v_face[i] * (mat_a_ll[i] + sigma_drag_v_arr[i])
            ) / mat_det[i]

        # Explicit correction per face (dp=0 contribution from friction only)
        # No explicit drag in RHS — drag enters only through sigma
        corr = np.zeros(N + 1)
        for i in range(N + 1):
            R_l_0 = -dt * fric_l[i]
            R_v_0 = -dt * fric_v[i]
            corr[i] = ((mat_a_vv[i] + sigma_drag_l_arr[i]) * R_l_0
                      + (mat_a_ll[i] + sigma_drag_v_arr[i]) * R_v_0
                      ) / mat_det[i]

        a_tri = np.zeros(N); b_tri = np.zeros(N)
        c_tri = np.zeros(N); d_tri = np.zeros(N)

        for i in range(N):
            alpha_coeff = self.V_cell * drho_dp[i] / dt

            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_total[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_total[i + 1]

            a_tri[i] = -bL if i > 0 else 0.0
            c_tri[i] = -bR if i < N - 1 else 0.0
            b_tri[i] = alpha_coeff + bL + bR
            d_tri[i] = alpha_coeff * p_old[i]

            mdot_total_in = mdot_l_old[i] + mdot_v_old[i]
            mdot_total_out = mdot_l_old[i + 1] + mdot_v_old[i + 1]
            d_tri[i] += (mdot_total_in - mdot_total_out)

            # Explicit friction + drag correction from 2x2 block solve
            corr_in = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else corr[i])
            corr_out = 0.0 if (i == N - 1 and outlet_choked) else corr[i + 1]
            d_tri[i] += (corr_in - corr_out)

            if i == N - 1:
                if outlet_choked:
                    d_tri[i] += (mdot_total_out - mdot_crit)
                else:
                    d_tri[i] += bR * self.p_out

        p[:] = self._thomas_solve(a_tri, b_tri, c_tri, d_tri)

        for i in range(N):
            if not np.isfinite(p[i]):
                p[i] = p_old[i]
            p[i] = max(self.bridge.p_min, min(self.bridge.p_max, p[i]))

        # ── Phasic momentum updates (2x2 Cramer solve per face) ──
        # Derived in: derivations/two_fluid_coupled_momentum.py
        # Delta_l = (R_l*a_vv + sd_v*R_v) / det
        # Delta_v = (a_ll*R_v + sd_l*R_l) / det

        mdot_l[0] = 0.0
        mdot_v[0] = 0.0

        for i in range(1, N):
            dp_face = p[i - 1] - p[i]

            R_l = beta * alpha_l_face[i] * dp_face - dt * fric_l[i]
            R_v = beta * alpha_v_face[i] * dp_face - dt * fric_v[i]

            Delta_l = (R_l * mat_a_vv[i] + sigma_drag_v_arr[i] * R_v) / mat_det[i]
            Delta_v = (mat_a_ll[i] * R_v + sigma_drag_l_arr[i] * R_l) / mat_det[i]

            mdot_l[i] = mdot_l_old[i] + Delta_l
            mdot_v[i] = mdot_v_old[i] + Delta_v

        # Outlet face
        dp_out = p[N - 1] - self.p_out

        R_l_out = beta * alpha_l_face[N] * dp_out - dt * fric_l[N]
        R_v_out = beta * alpha_v_face[N] * dp_out - dt * fric_v[N]

        Delta_l_out = (R_l_out * mat_a_vv[N] + sigma_drag_v_arr[N] * R_v_out) / mat_det[N]
        Delta_v_out = (mat_a_ll[N] * R_v_out + sigma_drag_l_arr[N] * R_l_out) / mat_det[N]

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

        # ── Void fraction transport ──
        for i in range(N):
            al = alpha_old[i]
            rv = max(rho_v[i], 0.01)
            flux_v = mdot_v[i] - mdot_v[i + 1]
            alpha_rho_v_new = al * rv + dt / self.V_cell * (flux_v + self.V_cell * Gamma[i])
            rv_new = max(rho_v[i], 0.01)
            alpha_new = max(ALPHA_MIN, min(ALPHA_MAX, alpha_rho_v_new / rv_new))
            if Gamma[i] > 0:
                alpha_new = max(alpha_new, 1e-3)
            alpha[i] = alpha_new

        # ── Phasic energy ──
        dp_dt = (p - p_old) / dt

        for i in range(N):
            al = alpha_old[i]

            m_l = max((1 - al) * rho_l[i] * self.V_cell, 1e-12)
            if (1 - al) > 1e-6:
                ml_in = mdot_l[i]; ml_out = mdot_l[i + 1]
                h_in = h_l_old[i - 1] if (i > 0 and ml_in >= 0) else h_l_old[i]
                h_out = h_l_old[i] if ml_out >= 0 else (h_l_old[i + 1] if i < N - 1 else h_l_old[i])
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
                hv_out = h_v_old[i] if mv_out >= 0 else (h_v_old[i + 1] if i < N - 1 else h_v_old[i])
                flux_v = mv_in * (hv_in - h_v_old[i]) - mv_out * (hv_out - h_v_old[i])
                pw_v = al * self.V_cell * dp_dt[i]
                qi_v = q_i_v[i] * self.V_cell
                phase_v = Gamma[i] * h_v_old[i] * self.V_cell
                h_v[i] = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(h_sat_v[i], min(h_v[i], 4e6))
            else:
                h_v[i] = h_sat_v[i]
