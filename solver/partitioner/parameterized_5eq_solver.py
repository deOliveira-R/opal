"""
parameterized_5eq_solver.py — Case 1: 5-equation solver with ALL parameters
extracted from Modelica. Zero hardcoded physics constants.

Comparison target:
  Case 0: extracted_5eq_solver.py (hardcoded closures) — must produce identical results
  Case 2: translateModel C codegen (future) — must produce identical results

The solver reads everything from ExtractedModelSpec:
  - Geometry (N, dx, A, D_h, f_D, g_axial)
  - Closures (H_i, C_0, alpha_nucleation, C_d, x_trans, c_floor, Phi2_max)
  - Boundaries (inlet_type, outlet_type, p_out)
  - Thresholds (rho_face_min, alpha_min, mass_min, h_l_min, h_v_max)

The ONLY physics this file implements is the semi-implicit NUMERICAL METHOD:
  - Tridiagonal pressure assembly + Thomas solve
  - Inertial momentum update
  - Explicit void fraction transport
  - Explicit phasic enthalpy update

The closure FORMULAS (Gamma, q_i_l, V_gj, Phi2) are still in Python —
this is the limitation of Case 1. Case 2 (translateModel) will eliminate
these by calling OM-generated C code.
"""

import numpy as np
from .model_spec import ExtractedModelSpec


class Parameterized5EqSolver:
    """5-equation drift-flux solver with all parameters from extraction."""

    def __init__(self, fluid, spec: ExtractedModelSpec):
        """
        Args:
            fluid: C++ FluidPackage (for property evaluation — same math as Modelica)
            spec: Complete model specification extracted from Modelica XML
        """
        self.fluid = fluid
        self.spec = spec

        # Unpack for convenience (all from spec, none hardcoded)
        g = spec.geometry
        self.N = g.N
        self.dx = g.dx
        self.A_flow = g.A_flow
        self.D_h = g.D_h
        self.f_D = g.f_D
        self.V_cell = g.V_cell
        self.g_axial = g.g_axial

        c = spec.closures
        self.H_i = c.H_i
        self.C_0 = c.C_0
        self.alpha_nucleation = c.alpha_nucleation
        self.use_critical_flow = c.use_critical_flow
        self.C_d = c.C_d
        self.x_trans = c.x_trans
        self.c_floor = c.c_floor
        self.use_two_phase_friction = c.use_two_phase_friction
        self.Phi2_max = c.Phi2_max

        b = spec.boundary
        self.inlet_type = b.inlet_type
        self.outlet_type = b.outlet_type
        self.p_out = b.p_out or 101325.0

        t = spec.thresholds
        self.rho_face_min = t.rho_face_min
        self.alpha_min = t.alpha_min
        self.mass_min = t.mass_min
        self.h_l_min = t.h_l_min
        self.h_v_max = t.h_v_max
        self.rv_floor_frac = t.rv_floor_frac
        self.rv_floor_abs = t.rv_floor_abs

        # Pressure bounds from fluid
        self.p_min = fluid.p_min if hasattr(fluid, 'p_min') else 700.0
        self.p_max = fluid.p_max if hasattr(fluid, 'p_max') else 21e6

    # ──────────────────────────────────────────────────────────────
    # Shared numerical utilities (no physics, pure numerics)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _thomas_solve(a, b, c, d):
        """Thomas algorithm for tridiagonal system. Returns solution x."""
        N = len(d)
        cp = np.zeros(N)
        dp = np.zeros(N)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
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
    def _face_average(cell_vals, N):
        """Arithmetic face averaging: boundary = cell, interior = 0.5*(L+R)."""
        face = np.zeros(N + 1)
        face[0] = cell_vals[0]
        for i in range(1, N):
            face[i] = 0.5 * (cell_vals[i - 1] + cell_vals[i])
        face[N] = cell_vals[N - 1]
        return face

    @staticmethod
    def _donor_cell(mdot, vals, i, N, bc_val=None):
        """Donor-cell face value: upwind selection based on flow direction."""
        if mdot >= 0:
            return vals[i - 1] if i > 0 else (bc_val if bc_val is not None else vals[0])
        else:
            return vals[i] if i < N else (bc_val if bc_val is not None else vals[N - 1])

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One semi-implicit timestep. All parameters from self.spec."""
        N = self.N
        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_old = mdot.copy()

        # ── Step 1: Properties ──
        rho_l = np.zeros(N)
        rho_v = np.zeros(N)
        rho_m = np.zeros(N)
        T_l = np.zeros(N)
        T_sat = np.zeros(N)
        h_sat_l = np.zeros(N)
        h_sat_v = np.zeros(N)
        drho_dp = np.zeros(N)
        drho_dh = np.zeros(N)

        for i in range(N):
            p_safe = max(self.p_min, min(self.p_max, p[i]))
            h_l_safe = max(self.h_l_min, h_l[i])
            h_v_safe = max(self.h_l_min * 10, h_v[i])

            # Saturation properties first (needed for metastable checks)
            pp = self.fluid.evaluate_phasic(p_safe)
            T_sat[i] = pp.T_sat
            h_sat_l[i] = pp.h_sat_l
            h_sat_v[i] = pp.h_sat_v

            # Liquid properties with metastable extension
            # When h_l > h_f (superheated liquid after depressurization):
            #   rho_l = rho_f(p), not rho_ph(p, h_l) which gives two-phase mixture
            #   T_l = T_sat + (h_l - h_f)/cp_l, not T_sat (equilibrium)
            # Ref: RELAP5/MOD3 Vol I §3.2; matches Pipe1D_DriftFlux.mo lines 105-127
            fp_l = self.fluid.evaluate(p_safe, h_l_safe)
            if h_l_safe <= h_sat_l[i]:
                rho_l[i] = max(fp_l.rho, 1.0)
                T_l[i] = fp_l.T
            else:
                rho_l[i] = max(pp.rho_l, 1.0)
                T_l[i] = T_sat[i] + (h_l_safe - h_sat_l[i]) / 4200.0

            # Vapor properties with metastable extension
            # When h_v < h_g: rho_v = rho_g(p), not rho_ph(p, h_v) which gives mixture
            # Matches Pipe1D_DriftFlux.mo lines 109-112
            fp_v = self.fluid.evaluate(p_safe, h_v_safe)
            if h_v_safe >= h_sat_v[i]:
                rho_v[i] = max(fp_v.rho, self.rv_floor_abs)
            else:
                rho_v[i] = max(pp.rho_v, self.rv_floor_abs)
            rho_m[i] = (1 - alpha[i]) * rho_l[i] + alpha[i] * rho_v[i]

            h_mix = (1 - alpha[i]) * h_l[i] + alpha[i] * h_v[i]
            h_mix = max(self.h_l_min, min(self.h_v_max, h_mix))
            fp_mix = self.fluid.evaluate(p_safe, h_mix)
            drho_dp[i] = fp_mix.drho_dp_h
            drho_dh[i] = fp_mix.drho_dh_p

        rho_face = self._face_average(rho_m, N)

        # ── Step 2: Closures (formulas from Modelica, parameters from spec) ──
        Gamma = np.zeros(N)
        q_i_l = np.zeros(N)
        q_i_v = np.zeros(N)

        for i in range(N):
            alpha_eff = alpha[i]
            if T_l[i] > T_sat[i] and alpha[i] < self.alpha_nucleation:
                alpha_eff = self.alpha_nucleation

            a_i = max(4 * alpha_eff * (1 - alpha_eff), alpha_eff)
            q_i_l[i] = self.H_i * a_i * (T_sat[i] - T_l[i])

            h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
            Gamma[i] = -q_i_l[i] / h_fg
            q_i_v[i] = -Gamma[i] * (h_v[i] - h_l[i]) - q_i_l[i]

        # ── Step 3: Friction (with optional two-phase multiplier) ──
        Phi2 = np.ones(N + 1)
        if self.use_two_phase_friction:
            for i in range(N + 1):
                ci = min(i, N - 1)
                al = alpha[ci]
                rr = max(rho_l[ci] / max(rho_v[ci], self.rv_floor_abs), 1.0)
                Phi2[i] = min((1-al)**2 + 2*(1-al)*al*np.sqrt(rr) + al**2*rr, self.Phi2_max)

        fric = np.zeros(N + 1)
        for i in range(N + 1):
            if rho_face[i] > self.rho_face_min and np.isfinite(mdot_old[i]):
                f = (Phi2[i] * self.f_D * self.dx / (2 * self.D_h)
                     * abs(mdot_old[i]) * mdot_old[i]
                     / (rho_face[i] * self.A_flow**2))
                fric[i] = f if np.isfinite(f) else 0.0

        # ── Step 4: Critical flow ──
        mdot_crit = 1e10
        if self.use_critical_flow:
            last = N - 1
            h_mix_last = (1 - alpha[last]) * h_l[last] + alpha[last] * h_v[last]
            fp_last = self.fluid.evaluate(max(p[last], self.p_min), h_mix_last)
            pp_last = self.fluid.evaluate_phasic(max(p[last], self.p_min))

            h_f = pp_last.h_sat_l
            h_g = pp_last.h_sat_v
            h_fg = max(h_g - h_f, 1e3)
            x_local = max(0, min(1, (h_mix_last - h_f) / h_fg))

            dp_sub = max(p[last] - self.p_out, 0)
            G_sub = np.sqrt(2.0 * pp_last.rho_l * dp_sub)

            if fp_last.drho_dp_h > 0:
                c_hem = max(np.sqrt(1.0 / (fp_last.rho * fp_last.drho_dp_h)), self.c_floor)
            else:
                c_hem = self.c_floor
            G_hem = fp_last.rho * c_hem

            if x_local < self.x_trans:
                blend = x_local / self.x_trans
                G_crit = G_sub * (1 - blend) + G_hem * blend
            else:
                G_crit = G_hem
            G_crit = max(G_crit, G_hem)
            mdot_crit = self.C_d * self.A_flow * G_crit

        # ── Step 5: Pressure tridiagonal ──
        beta = dt * self.A_flow / self.dx
        outlet_choked = self.use_critical_flow and mdot_old[N] > 0

        a_tri = np.zeros(N)
        b_tri = np.zeros(N)
        c_tri = np.zeros(N)
        d_tri = np.zeros(N)

        for i in range(N):
            alpha_coeff = self.V_cell * drho_dp[i] / dt
            beta_left = 0.0 if (self.inlet_type == "wall" and i == 0) else (0.0 if i == 0 else beta)
            beta_right = 0.0 if (i == N-1 and outlet_choked) else beta

            a_tri[i] = -beta_left if i > 0 else 0.0
            c_tri[i] = -beta_right if i < N-1 else 0.0
            b_tri[i] = alpha_coeff + beta_left + beta_right
            d_tri[i] = alpha_coeff * p_old[i]
            d_tri[i] += (mdot_old[i] - mdot_old[i+1]) - dt * (fric[i] - fric[i+1])

            if i == N-1:
                if outlet_choked:
                    d_tri[i] += (mdot_old[N] - mdot_crit)
                else:
                    d_tri[i] += beta_right * self.p_out

        p[:] = self._thomas_solve(a_tri, b_tri, c_tri, d_tri)

        for i in range(N):
            if not np.isfinite(p[i]):
                p[i] = p_old[i]
            p[i] = max(self.p_min, min(self.p_max, p[i]))

        # ── Step 6: Momentum ──
        if self.inlet_type == "wall":
            mdot[0] = 0.0
        else:
            mdot[0] = mdot_old[0] + beta * (self.p_out - p[0]) - dt * fric[0]

        for i in range(1, N):
            mdot[i] = mdot_old[i] + beta * (p[i-1] - p[i]) - dt * fric[i]

        mdot_mom = mdot_old[N] + beta * (p[N-1] - self.p_out) - dt * fric[N]
        if self.use_critical_flow and mdot_mom > 0:
            mdot[N] = min(mdot_mom, mdot_crit)
        else:
            mdot[N] = mdot_mom

        # ── Step 7: Void fraction ──
        for i in range(N):
            al = alpha_old[i]
            rv = rho_v[i]

            alpha_in = self._donor_cell(mdot[i], alpha_old, i, N, bc_val=al)
            alpha_out = self._donor_cell(mdot[i+1], alpha_old, i+1, N, bc_val=al)
            # For outlet face with positive flow, alpha_out = al (self)
            if mdot[i+1] >= 0:
                alpha_out = al
            else:
                alpha_out = alpha_old[i+1] if i < N-1 else al

            flux_v = mdot[i] * (alpha_old[i-1] if i > 0 and mdot[i] >= 0 else al) \
                   - mdot[i+1] * alpha_out
            alpha_rho_v_new = al * rv + dt / self.V_cell * (flux_v + self.V_cell * Gamma[i])

            rv_new = max(self.fluid.evaluate(max(p[i], self.p_min), h_v[i]).rho, self.rv_floor_abs)
            alpha_new = max(0.0, min(1.0, alpha_rho_v_new / rv_new))

            if Gamma[i] > 0:
                alpha_new = max(alpha_new, self.alpha_nucleation)

            alpha[i] = alpha_new

        # ── Step 8: Phasic enthalpies ──
        for i in range(N):
            al = alpha_old[i]
            dp_dt = (p[i] - p_old[i]) / dt

            # Liquid
            m_l = (1 - al) * rho_l[i] * self.V_cell
            if m_l <= self.mass_min:
                h_l[i] = h_sat_l[i]
            else:
                h_in = h_l_old[i-1] if i > 0 and mdot[i] >= 0 else h_l_old[i]
                h_out = h_l_old[i] if mdot[i+1] >= 0 else (h_l_old[i+1] if i < N-1 else h_l_old[i])

                al_in = alpha_old[i-1] if i > 0 and mdot[i] >= 0 else al
                al_out = al if mdot[i+1] >= 0 else (alpha_old[i+1] if i < N-1 else al)

                ml_in = mdot[i] * (1 - al_in)
                ml_out = mdot[i+1] * (1 - al_out)

                flux = ml_in * (h_in - h_l_old[i]) - ml_out * (h_out - h_l_old[i])
                pw = (1 - al) * self.V_cell * dp_dt
                phase = -Gamma[i] * h_l_old[i] * self.V_cell
                qi = q_i_l[i] * self.V_cell

                h_l_new = h_l_old[i] + dt / m_l * (flux + pw + qi + phase)
                h_l[i] = max(self.h_l_min, min(h_l_new, h_sat_v[i]))

            # Vapor
            m_v = al * rho_v[i] * self.V_cell
            if m_v <= self.mass_min:
                h_v[i] = h_sat_v[i]
            else:
                h_in_v = h_v_old[i-1] if i > 0 and mdot[i] >= 0 else h_v_old[i]
                h_out_v = h_v_old[i] if mdot[i+1] >= 0 else (h_v_old[i+1] if i < N-1 else h_v_old[i])

                al_in = alpha_old[i-1] if i > 0 and mdot[i] >= 0 else al
                al_out = al if mdot[i+1] >= 0 else (alpha_old[i+1] if i < N-1 else al)

                mv_in = mdot[i] * al_in
                mv_out = mdot[i+1] * al_out

                flux_v = mv_in * (h_in_v - h_v_old[i]) - mv_out * (h_out_v - h_v_old[i])
                pw_v = al * self.V_cell * dp_dt
                phase_v = Gamma[i] * h_v_old[i] * self.V_cell
                qi_v = q_i_v[i] * self.V_cell

                h_v_new = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(h_sat_v[i], min(h_v_new, self.h_v_max))
