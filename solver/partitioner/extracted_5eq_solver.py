"""
extracted_5eq_solver.py — 5-equation drift-flux semi-implicit solver
driven by extracted equation structure from Modelica.

State variables: p[N], alpha[N], h_l[N], h_v[N], mdot[N+1]

Semi-implicit splitting:
  1. Evaluate properties (mixture + phasic) at old state
  2. Evaluate closures (Gamma, q_i_l, q_i_v, V_gj)
  3. Assemble pressure tridiagonal (mass conservation)
  4. Solve pressure (Thomas algorithm)
  5. Update momentum (inertial)
  6. Update void fraction (explicit, vapor mass transport)
  7. Update liquid enthalpy (explicit, phasic energy)
  8. Update vapor enthalpy (explicit, phasic energy)

All physics from Modelica Pipe1D_DriftFlux.mo. Property evaluation via
C++ FluidPackage (same math as Modelica Water.mo / SimpleFluid.mo).
"""

import numpy as np


class Extracted5EqSolver:

    def __init__(self, fluid, spec, H_i=1e5, C_0=1.0, alpha_nucleation=1e-3,
                 use_critical_flow=False, C_d=1.0, x_trans=0.10, c_floor=1200.0,
                 use_two_phase_friction=False):
        self.fluid = fluid
        self.N = spec.N
        self.dx = spec.dx
        self.A_flow = spec.A_flow
        self.D_h = spec.D_h
        self.f_D = spec.f_D
        self.V_cell = spec.V_cell
        self.inlet_closed = spec.inlet_closed
        self.p_out = spec.p_out or 101325.0

        # Closure parameters (from Modelica model parameters)
        self.H_i = H_i
        self.C_0 = C_0
        self.alpha_nucleation = alpha_nucleation

        # Critical flow
        self.use_critical_flow = use_critical_flow
        self.C_d = C_d
        self.x_trans = x_trans
        self.c_floor = c_floor

        # Two-phase friction
        self.use_two_phase_friction = use_two_phase_friction

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One semi-implicit timestep for the 5-equation drift-flux model."""
        N = self.N
        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_old = mdot.copy()

        # ──────────────────────────────────────────────────
        # Step 1: Evaluate properties at old state
        # ──────────────────────────────────────────────────
        rho_l = np.zeros(N)
        rho_v = np.zeros(N)
        rho_m = np.zeros(N)
        T_l = np.zeros(N)
        T_sat = np.zeros(N)
        h_sat_l = np.zeros(N)
        h_sat_v = np.zeros(N)
        drho_dp = np.zeros(N)
        drho_dh = np.zeros(N)
        sigma_arr = np.zeros(N)

        for i in range(N):
            # Clamp state variables to valid range
            p_safe = max(p[i], self.fluid.p_min if hasattr(self.fluid, 'p_min') else 700.0)
            p_safe = min(p_safe, self.fluid.p_max if hasattr(self.fluid, 'p_max') else 21e6)
            h_l_safe = max(h_l[i], 1e4)
            h_v_safe = max(h_v[i], 1e5)

            # Saturation properties first (needed for metastable checks)
            pp = self.fluid.evaluate_phasic(p_safe)
            T_sat[i] = pp.T_sat
            h_sat_l[i] = pp.h_sat_l
            h_sat_v[i] = pp.h_sat_v
            sigma_arr[i] = pp.sigma

            # Liquid properties with metastable extension
            # When h_l > h_f: rho_l = rho_f(p), T_l = T_sat + (h_l - h_f)/cp_l
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
                rho_v[i] = max(fp_v.rho, 0.01)
            else:
                rho_v[i] = max(pp.rho_v, 0.01)
            rho_m[i] = (1 - alpha[i]) * rho_l[i] + alpha[i] * rho_v[i]

            # Mixture properties for pressure linearization
            h_mix = (1 - alpha[i]) * h_l[i] + alpha[i] * h_v[i]
            h_mix = max(1e4, min(h_mix, 4e6))
            fp_mix = self.fluid.evaluate(p_safe, h_mix)
            drho_dp[i] = fp_mix.drho_dp_h
            drho_dh[i] = fp_mix.drho_dh_p

        # Face densities
        rho_face = np.zeros(N + 1)
        rho_face[0] = rho_m[0]
        for i in range(1, N):
            rho_face[i] = 0.5 * (rho_m[i - 1] + rho_m[i])
        rho_face[N] = rho_m[N - 1]

        # ──────────────────────────────────────────────────
        # Step 2: Evaluate closures
        # ──────────────────────────────────────────────────
        Gamma = np.zeros(N)
        q_i_l = np.zeros(N)
        q_i_v = np.zeros(N)

        for i in range(N):
            al = alpha[i]
            # Nucleation
            alpha_eff = al
            if T_l[i] > T_sat[i] and al < self.alpha_nucleation:
                alpha_eff = self.alpha_nucleation

            # Interfacial area
            a_i = max(4 * alpha_eff * (1 - alpha_eff), alpha_eff)

            # Interfacial heat transfer
            q_i_l[i] = self.H_i * a_i * (T_sat[i] - T_l[i])

            # Mass transfer
            h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
            Gamma[i] = -q_i_l[i] / h_fg

            # Vapor heat: energy balance
            q_i_v[i] = -Gamma[i] * (h_v[i] - h_l[i]) - q_i_l[i]

        # ──────────────────────────────────────────────────
        # Step 3: Two-phase friction multiplier
        # ──────────────────────────────────────────────────
        Phi2 = np.ones(N + 1)
        if self.use_two_phase_friction:
            for i in range(N + 1):
                ci = min(i, N - 1)
                al = alpha[ci]
                rl, rv = rho_l[ci], max(rho_v[ci], 0.1)
                rr = max(rl / rv, 1.0)
                Phi2[i] = min((1 - al)**2 + 2 * (1 - al) * al * np.sqrt(rr) + al**2 * rr, 20.0)

        # Friction (with overflow protection)
        fric = np.zeros(N + 1)
        for i in range(N + 1):
            if rho_face[i] > 0.01 and np.isfinite(mdot_old[i]):
                fric[i] = (Phi2[i] * self.f_D * self.dx / (2 * self.D_h)
                           * abs(mdot_old[i]) * mdot_old[i]
                           / (rho_face[i] * self.A_flow**2))
                if not np.isfinite(fric[i]):
                    fric[i] = 0.0

        # ──────────────────────────────────────────────────
        # Step 4: Critical flow at outlet
        # ──────────────────────────────────────────────────
        mdot_crit = 1e10
        if self.use_critical_flow:
            last = N - 1
            h_mix_last = (1 - alpha[last]) * h_l[last] + alpha[last] * h_v[last]
            fp_last = self.fluid.evaluate(p[last], h_mix_last)
            pp_last = self.fluid.evaluate_phasic(max(p[last], self.fluid.p_min))

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

        # ──────────────────────────────────────────────────
        # Step 5: Assemble pressure tridiagonal
        # ──────────────────────────────────────────────────
        beta = dt * self.A_flow / self.dx
        outlet_choked = self.use_critical_flow and mdot_old[N] > 0

        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N)

        for i in range(N):
            alpha_coeff = self.V_cell * drho_dp[i] / dt
            beta_left = 0.0 if (self.inlet_closed and i == 0) else (0.0 if i == 0 else beta)

            if i == N - 1 and outlet_choked:
                beta_right = 0.0
            else:
                beta_right = beta

            a[i] = -beta_left if i > 0 else 0.0
            c[i] = -beta_right if i < N - 1 else 0.0
            b[i] = alpha_coeff + beta_left + beta_right
            d[i] = alpha_coeff * p_old[i]
            d[i] += (mdot_old[i] - mdot_old[i + 1]) - dt * (fric[i] - fric[i + 1])

            if i == N - 1:
                if outlet_choked:
                    d[i] += (mdot_old[N] - mdot_crit)
                else:
                    d[i] += beta_right * self.p_out

        # Thomas solve
        c_p = np.zeros(N)
        d_p = np.zeros(N)
        c_p[0] = c[0] / b[0]
        d_p[0] = d[0] / b[0]
        for i in range(1, N):
            denom = b[i] - a[i] * c_p[i - 1]
            c_p[i] = c[i] / denom
            d_p[i] = (d[i] - a[i] * d_p[i - 1]) / denom
        p[N - 1] = d_p[N - 1]
        for i in range(N - 2, -1, -1):
            p[i] = d_p[i] - c_p[i] * p[i + 1]

        # Pressure bounds + NaN protection
        p_floor = self.fluid.p_min if hasattr(self.fluid, 'p_min') else 700.0
        p_ceil = self.fluid.p_max if hasattr(self.fluid, 'p_max') else 21e6
        for i in range(N):
            if not np.isfinite(p[i]):
                p[i] = p_old[i]
            p[i] = max(p_floor, min(p_ceil, p[i]))

        # ──────────────────────────────────────────────────
        # Step 6: Update momentum
        # ──────────────────────────────────────────────────
        mdot[0] = 0.0  # wall

        for i in range(1, N):
            mdot[i] = mdot_old[i] + beta * (p[i - 1] - p[i]) - dt * fric[i]

        # Outlet
        mdot_mom = mdot_old[N] + beta * (p[N - 1] - self.p_out) - dt * fric[N]
        if self.use_critical_flow and mdot_mom > 0:
            mdot[N] = min(mdot_mom, mdot_crit)
        else:
            mdot[N] = mdot_mom

        # ──────────────────────────────────────────────────
        # Step 7: Update void fraction (vapor mass transport)
        # ──────────────────────────────────────────────────
        for i in range(N):
            al = alpha_old[i]
            rv = rho_v[i]

            # Donor-cell vapor flux
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

            # New vapor density at new pressure
            rv_new = max(self.fluid.evaluate(p[i], h_v[i]).rho, 0.01)
            alpha_new = alpha_rho_v_new / rv_new
            alpha_new = max(0.0, min(1.0, alpha_new))

            # Nucleation floor
            if Gamma[i] > 0:
                alpha_new = max(alpha_new, 1e-3)

            alpha[i] = alpha_new

        # ──────────────────────────────────────────────────
        # Step 8: Update phasic enthalpies
        # ──────────────────────────────────────────────────
        for i in range(N):
            al = alpha_old[i]
            rl = rho_l[i]
            rv = rho_v[i]
            dp_dt = (p[i] - p_old[i]) / dt

            # --- Liquid enthalpy ---
            m_l = (1 - al) * rl * self.V_cell
            if m_l <= 1e-12:
                h_l[i] = h_sat_l[i]
            else:
                # Donor-cell for liquid
                if mdot[i] >= 0:
                    h_face_in = h_l_old[i - 1] if i > 0 else h_l_old[0]
                else:
                    h_face_in = h_l_old[i]

                if mdot[i + 1] >= 0:
                    h_face_out = h_l_old[i]
                else:
                    h_face_out = h_l_old[i + 1] if i < N - 1 else h_l_old[i]

                # Liquid mass fluxes (approximate: (1-alpha)*mdot)
                if mdot[i] >= 0:
                    al_in = alpha_old[i - 1] if i > 0 else al
                else:
                    al_in = al
                mdot_l_in = mdot[i] * (1 - al_in)

                if mdot[i + 1] >= 0:
                    al_out = al
                else:
                    al_out = alpha_old[i + 1] if i < N - 1 else al
                mdot_l_out = mdot[i + 1] * (1 - al_out)

                flux_l = mdot_l_in * (h_face_in - h_l_old[i]) - mdot_l_out * (h_face_out - h_l_old[i])
                p_work_l = (1 - al) * self.V_cell * dp_dt
                phase_l = -Gamma[i] * h_l_old[i] * self.V_cell
                qi_l = q_i_l[i] * self.V_cell

                h_l_new = h_l_old[i] + dt / m_l * (flux_l + p_work_l + qi_l + phase_l)
                h_l[i] = max(1e4, min(h_l_new, h_sat_v[i]))

            # --- Vapor enthalpy ---
            m_v = al * rv * self.V_cell
            if m_v <= 1e-12:
                h_v[i] = h_sat_v[i]
            else:
                if mdot[i] >= 0:
                    h_face_in_v = h_v_old[i - 1] if i > 0 else h_v_old[0]
                else:
                    h_face_in_v = h_v_old[i]

                if mdot[i + 1] >= 0:
                    h_face_out_v = h_v_old[i]
                else:
                    h_face_out_v = h_v_old[i + 1] if i < N - 1 else h_v_old[i]

                if mdot[i] >= 0:
                    al_in = alpha_old[i - 1] if i > 0 else al
                else:
                    al_in = al
                mdot_v_in = mdot[i] * al_in

                if mdot[i + 1] >= 0:
                    al_out = al
                else:
                    al_out = alpha_old[i + 1] if i < N - 1 else al
                mdot_v_out = mdot[i + 1] * al_out

                flux_v = mdot_v_in * (h_face_in_v - h_v_old[i]) - mdot_v_out * (h_face_out_v - h_v_old[i])
                p_work_v = al * self.V_cell * dp_dt
                phase_v = Gamma[i] * h_v_old[i] * self.V_cell
                qi_v = q_i_v[i] * self.V_cell

                h_v_new = h_v_old[i] + dt / m_v * (flux_v + p_work_v + qi_v + phase_v)
                h_v[i] = max(h_sat_v[i], min(h_v_new, 4e6))
