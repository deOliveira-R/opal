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

    def __init__(self, bridge: OMEquationBridge, spec):
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

        # Closure flags (from spec if available)
        self.use_critical_flow = getattr(spec, 'use_critical_flow', False)

        # Set bridge parameters from spec
        bridge.set_params_from_spec(spec)

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
        mdot_old = mdot.copy()

        # ══════════════════════════════════════════════════════════
        # PHYSICS EVALUATION — ALL from OM-generated C
        # ══════════════════════════════════════════════════════════
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

        # Phi2 may have N or N+1 entries depending on model
        Phi2 = self.bridge.get('Phi2') if self.bridge.has('Phi2') else np.ones(N + 1)

        # ══════════════════════════════════════════════════════════
        # NUMERICAL METHOD ONLY BELOW
        # ══════════════════════════════════════════════════════════

        # ── Friction (Darcy * Phi2, from geometry + old momentum + face density) ──
        fric = np.zeros(N + 1)
        for i in range(N + 1):
            phi2_i = Phi2[min(i, len(Phi2) - 1)]
            if rho_face[i] > 0.01:
                fric[i] = (phi2_i * self.f_D * self.dx / (2 * self.D_h)
                           * abs(mdot_old[i]) * mdot_old[i]
                           / (rho_face[i] * self.A_flow**2))

        # ── Pressure tridiagonal ──
        beta = dt * self.A_flow / self.dx
        a_tri = np.zeros(N); b_tri = np.zeros(N)
        c_tri = np.zeros(N); d_tri = np.zeros(N)

        for i in range(N):
            alpha_coeff = self.V_cell * drho_dp[i] / dt
            beta_left = 0.0 if (self.inlet_closed and i == 0) else (0.0 if i == 0 else beta)
            beta_right = beta

            a_tri[i] = -beta_left if i > 0 else 0.0
            c_tri[i] = -beta_right if i < N - 1 else 0.0
            b_tri[i] = alpha_coeff + beta_left + beta_right
            d_tri[i] = alpha_coeff * p_old[i]
            d_tri[i] += (mdot_old[i] - mdot_old[i + 1]) - dt * (fric[i] - fric[i + 1])

            if i == N - 1:
                d_tri[i] += beta_right * self.p_out

        p[:] = self._thomas_solve(a_tri, b_tri, c_tri, d_tri)
        for i in range(N):
            p[i] = max(self.bridge.p_min, min(self.bridge.p_max, p[i]))

        # ── Momentum ──
        mdot[0] = 0.0  # wall BC
        for i in range(1, N):
            mdot[i] = mdot_old[i] + beta * (p[i - 1] - p[i]) - dt * fric[i]
        mdot[N] = mdot_old[N] + beta * (p[N - 1] - self.p_out) - dt * fric[N]

        # ── Void fraction transport (explicit) ──
        for i in range(N):
            al = alpha_old[i]
            rv = max(rho_v[i], 0.01)

            # Donor-cell void fraction advection
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

            rv_new = max(rho_v[i], 0.01)  # Use old-state rho_v for explicit update
            alpha[i] = max(0.0, min(1.0, alpha_rho_v_new / rv_new))

        # ── Phasic energy (explicit) ──
        dp_dt = (p - p_old) / dt

        for i in range(N):
            al = alpha_old[i]

            # Liquid energy
            m_l = (1 - al) * rho_l[i] * self.V_cell
            if m_l > 1e-12:
                # Donor-cell liquid enthalpy
                if mdot[i] >= 0:
                    h_in = h_l[i - 1] if i > 0 else h_l[i]
                else:
                    h_in = h_l[i]
                if mdot[i + 1] >= 0:
                    h_out = h_l[i]
                else:
                    h_out = h_l[i + 1] if i < N - 1 else h_l[i]

                al_in = alpha_old[i - 1] if i > 0 and mdot[i] >= 0 else al
                al_out = al if mdot[i + 1] >= 0 else (alpha_old[i + 1] if i < N - 1 else al)

                ml_in = mdot[i] * (1 - al_in)
                ml_out = mdot[i + 1] * (1 - al_out)

                flux = ml_in * (h_in - h_l[i]) - ml_out * (h_out - h_l[i])
                pw = (1 - al) * self.V_cell * dp_dt[i]
                qi = q_i_l[i] * self.V_cell
                phase = -Gamma[i] * h_l[i] * self.V_cell

                h_l[i] = h_l[i] + dt / m_l * (flux + pw + qi + phase)
                h_l[i] = max(1e4, min(h_l[i], h_sat_v[i]))
            else:
                h_l[i] = h_sat_l[i]

            # Vapour energy
            m_v = al * rho_v[i] * self.V_cell
            if m_v > 1e-12:
                if mdot[i] >= 0:
                    hv_in = h_v[i - 1] if i > 0 else h_v[i]
                else:
                    hv_in = h_v[i]
                if mdot[i + 1] >= 0:
                    hv_out = h_v[i]
                else:
                    hv_out = h_v[i + 1] if i < N - 1 else h_v[i]

                al_in = alpha_old[i - 1] if i > 0 and mdot[i] >= 0 else al
                al_out = al if mdot[i + 1] >= 0 else (alpha_old[i + 1] if i < N - 1 else al)

                mv_in = mdot[i] * al_in
                mv_out = mdot[i + 1] * al_out

                flux_v = mv_in * (hv_in - h_v[i]) - mv_out * (hv_out - h_v[i])
                pw_v = al * self.V_cell * dp_dt[i]
                qi_v = q_i_v[i] * self.V_cell
                phase_v = Gamma[i] * h_v[i] * self.V_cell

                h_v[i] = h_v[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(1e4, min(h_v[i], 4e6))
            else:
                h_v[i] = h_sat_v[i]
