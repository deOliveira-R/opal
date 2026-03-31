"""
bridge_5eq_solver_v11_energy_corrector.py — V11 with energy re-evaluation.

Targets the specific operator splitting weakness: V11's energy equations use
OLD enthalpies for advection. This corrector re-evaluates the bridge at the
V11 solution state and re-computes energy with UPDATED enthalpies.

The pressure/void/momentum stay at V11's values (already well-coupled
via the block Thomas solve). Only the energy equations are corrected.

Cost: 1-2 extra bridge evaluations per step (compared to V11's single eval).
Much cheaper than full JFNK (~100 evaluations per step).
"""

import numpy as np

from .codegen.equation_bridge import OMEquationBridge
from .bridge_5eq_solver_v11_a12mod import BridgeDriftFluxSolver


class V11EnergyCorrector:
    """V11 solver with iterated energy correction.

    Each timestep:
    1. Run V11.step() → new p, alpha, mdot, h_l_v11, h_v_v11
    2. Re-evaluate bridge at the V11 solution state
    3. Re-compute energy equations using CURRENT enthalpies for advection
       (not frozen old enthalpies as in V11)
    4. Repeat 2-3 for n_energy_iters iterations

    This captures the enthalpy-pressure coupling that V11 misses due to
    operator splitting, at a fraction of JFNK's cost.
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 tau_mix=4.5e-4, use_isentropic_a11=True,
                 n_energy_iters=2):
        self.v11 = BridgeDriftFluxSolver(
            bridge, spec, es=es,
            tau_mix=tau_mix, use_isentropic_a11=use_isentropic_a11)
        self.bridge = bridge
        self.N = bridge.N
        self.spec = spec
        self.n_energy_iters = n_energy_iters

        # Geometry
        self.V_cell = spec.V_cell
        self.time = 0.0

        # Diagnostics
        self.energy_corrections = []

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """V11 step followed by energy correction iterations.

        Modifies arrays in-place.
        """
        N = self.N

        # Save REAL old state (for time derivatives in energy equation)
        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_old = mdot.copy()

        # ---- V11 step (standard, all-in-one) ----
        self.v11.time = self.time
        self.v11.step(p, alpha, h_l, h_v, mdot, dt)

        # p, alpha, mdot are now at V11's new values (FINAL — don't change)
        # h_l, h_v are V11's energy output (will be corrected below)
        p_new = p     # reference (not copy — p is already updated)
        alpha_new = alpha
        mdot_new = mdot

        dp_dt = (p_new - p_old) / dt

        # ---- Energy correction iterations ----
        max_dh = 0.0
        for it in range(self.n_energy_iters):
            # Save current enthalpies (Jacobi-style iteration)
            h_l_curr = h_l.copy()
            h_v_curr = h_v.copy()

            # Re-evaluate bridge at CURRENT full state
            self.bridge.set_time(self.time + dt)
            self.bridge.set_state(p_new, alpha=alpha_new,
                                  h_l=h_l_curr, h_v=h_v_curr, mdot=mdot_new)
            self.bridge.evaluate()

            rho_l = self.bridge.get('rho_l')
            rho_v = self.bridge.get('rho_v')
            Gamma = self.bridge.get('Gamma')
            q_i_l = self.bridge.get('q_i_l')
            q_i_v = self.bridge.get('q_i_v')
            h_sat_l = self.bridge.get('h_sat_l')
            h_sat_v = self.bridge.get('h_sat_v')

            has_df = self.bridge.has('mdot_v') and self.bridge.has('mdot_l')
            if has_df:
                mdot_l_face = self.bridge.get('mdot_l')
                mdot_v_face = self.bridge.get('mdot_v')

            # Re-compute energy with CURRENT enthalpies for advection
            for i in range(N):
                al = alpha_old[i]

                # ---- Liquid energy ----
                m_l = max((1 - al) * rho_l[i] * self.V_cell, 1e-12)
                if (1 - al) > 1e-6:
                    if has_df:
                        ml_in = mdot_l_face[i]
                        ml_out = mdot_l_face[i + 1]
                    else:
                        al_in = (alpha_old[i - 1]
                                 if i > 0 and mdot_new[i] >= 0 else al)
                        al_out = (al if mdot_new[i + 1] >= 0
                                  else (alpha_old[i + 1]
                                        if i < N - 1 else al))
                        ml_in = mdot_new[i] * (1 - al_in)
                        ml_out = mdot_new[i + 1] * (1 - al_out)

                    # Face enthalpies: CURRENT iterate (not h_l_old!)
                    flow_in = ml_in if has_df else mdot_new[i]
                    flow_out = ml_out if has_df else mdot_new[i + 1]
                    h_in = (h_l_curr[i - 1]
                            if (i > 0 and flow_in >= 0) else h_l_curr[i])
                    h_out = (h_l_curr[i] if flow_out >= 0
                             else (h_l_curr[i + 1]
                                   if i < N - 1 else h_l_curr[i]))

                    flux = (ml_in * (h_in - h_l_curr[i])
                            - ml_out * (h_out - h_l_curr[i]))
                    pw = (1 - al) * self.V_cell * dp_dt[i]
                    qi = q_i_l[i] * self.V_cell
                    phase = -Gamma[i] * h_l_curr[i] * self.V_cell

                    h_l[i] = h_l_old[i] + dt / m_l * (
                        flux + pw + qi + phase)
                    h_l[i] = max(1e4, min(h_l[i], h_sat_v[i]))
                else:
                    h_l[i] = h_sat_l[i]

                # ---- Vapour energy ----
                m_v = max(al * rho_v[i] * self.V_cell, 1e-12)
                if al > 1e-6:
                    if has_df:
                        mv_in = mdot_v_face[i]
                        mv_out = mdot_v_face[i + 1]
                    else:
                        al_in = (alpha_old[i - 1]
                                 if i > 0 and mdot_new[i] >= 0 else al)
                        al_out = (al if mdot_new[i + 1] >= 0
                                  else (alpha_old[i + 1]
                                        if i < N - 1 else al))
                        mv_in = mdot_new[i] * al_in
                        mv_out = mdot_new[i + 1] * al_out

                    flow_in_v = mv_in if has_df else mdot_new[i]
                    flow_out_v = mv_out if has_df else mdot_new[i + 1]
                    hv_in = (h_v_curr[i - 1]
                             if (i > 0 and flow_in_v >= 0) else h_v_curr[i])
                    hv_out = (h_v_curr[i] if flow_out_v >= 0
                              else (h_v_curr[i + 1]
                                    if i < N - 1 else h_v_curr[i]))

                    flux_v = (mv_in * (hv_in - h_v_curr[i])
                              - mv_out * (hv_out - h_v_curr[i]))
                    pw_v = al * self.V_cell * dp_dt[i]
                    qi_v = q_i_v[i] * self.V_cell
                    phase_v = Gamma[i] * h_v_curr[i] * self.V_cell

                    h_v[i] = h_v_old[i] + dt / m_v * (
                        flux_v + pw_v + qi_v + phase_v)
                    h_v[i] = max(h_sat_v[i], min(h_v[i], 4e6))
                else:
                    h_v[i] = h_sat_v[i]

                # Nucleation floor (same as V11)
                if Gamma[i] > 0:
                    alpha[i] = max(alpha[i], 1e-3)

            # Track correction magnitude
            max_dh = max(np.max(np.abs(h_l - h_l_curr)),
                         np.max(np.abs(h_v - h_v_curr)))

        self.energy_corrections.append(max_dh)
