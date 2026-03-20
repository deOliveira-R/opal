"""
bridge_solver.py — Semi-implicit solver using the OM equation bridge.

True Case 2: ALL physics evaluation comes from OM-generated C code
called through the equation bridge. The solver provides ONLY the
semi-implicit numerical method (operator splitting, Thomas algorithm,
donor-cell reconstruction).

What comes from the bridge (Modelica physics, evaluated at old state):
  - Thermodynamic properties: rho, drho_dp, drho_dh, T (from PartialMedium)
  - Face density averaging (from PartialPipe1D equations)
  - Cell density (from PartialMedium.rho_ph)

What stays in the solver (numerical methods):
  - Friction computation (from extracted geometry + old momentum)
  - Pressure tridiagonal assembly + Thomas solve
  - Semi-implicit momentum update (NEW pressure, OLD friction)
  - Donor-cell upwind selection (using NEW mdot for transport consistency)
  - Explicit energy update (forward Euler)
  - Pressure clamping

Note on donor-cell: The upwind direction selection `if mdot >= 0 then h_upwind`
is a numerical discretization choice (like the Thomas algorithm or operator
splitting), not a physical closure. It appears in Modelica because Modelica
needs to express the complete equation system, but it is in the same category
as Numerics/Limiters.mo — a scheme choice, not a physical law. The semi-implicit
method requires evaluating donor-cell with the UPDATED mdot (post-momentum)
for consistency with the implicit pressure solve. The bridge evaluates h_face
at the old state; the solver applies the correct time level.
"""

import numpy as np

from .codegen.equation_bridge import OMEquationBridge


class BridgeSolver:
    """Semi-implicit HEM solver driven by the OM equation bridge.

    Physics (properties, face densities) from OM-generated C.
    Numerics (pressure solve, momentum, transport) from Python.
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None):
        """
        Args:
            bridge: OMEquationBridge (compiled from translateModel output)
            spec: Pipe1DGridSpec with geometry and BCs
            es: EquationSystem (optional) — if provided, ALL parameter values
                are read from the extracted XML (authoritative source).
        """
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
        self.inlet_closed = spec.inlet_closed
        self.p_out = spec.p_out or 101325.0

        # Set bridge parameters (from XML if available, else from spec)
        bridge.set_params_from_spec(spec, es=es)

        # Exposed internals for L0 testing
        self.last_rho_face = None
        self.last_drho_dp = None
        self.last_fric = None

    def step(self, p, h, mdot, dt):
        """One semi-implicit timestep. Modifies p, h, mdot in-place."""
        N = self.N
        p_old = p.copy()
        mdot_old = mdot.copy()

        # ══════════════════════════════════════════════════════════
        # PHYSICS EVALUATION — from OM-generated C (old state)
        # Properties, face densities, cell density — all from Modelica
        # ══════════════════════════════════════════════════════════
        self.bridge.set_state(p, h, mdot)
        self.bridge.evaluate()

        drho_dp = self.bridge.get_drho_dp()
        rho_face = self.bridge.get_rho_face()
        rho_cell = self.bridge.get_rho_cell()

        self.last_rho_face = rho_face.copy()
        self.last_drho_dp = drho_dp.copy()

        # ══════════════════════════════════════════════════════════
        # NUMERICAL METHOD — semi-implicit operator splitting
        # ══════════════════════════════════════════════════════════

        # ── Friction (Darcy, from geometry + old momentum + face density) ──
        fric = np.zeros(N + 1)
        for i in range(N + 1):
            if rho_face[i] > 0.01:
                fric[i] = (self.f_D * self.dx / (2 * self.D_h)
                           * abs(mdot_old[i]) * mdot_old[i]
                           / (rho_face[i] * self.A_flow**2))
        self.last_fric = fric.copy()

        # ── Pressure tridiagonal assembly ──
        beta = dt * self.A_flow / self.dx

        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)
        d = np.zeros(N)

        for i in range(N):
            alpha_i = self.V_cell * drho_dp[i] / dt

            beta_left = 0.0 if (self.inlet_closed and i == 0) else (0.0 if i == 0 else beta)
            beta_right = beta

            a[i] = -beta_left if i > 0 else 0.0
            c[i] = -beta_right if i < N - 1 else 0.0
            b[i] = alpha_i + beta_left + beta_right
            d[i] = alpha_i * p_old[i]
            d[i] += (mdot_old[i] - mdot_old[i + 1]) - dt * (fric[i] - fric[i + 1])

            if i == N - 1:
                d[i] += beta_right * self.p_out

        # ── Thomas algorithm ──
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

        # Pressure bounds
        for i in range(N):
            p[i] = max(self.bridge.p_min, min(self.bridge.p_max, p[i]))

        # ── Momentum update (semi-implicit: NEW pressure, OLD friction) ──
        mdot[0] = 0.0  # wall BC

        for i in range(1, N):
            mdot[i] = mdot_old[i] + beta * (p[i - 1] - p[i]) - dt * fric[i]

        mdot_mom = mdot_old[N] + beta * (p[N - 1] - self.p_out) - dt * fric[N]

        # Critical flow limiter at outlet (if model has it)
        mdot_crit_name = f'{self.bridge.prefix}.mdot_crit'
        if mdot_crit_name in self.bridge.info.all_vars and mdot_mom > 0:
            crit_idx = self.bridge.info.var_index(mdot_crit_name)
            mdot_crit_val = self.bridge.lib.opal_bridge_get_var(crit_idx)
            mdot[N] = min(mdot_mom, mdot_crit_val)
        else:
            mdot[N] = mdot_mom

        # ── Energy update (explicit, donor-cell with UPDATED mdot) ──
        # Donor-cell upwind selection uses the new mdot (post-momentum)
        # for consistency with the implicit pressure solve. This is the
        # standard approach in semi-implicit TH codes (RELAP5, TRACE).
        for i in range(N):
            # Donor-cell face enthalpies (upwind selection with new mdot)
            if mdot[i] >= 0:
                h_in = h[i - 1] if i > 0 else h[0]
            else:
                h_in = h[i]

            if mdot[i + 1] >= 0:
                h_out = h[i]
            else:
                h_out = h[i + 1] if i < N - 1 else h[i]

            flux = mdot[i] * (h_in - h[i]) - mdot[i + 1] * (h_out - h[i])
            p_work = self.V_cell * (p[i] - p_old[i]) / dt

            h[i] = h[i] + dt / (rho_cell[i] * self.V_cell) * (flux + p_work)
