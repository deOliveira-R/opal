"""
bridge_5eq_solver_v10_offdiag.py -- Variant 10: Block Thomas with drift-flux
linearized off-diagonals.

Based on V5 (block Thomas). Adds vapor flux coupling to the off-diagonal blocks
of the 2x2 block tridiagonal system. In V5, the off-diagonal (1,0) entries are
zero, meaning the vapor mass equation in cell i has no coupling to neighbor
pressures through the block solve. Physically, when a neighbor's pressure
changes, the mixture momentum at the shared face changes (via beta_eff), and
the drift-flux split allocates a fraction of that momentum change to the vapor
phase. This fraction is:

    d(mdot_v)/dp_neighbor = alpha_face * rho_v_face * C_0 * beta_eff / rho_face

where C_0 is the drift-flux distribution parameter. For simplicity and to avoid
hard-coding a Modelica parameter, C_0 = 1.0 is used here (the exact value is a
second-order effect on the off-diagonal coupling).

The linearized vapor flux contribution to Row 2 for cell i:
  - From dp[i-1]: +beta_v_L  (vapor enters from left when left pressure rises)
  - From dp[i]:   -(beta_v_L + beta_v_R)  (vapor leaves when this cell's pressure rises)
  - From dp[i+1]: +beta_v_R  (vapor enters from right when right pressure rises)

The block system per cell is:
    [A11  A12] [delta_p    ]   [R1]
    [A21  A22] [delta_alpha] = [R2]

Row 1: mixture mass conservation (pressure equation) -- unchanged from V5
Row 2: vapour mass conservation (void equation) -- now with phasic off-diagonals

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
    """5-equation drift-flux solver with 2x2 block Thomas and vapor off-diagonals.

    Extends V5 by adding drift-flux vapor flux coupling to the off-diagonal
    blocks of the 2x2 block tridiagonal system. This allows the block solve
    to propagate void information between neighboring cells through the
    pressure-vapor momentum coupling.

    Physics (properties, closures, friction multiplier, interfacial transfer)
    from OM-generated C. Numerics (block tridiagonal solve, momentum, phasic
    energy) from Python.
    """

    def __init__(self, bridge: OMEquationBridge, spec, es=None,
                 reconstruction='donor_cell', use_block_coupling=False,
                 corrector_steps=0):
        """
        Args:
            bridge: OMEquationBridge with compiled Modelica model
            spec: Pipe1DGridSpec (geometry, BCs)
            es: EquationSystem from XML -- pass this to load ALL parameter values
                from Modelica (H_i, C_0, etc.). Without es, only geometry is set.
            reconstruction: 'donor_cell' (1st order) or 'muscl' (2nd order, minmod TVD)
            use_block_coupling: accepted for API compatibility but ignored --
                block coupling is always active in V10.
            corrector_steps: accepted for API compatibility but ignored --
                block solve is implicit, no corrector needed.
        """
        self.reconstruction = reconstruction
        # V10: block coupling is always active; corrector is unnecessary
        self.use_block_coupling = True
        self.corrector_steps = 0
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
        """Scalar Thomas algorithm (kept for fallback)."""
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

    @staticmethod
    def _block_thomas_solve(a_blocks, b_blocks, c_blocks, d_vecs):
        """Block Thomas algorithm for 2x2 block tridiagonal system.

        Solves the system:
            a_blocks[i] * x[i-1] + b_blocks[i] * x[i] + c_blocks[i] * x[i+1] = d_vecs[i]

        where each block is a 2x2 matrix and each x[i], d_vecs[i] is a 2-vector.

        Args:
            a_blocks[i]: 2x2 lower diagonal block (a_blocks[0] unused)
            b_blocks[i]: 2x2 diagonal block
            c_blocks[i]: 2x2 upper diagonal block (c_blocks[N-1] unused)
            d_vecs[i]: 2-vector RHS

        Returns:
            list of 2-vectors x[i]
        """
        N = len(d_vecs)
        Z = np.zeros((2, 2))

        # Forward sweep
        cp = [None] * N  # modified upper diagonal blocks
        dp = [None] * N  # modified RHS vectors

        # i=0: no lower diagonal
        b_inv = np.linalg.inv(b_blocks[0])
        cp[0] = b_inv @ c_blocks[0]
        dp[0] = b_inv @ d_vecs[0]

        for i in range(1, N):
            temp = b_blocks[i] - a_blocks[i] @ cp[i - 1]
            temp_inv = np.linalg.inv(temp)
            if i < N - 1:
                cp[i] = temp_inv @ c_blocks[i]
            else:
                cp[i] = Z.copy()
            dp[i] = temp_inv @ (d_vecs[i] - a_blocks[i] @ dp[i - 1])

        # Backward substitution
        x = [None] * N
        x[N - 1] = dp[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] @ x[i + 1]

        return x

    def step(self, p, alpha, h_l, h_v, mdot, dt):
        """One semi-implicit timestep with 2x2 block Thomas solve.

        The block solve couples pressure and void fraction implicitly via
        mixture mass (Row 1) and vapour mass (Row 2) conservation. This
        eliminates the explicit void transport step and predictor-corrector.

        Modifies all arrays in-place.
        """
        N = self.N
        p_old = p.copy()
        alpha_old = alpha.copy()
        h_l_old = h_l.copy()
        h_v_old = h_v.copy()
        mdot_old = mdot.copy()

        # ====================================================================
        # PHYSICS EVALUATION -- ALL from OM-generated C
        # ====================================================================
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

        # Phasic compressibility -- required for block-coupled solve
        if self.bridge.has('drho_l_dp') and self.bridge.has('drho_v_dp'):
            drho_l_dp = self.bridge.get('drho_l_dp')
            drho_v_dp = self.bridge.get('drho_v_dp')
        else:
            # Fallback: approximate phasic compressibilities from mixture
            # This is less accurate but allows the solver to run without
            # the phasic derivatives from the bridge.
            drho_l_dp = drho_dp * 0.1   # liquid is ~10x less compressible
            drho_v_dp = drho_dp * 10.0   # vapor is ~10x more compressible

        # ====================================================================
        # NUMERICAL METHOD ONLY BELOW
        # ====================================================================

        # -- Critical flow (from Modelica via bridge) --
        mdot_crit = 1e10
        outlet_choked = False
        if self.use_critical_flow and self.bridge.has('mdot_crit'):
            mdot_crit_arr = self.bridge.get('mdot_crit')
            mdot_crit = mdot_crit_arr[0] if len(mdot_crit_arr) > 0 else 1e10
            mdot_crit *= getattr(self, 'C_d_factor', 1.0)
            outlet_choked = mdot_old[N] > 0

        # -- Friction with implicit resistance (semi-implicit friction treatment) --
        K_geom = self.f_D * self.dx / (2 * self.D_h)
        fric = np.zeros(N + 1)
        sigma = np.zeros(N + 1)

        for i in range(N + 1):
            phi2_i = Phi2[min(i, len(Phi2) - 1)]
            if rho_face[i] > 0.01 and np.isfinite(mdot_old[i]):
                f = (phi2_i * K_geom * abs(mdot_old[i]) * mdot_old[i]
                     / (rho_face[i] * self.A_flow**2))
                fric[i] = f if np.isfinite(f) else 0.0
                sigma[i] = (2 * dt * phi2_i * K_geom * abs(mdot_old[i])
                            / (rho_face[i] * self.A_flow**2))

        beta = dt * self.A_flow / self.dx
        beta_eff = beta / (1.0 + sigma)

        # -- Drift-flux phasic mass flows from bridge (if available) --
        has_drift_flux = self.bridge.has('mdot_v') and self.bridge.has('mdot_l')
        if has_drift_flux:
            mdot_v_face = self.bridge.get('mdot_v')
            mdot_l_face = self.bridge.get('mdot_l')
        else:
            mdot_v_face = None
            mdot_l_face = None

        # ====================================================================
        # 2x2 BLOCK TRIDIAGONAL ASSEMBLY (V10: with vapor off-diagonals)
        # ====================================================================
        # For each cell i, the coupled system is:
        #   [A11  A12] [delta_p    ]   [R1]
        #   [A21  A22] [delta_alpha] = [R2]
        #
        # Row 1 (mixture mass): V/dt * drho_mech * dp + V/dt * (rho_v - rho_l) * dalpha
        #   = (mdot_in - mdot_out) + pressure coupling from neighbors
        # Row 2 (vapour mass): V/dt * alpha * drho_v_dp * dp + V/dt * rho_v * dalpha
        #   = (mdot_v_in - mdot_v_out) + V * Gamma
        #
        # V10 addition: off-diagonal (1,0) entries carry the linearized vapor
        # flux response to neighbor pressure changes. The drift-flux split
        # allocates a fraction of the mixture momentum change to vapor:
        #   d(mdot_v)/dp_neighbor = alpha_face * rho_v_face * beta_eff / rho_face
        # (C_0 = 1.0 approximation; exact value is second-order here.)

        # -- Pre-compute face-level vapor properties for off-diagonal coupling --
        # Use donor-cell (upwind) values at each face. Face i sits between
        # cell i-1 and cell i; use the upwind cell's alpha and rho_v.
        alpha_face = np.zeros(N + 1)
        rho_v_face = np.zeros(N + 1)
        for fi in range(N + 1):
            if fi == 0:
                # Inlet face: use cell 0 values (closed wall or inlet BC)
                alpha_face[fi] = alpha_old[0]
                rho_v_face[fi] = max(rho_v[0], 0.01)
            elif fi == N:
                # Outlet face: use cell N-1 values
                alpha_face[fi] = alpha_old[N - 1]
                rho_v_face[fi] = max(rho_v[N - 1], 0.01)
            else:
                # Interior face: donor-cell based on old flow direction
                if mdot_old[fi] >= 0:
                    alpha_face[fi] = alpha_old[fi - 1]
                    rho_v_face[fi] = max(rho_v[fi - 1], 0.01)
                else:
                    alpha_face[fi] = alpha_old[fi]
                    rho_v_face[fi] = max(rho_v[fi], 0.01)

        a_blocks = [np.zeros((2, 2)) for _ in range(N)]
        b_blocks = [np.zeros((2, 2)) for _ in range(N)]
        c_blocks = [np.zeros((2, 2)) for _ in range(N)]
        d_vecs = [np.zeros(2) for _ in range(N)]

        for i in range(N):
            al = alpha_old[i]
            rv_i = max(rho_v[i], 0.01)
            rl_i = max(rho_l[i], 1.0)

            # Face coupling coefficients (same logic as scalar solver)
            bL = 0.0 if (self.inlet_closed and i == 0) else (
                0.0 if i == 0 else beta_eff[i])
            bR = 0.0 if (i == N - 1 and outlet_choked) else beta_eff[i + 1]

            # ---- Drift-flux vapor momentum coupling (V10) ----
            # Linearized sensitivity of vapor mass flux to neighbor pressure.
            # mdot_v = alpha_f * rho_v_f * C_0 * j * A, where j = mdot/(rho*A).
            # d(mdot_v)/dp_neighbor = alpha_f * rho_v_f * C_0 * d(j)/dp_neighbor
            # d(j)/dp_neighbor = beta_eff / (A * rho_face) [from momentum eq]
            # So: d(mdot_v)/dp_neighbor = alpha_f * rho_v_f * beta_eff / rho_face
            # (using C_0 = 1.0)
            #
            # Left face (face i): between cell i-1 and cell i
            if has_drift_flux and i > 0 and bL > 0:
                rho_f_L = max(rho_face[i], 1.0)
                beta_v_L = alpha_face[i] * rho_v_face[i] * bL / rho_f_L
            else:
                beta_v_L = 0.0

            # Right face (face i+1): between cell i and cell i+1
            if has_drift_flux and i < N - 1 and bR > 0:
                rho_f_R = max(rho_face[i + 1], 1.0)
                beta_v_R = alpha_face[i + 1] * rho_v_face[i + 1] * bR / rho_f_R
            else:
                beta_v_R = 0.0

            # ---- Row 1: Mixture mass conservation ----
            # Mechanical compressibility: how mixture density changes with p
            # at fixed void fraction
            drho_mech = (1 - al) * drho_l_dp[i] + al * drho_v_dp[i]
            A11 = self.V_cell / dt * drho_mech + bL + bR
            # Void changes mixture density: d(rho_m)/d(alpha) = rho_v - rho_l
            A12 = self.V_cell / dt * (rv_i - rl_i)

            # RHS Row 1: mass residual + implicit friction correction
            # The delta formulation requires the old pressure gradient terms:
            #   R1 = (mdot_old_in - mdot_old_out)
            #        - bL*(p_old[i] - p_old[i-1]) - bR*(p_old[i] - p_old[i+1])
            #        - friction_correction
            # because we're solving for delta_p, not absolute p.
            R1 = (mdot_old[i] - mdot_old[i + 1])
            R1 -= dt * (fric[i] / (1.0 + sigma[i])
                        - fric[i + 1] / (1.0 + sigma[i + 1]))

            # Old pressure gradient contributions from momentum coupling
            if not (self.inlet_closed and i == 0) and i > 0:
                R1 -= bL * (p_old[i] - p_old[i - 1])
            if i < N - 1:
                R1 -= bR * (p_old[i] - p_old[i + 1])

            # Outlet BC contribution to R1
            if i == N - 1:
                if outlet_choked:
                    R1 += (mdot_old[N] - mdot_crit)
                else:
                    # For unchoked outlet: the right face couples to p_out
                    # Old gradient term: -bR*(p_old[i] - p_out)
                    R1 -= bR * (p_old[i] - self.p_out)

            # ---- Row 2: Vapour mass conservation ----
            # Pressure changes vapour density in the alpha*rho_v product:
            # d(alpha * rho_v)/dp = alpha * drho_v_dp
            A21 = self.V_cell / dt * al * drho_v_dp[i]
            # Void accumulation: d(alpha * rho_v)/d(alpha) = rho_v
            A22 = self.V_cell / dt * rv_i

            # Linearize Gamma response to pressure (dGamma/dp term)
            if Gamma[i] > 0 and T_l[i] > T_sat[i]:
                superheat = max(T_l[i] - T_sat[i], 0.1)
                h_fg = max(h_sat_v[i] - h_sat_l[i], 1.0)
                dTsat_dp = T_sat[i] * (1.0 / rv_i - 1.0 / rl_i) / h_fg
                dGamma_dp = -Gamma[i] * dTsat_dp / superheat
                A21 += self.V_cell * dGamma_dp

            # V10: vapor flux coupling to this cell's pressure (diagonal part)
            # When p[i] increases, vapor flux at left face increases INTO cell i
            # (beta_v_L * dp[i]) but also vapor flux at right face increases
            # OUT of cell i (beta_v_R * dp[i]). Net effect on vapor in cell i:
            #   -(beta_v_L + beta_v_R) * dp[i]
            A21 -= (beta_v_L + beta_v_R)

            # RHS Row 2: vapour flux + phase change source
            if has_drift_flux:
                flux_v_old = mdot_v_face[i] - mdot_v_face[i + 1]
            else:
                # Donor-cell vapour flux using old-state values
                if mdot_old[i] >= 0:
                    alpha_in = alpha_old[i - 1] if i > 0 else al
                else:
                    alpha_in = al
                if mdot_old[i + 1] >= 0:
                    alpha_out = al
                else:
                    alpha_out = alpha_old[i + 1] if i < N - 1 else al
                flux_v_old = mdot_old[i] * alpha_in - mdot_old[i + 1] * alpha_out

            # V10: old-state vapor flux coupling to R2 (delta formulation)
            # The off-diagonal terms couple to delta_p, so we need the old
            # vapor flux gradient contribution subtracted from R2:
            #   R2 -= -(beta_v_L + beta_v_R) * p_old[i]
            #         + beta_v_L * p_old[i-1] + beta_v_R * p_old[i+1]
            # This simplifies to: R2 += beta_v_L*(p_old[i] - p_old[i-1])
            #                         + beta_v_R*(p_old[i] - p_old[i+1])
            # But wait -- we need the opposite sign. The vapor flux INTO cell i
            # from the left is +beta_v_L*(dp[i-1] - dp[i]). In the delta
            # formulation, the old-state contribution must be removed:
            #   R2 -= beta_v_L*(p_old[i-1] - p_old[i]) + beta_v_R*(p_old[i+1] - p_old[i])
            # which is: R2 += beta_v_L*(p_old[i] - p_old[i-1])
            #               + beta_v_R*(p_old[i] - p_old[i+1])
            # No -- let me be precise. The full vapor flux coupling is:
            #   vapor_flux_coupling = beta_v_L*(p[i-1] - p[i]) + beta_v_R*(p[i+1] - p[i])
            # In delta form (p = p_old + delta_p):
            #   = beta_v_L*(delta_p[i-1] - delta_p[i]) + beta_v_R*(delta_p[i+1] - delta_p[i])
            #     + beta_v_L*(p_old[i-1] - p_old[i]) + beta_v_R*(p_old[i+1] - p_old[i])
            # The delta_p terms go into the matrix (A21 diagonal, off-diag blocks).
            # The old-state terms go into R2.
            R2_v_old = 0.0
            if beta_v_L > 0 and i > 0:
                R2_v_old += beta_v_L * (p_old[i - 1] - p_old[i])
            if beta_v_R > 0 and i < N - 1:
                R2_v_old += beta_v_R * (p_old[i + 1] - p_old[i])

            R2 = flux_v_old + self.V_cell * Gamma[i] + R2_v_old

            # ---- Assemble blocks ----
            b_blocks[i] = np.array([[A11, A12],
                                    [A21, A22]])
            d_vecs[i] = np.array([R1, R2])

            # Off-diagonal blocks with vapor flux coupling (V10)
            # Row 0 (mixture mass): (0,0) = -bL/-bR (pressure coupling, same as V5)
            # Row 1 (vapor mass): (1,0) = +beta_v_L/+beta_v_R
            #   Positive sign because when p[i-1] increases, vapor flows INTO cell i
            #   (same sign convention as the mixture mass off-diagonal, but for the
            #   vapor fraction of the flux).
            if i > 0:
                a_blocks[i] = np.array([[-bL, 0.0],
                                        [beta_v_L, 0.0]])
            if i < N - 1:
                c_blocks[i] = np.array([[-bR, 0.0],
                                        [beta_v_R, 0.0]])

        # ====================================================================
        # BLOCK THOMAS SOLVE
        # ====================================================================
        x = self._block_thomas_solve(a_blocks, b_blocks, c_blocks, d_vecs)

        # Extract delta_p and delta_alpha from solution
        for i in range(N):
            delta_p = x[i][0]
            delta_alpha = x[i][1]

            # Update pressure
            p_new = p_old[i] + delta_p
            if not np.isfinite(p_new):
                p_new = p_old[i]
            p[i] = max(self.bridge.p_min, min(self.bridge.p_max, p_new))

            # Update void fraction
            alpha_new = alpha_old[i] + delta_alpha
            alpha[i] = max(0.0, min(1.0, alpha_new))

            # Nucleation floor: if phase change is active, ensure minimum void
            if Gamma[i] > 0:
                alpha[i] = max(alpha[i], 1e-3)

        # -- Momentum (with implicit friction) --
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

        # -- Phasic energy (explicit, using _old enthalpy values) --
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
                    h_in = self._muscl_face(h_l_old, i, N, flow_in, h_l_old[0], h_l_old[N - 1])
                    h_out = self._muscl_face(h_l_old, i + 1, N, flow_out, h_l_old[0], h_l_old[N - 1])
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
                    hv_in = self._muscl_face(h_v_old, i, N, flow_in_v, h_v_old[0], h_v_old[N - 1])
                    hv_out = self._muscl_face(h_v_old, i + 1, N, flow_out_v, h_v_old[0], h_v_old[N - 1])
                else:
                    hv_in = h_v_old[i - 1] if (i > 0 and flow_in_v >= 0) else h_v_old[i]
                    hv_out = h_v_old[i] if flow_out_v >= 0 else (h_v_old[i + 1] if i < N - 1 else h_v_old[i])

                flux_v = mv_in * (hv_in - h_v_old[i]) - mv_out * (hv_out - h_v_old[i])
                pw_v = al * self.V_cell * dp_dt[i]
                qi_v = q_i_v[i] * self.V_cell
                phase_v = Gamma[i] * h_v_old[i] * self.V_cell

                h_v[i] = h_v_old[i] + dt / m_v * (flux_v + pw_v + qi_v + phase_v)
                h_v[i] = max(h_sat_v[i], min(h_v[i], 4e6))
            else:
                h_v[i] = h_sat_v[i]
