"""
test_two_phase.py — Phase 2 two-phase solver verification tests.

All tests use SimpleFluid (linear properties) so that every number is
hand-verifiable.  IAPWS-IF97 validation comes later.

Tests
-----
1. Single-phase Hagen-Poiseuille steady state
2. Global mass conservation
3. Global energy conservation
4. Temporal convergence rate (first-order)
5. Heated channel with boiling
6. C++ SimpleFluid vs Modelica reference values
"""

import sys
import os
import numpy as np
import pytest

# Ensure the built .so is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "two_phase"))
import opal_two_phase as tp


# ============================================================================
# Reference geometry and conditions
# ============================================================================

# Pipe geometry (simple round pipe)
N_REF    = 5
DX       = 1.0       # m
A_FLOW   = 0.01      # m^2  (D ~ 0.113 m)
D_H      = 0.1       # m
F_D      = 0.02      # Darcy friction factor

# Boundary conditions (subcooled liquid, Region 1 for SimpleFluid)
P_IN     = 10.1e6    # Pa
P_OUT    = 10.0e6    # Pa
H_IN     = 700.0e3   # J/kg  (well below h_f = 800 kJ/kg at p_ref)

# SimpleFluid constants (replicated for reference)
P_REF    = 10.0e6
RHO_F_0  = 750.0
RHO_F_1  = 20.0
A_L      = 6.25e-5
H_F_0    = 800.0e3
H_F_1    = 100.0e3
H_G_0    = 2800.0e3
H_G_1    = 50.0e3
RHO_G_0  = 40.0
RHO_G_1  = 5.0
CP_L     = 4000.0


def make_solver(N=N_REF):
    """Create solver + fluid + BCs for standard test conditions."""
    fluid  = tp.SimpleFluidProperties()
    solver = tp.TwoPhaseSolver(N, DX, A_FLOW, D_H, F_D, fluid)
    bc     = tp.TwoPhaseBCs(P_IN, P_OUT, H_IN)
    return solver, fluid, bc


def sf_rho_ph(p, h):
    """Python reference: SimpleFluid rho(p,h) for Region 1."""
    p_hat = (p - P_REF) / P_REF
    hf = H_F_0 + H_F_1 * p_hat
    rf = RHO_F_0 + RHO_F_1 * p_hat
    return rf + A_L * (hf - h)


def initial_state(N, p_init=None, h_init=H_IN):
    """Return initial arrays for solver."""
    p    = np.full(N, p_init if p_init is not None else 0.5 * (P_IN + P_OUT))
    h    = np.full(N, h_init)
    mdot = np.zeros(N + 1)
    return p, h, mdot


# ============================================================================
# Test 1: Single-phase Hagen-Poiseuille steady state
# ============================================================================

class TestHagenPoiseuille:
    """Variable-coefficient solver must reproduce H-P in single-phase limit."""

    @pytest.mark.parametrize("N", [1, 5, 10, 20])
    def test_steady_state_flow(self, N):
        """At steady state, mdot is uniform and matches H-P prediction."""
        fluid  = tp.SimpleFluidProperties()
        solver = tp.TwoPhaseSolver(N, DX, A_FLOW, D_H, F_D, fluid)
        bc     = tp.TwoPhaseBCs(P_IN, P_OUT, H_IN)
        p, h, mdot = initial_state(N)

        # CFL for explicit enthalpy: dt < rho*V/mdot.
        # mdot ~ dp/((N+1)*R), so CFL ~ (N+1). Use conservative dt.
        rho_approx = 756.0
        R_approx   = F_D * DX / (2 * D_H * A_FLOW**2 * rho_approx)
        mdot_approx = (P_IN - P_OUT) / ((N + 1) * R_approx)
        dt_cfl = rho_approx * DX * A_FLOW / abs(mdot_approx)
        dt = min(1e-3, 0.5 * dt_cfl)  # safety factor 0.5
        n_steps = max(20000, int(20.0 / dt))  # at least 20 s of simulation

        # Run to steady state
        hist = solver.solve(p, h, mdot, bc, dt, n_steps, n_steps)
        p_ss    = hist[-1, :N]
        mdot_ss = hist[-1, 2*N:]

        # All face flows should be equal (steady state)
        assert np.allclose(mdot_ss, mdot_ss[0], rtol=1e-6), \
            f"Flows not uniform: max spread {np.ptp(mdot_ss):.3e}"

        # Compute analytical H-P reference
        # R_face = f_D*dx/(2*D_h*A^2*rho_face), rho varies by cell but is ~constant
        # Use average pressure to get reference density
        p_avg = np.mean(p_ss)
        rho_avg = sf_rho_ph(p_avg, H_IN)
        R_face = F_D * DX / (2 * D_H * A_FLOW**2 * rho_avg)
        mdot_analytical = (P_IN - P_OUT) / ((N + 1) * R_face)

        rel_err = abs(mdot_ss[0] - mdot_analytical) / mdot_analytical
        assert rel_err < 1e-3, \
            f"H-P mismatch: mdot={mdot_ss[0]:.4f}, expected={mdot_analytical:.4f}, rel_err={rel_err:.2e}"

    def test_linear_pressure_profile(self):
        """Steady-state pressure should be approximately linear."""
        N = 10
        solver, fluid, bc = make_solver(N)
        p, h, mdot = initial_state(N)
        hist = solver.solve(p, h, mdot, bc, 1e-3, 20000, 20000)
        p_ss = hist[-1, :N]

        # Linear interpolation between inlet and outlet
        # Cell i is at position (i+0.5)/(N+1) of the way from inlet to outlet
        # (accounting for half-cells at boundaries)
        p_linear = np.array([
            P_IN - (P_IN - P_OUT) * (i + 1) / (N + 1) for i in range(N)
        ])
        rel_err = np.max(np.abs(p_ss - p_linear) / (P_IN - P_OUT))
        assert rel_err < 1e-3, f"Pressure not linear, max rel_err={rel_err:.2e}"


# ============================================================================
# Test 2: Global mass conservation
# ============================================================================

class TestMassConservation:
    """The tridiagonal system IS the mass balance — must hold to ~machine eps."""

    def test_mass_balance(self):
        N = 5
        solver, fluid, bc = make_solver(N)
        p, h, mdot = initial_state(N)
        dt = 1e-3

        # Step-by-step mass balance check
        max_residual = 0.0
        for _ in range(1000):
            p_old = p.copy()
            solver.step(p, h, mdot, bc, dt)

            # Mass stored: sum_i V * rho_new_i
            # d(mass)/dt = mdot_in - mdot_out
            for i in range(N):
                rho_old = sf_rho_ph(p_old[i], h[i])
                rho_new = sf_rho_ph(p[i], h[i])
                V = DX * A_FLOW
                dm_stored = V * (rho_new - rho_old) / dt
                dm_flow   = mdot[i] - mdot[i + 1]
                residual  = abs(dm_stored - dm_flow)
                # Normalize by typical mass flow
                if abs(mdot[0]) > 1e-10:
                    residual /= abs(mdot[0])
                max_residual = max(max_residual, residual)

        assert max_residual < 1e-4, \
            f"Mass conservation violated: max_residual={max_residual:.2e}"


# ============================================================================
# Test 3: Global energy conservation
# ============================================================================

class TestEnergyConservation:
    """Energy balance: stored + advected + pressure work should balance."""

    def test_energy_balance_steady(self):
        """At steady state: mdot*h_in = mdot*h_out (no wall heat, no dp/dt)."""
        N = 5
        solver, fluid, bc = make_solver(N)
        p, h, mdot = initial_state(N)

        # Run to steady state (no wall heat → h should remain at h_in)
        hist = solver.solve(p, h, mdot, bc, 1e-3, 20000, 20000)
        h_ss = hist[-1, N:2*N]
        mdot_ss = hist[-1, 2*N:]

        # At steady state with no wall heat, enthalpy should be ~h_in everywhere
        # (small pressure work term may cause tiny drift)
        rel_err = np.max(np.abs(h_ss - H_IN) / H_IN)
        assert rel_err < 1e-4, \
            f"Enthalpy drifted from h_in: max rel_err={rel_err:.2e}"

    def test_energy_balance_with_heating(self):
        """With wall heat q: at steady state, h_out = h_in + q_total/mdot."""
        N = 10
        fluid  = tp.SimpleFluidProperties()
        solver = tp.TwoPhaseSolver(N, DX, A_FLOW, D_H, F_D, fluid)
        bc     = tp.TwoPhaseBCs(P_IN, P_OUT, H_IN)
        p, h, mdot = initial_state(N)

        # Apply 50 kW per cell (total 500 kW) — stay subcooled
        q_wall = np.full(N, 50.0e3)  # W per cell

        hist = solver.solve(p, h, mdot, bc, 1e-3, 30000, 30000, q_wall)
        h_ss    = hist[-1, N:2*N]
        mdot_ss = hist[-1, 2*N:]

        # h_out should be h_in + q_total / mdot
        q_total = np.sum(q_wall)
        mdot_out = mdot_ss[-1]
        h_out_expected = H_IN + q_total / mdot_out
        h_out_actual   = h_ss[-1]

        # The last cell enthalpy is close to the outlet enthalpy
        # (donor-cell means h_out = h[N-1] for positive flow)
        rel_err = abs(h_out_actual - h_out_expected) / h_out_expected
        assert rel_err < 5e-3, \
            f"Energy balance: h_out={h_out_actual:.1f}, expected={h_out_expected:.1f}, " \
            f"rel_err={rel_err:.2e}"


# ============================================================================
# Test 4: Temporal convergence rate
# ============================================================================

class TestConvergenceRate:
    """Semi-implicit scheme should be first-order in dt."""

    def test_first_order_enthalpy(self):
        """Halving dt should halve the enthalpy error (explicit Euler)."""
        # Pressure converges in nanoseconds (acoustic), so test the explicit
        # enthalpy update instead (much slower thermal timescale).
        N = 5
        fluid  = tp.SimpleFluidProperties()
        solver = tp.TwoPhaseSolver(N, DX, A_FLOW, D_H, F_D, fluid)
        bc     = tp.TwoPhaseBCs(P_IN, P_OUT, H_IN)
        q_wall = np.full(N, 1.0e6)  # strong heating for visible transient

        # Reference at very small dt
        T_end = 0.5e-3  # short window where enthalpy is evolving
        dt_ref = 1e-7
        n_ref = int(T_end / dt_ref)
        p0, h0, mdot0 = initial_state(N)
        hist_ref = solver.solve(p0, h0, mdot0, bc, dt_ref, n_ref, n_ref, q_wall)
        h_fine = hist_ref[-1, N]  # enthalpy of cell 0

        errors = []
        for dt in [1e-4, 5e-5, 2.5e-5]:
            n_steps = int(T_end / dt)
            p0, h0, mdot0 = initial_state(N)
            hist = solver.solve(p0, h0, mdot0, bc, dt, n_steps, n_steps, q_wall)
            err = abs(hist[-1, N] - h_fine)
            errors.append(err)

        # Convergence rates
        rates = []
        for i in range(len(errors) - 1):
            if errors[i + 1] > 1e-10 and errors[i] > 1e-10:
                rate = np.log2(errors[i] / errors[i + 1])
                rates.append(rate)

        avg_rate = np.mean(rates) if rates else 0.0
        assert avg_rate > 0.7, \
            f"Expected ~first order (rate~1.0), got {avg_rate:.2f}, errors={errors}"


# ============================================================================
# Test 5: Heated channel with boiling
# ============================================================================

class TestHeatedChannelBoiling:
    """Subcooled inlet, uniform wall heat → boiling starts mid-channel."""

    def test_boiling_channel(self):
        """Verify enthalpy profile and mass/energy conservation with two-phase."""
        N = 10
        fluid  = tp.SimpleFluidProperties()

        # High friction → low flow → wall heat causes boiling.
        # f_D=200, R_face ~ 13333 Pa/(kg/s), mdot ~ 0.7 kg/s
        # q=10kW/cell → delta_h ~ 100kW/0.7 = 143 kJ/kg
        # h_out ~ 700+143 = 843 kJ/kg (into two-phase, h_f=800)
        solver = tp.TwoPhaseSolver(N, DX, A_FLOW, D_H, 200.0, fluid)
        bc = tp.TwoPhaseBCs(P_IN, P_OUT, H_IN)
        p, h, mdot = initial_state(N)
        q_wall = np.full(N, 10.0e3)  # 10 kW per cell, 100 kW total

        # Thermal time constant ~200s; run 300s to approach steady state
        # dt=1e-3 is well within CFL for this low-flow case
        dt = 1e-3
        n_steps = 300000
        for chunk in range(3):
            hist = solver.solve(p, h, mdot, bc, dt, n_steps // 3,
                                n_steps // 3, q_wall)
            p    = hist[-1, :N].copy()
            h    = hist[-1, N:2*N].copy()
            mdot = hist[-1, 2*N:].copy()

        p_ss, h_ss, mdot_ss = p, h, mdot

        # 1. Enthalpy should increase monotonically
        dh = np.diff(h_ss)
        assert np.all(dh > 0), \
            f"Enthalpy should increase along heated channel, dh={dh}"

        # 2. Should start subcooled and end in two-phase
        h_f_inlet = H_F_0 + H_F_1 * (p_ss[0] - P_REF) / P_REF
        assert h_ss[0] < h_f_inlet, "First cell should be subcooled"

        h_f_outlet = H_F_0 + H_F_1 * (p_ss[-1] - P_REF) / P_REF
        assert h_ss[-1] > h_f_outlet, \
            f"Last cell should be two-phase: h={h_ss[-1]:.0f}, h_f={h_f_outlet:.0f}"

        # 3. Energy balance: h_out ~ h_in + q_total/mdot
        q_total = np.sum(q_wall)
        mdot_avg = np.mean(mdot_ss)
        h_out_expected = H_IN + q_total / mdot_avg
        rel_err = abs(h_ss[-1] - h_out_expected) / h_out_expected
        assert rel_err < 0.05, \
            f"Energy balance: h_out={h_ss[-1]:.0f}, expected={h_out_expected:.0f}, " \
            f"rel_err={rel_err:.2e}"


# ============================================================================
# Test 6: C++ SimpleFluid vs Modelica reference values
# ============================================================================

class TestSimpleFluidProperties:
    """Verify C++ SimpleFluid matches the Modelica reference exactly."""

    @pytest.fixture
    def fluid(self):
        return tp.SimpleFluidProperties()

    def test_region1_liquid(self, fluid):
        """Subcooled liquid at p=10MPa, h=700kJ/kg."""
        fp = fluid.evaluate(10.0e6, 700.0e3)

        # rho = 750 + 6.25e-5*(800e3-700e3) = 756.25
        assert abs(fp.rho - 756.25) < 1e-10

        # drho_dp_h = (20 + 6.25e-5*100e3)/10e6 = 26.25/10e6 = 2.625e-6
        assert abs(fp.drho_dp_h - 2.625e-6) < 1e-16

        # drho_dh_p = -6.25e-5
        assert abs(fp.drho_dh_p - (-6.25e-5)) < 1e-16

        # T = 400 - (800e3-700e3)/4000 = 375
        assert abs(fp.T - 375.0) < 1e-10

    def test_region2_steam(self, fluid):
        """Superheated steam at p=10MPa, h=3000kJ/kg."""
        fp = fluid.evaluate(10.0e6, 3000.0e3)

        # rho = 40 - 2e-5*(3000e3-2800e3) = 40 - 4 = 36
        assert abs(fp.rho - 36.0) < 1e-10

        # drho_dp_h = (5 + 2e-5*50e3)/10e6 = 6/10e6 = 6e-7
        assert abs(fp.drho_dp_h - 6.0e-7) < 1e-16

        # drho_dh_p = -2e-5
        assert abs(fp.drho_dh_p - (-2.0e-5)) < 1e-16

        # T = 400 + (3000e3-2800e3)/2000 = 400 + 100 = 500
        assert abs(fp.T - 500.0) < 1e-10

    def test_region4_two_phase(self, fluid):
        """Two-phase at p=10MPa, h=1800kJ/kg (mid saturation)."""
        fp = fluid.evaluate(10.0e6, 1800.0e3)

        # At p_ref: h_f=800e3, h_g=2800e3, h_fg=2000e3
        # x = (1800e3 - 800e3)/2000e3 = 0.5
        # rho_f=750, rho_g=40
        # v = 0.5/40 + 0.5/750 = 0.0125 + 6.667e-4 = 0.013167
        # rho = 1/v = 75.949...
        x = 0.5
        vf = 1.0 / 750.0
        vg = 1.0 / 40.0
        v = x * vg + (1 - x) * vf
        rho_expected = 1.0 / v
        assert abs(fp.rho - rho_expected) < 1e-8

        # drho_dh_p = -rho^2 * (1/rho_g - 1/rho_f) / h_fg
        drho_dh_expected = -rho_expected**2 * (vg - vf) / 2000.0e3
        assert abs(fp.drho_dh_p - drho_dh_expected) < 1e-12

        # T = T_sat = 400 K
        assert abs(fp.T - 400.0) < 1e-10

    def test_derivative_consistency_fd(self, fluid):
        """Central finite difference should match analytical derivatives."""
        test_points = [
            (10.0e6, 700.0e3),   # Region 1
            (10.0e6, 3000.0e3),  # Region 2
            (10.0e6, 1800.0e3),  # Region 4
        ]
        for p, h in test_points:
            fp = fluid.evaluate(p, h)

            # drho_dp_h via central FD
            dp = 100.0  # Pa
            rho_plus  = fluid.evaluate(p + dp, h).rho
            rho_minus = fluid.evaluate(p - dp, h).rho
            fd_dp = (rho_plus - rho_minus) / (2 * dp)
            rel_err = abs(fd_dp - fp.drho_dp_h) / (abs(fp.drho_dp_h) + 1e-20)
            assert rel_err < 1e-4, \
                f"drho_dp_h FD mismatch at ({p:.0e},{h:.0e}): " \
                f"analytical={fp.drho_dp_h:.6e}, FD={fd_dp:.6e}, rel_err={rel_err:.2e}"

            # drho_dh_p via central FD
            dh = 10.0  # J/kg
            rho_plus  = fluid.evaluate(p, h + dh).rho
            rho_minus = fluid.evaluate(p, h - dh).rho
            fd_dh = (rho_plus - rho_minus) / (2 * dh)
            rel_err = abs(fd_dh - fp.drho_dh_p) / (abs(fp.drho_dh_p) + 1e-20)
            assert rel_err < 1e-4, \
                f"drho_dh_p FD mismatch at ({p:.0e},{h:.0e}): " \
                f"analytical={fp.drho_dh_p:.6e}, FD={fd_dh:.6e}, rel_err={rel_err:.2e}"


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
