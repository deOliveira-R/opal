"""
P0 tests — Would have caught the two escaped bugs.

Bug 1: q_i_l sign convention (closures.hpp)
  q_i_l = H_i * a_i * (T_sat - T_l) = heat INTO liquid.
  Negative when superheated (heat leaves liquid to drive evaporation).

Bug 2: Vapor enthalpy floor (five_eq_model.cpp)
  h_v must be clamped to [h_sat_v, h_max], not [h_sat_l, h_max].
  h_v < h_sat_v produces invalid IAPWS R2 inputs (negative density).

Each test verifies ONE behavior of ONE term. No emergent behavior tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_phase"))
import opal_two_phase as tp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bc_helpers import step_5eq_migrated, step_hem_migrated, solve_migrated, reset_time


# ============================================================================
# Helpers
# ============================================================================

def make_interfacial_state(
    T_l, T_sat, alpha=0.3,
    p=10e6, rho_l=750.0, rho_v=35.0,
    h_l=900e3, h_v=2800e3,
    h_sat_l=800e3, h_sat_v=2780e3,
    cp_l=4200.0, sigma=0.05, D_h=0.073,
    T_v=None,
):
    """Construct an InterfacialState with controlled (T_l, T_sat)."""
    s = tp.InterfacialState()
    s.p = p
    s.alpha = alpha
    s.rho_l = rho_l
    s.rho_v = rho_v
    s.h_l = h_l
    s.h_v = h_v
    s.T_l = T_l
    s.T_v = T_v if T_v is not None else T_sat + 5.0
    s.T_sat = T_sat
    s.h_sat_l = h_sat_l
    s.h_sat_v = h_sat_v
    s.cp_l = cp_l
    s.sigma = sigma
    s.D_h = D_h
    return s


def make_5eq_solver(N=5, fluid=None):
    """Standard 5-eq solver with SimpleFluid for isolated testing."""
    if fluid is None:
        fluid = tp.SimpleFluidProperties()
    closures = tp.DriftFluxClosures(H_i=1e6, C_0=1.0, alpha_nucleation=1e-3)
    model = tp.FiveEqModel(fluid, closures)
    dx = 0.5
    A = 0.01
    D_h = 0.1
    f_D = 0.02
    solver = tp.TwoPhaseSolver(N, dx, A, D_h, f_D, fluid,
                                tp.DonorCell(), model, tp.InertialMomentum())
    return solver, fluid


# ============================================================================
# P0-1: Closure sign convention — q_i_l direction
# ============================================================================

class TestClosureSignConvention:
    """Direct unit tests of DriftFluxClosures::compute() sign convention.
    q_i_l = heat INTO liquid. Negative when liquid is superheated."""

    def test_superheated_liquid_q_i_l_negative(self):
        """When T_l > T_sat, heat LEAVES liquid: q_i_l < 0."""
        s = make_interfacial_state(T_l=510.0, T_sat=500.0)
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        r = c.compute(s)
        assert r.q_i_l < 0, f"q_i_l should be negative when superheated, got {r.q_i_l}"

    def test_superheated_liquid_Gamma_positive(self):
        """When T_l > T_sat, evaporation occurs: Gamma > 0."""
        s = make_interfacial_state(T_l=510.0, T_sat=500.0)
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        r = c.compute(s)
        assert r.Gamma > 0, f"Gamma should be positive (evaporation), got {r.Gamma}"

    def test_subcooled_liquid_q_i_l_positive(self):
        """When T_l < T_sat, heat ENTERS liquid: q_i_l > 0."""
        s = make_interfacial_state(T_l=490.0, T_sat=500.0)
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        r = c.compute(s)
        assert r.q_i_l > 0, f"q_i_l should be positive when subcooled, got {r.q_i_l}"

    def test_subcooled_liquid_Gamma_negative(self):
        """When T_l < T_sat, condensation occurs: Gamma < 0."""
        s = make_interfacial_state(T_l=490.0, T_sat=500.0)
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        r = c.compute(s)
        assert r.Gamma < 0, f"Gamma should be negative (condensation), got {r.Gamma}"

    def test_equilibrium_zero_transfer(self):
        """When T_l = T_sat exactly, no interfacial transfer."""
        s = make_interfacial_state(T_l=500.0, T_sat=500.0)
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        r = c.compute(s)
        assert r.q_i_l == pytest.approx(0.0, abs=1e-10)
        assert r.Gamma == pytest.approx(0.0, abs=1e-10)


# ============================================================================
# P0-2: Closure energy balance identity
# ============================================================================

class TestClosureEnergyBalance:
    """The interfacial energy balance q_i_l + q_i_v + Gamma*(h_v - h_l) = 0
    must hold exactly (enforced by construction)."""

    @pytest.mark.parametrize("T_l,T_sat,alpha,h_l,h_v,desc", [
        (510, 500, 0.3, 900e3, 2800e3, "superheated, moderate void"),
        (490, 500, 0.3, 750e3, 2800e3, "subcooled, moderate void"),
        (500.1, 500, 0.5, 800e3, 2780e3, "near-saturation, 50% void"),
        (520, 500, 0.95, 950e3, 2900e3, "high void, strong superheat"),
        (480, 500, 0.01, 700e3, 2750e3, "low void, subcooled"),
        (530, 500, 0.001, 1000e3, 2800e3, "nucleation-level void"),
    ])
    def test_energy_balance_identity(self, T_l, T_sat, alpha, h_l, h_v, desc):
        """q_i_l + q_i_v + Gamma*(h_v - h_l) = 0 for various states."""
        s = make_interfacial_state(T_l=T_l, T_sat=T_sat, alpha=alpha,
                                   h_l=h_l, h_v=h_v)
        c = tp.DriftFluxClosures(H_i=1e6, C_0=1.0)
        r = c.compute(s)

        residual = r.q_i_l + r.q_i_v + r.Gamma * (h_v - h_l)
        scale = max(abs(r.q_i_l), abs(r.q_i_v),
                    abs(r.Gamma * (h_v - h_l)), 1.0)
        assert abs(residual) < 1e-8 * scale, (
            f"Energy balance violated ({desc}): "
            f"q_i_l={r.q_i_l:.2e} + q_i_v={r.q_i_v:.2e} + "
            f"Gamma*dh={r.Gamma*(h_v-h_l):.2e} = {residual:.2e}"
        )


# ============================================================================
# P0-3: Closure Gamma magnitude (hand calculation)
# ============================================================================

class TestClosureGammaMagnitude:
    """Verify Gamma = -q_i_l / h_fg against exact hand calculation."""

    def test_gamma_exact_value(self):
        """Compute Gamma from known inputs and verify against formula."""
        H_i = 1e5
        T_l = 510.0
        T_sat = 500.0
        dT = T_sat - T_l  # = -10 K
        alpha = 0.3
        h_sat_l = 800e3
        h_sat_v = 2800e3
        h_fg = h_sat_v - h_sat_l  # = 2e6

        # a_i = max(4*0.3*0.7, 0.3) = max(0.84, 0.3) = 0.84
        a_i_expected = max(4 * alpha * (1 - alpha), alpha)
        assert a_i_expected == pytest.approx(0.84, rel=1e-10)

        # q_i_l = H_i * a_i * (T_sat - T_l) = 1e5 * 0.84 * (-10) = -840000
        q_i_l_expected = H_i * a_i_expected * dT
        assert q_i_l_expected == pytest.approx(-840000.0, rel=1e-10)

        # Gamma = -q_i_l / h_fg = 840000 / 2e6 = 0.42
        Gamma_expected = -q_i_l_expected / h_fg
        assert Gamma_expected == pytest.approx(0.42, rel=1e-10)

        # Now verify C++ closure matches
        s = make_interfacial_state(T_l=T_l, T_sat=T_sat, alpha=alpha,
                                   h_sat_l=h_sat_l, h_sat_v=h_sat_v)
        c = tp.DriftFluxClosures(H_i=H_i, C_0=1.0)
        r = c.compute(s)

        assert r.q_i_l == pytest.approx(q_i_l_expected, rel=1e-10)
        assert r.Gamma == pytest.approx(Gamma_expected, rel=1e-10)

    def test_gamma_scales_with_H_i(self):
        """Doubling H_i doubles Gamma."""
        s = make_interfacial_state(T_l=510.0, T_sat=500.0, alpha=0.3)

        c1 = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        c2 = tp.DriftFluxClosures(H_i=2e5, C_0=1.0)

        r1 = c1.compute(s)
        r2 = c2.compute(s)

        assert r2.Gamma == pytest.approx(2.0 * r1.Gamma, rel=1e-10)

    def test_gamma_scales_with_superheat(self):
        """Doubling dT doubles Gamma."""
        s1 = make_interfacial_state(T_l=510.0, T_sat=500.0, alpha=0.3)
        s2 = make_interfacial_state(T_l=520.0, T_sat=500.0, alpha=0.3)

        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)

        r1 = c.compute(s1)
        r2 = c.compute(s2)

        assert r2.Gamma == pytest.approx(2.0 * r1.Gamma, rel=1e-10)


# ============================================================================
# P0-4: Vapor enthalpy floor at h_sat_v
# ============================================================================

class TestVaporEnthalpyFloor:
    """After every transport update, h_v[i] >= h_sat_v(p[i])."""

    def test_h_v_above_h_sat_v_after_step(self):
        """Run several steps and verify h_v >= h_sat_v for all cells."""
        solver, fluid = make_5eq_solver(N=5)
        N = 5

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6
        bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_l + 50e3  # slightly above saturation
        bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.1)
        h_l = np.full(N, pp.h_sat_l + 50e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)
        dt = 1e-4

        for step in range(200):
            step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, dt)
            for i in range(N):
                pp_i = fluid.evaluate_phasic(p[i])
                assert h_v[i] >= pp_i.h_sat_v - 1.0, (
                    f"step={step}, cell={i}: h_v={h_v[i]/1e3:.1f} kJ/kg < "
                    f"h_sat_v={pp_i.h_sat_v/1e3:.1f} kJ/kg at p={p[i]/1e6:.3f} MPa"
                )

    def test_h_v_floor_enforced_on_undershoot(self):
        """Initialize h_v below h_sat_v; after one step it should be clamped up."""
        solver, fluid = make_5eq_solver(N=3)
        N = 3

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 10e6
        bc.h_in = pp.h_sat_l; bc.h_l_in = pp.h_sat_l; bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.05)
        h_l = np.full(N, pp.h_sat_l)
        h_v = np.full(N, pp.h_sat_v - 200e3)  # 200 kJ/kg below floor
        mdot = np.zeros(N + 1)

        step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, 1e-4)

        for i in range(N):
            pp_i = fluid.evaluate_phasic(p[i])
            assert h_v[i] >= pp_i.h_sat_v - 1.0, (
                f"cell={i}: h_v={h_v[i]/1e3:.1f} < h_sat_v={pp_i.h_sat_v/1e3:.1f}"
            )


# ============================================================================
# P0-5: Vapor density positivity
# ============================================================================

class TestVaporDensityPositivity:
    """rho_vapor(p, h_v) must be positive for all solver-computed states."""

    def test_density_positive_during_flashing(self):
        """Run a flashing transient and check rho_v > 0 at every step."""
        solver, fluid = make_5eq_solver(N=5)
        N = 5

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.0e6
        bc.h_in = pp.h_sat_l + 100e3  # superheated
        bc.h_l_in = bc.h_in; bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.01)
        h_l = np.full(N, pp.h_sat_l + 100e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for step in range(500):
            step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, 1e-4)
            for i in range(N):
                if alpha[i] > 1e-8:
                    rho_v = fluid.rho_vapor(p[i], h_v[i])
                    assert rho_v > 0, (
                        f"step={step}, cell={i}: rho_v={rho_v:.2f} <= 0 at "
                        f"p={p[i]/1e6:.3f} MPa, h_v={h_v[i]/1e3:.1f} kJ/kg"
                    )

    def test_density_positive_during_condensation(self):
        """Run a condensation transient and check rho_v > 0."""
        solver, fluid = make_5eq_solver(N=5)
        N = 5

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_l - 50e3  # subcooled inlet
        bc.h_l_in = bc.h_in; bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.3)  # start with vapor present
        h_l = np.full(N, pp.h_sat_l)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for step in range(500):
            step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, 1e-4)
            for i in range(N):
                if alpha[i] > 1e-8:
                    rho_v = fluid.rho_vapor(p[i], h_v[i])
                    assert rho_v > 0, (
                        f"step={step}, cell={i}: rho_v={rho_v:.2f} at "
                        f"h_v={h_v[i]/1e3:.1f}, h_sat_v={pp.h_sat_v/1e3:.1f}"
                    )


# ============================================================================
# P0-6: Liquid enthalpy bounds
# ============================================================================

class TestLiquidEnthalpyBounds:
    """h_l is clamped to [h_min=1e4, h_sat_v(p)]."""

    def test_h_l_below_ceiling(self):
        """h_l should never exceed h_sat_v(p), even with strong heating."""
        solver, fluid = make_5eq_solver(N=3)
        N = 3

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_v - 10e3  # very high inlet enthalpy
        bc.h_l_in = bc.h_in; bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.01)
        h_l = np.full(N, pp.h_sat_v - 10e3)  # near ceiling
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        # Strong wall heating to push h_l up
        q_wall = np.full(N, 1e8)  # 100 MW/m3

        for step in range(100):
            step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, 1e-4, q_wall)
            for i in range(N):
                pp_i = fluid.evaluate_phasic(p[i])
                assert h_l[i] <= pp_i.h_sat_v + 1.0, (
                    f"step={step}, cell={i}: h_l={h_l[i]/1e3:.1f} > "
                    f"h_sat_v={pp_i.h_sat_v/1e3:.1f}"
                )

    def test_h_l_above_floor(self):
        """h_l should never go below h_min=1e4 J/kg, even with strong cooling."""
        solver, fluid = make_5eq_solver(N=3)
        N = 3

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = 50e3  # very low enthalpy
        bc.h_l_in = bc.h_in; bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.01)
        h_l = np.full(N, 50e3)  # 50 kJ/kg
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        # Strong wall cooling
        q_wall = np.full(N, -1e8)

        for step in range(100):
            step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, 1e-4, q_wall)
            for i in range(N):
                assert h_l[i] >= 1e4 - 1.0, (
                    f"step={step}, cell={i}: h_l={h_l[i]:.1f} < h_min=1e4"
                )


# ============================================================================
# P0-7: Superheated liquid enthalpy DECREASES toward saturation
# ============================================================================

class TestSuperheatedEnthalpyDirection:
    """When liquid is superheated (h_l > h_sat_l at local p),
    interfacial heat transfer should COOL the liquid (h_l decreases).
    This test would have caught the sign bug directly."""

    def test_h_l_decreases_when_superheated(self):
        """Superheated liquid: h_l must decrease toward h_sat_l over time."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e6, C_0=1.0, alpha_nucleation=1e-3)
        model = tp.FiveEqModel(fluid, closures)
        N = 5
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        h_l_init = pp.h_sat_l + 100e3  # 100 kJ/kg above saturation

        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 10e6  # equal pressure → minimal flow
        bc.h_in = h_l_init; bc.h_l_in = h_l_init; bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.1)  # two-phase present
        h_l = np.full(N, h_l_init)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for step in range(500):
            step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, 1e-4)

        # Interior cells (avoid boundary effects at cells 0 and N-1)
        for i in range(1, N - 1):
            assert h_l[i] < h_l_init, (
                f"cell={i}: h_l={h_l[i]/1e3:.1f} kJ/kg did NOT decrease "
                f"from initial {h_l_init/1e3:.1f} kJ/kg. "
                f"Superheated liquid must cool toward saturation."
            )

    def test_h_l_approaches_saturation(self):
        """With strong H_i, h_l should approach h_sat_l (not overshoot to zero)."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e7, C_0=1.0, alpha_nucleation=1e-3)
        model = tp.FiveEqModel(fluid, closures)
        N = 3
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        h_l_init = pp.h_sat_l + 50e3

        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 10e6
        bc.h_in = h_l_init; bc.h_l_in = h_l_init; bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.1)
        h_l = np.full(N, h_l_init)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for step in range(2000):
            step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, 1e-4)

        # h_l should be closer to h_sat_l than to initial value
        mid = N // 2
        pp_mid = fluid.evaluate_phasic(p[mid])
        dist_to_sat = abs(h_l[mid] - pp_mid.h_sat_l)
        dist_initial = abs(h_l_init - pp_mid.h_sat_l)
        assert dist_to_sat < 0.5 * dist_initial, (
            f"h_l={h_l[mid]/1e3:.1f} kJ/kg not approaching h_sat_l="
            f"{pp_mid.h_sat_l/1e3:.1f} kJ/kg (initial was {h_l_init/1e3:.1f})"
        )


# ============================================================================
# P0-extra: IAPWS rho_vapor returns garbage for invalid inputs
# ============================================================================

class TestIAPWSInvalidInput:
    """Document that IAPWS rho_vapor produces non-physical output
    when h_v < h_sat_v. This proves the solver MUST prevent such inputs."""

    def test_rho_vapor_invalid_below_h_sat_v(self):
        """rho_vapor(p, h_v < h_sat_v) returns non-physical value."""
        fluid = tp.IAPWSIF97Properties()
        p = 1.5e6
        pp = fluid.evaluate_phasic(p)

        # Valid input: h_v = h_sat_v → should give positive rho_v
        rho_valid = fluid.rho_vapor(p, pp.h_sat_v)
        assert rho_valid > 0, f"rho_vapor at saturation should be positive"
        assert rho_valid == pytest.approx(pp.rho_v, rel=0.01)

        # Invalid input: h_v well below h_sat_v → non-physical
        h_v_invalid = pp.h_sat_v - 250e3
        rho_invalid = fluid.rho_vapor(p, h_v_invalid)
        # The result is non-physical (negative or wildly wrong)
        is_non_physical = (rho_invalid <= 0 or rho_invalid > 1000
                          or np.isnan(rho_invalid))
        assert is_non_physical, (
            f"IAPWS rho_vapor({p/1e6:.1f} MPa, {h_v_invalid/1e3:.0f} kJ/kg) = "
            f"{rho_invalid:.2f} — expected non-physical result for invalid input"
        )
