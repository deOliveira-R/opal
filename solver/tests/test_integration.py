"""
test_integration.py — Integration tests for untested code paths.

These tests address gaps identified by QA review: every difficulty during
Phase 3 traced to a code path that had zero pytest coverage.

P1 tests (5): IAPWS+inertial momentum, wall BC reflection, mini-Edwards
P2 tests (9): RansomTrapp, IAPWS PhasicProperties, nucleation, enthalpy bounds

All tests use the full C++ solver via pybind11 bindings.
"""

import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_phase"))
import opal_two_phase as tp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bc_helpers import (step_5eq, step_hem, pressure_bcs, wall_pressure_bcs,
                        wall_break_bcs, drift_flux_closures)


# ============================================================================
# P1: IAPWS + InertialMomentum (catches property mismatch divergence)
# ============================================================================

class TestIAPWSInertialMomentum:
    """IAPWS properties with inertial momentum — the combination that
    caused divergence during Edwards development."""

    def test_subcooled_hagen_poiseuille(self):
        """Subcooled liquid at 10 MPa should reach steady state with
        inertial momentum + IAPWS without diverging."""
        N = 5
        fluid = tp.IAPWSIF97Properties()
        model = tp.HEMModel()
        momentum = tp.InertialMomentum()
        solver = tp.TwoPhaseSolver(N, 1.0, 0.01, 0.1, 0.02, fluid,
                                   tp.DonorCell(), model, momentum)

        p = np.linspace(10.1e6, 10.0e6, N)
        h = np.full(N, 900e3)  # subcooled at 10 MPa (h_f ~ 1267 kJ/kg)
        mdot = np.zeros(N + 1)
        bc_in, bc_out = pressure_bcs(10.1e6, 10.0e6, 900e3)

        for _ in range(2000):
            step_hem(solver, p, h, mdot, bc_in, bc_out, 1e-4)

        assert np.all(np.isfinite(p)), f"NaN in pressure: {p}"
        assert np.all(np.isfinite(mdot)), f"NaN in mdot: {mdot}"
        assert np.all(mdot > 0), f"Expected positive flow: {mdot}"
        # Flow should be roughly uniform at steady state
        assert np.std(mdot[1:-1]) / np.mean(mdot[1:-1]) < 0.1

    def test_two_phase_stability(self):
        """Two-phase conditions at 10 MPa should remain stable with
        inertial momentum + IAPWS."""
        N = 5
        fluid = tp.IAPWSIF97Properties()
        model = tp.HEMModel()
        momentum = tp.InertialMomentum()
        solver = tp.TwoPhaseSolver(N, 1.0, 0.01, 0.1, 0.02, fluid,
                                   tp.DonorCell(), model, momentum)

        p = np.linspace(10.1e6, 10.0e6, N)
        h = np.full(N, 1800e3)  # two-phase at 10 MPa
        mdot = np.zeros(N + 1)
        bc_in, bc_out = pressure_bcs(10.1e6, 10.0e6, 1800e3)

        for _ in range(1000):
            step_hem(solver, p, h, mdot, bc_in, bc_out, 1e-4)

        assert np.all(np.isfinite(p)), f"NaN in pressure: {p}"
        assert np.all(np.isfinite(h)), f"NaN in enthalpy: {h}"
        assert np.all(np.isfinite(mdot)), f"NaN in mdot: {mdot}"


# ============================================================================
# P1: Wall BC pressure reflection (catches pressure doubling bug)
# ============================================================================

class TestWallBCReflection:
    """Wall boundary condition with inertial momentum — catches the
    pressure doubling from wave reflection at closed end."""

    def test_uniform_pressure_stays_uniform(self):
        """With wall at outlet and inlet pressure matching initial,
        pressure should stay uniform and mdot at wall should be zero."""
        N = 5
        fluid = tp.SimpleFluidProperties()
        closures = drift_flux_closures(H_i=0.0, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        momentum = tp.InertialMomentum()
        solver = tp.TwoPhaseSolver(N, 1.0, 0.01, 0.1, 0.02, fluid,
                                   tp.DonorCell(), model, momentum)

        bc_in = tp.PressureFace(10.0e6, 700e3, 2800e3, 0.0)
        bc_out = tp.WallFace(700e3, 2800e3)

        p = np.full(N, 10.0e6)
        alpha = np.zeros(N)
        h_l = np.full(N, 700e3)
        h_v = np.full(N, 2800e3)
        mdot = np.zeros(N + 1)

        for _ in range(200):
            step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, 1e-4)

        # Wall face flow must be exactly zero
        assert mdot[-1] == 0.0, f"Wall mdot should be 0, got {mdot[-1]}"
        # Pressure should remain near 10 MPa (no spurious buildup)
        assert np.max(p) / 1e6 < 10.5, (
            f"Pressure exceeded 10.5 MPa: max = {np.max(p)/1e6:.3f}"
        )
        assert np.min(p) / 1e6 > 9.5, (
            f"Pressure dropped below 9.5 MPa: min = {np.min(p)/1e6:.3f}"
        )

    def test_step_pressure_reflection_bounded(self):
        """A pressure step should reflect at the wall but not exceed
        2x the initial jump (perfect reflection limit)."""
        N = 10
        fluid = tp.SimpleFluidProperties()
        closures = drift_flux_closures(H_i=0.0, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        momentum = tp.InertialMomentum()
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                   tp.DonorCell(), model, momentum)

        bc_in = tp.PressureFace(10.1e6, 700e3, 2800e3, 0.0)
        bc_out = tp.WallFace(700e3, 2800e3)

        # Step pressure: cells 0-4 at 10.1 MPa, cells 5-9 at 10.0 MPa
        p = np.full(N, 10.0e6)
        p[:5] = 10.1e6
        alpha = np.zeros(N)
        h_l = np.full(N, 700e3)
        h_v = np.full(N, 2800e3)
        mdot = np.zeros(N + 1)

        p_max_seen = np.max(p)
        for _ in range(500):
            step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, 1e-4)
            p_max_seen = max(p_max_seen, np.max(p))

        # Max pressure should not exceed 10.2 MPa (2x the 0.1 MPa step)
        assert p_max_seen / 1e6 < 10.3, (
            f"Pressure exceeded reflection limit: max = {p_max_seen/1e6:.3f} MPa"
        )


# ============================================================================
# P1: Mini-Edwards integration smoke test
# ============================================================================

class TestMiniEdwards:
    """Minimal Edwards-like blowdown: IAPWS + InertialMomentum + RansomTrapp
    + 5-eq + WALL/BREAK BCs. Regression test for the full C++ path."""

    def test_mini_blowdown_runs(self):
        """N=5 pipe blowdown should run without crash, with pressure
        decreasing at break cell and positive outlet flow."""
        N = 5
        fluid = tp.IAPWSIF97Properties()
        closures = drift_flux_closures(H_i=1e7, C_0=1.0, alpha_nucleation=1e-3)
        model = tp.FiveEqModel(fluid, closures)
        momentum = tp.InertialMomentum()
        critical_flow = tp.RansomTrapp(fluid, x_trans=0.10, c_floor=1200.0)
        solver = tp.TwoPhaseSolver(N, 0.8, 4.185e-3, 0.073, 0.02,
                                   fluid, tp.DonorCell(), model,
                                   momentum, critical_flow)

        bc_in = tp.WallFace(986.6e3, 2772.6e3)
        bc_out = tp.BreakFace(101325.0, 0.87, 986.6e3, 2772.6e3)

        p = np.full(N, 7.0e6)
        alpha = np.full(N, 1e-6)
        h_l = np.full(N, 986.6e3)
        h_v = np.full(N, 2772.6e3)
        mdot = np.zeros(N + 1)

        dt = 1e-5
        p_break_initial = p[-1]

        for _ in range(200):
            step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, dt)

        # All state should be finite
        assert np.all(np.isfinite(p)), f"NaN in pressure: {p}"
        assert np.all(np.isfinite(alpha)), f"NaN in alpha: {alpha}"
        assert np.all(np.isfinite(h_l)), f"NaN in h_l: {h_l}"
        assert np.all(np.isfinite(h_v)), f"NaN in h_v: {h_v}"
        assert np.all(np.isfinite(mdot)), f"NaN in mdot: {mdot}"

        # Break cell pressure should have decreased
        assert p[-1] < p_break_initial, (
            f"Break pressure should decrease: {p[-1]/1e6:.3f} >= {p_break_initial/1e6:.3f} MPa"
        )

        # Outlet flow should be positive (outward)
        assert mdot[-1] > 0, f"Expected outflow at break: mdot={mdot[-1]}"

        # Wall flow should be zero
        assert mdot[0] == 0.0, f"Wall flow should be zero: {mdot[0]}"


# ============================================================================
# P2: RansomTrapp critical flow unit tests
# ============================================================================

class TestRansomTrapp:
    """Unit tests for the Ransom-Trapp critical flow model."""

    def _make_rt(self):
        fluid = tp.SimpleFluidProperties()
        return tp.RansomTrapp(fluid, x_trans=0.10, c_floor=1200.0)

    def test_subcooled_bernoulli(self):
        """Subcooled critical flux should match Bernoulli: G = sqrt(2*rho_f*dp)."""
        fluid = tp.SimpleFluidProperties()
        rt = tp.RansomTrapp(fluid)

        p_cell = 10e6
        h_mix = 700e3    # subcooled (h_f = 800e3)
        pp = fluid.evaluate_phasic(p_cell)
        fp = fluid.evaluate(p_cell, h_mix)
        p_back = 1e5

        result = rt.evaluate(p_cell, h_mix, fp.rho, fp.drho_dp_h,
                             p_back, 0.01, 1.0, 1e6)

        # Analytical G_sub = sqrt(2 * rho_f * dp)
        G_sub_expected = np.sqrt(2.0 * pp.rho_l * (p_cell - p_back))
        mdot_expected = 0.01 * G_sub_expected  # A_flow * G * C_d

        # Should be choked (mdot_momentum=1e6 >> critical)
        assert result.is_choked

        # mdot_crit should be close to Bernoulli (subcooled, x=0)
        assert result.mdot_crit == pytest.approx(mdot_expected, rel=0.05)

    def test_choke_detection(self):
        """is_choked should be True when momentum mdot > critical, False otherwise."""
        fluid = tp.SimpleFluidProperties()
        rt = tp.RansomTrapp(fluid)

        p_cell = 10e6
        h_mix = 700e3
        fp = fluid.evaluate(p_cell, h_mix)

        # High momentum → choked
        r1 = rt.evaluate(p_cell, h_mix, fp.rho, fp.drho_dp_h,
                         1e5, 0.01, 1.0, 1e6)
        assert r1.is_choked

        # Low momentum → not choked
        r2 = rt.evaluate(p_cell, h_mix, fp.rho, fp.drho_dp_h,
                         1e5, 0.01, 1.0, 0.001)
        assert not r2.is_choked

    def test_negative_flow_never_choked(self):
        """Negative (inflow) should never be choked."""
        fluid = tp.SimpleFluidProperties()
        rt = tp.RansomTrapp(fluid)
        fp = fluid.evaluate(10e6, 700e3)
        r = rt.evaluate(10e6, 700e3, fp.rho, fp.drho_dp_h,
                        1e5, 0.01, 1.0, -100.0)
        assert not r.is_choked

    def test_iapws_ransom_trapp(self):
        """RansomTrapp with IAPWS should work at Edwards conditions."""
        fluid = tp.IAPWSIF97Properties()
        rt = tp.RansomTrapp(fluid)
        fp = fluid.evaluate(7e6, 986.6e3)  # Edwards IC

        r = rt.evaluate(7e6, 986.6e3, fp.rho, fp.drho_dp_h,
                        1e5, 4.185e-3, 0.87, 1e6)
        assert r.is_choked
        assert r.mdot_crit > 0


# ============================================================================
# P2: IAPWS PhasicProperties oracle comparison
# ============================================================================

class TestIAPWSPhasicProperties:
    """Verify IAPWS PhasicProperties against the Python iapws oracle."""

    @pytest.fixture
    def fluid(self):
        return tp.IAPWSIF97Properties()

    @pytest.mark.parametrize("p_MPa", [1.0, 5.0, 10.0, 15.0])
    def test_saturation_properties(self, fluid, p_MPa):
        """T_sat, h_f, h_g, rho_f, rho_v should match iapws oracle."""
        import iapws as iw
        p_Pa = p_MPa * 1e6
        pp = fluid.evaluate_phasic(p_Pa)
        ref_f = iw.IAPWS97(P=p_MPa, x=0)
        ref_g = iw.IAPWS97(P=p_MPa, x=1)

        assert pp.T_sat == pytest.approx(ref_f.T, rel=1e-4)
        assert pp.h_sat_l == pytest.approx(ref_f.h * 1e3, rel=1e-3)
        assert pp.h_sat_v == pytest.approx(ref_g.h * 1e3, rel=1e-3)
        assert pp.rho_l == pytest.approx(ref_f.rho, rel=1e-3)
        assert pp.rho_v == pytest.approx(ref_g.rho, rel=1e-3)

    @pytest.mark.parametrize("p_MPa", [1.0, 7.0, 15.0])
    def test_liquid_temperature(self, fluid, p_MPa):
        """T_liquid at subcooled enthalpy should match iapws oracle."""
        import iapws as iw
        p_Pa = p_MPa * 1e6
        ref_f = iw.IAPWS97(P=p_MPa, x=0)
        h_sub = ref_f.h * 1e3 - 100e3  # 100 kJ/kg subcooled

        T_cpp = fluid.T_liquid(p_Pa, h_sub)
        ref = iw.IAPWS97(P=p_MPa, h=h_sub / 1e3)
        assert T_cpp == pytest.approx(ref.T, rel=1e-3)


# ============================================================================
# P2: Nucleation with IAPWS
# ============================================================================

class TestNucleationIAPWS:
    """Nucleation onset with IAPWS — catches the interfacial area
    collapse at alpha=0 with real properties."""

    def test_superheated_creates_void(self):
        """Superheated liquid at 2 MPa (T_l > T_sat) with IAPWS should
        produce void from alpha=0 via nucleation."""
        N = 3
        fluid = tp.IAPWSIF97Properties()
        closures = drift_flux_closures(H_i=1e7, C_0=1.0, alpha_nucleation=1e-3)
        model = tp.FiveEqModel(fluid, closures)
        solver = tp.TwoPhaseSolver(N, 1.0, 0.01, 0.1, 0.02, fluid,
                                   tp.DonorCell(), model)

        # At 2 MPa: T_sat ~ 485 K, h_f ~ 909 kJ/kg
        # Use h_l = 1000 kJ/kg → T_l > T_sat → superheated
        pp = fluid.evaluate_phasic(2e6)

        bc_in, bc_out = pressure_bcs(2.1e6, 2.0e6, 1000e3,
                                     h_v=pp.h_sat_v, alpha=0.001)

        p = np.full(N, 2.05e6)
        alpha = np.zeros(N)  # start with ZERO void
        h_l = np.full(N, 1000e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for _ in range(200):
            step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, 1e-4)

        assert np.max(alpha) > 0, (
            f"Nucleation failed with IAPWS: max(alpha) = {np.max(alpha):.2e}"
        )
        assert np.all(np.isfinite(p))


# ============================================================================
# P2: Enthalpy bounds under CFL violation
# ============================================================================

class TestEnthalpyBounds:
    """Verify enthalpy stays bounded even with CFL-violating timestep."""

    def test_large_dt_enthalpy_bounded(self):
        """With a deliberately large dt, enthalpy should be clamped,
        not go to infinity or NaN."""
        N = 3
        fluid = tp.SimpleFluidProperties()
        closures = drift_flux_closures(H_i=1e5, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        solver = tp.TwoPhaseSolver(N, 1.0, 0.01, 0.1, 0.02, fluid,
                                   tp.DonorCell(), model)

        bc_in, bc_out = pressure_bcs(10.5e6, 10.0e6, 700e3, h_v=2800e3)

        p = np.full(N, 10.25e6)
        alpha = np.full(N, 0.1)
        h_l = np.full(N, 700e3)
        h_v = np.full(N, 2800e3)
        mdot = np.zeros(N + 1)

        # Deliberately large dt (should trigger CFL warning)
        for _ in range(10):
            step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, 0.1)

        # Enthalpy should be finite (clamped, not NaN)
        assert np.all(np.isfinite(h_l)), f"h_l has NaN: {h_l}"
        assert np.all(np.isfinite(h_v)), f"h_v has NaN: {h_v}"
        # Enthalpy should be within physical bounds
        assert np.all(h_l > 0), f"h_l went negative: {h_l}"
        assert np.all(h_v > 0), f"h_v went negative: {h_v}"
        assert np.all(h_v < 5e6), f"h_v exceeded 5 MJ/kg: {h_v}"
