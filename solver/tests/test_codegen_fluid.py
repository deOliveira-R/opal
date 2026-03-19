"""
test_codegen_fluid.py — L0/L1/L2 verification of Case 2 (translateModel C codegen).

Verifies that OM-generated C code, compiled to .so and wrapped in
ModelicaFluidPackage, produces identical results to the C++ FluidPackage
and hand calculations.
"""

import sys
import os
import math
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "two_phase"))

OPAL_ROOT = Path(__file__).resolve().parents[2]
SO_PATH = OPAL_ROOT / "feasibility" / "results" / "opal_codegen_InlineTest.so"


@pytest.fixture(scope="module")
def mfp():
    """ModelicaFluidPackage from InlineTest .so."""
    if not SO_PATH.exists():
        pytest.skip("InlineTest .so not available — run build_codegen.py first")
    from partitioner.codegen.modelica_fluid import ModelicaFluidPackage
    return ModelicaFluidPackage(SO_PATH)


@pytest.fixture(scope="module")
def cpp_fluid():
    """C++ SimpleFluidProperties for comparison."""
    try:
        import opal_two_phase as tp
        return tp.SimpleFluidProperties()
    except ImportError:
        pytest.skip("C++ opal_two_phase not available")


# ============================================================================
# SimpleFluid hand-calculated reference values
# ============================================================================

class SF:
    """SimpleFluid constants for hand calculations."""
    p_ref = 10e6
    rho_f_0, rho_f_1 = 750.0, 20.0
    rho_g_0, rho_g_1 = 40.0, 5.0
    T_sat_0, T_sat_1 = 400.0, 20.0
    h_f_0, h_f_1 = 800e3, 100e3
    h_g_0, h_g_1 = 2800e3, 50e3

    @staticmethod
    def p_hat(p): return (p - SF.p_ref) / SF.p_ref


# ============================================================================
# L0: Function-level verification against hand calculations
# ============================================================================

class TestL0Properties:
    """L0: Every property function at known test points."""

    def test_rho_ph_subcooled(self, mfp):
        """Region 1 (subcooled): rho = rho_f = 750 at p=10MPa, h=800kJ/kg."""
        fp = mfp.evaluate(10e6, 800e3)
        assert fp.rho == pytest.approx(750.0)

    def test_rho_ph_superheated(self, mfp):
        """Region 2 (superheated): rho = rho_g = 40 at p=10MPa, h=2800kJ/kg."""
        fp = mfp.evaluate(10e6, 2800e3)
        assert fp.rho == pytest.approx(40.0)

    def test_rho_ph_two_phase(self, mfp):
        """Region 4 (two-phase): rho interpolated at h=1800kJ/kg (quality=0.5)."""
        fp = mfp.evaluate(10e6, 1800e3)
        # quality = (1800-800)/(2800-800) = 0.5
        # rho_2phase = 1/(x/rho_g + (1-x)/rho_f) = 1/(0.5/40 + 0.5/750)
        x = 0.5
        rho_expected = 1.0 / (x / 40.0 + (1 - x) / 750.0)
        assert fp.rho == pytest.approx(rho_expected, rel=1e-6)

    def test_T_ph_subcooled(self, mfp):
        """T in Region 1: T = T_sat - (h_f - h) / cp_l."""
        fp = mfp.evaluate(10e6, 700e3)
        # T = 400 - (800e3 - 700e3) / 4000 = 400 - 25 = 375
        assert fp.T == pytest.approx(375.0)

    def test_T_ph_saturation(self, mfp):
        """T at saturation: T = T_sat = 400 at p=10MPa."""
        fp = mfp.evaluate(10e6, 800e3)
        assert fp.T == pytest.approx(400.0)

    def test_T_sat(self, mfp):
        """T_sat = 400 + 20*(p-p_ref)/p_ref at p=10MPa → 400."""
        pp = mfp.evaluate_phasic(10e6)
        assert pp.T_sat == pytest.approx(400.0)

    def test_T_sat_offset(self, mfp):
        """T_sat at p=11MPa: 400 + 20*0.1 = 402."""
        pp = mfp.evaluate_phasic(11e6)
        assert pp.T_sat == pytest.approx(402.0)

    def test_h_f(self, mfp):
        """h_f = 800e3 at p=10MPa."""
        pp = mfp.evaluate_phasic(10e6)
        assert pp.h_sat_l == pytest.approx(800e3)

    def test_h_g(self, mfp):
        """h_g = 2800e3 at p=10MPa."""
        pp = mfp.evaluate_phasic(10e6)
        assert pp.h_sat_v == pytest.approx(2800e3)

    def test_rho_f(self, mfp):
        """rho_f = 750 at p=10MPa."""
        pp = mfp.evaluate_phasic(10e6)
        assert pp.rho_l == pytest.approx(750.0)

    def test_rho_g(self, mfp):
        """rho_g = 40 at p=10MPa."""
        pp = mfp.evaluate_phasic(10e6)
        assert pp.rho_g == pytest.approx(40.0)

    def test_drho_dp_h_subcooled(self, mfp, cpp_fluid):
        """drho/dp at const h in subcooled region — compare to C++ reference."""
        fp_om = mfp.evaluate(10e6, 700e3)
        fp_cpp = cpp_fluid.evaluate(10e6, 700e3)
        assert fp_om.drho_dp_h == pytest.approx(fp_cpp.drho_dp_h, rel=1e-12)
        assert fp_om.drho_dp_h > 0, "drho/dp must be positive (compressibility)"

    def test_drho_dh_p_subcooled(self, mfp, cpp_fluid):
        """drho/dh at const p in subcooled region — compare to C++ reference."""
        fp_om = mfp.evaluate(10e6, 700e3)
        fp_cpp = cpp_fluid.evaluate(10e6, 700e3)
        assert fp_om.drho_dh_p == pytest.approx(fp_cpp.drho_dh_p, rel=1e-12)

    def test_drho_dh_p_two_phase(self, mfp):
        """drho/dh in two-phase region is negative (increasing h → more vapor → lighter)."""
        fp = mfp.evaluate(10e6, 1800e3)
        assert fp.drho_dh_p < 0, "drho/dh must be negative in two-phase"


# ============================================================================
# L0: Comparison with C++ FluidPackage (must match to machine precision)
# ============================================================================

class TestL0CppComparison:
    """L0: OM-generated C matches C++ FluidPackage for SimpleFluid."""

    TEST_POINTS = [
        (10e6, 500e3),     # Region 1 (deep subcooled)
        (10e6, 750e3),     # Region 1 (near saturation)
        (10e6, 800e3),     # Region 1/4 boundary
        (10e6, 1200e3),    # Region 4 (two-phase, low quality)
        (10e6, 1800e3),    # Region 4 (two-phase, mid quality)
        (10e6, 2400e3),    # Region 4 (two-phase, high quality)
        (10e6, 2800e3),    # Region 2/4 boundary
        (10e6, 3000e3),    # Region 2 (superheated)
        (8e6, 800e3),      # Lower pressure
        (12e6, 800e3),     # Higher pressure
    ]

    def test_rho_matches_cpp(self, mfp, cpp_fluid):
        """rho_ph matches C++ at all test points."""
        for p, h in self.TEST_POINTS:
            cpp_fp = cpp_fluid.evaluate(p, h)
            om_fp = mfp.evaluate(p, h)
            assert om_fp.rho == pytest.approx(cpp_fp.rho, rel=1e-12), \
                f"rho mismatch at p={p}, h={h}: OM={om_fp.rho}, C++={cpp_fp.rho}"

    def test_drho_dp_matches_cpp(self, mfp, cpp_fluid):
        """drho_dp_h matches C++ at all test points."""
        for p, h in self.TEST_POINTS:
            cpp_fp = cpp_fluid.evaluate(p, h)
            om_fp = mfp.evaluate(p, h)
            assert om_fp.drho_dp_h == pytest.approx(cpp_fp.drho_dp_h, rel=1e-10, abs=1e-15), \
                f"drho_dp mismatch at p={p}, h={h}"

    def test_drho_dh_matches_cpp(self, mfp, cpp_fluid):
        """drho_dh_p matches C++ at all test points."""
        for p, h in self.TEST_POINTS:
            cpp_fp = cpp_fluid.evaluate(p, h)
            om_fp = mfp.evaluate(p, h)
            assert om_fp.drho_dh_p == pytest.approx(cpp_fp.drho_dh_p, rel=1e-10, abs=1e-15), \
                f"drho_dh mismatch at p={p}, h={h}"

    def test_T_matches_cpp(self, mfp, cpp_fluid):
        """T_ph matches C++ at all test points."""
        for p, h in self.TEST_POINTS:
            cpp_fp = cpp_fluid.evaluate(p, h)
            om_fp = mfp.evaluate(p, h)
            assert om_fp.T == pytest.approx(cpp_fp.T, rel=1e-12), \
                f"T mismatch at p={p}, h={h}"

    def test_phasic_matches_cpp(self, mfp, cpp_fluid):
        """Saturation properties match C++ at several pressures."""
        for p in [8e6, 10e6, 12e6, 15e6]:
            cpp_pp = cpp_fluid.evaluate_phasic(p)
            om_pp = mfp.evaluate_phasic(p)
            assert om_pp.T_sat == pytest.approx(cpp_pp.T_sat, rel=1e-12), f"T_sat at p={p}"
            assert om_pp.h_sat_l == pytest.approx(cpp_pp.h_sat_l, rel=1e-12), f"h_f at p={p}"
            assert om_pp.h_sat_v == pytest.approx(cpp_pp.h_sat_v, rel=1e-12), f"h_g at p={p}"
            assert om_pp.rho_l == pytest.approx(cpp_pp.rho_l, rel=1e-12), f"rho_f at p={p}"
            # C++ uses rho_v, ModelicaFluidPackage uses rho_g (Modelica naming)
            assert om_pp.rho_g == pytest.approx(cpp_pp.rho_v, rel=1e-12), f"rho_g at p={p}"


# ============================================================================
# L1: Interface compatibility with Parameterized5EqSolver
# ============================================================================

class TestL1Interface:
    """L1: ModelicaFluidPackage plugs into the solver."""

    def test_evaluate_returns_correct_attrs(self, mfp):
        """evaluate() result has .rho, .drho_dp_h, .drho_dh_p, .T."""
        fp = mfp.evaluate(10e6, 800e3)
        assert hasattr(fp, 'rho')
        assert hasattr(fp, 'drho_dp_h')
        assert hasattr(fp, 'drho_dh_p')
        assert hasattr(fp, 'T')

    def test_evaluate_phasic_returns_correct_attrs(self, mfp):
        """evaluate_phasic() result has all required attributes including aliases."""
        pp = mfp.evaluate_phasic(10e6)
        for attr in ['T_sat', 'h_sat_l', 'h_sat_v', 'rho_l', 'rho_g',
                     'rho_v', 'sigma', 'cp_l', 'cp_v', 'drho_l_dp', 'drho_v_dp']:
            assert hasattr(pp, attr), f"PhasicProperties missing attribute: {attr}"

    def test_has_pressure_bounds(self, mfp):
        """ModelicaFluidPackage has p_min and p_max."""
        assert hasattr(mfp, 'p_min')
        assert hasattr(mfp, 'p_max')
        assert mfp.p_min > 0
        assert mfp.p_max > mfp.p_min

    def test_solver_construction(self, mfp):
        """Parameterized5EqSolver accepts ModelicaFluidPackage as fluid."""
        from partitioner.model_spec import (
            ExtractedModelSpec, GeometrySpec, ClosureSpec,
            BoundarySpec, InitialConditions, SafetyThresholds,
        )
        spec = ExtractedModelSpec(
            prefix="pipe",
            model_type="drift_flux",
            geometry=GeometrySpec(N=3, dx=1.0, A_flow=0.01, D_h=0.1,
                                  f_D=0.02, V_cell=0.01, g_axial=0.0),
            closures=ClosureSpec(H_i=1e5, C_0=1.13, alpha_nucleation=1e-3,
                                 use_critical_flow=False, C_d=1.0, x_trans=0.1,
                                 c_floor=1200.0, use_two_phase_friction=False,
                                 Phi2_max=20.0),
            boundary=BoundarySpec(inlet_type="wall", outlet_type="pressure",
                                  p_out=101325.0, h_out=800e3),
            ic=InitialConditions(p0=[10e6]*3, h0=[800e3]*3,
                                 h_v0=[2800e3]*3, alpha0=[1e-6]*3,
                                 mdot0=[0.0]*4),
            thresholds=SafetyThresholds(),
        )
        from partitioner.parameterized_5eq_solver import Parameterized5EqSolver
        solver = Parameterized5EqSolver(mfp, spec)
        assert solver is not None


# ============================================================================
# L2: Solver integration — single timestep comparison
# ============================================================================

class TestL2SolverStep:
    """L2: One timestep with ModelicaFluidPackage matches C++ FluidPackage."""

    def test_single_step_matches(self, mfp, cpp_fluid):
        """Run one 5-eq step with both fluids, compare states."""
        import numpy as np
        from partitioner.model_spec import (
            ExtractedModelSpec, GeometrySpec, ClosureSpec,
            BoundarySpec, InitialConditions, SafetyThresholds,
        )
        from partitioner.parameterized_5eq_solver import Parameterized5EqSolver

        spec = ExtractedModelSpec(
            prefix="pipe",
            model_type="drift_flux",
            geometry=GeometrySpec(N=3, dx=1.0, A_flow=0.004, D_h=0.073,
                                  f_D=0.02, V_cell=0.004, g_axial=0.0),
            closures=ClosureSpec(H_i=1e5, C_0=1.13, alpha_nucleation=1e-3,
                                 use_critical_flow=False, C_d=1.0, x_trans=0.1,
                                 c_floor=1200.0, use_two_phase_friction=False,
                                 Phi2_max=20.0),
            boundary=BoundarySpec(inlet_type="wall", outlet_type="pressure",
                                  p_out=101325.0, h_out=800e3),
            ic=InitialConditions(p0=[10e6]*3, h0=[800e3]*3,
                                 h_v0=[2800e3]*3, alpha0=[1e-6]*3,
                                 mdot0=[0.0]*4),
            thresholds=SafetyThresholds(),
        )

        # Case 1: C++ fluid
        solver1 = Parameterized5EqSolver(cpp_fluid, spec)
        p1 = np.array([10e6, 10e6, 10e6])
        a1 = np.array([1e-6, 1e-6, 1e-6])
        hl1 = np.array([800e3, 800e3, 800e3])
        hv1 = np.array([2800e3, 2800e3, 2800e3])
        m1 = np.array([0.0, 0.0, 0.0, 0.0])
        solver1.step(p1, a1, hl1, hv1, m1, 1e-4)

        # Case 2: Modelica fluid
        solver2 = Parameterized5EqSolver(mfp, spec)
        p2 = np.array([10e6, 10e6, 10e6])
        a2 = np.array([1e-6, 1e-6, 1e-6])
        hl2 = np.array([800e3, 800e3, 800e3])
        hv2 = np.array([2800e3, 2800e3, 2800e3])
        m2 = np.array([0.0, 0.0, 0.0, 0.0])
        solver2.step(p2, a2, hl2, hv2, m2, 1e-4)

        # Compare — should be identical (same math from same Modelica source)
        np.testing.assert_allclose(p1, p2, rtol=1e-12,
                                   err_msg="Pressure diverged between Case 1 and Case 2")
        np.testing.assert_allclose(a1, a2, rtol=1e-10, atol=1e-15,
                                   err_msg="Alpha diverged")
        np.testing.assert_allclose(hl1, hl2, rtol=1e-12,
                                   err_msg="h_l diverged")
        np.testing.assert_allclose(hv1, hv2, rtol=1e-12,
                                   err_msg="h_v diverged")
        np.testing.assert_allclose(m1, m2, rtol=1e-12, atol=1e-15,
                                   err_msg="mdot diverged")


# ============================================================================
# Build verification
# ============================================================================

class TestBuild:
    """Verify the .so was built correctly."""

    def test_so_exists(self):
        """The compiled .so exists."""
        assert SO_PATH.exists(), f"Expected .so at {SO_PATH}"

    def test_so_has_expected_symbols(self):
        """The .so exports all expected opal_* symbols."""
        if not SO_PATH.exists():
            pytest.skip(".so not available")
        import subprocess
        result = subprocess.run(['nm', '-gU', str(SO_PATH)],
                                capture_output=True, text=True)
        symbols = result.stdout
        expected = ['opal_rho_ph', 'opal_T_ph', 'opal_T_sat',
                    'opal_drho_dp_h', 'opal_drho_dh_p',
                    'opal_h_f', 'opal_h_g', 'opal_rho_f', 'opal_rho_g']
        for sym in expected:
            assert sym in symbols or f'_{sym}' in symbols, \
                f"Missing symbol: {sym}"


# ============================================================================
# QA Round 1 gap closure: rho_v alias, drho_l_dp, sigma, diagnostics,
# off-reference derivatives, region boundaries, multi-step
# ============================================================================

class TestRhoVAlias:
    """FINDING 1: rho_v alias on PhasicProperties."""

    def test_rho_v_equals_rho_g(self, mfp):
        """pp.rho_v must exist and equal pp.rho_g."""
        pp = mfp.evaluate_phasic(10e6)
        assert hasattr(pp, 'rho_v'), "PhasicProperties must have rho_v alias"
        assert pp.rho_v == pp.rho_g

    def test_rho_v_at_off_reference(self, mfp):
        """rho_v alias works at non-reference pressure."""
        pp = mfp.evaluate_phasic(12e6)
        assert pp.rho_v == pytest.approx(pp.rho_g)
        assert pp.rho_v > 0


class TestPhasicDerivatives:
    """FINDING 2: drho_l_dp and drho_v_dp via finite difference."""

    def test_drho_l_dp_nonzero(self, mfp):
        """drho_f/dp must be nonzero for SimpleFluid (rho_f has pressure dependence)."""
        pp = mfp.evaluate_phasic(10e6)
        # SimpleFluid: rho_f = 750 + 20*(p-p_ref)/p_ref → drho_f/dp = 20/p_ref = 2e-6
        assert pp.drho_l_dp == pytest.approx(20.0 / 10e6, rel=0.01)

    def test_drho_v_dp_nonzero(self, mfp):
        """drho_g/dp must be nonzero for SimpleFluid."""
        pp = mfp.evaluate_phasic(10e6)
        # SimpleFluid: rho_g = 40 + 5*(p-p_ref)/p_ref → drho_g/dp = 5/p_ref = 5e-7
        assert pp.drho_v_dp == pytest.approx(5.0 / 10e6, rel=0.01)

    def test_drho_l_dp_at_off_reference(self, mfp):
        """Derivative should be similar at different pressures (linear model)."""
        pp8 = mfp.evaluate_phasic(8e6)
        pp12 = mfp.evaluate_phasic(12e6)
        assert pp8.drho_l_dp == pytest.approx(pp12.drho_l_dp, rel=0.01)


class TestSigmaFallback:
    """FINDING 3: sigma computed from SimpleFluid formula when not exported."""

    def test_sigma_at_reference(self, mfp):
        """sigma(10MPa) = 0.06 (SimpleFluid reference)."""
        pp = mfp.evaluate_phasic(10e6)
        assert pp.sigma == pytest.approx(0.06, abs=1e-10)

    def test_sigma_varies_with_pressure(self, mfp):
        """sigma must change with pressure (not a constant 0.06 everywhere)."""
        pp8 = mfp.evaluate_phasic(8e6)
        pp12 = mfp.evaluate_phasic(12e6)
        # SimpleFluid: sigma = 0.06 - 0.04*(p-p_ref)/p_ref
        # At 8MPa: p_hat = -0.2, sigma = 0.06 + 0.008 = 0.068
        # At 12MPa: p_hat = 0.2, sigma = 0.06 - 0.008 = 0.052
        assert pp8.sigma == pytest.approx(0.068, abs=1e-6)
        assert pp12.sigma == pytest.approx(0.052, abs=1e-6)
        assert pp8.sigma != pp12.sigma


class TestDiagnosticFunctions:
    """FINDING 5: quality_ph and region_ph exposed and working."""

    def test_quality_subcooled(self, mfp):
        """quality = 0 in subcooled region."""
        assert mfp.quality_ph(10e6, 700e3) == pytest.approx(0.0)

    def test_quality_superheated(self, mfp):
        """quality = 1 in superheated region."""
        assert mfp.quality_ph(10e6, 3000e3) == pytest.approx(1.0)

    def test_quality_midpoint(self, mfp):
        """quality = 0.5 at midpoint enthalpy."""
        assert mfp.quality_ph(10e6, 1800e3) == pytest.approx(0.5)

    def test_region_subcooled(self, mfp):
        """region = 1 (subcooled)."""
        assert mfp.region_ph(10e6, 700e3) == 1

    def test_region_two_phase(self, mfp):
        """region = 4 (two-phase)."""
        assert mfp.region_ph(10e6, 1800e3) == 4

    def test_region_superheated(self, mfp):
        """region = 2 (superheated)."""
        assert mfp.region_ph(10e6, 3000e3) == 2


class TestOffReferenceDerivatives:
    """MISSING: Derivatives at off-reference pressures in two-phase region."""

    def test_drho_dp_two_phase_at_8MPa(self, mfp, cpp_fluid):
        """drho_dp in two-phase at p=8MPa must match C++."""
        fp_om = mfp.evaluate(8e6, 1800e3)
        fp_cpp = cpp_fluid.evaluate(8e6, 1800e3)
        assert fp_om.drho_dp_h == pytest.approx(fp_cpp.drho_dp_h, rel=1e-10)

    def test_drho_dh_two_phase_at_12MPa(self, mfp, cpp_fluid):
        """drho_dh in two-phase at p=12MPa must match C++."""
        fp_om = mfp.evaluate(12e6, 1500e3)
        fp_cpp = cpp_fluid.evaluate(12e6, 1500e3)
        assert fp_om.drho_dh_p == pytest.approx(fp_cpp.drho_dh_p, rel=1e-10)

    def test_drho_dp_two_phase_positive(self, mfp):
        """drho/dp must be positive in two-phase (compressibility)."""
        fp = mfp.evaluate(8e6, 1800e3)
        assert fp.drho_dp_h > 0


class TestRegionBoundaries:
    """MISSING: Behavior exactly at h_f and h_g boundaries."""

    def test_at_h_f_boundary(self, mfp, cpp_fluid):
        """rho at h exactly equal to h_f matches C++."""
        h_f = mfp.evaluate_phasic(10e6).h_sat_l  # 800e3
        fp_om = mfp.evaluate(10e6, h_f)
        fp_cpp = cpp_fluid.evaluate(10e6, h_f)
        assert fp_om.rho == pytest.approx(fp_cpp.rho, rel=1e-12)

    def test_at_h_g_boundary(self, mfp, cpp_fluid):
        """rho at h exactly equal to h_g matches C++."""
        h_g = mfp.evaluate_phasic(10e6).h_sat_v  # 2800e3
        fp_om = mfp.evaluate(10e6, h_g)
        fp_cpp = cpp_fluid.evaluate(10e6, h_g)
        assert fp_om.rho == pytest.approx(fp_cpp.rho, rel=1e-12)


class TestMultiStep:
    """MISSING: Multi-step L2 test to catch cumulative drift."""

    def test_100_steps_match(self, mfp, cpp_fluid):
        """100 timesteps with both fluids — no cumulative drift."""
        import numpy as np
        from partitioner.model_spec import (
            ExtractedModelSpec, GeometrySpec, ClosureSpec,
            BoundarySpec, InitialConditions, SafetyThresholds,
        )
        from partitioner.parameterized_5eq_solver import Parameterized5EqSolver

        spec = ExtractedModelSpec(
            prefix="pipe",
            model_type="drift_flux",
            geometry=GeometrySpec(N=3, dx=1.0, A_flow=0.004, D_h=0.073,
                                  f_D=0.02, V_cell=0.004, g_axial=0.0),
            closures=ClosureSpec(H_i=1e5, C_0=1.13, alpha_nucleation=1e-3,
                                 use_critical_flow=False, C_d=1.0, x_trans=0.1,
                                 c_floor=1200.0, use_two_phase_friction=False,
                                 Phi2_max=20.0),
            boundary=BoundarySpec(inlet_type="wall", outlet_type="pressure",
                                  p_out=101325.0, h_out=800e3),
            ic=InitialConditions(p0=[10e6]*3, h0=[800e3]*3,
                                 h_v0=[2800e3]*3, alpha0=[1e-6]*3,
                                 mdot0=[0.0]*4),
            thresholds=SafetyThresholds(),
        )

        solver1 = Parameterized5EqSolver(cpp_fluid, spec)
        p1 = np.array([10e6, 10e6, 10e6])
        a1 = np.array([1e-6, 1e-6, 1e-6])
        hl1 = np.array([800e3, 800e3, 800e3])
        hv1 = np.array([2800e3, 2800e3, 2800e3])
        m1 = np.array([0.0, 0.0, 0.0, 0.0])

        solver2 = Parameterized5EqSolver(mfp, spec)
        p2 = np.array([10e6, 10e6, 10e6])
        a2 = np.array([1e-6, 1e-6, 1e-6])
        hl2 = np.array([800e3, 800e3, 800e3])
        hv2 = np.array([2800e3, 2800e3, 2800e3])
        m2 = np.array([0.0, 0.0, 0.0, 0.0])

        dt = 1e-4
        for _ in range(100):
            solver1.step(p1, a1, hl1, hv1, m1, dt)
            solver2.step(p2, a2, hl2, hv2, m2, dt)

        np.testing.assert_allclose(p1, p2, rtol=1e-10,
                                   err_msg="Pressure diverged after 100 steps")
        np.testing.assert_allclose(hl1, hl2, rtol=1e-10,
                                   err_msg="h_l diverged after 100 steps")
        np.testing.assert_allclose(m1, m2, rtol=1e-10, atol=1e-15,
                                   err_msg="mdot diverged after 100 steps")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
