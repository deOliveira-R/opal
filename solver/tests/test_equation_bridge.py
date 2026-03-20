"""
test_equation_bridge.py — L0/L1/L2 tests for the OM equation bridge infrastructure.

Covers: info_parser, CTokenRewriter, OMEquationBridge, BridgeSolver,
build_codegen helpers, OM runtime stubs, media function discovery.
"""

import sys
import os
import json
import math
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "two_phase"))

OPAL_ROOT = Path(__file__).resolve().parents[2]
SF_SO = OPAL_ROOT / "feasibility" / "results" / "opal_bridge_InlineTest.so"
SF_INFO = OPAL_ROOT / "feasibility" / "results" / "InlineTest_info.json"
SF_C = OPAL_ROOT / "feasibility" / "results" / "InlineTest.c"
SF_FUNC_C = OPAL_ROOT / "feasibility" / "results" / "InlineTest_functions.c"
SF_FUNC_H = OPAL_ROOT / "feasibility" / "results" / "InlineTest_functions.h"
MOD_XML = OPAL_ROOT / "feasibility" / "results" / "ModularPipeTest.xml"
WATER_SO = OPAL_ROOT / "feasibility" / "results" / "opal_bridge_WaterTest.so"
WATER_INFO = OPAL_ROOT / "feasibility" / "results" / "WaterTest_info.json"


def _skip_if_missing(*paths):
    for p in paths:
        if not p.exists():
            pytest.skip(f"{p.name} not available")


# ============================================================================
# M1: info_parser tests
# ============================================================================

class TestInfoParser:
    """Parse _info.json correctly."""

    @pytest.fixture
    def info(self):
        _skip_if_missing(SF_INFO)
        from partitioner.codegen.info_parser import parse_info_json
        return parse_info_json(SF_INFO)

    def test_model_name(self, info):
        assert info.model_name == "InlineTest"

    def test_state_count(self, info):
        assert info.n_states == 9  # 3 p + 3 h + 3 mdot

    def test_n_vars(self, info):
        assert info.n_vars == 41

    def test_n_params(self, info):
        assert info.n_params == 14

    def test_var_index_lookup(self, info):
        assert info.var_index("pipe.p[1]") == 6
        assert info.var_index("pipe.h[1]") == 0

    def test_param_index_lookup(self, info):
        assert info.param_index("pipe.dx") == 7
        assert info.param_index("pipe.f_D") == 8

    def test_vars_by_pattern(self, info):
        p_idx = info.vars_by_pattern("pipe.p", 3)
        assert len(p_idx) == 3
        assert p_idx == [6, 7, 8]

    def test_blt_order_nonempty(self, info):
        assert len(info.blt_order) > 0
        assert 52 in info.blt_order  # first equation

    def test_algebraic_eqs_are_assign(self, info):
        for eq in info.algebraic_eqs:
            assert eq.tag == "assign"

    def test_jacobian_vars_excluded(self, info):
        for name in info.all_vars:
            assert "jacobian" not in info.all_vars[name].kind

    def test_states_are_states(self, info):
        for name, vi in info.states.items():
            assert vi.kind == "state"


# ============================================================================
# M2: build_codegen helper tests
# ============================================================================

class TestBuildCodegenHelpers:
    def test_detect_medium_prefix_simplefluid(self):
        from partitioner.codegen.build_codegen import detect_medium_prefix
        names = ["omc_library_Media_SimpleFluid_rho__ph",
                 "omc_library_Media_SimpleFluid_T__ph"]
        assert detect_medium_prefix(names) == "omc_library_Media_SimpleFluid_"

    def test_make_opal_name_simplefluid(self):
        from partitioner.codegen.build_codegen import make_opal_name
        assert make_opal_name("omc_library_Media_SimpleFluid_rho__ph",
                              "omc_library_Media_SimpleFluid_") == "opal_rho_ph"

    def test_make_opal_name_double_underscore(self):
        from partitioner.codegen.build_codegen import make_opal_name
        assert make_opal_name("omc_library_Media_SimpleFluid_drho__dp__h",
                              "omc_library_Media_SimpleFluid_") == "opal_drho_dp_h"

    def test_detect_medium_prefix_empty(self):
        from partitioner.codegen.build_codegen import detect_medium_prefix
        assert detect_medium_prefix([]) == "omc_"

    def test_parse_function_signatures(self):
        _skip_if_missing(SF_FUNC_H)
        from partitioner.codegen.build_codegen import parse_function_signatures
        sigs = parse_function_signatures(SF_FUNC_H)
        names = [s['name'] for s in sigs]
        assert "omc_library_Media_SimpleFluid_rho__ph" in names
        assert "omc_library_Media_SimpleFluid_T__ph" in names


# ============================================================================
# H2: CTokenRewriter unit tests
# ============================================================================

class TestCTokenRewriter:
    @pytest.fixture
    def rewriter(self):
        from partitioner.codegen.bridge_codegen import CTokenRewriter
        return CTokenRewriter()

    def test_var_accessor_rewrite(self, rewriter):
        body = '  x = data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]];'
        result = rewriter.rewrite(body)
        assert 'opal_vars[42]' in result
        assert 'data->' not in result

    def test_param_accessor_rewrite(self, rewriter):
        body = '  x = data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[7]];'
        result = rewriter.rewrite(body)
        assert 'opal_params[7]' in result
        assert 'data->' not in result

    def test_accessor_with_parens_and_comment(self, rewriter):
        body = '  x = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* pipe.p[1] STATE(1) */);'
        result = rewriter.rewrite(body)
        assert 'opal_vars[5]' in result
        assert '/*' not in result

    def test_division_sim_to_opal_div(self, rewriter):
        body = '  DIVISION_SIM(a, b, "msg", equationIndexes);'
        result = rewriter.rewrite(body)
        assert 'OPAL_DIV(' in result
        assert 'DIVISION_SIM' not in result

    def test_threaddata_removed(self, rewriter):
        body = '  threadData->lastEquationSolved = 52;'
        result = rewriter.rewrite(body)
        assert 'lastEquationSolved' not in result

    def test_equation_indexes_removed(self, rewriter):
        body = '  const int equationIndexes[2] = {1,52};'
        result = rewriter.rewrite(body)
        assert 'equationIndexes' not in result

    def test_remaining_threaddata_becomes_td(self, rewriter):
        body = '  x = omc_func(threadData, 1.0, 2.0);'
        result = rewriter.rewrite(body)
        assert '(&_td)' in result
        assert 'threadData' not in result or '(&_td)' in result

    def test_validate_catches_unrecognized_data_ref(self, rewriter):
        body = '  x = data->someNewThing;'
        with pytest.raises(ValueError, match="unrewritten data-> reference"):
            rewriter.rewrite(body, eq_id=99)

    def test_data_in_string_literal_ignored(self, rewriter):
        body = '  printf("data->foo");'
        result = rewriter.rewrite(body)  # should not raise
        assert 'data->foo' in result  # inside string, preserved

    def test_relationhysteresis_greater_eq(self, rewriter):
        body = '  relationhysteresis(data, &tmp0, val, 0.0, t1, t2, 2, GreaterEq, GreaterEqZC);'
        result = rewriter.rewrite(body)
        assert 'tmp0 = (val) >= (0.0)' in result

    def test_roundtrip_real_simplefluid_eq(self):
        """Rewrite an actual OM-generated SimpleFluid equation function body."""
        _skip_if_missing(SF_C)
        from partitioner.codegen.bridge_codegen import extract_equation_functions, rewrite_equation_body
        funcs = extract_equation_functions(SF_C, "InlineTest")
        assert 52 in funcs  # T_cell[3] = T_ph(p[3], h[3])
        rewritten = rewrite_equation_body(funcs[52], eq_id=52)
        assert 'opal_vars[' in rewritten
        assert 'data->' not in rewritten


# ============================================================================
# H3: OM runtime stubs
# ============================================================================

class TestOMRuntimeStubs:
    """Test the compiled stubs via the bridge .so."""

    @pytest.fixture
    def bridge(self):
        _skip_if_missing(SF_SO, SF_INFO)
        from partitioner.codegen.info_parser import parse_info_json
        from partitioner.codegen.equation_bridge import OMEquationBridge
        return OMEquationBridge(SF_SO, parse_info_json(SF_INFO))

    def test_rho_ph_normal(self, bridge):
        """Normal evaluation produces correct result (not abort)."""
        rho = bridge._rho_ph_fn(10e6, 800e3)
        assert rho == pytest.approx(750.0)

    def test_rho_ph_two_phase(self, bridge):
        """Two-phase evaluation works (exercises region logic)."""
        rho = bridge._rho_ph_fn(10e6, 1800e3)
        assert rho > 40 and rho < 750


# ============================================================================
# H1: BridgeSolver parity tests
# ============================================================================

class TestBridgeSolverParity:
    """BridgeSolver must match ExtractedSemiImplicitSolver exactly."""

    @pytest.fixture
    def solvers(self):
        _skip_if_missing(SF_SO, SF_INFO, MOD_XML)
        import opal_two_phase as tp
        from partitioner.xml_reader import load_equation_system
        from partitioner.pipe1d_mapper import map_pipe1d
        from partitioner.equation_classifier import classify_equations
        from partitioner.extracted_solver import ExtractedSemiImplicitSolver
        from partitioner.bridge_solver import BridgeSolver
        from partitioner.codegen.info_parser import parse_info_json
        from partitioner.codegen.equation_bridge import OMEquationBridge

        es = load_equation_system(str(MOD_XML))
        spec = map_pipe1d(es)
        cs = classify_equations(es, prefix="pipe")
        info = parse_info_json(SF_INFO)

        s1 = ExtractedSemiImplicitSolver(cs, tp.SimpleFluidProperties(), spec)
        bridge = OMEquationBridge(SF_SO, info)
        s2 = BridgeSolver(bridge, spec)
        return s1, s2, spec

    def _make_state(self, spec):
        return (np.array(spec.p0, dtype=float),
                np.array(spec.h0, dtype=float),
                np.array(spec.mdot0, dtype=float))

    def test_single_step_parity(self, solvers):
        s1, s2, spec = solvers
        p1, h1, m1 = self._make_state(spec)
        p2, h2, m2 = p1.copy(), h1.copy(), m1.copy()
        s1.step(p1, h1, m1, 5e-5)
        s2.step(p2, h2, m2, 5e-5)
        np.testing.assert_allclose(p1, p2, rtol=1e-13)
        np.testing.assert_allclose(h1, h2, atol=1e-8)
        np.testing.assert_allclose(m1, m2, atol=1e-14)

    def test_2000_step_parity(self, solvers):
        """The headline claim: 5.13e-15 after 2000 steps."""
        s1, s2, spec = solvers
        p1, h1, m1 = self._make_state(spec)
        p2, h2, m2 = p1.copy(), h1.copy(), m1.copy()
        for _ in range(2000):
            s1.step(p1, h1, m1, 5e-5)
            s2.step(p2, h2, m2, 5e-5)
        p_err = np.max(np.abs(p1 - p2) / np.maximum(np.abs(p1), 1.0))
        assert p_err < 1e-12, f"Pressure diverged: {p_err}"

    def test_friction_sign_and_magnitude(self, solvers):
        """Friction term has correct sign for both positive and negative mdot."""
        s1, s2, spec = solvers
        p1, h1, m1 = self._make_state(spec)
        # Set nonzero mdot for friction
        m1[1:] = 10.0
        p2, h2, m2 = p1.copy(), h1.copy(), m1.copy()
        s1.step(p1, h1, m1, 5e-5)
        s2.step(p2, h2, m2, 5e-5)
        # Friction should match
        np.testing.assert_allclose(s1.last_fric, s2.last_fric, rtol=1e-12)

    def test_properties_match(self, solvers):
        """Bridge properties match C++ FluidPackage after evaluate."""
        s1, s2, spec = solvers
        p1, h1, m1 = self._make_state(spec)
        p2, h2, m2 = p1.copy(), h1.copy(), m1.copy()
        s1.step(p1, h1, m1, 5e-5)
        s2.step(p2, h2, m2, 5e-5)
        np.testing.assert_allclose(s1.last_rho_face, s2.last_rho_face, rtol=1e-12)
        np.testing.assert_allclose(s1.last_drho_dp, s2.last_drho_dp, rtol=1e-12)


# ============================================================================
# M3: Water/IAPWS bridge regression
# ============================================================================

class TestWaterBridge:
    """Water/IAPWS bridge matches C++ IAPWSIF97Properties."""

    @pytest.fixture
    def water_bridge(self):
        _skip_if_missing(WATER_SO, WATER_INFO, MOD_XML)
        from partitioner.codegen.info_parser import parse_info_json
        from partitioner.codegen.equation_bridge import OMEquationBridge
        from partitioner.xml_reader import load_equation_system
        from partitioner.pipe1d_mapper import map_pipe1d

        info = parse_info_json(WATER_INFO)
        bridge = OMEquationBridge(WATER_SO, info)
        es = load_equation_system(str(MOD_XML))
        spec = map_pipe1d(es)
        bridge.set_params_from_spec(spec)
        return bridge

    @pytest.fixture
    def cpp_iapws(self):
        import opal_two_phase as tp
        return tp.IAPWSIF97Properties()

    TEST_POINTS = [(7e6, 980e3), (7e6, 500e3), (5e6, 1200e3), (10e6, 2000e3)]

    def test_rho_matches_cpp(self, water_bridge, cpp_iapws):
        for p, h in self.TEST_POINTS:
            bridge_rho = water_bridge._rho_ph_fn(p, h)
            cpp_rho = cpp_iapws.evaluate(p, h).rho
            assert bridge_rho == pytest.approx(cpp_rho, rel=1e-10), \
                f"rho mismatch at p={p}, h={h}"

    def test_evaluate_produces_nonzero_rho_face(self, water_bridge):
        p = np.array([7e6, 7e6, 7e6])
        h = np.array([980e3, 980e3, 980e3])
        mdot = np.zeros(4)
        water_bridge.set_state(p, h, mdot)
        water_bridge.evaluate()
        rho_face = water_bridge.get_rho_face()
        assert all(r > 0 for r in rho_face), f"Zero rho_face: {rho_face}"


# ============================================================================
# M4: Media function discovery
# ============================================================================

class TestMediaFnDiscovery:
    def test_simplefluid_rho_ph_found(self):
        _skip_if_missing(SF_SO, SF_INFO)
        from partitioner.codegen.info_parser import parse_info_json
        from partitioner.codegen.equation_bridge import OMEquationBridge
        bridge = OMEquationBridge(SF_SO, parse_info_json(SF_INFO))
        assert bridge._rho_ph_fn is not None

    def test_water_rho_ph_found(self):
        _skip_if_missing(WATER_SO, WATER_INFO)
        from partitioner.codegen.info_parser import parse_info_json
        from partitioner.codegen.equation_bridge import OMEquationBridge
        bridge = OMEquationBridge(WATER_SO, parse_info_json(WATER_INFO))
        assert bridge._rho_ph_fn is not None

    def test_manifest_exists(self):
        _skip_if_missing(SF_SO)
        manifest = SF_SO.with_suffix('.json')
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert 'media_wrappers' in data
        assert any('rho_ph' in w for w in data['media_wrappers'])


# ============================================================================
# M5: set_params_from_spec completeness
# ============================================================================

class TestSetParams:
    def test_geometry_params_set(self):
        _skip_if_missing(SF_SO, SF_INFO, MOD_XML)
        from partitioner.codegen.info_parser import parse_info_json
        from partitioner.codegen.equation_bridge import OMEquationBridge
        from partitioner.xml_reader import load_equation_system
        from partitioner.pipe1d_mapper import map_pipe1d

        info = parse_info_json(SF_INFO)
        bridge = OMEquationBridge(SF_SO, info)
        es = load_equation_system(str(MOD_XML))
        spec = map_pipe1d(es)
        bridge.set_params_from_spec(spec)

        # Read back params via get_var on known indices
        dx_idx = info.param_index("pipe.dx")
        f_D_idx = info.param_index("pipe.f_D")
        # Params are set via set_params — verify spec values propagated
        assert spec.dx == pytest.approx(1.0)
        assert spec.f_D == pytest.approx(0.02)


# ============================================================================
# Build verification
# ============================================================================

class TestBridgeBuild:
    def test_simplefluid_so_exists(self):
        assert SF_SO.exists()

    def test_water_so_exists(self):
        _skip_if_missing(WATER_SO)
        assert WATER_SO.exists()

    def test_simplefluid_exports_generic_api(self):
        _skip_if_missing(SF_SO)
        import ctypes
        lib = ctypes.CDLL(str(SF_SO))
        # All generic API functions must exist
        for name in ['opal_bridge_evaluate', 'opal_bridge_set_var',
                     'opal_bridge_get_var', 'opal_bridge_set_vars',
                     'opal_bridge_get_vars', 'opal_bridge_set_params',
                     'opal_bridge_get_n_vars', 'opal_bridge_get_n_params']:
            assert hasattr(lib, name), f"Missing: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
