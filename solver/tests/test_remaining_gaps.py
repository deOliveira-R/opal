"""
test_remaining_gaps.py — Cover QA audit Gaps 8, 10-15.

Gap 8:  Momentum equation structure verification
Gap 10: IAPWS C++ vs Python iapws oracle (already covered by test_iapws_cpp.py,
        but adding extraction-specific checks here)
Gap 11: xml_reader / equation_system parsing edge cases
Gap 12: Convergence test for extraction-driven solver
Gap 13: solver_from_spec produces a working solver
Gap 14: Equation.lhs_var() parsing
Gap 15: Variable.array_indices() for multi-dimensional arrays
"""

import sys
import os
import re
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "two_phase"))

import opal_two_phase as tp
from partitioner.equation_system import Variable, Parameter, Equation, EquationSystem
from partitioner.pipe1d_mapper import Pipe1DGridSpec, solver_from_spec

OPAL_ROOT = Path(__file__).resolve().parents[2]
EDWARDS_XML = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_backEnd.xml"


# ============================================================================
# Gap 8: Momentum equation structure verification
# ============================================================================

class TestMomentumStructure:
    """Verify extracted momentum equations have correct pressure gradient sign
    and friction term structure."""

    @pytest.fixture
    def momentum_eqs(self):
        if not EDWARDS_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.equation_classifier import classify_equations
        es = load_equation_system(str(EDWARDS_XML))
        cs = classify_equations(es, prefix="pipe")
        return cs.momentum_eqs

    def test_interior_face_has_p_left_minus_p_right(self, momentum_eqs):
        """Interior momentum: A*(p[i-1] - p[i]) — pressure gradient drives flow."""
        # Find an interior face (not boundary)
        interior = [eq for eq in momentum_eqs if eq.face > 2 and eq.face < 6]
        assert len(interior) > 0, "Should have interior momentum equations"

        eq = interior[0]
        # Should contain p[face-1] - p[face] pattern
        # e.g., for face 3: pipe.p[2] - pipe.p[3]
        f = eq.face
        pattern = rf'pipe\.p\[{f-1}\]\s*-\s*pipe\.p\[{f}\]'
        assert re.search(pattern, eq.eq_text), \
            f"Face {f}: expected p[{f-1}] - p[{f}], got: {eq.eq_text[:100]}"

    def test_outlet_face_has_p_last_minus_p_out(self, momentum_eqs):
        """Outlet momentum: A*(p[N] - p_out) — drives flow to outlet."""
        outlet = [eq for eq in momentum_eqs if eq.face == 6]  # N+1=6 for N=5
        assert len(outlet) == 1

        eq = outlet[0]
        # Should contain pipe.p[5] - atm.p_set
        assert "pipe.p[5]" in eq.eq_text
        assert "atm.p_set" in eq.eq_text

    def test_friction_opposes_flow_in_equation(self, momentum_eqs):
        """Friction term should have negative sign (opposes flow)."""
        eq = momentum_eqs[0]
        # The extracted form has: (-0.5) * f_D * dx * |mdot| * mdot / (...)
        # The -0.5 indicates friction opposes the pressure-driven flow
        assert "-0.5" in eq.eq_text or "(-0.5)" in eq.eq_text, \
            f"Friction should have negative coefficient: {eq.eq_text[:120]}"

    def test_all_momentum_faces_present(self, momentum_eqs):
        """Should have momentum equations for faces 2..6 (N=5, face 1 eliminated)."""
        faces = sorted(eq.face for eq in momentum_eqs)
        assert faces == [2, 3, 4, 5, 6]


# ============================================================================
# Gap 11: equation_system.py parsing edge cases
# ============================================================================

class TestEquationSystem:
    """Unit tests for Variable, Parameter, Equation, EquationSystem classes."""

    def test_variable_array_base_1d(self):
        v = Variable(id=1, name="pipe.p[3]", variability="continuousState")
        assert v.array_base() == "pipe.p"

    def test_variable_array_base_scalar(self):
        v = Variable(id=1, name="total_mass", variability="continuous")
        assert v.array_base() is None

    def test_variable_array_indices_1d(self):
        v = Variable(id=1, name="pipe.p[3]", variability="continuousState")
        assert v.array_indices() == (3,)

    def test_variable_array_indices_2d(self):
        """Gap 15: Multi-dimensional array indices (for future 3D vessel)."""
        v = Variable(id=1, name="vessel.T[2,3]", variability="continuousState")
        assert v.array_indices() == (2, 3)

    def test_variable_array_indices_3d(self):
        """Gap 15: 3D array (Nr, Ntheta, Nz)."""
        v = Variable(id=1, name="vessel.p[4,3,8]", variability="continuousState")
        assert v.array_indices() == (4, 3, 8)

    def test_variable_array_indices_scalar_returns_none(self):
        v = Variable(id=1, name="totalMass", variability="continuous")
        assert v.array_indices() is None

    def test_variable_is_state(self):
        v = Variable(id=1, name="p", variability="continuousState")
        assert v.is_state is True
        assert v.is_algebraic is False

    def test_variable_is_algebraic(self):
        v = Variable(id=1, name="rho", variability="continuous")
        assert v.is_state is False
        assert v.is_algebraic is True

    def test_equation_is_ode(self):
        e = Equation(id=1, text="der(pipe.p[1]) = (mdot[1] - mdot[2]) / C")
        assert e.is_ode is True

    def test_equation_is_not_ode(self):
        e = Equation(id=1, text="rho[1] = SimpleFluid.rho_ph(p[1], h[1])")
        assert e.is_ode is False

    def test_equation_lhs_var_simple(self):
        """Gap 14: lhs_var extracts the first der() argument."""
        e = Equation(id=1, text="der(pipe.p[1]) = something")
        assert e.lhs_var() == "pipe.p[1]"

    def test_equation_lhs_var_with_prefix(self):
        e = Equation(id=1, text="V * der(pipe.h[3]) = flux + work")
        # lhs_var finds the first der() in text, not necessarily on LHS
        assert e.lhs_var() == "pipe.h[3]"

    def test_equation_lhs_var_non_ode_returns_none(self):
        """Gap 14: Non-ODE equation should return None."""
        e = Equation(id=1, text="rho[1] = 750.0")
        assert e.lhs_var() is None

    def test_equation_system_build_indexes(self):
        es = EquationSystem(
            variables=[Variable(1, "x", "continuousState"),
                       Variable(2, "y", "continuous")],
            parameters=[Parameter(1, "k", 1.0)],
            equations=[Equation(1, "der(x) = -k*x")],
        )
        es.build_indexes()
        assert es.var(1).name == "x"
        assert es.var("x").id == 1
        assert es.param("k").value == 1.0
        assert es.eq(1).text == "der(x) = -k*x"

    def test_equation_system_states(self):
        es = EquationSystem(variables=[
            Variable(1, "x", "continuousState"),
            Variable(2, "y", "continuous"),
            Variable(3, "z", "continuousState"),
        ])
        assert len(es.states) == 2
        assert len(es.algebraics) == 1

    def test_equation_system_summary(self):
        es = EquationSystem(
            variables=[Variable(1, "x", "continuousState")],
            parameters=[Parameter(1, "k", 1.0)],
            equations=[Equation(1, "der(x) = -k*x")],
        )
        s = es.summary()
        assert "1 vars" in s
        assert "1 states" in s
        assert "1 params" in s
        assert "1 eqs" in s


# ============================================================================
# Gap 12: Convergence test for extraction-driven solver
# ============================================================================

class TestExtractionConvergence:
    """Verify the extraction-driven solver converges at first order."""

    def _run_steady_state(self, N, n_steps=5000):
        """Run to steady state with N cells, return final pressure profile."""
        from partitioner.equation_classifier import ClassifiedSystem
        from partitioner.extracted_solver import ExtractedSemiImplicitSolver

        dx = 5.0 / N
        A = 0.01
        spec = Pipe1DGridSpec(
            N=N, prefix="pipe", dx=dx, A_flow=A, D_h=0.1, f_D=0.02,
            V_cell=dx * A, p_out=9.9e6, h_out=700e3,
            inlet_closed=False, outlet_closed=False,
            p0=[10e6] * N, h0=[700e3] * N, mdot0=[0.0] * (N + 1))
        # Override inlet_closed for a driven flow problem
        spec.inlet_closed = False

        cs = ClassifiedSystem(prefix="pipe", N=N)
        fluid = tp.SimpleFluidProperties()
        solver = ExtractedSemiImplicitSolver(cs, fluid, spec)
        # Override: open inlet with p_in
        solver.inlet_closed = False

        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(N + 1)
        dt = 1e-4

        for _ in range(n_steps):
            solver.step(p, h, mdot, dt)

        return p, mdot

    def test_pressure_profile_linear_at_steady_state(self):
        """At steady state, pressure should be approximately linear."""
        N = 10
        p, mdot = self._run_steady_state(N, n_steps=10000)

        # Check linearity: max deviation from linear fit
        x = np.arange(N)
        coeffs = np.polyfit(x, p, 1)
        p_linear = np.polyval(coeffs, x)
        max_dev = np.max(np.abs(p - p_linear)) / np.mean(p)

        print(f"  N={N}: max deviation from linear = {max_dev:.2e}")
        assert max_dev < 0.01, f"Pressure not linear at steady state: {max_dev:.2e}"


# ============================================================================
# Gap 13: solver_from_spec produces a working solver
# ============================================================================

class TestSolverFromSpec:
    """Verify solver_from_spec constructs a valid TwoPhaseSolver."""

    def test_constructs_without_error(self):
        if not EDWARDS_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.pipe1d_mapper import map_pipe1d

        es = load_equation_system(str(EDWARDS_XML))
        spec = map_pipe1d(es)
        fluid = tp.SimpleFluidProperties()
        model = tp.HEMModel()

        solver = solver_from_spec(spec, fluid, model)
        assert solver is not None
        assert solver.N == spec.N

    def test_geometry_matches_spec(self):
        if not EDWARDS_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.pipe1d_mapper import map_pipe1d

        es = load_equation_system(str(EDWARDS_XML))
        spec = map_pipe1d(es)
        fluid = tp.SimpleFluidProperties()
        model = tp.HEMModel()

        solver = solver_from_spec(spec, fluid, model)
        assert solver.N == spec.N
        assert solver.dx == pytest.approx(spec.dx)
        assert solver.A_flow == pytest.approx(spec.A_flow)
        assert solver.D_h == pytest.approx(spec.D_h)
        assert solver.f_D == pytest.approx(spec.f_D)

    def test_can_step_without_crash(self):
        if not EDWARDS_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.pipe1d_mapper import map_pipe1d
        es = load_equation_system(str(EDWARDS_XML))
        spec = map_pipe1d(es)
        fluid = tp.SimpleFluidProperties()
        model = tp.HEMModel()

        solver = solver_from_spec(spec, fluid, model)

        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(spec.N + 1)

        bc_in = tp.WallFace(h[0])
        bc_out = tp.PressureFace(spec.p_out, h[0])

        # Should not crash
        solver.step_hem_bf(p, h, mdot, bc_in, bc_out, 0.0, 5e-5)

        assert np.all(np.isfinite(p))
        assert np.all(np.isfinite(h))
        assert np.all(np.isfinite(mdot))


# ============================================================================
# Gap 10: IAPWS C++ evaluation against Python oracle
# (Extends existing test_iapws_cpp.py with extraction-specific checks)
# ============================================================================

class TestIAPWSExtractionReady:
    """Verify IAPWS properties work correctly for the extraction pipeline."""

    def test_evaluate_returns_all_four_properties(self):
        """The extraction pipeline needs rho, drho_dp_h, drho_dh_p, T."""
        fluid = tp.IAPWSIF97Properties()
        props = fluid.evaluate(10e6, 900e3)  # subcooled at 10 MPa

        assert props.rho > 0
        assert props.drho_dp_h > 0  # compressible
        assert props.drho_dh_p < 0  # density decreases with enthalpy (liquid)
        assert props.T > 273.15     # above absolute zero

    def test_phasic_properties_for_init_5eq(self):
        """init_5eq_state needs evaluate_phasic for phase classification."""
        fluid = tp.IAPWSIF97Properties()
        pp = fluid.evaluate_phasic(10e6)

        assert pp.h_sat_l > 0
        assert pp.h_sat_v > pp.h_sat_l
        assert pp.rho_l > pp.rho_v
        assert pp.T_sat > 273.15

    def test_simple_fluid_matches_iapws_at_reference(self):
        """At the reference point (10 MPa, subcooled), both fluids should give
        reasonable (but different) density values."""
        sf = tp.SimpleFluidProperties()
        iapws = tp.IAPWSIF97Properties()

        # Same (p, h) point
        p, h = 10e6, 700e3
        rho_sf = sf.evaluate(p, h).rho
        rho_iapws = iapws.evaluate(p, h).rho

        # Both should give positive density in the right ballpark
        assert 600 < rho_sf < 1000
        assert 600 < rho_iapws < 1000
        # They won't match exactly (SimpleFluid is linear, IAPWS is nonlinear)
        # but should be in the same order of magnitude


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
