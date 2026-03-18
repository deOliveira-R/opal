"""
test_pipe1d_integration.py — Pipe1D extraction → partitioner integration test.

Verifies the full pipeline from Modelica Pipe1D.mo through OpenModelica
extraction to the OPAL partitioner. This is the Phase 2.5a integration
test that proves the extraction-mapping pipeline works for the new
Pipe1D component with inertial momentum.

Note: does NOT run the C++ solver (which lacks inertial momentum).
That integration is Phase 3 scope. This test verifies:
1. XML loads correctly via xml_reader
2. pipe1d_mapper recognizes the equation structure
3. State variables, parameters, and BCs are correctly extracted
4. Equation classification (mass, momentum, energy) is correct
"""

import sys
import os
import pytest
from pathlib import Path

# Add partitioner to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d

OPAL_ROOT = Path(__file__).resolve().parents[2]
EDWARDS_XML = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_backEnd.xml"


@pytest.fixture
def edwards_es():
    """Load the Edwards test XML into an EquationSystem."""
    if not EDWARDS_XML.exists():
        pytest.skip("EdwardsTest XML not available")
    return load_equation_system(EDWARDS_XML)


@pytest.fixture
def edwards_spec(edwards_es):
    """Map the Edwards EquationSystem to a Pipe1DGridSpec."""
    return map_pipe1d(edwards_es)


# ============================================================================
# Test 1: XML loading
# ============================================================================

class TestXMLLoading:
    """Verify the extracted XML loads correctly."""

    def test_loads_without_error(self, edwards_es):
        assert edwards_es is not None

    def test_variable_count(self, edwards_es):
        """EdwardsTest N=5: 15 ODE states + 30 algebraic = 45 ordered variables."""
        assert len(edwards_es.variables) == 45

    def test_state_count(self, edwards_es):
        """5 pressure + 5 enthalpy + 5 momentum = 15 ODE states."""
        states = edwards_es.states
        assert len(states) == 15

    def test_equation_count(self, edwards_es):
        """45 equations for 45 ordered variables."""
        assert len(edwards_es.equations) == 45

    def test_state_names(self, edwards_es):
        """State arrays should be pipe.p[1..5], pipe.h[1..5], pipe.mdot[2..6]."""
        state_names = sorted(v.name for v in edwards_es.states)
        expected = sorted(
            [f"pipe.h[{i}]" for i in range(1, 6)] +
            [f"pipe.p[{i}]" for i in range(1, 6)] +
            [f"pipe.mdot[{i}]" for i in range(2, 7)]
        )
        assert state_names == expected

    def test_mdot1_eliminated(self, edwards_es):
        """pipe.mdot[1] should NOT be an ODE state (ClosedEnd forces it to 0)."""
        state_names = {v.name for v in edwards_es.states}
        assert "pipe.mdot[1]" not in state_names


# ============================================================================
# Test 2: Partitioner mapping
# ============================================================================

class TestPartitionerMapping:
    """Verify pipe1d_mapper correctly extracts grid spec."""

    def test_cell_count(self, edwards_spec):
        assert edwards_spec.N == 5

    def test_prefix(self, edwards_spec):
        assert edwards_spec.prefix == "pipe"

    def test_geometry(self, edwards_spec):
        """Geometry derived from L=4.096, D=0.073, N=5."""
        import math
        assert abs(edwards_spec.dx - 4.096 / 5) < 1e-10
        assert abs(edwards_spec.A_flow - math.pi / 4 * 0.073**2) < 1e-10
        assert abs(edwards_spec.D_h - 0.073) < 1e-10
        assert edwards_spec.f_D == 0.02

    def test_boundary_conditions(self, edwards_spec):
        """Inlet closed (ClosedEnd), outlet open (PressureSource at 101325 Pa)."""
        assert edwards_spec.inlet_closed is True
        assert edwards_spec.outlet_closed is False
        assert edwards_spec.p_out == 101325.0

    def test_initial_conditions(self, edwards_spec):
        """All cells initialized to p=7 MPa, h=980 kJ/kg."""
        assert all(p == 7e6 for p in edwards_spec.p0)
        assert all(h == 980e3 for h in edwards_spec.h0)
        assert len(edwards_spec.p0) == 5
        assert len(edwards_spec.h0) == 5


# ============================================================================
# Test 3: Equation classification
# ============================================================================

class TestEquationClassification:
    """Verify the extracted equations have the expected structure."""

    def test_ode_equations_exist(self, edwards_es):
        """Should have ODE equations (containing 'der(')."""
        ode_eqs = [e for e in edwards_es.equations if e.is_ode]
        # 5 mass + 5 momentum + 5 energy = 15 ODEs
        # But OM may rearrange — at least check some exist
        assert len(ode_eqs) >= 10, \
            f"Expected at least 10 ODE equations, found {len(ode_eqs)}"

    def test_mass_equations(self, edwards_es):
        """Mass conservation: V*(drho_dp*der(p) + drho_dh*der(h)) = mdot_in - mdot_out."""
        mass_eqs = [e for e in edwards_es.equations
                    if "drho_dp" in e.text and "der(pipe.p" in e.text]
        assert len(mass_eqs) == 5, \
            f"Expected 5 mass conservation equations, found {len(mass_eqs)}"

    def test_momentum_equations(self, edwards_es):
        """Momentum: der(pipe.mdot[i]) = ... (inertial, NOT algebraic)."""
        mom_eqs = [e for e in edwards_es.equations
                   if "der(pipe.mdot" in e.text]
        assert len(mom_eqs) == 5, \
            f"Expected 5 momentum ODEs, found {len(mom_eqs)}"

    def test_energy_equations(self, edwards_es):
        """Energy: rho*V*der(h[i]) = advection + pressure_work + q_wall."""
        energy_eqs = [e for e in edwards_es.equations
                      if "der(pipe.h" in e.text and "q_wall" in e.text]
        assert len(energy_eqs) == 5, \
            f"Expected 5 energy equations, found {len(energy_eqs)}"

    def test_property_calls(self, edwards_es):
        """Property evaluation calls should reference SimpleFluid functions."""
        prop_eqs = [e for e in edwards_es.equations
                    if "SimpleFluid.rho_ph" in e.text]
        assert len(prop_eqs) >= 4, \
            f"Expected at least 4 property evaluation equations"

    def test_donor_cell_enthalpy(self, edwards_es):
        """Donor-cell face enthalpy: if mdot >= 0 then h_upwind else h_downwind."""
        dc_eqs = [e for e in edwards_es.equations
                  if "h_face" in e.text and "if" in e.text]
        assert len(dc_eqs) == 5, \
            f"Expected 5 donor-cell face enthalpy equations, found {len(dc_eqs)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
