"""
test_partitioner.py — Unit tests for the equation partitioner.

Tests:
  1. Variable — array_base, array_indices, is_state, is_algebraic
  2. Equation — is_ode, lhs_var
  3. EquationSystem — build_indexes, var/param/eq lookups, states/algebraics
  4. xml_reader — parse a synthetic XML, verify all fields
  5. grid_mapper — happy path + error cases (missing vars, bad indices, missing params)

Run with:
  python -m pytest solver/tests/test_partitioner.py -v
"""

from __future__ import annotations
import sys
import textwrap
import tempfile
from pathlib import Path

SOLVER_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SOLVER_DIR))

from partitioner.equation_system import Variable, Parameter, Equation, EquationSystem
from partitioner.xml_reader import load_equation_system
from partitioner.grid_mapper import map_pipe_grid, PipeGridSpec

import pytest


# ===========================================================================
# Test 1 — Variable dataclass
# ===========================================================================

class TestVariable:
    def test_array_base_scalar(self):
        v = Variable(id=1, name="rho", variability="continuous")
        assert v.array_base() is None

    def test_array_base_1d(self):
        v = Variable(id=1, name="p[3]", variability="continuousState")
        assert v.array_base() == "p"

    def test_array_base_2d(self):
        v = Variable(id=1, name="T[2,3]", variability="continuous")
        assert v.array_base() == "T"

    def test_array_indices_scalar(self):
        v = Variable(id=1, name="rho", variability="continuous")
        assert v.array_indices() is None

    def test_array_indices_1d(self):
        v = Variable(id=1, name="p[5]", variability="continuousState")
        assert v.array_indices() == (5,)

    def test_array_indices_2d(self):
        v = Variable(id=1, name="T[2,3]", variability="continuous")
        assert v.array_indices() == (2, 3)

    def test_is_state(self):
        v = Variable(id=1, name="p[1]", variability="continuousState")
        assert v.is_state is True
        assert v.is_algebraic is False

    def test_is_algebraic(self):
        v = Variable(id=1, name="mdot[1]", variability="continuous")
        assert v.is_algebraic is True
        assert v.is_state is False


# ===========================================================================
# Test 2 — Equation dataclass
# ===========================================================================

class TestEquation:
    def test_is_ode_true(self):
        eq = Equation(id=1, text="der(p[1]) = (mdot[1] - mdot[2]) / C")
        assert eq.is_ode is True

    def test_is_ode_false(self):
        eq = Equation(id=2, text="mdot[1] = (p_in - p[1]) / R")
        assert eq.is_ode is False

    def test_lhs_var(self):
        eq = Equation(id=1, text="der(p[1]) = (mdot[1] - mdot[2]) / C")
        assert eq.lhs_var() == "p[1]"

    def test_lhs_var_none(self):
        eq = Equation(id=2, text="mdot[1] = (p_in - p[1]) / R")
        assert eq.lhs_var() is None

    def test_lhs_var_temperature(self):
        eq = Equation(id=3, text="der(T[2]) = mdot[2] * (T_in - T[2])")
        assert eq.lhs_var() == "T[2]"


# ===========================================================================
# Test 3 — EquationSystem dataclass
# ===========================================================================

class TestEquationSystem:
    @pytest.fixture
    def system(self):
        es = EquationSystem(
            variables=[
                Variable(id=1, name="p[1]", variability="continuousState", initial_value=15.5e6),
                Variable(id=2, name="p[2]", variability="continuousState", initial_value=15.5e6),
                Variable(id=3, name="T[1]", variability="continuousState", initial_value=563.0),
                Variable(id=4, name="T[2]", variability="continuousState", initial_value=563.0),
                Variable(id=5, name="mdot[1]", variability="continuous"),
                Variable(id=6, name="mdot[2]", variability="continuous"),
                Variable(id=7, name="mdot[3]", variability="continuous"),
            ],
            parameters=[
                Parameter(id=1, name="R", value=1e4),
                Parameter(id=2, name="C", value=1e-9),
                Parameter(id=3, name="rho", value=720.0),
            ],
            equations=[
                Equation(id=1, text="der(p[1]) = (mdot[1] - mdot[2]) / C"),
                Equation(id=2, text="der(p[2]) = (mdot[2] - mdot[3]) / C"),
                Equation(id=3, text="mdot[1] = (p_in - p[1]) / R"),
            ],
        )
        es.build_indexes()
        return es

    def test_var_by_id(self, system):
        v = system.var(1)
        assert v.name == "p[1]"

    def test_var_by_name(self, system):
        v = system.var("mdot[2]")
        assert v.id == 6

    def test_var_missing_raises(self, system):
        with pytest.raises(KeyError):
            system.var("nonexistent")

    def test_param_lookup(self, system):
        p = system.param("R")
        assert p.value == 1e4

    def test_states(self, system):
        states = system.states
        assert len(states) == 4
        assert all(s.is_state for s in states)

    def test_algebraics(self, system):
        algs = system.algebraics
        assert len(algs) == 3
        assert all(a.is_algebraic for a in algs)

    def test_summary(self, system):
        s = system.summary()
        assert "7 vars" in s
        assert "4 states" in s
        assert "3 algebraic" in s
        assert "3 params" in s


# ===========================================================================
# Test 4 — xml_reader with synthetic XML
# ===========================================================================

_SYNTHETIC_XML = textwrap.dedent("""\
<?xml version="1.0" encoding="UTF-8"?>
<dae xmlns:mml="http://www.w3.org/1998/Math/MathML">
  <orderedVariables>
    <variable id="1" name="p[1]" variability="continuousState" fixed="true">
      <initialValue string="15500000.0"/>
    </variable>
    <variable id="2" name="p[2]" variability="continuousState" fixed="true">
      <initialValue string="15500000.0"/>
    </variable>
    <variable id="3" name="T[1]" variability="continuousState" fixed="true">
      <initialValue string="563.0"/>
    </variable>
    <variable id="4" name="T[2]" variability="continuousState" fixed="true">
      <initialValue string="563.0"/>
    </variable>
    <variable id="5" name="mdot[1]" variability="continuous"/>
    <variable id="6" name="mdot[2]" variability="continuous"/>
    <variable id="7" name="mdot[3]" variability="continuous"/>
  </orderedVariables>
  <knownVariables>
    <variable id="1" name="R" type="Real"><bindExpression string="10000.0"/></variable>
    <variable id="2" name="C" type="Real"><bindExpression string="1e-09"/></variable>
    <variable id="3" name="rho" type="Real"><bindExpression string="720.0"/></variable>
    <variable id="4" name="Cp" type="Real"><bindExpression string="5000.0"/></variable>
    <variable id="5" name="V" type="Real"><bindExpression string="0.01"/></variable>
    <variable id="6" name="p_in" type="Real"><bindExpression string="15500000.0"/></variable>
    <variable id="7" name="p_out" type="Real"><bindExpression string="15400000.0"/></variable>
    <variable id="8" name="T_in" type="Real"><bindExpression string="563.0"/></variable>
  </knownVariables>
  <equations>
    <equation id="1">der(p[1]) = (mdot[1] - mdot[2]) / C</equation>
    <equation id="2">der(p[2]) = (mdot[2] - mdot[3]) / C</equation>
    <equation id="3">der(T[1]) = mdot[1] * (T_in - T[1])</equation>
    <equation id="4">der(T[2]) = mdot[2] * (T_in - T[2])</equation>
    <equation id="5">mdot[1] = (p_in - p[1]) / R</equation>
    <equation id="6">mdot[2] = (p[1] - p[2]) / R</equation>
    <equation id="7">mdot[3] = (p[2] - p_out) / R</equation>
  </equations>
  <originalAdjacencyMatrix>
    <mml:matrix>
      <mml:matrixrow id="1"><mml:ci>1</mml:ci><mml:ci>5</mml:ci><mml:ci>6</mml:ci></mml:matrixrow>
      <mml:matrixrow id="2"><mml:ci>2</mml:ci><mml:ci>6</mml:ci><mml:ci>7</mml:ci></mml:matrixrow>
      <mml:matrixrow id="5"><mml:ci>1</mml:ci><mml:ci>5</mml:ci></mml:matrixrow>
    </mml:matrix>
  </originalAdjacencyMatrix>
  <bltRepresentation>
    <bltBlock>
      <involvedEquation equationId="5"/>
      <involvedEquation equationId="1"/>
    </bltBlock>
  </bltRepresentation>
  <solvedIn variableId="1" equationId="1"/>
  <solvedIn variableId="5" equationId="5"/>
</dae>
""")


class TestXmlReader:
    @pytest.fixture
    def system(self, tmp_path):
        xml_file = tmp_path / "test_dae.xml"
        xml_file.write_text(_SYNTHETIC_XML)
        return load_equation_system(xml_file)

    def test_variable_count(self, system):
        assert len(system.variables) == 7

    def test_variable_properties(self, system):
        p1 = system.var("p[1]")
        assert p1.id == 1
        assert p1.variability == "continuousState"
        assert p1.initial_value == 15500000.0
        assert p1.fixed is True

    def test_parameter_count(self, system):
        assert len(system.parameters) == 8

    def test_parameter_values(self, system):
        assert system.param("R").value == 10000.0
        assert system.param("C").value == 1e-9
        assert system.param("p_in").value == 15500000.0

    def test_equation_count(self, system):
        assert len(system.equations) == 7

    def test_equation_ode_detection(self, system):
        eq1 = system.eq(1)
        assert eq1.is_ode is True
        assert eq1.lhs_var() == "p[1]"
        eq5 = system.eq(5)
        assert eq5.is_ode is False

    def test_adjacency(self, system):
        assert 1 in system.adjacency
        assert sorted(system.adjacency[1]) == [1, 5, 6]

    def test_matching(self, system):
        assert system.matching[1] == 1   # var 1 (p[1]) solved by eq 1
        assert system.matching[5] == 5   # var 5 (mdot[1]) solved by eq 5

    def test_blt_order(self, system):
        assert system.blt_order == [5, 1]

    def test_states_and_algebraics(self, system):
        assert len(system.states) == 4
        assert len(system.algebraics) == 3
        state_names = {s.name for s in system.states}
        assert state_names == {"p[1]", "p[2]", "T[1]", "T[2]"}


# ===========================================================================
# Test 5 — grid_mapper
# ===========================================================================

class TestGridMapper:
    @pytest.fixture
    def system(self, tmp_path):
        xml_file = tmp_path / "test_dae.xml"
        xml_file.write_text(_SYNTHETIC_XML)
        return load_equation_system(xml_file)

    def test_happy_path(self, system):
        grid = map_pipe_grid(system)
        assert isinstance(grid, PipeGridSpec)
        assert grid.N == 2
        assert grid.R == 10000.0
        assert grid.C == 1e-9
        assert grid.rho == 720.0
        assert grid.Cp == 5000.0
        assert grid.V == 0.01
        assert grid.p_in == 15500000.0
        assert grid.p_out == 15400000.0
        assert grid.T_in == 563.0
        assert len(grid.p0) == 2
        assert len(grid.T0) == 2
        assert grid.p0[0] == 15500000.0
        assert grid.T0[0] == 563.0

    def test_missing_p_states(self):
        """No p[] states → should raise ValueError."""
        es = EquationSystem(
            variables=[
                Variable(id=1, name="T[1]", variability="continuousState"),
                Variable(id=2, name="mdot[1]", variability="continuous"),
            ],
        )
        es.build_indexes()
        with pytest.raises(ValueError, match="Expected state arrays.*p.*and.*T"):
            map_pipe_grid(es)

    def test_missing_mdot(self):
        """No mdot[] algebraics → should raise ValueError."""
        es = EquationSystem(
            variables=[
                Variable(id=1, name="p[1]", variability="continuousState"),
                Variable(id=2, name="T[1]", variability="continuousState"),
                Variable(id=3, name="flow[1]", variability="continuous"),
            ],
        )
        es.build_indexes()
        with pytest.raises(ValueError, match="Expected algebraic array.*mdot"):
            map_pipe_grid(es)

    def test_non_contiguous_indices(self):
        """p[1], p[3] (missing p[2]) → should raise ValueError."""
        es = EquationSystem(
            variables=[
                Variable(id=1, name="p[1]", variability="continuousState"),
                Variable(id=2, name="p[3]", variability="continuousState"),
                Variable(id=3, name="T[1]", variability="continuousState"),
                Variable(id=4, name="T[3]", variability="continuousState"),
                Variable(id=5, name="mdot[1]", variability="continuous"),
                Variable(id=6, name="mdot[2]", variability="continuous"),
                Variable(id=7, name="mdot[3]", variability="continuous"),
            ],
            parameters=[
                Parameter(id=i, name=n, value=v)
                for i, (n, v) in enumerate([
                    ("R", 1e4), ("C", 1e-9), ("rho", 720.0), ("Cp", 5000.0),
                    ("V", 0.01), ("p_in", 15.5e6), ("p_out", 15.4e6), ("T_in", 563.0),
                ], 1)
            ],
        )
        es.build_indexes()
        with pytest.raises(ValueError, match="p indices not contiguous"):
            map_pipe_grid(es)

    def test_missing_parameters(self):
        """Missing required parameter → should raise ValueError."""
        es = EquationSystem(
            variables=[
                Variable(id=1, name="p[1]", variability="continuousState"),
                Variable(id=2, name="T[1]", variability="continuousState"),
                Variable(id=3, name="mdot[1]", variability="continuous"),
                Variable(id=4, name="mdot[2]", variability="continuous"),
            ],
            parameters=[
                Parameter(id=1, name="R", value=1e4),
                # Missing C, rho, Cp, V, p_in, p_out, T_in
            ],
        )
        es.build_indexes()
        with pytest.raises(ValueError, match="Missing parameters"):
            map_pipe_grid(es)

    def test_summary(self, system):
        grid = map_pipe_grid(system)
        s = grid.summary()
        assert "N=2" in s
        assert "p_in" in s


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
