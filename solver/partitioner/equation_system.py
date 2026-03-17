"""
equation_system.py — Core dataclasses representing a parsed DAE system.

These are populated by xml_reader.py from dumpXMLDAE backEnd output and
consumed by grid_mapper.py (and future solver routers).
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Variable:
    """An ordered (unknown) variable extracted from the DAE."""
    id: int
    name: str
    variability: str          # 'continuousState' | 'continuous' | 'discrete' | ...
    initial_value: float | None = None
    fixed: bool = False

    @property
    def is_state(self) -> bool:
        return self.variability == "continuousState"

    @property
    def is_algebraic(self) -> bool:
        return self.variability == "continuous"

    def array_base(self) -> str | None:
        """Return 'p' for 'p[3]', None for scalar names."""
        if "[" in self.name:
            return self.name[: self.name.index("[")]
        return None

    def array_indices(self) -> tuple[int, ...] | None:
        """Return (3,) for 'p[3]', (2,3) for 'T[2,3]', None for scalars."""
        if "[" not in self.name:
            return None
        inner = self.name[self.name.index("[") + 1 : self.name.index("]")]
        return tuple(int(x) for x in inner.split(","))


@dataclass
class Parameter:
    """A known (parameter/constant) variable."""
    id: int
    name: str
    value: float | None = None
    type_: str = "Real"


@dataclass
class Equation:
    """One scalar equation from the extracted DAE."""
    id: int
    text: str                 # e.g. "der(p[1]) = (mdot[1] - mdot[2]) / C"

    @property
    def is_ode(self) -> bool:
        return "der(" in self.text

    def lhs_var(self) -> str | None:
        """Return 'p[1]' for 'der(p[1]) = ...', or None."""
        if not self.is_ode:
            return None
        start = self.text.index("der(") + 4
        end = self.text.index(")", start)
        return self.text[start:end]


@dataclass
class EquationSystem:
    """Complete parsed DAE extracted from OpenModelica backEnd XML."""
    variables:  list[Variable]  = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    equations:  list[Equation]  = field(default_factory=list)

    # Adjacency: eq_id -> [var_ids that appear in it]
    adjacency: dict[int, list[int]] = field(default_factory=dict)
    # Matching: var_id -> eq_id that solves for it
    matching:  dict[int, int]       = field(default_factory=dict)
    # BLT order: list of eq_ids in evaluation order
    blt_order: list[int]            = field(default_factory=list)

    # Convenience indexes (populated by xml_reader after loading)
    _var_by_id:   dict[int, Variable]   = field(default_factory=dict, repr=False)
    _var_by_name: dict[str, Variable]   = field(default_factory=dict, repr=False)
    _param_by_name: dict[str, Parameter] = field(default_factory=dict, repr=False)
    _eq_by_id:    dict[int, Equation]   = field(default_factory=dict, repr=False)

    def build_indexes(self) -> None:
        self._var_by_id   = {v.id: v for v in self.variables}
        self._var_by_name = {v.name: v for v in self.variables}
        self._param_by_name = {p.name: p for p in self.parameters}
        self._eq_by_id    = {e.id: e for e in self.equations}

    def var(self, id_or_name: int | str) -> Variable:
        if isinstance(id_or_name, int):
            return self._var_by_id[id_or_name]
        return self._var_by_name[id_or_name]

    def param(self, name: str) -> Parameter:
        return self._param_by_name[name]

    def eq(self, id_: int) -> Equation:
        return self._eq_by_id[id_]

    @property
    def states(self) -> list[Variable]:
        return [v for v in self.variables if v.is_state]

    @property
    def algebraics(self) -> list[Variable]:
        return [v for v in self.variables if v.is_algebraic]

    def summary(self) -> str:
        return (
            f"EquationSystem: {len(self.variables)} vars "
            f"({len(self.states)} states, {len(self.algebraics)} algebraic), "
            f"{len(self.parameters)} params, {len(self.equations)} eqs, "
            f"{len(self.blt_order)} BLT blocks"
        )
