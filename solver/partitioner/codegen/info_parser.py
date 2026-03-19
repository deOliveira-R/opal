"""
info_parser.py — Parse OpenModelica's _info.json for variable/parameter/equation maps.

translateModel generates {Model}_info.json with authoritative metadata:
- Variable names → array indices (states, derivatives, algebraic)
- Parameter names → array indices
- Equation catalog (eqIndex, tag, defines, uses)

This is the ground truth for the C bridge — no C code comment parsing needed.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VarInfo:
    """A single variable from the _info.json."""
    name: str
    kind: str       # "state", "derivative", "variable", "dummy state", "parameter"
    index: int      # position in realVars[] or realParameter[]
    comment: str = ""


@dataclass
class EqInfo:
    """A single equation from the _info.json."""
    eq_index: int
    tag: str        # "assign", "tornsystem", "torn", "residual", "jacobian", "alias", "dummy"
    section: str    # "regular", "initial", or empty
    defines: list[str] = field(default_factory=list)
    uses: list[str] = field(default_factory=list)


@dataclass
class ModelInfo:
    """Complete parsed model metadata from _info.json."""
    model_name: str

    # Variable maps (name → index in realVars[])
    states: dict[str, VarInfo] = field(default_factory=dict)
    derivatives: dict[str, VarInfo] = field(default_factory=dict)
    algebraic: dict[str, VarInfo] = field(default_factory=dict)
    all_vars: dict[str, VarInfo] = field(default_factory=dict)

    # Parameter map (name → index in realParameter[])
    parameters: dict[str, VarInfo] = field(default_factory=dict)

    # Sizes
    n_vars: int = 0       # total realVars slots
    n_params: int = 0     # total realParameter slots
    n_states: int = 0

    # Equation catalog
    equations: list[EqInfo] = field(default_factory=list)

    # Runtime algebraic equations (tag="assign", callable from bridge)
    algebraic_eqs: list[EqInfo] = field(default_factory=list)

    # BLT evaluation order (equation indices for algebraic eqs)
    blt_order: list[int] = field(default_factory=list)

    def var_index(self, name: str) -> int:
        """Get the realVars index for a variable name."""
        return self.all_vars[name].index

    def param_index(self, name: str) -> int:
        """Get the realParameter index for a parameter name."""
        return self.parameters[name].index

    def vars_by_pattern(self, prefix: str, n: int) -> list[int]:
        """Get indices for an array variable, e.g., vars_by_pattern('pipe.p', 3) → [6,7,8]."""
        indices = []
        for i in range(1, n + 1):
            name = f"{prefix}[{i}]"
            if name in self.all_vars:
                indices.append(self.all_vars[name].index)
        return indices

    def summary(self) -> str:
        return (
            f"ModelInfo: {self.model_name}\n"
            f"  Variables: {self.n_vars} ({self.n_states} states, "
            f"{len(self.derivatives)} derivatives, {len(self.algebraic)} algebraic)\n"
            f"  Parameters: {self.n_params}\n"
            f"  Equations: {len(self.equations)} total, "
            f"{len(self.algebraic_eqs)} algebraic (callable)"
        )


def parse_info_json(path: Path) -> ModelInfo:
    """Parse a translateModel _info.json file into a ModelInfo.

    Args:
        path: Path to {Model}_info.json

    Returns:
        ModelInfo with complete variable, parameter, and equation metadata.
    """
    with open(path) as f:
        data = json.load(f)

    model_name = data.get("info", {}).get("name", "Unknown")
    info = ModelInfo(model_name=model_name)

    # Parse variables
    max_var_idx = -1
    max_param_idx = -1

    for name, vdata in data.get("variables", {}).items():
        kind = vdata.get("kind", "")
        index = vdata.get("index", 0)
        comment = vdata.get("comment", "")

        vi = VarInfo(name=name, kind=kind, index=index, comment=comment)

        if kind == "parameter":
            info.parameters[name] = vi
            max_param_idx = max(max_param_idx, index)
        elif "jacobian" in kind:
            # Skip jacobian variables — internal to OM's linear solver
            continue
        else:
            info.all_vars[name] = vi
            max_var_idx = max(max_var_idx, index)

            if kind == "state":
                info.states[name] = vi
            elif kind == "derivative":
                info.derivatives[name] = vi
            elif kind in ("variable", "dummy state"):
                info.algebraic[name] = vi

    info.n_vars = max_var_idx + 1
    info.n_params = max_param_idx + 1
    info.n_states = len(info.states)

    # Parse equations
    for eq_data in data.get("equations", []):
        eq_index = eq_data.get("eqIndex", 0)
        tag = eq_data.get("tag", "")
        section = eq_data.get("section", "regular")
        defines = eq_data.get("defines", [])
        uses = eq_data.get("uses", [])

        eq = EqInfo(
            eq_index=eq_index,
            tag=tag,
            section=section,
            defines=defines,
            uses=uses,
        )
        info.equations.append(eq)

        # Collect runtime algebraic equations (tag="assign", not initial/jacobian)
        if tag == "assign" and section in ("regular", None, ""):
            info.algebraic_eqs.append(eq)

    # BLT order: the order algebraic_eqs appear is the evaluation order
    # (OM generates them in BLT-sorted order in the JSON)
    info.blt_order = [eq.eq_index for eq in info.algebraic_eqs]

    return info
