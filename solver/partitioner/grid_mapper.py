"""
grid_mapper.py — Recognise a staggered-mesh pipe grid from an EquationSystem.

The recogniser looks for the pattern produced by ScalablePipe (and, more
generally, any N-cell 1D pipe with constant properties):

  State variables:   p[1..N], T[1..N]     (continuousState, 1-indexed)
  Algebraic vars:    mdot[1..N+1]         (continuous, 1-indexed)
  Parameters:        R, C, rho, Cp, V, p_in, p_out, T_in

Returns a PipeGridSpec dataclass that the single-phase C++ solver consumes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .equation_system import EquationSystem


@dataclass
class PipeGridSpec:
    """Fully-specified grid ready to hand to the single-phase C++ solver."""
    N: int           # number of cells

    # Grid / fluid parameters (extracted from parsed Parameters)
    R:   float       # cell friction resistance [Pa/(kg/s)]
    C:   float       # cell compressibility [kg/Pa]
    rho: float       # fluid density [kg/m³]
    Cp:  float       # specific heat [J/(kg·K)]
    V:   float       # cell volume [m³]

    # Boundary conditions
    p_in:  float     # inlet pressure [Pa]
    p_out: float     # outlet pressure [Pa]
    T_in:  float     # inlet temperature [K]

    # Initial state (length-N arrays, 1-indexed Modelica → 0-indexed here)
    p0: list[float]  # initial cell pressures
    T0: list[float]  # initial cell temperatures

    def summary(self) -> str:
        return (
            f"PipeGridSpec: N={self.N}, R={self.R:.3g}, C={self.C:.3g}, "
            f"rho={self.rho:.3g}, Cp={self.Cp:.3g}, V={self.V:.3g}\n"
            f"  BCs: p_in={self.p_in:.6g} Pa, p_out={self.p_out:.6g} Pa, "
            f"T_in={self.T_in:.3g} K"
        )


# Required parameter names (all must be present in the EquationSystem)
_REQUIRED_PARAMS = {"R", "C", "rho", "Cp", "V", "p_in", "p_out", "T_in"}


def map_pipe_grid(es: "EquationSystem") -> PipeGridSpec:
    """
    Extract a PipeGridSpec from an EquationSystem.

    Raises ValueError if the system does not look like a staggered-mesh pipe.
    """
    # ---- Identify array bases ----------------------------------------
    state_bases = _array_bases(es.states)
    alg_bases   = _array_bases(es.algebraics)

    if "p" not in state_bases or "T" not in state_bases:
        raise ValueError(
            f"Expected state arrays 'p' and 'T', found: {sorted(state_bases)}"
        )
    if "mdot" not in alg_bases:
        raise ValueError(
            f"Expected algebraic array 'mdot', found: {sorted(alg_bases)}"
        )

    # ---- Determine N from the p[i] states ----------------------------
    p_indices = sorted(
        v.array_indices()[0]
        for v in es.states
        if v.array_base() == "p"
    )
    T_indices = sorted(
        v.array_indices()[0]
        for v in es.states
        if v.array_base() == "T"
    )
    mdot_indices = sorted(
        v.array_indices()[0]
        for v in es.algebraics
        if v.array_base() == "mdot"
    )

    N = len(p_indices)
    if p_indices != list(range(1, N + 1)):
        raise ValueError(f"p indices not contiguous 1..N: {p_indices}")
    if T_indices != list(range(1, N + 1)):
        raise ValueError(f"T indices not contiguous 1..N: {T_indices}")
    if mdot_indices != list(range(1, N + 2)):
        raise ValueError(f"mdot indices not contiguous 1..N+1: {mdot_indices}")

    # ---- Extract parameters ------------------------------------------
    missing = _REQUIRED_PARAMS - {p.name for p in es.parameters}
    if missing:
        raise ValueError(f"Missing parameters: {missing}")

    def pval(name: str) -> float:
        p = es.param(name)
        if p.value is None:
            raise ValueError(f"Parameter '{name}' has no numeric value")
        return p.value

    # ---- Build initial state (0-indexed) -----------------------------
    p0 = [
        es.var(f"p[{i}]").initial_value or pval("p_in")
        for i in range(1, N + 1)
    ]
    T0 = [
        es.var(f"T[{i}]").initial_value or pval("T_in")
        for i in range(1, N + 1)
    ]

    return PipeGridSpec(
        N=N,
        R=pval("R"),  C=pval("C"),
        rho=pval("rho"), Cp=pval("Cp"), V=pval("V"),
        p_in=pval("p_in"), p_out=pval("p_out"), T_in=pval("T_in"),
        p0=p0, T0=T0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _array_bases(variables) -> set[str]:
    """Return the set of array base names (e.g. {'p', 'T'}) present."""
    bases = set()
    for v in variables:
        b = v.array_base()
        if b is not None:
            bases.add(b)
    return bases
