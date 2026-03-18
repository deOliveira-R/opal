"""
pipe1d_mapper.py — Recognise a Pipe1D component from an EquationSystem.

Recognises the pattern produced by library.Pipes.Pipe1D with inertial momentum:

  ODE states:      {prefix}.p[1..N], {prefix}.h[1..N], {prefix}.mdot[2..N+1]
  Algebraic vars:  {prefix}.rho[...], {prefix}.drho_dp[...], {prefix}.drho_dh[...],
                   {prefix}.rho_face[...], {prefix}.h_face[...], {prefix}.T_cell[...]
  Parameters:      {prefix}.N, {prefix}.L, {prefix}.D, {prefix}.f_D,
                   {prefix}.dx, {prefix}.A_flow, {prefix}.V_cell, {prefix}.D_h

The prefix (e.g., "pipe") is the component instance name in the system model.
mdot[1] may be eliminated (dummyState) if connected to a ClosedEnd.

Returns a Pipe1DGridSpec that the two-phase solver consumes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .equation_system import EquationSystem


@dataclass
class Pipe1DGridSpec:
    """Fully-specified Pipe1D grid for the two-phase solver."""
    N: int                  # number of cells
    prefix: str             # component instance name (e.g., "pipe")

    # Geometry
    dx: float               # cell length [m]
    A_flow: float           # flow area [m^2]
    D_h: float              # hydraulic diameter [m]
    f_D: float              # Darcy friction factor [-]
    V_cell: float           # cell volume [m^3]

    # Boundary conditions
    p_out: float | None     # outlet pressure [Pa] (None if closed)
    h_out: float | None     # outlet enthalpy [J/kg]
    inlet_closed: bool      # True if inlet is a wall (mdot[1] = 0)
    outlet_closed: bool     # True if outlet is a wall

    # Initial state (0-indexed, length N)
    p0: list[float]
    h0: list[float]
    mdot0: list[float]      # length N+1

    def summary(self) -> str:
        return (
            f"Pipe1DGridSpec: N={self.N}, prefix='{self.prefix}'\n"
            f"  Geometry: dx={self.dx:.4f}, A={self.A_flow:.6f}, "
            f"D_h={self.D_h:.4f}, f_D={self.f_D}\n"
            f"  BCs: inlet_closed={self.inlet_closed}, "
            f"outlet_closed={self.outlet_closed}, "
            f"p_out={self.p_out}"
        )


def map_pipe1d(es: "EquationSystem") -> Pipe1DGridSpec:
    """
    Extract a Pipe1DGridSpec from an EquationSystem.

    Detects the pipe component prefix automatically by looking for
    state variable arrays named {something}.p[...].

    Raises ValueError if the pattern is not recognised.
    """
    # ---- Find prefix by looking for {prefix}.p[i] state variables ----
    prefix = _detect_prefix(es)

    # ---- Identify N from p[1..N] states ----
    p_indices = _get_state_indices(es, prefix, "p")
    h_indices = _get_state_indices(es, prefix, "h")

    N = len(p_indices)
    if N < 1:
        raise ValueError(f"No pressure states found for prefix '{prefix}'")
    if p_indices != list(range(1, N + 1)):
        raise ValueError(f"p indices not contiguous 1..{N}: {p_indices}")
    if h_indices != list(range(1, N + 1)):
        raise ValueError(f"h indices not contiguous 1..{N}: {h_indices}")

    # ---- mdot states: may be [2..N+1] if inlet closed, or [1..N+1] ----
    mdot_state_indices = _get_state_indices(es, prefix, "mdot")
    # Check for dummy states (eliminated by constraints)
    mdot_dummy_indices = _get_indices(es, prefix, "mdot", "continuousDummyState")
    # Also check if mdot[1] or mdot[N+1] ended up in knownVariables (OM sometimes
    # moves constrained variables there instead of making them dummyStates)
    mdot_known = set()
    for p in es.parameters:
        if p.name.startswith(f"{prefix}.mdot["):
            idx = int(p.name.split("[")[1].rstrip("]"))
            mdot_known.add(idx)

    all_mdot = sorted(set(mdot_state_indices + mdot_dummy_indices) | mdot_known)

    inlet_closed = (1 not in mdot_state_indices)
    outlet_closed = ((N + 1) not in mdot_state_indices)

    # ---- Extract parameters ----
    def pval(name: str) -> float:
        full_name = f"{prefix}.{name}"
        try:
            p = es.param(full_name)
            if p.value is not None:
                return p.value
        except KeyError:
            pass
        raise ValueError(f"Parameter '{full_name}' not found or has no value")

    # Base parameters (OM may inline derived ones like dx, A_flow, V_cell)
    L = pval("L")
    D = pval("D")
    f_D = pval("f_D")
    N_param = int(pval("N"))

    if N_param != N:
        raise ValueError(f"N from states ({N}) != N parameter ({N_param})")

    import math
    dx = L / N
    A_flow = math.pi / 4 * D**2
    D_h = D
    V_cell = dx * A_flow

    # ---- Boundary pressure: look for atm.p_set or similar ----
    # Scan parameters for a pressure source value
    p_out = None
    h_out = None
    for p in es.parameters:
        if p.name.endswith(".p_set") and p.value is not None:
            p_out = p.value
        if p.name.endswith(".h_set") and p.value is not None:
            h_out = p.value

    # ---- Initial conditions ----
    p_init = pval("p_init")
    h_init = pval("h_init")

    p0 = []
    h0 = []
    for i in range(1, N + 1):
        # Try to get specific initial value, fall back to p_init/h_init
        pv = _get_initial_value(es, f"{prefix}.p[{i}]", p_init)
        hv = _get_initial_value(es, f"{prefix}.h[{i}]", h_init)
        p0.append(pv)
        h0.append(hv)

    mdot0 = [0.0] * (N + 1)  # initial mass flow (all zero for blowdown)

    return Pipe1DGridSpec(
        N=N, prefix=prefix,
        dx=dx, A_flow=A_flow, D_h=D_h, f_D=f_D, V_cell=V_cell,
        p_out=p_out, h_out=h_out,
        inlet_closed=inlet_closed, outlet_closed=outlet_closed,
        p0=p0, h0=h0, mdot0=mdot0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_prefix(es: "EquationSystem") -> str:
    """Find the component prefix from state variable names like 'pipe.p[1]'."""
    for v in es.states:
        base = v.array_base()
        if base and ".p" in v.name and base.endswith(".p"):
            # base is "pipe.p" → prefix is "pipe"
            return base.rsplit(".p", 1)[0]
    raise ValueError("Cannot detect pipe prefix — no {prefix}.p[i] state found")


def _get_state_indices(es: "EquationSystem", prefix: str, var: str) -> list[int]:
    """Get sorted indices of state variables matching {prefix}.{var}[i]."""
    target = f"{prefix}.{var}"
    indices = []
    for v in es.states:
        if v.array_base() == target:
            idx = v.array_indices()
            if idx and len(idx) == 1:
                indices.append(idx[0])
    return sorted(indices)


def _get_indices(es: "EquationSystem", prefix: str, var: str,
                 variability: str) -> list[int]:
    """Get indices of variables with specific variability."""
    target = f"{prefix}.{var}"
    indices = []
    for v in es.variables:
        if v.variability == variability and v.array_base() == target:
            idx = v.array_indices()
            if idx and len(idx) == 1:
                indices.append(idx[0])
    return sorted(indices)


def _get_initial_value(es: "EquationSystem", name: str,
                       default: float) -> float:
    """Get initial value for a variable, with fallback."""
    try:
        v = es.var(name)
        if v.initial_value is not None:
            return v.initial_value
    except KeyError:
        pass
    return default
