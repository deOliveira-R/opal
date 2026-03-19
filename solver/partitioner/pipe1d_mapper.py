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

Factory functions:
  solver_from_spec()          — Pipe1DGridSpec → TwoPhaseSolver
  boundary_faces_from_spec()  — Pipe1DGridSpec → (bc_in, bc_out)
  init_5eq_state()            — Pipe1DGridSpec + fluid → (p, alpha, h_l, h_v, mdot)
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
# Factory functions: Pipe1DGridSpec → Solver construction
# ---------------------------------------------------------------------------

def solver_from_spec(spec, fluid, model, recon=None, momentum=None,
                     critical_flow=None):
    """Construct a TwoPhaseSolver from an extracted Pipe1DGridSpec.

    The spec provides geometry (N, dx, A_flow, D_h, f_D).
    Strategy objects (fluid, model, recon, momentum, critical_flow) are
    the user's solver configuration — NOT extracted from the Modelica model.
    """
    import opal_two_phase as tp
    return tp.TwoPhaseSolver(
        spec.N, spec.dx, spec.A_flow, spec.D_h, spec.f_D,
        fluid,
        recon or tp.DonorCell(),
        model,
        momentum or tp.InertialMomentum(),
        critical_flow,
    )


def boundary_faces_from_spec(spec, h_l, h_v, C_d=None):
    """Map Pipe1DGridSpec BC flags to BoundaryFace strategy objects.

    Args:
        spec:  Extracted grid spec with inlet_closed, outlet_closed, p_out.
        h_l:   Liquid enthalpy for BC ghost cells [J/kg].
        h_v:   Vapor enthalpy for BC ghost cells [J/kg].
        C_d:   Break discharge coefficient [-]. If provided and outlet is open,
               creates a BreakFace instead of PressureFace.

    Returns:
        (bc_in, bc_out) tuple of BoundaryFace objects.
    """
    import opal_two_phase as tp

    if spec.inlet_closed:
        bc_in = tp.WallFace(h_l, h_v)
    else:
        bc_in = tp.PressureFace(spec.p_out, h_l, h_v, 0.0)  # TODO: p_in field

    if spec.outlet_closed:
        bc_out = tp.WallFace(h_l, h_v)
    elif C_d is not None:
        bc_out = tp.BreakFace(spec.p_out, C_d, h_l, h_v)
    else:
        bc_out = tp.PressureFace(spec.p_out, h_l, h_v, 0.0)

    return bc_in, bc_out


def init_5eq_state(spec, fluid):
    """Initialize 5-equation state arrays from extracted HEM initial conditions.

    The spec gives mixture enthalpy h0. For the 5-eq model we need
    separate (alpha, h_l, h_v). Classification:
      h < h_sat_l → subcooled: alpha=0, h_l=h, h_v=h_sat_v
      h > h_sat_v → superheated: alpha=1, h_l=h_sat_l, h_v=h
      otherwise   → two-phase: alpha from quality, h_l=h_sat_l, h_v=h_sat_v

    Returns:
        (p, alpha, h_l, h_v, mdot) as numpy arrays.
    """
    import numpy as np

    N = spec.N
    p     = np.array(spec.p0, dtype=float)
    alpha = np.zeros(N)
    h_l   = np.zeros(N)
    h_v   = np.zeros(N)
    mdot  = np.array(spec.mdot0, dtype=float)

    for i in range(N):
        pp = fluid.evaluate_phasic(p[i])
        h = spec.h0[i]

        if h <= pp.h_sat_l:
            # Subcooled liquid
            alpha[i] = 1e-6  # near-zero void for numerical stability
            h_l[i] = h
            h_v[i] = pp.h_sat_v
        elif h >= pp.h_sat_v:
            # Superheated vapor
            alpha[i] = 1.0 - 1e-6
            h_l[i] = pp.h_sat_l
            h_v[i] = h
        else:
            # Two-phase: quality x = (h - h_f) / (h_g - h_f)
            h_fg = pp.h_sat_v - pp.h_sat_l
            x = (h - pp.h_sat_l) / h_fg if h_fg > 1.0 else 0.5
            # Void fraction from quality (HEM: alpha = x*rho_l / (x*rho_l + (1-x)*rho_v))
            alpha[i] = x * pp.rho_l / (x * pp.rho_l + (1 - x) * pp.rho_v)
            h_l[i] = pp.h_sat_l
            h_v[i] = pp.h_sat_v

    return p, alpha, h_l, h_v, mdot


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
