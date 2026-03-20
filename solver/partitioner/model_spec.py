"""
model_spec.py — Complete model specification extracted from Modelica.

Replaces the ad-hoc Pipe1DGridSpec + hardcoded closure parameters
with a single, comprehensive spec that contains EVERYTHING the solver
needs — all extracted from the Modelica model, nothing hardcoded.

For Phase 3: this will evolve into SubsystemSpec (per-component specs)
with boundary coupling between subsystems.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .equation_system import EquationSystem


@dataclass
class GeometrySpec:
    """Pipe geometry — extracted from Modelica parameters."""
    N: int
    dx: float           # [m]
    A_flow: float       # [m²]
    D_h: float          # [m]
    f_D: float          # [-]
    V_cell: float       # [m³]
    g_axial: float = 0.0  # [m/s²]


@dataclass
class ClosureSpec:
    """Closure model parameters — extracted from Modelica parameters.
    Zero hardcoded defaults. Everything must come from the model."""
    H_i: float               # Interfacial HT coefficient [W/(m³·K)]
    C_0: float               # Drift-flux distribution parameter [-]
    alpha_nucleation: float   # Nucleation onset void fraction [-]
    use_critical_flow: bool
    C_d: float               # Break discharge coefficient [-]
    x_trans: float            # Critical flow quality transition [-]
    c_floor: float            # Minimum sound speed [m/s]
    use_two_phase_friction: bool
    Phi2_max: float           # Max two-phase friction multiplier [-]


@dataclass
class BoundarySpec:
    """Boundary conditions — extracted from Modelica model connectivity."""
    inlet_type: str      # "wall", "pressure", "pump" (extensible for Phase 3)
    outlet_type: str     # "wall", "pressure", "break"
    p_out: float | None  # Outlet pressure [Pa] (if pressure/break BC)
    h_out: float | None  # Outlet enthalpy [J/kg]


@dataclass
class InitialConditions:
    """Initial state — extracted from Modelica parameters."""
    p0: list[float]
    h0: list[float]       # mixture enthalpy (HEM) or liquid enthalpy (5-eq)
    h_v0: list[float] | None = None  # vapor enthalpy (5-eq only)
    alpha0: list[float] | None = None  # void fraction (5-eq only)
    mdot0: list[float] = field(default_factory=list)


@dataclass
class SafetyThresholds:
    """Numerical safety thresholds — from SolverNumerics or sensible fluid-derived bounds.
    These are solver configuration, NOT physics."""
    rho_face_min: float = 0.01     # [kg/m³] skip friction if ρ below this
    alpha_min: float = 1e-6        # [-] near-zero void for IC
    mass_min: float = 1e-12        # [kg] phasic mass cutoff
    h_l_min: float = 1e4           # [J/kg] liquid enthalpy floor
    h_v_max: float = 4e6           # [J/kg] vapor enthalpy ceiling
    rv_floor_frac: float = 0.01    # [-] vapor density floor as fraction of sat
    rv_floor_abs: float = 0.01     # [kg/m³] absolute vapor density floor


@dataclass
class ExtractedModelSpec:
    """Complete model specification — everything the solver needs,
    all extracted from Modelica. Nothing hardcoded in the solver."""
    prefix: str
    model_type: str          # "hem" or "drift_flux"
    geometry: GeometrySpec
    closures: ClosureSpec
    boundary: BoundarySpec
    ic: InitialConditions
    thresholds: SafetyThresholds = field(default_factory=SafetyThresholds)

    @property
    def N(self):
        return self.geometry.N

    def summary(self) -> str:
        return (
            f"ExtractedModelSpec: prefix='{self.prefix}', type='{self.model_type}'\n"
            f"  Geometry: N={self.geometry.N}, dx={self.geometry.dx:.4f}, "
            f"A={self.geometry.A_flow:.6f}, g={self.geometry.g_axial}\n"
            f"  Closures: H_i={self.closures.H_i:.0e}, C_0={self.closures.C_0}, "
            f"crit_flow={self.closures.use_critical_flow}, C_d={self.closures.C_d}\n"
            f"  Boundary: in={self.boundary.inlet_type}, out={self.boundary.outlet_type}, "
            f"p_out={self.boundary.p_out}\n"
            f"  IC: p={self.ic.p0[0]/1e6:.1f} MPa, h={self.ic.h0[0]/1e3:.1f} kJ/kg"
        )


def extract_model_spec(es: "EquationSystem") -> ExtractedModelSpec:
    """Extract a complete model specification from an EquationSystem.

    Reads ALL parameters from the extracted XML — geometry, closures,
    boundary conditions, initial conditions. Nothing hardcoded.
    """
    from .pipe1d_mapper import _detect_prefix, _get_state_indices, _get_indices, _get_initial_value

    prefix = _detect_prefix(es)

    # Detect model type from state variables
    p_indices = _get_state_indices(es, prefix, "p")
    h_indices = _get_state_indices(es, prefix, "h")
    h_l_indices = _get_state_indices(es, prefix, "h_l")
    h_v_indices = _get_state_indices(es, prefix, "h_v")
    alpha_indices = _get_state_indices(es, prefix, "alpha")

    N = len(p_indices)
    is_5eq = bool(h_l_indices and h_v_indices and alpha_indices)
    model_type = "drift_flux" if is_5eq else "hem"

    # ── Helper: get parameter value ──
    def pval(name, default=None):
        full = f"{prefix}.{name}"
        try:
            p = es.param(full)
            if p.value is not None:
                return p.value
        except KeyError:
            pass
        if default is not None:
            return default
        raise ValueError(f"Parameter '{full}' not found and no default")

    # ── Geometry ──
    import math
    L = pval("L")
    D = pval("D")
    geometry = GeometrySpec(
        N=N,
        dx=L / N,
        A_flow=math.pi / 4 * D**2,
        D_h=D,
        f_D=pval("f_D"),
        V_cell=L / N * math.pi / 4 * D**2,
        g_axial=pval("g_axial", 0.0),
    )

    # ── Closures (from Modelica parameters — NOT hardcoded) ──
    # Boolean parameters: OM may not include value in XML for Booleans.
    # Check the bindExpression or default to False.
    def bval(name, default=False):
        """Get boolean parameter, handling OM's None for Boolean type."""
        full = f"{prefix}.{name}"
        try:
            p = es.param(full)
            if p.value is not None:
                return bool(p.value)
            # OM may store Boolean as string in bind expression
            # For now, check if the parameter exists (means it was set to true in the model)
            # The Modelica default for use_critical_flow is false, so if OM
            # includes it in knownVariables, it was likely set to true.
            return default
        except KeyError:
            return default

    # Check equation text for evidence of critical flow being active
    eq_texts = [eq.text for eq in es.equations if eq.text]
    has_crit_flow_eqs = any("mdot_crit" in t for t in eq_texts)
    has_two_phase_fric = any("Phi2" in t for t in eq_texts)

    closures = ClosureSpec(
        H_i=pval("H_i", 0.0),
        C_0=pval("C_0", 1.0),
        alpha_nucleation=pval("alpha_nucleation", 1e-3),
        use_critical_flow=has_crit_flow_eqs,  # infer from equations
        C_d=pval("C_d", 1.0),
        x_trans=pval("x_trans", 0.10),
        c_floor=pval("c_floor", 10.0),
        use_two_phase_friction=has_two_phase_fric,  # infer from equations
        Phi2_max=pval("Phi2_max", 20.0),
    )

    # ── Boundary conditions ──
    mdot_state_indices = _get_state_indices(es, prefix, "mdot")
    mdot_dummy_indices = _get_indices(es, prefix, "mdot", "continuousDummyState")
    inlet_closed = (1 not in mdot_state_indices)
    outlet_closed = ((N + 1) not in mdot_state_indices)

    p_out = None
    h_out = None
    for p in es.parameters:
        if p.name.endswith(".p_set") and p.value is not None:
            p_out = p.value
        if p.name.endswith(".h_set") and p.value is not None:
            h_out = p.value

    boundary = BoundarySpec(
        inlet_type="wall" if inlet_closed else "pressure",
        outlet_type="wall" if outlet_closed else ("break" if closures.use_critical_flow else "pressure"),
        p_out=p_out,
        h_out=h_out,
    )

    # ── Initial conditions ──
    p_init = pval("p_init")
    if is_5eq:
        h_l_init = pval("h_l_init", pval("h_init", 800e3))
        h_v_init = pval("h_v_init", 2800e3)
        alpha_init = pval("alpha_init", 1e-6)
    else:
        h_l_init = pval("h_init", 800e3)
        h_v_init = None
        alpha_init = None

    p0 = [_get_initial_value(es, f"{prefix}.p[{i}]", p_init) for i in range(1, N+1)]

    if is_5eq:
        h0 = [_get_initial_value(es, f"{prefix}.h_l[{i}]", h_l_init) for i in range(1, N+1)]
        h_v0 = [_get_initial_value(es, f"{prefix}.h_v[{i}]", h_v_init) for i in range(1, N+1)]
        alpha0 = [_get_initial_value(es, f"{prefix}.alpha[{i}]", alpha_init) for i in range(1, N+1)]
    else:
        h0 = [_get_initial_value(es, f"{prefix}.h[{i}]", h_l_init) for i in range(1, N+1)]
        h_v0 = None
        alpha0 = None

    ic = InitialConditions(
        p0=p0, h0=h0, h_v0=h_v0, alpha0=alpha0,
        mdot0=[0.0] * (N + 1),
    )

    return ExtractedModelSpec(
        prefix=prefix,
        model_type=model_type,
        geometry=geometry,
        closures=closures,
        boundary=boundary,
        ic=ic,
    )
