"""
equation_classifier.py — Classify extracted equations by role for semi-implicit solving.

Takes an EquationSystem from xml_reader and identifies:
  - Mass conservation equations (for pressure linearisation)
  - Momentum equations (for velocity update)
  - Energy equations (for enthalpy update)
  - Property evaluation equations (rho, drho_dp, drho_dh, T)
  - Face density averaging equations
  - Donor-cell face enthalpy equations
  - Boundary constraint equations

Each classified equation carries the cell/face index it applies to and the
variable names involved, so the semi-implicit solver knows exactly what to
compute and in what order.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .equation_system import EquationSystem


@dataclass
class MassEq:
    """Mass conservation: V*(drho_dp*der(p) + drho_dh*der(h)) = mdot_in - mdot_out"""
    cell: int           # 1-based cell index
    eq_text: str


@dataclass
class MomentumEq:
    """Momentum: der(mdot[i]) = f(p_left, p_right, mdot, rho_face, friction)"""
    face: int           # 1-based face index (mdot index)
    eq_text: str


@dataclass
class EnergyEq:
    """Energy: rho*V*der(h[i]) = advection + pressure_work + q_wall"""
    cell: int           # 1-based cell index
    eq_text: str


@dataclass
class PropertyEq:
    """Property evaluation: var = Medium.func(p[i], h[i])"""
    cell: int           # 1-based cell index
    output_var: str     # e.g., "pipe.rho[2]"
    func_name: str      # e.g., "rho_ph"
    eq_text: str


@dataclass
class FaceDensityEq:
    """Face density: rho_face[i] = 0.5*(rho_L + rho_R) or rho_face[1] = rho[1]"""
    face: int           # 1-based face index
    eq_text: str


@dataclass
class DonorCellEq:
    """Donor-cell: h_face[i] = if mdot >= 0 then h_upwind else h_downwind"""
    face: int           # 1-based face index
    eq_text: str


@dataclass
class ConstraintEq:
    """Boundary constraint: e.g., closed_end.port.p = pipe.p[1]"""
    eq_text: str


@dataclass
class ClassifiedSystem:
    """All equations classified by role, ready for semi-implicit solving."""
    prefix: str                              # pipe instance name
    N: int                                   # number of cells
    mass_eqs: list[MassEq] = field(default_factory=list)
    momentum_eqs: list[MomentumEq] = field(default_factory=list)
    energy_eqs: list[EnergyEq] = field(default_factory=list)
    property_eqs: list[PropertyEq] = field(default_factory=list)
    face_density_eqs: list[FaceDensityEq] = field(default_factory=list)
    donor_cell_eqs: list[DonorCellEq] = field(default_factory=list)
    constraint_eqs: list[ConstraintEq] = field(default_factory=list)
    unclassified: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ClassifiedSystem: prefix='{self.prefix}', N={self.N}\n"
            f"  Mass:        {len(self.mass_eqs)}\n"
            f"  Momentum:    {len(self.momentum_eqs)}\n"
            f"  Energy:      {len(self.energy_eqs)}\n"
            f"  Property:    {len(self.property_eqs)}\n"
            f"  Face density:{len(self.face_density_eqs)}\n"
            f"  Donor-cell:  {len(self.donor_cell_eqs)}\n"
            f"  Constraint:  {len(self.constraint_eqs)}\n"
            f"  Unclassified:{len(self.unclassified)}"
        )


def classify_equations(es: "EquationSystem", prefix: str = "pipe") -> ClassifiedSystem:
    """
    Classify all equations in an extracted EquationSystem by their role
    in the semi-implicit solve.

    Classification uses regex on equation text — the same patterns validated
    in test_pipe1d_integration.py.
    """
    cs = ClassifiedSystem(prefix=prefix, N=0)

    # Detect N from state variables
    p_states = [v for v in es.states if re.match(rf'{prefix}\.p\[\d+\]', v.name)]
    cs.N = len(p_states)

    for eq in es.equations:
        text = eq.text.strip() if eq.text else ""
        if not text:
            continue

        classified = False

        # Mass conservation: contains drho_dp AND der(prefix.p[
        if f"drho_dp" in text and f"der({prefix}.p" in text:
            cell = _extract_cell_from_der_p(text, prefix)
            cs.mass_eqs.append(MassEq(cell=cell, eq_text=text))
            classified = True

        # Momentum: contains der(prefix.mdot[
        elif f"der({prefix}.mdot" in text or f"der({prefix}.m_flow" in text:
            face = _extract_face_from_der_mdot(text, prefix)
            cs.momentum_eqs.append(MomentumEq(face=face, eq_text=text))
            classified = True

        # Energy: contains der(prefix.h[ but NOT drho_dh (which is in mass eq)
        elif f"der({prefix}.h" in text and "drho_dh" not in text:
            cell = _extract_cell_from_der_h(text, prefix)
            cs.energy_eqs.append(EnergyEq(cell=cell, eq_text=text))
            classified = True

        # Property evaluation: contains Medium.rho_ph or Medium.drho_dp_h etc.
        elif "Medium." in text or "SimpleFluid." in text:
            cell, output_var, func_name = _extract_property_info(text, prefix)
            cs.property_eqs.append(PropertyEq(
                cell=cell, output_var=output_var,
                func_name=func_name, eq_text=text))
            classified = True

        # Face density averaging: rho_face AND "0.5 *"
        elif f"{prefix}.rho_face" in text and "0.5" in text:
            face = _extract_face_index(text, prefix, "rho_face")
            cs.face_density_eqs.append(FaceDensityEq(face=face, eq_text=text))
            classified = True

        # Face density boundary: rho_face = Medium.rho_ph (already caught by property)
        elif f"{prefix}.rho_face" in text and "rho_ph" in text:
            # This is a property eq that outputs to rho_face — already classified above
            pass

        # Donor-cell: h_face AND "if"
        elif f"{prefix}.h_face" in text and "if" in text:
            face = _extract_face_index(text, prefix, "h_face")
            cs.donor_cell_eqs.append(DonorCellEq(face=face, eq_text=text))
            classified = True

        # Boundary constraint: e.g., closed_end.port.p = prefix.p[1]
        elif "port.p" in text and f"{prefix}.p[" in text:
            cs.constraint_eqs.append(ConstraintEq(eq_text=text))
            classified = True

        if not classified:
            cs.unclassified.append(text)

    # Sort by index
    cs.mass_eqs.sort(key=lambda e: e.cell)
    cs.momentum_eqs.sort(key=lambda e: e.face)
    cs.energy_eqs.sort(key=lambda e: e.cell)
    cs.property_eqs.sort(key=lambda e: e.cell)
    cs.face_density_eqs.sort(key=lambda e: e.face)
    cs.donor_cell_eqs.sort(key=lambda e: e.face)

    return cs


# ---------------------------------------------------------------------------
# Helpers — extract cell/face indices from equation text
# ---------------------------------------------------------------------------

def _extract_cell_from_der_p(text: str, prefix: str) -> int:
    """Extract cell index from der(prefix.p[i])."""
    m = re.search(rf'der\({prefix}\.p\[(\d+)\]\)', text)
    return int(m.group(1)) if m else 0


def _extract_face_from_der_mdot(text: str, prefix: str) -> int:
    """Extract face index from der(prefix.mdot[i]) or der(prefix.m_flow[i])."""
    m = re.search(rf'der\({prefix}\.(mdot|m_flow)\[(\d+)\]\)', text)
    return int(m.group(2)) if m else 0


def _extract_cell_from_der_h(text: str, prefix: str) -> int:
    """Extract cell index from der(prefix.h[i])."""
    m = re.search(rf'der\({prefix}\.h\[(\d+)\]\)', text)
    return int(m.group(1)) if m else 0


def _extract_face_index(text: str, prefix: str, var: str) -> int:
    """Extract face index from prefix.var[i] on LHS."""
    m = re.search(rf'{prefix}\.{var}\[(\d+)\]', text)
    return int(m.group(1)) if m else 0


def _extract_property_info(text: str, prefix: str) -> tuple:
    """Extract cell index, output variable name, and function name from property eq."""
    # Output variable is on the LHS (before =)
    lhs = text.split("=")[0].strip()

    # Function name: last component of the qualified call
    func_match = re.search(r'\.(\w+)\(', text.split("=")[1]) if "=" in text else None
    func_name = func_match.group(1) if func_match else "unknown"

    # Cell index from the argument p[i]
    cell_match = re.search(rf'{prefix}\.p\[(\d+)\]', text)
    cell = int(cell_match.group(1)) if cell_match else 0

    return cell, lhs, func_name
