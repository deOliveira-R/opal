"""
xml_reader.py — Parse dumpXMLDAE backEnd XML into an EquationSystem.

Usage:
    from opal.solver.partitioner.xml_reader import load_equation_system
    sys = load_equation_system(Path("scale_N5_backEnd.xml"))
"""

from __future__ import annotations
import xml.etree.ElementTree as ET
from pathlib import Path

from .equation_system import EquationSystem, Variable, Parameter, Equation

# MathML namespace used in adjacency matrix
MML = "http://www.w3.org/1998/Math/MathML"


def load_equation_system(xml_path: Path) -> EquationSystem:
    """Parse a dumpXMLDAE backEnd XML file and return an EquationSystem."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    sys = EquationSystem()
    _parse_variables(root, sys)
    _parse_parameters(root, sys)
    _parse_equations(root, sys)
    _parse_adjacency(root, sys)
    _parse_matching_and_blt(root, sys)
    sys.build_indexes()
    return sys


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------

def _parse_variables(root: ET.Element, sys: EquationSystem) -> None:
    ordered = root.find(".//orderedVariables")
    if ordered is None:
        return
    for var_el in ordered.findall(".//variable"):
        vid   = int(var_el.get("id", 0))
        name  = var_el.get("name", "")
        vari  = var_el.get("variability", "continuous")
        fixed = var_el.get("fixed", "false").lower() == "true"

        init_val = None
        iv_el = var_el.find(".//initialValue")
        if iv_el is not None and iv_el.get("string"):
            try:
                init_val = float(iv_el.get("string"))
            except ValueError:
                pass

        sys.variables.append(Variable(
            id=vid, name=name, variability=vari,
            initial_value=init_val, fixed=fixed,
        ))


def _parse_parameters(root: ET.Element, sys: EquationSystem) -> None:
    known = root.find(".//knownVariables")
    if known is None:
        return
    for var_el in known.findall(".//variable"):
        pid   = int(var_el.get("id", 0))
        name  = var_el.get("name", "")
        type_ = var_el.get("type", "Real")

        value = None
        be_el = var_el.find(".//bindExpression")
        if be_el is not None and be_el.get("string"):
            try:
                value = float(be_el.get("string"))
            except ValueError:
                pass

        sys.parameters.append(Parameter(id=pid, name=name, value=value, type_=type_))


def _parse_equations(root: ET.Element, sys: EquationSystem) -> None:
    eqs_el = root.find(".//equations")
    if eqs_el is None:
        return
    for eq_el in eqs_el.findall("equation"):
        eid  = int(eq_el.get("id", 0))
        text = (eq_el.text or "").strip()
        sys.equations.append(Equation(id=eid, text=text))


def _parse_adjacency(root: ET.Element, sys: EquationSystem) -> None:
    adj_el = root.find(".//originalAdjacencyMatrix")
    if adj_el is None:
        return
    for row in adj_el.findall(f".//{{{MML}}}matrixrow"):
        eq_id = int(row.get("id", 0))
        var_ids = []
        for ci in row.findall(f"{{{MML}}}ci"):
            raw = (ci.text or "").strip()
            try:
                # Negative values indicate derivative appears; store abs value
                var_ids.append(abs(int(raw)))
            except ValueError:
                pass
        sys.adjacency[eq_id] = var_ids


def _parse_matching_and_blt(root: ET.Element, sys: EquationSystem) -> None:
    # Matching: varId solved by equationId
    for el in root.findall(".//solvedIn"):
        var_id = int(el.get("variableId", 0))
        eq_id  = int(el.get("equationId", 0))
        sys.matching[var_id] = eq_id

    # BLT evaluation order (list of equation ids, in order)
    # The bltRepresentation is nested in the XML (inner one has the blocks)
    for blt_block in root.findall(".//bltBlock"):
        for inv_eq in blt_block.findall("involvedEquation"):
            eq_id = int(inv_eq.get("equationId", 0))
            sys.blt_order.append(eq_id)
