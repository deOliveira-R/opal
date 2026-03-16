"""
extraction_utils.py — Shared helpers for OPAL feasibility tests.

All test scripts import from here to avoid duplicating OMC session setup,
path logic, XML parsing, and report formatting.
"""

import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canonical paths
# ---------------------------------------------------------------------------
FEASIBILITY_DIR = Path(__file__).parent.resolve()
OPAL_ROOT       = FEASIBILITY_DIR.parent
OM_HOME         = OPAL_ROOT / "external/OpenModelica/build_cmake/install_cmake"
OM_BIN          = OM_HOME / "bin" / "omc"
MODELS_DIR      = FEASIBILITY_DIR / "models"
RESULTS_DIR     = FEASIBILITY_DIR / "results"

# MathML namespace used in adjacency matrix output
MML = "http://www.w3.org/1998/Math/MathML"


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _ensure_venv():
    """Add the venv site-packages to sys.path if OMPython isn't importable."""
    venv_site = OPAL_ROOT / "external/venv/lib"
    for p in venv_site.glob("python*/site-packages"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def start_omc_session(load_msl: bool = False):
    """
    Create and return a ready OMCSessionZMQ.

    Sets omc working directory to RESULTS_DIR so file-writing APIs
    (dumpXMLDAE, translateModelXML) land there automatically.

    If load_msl=True, loads Modelica 4.x from ~/.openmodelica/libraries.
    """
    if not OM_BIN.exists():
        sys.exit(f"ERROR: omc not found at {OM_BIN}\n"
                 "       Build OpenModelica first.")

    RESULTS_DIR.mkdir(exist_ok=True)
    _ensure_venv()

    from OMPython import OMCSessionZMQ  # noqa: E402
    omc = OMCSessionZMQ(omhome=str(OM_HOME))

    # Warmup (avoids timing the first JIT hit)
    omc.sendExpression("getVersion()", parsed=False)

    # Set working directory
    omc.sendExpression(f'cd("{RESULTS_DIR.as_posix()}")', parsed=False)
    omc_check(omc, "cd")

    if load_msl:
        t0 = time.perf_counter()
        r  = omc.sendExpression("loadModel(Modelica)", parsed=False)
        dt = time.perf_counter() - t0
        err = omc_check(omc, "loadModel(Modelica)")
        if "true" not in r.lower():
            sys.exit(f"ERROR: loadModel(Modelica) failed: {r}\n{err}")
        print(f"  loadModel(Modelica): {dt:.1f}s")

    return omc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def omc_check(omc, step: str) -> str:
    """Retrieve and print OM error/warning messages. Returns the raw string."""
    err = omc.sendExpression("getErrorString()", parsed=False).strip().strip('"')
    if err:
        print(f"  [OM messages @ {step}]: {err[:400]}")
    return err


def parse_dumpxml_return(raw: str) -> Path | None:
    """Extract the XML file path from dumpXMLDAE's '(true, "path.xml")' return."""
    m = re.search(r'"([^"]+\.xml)"', raw)
    return Path(m.group(1)) if m else None


def resolve_xml_path(raw: str, prefix: str, fallback_names: list[str] = None) -> Path:
    """
    Resolve the output XML file from a dumpXMLDAE return value.

    Tries:
      1. Path extracted from the return string
      2. RESULTS_DIR / f"{prefix}.xml"
      3. Each name in fallback_names (as RESULTS_DIR / name)
    Raises FileNotFoundError if none exist.
    """
    candidates = []
    p = parse_dumpxml_return(raw)
    if p is not None:
        candidates.append(p if p.is_absolute() else RESULTS_DIR / p.name)
    candidates.append(RESULTS_DIR / f"{prefix}.xml")
    for name in (fallback_names or []):
        candidates.append(RESULTS_DIR / name)

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"XML not found for prefix '{prefix}' (raw: {raw!r})"
    )


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_xml_report(path: Path) -> dict:
    """
    Parse a dumpXMLDAE XML file and return a metrics dict.

    Keys:
        n_ordered_vars, n_known_vars, n_alias_vars, n_equations,
        n_states, n_dummy_states, n_dummy_ders,
        n_blt_blocks, n_loop_blocks, has_adjacency, has_matching,
        n_zero_crossings, has_blt, parse_error (if any)
    """
    info: dict[str, Any] = {
        "n_ordered_vars":   None,
        "n_known_vars":     None,
        "n_alias_vars":     None,
        "n_equations":      None,
        "n_states":         0,
        "n_dummy_states":   0,
        "n_dummy_ders":     0,
        "n_blt_blocks":     0,
        "n_loop_blocks":    0,
        "has_adjacency":    False,
        "has_matching":     False,
        "has_blt":          False,
        "n_zero_crossings": 0,
    }
    if not path.exists():
        info["parse_error"] = "file not found"
        return info

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as e:
        info["parse_error"] = str(e)
        return info

    # Variable counts from dimension attributes
    for key, tag in [
        ("n_ordered_vars", "orderedVariables"),
        ("n_known_vars",   "knownVariables"),
        ("n_alias_vars",   "aliasVariables"),
    ]:
        el = root.find(f".//{tag}")
        if el is not None:
            info[key] = int(el.get("dimension", 0))

    # Equation count
    eqs = root.find(".//equations")
    if eqs is not None:
        info["n_equations"] = int(eqs.get("dimension", 0))

    # State variable flavours (count by variability attribute)
    for var in root.findall(".//variable"):
        v = var.get("variability", "")
        if v == "continuousState":
            info["n_states"] += 1
        elif v == "continuousDummyState":
            info["n_dummy_states"] += 1
        elif v == "continuousDummyDer":
            info["n_dummy_ders"] += 1

    # BLT blocks (nested <bltRepresentation> → <bltBlock>)
    blt_blocks = root.findall(".//bltBlock")
    info["n_blt_blocks"] = len(blt_blocks)
    info["has_blt"]      = len(blt_blocks) > 0
    info["n_loop_blocks"] = sum(1 for b in blt_blocks if len(list(b)) > 1)

    # Matching and adjacency
    info["has_matching"]  = root.find(".//matchingAlgorithm") is not None
    adj_el = root.find(".//originalAdjacencyMatrix")
    info["has_adjacency"] = (
        adj_el is not None
        and adj_el.find(f".//{{{MML}}}matrix") is not None
    )

    # Events
    zc = root.find(".//zeroCrossingList")
    if zc is not None:
        info["n_zero_crossings"] = int(zc.get("dimension", 0))

    return info


def decode_adjacency(path: Path) -> dict:
    """
    Parse the MathML adjacency (incidence) matrix from a dumpXMLDAE XML.

    Returns:
        {
          "rows": { eq_id: [var_ids] },       # positive → appears normally
          "ders": { eq_id: [var_ids] },       # negative in MathML → derivative
          "sparsity": float,                  # nnz / (n_eq * n_var)
          "n_eq": int,
          "n_var": int,
        }
    """
    result = {"rows": {}, "ders": {}, "sparsity": None, "n_eq": 0, "n_var": 0}
    if not path.exists():
        return result

    tree = ET.parse(path)
    root = tree.getroot()

    # Variable count (denominator for sparsity)
    ov = root.find(".//orderedVariables")
    n_var = int(ov.get("dimension", 0)) if ov is not None else 0

    adj = root.find(".//originalAdjacencyMatrix")
    if adj is None:
        return result

    rows = {}
    ders = {}
    for row in adj.findall(f".//{{{MML}}}matrixrow"):
        eq_id = int(row.get("id", 0))
        pos_vars, neg_vars = [], []
        for ci in row.findall(f"{{{MML}}}ci"):
            v = ci.text.strip() if ci.text else "0"
            val = int(v)
            if val > 0:
                pos_vars.append(val)
            elif val < 0:
                neg_vars.append(-val)
        rows[eq_id] = pos_vars
        ders[eq_id] = neg_vars

    n_eq  = len(rows)
    nnz   = sum(len(v) + len(d) for v, d in zip(rows.values(), ders.values()))
    sparsity = nnz / (n_eq * n_var) if n_eq > 0 and n_var > 0 else None

    result.update({"rows": rows, "ders": ders,
                   "sparsity": sparsity, "n_eq": n_eq, "n_var": n_var})
    return result


def get_variable_names(path: Path, section: str = "orderedVariables") -> dict[int, str]:
    """
    Return {variable_id: name} from a dumpXMLDAE XML.

    section: 'orderedVariables' (default), 'knownVariables', 'aliasVariables', or None (all)
    """
    if not path.exists():
        return {}
    tree = ET.parse(path)
    root = tree.getroot()
    names = {}
    scope = root.find(f".//{section}") if section else root
    if scope is None:
        return names
    for var in scope.findall(".//variable"):
        vid  = var.get("id")
        name = var.get("name")
        if vid and name:
            names[int(vid)] = name
    return names


def get_equation_texts(path: Path) -> dict[int, str]:
    """Return {equation_id: text} from a dumpXMLDAE XML."""
    if not path.exists():
        return {}
    tree = ET.parse(path)
    root = tree.getroot()
    eqs = {}
    for eq in root.findall(".//equation"):
        eid  = eq.get("id")
        text = (eq.text or "").strip()
        if eid:
            eqs[int(eid)] = text
    return eqs


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(label: str, fn) -> dict:
    """Run one extraction step, catch exceptions, return result dict."""
    result = {"label": label, "status": "FAIL", "error": None, "info": {}, "time_s": None}
    t0 = time.perf_counter()
    try:
        info = fn()
        result["status"] = "PASS"
        result["info"]   = info or {}
    except Exception as exc:
        result["error"] = str(exc)
        print(f"  EXCEPTION in {label}: {exc}")
    result["time_s"] = time.perf_counter() - t0
    return result


# ---------------------------------------------------------------------------
# dumpXMLDAE convenience wrapper
# ---------------------------------------------------------------------------

def dump_xml_dae(omc, model: str, level: str, prefix: str,
                 fallback_names: list[str] = None) -> tuple[Path, dict]:
    """
    Call dumpXMLDAE at the given translation level and return (xml_path, info).
    Raises on failure.
    """
    expr = (
        f'dumpXMLDAE({model}, '
        f'translationLevel="{level}", '
        f'addOriginalAdjacencyMatrix=true, '
        f'addSolvingInfo=true, '
        f'addMathMLCode=false, '
        f'dumpResiduals=false, '
        f'fileNamePrefix="{prefix}")'
    )
    raw = omc.sendExpression(expr, parsed=False)
    omc_check(omc, f"dumpXMLDAE({level})")
    xml_path = resolve_xml_path(raw, prefix, fallback_names)
    info = parse_xml_report(xml_path)
    return xml_path, info


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_tabular_report(
    title: str,
    results: list[dict],
    omc_ver: str,
    model_desc: str,
    extra_lines: list[str],
    out_path: Path,
):
    """Write a standardised tabular report to out_path and stdout."""
    from datetime import datetime
    lines = []
    lines.append(title)
    lines.append(f"Run  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"omc  : {omc_ver}")
    lines.append(f"Model: {model_desc}")
    lines.append("")

    hdr = (
        f"{'Step':<32} {'Status':^6}  {'Vars':>5}  {'Eqs':>5}  "
        f"{'States':>6}  {'BLT':>5}  {'Loops':>5}  "
        f"{'ZC':>4}  {'Match':^5}  {'Adj':^5}  {'Time(s)':>7}"
    )
    sep = "-" * len(hdr)
    lines += [hdr, sep]

    for r in results:
        i    = r["info"]
        st   = r["status"]
        name = r["label"]
        vars_ = i.get("n_ordered_vars", "?")
        eqs   = i.get("n_equations",    "?")
        states= i.get("n_states",       "?")
        blt   = i.get("n_blt_blocks",   "?")
        loops = i.get("n_loop_blocks",  "?")
        zc    = i.get("n_zero_crossings","?")
        match_= "yes" if i.get("has_matching")  else ("?" if vars_ == "?" else "no")
        adj   = "yes" if i.get("has_adjacency") else ("?" if vars_ == "?" else "no")
        t     = f"{r['time_s']:.1f}" if r.get("time_s") is not None else "?"
        err   = f" ← {r['error'][:55]}" if r.get("error") else ""
        lines.append(
            f"{name:<32} {st:^6}  {str(vars_):>5}  {str(eqs):>5}  "
            f"{str(states):>6}  {str(blt):>5}  {str(loops):>5}  "
            f"{str(zc):>4}  {match_:^5}  {adj:^5}  {t:>7}{err}"
        )

    lines += [sep, ""]
    lines += extra_lines

    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    print(f"\n{'='*70}\n{text}\nReport → {out_path}")
    return text
