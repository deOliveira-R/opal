#!/usr/bin/env python3
"""
OPAL Feasibility Test 1 — Equation Extraction
==============================================
Exercises all OpenModelica extraction APIs on SimpleLoop.mo and
reports what each level exposes.  Intended to answer the question:
"Can we get enough information out of OM to drive a custom solver?"

Run with:
    ../external/venv/bin/python test_extraction.py

Results written to:  feasibility/results/
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (all relative to this file — works from any cwd)
# ---------------------------------------------------------------------------
FEASIBILITY_DIR = Path(__file__).parent.resolve()
OPAL_ROOT       = FEASIBILITY_DIR.parent
OM_HOME         = OPAL_ROOT / "external/OpenModelica/build_cmake/install_cmake"
VENV_PYTHON     = OPAL_ROOT / "external/venv/bin/python"
MODEL_FILE      = FEASIBILITY_DIR / "models/SimpleLoop.mo"
RESULTS_DIR     = FEASIBILITY_DIR / "results"

OM_BIN = OM_HOME / "bin" / "omc"

# ---------------------------------------------------------------------------
# Sanity checks before importing OMPython
# ---------------------------------------------------------------------------
if not OM_BIN.exists():
    sys.exit(f"ERROR: omc not found at {OM_BIN}\n"
             "       Build OpenModelica first: see external/OpenModelica/README.cmake.md")

if not MODEL_FILE.exists():
    sys.exit(f"ERROR: model file not found: {MODEL_FILE}")

RESULTS_DIR.mkdir(exist_ok=True)

# OMPython is in the venv next to this interpreter; add it to sys.path if needed.
venv_site = OPAL_ROOT / "external/venv/lib"
for p in venv_site.glob("python*/site-packages"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from OMPython import OMCSessionZMQ  # noqa: E402 (import after path setup)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def omc_check(omc: OMCSessionZMQ, step: str) -> str:
    """Return the error/warning string from OM and print if non-empty."""
    err = omc.sendExpression("getErrorString()", parsed=False).strip().strip('"')
    if err:
        print(f"  [OM messages @ {step}]: {err[:300]}")
    return err


def parse_dumpxml_return(raw: str) -> Path | None:
    """Extract the XML file path from dumpXMLDAE's (true, "path.xml") return."""
    m = re.search(r'"([^"]+\.xml)"', raw)
    if m:
        return Path(m.group(1))
    return None


def parse_xml_report(path: Path) -> dict:
    """Parse a dumpXMLDAE XML file and return key metrics."""
    info: dict = {
        "n_ordered_vars":   None,
        "n_known_vars":     None,
        "n_alias_vars":     None,
        "n_equations":      None,
        "n_states":         0,
        "n_dummy_states":   0,
        "n_dummy_ders":     0,
        "n_blt_blocks":     0,
        "n_loop_blocks":    0,   # blocks with > 1 equation (algebraic loops)
        "has_adjacency":    False,
        "has_matching":     False,
        "has_blt":          False,
        "n_zero_crossings": 0,
    }
    if not path.exists():
        return info

    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as e:
        info["parse_error"] = str(e)
        return info

    # Variable counts
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

    # State variable flavours
    for var in root.findall(".//variable"):
        v = var.get("variability", "")
        if v == "continuousState":
            info["n_states"] += 1
        elif v == "continuousDummyState":
            info["n_dummy_states"] += 1
        elif v == "continuousDummyDer":
            info["n_dummy_ders"] += 1

    # BLT structure
    # The XML nests two <bltRepresentation> tags; look for <bltBlock> anywhere.
    blt_blocks = root.findall(".//bltBlock")
    info["n_blt_blocks"] = len(blt_blocks)
    info["has_blt"] = len(blt_blocks) > 0
    info["n_loop_blocks"] = sum(
        1 for b in blt_blocks if len(list(b)) > 1
    )

    # Matching and adjacency
    info["has_matching"]  = root.find(".//matchingAlgorithm") is not None
    # OM uses <originalAdjacencyMatrix> with MathML-namespaced <matrixrow> children
    MML = "http://www.w3.org/1998/Math/MathML"
    adj_el = root.find(".//originalAdjacencyMatrix")
    info["has_adjacency"] = (
        adj_el is not None
        and adj_el.find(f".//{{{MML}}}matrix") is not None
    )

    # Zero crossings / events
    zc = root.find(".//zeroCrossingList")
    if zc is not None:
        info["n_zero_crossings"] = int(zc.get("dimension", 0))

    return info


def run_step(label: str, fn) -> dict:
    """Run one extraction step, catch exceptions, return result dict."""
    result = {"label": label, "status": "FAIL", "error": None, "info": {}}
    try:
        info = fn()
        result["status"] = "PASS"
        result["info"]   = info or {}
    except Exception as exc:
        result["error"] = str(exc)
        print(f"  EXCEPTION in {label}: {exc}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("OPAL Feasibility Test 1 — Equation Extraction")
    print(f"  omc    : {OM_HOME}/bin/omc")
    print(f"  model  : {MODEL_FILE}")
    print(f"  results: {RESULTS_DIR}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Start OMC session
    # ------------------------------------------------------------------
    print("\n[0] Starting OMC session …")
    omc = OMCSessionZMQ(omhome=str(OM_HOME))
    ver = omc.sendExpression("getVersion()", parsed=False).strip().strip('"')
    print(f"  omc version: {ver}")
    omc_check(omc, "startup")

    # Set OMC working directory so file-writing APIs land in results/
    omc.sendExpression(f'cd("{RESULTS_DIR.as_posix()}")', parsed=False)
    omc_check(omc, "cd")

    # ------------------------------------------------------------------
    # Load model (no loadModel(Modelica) — self-contained)
    # ------------------------------------------------------------------
    print("\n[1] Loading SimpleLoop.mo …")
    loaded = omc.sendExpression(f'loadFile("{MODEL_FILE.as_posix()}")', parsed=False)
    errors = omc_check(omc, "loadFile")
    if "true" not in loaded.lower():
        print(f"  ERROR: loadFile returned: {loaded}")
        omc.sendExpression("quit()")
        return 1
    print(f"  loadFile: {loaded.strip()}")

    # Confirm the model is there
    classes = omc.sendExpression("getClassNames()", parsed=False)
    print(f"  top-level classes: {classes.strip()}")

    results = []

    # ------------------------------------------------------------------
    # Step A: instantiateModel
    # ------------------------------------------------------------------
    print("\n[A] instantiateModel …")
    def step_instantiate():
        raw = omc.sendExpression("instantiateModel(SimpleLoop)", parsed=False)
        omc_check(omc, "instantiateModel")
        # Strip surrounding quotes that OMC adds to string results
        flat = raw.strip().strip('"').replace('\\"', '"').replace('\\n', '\n')
        out_path = RESULTS_DIR / "instantiateModel.mo"
        out_path.write_text(flat)
        print(f"  wrote {out_path.name} ({out_path.stat().st_size} bytes)")
        lines = flat.splitlines()
        n_real_vars  = sum(1 for l in lines if re.search(r'\bReal\b', l))
        n_der        = sum(1 for l in lines if 'der(' in l)
        n_connect    = sum(1 for l in lines if 'connect(' in l)
        n_eq_lines   = sum(1 for l in lines if l.strip().endswith(';') and '=' in l)
        info = {
            "lines":      len(lines),
            "Real_decls": n_real_vars,
            "der_calls":  n_der,
            "connects":   n_connect,
            "eq_lines":   n_eq_lines,
        }
        print(f"  lines={info['lines']}  Real={info['Real_decls']}  "
              f"der()={info['der_calls']}  connect()={info['connects']}")
        return info

    results.append(run_step("instantiateModel", step_instantiate))

    # ------------------------------------------------------------------
    # Steps B-E: dumpXMLDAE at each translation level
    # ------------------------------------------------------------------
    xml_levels = [
        ("flat",       "dumpXMLDAE_flat"),
        ("optimiser",  "dumpXMLDAE_optimiser"),
        ("backEnd",    "dumpXMLDAE_backEnd"),
        ("stateSpace", "dumpXMLDAE_stateSpace"),
    ]

    for level, prefix in xml_levels:
        lbl = f"dumpXMLDAE({level})"
        print(f"\n[B-E] {lbl} …")

        def step_dump(level=level, prefix=prefix):
            expr = (
                f'dumpXMLDAE(SimpleLoop, '
                f'translationLevel="{level}", '
                f'addOriginalAdjacencyMatrix=true, '
                f'addSolvingInfo=true, '
                f'addMathMLCode=false, '
                f'dumpResiduals=false, '
                f'fileNamePrefix="{prefix}")'
            )
            raw  = omc.sendExpression(expr, parsed=False)
            omc_check(omc, lbl)

            # dumpXMLDAE writes the file to omc's cwd; find it
            xml_path = parse_dumpxml_return(raw)
            if xml_path is None:
                # Fallback: look for the file directly
                xml_path = RESULTS_DIR / f"{prefix}.xml"
            if not xml_path.is_absolute():
                xml_path = RESULTS_DIR / xml_path.name
            if not xml_path.exists():
                raise FileNotFoundError(
                    f"XML file not found (raw return: {raw!r})"
                )

            print(f"  file: {xml_path.name} ({xml_path.stat().st_size:,} bytes)")
            info = parse_xml_report(xml_path)
            print(f"  vars(ordered/known/alias): "
                  f"{info['n_ordered_vars']}/{info['n_known_vars']}/{info['n_alias_vars']}  "
                  f"eqs: {info['n_equations']}  states: {info['n_states']}  "
                  f"dummyDer: {info['n_dummy_ders']}  "
                  f"BLT blocks: {info['n_blt_blocks']}  loops: {info['n_loop_blocks']}  "
                  f"adjacency: {info['has_adjacency']}  matching: {info['has_matching']}")
            return info

        results.append(run_step(lbl, step_dump))

    # ------------------------------------------------------------------
    # Step F: translateModelXML
    # ------------------------------------------------------------------
    print("\n[F] translateModelXML …")
    def step_translate():
        raw = omc.sendExpression(
            'translateModelXML(SimpleLoop, fileNamePrefix="translateModelXML")',
            parsed=False,
        )
        omc_check(omc, "translateModelXML")
        # Returns the filename on success, or empty string on failure
        xml_path = None
        m = re.search(r'"([^"]+)"', raw)
        if m:
            candidate = Path(m.group(1))
            if not candidate.is_absolute():
                candidate = RESULTS_DIR / candidate.name
            if candidate.exists():
                xml_path = candidate

        if xml_path is None:
            # OM writes {ClassName}.xml regardless of prefix — try that
            fallback = RESULTS_DIR / "SimpleLoop.xml"
            if fallback.exists():
                xml_path = fallback
            else:
                raise FileNotFoundError(
                    f"translateModelXML produced no file (raw: {raw!r})"
                )

        size = xml_path.stat().st_size
        print(f"  file: {xml_path.name} ({size:,} bytes)")
        # Quick structural check
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            tag  = root.tag
        except ET.ParseError:
            tag = "(parse error)"
        print(f"  root element: {tag}")
        return {"file": xml_path.name, "size_bytes": size, "root_tag": tag}

    results.append(run_step("translateModelXML", step_translate))

    # ------------------------------------------------------------------
    # Shut down OMC
    # ------------------------------------------------------------------
    omc.sendExpression("quit()", parsed=False)

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    write_report(results, ver, RESULTS_DIR)

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n{'='*70}")
    print(f"Result: {len(results) - n_fail}/{len(results)} steps passed.")
    if n_fail:
        print(f"FAILED steps: {[r['label'] for r in results if r['status'] == 'FAIL']}")

    return 0 if n_fail == 0 else 1


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(results: list, omc_ver: str, out_dir: Path):
    lines = []
    lines.append("OPAL Feasibility Test 1 — Equation Extraction Report")
    lines.append(f"Run  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"omc  : {omc_ver}")
    lines.append(f"Model: SimpleLoop (pump + compressible pipe + heat sink, isothermal loop)")
    lines.append("")

    hdr = (
        f"{'Step':<28} {'Status':^6}  {'Vars':>5}  {'Eqs':>5}  "
        f"{'States':>6}  {'DumDer':>6}  {'BLT':>5}  {'Loops':>5}  "
        f"{'ZC':>4}  {'Match':^5}  {'Adj':^5}"
    )
    sep = "-" * len(hdr)
    lines.append(hdr)
    lines.append(sep)

    for r in results:
        i    = r["info"]
        st   = r["status"]
        name = r["label"]

        if "dumpXMLDAE" in name or name == "instantiateModel":
            vars_  = i.get("n_ordered_vars", i.get("Real_decls", "?"))
            eqs    = i.get("n_equations",    i.get("eq_lines",   "?"))
            states = i.get("n_states",       "?")
            ddder  = i.get("n_dummy_ders",   "?")
            blt    = i.get("n_blt_blocks",   "?")
            loops  = i.get("n_loop_blocks",  "?")
            zc     = i.get("n_zero_crossings","?")
            match_ = "yes" if i.get("has_matching")  else "no"
            adj    = "yes" if i.get("has_adjacency") else "no"
            if name == "instantiateModel":
                match_ = "-"; adj = "-"; states = "-" if not i.get("n_states") else states
        else:
            # translateModelXML
            vars_ = eqs = states = ddder = blt = loops = zc = match_ = adj = "-"

        err = f" ← {r['error'][:60]}" if r["error"] else ""
        lines.append(
            f"{name:<28} {st:^6}  {str(vars_):>5}  {str(eqs):>5}  "
            f"{str(states):>6}  {str(ddder):>6}  {str(blt):>5}  {str(loops):>5}  "
            f"{str(zc):>4}  {match_:^5}  {adj:^5}{err}"
        )

    lines.append(sep)
    lines.append("")
    lines.append("Decision criteria (Test 1):")
    lines.append("  ODE/DAE reconstructable : states + matching present at backEnd level?")
    lines.append("  Jacobian sparsity        : adjacencyMatrix present?")
    lines.append("  BLT structure            : bltRepresentation present at optimiser/backEnd?")
    lines.append("  Events preserved         : zero crossings in output? (none expected here)")
    lines.append("  Component traceability   : variable names retain pump./pipe./sink. prefix?")
    lines.append("  stream semantics         : N/A — no stream connectors in this model")
    lines.append("  Media calls              : N/A — no media library, pure equations visible")
    lines.append("")
    lines.append("Files written:")
    for f in sorted(out_dir.glob("*.xml")) + sorted(out_dir.glob("*.mo")):
        lines.append(f"  {f.name}  ({f.stat().st_size:,} bytes)")

    report_text = "\n".join(lines) + "\n"
    report_path = out_dir / "report.txt"
    report_path.write_text(report_text)
    print(f"\n{'='*70}")
    print(report_text)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
