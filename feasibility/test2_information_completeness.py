#!/usr/bin/env python3
"""
OPAL Feasibility Test 2 — Information Completeness
====================================================
Answers the questions left open by Test 1:

  A. Can we decode the adjacency (incidence) matrix into a human-readable table?
  B. Do when-equations and events survive into the XML?
  C. How does OM expand stream connectors / inStream()?
  D. Are variable names traceable to source components after alias elimination?

Run:
    ../external/venv/bin/python test2_information_completeness.py
"""

import re
import sys
from pathlib import Path

# Make sure the venv and utils are importable from any cwd.
sys.path.insert(0, str(Path(__file__).parent))

from extraction_utils import (
    RESULTS_DIR, MODELS_DIR,
    start_omc_session, omc_check,
    dump_xml_dae, parse_xml_report, decode_adjacency,
    get_variable_names, get_equation_texts,
    run_step, write_tabular_report,
)

# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("OPAL Feasibility Test 2 — Information Completeness")
    print("=" * 70)

    omc = start_omc_session(load_msl=False)
    ver = omc.sendExpression("getVersion()", parsed=False).strip().strip('"')
    print(f"  omc: {ver}")

    results  = []
    findings = []  # accumulated text findings for the report

    # =========================================================================
    # PART A — Incidence matrix decoding (reuse Test 1 backEnd XML)
    # =========================================================================
    print("\n[A] Decoding adjacency matrix from SimpleLoop backEnd XML …")

    backend_xml = RESULTS_DIR / "dumpXMLDAE_backEnd.xml"
    if not backend_xml.exists():
        print("  WARNING: run test_extraction.py first to generate backEnd XML")
    else:
        adj   = decode_adjacency(backend_xml)
        names = get_variable_names(backend_xml)
        eqs   = get_equation_texts(backend_xml)

        if adj["n_eq"] > 0:
            print(f"  Adjacency: {adj['n_eq']} eqs × {adj['n_var']} vars, "
                  f"sparsity = {adj['sparsity']:.3f}")

            # Print compact incidence table
            header = f"  {'Eq':<4}  " + "  ".join(f"v{i}" for i in range(1, adj["n_var"]+1))
            print(f"\n  Incidence table (+ = variable, d = derivative):")
            print(f"  {'Eq':<4}  {'Equation text':<45} " +
                  "  ".join(f"v{i:>2}" for i in range(1, adj["n_var"]+1)))
            print("  " + "-" * (52 + 5 * adj["n_var"]))
            for eq_id in sorted(adj["rows"].keys()):
                eq_txt = eqs.get(eq_id, "?")[:43]
                row = []
                for vi in range(1, adj["n_var"] + 1):
                    if vi in adj["ders"].get(eq_id, []):
                        row.append(" d")
                    elif vi in adj["rows"].get(eq_id, []):
                        row.append(" +")
                    else:
                        row.append("  ")
                print(f"  {eq_id:<4}  {eq_txt:<45} {'  '.join(row)}")

            # Variable legend
            print("\n  Variable legend:")
            for vid, name in sorted(names.items()):
                print(f"    v{vid}: {name}")

            findings.append(f"[A] Adjacency matrix decoded: "
                            f"{adj['n_eq']} eqs, {adj['n_var']} vars, "
                            f"sparsity={adj['sparsity']:.3f}, parseable=YES")
        else:
            findings.append("[A] Adjacency matrix: no rows found (run Test 1 first)")

    # =========================================================================
    # PART B — When-equation and event survival (WhenLoop model)
    # =========================================================================
    print("\n[B] Loading WhenLoop.mo …")
    model_file = MODELS_DIR / "WhenLoop.mo"
    omc.sendExpression(f'loadFile("{model_file.as_posix()}")', parsed=False)
    omc_check(omc, "loadFile WhenLoop")

    xml_levels = [
        ("flat",      "when_flat"),
        ("optimiser", "when_optimiser"),
        ("backEnd",   "when_backEnd"),
    ]
    for level, prefix in xml_levels:
        lbl = f"WhenLoop dumpXMLDAE({level})"
        print(f"\n  {lbl} …")
        def step_when(level=level, prefix=prefix):
            xml_path, info = dump_xml_dae(omc, "WhenLoop", level, prefix)
            print(f"    file: {xml_path.name} ({xml_path.stat().st_size:,} bytes)")
            print(f"    vars={info['n_ordered_vars']}  eqs={info['n_equations']}  "
                  f"states={info['n_states']}  ZC={info['n_zero_crossings']}  "
                  f"BLT={info['n_blt_blocks']}")

            # Check for discrete variables
            import xml.etree.ElementTree as ET
            root = ET.parse(xml_path).getroot()
            discrete_vars = [
                v.get("name") for v in root.findall(".//variable")
                if v.get("variability") == "discrete"
            ]
            info["discrete_vars"] = discrete_vars
            info["n_discrete"]    = len(discrete_vars)
            if discrete_vars:
                print(f"    discrete vars: {discrete_vars}")

            # Check for when-equation tags
            when_eqs = root.findall(".//whenEquation")
            info["n_when_equations"] = len(when_eqs)
            if when_eqs:
                print(f"    whenEquation elements: {len(when_eqs)}")

            return info

        results.append(run_step(lbl, step_when))

    # Summarise event findings
    when_back = next((r for r in results if "backEnd" in r["label"] and "When" in r["label"]), None)
    if when_back and when_back["status"] == "PASS":
        i = when_back["info"]
        findings.append(
            f"[B] WhenLoop backEnd: ZC={i.get('n_zero_crossings')}, "
            f"discrete={i.get('n_discrete')}, "
            f"whenEq={i.get('n_when_equations')}, "
            f"events_survived={'YES' if i.get('n_zero_crossings', 0) > 0 else 'NO'}"
        )

    # =========================================================================
    # PART C — Stream connector expansion (StreamLoop model)
    # =========================================================================
    print("\n[C] Loading StreamLoop.mo …")
    omc.sendExpression("clear()", parsed=False)
    omc_check(omc, "clear")
    model_file = MODELS_DIR / "StreamLoop.mo"
    omc.sendExpression(f'loadFile("{model_file.as_posix()}")', parsed=False)
    omc_check(omc, "loadFile StreamLoop")

    for level, prefix in [("flat", "stream_flat"), ("backEnd", "stream_backEnd")]:
        lbl = f"StreamLoop dumpXMLDAE({level})"
        print(f"\n  {lbl} …")
        def step_stream(level=level, prefix=prefix):
            xml_path, info = dump_xml_dae(omc, "StreamLoop", level, prefix)
            print(f"    file: {xml_path.name} ({xml_path.stat().st_size:,} bytes)")
            print(f"    vars(ord/kn/alias)={info['n_ordered_vars']}/"
                  f"{info['n_known_vars']}/{info['n_alias_vars']}  "
                  f"eqs={info['n_equations']}")

            # Check for inStream in equation text
            import xml.etree.ElementTree as ET
            root = ET.parse(xml_path).getroot()
            eq_texts = [eq.text or "" for eq in root.findall(".//equation")]
            instream_eqs = [t.strip() for t in eq_texts if "inStream" in t]
            info["instream_calls"] = len(instream_eqs)
            info["instream_sample"] = instream_eqs[:2]
            if instream_eqs:
                print(f"    inStream() calls found in equations: {len(instream_eqs)}")
                for s in instream_eqs[:2]:
                    print(f"      {s[:80]}")
            else:
                print(f"    inStream() NOT found in equations (expanded or eliminated)")

            # Check h_outflow in alias vs ordered
            alias_vars = [
                v.get("name", "") for v in root.findall(".//variable")
                if v.get("name", "").endswith("h_outflow")
                or "h_outflow" in v.get("name", "")
            ]
            info["h_outflow_count"] = len(alias_vars)
            info["h_outflow_sample"] = alias_vars[:4]
            print(f"    h_outflow variables: {alias_vars[:4]}")
            return info

        results.append(run_step(lbl, step_stream))

    stream_flat = next((r for r in results if "flat" in r["label"] and "Stream" in r["label"]), None)
    stream_back = next((r for r in results if "backEnd" in r["label"] and "Stream" in r["label"]), None)
    if stream_flat and stream_back and stream_flat["status"] == "PASS":
        fi = stream_flat["info"]
        bi = stream_back["info"]
        findings.append(
            f"[C] inStream() in flat XML: {fi.get('instream_calls', '?')} calls, "
            f"in backEnd: {bi.get('instream_calls', '?')} calls  "
            f"(0 = expanded, >0 = opaque call remaining)"
        )
        findings.append(
            f"[C] h_outflow vars visible: flat={fi.get('h_outflow_count', '?')}, "
            f"backEnd={bi.get('h_outflow_count', '?')}"
        )

    # =========================================================================
    # PART D — Component name traceability post-alias-elimination
    # =========================================================================
    print("\n[D] Checking component name traceability …")
    def step_traceability():
        backend_xml = RESULTS_DIR / "dumpXMLDAE_backEnd.xml"
        if not backend_xml.exists():
            raise FileNotFoundError("Run test_extraction.py first")

        import xml.etree.ElementTree as ET
        root = ET.parse(backend_xml).getroot()

        all_vars = root.findall(".//variable")
        traceable    = sum(1 for v in all_vars if "." in v.get("name", ""))
        total        = len(all_vars)
        pct          = 100 * traceable / total if total else 0

        # Categorise by section
        by_section: dict = {}
        for section in ["orderedVariables", "knownVariables", "aliasVariables"]:
            el = root.find(f".//{section}")
            if el is None:
                continue
            vars_in = el.findall(".//variable")
            tr = sum(1 for v in vars_in if "." in v.get("name", ""))
            by_section[section] = (tr, len(vars_in))

        print(f"  Overall: {traceable}/{total} ({pct:.0f}%) have component prefix")
        for sec, (tr, tot) in by_section.items():
            print(f"    {sec}: {tr}/{tot}")
        return {
            "traceable": traceable, "total": total, "pct_traceable": pct,
            "by_section": by_section,
        }

    r_trace = run_step("component_name_traceability", step_traceability)
    if r_trace["status"] == "PASS":
        i = r_trace["info"]
        findings.append(
            f"[D] Component name traceability: {i['traceable']}/{i['total']} "
            f"= {i['pct_traceable']:.0f}% have dot-prefix (FM5 assessment)"
        )

    # =========================================================================
    # Shut down and report
    # =========================================================================
    omc.sendExpression("quit()", parsed=False)

    extra = [
        "Findings:",
        *[f"  {f}" for f in findings],
        "",
        "Decision criteria:",
        "  FM1 (for-loop unrolling): tested in Test 5",
        "  FM2 (BLT scattering): see WhenLoop BLT block count",
        "  FM3 (opaque media): inStream() expansion shows mechanism",
        "  FM4 (scaling): tested in Tests 4 & 5",
        "  FM5 (name mangling): D check above",
    ]

    write_tabular_report(
        title     = "OPAL Feasibility Test 2 — Information Completeness",
        results   = results,
        omc_ver   = ver,
        model_desc= "WhenLoop (events) + StreamLoop (stream connectors)",
        extra_lines=extra,
        out_path  = RESULTS_DIR / "test2_report.txt",
    )

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
