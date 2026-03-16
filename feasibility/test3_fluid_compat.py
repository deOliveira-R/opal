#!/usr/bin/env python3
"""
OPAL Feasibility Test 3 — Modelica.Fluid Compatibility
=======================================================
Tests whether OM's extraction pipeline survives the full MSL Fluid stack.

Three sub-models (increasing complexity):
  FluidPipeLoopStatic   — StaticPipe + SteadyState + StandardWaterOnePhase
  FluidPipeLoopDynamic  — DynamicPipe (3 nodes) + FixedInitial + StandardWaterOnePhase
  FluidPipeLoopTwophase — StaticPipe + SteadyState + StandardWater (ph, two-phase)

Key questions (FM3 assessment):
  1. Does translation succeed at all?
  2. Are IAPWS-IF97 calls opaque (external "C") or inlined as equations?
  3. How bad is the equation count explosion from media inlining?
  4. Are stream connectors (h_outflow) properly eliminated?

Run:
    ../external/venv/bin/python test3_fluid_compat.py
"""

import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extraction_utils import (
    RESULTS_DIR, MODELS_DIR,
    start_omc_session, omc_check, dump_xml_dae,
    parse_xml_report, run_step, write_tabular_report,
)

SUB_MODELS = [
    ("FluidPipeLoopStatic",   "fluid_static"),
    ("FluidPipeLoopDynamic",  "fluid_dynamic"),
    ("FluidPipeLoopTwophase", "fluid_twophase"),
]


def check_for_external_calls(xml_path: Path) -> dict:
    """
    Scan equation text for signs of opaque external function calls vs.
    inlined polynomial arithmetic.

    Returns:
        external_func_calls: list of equation texts containing 'external' or
                             known IF97 function names
        if97_names_found: list of IF97 function name substrings detected
        total_equations: int
    """
    if not xml_path.exists():
        return {"external_func_calls": [], "if97_names_found": [], "total_equations": 0}

    root = ET.parse(xml_path).getroot()
    eq_texts = [eq.text or "" for eq in root.findall(".//equation")]

    if97_signatures = [
        "waterBaseProp", "IF97_Utilities", "rho_ph", "T_ph", "h_pT",
        "specificEnthalpy", "density_ph", "temperature_ph", "cp_ph",
        "setState_phX", "setState_pTX", "REGION", "if97", "IF97",
    ]

    ext_calls = []
    if97_found = set()
    for t in eq_texts:
        for sig in if97_signatures:
            if sig in t:
                if97_found.add(sig)
                ext_calls.append(t.strip()[:120])
                break

    return {
        "external_func_calls": ext_calls[:5],
        "if97_names_found":    list(if97_found),
        "total_equations":     len(eq_texts),
    }


def main() -> int:
    print("=" * 70)
    print("OPAL Feasibility Test 3 — Modelica.Fluid Compatibility")
    print("=" * 70)

    print("\n[0] Starting OMC session and loading MSL …")
    omc = start_omc_session(load_msl=True)
    ver = omc.sendExpression("getVersion()", parsed=False).strip().strip('"')
    print(f"  omc: {ver}")

    model_file = MODELS_DIR / "FluidPipeLoop.mo"
    t0 = time.perf_counter()
    omc.sendExpression(f'loadFile("{model_file.as_posix()}")', parsed=False)
    err = omc_check(omc, "loadFile FluidPipeLoop")
    print(f"  loadFile: {time.perf_counter()-t0:.2f}s")

    results  = []
    findings = []

    # Baseline for comparison
    baseline = {"n_ordered_vars": 5, "n_equations": 5, "n_blt_blocks": 2}
    findings.append(f"Baseline (SimpleLoop, Test 1): "
                    f"vars={baseline['n_ordered_vars']}  "
                    f"eqs={baseline['n_equations']}  "
                    f"BLT={baseline['n_blt_blocks']}")

    for model_name, prefix in SUB_MODELS:
        print(f"\n[?] {model_name} …")

        def step(model_name=model_name, prefix=prefix):
            # Attempt dumpXMLDAE at backEnd level
            t_start = time.perf_counter()
            try:
                xml_path, info = dump_xml_dae(
                    omc, model_name, "backEnd",
                    f"{prefix}_backEnd",
                    fallback_names=[f"{model_name}.xml"]
                )
            except Exception as e:
                # Also try flat level to see if the failure is in backEnd only
                try:
                    xml_path, info = dump_xml_dae(
                        omc, model_name, "flat",
                        f"{prefix}_flat",
                        fallback_names=[f"{model_name}.xml"]
                    )
                    info["extraction_level"] = "flat_only (backEnd failed)"
                except Exception as e2:
                    raise RuntimeError(
                        f"Both flat and backEnd failed: {e} / {e2}"
                    ) from e2

            dt = time.perf_counter() - t_start
            info["extraction_time_s"] = dt

            print(f"  vars(ord/kn/alias): "
                  f"{info['n_ordered_vars']}/{info['n_known_vars']}/{info['n_alias_vars']}")
            print(f"  eqs: {info['n_equations']}  states: {info['n_states']}  "
                  f"BLT: {info['n_blt_blocks']}  time: {dt:.1f}s")

            # FM3 check: are IF97 calls opaque or inlined?
            fm3 = check_for_external_calls(xml_path)
            info.update(fm3)
            if fm3["if97_names_found"]:
                print(f"  IF97 function names found in equations: {fm3['if97_names_found']}")
                print(f"  Sample equations with IF97 calls:")
                for eq in fm3["external_func_calls"][:2]:
                    print(f"    {eq}")
                info["fm3_verdict"] = "OPAQUE (external calls present)"
            else:
                print(f"  No IF97 named calls found → likely inlined as polynomial arithmetic")
                info["fm3_verdict"] = "INLINED (no named IF97 calls in equations)"

            # MSL equation count relative to SimpleLoop baseline
            n_eq = info.get("n_equations")
            if n_eq is not None:
                ratio = n_eq / baseline["n_equations"]
                print(f"  Equation count ratio vs baseline: {ratio:.0f}×")
                info["eq_count_ratio"] = ratio

            # translateModelXML attempt (non-critical)
            try:
                raw = omc.sendExpression(
                    f'translateModelXML({model_name})', parsed=False
                )
                omc_check(omc, f"translateModelXML({model_name})")
                info["translateModelXML"] = "attempted"
            except Exception:
                info["translateModelXML"] = "skipped"

            return info

        r = run_step(model_name, step)
        results.append(r)

        if r["status"] == "PASS":
            i = r["info"]
            findings.append(
                f"{model_name}: eqs={i.get('n_equations', '?')}  "
                f"states={i.get('n_states', '?')}  "
                f"FM3={i.get('fm3_verdict', '?')}  "
                f"time={i.get('extraction_time_s', 0):.1f}s"
            )
        else:
            findings.append(f"{model_name}: FAILED — {r['error']}")

    omc.sendExpression("quit()", parsed=False)

    # ------------------------------------------------------------------
    extra = [
        "FM3 (opaque external calls) assessment:",
        *[f"  {f}" for f in findings],
        "",
        "Decision criteria:",
        "  PASS: translation succeeds + equations accessible → proceed to solver",
        "  PARTIAL (opaque): external C calls present → need own media library",
        "  FAIL (translate error): FM3 confirmed → write pure-Modelica IAPWS-IF97",
    ]

    write_tabular_report(
        title      = "OPAL Feasibility Test 3 — Modelica.Fluid Compatibility",
        results    = results,
        omc_ver    = ver,
        model_desc = "FluidPipeLoop (static / dynamic / two-phase) with MSL 4.1.0",
        extra_lines= extra,
        out_path   = RESULTS_DIR / "test3_report.txt",
    )

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
