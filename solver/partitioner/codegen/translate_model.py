"""
translate_model.py — End-to-end: Modelica model → OM translateModel → bridge .so

Single command that:
1. Starts OMC session
2. Loads OPAL library + model definition
3. Runs translateModel → generates C code + _info.json
4. Builds the equation bridge .so (via bridge_codegen)
5. Returns paths to .so, _info.json, and manifest

Usage:
    from partitioner.codegen.translate_model import translate_and_build
    so, info_json = translate_and_build("EdwardsTest_DriftFlux")

    # Or from command line:
    python translate_model.py EdwardsTest_IAPWS_CritFlow
"""

import sys
import time
from pathlib import Path

# Canonical paths
OPAL_ROOT = Path(__file__).resolve().parents[3]
FEASIBILITY_DIR = OPAL_ROOT / "feasibility"
MODELS_DIR = FEASIBILITY_DIR / "models"
RESULTS_DIR = FEASIBILITY_DIR / "results"


def translate_and_build(model_name: str,
                        model_file: Path = None,
                        model_string: str = None,
                        output_dir: Path = None) -> tuple[Path, Path]:
    """Translate a Modelica model and build the equation bridge .so.

    Args:
        model_name: Modelica model name (e.g., "EdwardsTest_DriftFlux")
        model_file: Path to .mo file containing the model (optional — searches models/)
        model_string: Inline model definition string (optional)
        output_dir: Where to write generated files (default: feasibility/results/)

    Returns:
        (bridge_so_path, info_json_path)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(exist_ok=True)

    # ── Start OMC ──
    sys.path.insert(0, str(FEASIBILITY_DIR))
    from extraction_utils import start_omc_session, omc_check, OPAL_ROOT as _root

    print(f"Translating {model_name}...")
    t0 = time.perf_counter()

    omc = start_omc_session(load_msl=True)

    # Load OPAL library
    lib_pkg = (OPAL_ROOT / "library" / "package.mo").as_posix()
    r = omc.sendExpression(f'loadFile("{lib_pkg}")', parsed=False)
    if "false" in r.lower():
        raise RuntimeError(f"Failed to load OPAL library: {omc.sendExpression('getErrorString()')}")

    # Load model
    if model_string:
        r = omc.sendExpression(f'loadString("{model_string}")', parsed=False)
        if "false" in r.lower():
            raise RuntimeError(f"loadString failed: {omc.sendExpression('getErrorString()')}")
    elif model_file:
        r = omc.sendExpression(f'loadFile("{model_file.as_posix()}")', parsed=False)
        if "false" in r.lower():
            raise RuntimeError(f"loadFile({model_file}) failed: {omc.sendExpression('getErrorString()')}")
    else:
        # Search in models/ directory
        mo_path = MODELS_DIR / f"{model_name}.mo"
        if mo_path.exists():
            r = omc.sendExpression(f'loadFile("{mo_path.as_posix()}")', parsed=False)
            if "false" in r.lower():
                raise RuntimeError(f"loadFile({mo_path}) failed: {omc.sendExpression('getErrorString()')}")
        # else: model might already be loaded from the library

    # ── translateModel ──
    print(f"  Running translateModel({model_name})...")
    r = omc.sendExpression(f'translateModel({model_name})', parsed=False)
    err = omc_check(omc, "translateModel")
    if "true" not in r.lower():
        raise RuntimeError(f"translateModel failed: {r[:200]}\n{err}")

    dt_translate = time.perf_counter() - t0
    print(f"  translateModel: {dt_translate:.1f}s")

    # ── Locate generated files ──
    model_c = output_dir / f"{model_name}.c"
    functions_c = output_dir / f"{model_name}_functions.c"
    functions_h = output_dir / f"{model_name}_functions.h"
    info_json = output_dir / f"{model_name}_info.json"

    for f in [model_c, functions_c, functions_h, info_json]:
        if not f.exists():
            raise FileNotFoundError(f"Expected generated file not found: {f}")

    # ── Build bridge .so ──
    print(f"  Building bridge .so...")
    from .bridge_codegen import build_bridge

    bridge_so = build_bridge(
        model_c=model_c,
        functions_c=functions_c,
        functions_h=functions_h,
        info_json=info_json,
        output_so=output_dir / f"opal_bridge_{model_name}.so",
    )

    dt_total = time.perf_counter() - t0
    print(f"  Bridge: {bridge_so} ({bridge_so.stat().st_size} bytes)")
    print(f"  Total: {dt_total:.1f}s")

    return bridge_so, info_json


# ── Also run dumpXMLDAE for the extraction pipeline ──
def translate_and_extract(model_name: str,
                          model_file: Path = None,
                          output_dir: Path = None) -> tuple[Path, Path, Path]:
    """Full pipeline: translateModel (bridge .so) + dumpXMLDAE (XML for classifier).

    Returns:
        (bridge_so_path, info_json_path, xml_path)
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(FEASIBILITY_DIR))
    from extraction_utils import start_omc_session, omc_check, resolve_xml_path

    print(f"Full pipeline for {model_name}...")
    t0 = time.perf_counter()

    omc = start_omc_session(load_msl=True)

    # Load library + model
    lib_pkg = (OPAL_ROOT / "library" / "package.mo").as_posix()
    omc.sendExpression(f'loadFile("{lib_pkg}")', parsed=False)

    mo_path = MODELS_DIR / f"{model_name}.mo"
    if mo_path.exists():
        omc.sendExpression(f'loadFile("{mo_path.as_posix()}")', parsed=False)
    elif model_file and model_file.exists():
        omc.sendExpression(f'loadFile("{model_file.as_posix()}")', parsed=False)

    # translateModel
    print(f"  translateModel({model_name})...")
    r = omc.sendExpression(f'translateModel({model_name})', parsed=False)
    omc_check(omc, "translateModel")
    if "true" not in r.lower():
        raise RuntimeError(f"translateModel failed: {r[:200]}")

    # dumpXMLDAE
    print(f"  dumpXMLDAE({model_name})...")
    raw = omc.sendExpression(
        f'dumpXMLDAE({model_name}, translationLevel="backEnd", '
        'addOriginalAdjacencyMatrix=true, addSolvingInfo=true)',
        parsed=False
    )
    omc_check(omc, "dumpXMLDAE")
    xml_path = resolve_xml_path(raw, model_name, [f"{model_name}.xml"])

    # Build bridge
    print(f"  Building bridge...")
    from .bridge_codegen import build_bridge

    model_c = output_dir / f"{model_name}.c"
    functions_c = output_dir / f"{model_name}_functions.c"
    functions_h = output_dir / f"{model_name}_functions.h"
    info_json = output_dir / f"{model_name}_info.json"

    bridge_so = build_bridge(model_c, functions_c, functions_h, info_json,
                             output_dir / f"opal_bridge_{model_name}.so")

    dt = time.perf_counter() - t0
    print(f"  Done in {dt:.1f}s: {bridge_so.name}, {xml_path.name}")

    return bridge_so, info_json, xml_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ModelName> [model_file.mo]")
        print(f"  Example: {sys.argv[0]} EdwardsTest_DriftFlux")
        sys.exit(1)

    name = sys.argv[1]
    mo = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    so, info = translate_and_build(name, model_file=mo)
    print(f"\nReady: {so}")
