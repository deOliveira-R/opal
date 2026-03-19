#!/usr/bin/env python3
"""
extract_edwards_iapws.py — Extract Edwards blowdown model with IAPWS-IF97 properties.

Produces EdwardsTest_IAPWS_backEnd.xml with:
  - Pipe1D with N=24, Water medium (IAPWS-IF97)
  - ClosedEnd inlet (wall)
  - PressureSource outlet (atmospheric — critical flow handled by solver)
  - Edwards geometry: L=4.096m, D=0.073m, f_D=0.02
  - Edwards IC: p=7 MPa, h=986.6 kJ/kg (subcooled at ~500K)
"""

from extraction_utils import (
    start_omc_session, omc_check, resolve_xml_path,
    OPAL_ROOT, RESULTS_DIR
)

print("=" * 60)
print("Extracting Edwards blowdown with IAPWS-IF97 (N=24)")
print("=" * 60)

omc = start_omc_session(load_msl=True)

# Load OPAL library
lib_pkg = (OPAL_ROOT / "library" / "package.mo").as_posix()
r = omc.sendExpression(f'loadFile("{lib_pkg}")', parsed=False)
omc_check(omc, "loadFile(library)")
print(f"  loadFile(library): {r}")

# Load Edwards IAPWS model from file
model_path = (OPAL_ROOT / "feasibility" / "models" / "EdwardsTest_IAPWS.mo").as_posix()
r = omc.sendExpression(f'loadFile("{model_path}")', parsed=False)
err = omc_check(omc, "loadFile(EdwardsTest_IAPWS)")
print(f"  loadFile(model): {r}")
if "false" in r.lower():
    print(f"  ERROR: {err}")
    exit(1)

# Extract
print(f"\nExtracting at backEnd level...")
raw = omc.sendExpression(
    'dumpXMLDAE(EdwardsTest_IAPWS, translationLevel="backEnd", '
    'addOriginalAdjacencyMatrix=true, addSolvingInfo=true)',
    parsed=False
)
err = omc_check(omc, "dumpXMLDAE")

if "true" not in raw.lower():
    print(f"  FAILED: {raw[:200]}")
    exit(1)

xml_path = resolve_xml_path(raw, "EdwardsTest_IAPWS",
                            ["EdwardsTest_IAPWS.xml"])
print(f"  XML: {xml_path} ({xml_path.stat().st_size} bytes)")

# Copy to validation data directory
import shutil
dest = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_IAPWS_backEnd.xml"
shutil.copy2(xml_path, dest)
print(f"  Copied to: {dest}")

# Quick structural check
import xml.etree.ElementTree as ET
root = ET.parse(dest).getroot()
states = [v for v in root.findall('.//orderedVariables/variablesList/*')
          if v.get('variability') == 'continuousState']
eqs = root.findall('.//equation')
print(f"\n  States: {len(states)}")
print(f"  Equations: {len(eqs)}")
for s in states[:5]:
    print(f"    {s.get('name')}")
print(f"    ...")

# Check for IAPWS function calls
eq_texts = [eq.text.strip() for eq in eqs if eq.text]
water_refs = [t for t in eq_texts if "Water" in t or "IF97" in t]
print(f"\n  Equations referencing Water/IF97: {len(water_refs)}")
if water_refs:
    print(f"    Sample: {water_refs[0][:100]}")

print("\nDone.")
