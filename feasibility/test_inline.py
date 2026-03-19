#!/usr/bin/env python3
"""
test_inline.py — Test whether annotation(Inline=true) makes OM inline
SimpleFluid function calls in the extracted equation XML.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from extraction_utils import (
    start_omc_session, omc_check, resolve_xml_path,
    OPAL_ROOT, RESULTS_DIR
)

print("=" * 60)
print("Test: Does annotation(Inline=true) produce inlined equations?")
print("=" * 60)

omc = start_omc_session(load_msl=True)

# Load OPAL as a package from the root (handles 'within OPAL.library' paths)
package_mo = (OPAL_ROOT / "package.mo").as_posix()

# Check if package.mo exists at OPAL root; if not, load files individually
if (OPAL_ROOT / "package.mo").exists():
    r = omc.sendExpression(f'loadFile("{package_mo}")', parsed=False)
    omc_check(omc, "loadFile(OPAL/package.mo)")
    print(f"  loadFile(OPAL): {r}")
else:
    # Load library files individually with correct within paths
    # Need to set MODELICAPATH so OM finds 'library' package
    lib_dir = OPAL_ROOT.as_posix()
    omc.sendExpression(f'setModelicaPath("{lib_dir}")', parsed=False)
    omc_check(omc, "setModelicaPath")

    # Load the library package
    lib_pkg = (OPAL_ROOT / "library" / "package.mo").as_posix()
    r = omc.sendExpression(f'loadFile("{lib_pkg}")', parsed=False)
    err = omc_check(omc, "loadFile(library)")
    print(f"  loadFile(library): {r}")

    if "false" in r.lower() or "error" in err.lower():
        # Try loading individual files without within clause
        print("  Package load failed. Loading individual .mo files...")
        for mo in ["Media/SimpleFluid.mo", "Pipes/Pipe1D.mo",
                    "Connectors/FluidPort.mo", "Boundary/PressureSource.mo",
                    "Boundary/ClosedEnd.mo"]:
            fp = (OPAL_ROOT / "library" / mo).as_posix()
            r = omc.sendExpression(f'loadFile("{fp}")', parsed=False)
            omc_check(omc, f"loadFile({mo})")
            print(f"  {mo}: {r}")

# Define a minimal inline test model
model_def = r"""
model InlineTest
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.Pipe1D pipe(N=3, L=3.0, D=0.073, f_D=0.02,
                            p_init=7e6, h_init=980e3);
  library.Boundary.PressureSource atm(p_set=101325.0, h_set=980e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, atm.port);
end InlineTest;
"""

r = omc.sendExpression(f'loadString("{model_def}")', parsed=False)
omc_check(omc, "loadString")
print(f"\n  loadString(InlineTest): {r}")

# Check what classes are loaded
classes = omc.sendExpression("getClassNames()", parsed=False)
print(f"  Loaded classes: {classes}")

# Extract
print(f"\nExtracting InlineTest at backEnd level...")
raw = omc.sendExpression(
    'dumpXMLDAE(InlineTest, translationLevel="backEnd", '
    'addOriginalAdjacencyMatrix=true, addSolvingInfo=true)',
    parsed=False
)
omc_check(omc, "dumpXMLDAE")
print(f"  Raw return: {raw[:200]}")

xml_path = resolve_xml_path(raw, "InlineTest",
                            ["InlineTest.xml", "InlineTest_backEnd.xml"])
print(f"  XML: {xml_path} ({xml_path.stat().st_size} bytes)")

# Parse equations
root = ET.parse(xml_path).getroot()
eq_texts = []
for eq in root.findall(".//equation"):
    if eq.text:
        eq_texts.append(eq.text.strip())

print(f"\n  Total equations: {len(eq_texts)}")

# Check for SimpleFluid function calls
sf_calls = [t for t in eq_texts if "SimpleFluid" in t]
print(f"  Equations with 'SimpleFluid' calls: {len(sf_calls)}")

if sf_calls:
    print("\n  RESULT: Functions NOT inlined.")
    print("  Sample equations with function calls:")
    for eq in sf_calls[:4]:
        print(f"    {eq[:140]}")
else:
    print("\n  RESULT: Functions INLINED!")
    print("  Sample equations (should be flat arithmetic):")
    for eq in eq_texts[:6]:
        print(f"    {eq[:140]}")

# Check for ODE equations (der(...))
ode_eqs = [t for t in eq_texts if "der(" in t]
print(f"\n  ODE equations: {len(ode_eqs)}")
for eq in ode_eqs[:3]:
    print(f"    {eq[:140]}")

print("\nDone.")
