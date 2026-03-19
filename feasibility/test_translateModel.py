#!/usr/bin/env python3
"""
test_translateModel.py — Check if translateModel() generates C code with
inlined SimpleFluid functions (no opaque calls).
"""

import sys
from pathlib import Path
from extraction_utils import (
    start_omc_session, omc_check, OPAL_ROOT, RESULTS_DIR
)

print("=" * 60)
print("Test: Does translateModel() produce C code with inlined functions?")
print("=" * 60)

omc = start_omc_session(load_msl=True)

# Load library
lib_pkg = (OPAL_ROOT / "library" / "package.mo").as_posix()
r = omc.sendExpression(f'loadFile("{lib_pkg}")', parsed=False)
omc_check(omc, "loadFile(library)")
print(f"  loadFile(library): {r}")

# Define test model
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
print(f"  loadString: {r}")

# translateModel generates C code
print(f"\nCalling translateModel(InlineTest)...")
r = omc.sendExpression('translateModel(InlineTest)', parsed=False)
err = omc_check(omc, "translateModel")
print(f"  translateModel: {r}")

# Check generated files
gen_files = list(RESULTS_DIR.glob("InlineTest*"))
print(f"\n  Generated files ({len(gen_files)}):")
for f in sorted(gen_files):
    print(f"    {f.name} ({f.stat().st_size} bytes)")

# Look at the main C file for function bodies
c_file = RESULTS_DIR / "InlineTest.c"
if c_file.exists():
    content = c_file.read_text()
    # Check if SimpleFluid function names appear
    if "SimpleFluid" in content:
        print(f"\n  'SimpleFluid' found in .c file — checking context...")
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "SimpleFluid" in line:
                print(f"    L{i+1}: {line.strip()[:120]}")
                if i + 1 < len(lines):
                    print(f"    L{i+2}: {lines[i+1].strip()[:120]}")
                break
    else:
        print(f"\n  'SimpleFluid' NOT in .c file — functions fully inlined in C!")

    # Check for rho_ph patterns
    if "rho_ph" in content:
        print(f"  'rho_ph' found — function calls may exist")
    else:
        print(f"  'rho_ph' NOT found — fully compiled")

# Check _functions.c
func_c = RESULTS_DIR / "InlineTest_functions.c"
if func_c.exists():
    fc = func_c.read_text()
    print(f"\n  InlineTest_functions.c ({len(fc)} bytes)")
    if "SimpleFluid" in fc:
        lines = fc.split('\n')
        sf_lines = [(i+1, l.strip()) for i, l in enumerate(lines) if "SimpleFluid" in l]
        print(f"  SimpleFluid references: {len(sf_lines)}")
        for ln, text in sf_lines[:5]:
            print(f"    L{ln}: {text[:120]}")
    if "rho_f_0" in fc or "750.0" in fc:
        print(f"  Constant values (rho_f_0/750.0) found — function bodies ARE in C code!")

print("\nDone.")
