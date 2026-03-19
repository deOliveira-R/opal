"""
build_codegen.py — Compile OM-generated _functions.c into a standalone .so.

Takes the translateModel output (_functions.c, _functions.h) and produces
a shared library with clean C function names (opal_*) callable from Python
via ctypes. Zero OpenModelica runtime dependency.

Strategy:
  1. Parse _functions.h to extract function declarations (omc_* signatures)
  2. Extract function bodies from _functions.c (everything between extern "C" guards)
  3. Generate a standalone C file with:
     - OPAL type stubs (no OM headers)
     - The extracted function bodies
     - Clean wrapper functions (opal_rho_ph etc.)
  4. Compile to .so
"""

import re
import subprocess
import sys
from pathlib import Path


STUBS_H = Path(__file__).parent / "opal_om_stubs.h"


def parse_function_signatures(functions_h: Path) -> list[dict]:
    """Extract omc_* function signatures from the OM-generated header.

    Returns list of dicts: {name, return_type, params: [(type, name), ...]}
    """
    text = functions_h.read_text()
    # Match: modelica_real omc_library_Media_...(threadData_t *threadData, modelica_real _p, ...);
    pattern = re.compile(
        r'^(modelica_real|modelica_integer)\s+'
        r'(omc_\w+)\s*\(([^)]*)\)\s*;',
        re.MULTILINE
    )
    funcs = []
    for m in pattern.finditer(text):
        ret_type = m.group(1)
        name = m.group(2)
        params_str = m.group(3)
        params = []
        for p in params_str.split(','):
            p = p.strip()
            if not p:
                continue
            # e.g. "threadData_t *threadData" or "modelica_real _p"
            parts = p.rsplit(None, 1)
            if len(parts) == 2:
                ptype, pname = parts[0].strip(), parts[1].strip()
                params.append((ptype, pname))
        funcs.append({
            'name': name,
            'return_type': ret_type,
            'params': params,
        })
    return funcs


def extract_function_bodies(functions_c: Path) -> str:
    """Extract the C function bodies from _functions.c.

    Strips all preprocessor directives (#include, #ifdef, #endif, etc.)
    and the extern "C" wrapper, keeping only the function definitions.
    The generated standalone file provides its own types via inline stubs.
    """
    text = functions_c.read_text()
    lines = text.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Skip all preprocessor directives
        if stripped.startswith('#'):
            continue
        # Skip extern "C" wrapper
        if stripped == 'extern "C" {':
            continue
        filtered.append(line)

    # The extern "C" block has a trailing "}" that we need to remove.
    # It's the last non-empty line in the file.
    while filtered and not filtered[-1].strip():
        filtered.pop()
    if filtered and filtered[-1].strip() == '}':
        filtered.pop()

    return '\n'.join(filtered)


def detect_medium_prefix(func_names: list[str]) -> str:
    """Detect the OM medium package prefix from function names.

    E.g., from ['omc_library_Media_SimpleFluid_rho__ph', ...] → 'omc_library_Media_SimpleFluid_'
    From ['omc_library_Media_Water_rho__ph', ...] → 'omc_library_Media_Water_'

    Strategy: find the longest common prefix ending with '_' among all names.
    """
    if not func_names:
        return 'omc_'
    # All names should start with omc_library_Media_
    prefix = 'omc_library_Media_'
    # Find the shortest name and work backwards from the last function-name part
    sample = func_names[0]
    # The medium name is between 'omc_library_Media_' and the function name.
    # Function names always contain '__' (double underscore from Modelica encoding).
    # The medium name does NOT contain '__' (it's a package name).
    # So find the first '__' after the Media_ prefix.
    rest = sample[len(prefix):]
    idx = rest.find('__')
    if idx == -1:
        # Fallback: find last '_' before the function suffix
        idx = rest.rfind('_')
    if idx == -1:
        return prefix
    # The medium name is rest[:idx], but we need to find the last '_' before '__'
    # that separates the medium name from the function name.
    # E.g., rest = "SimpleFluid_rho__ph" → medium = "SimpleFluid", func = "rho__ph"
    medium_and_func = rest[:idx]  # "SimpleFluid_rho"
    last_under = medium_and_func.rfind('_')
    if last_under == -1:
        return prefix + rest[:idx + 1]
    medium = medium_and_func[:last_under]  # "SimpleFluid"
    return prefix + medium + '_'


def make_opal_name(omc_name: str, medium_prefix: str) -> str:
    """Convert omc_library_Media_SimpleFluid_rho__ph -> opal_rho_ph.

    Uses the detected medium prefix for reliable stripping.
    """
    if omc_name.startswith(medium_prefix):
        short = omc_name[len(medium_prefix):]
    else:
        short = omc_name.replace('omc_', '')
    # Replace __ (Modelica name encoding) with _
    short = short.replace('__', '_')
    return f'opal_{short}'


def generate_standalone_c(functions_c: Path, functions_h: Path,
                          model_name: str) -> str:
    """Generate a single standalone C file ready for compilation."""

    funcs = parse_function_signatures(functions_h)
    bodies = extract_function_bodies(functions_c)
    medium_prefix = detect_medium_prefix([f['name'] for f in funcs])

    lines = []
    lines.append('/* Auto-generated by OPAL build_codegen.py — do not edit */')
    lines.append(f'/* Source model: {model_name} */')
    lines.append('')

    # Include stubs
    lines.append('#include <stdio.h>')
    lines.append('#include <stdlib.h>')
    lines.append('#include <stdarg.h>')
    lines.append('#include <math.h>')
    lines.append('')

    # Inline the type stubs (so this file is self-contained)
    lines.append('/* ── OM type stubs ── */')
    lines.append('typedef double modelica_real;')
    lines.append('typedef long modelica_integer;')
    lines.append('typedef int modelica_boolean;')
    lines.append('typedef const char* modelica_string;')
    lines.append('typedef void* modelica_metatype;')
    lines.append('typedef struct { int lastEquationSolved; } threadData_t;')
    lines.append('')
    lines.append('#define DLLDirection')
    lines.append('#define OMC_LABEL_UNUSED')
    lines.append('')
    lines.append('static void throwStreamPrint(threadData_t *td, const char *fmt, ...) {')
    lines.append('    (void)td; (void)fmt;')
    lines.append('    fprintf(stderr, "OPAL codegen: division by zero in OM-generated code\\n");')
    lines.append('    abort();')
    lines.append('}')
    lines.append('')

    # Boxptr stubs (never called, but must compile)
    lines.append('static inline modelica_real mmc_unbox_real(modelica_metatype x) { (void)x; return 0.0; }')
    lines.append('static inline modelica_metatype mmc_mk_rcon(modelica_real x) { (void)x; return (void*)0; }')
    lines.append('static inline modelica_metatype mmc_mk_icon(modelica_integer x) { (void)x; return (void*)0; }')
    lines.append('')

    # Forward declarations for all omc_ functions (needed for mutual recursion)
    lines.append('/* ── Forward declarations ── */')
    for f in funcs:
        param_str = ', '.join(f'{t} {n}' for t, n in f['params'])
        lines.append(f'{f["return_type"]} {f["name"]}({param_str});')
    lines.append('')

    # The actual function bodies
    lines.append('/* ── OM-generated function bodies ── */')
    lines.append(bodies)
    lines.append('')

    # Static threadData for wrappers
    lines.append('/* ── Clean wrapper API ── */')
    lines.append('static threadData_t _opal_td = {0};')
    lines.append('')

    # Generate clean wrappers
    for f in funcs:
        opal_name = make_opal_name(f['name'], medium_prefix)
        c_ret = 'double' if f['return_type'] == 'modelica_real' else 'long'

        # Skip threadData from wrapper params
        wrapper_params = [(t, n) for t, n in f['params']
                          if 'threadData' not in t]
        param_decl = ', '.join(f'double {n}' for _, n in wrapper_params)
        param_call = ', '.join(n for _, n in wrapper_params)

        lines.append(f'{c_ret} {opal_name}({param_decl}) {{')
        lines.append(f'    return {f["name"]}(&_opal_td, {param_call});')
        lines.append('}')
        lines.append('')

    return '\n'.join(lines)


def build_so(functions_c: Path, functions_h: Path,
             output_so: Path = None, model_name: str = None) -> Path:
    """Build a standalone .so from OM-generated _functions.c.

    Args:
        functions_c: Path to {Model}_functions.c
        functions_h: Path to {Model}_functions.h
        output_so: Output .so path (default: same dir as functions_c)
        model_name: Model name for comments (default: from filename)

    Returns:
        Path to the compiled .so
    """
    if model_name is None:
        model_name = functions_c.stem.replace('_functions', '')

    if output_so is None:
        output_so = functions_c.parent / f'opal_codegen_{model_name}.so'

    # Generate standalone C
    c_code = generate_standalone_c(functions_c, functions_h, model_name)
    gen_c = output_so.with_suffix('.c')
    gen_c.write_text(c_code)

    # Compile
    cmd = [
        'cc', '-shared', '-fPIC', '-O2',
        '-Wno-unused-function',
        '-Wno-sometimes-uninitialized',
        '-Wno-parentheses-equality',
        '-o', str(output_so),
        str(gen_c),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed:\n{result.stderr}\n"
            f"Command: {' '.join(cmd)}"
        )

    # Verify .so exists and has expected symbols
    if not output_so.exists():
        raise RuntimeError(f"Compiled .so not found at {output_so}")

    return output_so


def list_exports(so_path: Path) -> list[str]:
    """List exported opal_* symbols from a .so file."""
    result = subprocess.run(
        ['nm', '-gU', str(so_path)],
        capture_output=True, text=True
    )
    symbols = []
    for line in result.stdout.split('\n'):
        parts = line.split()
        if len(parts) >= 3 and parts[2].startswith('_opal_'):
            symbols.append(parts[2].lstrip('_'))  # strip leading _ (macOS)
    return sorted(symbols)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <functions.c> <functions.h> [output.so]")
        sys.exit(1)

    fc = Path(sys.argv[1])
    fh = Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    so = build_so(fc, fh, out)
    print(f"Built: {so} ({so.stat().st_size} bytes)")
    exports = list_exports(so)
    print(f"Exports ({len(exports)}):")
    for s in exports:
        print(f"  {s}")
