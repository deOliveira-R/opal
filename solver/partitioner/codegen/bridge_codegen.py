"""
bridge_codegen.py — Generate a C bridge from OM-generated per-equation functions.

Rewrites OM equation functions to use flat arrays instead of the DATA struct,
then compiles to a standalone .so callable from Python via ctypes.

The bridge provides:
  - opal_bridge_set_state(p[], h[], mdot[]) — write state into flat arrays
  - opal_bridge_set_params(values[]) — write parameters
  - opal_bridge_evaluate() — call all algebraic equations in BLT order
  - opal_bridge_get_*() — read computed properties, face densities, etc.
"""

import re
import subprocess
from pathlib import Path

from .info_parser import ModelInfo, parse_info_json
from .build_codegen import extract_function_bodies, parse_function_signatures


def extract_equation_functions(model_c: Path, model_name: str) -> dict[int, str]:
    """Extract per-equation function bodies from {Model}.c.

    Returns dict mapping equation index → function body (C code string).
    """
    text = model_c.read_text()
    # Pattern: void ModelName_eqFunction_XX(DATA *data, threadData_t *threadData) { ... }
    pattern = re.compile(
        r'void\s+' + re.escape(model_name) + r'_eqFunction_(\d+)\s*\('
        r'DATA\s*\*\s*data\s*,\s*threadData_t\s*\*\s*threadData\s*\)\s*\{',
        re.MULTILINE
    )

    functions = {}
    for m in pattern.finditer(text):
        eq_id = int(m.group(1))
        start = m.end()
        # Find matching closing brace (track nesting)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        body = text[start:i - 1]  # exclude closing brace
        functions[eq_id] = body

    return functions


def rewrite_equation_body(body: str) -> str:
    """Rewrite an OM equation function body to use flat opal_vars/opal_params arrays.

    Replaces:
      data->localData[0]->realVars[data->simulationInfo->realVarsIndex[N]] → opal_vars[N]
      data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[N]] → opal_params[N]
      DIVISION_SIM(x, y, "msg", equationIndexes) → ((y) != 0.0 ? (x)/(y) : 0.0)
      relationhysteresis(data, &out, val, threshold, ...) → simple comparison
      threadData->lastEquationSolved = N; → (removed)
      const int equationIndexes[2] = {1,N}; → (removed)
    """
    # Variable access: realVars[realVarsIndex[N]]
    body = re.sub(
        r'\(data->localData\[0\]->realVars\[data->simulationInfo->realVarsIndex\[(\d+)\]\]'
        r'\s*/\*[^*]*\*/\s*\)',
        r'opal_vars[\1]',
        body
    )
    # Fallback without comment
    body = re.sub(
        r'data->localData\[0\]->realVars\[data->simulationInfo->realVarsIndex\[(\d+)\]\]',
        r'opal_vars[\1]',
        body
    )

    # Parameter access: realParameter[realParamsIndex[N]]
    body = re.sub(
        r'\(data->simulationInfo->realParameter\[data->simulationInfo->realParamsIndex\[(\d+)\]\]'
        r'\s*/\*[^*]*\*/\s*\)',
        r'opal_params[\1]',
        body
    )
    body = re.sub(
        r'data->simulationInfo->realParameter\[data->simulationInfo->realParamsIndex\[(\d+)\]\]',
        r'opal_params[\1]',
        body
    )

    # DIVISION_SIM(numerator, denominator, "msg", equationIndexes)
    # Need to handle nested parens in numerator/denominator
    body = re.sub(
        r'DIVISION_SIM\(',
        'OPAL_DIV(',
        body
    )

    # relationhysteresis(data, &out, val, threshold, tmp1, tmp2, idx, GreaterEq, GreaterEqZC)
    # → out = (val >= threshold)
    body = re.sub(
        r'relationhysteresis\s*\(\s*data\s*,\s*&(\w+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,'
        r'\s*[^,]+\s*,\s*[^,]+\s*,\s*\d+\s*,\s*GreaterEq\s*,\s*GreaterEqZC\s*\)',
        r'\1 = (\2) >= (\3)',
        body
    )
    body = re.sub(
        r'relationhysteresis\s*\(\s*data\s*,\s*&(\w+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,'
        r'\s*[^,]+\s*,\s*[^,]+\s*,\s*\d+\s*,\s*Greater\s*,\s*GreaterZC\s*\)',
        r'\1 = (\2) > (\3)',
        body
    )

    # Remove threadData->lastEquationSolved = N;
    body = re.sub(r'\s*threadData->lastEquationSolved\s*=\s*\d+\s*;', '', body)

    # Remove equationIndexes declaration
    body = re.sub(r'\s*const\s+int\s+equationIndexes\[\d+\]\s*=\s*\{[^}]*\}\s*;', '', body)

    # Replace threadData with our static dummy
    body = body.replace('threadData', '(&_td)')

    return body


def generate_bridge_c(info: ModelInfo, model_c: Path, functions_c: Path,
                      functions_h: Path) -> str:
    """Generate a standalone C bridge file.

    The bridge provides flat-array access to OM-generated equation evaluations.
    """
    model_name = info.model_name
    eq_functions = extract_equation_functions(model_c, model_name)
    func_bodies = extract_function_bodies(functions_c)
    func_sigs = parse_function_signatures(functions_h)

    lines = []
    lines.append(f'/* Auto-generated OPAL equation bridge for {model_name} */')
    lines.append('/* Calls OM equation functions through flat arrays — no OM runtime */')
    lines.append('')

    # Standard includes
    lines.append('#include <stdio.h>')
    lines.append('#include <stdlib.h>')
    lines.append('#include <stdarg.h>')
    lines.append('#include <math.h>')
    lines.append('#include <string.h>')
    lines.append('')

    # Type stubs
    lines.append('/* ── OM type stubs ── */')
    lines.append('typedef double modelica_real;')
    lines.append('typedef long modelica_integer;')
    lines.append('typedef int modelica_boolean;')
    lines.append('typedef void* modelica_metatype;')
    lines.append('typedef struct { int dummy; } threadData_t;')
    lines.append('')
    lines.append('#define DLLDirection')
    lines.append('#define OMC_LABEL_UNUSED')
    lines.append('#define OMC_DISABLE_OPT')
    lines.append('')

    # Error/division stubs
    lines.append('static void throwStreamPrint(threadData_t *td, const char *fmt, ...) {')
    lines.append('    (void)td; (void)fmt; abort();')
    lines.append('}')
    lines.append('')
    lines.append('/* Safe division: returns 0 if denominator is zero */')
    lines.append('#define OPAL_DIV(num, den, msg, idx) \\')
    lines.append('    ((den) != 0.0 ? (num) / (den) : 0.0)')
    lines.append('')

    # Boxptr stubs
    lines.append('static inline modelica_real mmc_unbox_real(modelica_metatype x) { (void)x; return 0.0; }')
    lines.append('static inline modelica_metatype mmc_mk_rcon(modelica_real x) { (void)x; return (void*)0; }')
    lines.append('static inline modelica_metatype mmc_mk_icon(modelica_integer x) { (void)x; return (void*)0; }')
    lines.append('')

    # infoStreamPrint stub (used in linear system equations — we skip those)
    lines.append('#define infoStreamPrint(...)')
    lines.append('#define OMC_LOG_DT 0')
    lines.append('')

    # Flat arrays
    lines.append(f'/* ── Flat data arrays ({info.n_vars} vars, {info.n_params} params) ── */')
    lines.append(f'static double opal_vars[{info.n_vars}];')
    lines.append(f'static double opal_params[{info.n_params}];')
    lines.append(f'static threadData_t _td = {{0}};')
    lines.append('')

    # Forward declarations for media functions
    lines.append('/* ── Media function forward declarations ── */')
    for f in func_sigs:
        param_str = ', '.join(f'{t} {n}' for t, n in f['params'])
        lines.append(f'{f["return_type"]} {f["name"]}({param_str});')
    lines.append('')

    # Media function bodies (from _functions.c)
    lines.append('/* ── OM-generated media function bodies ── */')
    lines.append(func_bodies)
    lines.append('')

    # Rewritten equation functions
    lines.append('/* ── Rewritten equation functions (flat array access) ── */')
    for eq_id in info.blt_order:
        if eq_id not in eq_functions:
            continue
        eq = next((e for e in info.algebraic_eqs if e.eq_index == eq_id), None)
        defines_str = ', '.join(eq.defines) if eq else '?'

        body = eq_functions[eq_id]
        rewritten = rewrite_equation_body(body)

        lines.append(f'/* eq {eq_id}: {defines_str} */')
        lines.append(f'static void eq_{eq_id}(void) {{')
        lines.append(rewritten)
        lines.append('}')
        lines.append('')

    # Evaluate all algebraic equations in BLT order
    lines.append('/* ── Batch evaluator (BLT order) ── */')
    lines.append('void opal_bridge_evaluate(void) {')
    for eq_id in info.blt_order:
        if eq_id in eq_functions:
            lines.append(f'    eq_{eq_id}();')
    lines.append('}')
    lines.append('')

    # State setter
    _gen_state_setter(lines, info)

    # Parameter setter
    _gen_param_setter(lines, info)

    # Getters for variable groups
    _gen_getters(lines, info)

    return '\n'.join(lines)


def _gen_state_setter(lines: list, info: ModelInfo):
    """Generate opal_bridge_set_state() function."""
    prefix = _detect_prefix(info)
    N = _detect_N(info, prefix)

    p_indices = info.vars_by_pattern(f'{prefix}.p', N)
    h_indices = info.vars_by_pattern(f'{prefix}.h', N)

    # mdot: find state and dummy-state mdot variables (NOT derivatives)
    mdot_names = sorted(
        [(name, vi.index) for name, vi in info.all_vars.items()
         if f'{prefix}.mdot[' in name and vi.kind in ('state', 'dummy state')],
        key=lambda x: int(x[0].split('[')[1].rstrip(']'))
    )

    lines.append(f'/* State setter: write p[N], h[N], mdot into opal_vars */')
    lines.append(f'void opal_bridge_set_state(int n_p, double* p, int n_h, double* h,')
    lines.append(f'                           int n_mdot, double* mdot) {{')

    for i, idx in enumerate(p_indices):
        lines.append(f'    if ({i} < n_p) opal_vars[{idx}] = p[{i}];')
    for i, idx in enumerate(h_indices):
        lines.append(f'    if ({i} < n_h) opal_vars[{idx}] = h[{i}];')
    for i, (name, idx) in enumerate(mdot_names):
        lines.append(f'    if ({i} < n_mdot) opal_vars[{idx}] = mdot[{i}]; /* {name} */')

    lines.append('}')
    lines.append('')


def _gen_param_setter(lines: list, info: ModelInfo):
    """Generate opal_bridge_set_params() function."""
    lines.append(f'/* Parameter setter: write all {info.n_params} parameters */')
    lines.append(f'void opal_bridge_set_params(int n, double* values) {{')
    lines.append(f'    for (int i = 0; i < n && i < {info.n_params}; i++)')
    lines.append(f'        opal_params[i] = values[i];')
    lines.append('}')
    lines.append('')


def _gen_getters(lines: list, info: ModelInfo):
    """Generate getter functions for property groups."""
    prefix = _detect_prefix(info)
    N = _detect_N(info, prefix)

    # Helper to generate a getter for an array variable
    def _getter(func_name, var_pattern, count, comment):
        indices = info.vars_by_pattern(var_pattern, count)
        lines.append(f'/* {comment} */')
        lines.append(f'void {func_name}(int n, double* out) {{')
        for i, idx in enumerate(indices):
            lines.append(f'    if ({i} < n) out[{i}] = opal_vars[{idx}];')
        lines.append('}')
        lines.append('')

    _getter('opal_bridge_get_rho_face', f'{prefix}.rho_face', N + 1,
            f'Face densities rho_face[1..{N+1}]')
    _getter('opal_bridge_get_h_face', f'{prefix}.h_face', N + 1,
            f'Donor-cell face enthalpies h_face[1..{N+1}]')
    _getter('opal_bridge_get_drho_dp', f'{prefix}.drho_dp', N,
            f'Pressure derivative drho_dp[1..{N}]')
    _getter('opal_bridge_get_drho_dh', f'{prefix}.drho_dh', N,
            f'Enthalpy derivative drho_dh[1..{N}]')
    _getter('opal_bridge_get_T_cell', f'{prefix}.T_cell', N,
            f'Cell temperature T_cell[1..{N}]')

    # rho: may be sparse (OM may not store all cells' rho — some go directly to rho_face)
    rho_indices = info.vars_by_pattern(f'{prefix}.rho', N)
    if rho_indices:
        _getter('opal_bridge_get_rho', f'{prefix}.rho', N, f'Cell density rho[1..{N}]')

    # Generic getter: copy all vars
    lines.append(f'void opal_bridge_get_all_vars(int n, double* out) {{')
    lines.append(f'    for (int i = 0; i < n && i < {info.n_vars}; i++)')
    lines.append(f'        out[i] = opal_vars[i];')
    lines.append('}')
    lines.append('')

    # N getter
    lines.append(f'int opal_bridge_get_N(void) {{ return {N}; }}')
    lines.append(f'int opal_bridge_get_n_vars(void) {{ return {info.n_vars}; }}')
    lines.append(f'int opal_bridge_get_n_params(void) {{ return {info.n_params}; }}')
    lines.append('')


def _detect_prefix(info: ModelInfo) -> str:
    """Detect the component prefix from state variable names."""
    for name in info.states:
        if '.p[' in name:
            return name.split('.p[')[0]
    raise ValueError("Cannot detect prefix — no {prefix}.p[i] state found")


def _detect_N(info: ModelInfo, prefix: str) -> int:
    """Detect N (number of cells) from pressure state count."""
    return len([n for n in info.states if n.startswith(f'{prefix}.p[')])


def build_bridge(model_c: Path, functions_c: Path, functions_h: Path,
                 info_json: Path, output_so: Path = None) -> Path:
    """Build the equation bridge .so from OM-generated files.

    Args:
        model_c: Path to {Model}.c (per-equation functions)
        functions_c: Path to {Model}_functions.c (media function bodies)
        functions_h: Path to {Model}_functions.h (media function declarations)
        info_json: Path to {Model}_info.json (variable/equation metadata)
        output_so: Output .so path (default: same dir as model_c)

    Returns:
        Path to compiled .so
    """
    info = parse_info_json(info_json)
    model_name = info.model_name

    if output_so is None:
        output_so = model_c.parent / f'opal_bridge_{model_name}.so'

    c_code = generate_bridge_c(info, model_c, functions_c, functions_h)
    gen_c = output_so.with_suffix('.c')
    gen_c.write_text(c_code)

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
        raise RuntimeError(f"Bridge compilation failed:\n{result.stderr}\nCmd: {' '.join(cmd)}")

    return output_so
