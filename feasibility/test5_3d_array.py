#!/usr/bin/env python3
"""
OPAL Feasibility Test 5 — 3D Array Component Extraction
=========================================================
Tests whether OM preserves the Approach B vessel's array structure
when extracted.

Key questions (from docs/vessel.md):
  FM1: Are for-loops unrolled in instantiateModel output?
  FM2: Does BLT scatter vessel equations or keep column structure?
  FM4: How does extraction time scale with mesh size?
  FM5: Are array indices preserved as p[i,j,k] in variable names?

Uses Vessel3D.mo: 3D cylindrical mesh, axial flow only.
Tests mesh sizes: 2×2×3, 3×3×5 (baseline), 4×4×8.

Run:
    ../external/venv/bin/python test5_3d_array.py
"""

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from extraction_utils import (
    RESULTS_DIR, MODELS_DIR,
    start_omc_session, omc_check, dump_xml_dae,
    parse_xml_report, write_tabular_report, decode_adjacency,
)

# Mesh sizes to test: (Nr, Ntheta, Nz)
MESH_SIZES = [
    (2, 2, 3),   # tiny: 12 cells
    (3, 3, 5),   # baseline: 45 cells
    (4, 4, 8),   # medium: 128 cells
]

# Template for a Vessel3D variant with fixed mesh parameters
# (OM can't set integer parameters from scripting for structural array bounds)
TEMPLATE = """
// Auto-generated for Test 5, Nr={Nr} Ntheta={Ntheta} Nz={Nz}
model Vessel3D_{tag}
  "Vessel3D with Nr={Nr}, Ntheta={Ntheta}, Nz={Nz}"

  parameter Real dz    = 0.30;
  parameter Real dr    = 0.10;
  parameter Real rho0  = 720.0;
  parameter Real C_comp = 1e-9;
  parameter Real mu    = 1e-4;
  parameter Real Cp    = 5000.0;
  parameter Real K_fric = 12.0 * mu / (dr * dr);
  parameter Real p_in  = 15.50e6;
  parameter Real p_out = 15.40e6;
  parameter Real T_in  = 563.0;

  Real p[{Nr}, {Ntheta}, {Nz}](each start = 15.5e6, each fixed = true);
  Real T[{Nr}, {Ntheta}, {Nz}](each start = 563.0, each fixed = true);
  Real u_z[{Nr}, {Ntheta}, {Nz}+1];
  Real u_r[{Nr}+1, {Ntheta}, {Nz}];
  Real u_t[{Nr}, {Ntheta}, {Nz}];

equation
  for i in 1:{Nr}+1 loop for j in 1:{Ntheta} loop for k in 1:{Nz} loop
    u_r[i, j, k] = 0.0;
  end for; end for; end for;

  for i in 1:{Nr} loop for j in 1:{Ntheta} loop for k in 1:{Nz} loop
    u_t[i, j, k] = 0.0;
  end for; end for; end for;

  for i in 1:{Nr} loop
    for j in 1:{Ntheta} loop
      (p_in - p[i, j, 1]) / dz = K_fric * u_z[i, j, 1];
      for k in 2:{Nz} loop
        (p[i, j, k-1] - p[i, j, k]) / dz = K_fric * u_z[i, j, k];
      end for;
      (p[i, j, {Nz}] - p_out) / dz = K_fric * u_z[i, j, {Nz}+1];
    end for;
  end for;

  for i in 1:{Nr} loop for j in 1:{Ntheta} loop for k in 1:{Nz} loop
    rho0 * C_comp * der(p[i, j, k]) = (u_z[i, j, k] - u_z[i, j, k+1]) / dz;
  end for; end for; end for;

  for i in 1:{Nr} loop for j in 1:{Ntheta} loop for k in 1:{Nz} loop
    rho0 * Cp * der(T[i, j, k]) =
      rho0 * Cp * u_z[i, j, k]   * (T_in - T[i, j, k]) / dz
    + rho0 * Cp * u_z[i, j, k+1] * (T[i, j, k] - T_in) / dz;
  end for; end for; end for;

end Vessel3D_{tag};
"""


def check_array_structure(xml_path: Path, Nr: int, Ntheta: int, Nz: int) -> dict:
    """
    Check FM5: Are array indices preserved in variable names?
    Check FM1: Count how many variables have [i,j,k] style indexing.
    Check FM2: Analyse BLT block sizes and coherence.
    """
    if not xml_path.exists():
        return {}

    root = ET.parse(xml_path).getroot()
    n_cells = Nr * Ntheta * Nz

    # FM5: array index preservation
    all_vars = [(v.get("id"), v.get("name", ""), v.get("variability", ""))
                for v in root.findall(".//variable")]
    array_vars   = [(i, n, v) for i, n, v in all_vars if "[" in n]
    p_cell_vars  = [n for _, n, _ in array_vars if n.startswith("p[")]
    T_cell_vars  = [n for _, n, _ in array_vars if n.startswith("T[")]

    # FM1: for-loop unrolling — check instantiated flat text (not available here,
    # but count scalar equations as a proxy)
    eqs_el = root.find(".//equations")
    n_eqs  = int(eqs_el.get("dimension", 0)) if eqs_el is not None else 0

    # FM2: BLT block coherence
    blt_blocks = root.findall(".//bltBlock")
    blt_sizes  = [len(list(b)) for b in blt_blocks]
    n_single   = sum(1 for s in blt_sizes if s == 1)
    n_multi    = sum(1 for s in blt_sizes if s > 1)
    max_block  = max(blt_sizes) if blt_sizes else 0

    # Column-block structure: Nr*Ntheta columns, each of size Nz+1 (axial momentum)
    expected_col_block = Nz + 1
    expected_n_columns = Nr * Ntheta
    col_blocks = [s for s in blt_sizes if s == expected_col_block]

    return {
        "n_array_vars":       len(array_vars),
        "n_p_cell_vars":      len(p_cell_vars),
        "n_T_cell_vars":      len(T_cell_vars),
        "n_cells":            n_cells,
        "p_cells_complete":   len(p_cell_vars) >= n_cells,
        "sample_p_names":     p_cell_vars[:4],
        "n_equations":        n_eqs,
        "blt_sizes":          sorted(set(blt_sizes)),
        "n_single_blocks":    n_single,
        "n_multi_blocks":     n_multi,
        "max_blt_block":      max_block,
        "n_col_blocks":       len(col_blocks),
        "expected_col_block": expected_col_block,
        "expected_n_columns": expected_n_columns,
        "blt_coherent":       len(col_blocks) >= expected_n_columns,
    }


def run_mesh_size(Nr: int, Ntheta: int, Nz: int) -> dict:
    tag        = f"Nr{Nr}_Nt{Ntheta}_Nz{Nz}"
    model_name = f"Vessel3D_{tag}"
    n_cells    = Nr * Ntheta * Nz
    print(f"\n  Mesh {Nr}×{Ntheta}×{Nz} ({n_cells} cells) …")

    # Write model file
    mo_path = RESULTS_DIR / f"vessel3d_{tag}.mo"
    mo_path.write_text(TEMPLATE.format(Nr=Nr, Ntheta=Ntheta, Nz=Nz, tag=tag))

    omc = start_omc_session(load_msl=False)
    m = {"Nr": Nr, "Ntheta": Ntheta, "Nz": Nz, "n_cells": n_cells}

    try:
        omc.sendExpression(f'loadFile("{mo_path.as_posix()}")', parsed=False)
        omc_check(omc, f"loadFile {tag}")

        # instantiateModel (FM1: for-loop unrolling check)
        t0 = time.perf_counter()
        flat_mo = omc.sendExpression(f"instantiateModel({model_name})", parsed=False)
        omc_check(omc, f"instantiateModel {tag}")
        m["t_instantiate"] = time.perf_counter() - t0

        # FM1: check if for-loops are unrolled in flat Modelica
        flat_text = flat_mo.replace("\\n", "\n")
        eq_section = flat_text[flat_text.find("equation"):] if "equation" in flat_text else ""
        for_loops_in_eqs = eq_section.count("for ") - eq_section.count("end for")
        m["for_loops_in_flat"] = max(0, for_loops_in_eqs)
        scalar_eq_lines = sum(
            1 for l in eq_section.splitlines()
            if "=" in l and not l.strip().startswith("//")
               and not l.strip().startswith("for")
               and not l.strip().startswith("end")
        )
        m["scalar_eq_lines_flat"] = scalar_eq_lines
        print(f"    instantiateModel: {m['t_instantiate']:.2f}s  "
              f"for_loops_remaining={m['for_loops_in_flat']}  "
              f"scalar_eq_lines≈{scalar_eq_lines}")

        # dumpXMLDAE flat
        t0 = time.perf_counter()
        _, info_flat = dump_xml_dae(omc, model_name, "flat", f"vessel_{tag}_flat")
        m["t_flat"]   = time.perf_counter() - t0
        m["eqs_flat"] = info_flat.get("n_equations")

        # dumpXMLDAE backEnd
        t0 = time.perf_counter()
        xml_path, info_back = dump_xml_dae(omc, model_name, "backEnd", f"vessel_{tag}_back")
        m["t_backend"]  = time.perf_counter() - t0
        m["eqs_back"]   = info_back.get("n_equations")
        m["states"]     = info_back.get("n_states")
        m["blt_blocks"] = info_back.get("n_blt_blocks")
        m["alias_vars"] = info_back.get("n_alias_vars")

        # Deep array structure analysis
        arr = check_array_structure(xml_path, Nr, Ntheta, Nz)
        m.update(arr)

        print(f"    backEnd: eqs={m['eqs_back']}  states={m['states']}  "
              f"BLT={m['blt_blocks']}  alias={m['alias_vars']}  "
              f"time={m['t_backend']:.2f}s")
        print(f"    FM5: p[i,j,k] vars={m.get('n_p_cell_vars', '?')}/{n_cells}  "
              f"complete={m.get('p_cells_complete', '?')}  "
              f"sample={m.get('sample_p_names', [])[:2]}")
        print(f"    FM2: BLT block sizes={m.get('blt_sizes', '?')}  "
              f"col_blocks={m.get('n_col_blocks', '?')}/{m.get('expected_n_columns', '?')}  "
              f"coherent={m.get('blt_coherent', '?')}")

    finally:
        omc.sendExpression("quit()", parsed=False)

    return m


def main() -> int:
    print("=" * 70)
    print("OPAL Feasibility Test 5 — 3D Array Component Extraction")
    print("=" * 70)

    all_metrics = []
    for Nr, Ntheta, Nz in MESH_SIZES:
        try:
            m = run_mesh_size(Nr, Ntheta, Nz)
            all_metrics.append(m)
        except Exception as e:
            print(f"  ERROR for {Nr}×{Ntheta}×{Nz}: {e}")
            all_metrics.append({"Nr": Nr, "Ntheta": Ntheta, "Nz": Nz,
                                 "n_cells": Nr*Ntheta*Nz, "error": str(e)})

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("Scaling summary:")
    hdr = (f"  {'Mesh':<12}  {'Cells':>6}  {'Eqs(bk)':>8}  {'States':>7}  "
           f"{'BLT':>5}  {'FM1(loops)':>11}  {'FM5(p_ok)':>10}  "
           f"{'FM2(coh)':>9}  {'t_back':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for m in all_metrics:
        mesh = f"{m['Nr']}×{m['Ntheta']}×{m['Nz']}"
        if "error" in m:
            print(f"  {mesh:<12}  ERROR: {m['error'][:40]}")
            continue
        fm1 = f"{m.get('for_loops_in_flat', '?')} loops left"
        fm5 = f"{m.get('n_p_cell_vars', '?')}/{m['n_cells']}"
        fm2 = "YES" if m.get("blt_coherent") else ("NO" if "blt_coherent" in m else "?")
        print(
            f"  {mesh:<12}  {m['n_cells']:>6}  "
            f"{str(m.get('eqs_back', '?')):>8}  "
            f"{str(m.get('states', '?')):>7}  "
            f"{str(m.get('blt_blocks', '?')):>5}  "
            f"{fm1:>11}  {fm5:>10}  {fm2:>9}  "
            f"{m.get('t_backend', 0):>7.2f}"
        )

    # Findings
    findings = []
    valid = [m for m in all_metrics if "error" not in m]

    # FM1
    unrolled = all(m.get("for_loops_in_flat", 1) == 0 for m in valid)
    findings.append(
        f"FM1 (for-loop unrolling): "
        f"{'YES — for-loops unrolled in flat Modelica' if unrolled else 'NO — loops remain'}"
    )

    # FM5
    fm5_ok = all(m.get("p_cells_complete", False) for m in valid)
    findings.append(
        f"FM5 (array index preservation): "
        f"{'YES — p[i,j,k] names intact' if fm5_ok else 'PARTIAL or NO — check sample names'}"
    )

    # FM2
    coherent = [m.get("blt_coherent", False) for m in valid]
    findings.append(
        f"FM2 (BLT coherence): "
        f"{sum(coherent)}/{len(coherent)} mesh sizes show column-aligned BLT blocks"
    )

    # FM4
    if len(valid) >= 2:
        n1, n2 = valid[0], valid[-1]
        if n1.get("t_backend") and n2.get("t_backend"):
            ratio = n2["t_backend"] / n1["t_backend"]
            cell_ratio = n2["n_cells"] / n1["n_cells"]
            findings.append(
                f"FM4 (scaling): time {n1['t_backend']:.2f}s→{n2['t_backend']:.2f}s "
                f"for cells {n1['n_cells']}→{n2['n_cells']} "
                f"(time ratio={ratio:.1f}×, cell ratio={cell_ratio:.1f}×)"
            )

    from datetime import datetime
    lines = [
        "OPAL Feasibility Test 5 — 3D Array Component Extraction",
        f"Run : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        hdr, "  " + "-" * (len(hdr) - 2),
    ]
    for m in all_metrics:
        mesh = f"{m['Nr']}×{m['Ntheta']}×{m['Nz']}"
        if "error" in m:
            lines.append(f"  {mesh:<12}  ERROR: {m['error'][:40]}")
            continue
        fm1 = f"{m.get('for_loops_in_flat', '?')} loops left"
        fm5 = f"{m.get('n_p_cell_vars', '?')}/{m['n_cells']}"
        fm2 = "YES" if m.get("blt_coherent") else ("NO" if "blt_coherent" in m else "?")
        lines.append(
            f"  {mesh:<12}  {m['n_cells']:>6}  "
            f"{str(m.get('eqs_back', '?')):>8}  "
            f"{str(m.get('states', '?')):>7}  "
            f"{str(m.get('blt_blocks', '?')):>5}  "
            f"{fm1:>11}  {fm5:>10}  {fm2:>9}  "
            f"{m.get('t_backend', 0):>7.2f}"
        )
    lines += [
        "", "Findings:",
        *[f"  {f}" for f in findings],
        "",
        "Decision criteria:",
        "  FM1: for-loops unrolled → pattern-reconstruction pass needed (expected, Easy fix)",
        "  FM5: indices preserved → good; mangled → FM1 reconstruction handles it",
        "  FM2: BLT coherent → vessel stays together; scattered → use 'flat' level extraction",
        "  FM4: time linear in cells → proceed; super-linear → vessel internals to C++",
    ]

    report_path = RESULTS_DIR / "test5_report.txt"
    text = "\n".join(lines) + "\n"
    report_path.write_text(text)
    print(f"\n{text}\nReport → {report_path}")

    n_fail = sum(1 for m in all_metrics if "error" in m)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
