#!/usr/bin/env python3
"""
edwards_comparison.py — The OPAL architecture proof.

Runs the Edwards blowdown through BOTH paths:
  Path A: Modelica extraction → equation classifier → Python semi-implicit solver
  Path B: Hand-wired C++ TwoPhaseSolver (same physics, same SimpleFluid)

Both paths use:
  - Same geometry (from extracted XML: N=5, L=4.096m, D=0.073m)
  - Same fluid (SimpleFluid)
  - Same BCs (wall inlet, pressure outlet at 101325 Pa)
  - Same dt (5e-5 s)
  - Same physics (HEM, inertial momentum, Darcy friction, donor-cell)

If the architecture works, both paths produce IDENTICAL results.
"""

import sys
import pathlib
import numpy as np

OPAL_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OPAL_ROOT / "two_phase"))
sys.path.insert(0, str(OPAL_ROOT))

import opal_two_phase as tp
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d
from partitioner.equation_classifier import classify_equations
from partitioner.extracted_solver import ExtractedSemiImplicitSolver

EDWARDS_XML = OPAL_ROOT.parent / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_backEnd.xml"


def main():
    if not EDWARDS_XML.exists():
        print(f"ERROR: {EDWARDS_XML} not found")
        sys.exit(1)

    # ==================================================================
    # Setup
    # ==================================================================
    print("=" * 70)
    print("OPAL ARCHITECTURE PROOF: Edwards Blowdown Comparison")
    print("  Path A: Modelica extraction → Python semi-implicit solver")
    print("  Path B: C++ TwoPhaseSolver (same physics)")
    print("=" * 70)

    # Load and parse XML
    es = load_equation_system(str(EDWARDS_XML))
    spec = map_pipe1d(es)
    cs = classify_equations(es, prefix=spec.prefix)
    N = spec.N
    dt = 5e-5
    t_end = 0.6
    n_steps = int(t_end / dt)

    print(f"\nGeometry from Modelica extraction:")
    print(f"  N={N}, dx={spec.dx:.4f}m, A={spec.A_flow:.6f}m², D_h={spec.D_h:.4f}m")
    print(f"  Inlet: {'wall' if spec.inlet_closed else 'open'}")
    print(f"  Outlet: p_out={spec.p_out} Pa")
    print(f"  ICs: p={spec.p0[0]/1e6:.1f} MPa, h={spec.h0[0]/1e3:.1f} kJ/kg")
    print(f"  Time: dt={dt*1e3:.3f}ms, t_end={t_end}s, {n_steps} steps")
    print(f"  Equations classified: {cs.summary()}")

    fluid = tp.SimpleFluidProperties()

    # ==================================================================
    # Path A: Extraction-driven solver
    # ==================================================================
    print(f"\n{'─'*70}")
    print("Path A: Extraction-driven semi-implicit solver")
    print(f"{'─'*70}")

    ext_solver = ExtractedSemiImplicitSolver(cs, fluid, spec)
    p_A = np.array(spec.p0, dtype=float)
    h_A = np.array(spec.h0, dtype=float)
    m_A = np.array(spec.mdot0, dtype=float)

    hist_A = [(0.0, p_A.copy())]
    t = 0.0
    for step in range(n_steps):
        ext_solver.step(p_A, h_A, m_A, dt)
        t += dt
        if (step + 1) % (n_steps // 20) == 0:
            hist_A.append((t, p_A.copy()))

    print(f"  Final: p[0]={p_A[0]/1e6:.4f} MPa, p[-1]={p_A[-1]/1e6:.4f} MPa, "
          f"mdot_out={m_A[-1]:.4f}")

    # ==================================================================
    # Path B: C++ solver (same physics)
    # ==================================================================
    print(f"\n{'─'*70}")
    print("Path B: C++ TwoPhaseSolver (HEM, InertialMomentum, SimpleFluid)")
    print(f"{'─'*70}")

    cpp_solver = tp.TwoPhaseSolver(
        spec.N, spec.dx, spec.A_flow, spec.D_h, spec.f_D,
        fluid, tp.DonorCell(), tp.HEMModel(), tp.InertialMomentum())

    bc_in = tp.WallFace(spec.h0[0])
    bc_out = tp.PressureFace(spec.p_out, spec.h0[0])

    p_B = np.array(spec.p0, dtype=float)
    h_B = np.array(spec.h0, dtype=float)
    m_B = np.array(spec.mdot0, dtype=float)

    hist_B = [(0.0, p_B.copy())]
    t = 0.0
    for step in range(n_steps):
        cpp_solver.step_hem_bf(p_B, h_B, m_B, bc_in, bc_out, t, dt)
        t += dt
        if (step + 1) % (n_steps // 20) == 0:
            hist_B.append((t, p_B.copy()))

    print(f"  Final: p[0]={p_B[0]/1e6:.4f} MPa, p[-1]={p_B[-1]/1e6:.4f} MPa, "
          f"mdot_out={m_B[-1]:.4f}")

    # ==================================================================
    # Comparison
    # ==================================================================
    print(f"\n{'='*70}")
    print("COMPARISON: Path A (extraction) vs Path B (C++)")
    print(f"{'='*70}")

    # Final state comparison
    p_err = np.abs(p_A - p_B)
    h_err = np.abs(h_A - h_B)
    m_err = np.abs(m_A - m_B)

    p_rel = p_err / np.maximum(np.abs(p_B), 1.0)
    h_rel = h_err / np.maximum(np.abs(h_B), 1.0)

    print(f"\nFinal state (t={t_end}s):")
    print(f"  Pressure max relative error: {np.max(p_rel):.2e}")
    print(f"  Enthalpy max relative error: {np.max(h_rel):.2e}")
    print(f"  Mass flow max absolute error: {np.max(m_err):.2e}")

    # Per-cell comparison
    print(f"\nPer-cell pressure comparison (MPa):")
    print(f"  {'Cell':>4s} {'Path A':>12s} {'Path B':>12s} {'Diff':>12s} {'Rel Err':>12s}")
    for i in range(N):
        diff = p_A[i] - p_B[i]
        rel = abs(diff) / max(abs(p_B[i]), 1.0)
        print(f"  {i+1:4d} {p_A[i]/1e6:12.6f} {p_B[i]/1e6:12.6f} {diff:12.2f} {rel:12.2e}")

    # Time history comparison
    print(f"\nPressure history at cell 1 (MPa):")
    print(f"  {'t (ms)':>8s} {'Path A':>12s} {'Path B':>12s} {'Rel Err':>12s}")
    for (t_a, p_a), (t_b, p_b) in zip(hist_A, hist_B):
        rel = abs(p_a[0] - p_b[0]) / max(abs(p_b[0]), 1.0)
        print(f"  {t_a*1e3:8.2f} {p_a[0]/1e6:12.6f} {p_b[0]/1e6:12.6f} {rel:12.2e}")

    # Verdict
    print(f"\n{'='*70}")
    max_p_rel = np.max(p_rel)
    if max_p_rel < 1e-6:
        print(f"VERDICT: MATCH — max relative pressure error {max_p_rel:.2e} (< 1e-6)")
        print(f"  Physics from Modelica ≡ Physics from C++")
    elif max_p_rel < 1e-3:
        print(f"VERDICT: CLOSE — max relative pressure error {max_p_rel:.2e} (< 1e-3)")
        print(f"  Small differences from implementation details (operator splitting order)")
    else:
        print(f"VERDICT: DIVERGED — max relative pressure error {max_p_rel:.2e}")
        print(f"  Extraction path and C++ disagree — investigate!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
