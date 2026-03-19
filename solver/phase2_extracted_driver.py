#!/usr/bin/env python3
"""
phase2_extracted_driver.py — Extraction-driven semi-implicit solver.

THIS IS THE OPAL ARCHITECTURE IN ACTION:
  Modelica Pipe1D.mo → OpenModelica extraction → equation classifier
  → semi-implicit solver driven by extracted equation structure

The physics comes from Modelica. The numerics comes from our semi-implicit
engine. Property evaluation uses the C++ FluidPackage (same math as the
Modelica SimpleFluid/Water media).

Usage:
    PYTHONPATH=solver/two_phase:solver python solver/phase2_extracted_driver.py \\
        docs/validation/edwards/data/EdwardsTest_backEnd.xml
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


def main():
    xml_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
        OPAL_ROOT.parent / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_backEnd.xml"
    )

    if not xml_path.exists():
        print(f"ERROR: {xml_path} not found")
        sys.exit(1)

    # ==================================================================
    # Stage 1: Extract equation system from XML
    # ==================================================================
    print(f"Loading: {xml_path}")
    es = load_equation_system(str(xml_path))
    print(f"  {es.summary()}")

    # ==================================================================
    # Stage 2: Map to Pipe1DGridSpec (geometry + BCs)
    # ==================================================================
    spec = map_pipe1d(es)
    print(f"\n{spec.summary()}")

    # ==================================================================
    # Stage 3: Classify equations by role
    # ==================================================================
    cs = classify_equations(es, prefix=spec.prefix)
    print(f"\n{cs.summary()}")

    # Verify classification is complete
    total = (len(cs.mass_eqs) + len(cs.momentum_eqs) + len(cs.energy_eqs)
             + len(cs.property_eqs) + len(cs.face_density_eqs)
             + len(cs.donor_cell_eqs) + len(cs.constraint_eqs))
    print(f"\n  Classified: {total}/{len(es.equations)}")
    if cs.unclassified:
        print(f"  Unclassified equations:")
        for eq in cs.unclassified:
            print(f"    {eq[:100]}")

    # ==================================================================
    # Stage 4: Construct extraction-driven solver
    # ==================================================================
    # The fluid package provides property evaluation (same math as Modelica SimpleFluid)
    fluid = tp.SimpleFluidProperties()
    solver = ExtractedSemiImplicitSolver(cs, fluid, spec)
    print(f"\nSolver: N={solver.N}, dx={solver.dx:.4f}, A={solver.A_flow:.6f}")

    # ==================================================================
    # Stage 5: Initialize state from extracted initial conditions
    # ==================================================================
    p = np.array(spec.p0, dtype=float)
    h = np.array(spec.h0, dtype=float)
    mdot = np.array(spec.mdot0, dtype=float)

    print(f"\nInitial: p={p[0]/1e6:.2f} MPa, h={h[0]/1e3:.1f} kJ/kg")

    # ==================================================================
    # Stage 6: Time integration
    # ==================================================================
    dt = 5e-5
    t_end = 0.6
    n_steps = int(t_end / dt)

    print(f"Time: dt={dt*1e3:.3f} ms, t_end={t_end:.3f} s, {n_steps} steps")
    print(f"\n{'step':>8s} {'t_ms':>8s} {'p[0]':>10s} {'p[-1]':>10s} {'mdot_out':>12s}")

    history = [(0.0, p.copy(), h.copy(), mdot.copy())]
    t = 0.0

    for step in range(n_steps):
        solver.step(p, h, mdot, dt)
        t += dt

        if step % max(1, n_steps // 200) == 0:
            history.append((t, p.copy(), h.copy(), mdot.copy()))

        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            print(f"{step:8d} {t*1e3:8.2f} {p[0]/1e6:10.3f} {p[-1]/1e6:10.3f} "
                  f"{mdot[-1]:12.3f}")

    # ==================================================================
    # Stage 7: Verification
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"VERIFICATION")
    print(f"{'='*60}")

    all_finite = np.all(np.isfinite(p)) and np.all(np.isfinite(h)) and np.all(np.isfinite(mdot))
    print(f"  All state finite:      {'PASS' if all_finite else 'FAIL'}")

    p_decreased = p[-1] < spec.p0[-1]
    print(f"  Break pressure dropped: {'PASS' if p_decreased else 'FAIL'} "
          f"({spec.p0[-1]/1e6:.2f} → {p[-1]/1e6:.2f} MPa)")

    wall_zero = mdot[0] == 0.0
    print(f"  Wall flow zero:        {'PASS' if wall_zero else 'FAIL'}")

    outflow_pos = mdot[-1] > 0
    print(f"  Outlet flow positive:  {'PASS' if outflow_pos else 'FAIL'}")

    print(f"\nPipeline: XML → EquationSystem → Pipe1DGridSpec → ClassifiedSystem"
          f" → ExtractedSemiImplicitSolver → {n_steps} steps")
    print(f"Physics from Modelica. Numerics from OPAL.")


if __name__ == "__main__":
    main()
