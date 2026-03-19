#!/usr/bin/env python3
"""
phase2_driver.py — End-to-end two-phase extraction → solver pipeline.

Proves the OPAL architecture: Modelica → OpenModelica → XML → Partitioner → Solver.

Usage:
    PYTHONPATH=solver/two_phase:solver python solver/phase2_driver.py <xml_path> [options]

Example (Edwards blowdown):
    PYTHONPATH=solver/two_phase:solver python solver/phase2_driver.py \
        docs/validation/edwards/data/EdwardsTest_backEnd.xml \
        --dt 5e-5 --t-end 0.6 --H_i 1e7 --C_d 0.87

The XML provides geometry and initial conditions (extracted from Modelica).
Solver strategy parameters (H_i, C_0, C_d, dt, etc.) are user configuration.
"""

import sys
import argparse
import pathlib
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
OPAL_ROOT = pathlib.Path(__file__).resolve().parents[0]
sys.path.insert(0, str(OPAL_ROOT / "two_phase"))
sys.path.insert(0, str(OPAL_ROOT))

import opal_two_phase as tp
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import (
    map_pipe1d, solver_from_spec, boundary_faces_from_spec, init_5eq_state,
)


def main():
    parser = argparse.ArgumentParser(description="OPAL Phase 2 driver: XML → two-phase solver")
    parser.add_argument("xml_path", type=str, help="Path to backEnd XML from OpenModelica")
    parser.add_argument("--dt", type=float, default=5e-5, help="Timestep [s]")
    parser.add_argument("--t-end", type=float, default=0.6, help="End time [s]")
    parser.add_argument("--H_i", type=float, default=1e7, help="Interfacial HT coeff [W/(m³·K)]")
    parser.add_argument("--C_0", type=float, default=1.0, help="Distribution parameter [-]")
    parser.add_argument("--C_d", type=float, default=None, help="Break discharge coefficient [-]")
    parser.add_argument("--fluid", choices=["simple", "iapws"], default="iapws",
                        help="Fluid package")
    parser.add_argument("--model", choices=["hem", "5eq"], default="5eq",
                        help="Flow model")
    parser.add_argument("--save", type=str, default=None, help="Save results to .npz")
    args = parser.parse_args()

    xml_path = pathlib.Path(args.xml_path)
    if not xml_path.exists():
        print(f"ERROR: XML file not found: {xml_path}")
        sys.exit(1)

    # ==================================================================
    # Stage 1: Parse XML → EquationSystem
    # ==================================================================
    print(f"Loading XML: {xml_path}")
    es = load_equation_system(str(xml_path))
    print(f"  {es.summary()}")

    # ==================================================================
    # Stage 2: Map → Pipe1DGridSpec
    # ==================================================================
    spec = map_pipe1d(es)
    print(f"\n{spec.summary()}")

    # ==================================================================
    # Stage 3: Construct solver strategies (user configuration)
    # ==================================================================
    if args.fluid == "iapws":
        fluid = tp.IAPWSIF97Properties()
    else:
        fluid = tp.SimpleFluidProperties()

    ht = tp.LinearRelaxation(H_i=args.H_i)
    drift = tp.ZuberFindlay(C_0=args.C_0)
    closures = tp.DriftFluxClosures(ht, drift)

    if args.model == "5eq":
        model = tp.FiveEqModel(fluid, closures)
    else:
        model = tp.HEMModel()

    recon = tp.DonorCell()
    momentum = tp.InertialMomentum()

    critical_flow = None
    if args.C_d is not None:
        critical_flow = tp.RansomTrapp(fluid, x_trans=0.10, c_floor=1200.0)

    print(f"\nSolver strategies:")
    print(f"  Fluid:    {args.fluid}")
    print(f"  Model:    {model.name}")
    print(f"  Momentum: {momentum.name}")
    print(f"  Recon:    DonorCell")
    print(f"  Critical: {critical_flow.name if critical_flow else 'none'}")
    print(f"  H_i={args.H_i:.0e}, C_0={args.C_0}")
    if args.C_d is not None:
        print(f"  C_d={args.C_d}")

    # ==================================================================
    # Stage 4: Construct TwoPhaseSolver from spec
    # ==================================================================
    solver = solver_from_spec(spec, fluid, model, recon, momentum, critical_flow)
    print(f"\nSolver: N={solver.N}, dx={solver.dx:.4f} m, A={solver.A_flow:.6f} m²")

    # ==================================================================
    # Stage 5: Initialize state from extracted initial conditions
    # ==================================================================
    p, alpha, h_l, h_v, mdot = init_5eq_state(spec, fluid)

    print(f"\nInitial state:")
    print(f"  p:     {p[0]/1e6:.2f} .. {p[-1]/1e6:.2f} MPa")
    print(f"  alpha: {alpha[0]:.2e} .. {alpha[-1]:.2e}")
    print(f"  h_l:   {h_l[0]/1e3:.1f} kJ/kg")
    print(f"  h_v:   {h_v[0]/1e3:.1f} kJ/kg")

    # ==================================================================
    # Stage 6: Build boundary faces from spec
    # ==================================================================
    bc_in, bc_out = boundary_faces_from_spec(spec, h_l[0], h_v[0], C_d=args.C_d)
    print(f"\nBoundary conditions:")
    print(f"  Inlet:  {'WallFace' if spec.inlet_closed else 'PressureFace'}")
    print(f"  Outlet: {'WallFace' if spec.outlet_closed else 'BreakFace' if args.C_d else 'PressureFace'}")
    if spec.p_out is not None:
        print(f"  p_out:  {spec.p_out/1e6:.4f} MPa")

    # ==================================================================
    # Stage 7: Time integration
    # ==================================================================
    dt = args.dt
    t_end = args.t_end
    n_steps = int(t_end / dt)

    print(f"\nTime integration: dt={dt*1e3:.3f} ms, t_end={t_end:.3f} s, {n_steps} steps")

    history = [(0.0, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy())]
    save_stride = max(1, n_steps // 200)  # ~200 snapshots

    t = 0.0
    p_break_initial = p[-1]

    print(f"\n{'step':>8s} {'t_ms':>8s} {'p[0]_MPa':>10s} {'p[-1]_MPa':>10s} "
          f"{'alpha[0]':>10s} {'mdot_out':>12s}")

    for step in range(n_steps):
        solver.step_bf(p, alpha, h_l, h_v, mdot, bc_in, bc_out, t, dt)
        t += dt

        if (step + 1) % save_stride == 0 or step == n_steps - 1:
            history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))

        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            print(f"{step:8d} {t*1e3:8.2f} {p[0]/1e6:10.3f} {p[-1]/1e6:10.3f} "
                  f"{alpha[0]:10.4f} {mdot[-1]:12.3f}")

    # ==================================================================
    # Stage 8: Verification
    # ==================================================================
    print(f"\n{'='*60}")
    print(f"VERIFICATION")
    print(f"{'='*60}")

    # V1: All state finite
    all_finite = (np.all(np.isfinite(p)) and np.all(np.isfinite(alpha))
                  and np.all(np.isfinite(h_l)) and np.all(np.isfinite(h_v))
                  and np.all(np.isfinite(mdot)))
    print(f"  All state finite:        {'PASS' if all_finite else 'FAIL'}")

    # V2: Pressure bounded
    p_in_range = np.all(p > 0) and np.all(p < 50e6)
    print(f"  Pressure in range:       {'PASS' if p_in_range else 'FAIL'}")

    # V3: Void fraction bounded
    alpha_bounded = np.all(alpha >= 0) and np.all(alpha <= 1)
    print(f"  Void fraction bounded:   {'PASS' if alpha_bounded else 'FAIL'}")

    # V4: Break cell pressure decreased (for blowdown)
    if not spec.outlet_closed and args.C_d is not None:
        p_decreased = p[-1] < p_break_initial
        print(f"  Break pressure decreased: {'PASS' if p_decreased else 'FAIL'} "
              f"({p_break_initial/1e6:.2f} → {p[-1]/1e6:.2f} MPa)")

    # V5: Wall flow = 0 (if inlet closed)
    if spec.inlet_closed:
        wall_zero = mdot[0] == 0.0
        print(f"  Wall flow zero:          {'PASS' if wall_zero else 'FAIL'} "
              f"(mdot[0]={mdot[0]:.2e})")

    # V6: Outlet flow positive (for blowdown)
    if not spec.outlet_closed:
        outflow_pos = mdot[-1] > 0
        print(f"  Outlet flow positive:    {'PASS' if outflow_pos else 'FAIL'} "
              f"(mdot[-1]={mdot[-1]:.3f})")

    # ==================================================================
    # Save results
    # ==================================================================
    if args.save:
        t_arr = np.array([rec[0] for rec in history])
        p_arr = np.array([rec[1] for rec in history])
        a_arr = np.array([rec[2] for rec in history])
        hl_arr = np.array([rec[3] for rec in history])
        hv_arr = np.array([rec[4] for rec in history])
        mdot_arr = np.array([rec[5] for rec in history])
        np.savez(args.save, t=t_arr, p=p_arr, alpha=a_arr,
                 h_l=hl_arr, h_v=hv_arr, mdot=mdot_arr,
                 dx=spec.dx, N=spec.N, dt=dt, H_i=args.H_i)
        print(f"\nResults saved to {args.save}")

    print(f"\nPipeline complete: XML → EquationSystem → Pipe1DGridSpec → TwoPhaseSolver → {n_steps} steps")


if __name__ == "__main__":
    main()
