#!/usr/bin/env python3
"""
edwards_bridge_6eq_validation.py — Edwards blowdown with 6-equation two-fluid
via the OM equation bridge.

ALL physics from Modelica: metastable T_l/rho_l/rho_v, interfacial drag
(Ishii bubbly + Schiller-Naumann), per-phase Darcy friction, interfacial HT,
Henry-Fauske critical flow.
Solver provides ONLY: Thomas algorithm + semi-implicit splitting.

Usage:
    python edwards_bridge_6eq_validation.py [options]

Options:
    --dt FLOAT     Timestep in seconds (default: 5e-5)
    --t-end FLOAT  End time in seconds (default: 0.6)
    --save         Save .npz + MAPE JSON to results directory
"""

import sys
import json
import time as time_mod
import argparse
import pathlib
import numpy as np

SOLVER_ROOT = pathlib.Path(__file__).resolve().parent
OPAL_ROOT = SOLVER_ROOT.parent
sys.path.insert(0, str(SOLVER_ROOT / "two_phase"))
sys.path.insert(0, str(SOLVER_ROOT))
sys.path.insert(0, str(OPAL_ROOT / "docs" / "validation" / "edwards" / "data"))

from partitioner.codegen.info_parser import parse_info_json
from partitioner.codegen.equation_bridge import OMEquationBridge
from partitioner.two_fluid_variants.bridge_6eq_solver import BridgeTwoFluidSolver
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d

RESULTS_DIR = OPAL_ROOT / "feasibility" / "results" / "edwards_6eq"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Edwards blowdown — 6-eq two-fluid via OM bridge")
    parser.add_argument('--dt', type=float, default=5e-5,
                        help='Timestep in seconds (default: 5e-5)')
    parser.add_argument('--t-end', type=float, default=0.6,
                        help='End time in seconds (default: 0.6)')
    parser.add_argument('--save', action='store_true',
                        help='Save .npz and MAPE JSON to results directory')
    parser.add_argument('--isentropic', action='store_true',
                        help='Use isentropic compressibility for pressure diagonal')
    parser.add_argument('--no-break-form-loss', action='store_true',
                        help='Disable break form loss at outlet')
    return parser.parse_args(argv)


def run_validation(args):
    """Run Edwards blowdown with 6-eq two-fluid model."""
    model_name = '6eq_hf_ramp'
    dt = args.dt
    t_end = args.t_end

    # ── Resolve paths ──
    bridge_so = RESULTS_DIR / "opal_bridge_EdwardsTest_TwoFluid_HF_Ramp.so"
    info_json = RESULTS_DIR / "EdwardsTest_TwoFluid_HF_Ramp_info.json"
    edwards_xml = RESULTS_DIR / "EdwardsTest_TwoFluid_HF_Ramp.xml"

    for path, name in [(bridge_so, "Bridge .so"), (info_json, "Info JSON")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            print("Run: python -c 'from solver.partitioner.codegen.translate_model "
                  "import translate_and_build; translate_and_build(\"EdwardsTest_TwoFluid_HF_Ramp\")'")
            sys.exit(1)

    use_isentropic = getattr(args, 'isentropic', False)
    break_form_loss = not getattr(args, 'no_break_form_loss', False)

    print("=" * 70)
    print("Edwards Blowdown — 6-EQ TWO-FLUID via OM Bridge")
    print("  Critical flow: Henry-Fauske | Ramp: Modelica RampedBreak")
    print(f"  Compressibility: {'isentropic' if use_isentropic else 'h_mix (thermal)'}")
    print(f"  Break form loss: {'ON' if break_form_loss else 'OFF'}")
    print("  ALL physics from Modelica — solver provides ONLY numerical methods")
    print("=" * 70)

    # ── Load model ──
    info = parse_info_json(info_json)
    bridge = OMEquationBridge(bridge_so, info)
    N = bridge.N

    if not edwards_xml.exists():
        raise FileNotFoundError(f"Edwards XML not found: {edwards_xml}")
    es = load_equation_system(str(edwards_xml))
    spec = map_pipe1d(es)

    solver = BridgeTwoFluidSolver(bridge, spec, es=es,
                                   use_isentropic_a11=use_isentropic,
                                   break_form_loss=break_form_loss)

    print(f"\nModel: 6-eq two-fluid, N={N}")
    print(f"  {info.summary()}")
    print(f"  Bridge has mdot_l: {bridge.has('mdot_l')}, mdot_v: {bridge.has('mdot_v')}")
    print(f"  Bridge has F_drag: {bridge.has('F_drag')}, fric_l: {bridge.has('fric_l')}")
    print(f"  Bridge has v_l: {bridge.has('v_l')}, v_v: {bridge.has('v_v')}")

    # ── Load experimental data ──
    from edwards_blowdown_data import edwards_blowdown
    import iapws

    # ── Initial conditions ──
    p_init = 7e6
    p = np.full(N, p_init)
    alpha = np.full(N, 1e-6)
    mdot_l = np.zeros(N + 1)  # closed end, no flow
    mdot_v = np.zeros(N + 1)
    h_l = np.full(N, 986.6e3)
    h_v_sat = iapws.IAPWS97(P=p_init/1e6, x=1).h * 1e3
    h_v = np.full(N, h_v_sat)

    n_steps = int(t_end / dt)

    gauge_stations = edwards_blowdown["gauge_stations"]
    gs_cells = {}
    for name, gs in gauge_stations.items():
        gs_cells[name] = min(int(gs["x_m"] / spec.dx), N - 1)

    print(f"\nRunning {n_steps} steps (dt={dt*1e6:.0f}µs, t_end={t_end}s)...")
    print(f"{'step':>8s} {'t_ms':>8s} {'p_GS1':>10s} {'p_GS7':>10s} "
          f"{'a_max':>10s} {'mdot_l_out':>12s} {'mdot_v_out':>12s}")

    save_times = np.concatenate([
        np.arange(0, 0.01, 0.0005),
        np.arange(0.01, 0.1, 0.005),
        np.arange(0.1, t_end + 0.01, 0.01),
    ])
    history = []
    next_save_idx = 0
    t = 0.0

    t_wall_start = time_mod.perf_counter()

    for step in range(n_steps):
        while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
            history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(),
                           mdot_l.copy(), mdot_v.copy()))
            next_save_idx += 1

        solver.time = t
        solver.step(p, alpha, h_l, h_v, mdot_l, mdot_v, dt)
        t += dt

        if step % 2000 == 0 or step == n_steps - 1:
            p_gs1 = p[gs_cells["GS-1"]] / 1e6
            p_gs7 = p[gs_cells["GS-7"]] / 1e6
            a_max = np.max(alpha)
            print(f"{step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} "
                  f"{a_max:10.4f} {mdot_l[N]:12.3f} {mdot_v[N]:12.3f}")

    history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(),
                   mdot_l.copy(), mdot_v.copy()))

    t_wall_end = time_mod.perf_counter()
    wall_time = t_wall_end - t_wall_start

    # ── Compare to experiment ──
    data_dir = OPAL_ROOT / "docs" / "validation" / "edwards" / "data"
    t_sim = np.array([rec[0] for rec in history])

    exp_files = {
        "GS-1": "fig3-gs1.csv", "GS-2": "fig4-gs2.csv", "GS-3": "fig5-gs3.csv",
        "GS-4": "fig6-gs4.csv", "GS-5": "fig7-gs5.csv", "GS-6": "fig8-gs6.csv",
        "GS-7": "fig9-gs7.csv",
    }
    gs_x = {"GS-1": 3.927, "GS-2": 3.769, "GS-3": 2.935, "GS-4": 2.024,
             "GS-5": 1.469, "GS-6": 0.914, "GS-7": 0.079}
    PSIA_TO_MPA = 6894.76 / 1e6
    gs_errors = {}

    for gs_name, filename in exp_files.items():
        exp_path = data_dir / filename
        if not exp_path.exists():
            continue
        exp_data = np.loadtxt(exp_path, delimiter=",")
        t_exp, p_exp_MPa = exp_data[:, 0], exp_data[:, 1] * PSIA_TO_MPA

        cell_idx = gs_cells[gs_name]
        p_sim = np.array([rec[1][cell_idx] for rec in history]) / 1e6

        errors = []
        for i in range(len(t_exp)):
            p_interp = np.interp(t_exp[i], t_sim, p_sim)
            if p_exp_MPa[i] > 0.1:
                errors.append(abs((p_interp - p_exp_MPa[i]) / p_exp_MPa[i] * 100))
        if errors:
            gs_errors[gs_name] = np.mean(errors)

    overall = np.mean(list(gs_errors.values())) if gs_errors else float('nan')

    print(f"\n{'='*70}")
    print(f"Edwards Blowdown — 6-EQ TWO-FLUID via OM Bridge")
    print(f"{'='*70}")
    print(f"\nMAPE by station:")
    for gs_name in exp_files:
        if gs_name in gs_errors:
            print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): {gs_errors[gs_name]:.1f}%")
    print(f"\n  Overall MAPE: {overall:.1f}%")

    print(f"\nPhysical indicators:")
    alpha_final = history[-1][2]
    print(f"  alpha_max (final):     {np.max(alpha_final):.4f}")
    print(f"  mdot_l_out (final):    {history[-1][5][N]:.3f} kg/s")
    print(f"  mdot_v_out (final):    {history[-1][6][N]:.3f} kg/s")

    print(f"\nPerformance:")
    print(f"  Wall time: {wall_time:.2f}s ({n_steps/wall_time:.0f} steps/s)")

    print(f"\nComparison:")
    print(f"  5-eq production (V24+AC10):  22.7% MAPE")
    print(f"  6-eq previous (pre-J/L):     39.5% MAPE")
    print(f"  6-eq this run:               {overall:.1f}% MAPE")
    print(f"  RELAP5 reference:            ~20% MAPE")

    # ── Save results ──
    if args.save:
        save_dir = OPAL_ROOT / "docs" / "validation" / "edwards" / "results" / "six_eq_hf_ramp"
        save_dir.mkdir(parents=True, exist_ok=True)

        npz_path = save_dir / f"edwards_{model_name}_N{N}.npz"
        np.savez(npz_path,
                 t=np.array([r[0] for r in history]),
                 p=np.array([r[1] for r in history]),
                 alpha=np.array([r[2] for r in history]),
                 h_l=np.array([r[3] for r in history]),
                 h_v=np.array([r[4] for r in history]),
                 mdot_l=np.array([r[5] for r in history]),
                 mdot_v=np.array([r[6] for r in history]),
                 dx=np.array([spec.dx]),
                 N=np.array([N]),
                 model=np.array([model_name]),
                 dt=np.array([dt]),
                 t_end=np.array([t_end]))
        print(f"\n  Saved: {npz_path}")

        mape_path = save_dir / f"mape_{model_name}_N{N}.json"
        mape_data = {
            "model": model_name,
            "equations": "6-eq two-fluid",
            "critical_flow": "Henry-Fauske",
            "ramp_source": "Modelica RampedBreak",
            "interfacial_drag": "Ishii bubbly (Schiller-Naumann)",
            "N": N,
            "dt": dt,
            "t_end": t_end,
            "overall_mape": round(overall, 2),
            "per_station": {k: round(v, 2) for k, v in gs_errors.items()},
            "wall_time_s": round(wall_time, 2),
            "steps_per_second": round(n_steps / wall_time, 0),
        }
        with open(mape_path, 'w') as f:
            json.dump(mape_data, f, indent=2)
        print(f"  Saved: {mape_path}")

    return overall, gs_errors, history, N, spec


if __name__ == '__main__':
    args = parse_args()
    run_validation(args)
