#!/usr/bin/env python3
"""
compare_void_variants.py — Run Edwards blowdown with all solver variants and compare.

Variants:
  base     Original bridge_5eq_solver (baseline)
  v1       rho_v linearization fix
  v2       Two-stage pressure solve
  v3       Fine-grained alpha blend (sigmoid at ALPHA_MID=0.001)
  v4       Augmented diagonal (RELAP5-like smooth coupling)
  v5       2x2 Block Thomas (simultaneous pressure+void)

Usage:
    PYTHONPATH=solver python solver/compare_void_variants.py [--variants base,v1,v2,...] [--dt 5e-5]
"""

import sys
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
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d

# Solver imports — each variant
SOLVER_MAP = {}

def _import_solver(name):
    """Lazy import of solver variant."""
    if name == 'base':
        from partitioner.bridge_5eq_solver import BridgeDriftFluxSolver
    elif name == 'v1':
        from partitioner.bridge_5eq_solver_v1_rv_fix import BridgeDriftFluxSolver
    elif name == 'v2':
        from partitioner.bridge_5eq_solver_v2_twostage import BridgeDriftFluxSolver
    elif name == 'v3':
        from partitioner.bridge_5eq_solver_v3_alpha_blend import BridgeDriftFluxSolver
    elif name == 'v4':
        from partitioner.bridge_5eq_solver_v4_augmented_diag import BridgeDriftFluxSolver
    elif name == 'v5':
        from partitioner.bridge_5eq_solver_v5_block_thomas import BridgeDriftFluxSolver
    elif name == 'v6':
        from partitioner.bridge_5eq_solver_v6_block_blend import BridgeDriftFluxSolver
    elif name == 'v7':
        from partitioner.bridge_5eq_solver_v7_reeval import BridgeDriftFluxSolver
    elif name == 'v8':
        from partitioner.bridge_5eq_solver_v8_gamma_corr import BridgeDriftFluxSolver
    elif name == 'v9':
        from partitioner.bridge_5eq_solver_v9_newton import BridgeDriftFluxSolver
    elif name == 'v10':
        from partitioner.bridge_5eq_solver_v10_offdiag import BridgeDriftFluxSolver
    elif name == 'v11':
        from partitioner.bridge_5eq_solver_v11_a12mod import BridgeDriftFluxSolver
    elif name == 'v12':
        from partitioner.bridge_5eq_solver_v12_offdiag_a12mod import BridgeDriftFluxSolver
    elif name == 'v13':
        from partitioner.bridge_5eq_solver_v13_newton_offdiag import BridgeDriftFluxSolver
    elif name == 'v14':
        from partitioner.bridge_5eq_solver_v14_adaptive_tau import BridgeDriftFluxSolver
    elif name == 'v15':
        from partitioner.bridge_5eq_solver_v15_tau_flash_retune import BridgeDriftFluxSolver
    elif name == 'v16':
        from partitioner.bridge_5eq_solver_v16_energy_sub import BridgeDriftFluxSolver
    elif name == 'v17':
        from partitioner.bridge_5eq_solver_v17_a11_blend import BridgeDriftFluxSolver
    elif name == 'v18':
        from partitioner.bridge_5eq_solver_v18_joint_sweep import BridgeDriftFluxSolver
    else:
        raise ValueError(f"Unknown variant: {name}")
    return BridgeDriftFluxSolver


def run_variant(variant_name, dt=5e-5, t_end=0.6):
    """Run Edwards blowdown with a solver variant. Returns metrics dict."""
    RESULTS_DIR = OPAL_ROOT / "feasibility" / "results"
    model = "EdwardsTest_DriftFlux_HF_Ramp_Flash"
    so_path = RESULTS_DIR / f"opal_bridge_{model}.so"
    info_path = RESULTS_DIR / f"{model}_info.json"
    xml_path = RESULTS_DIR / f"{model}.xml"

    info = parse_info_json(str(info_path))
    bridge = OMEquationBridge(str(so_path), info)
    es = load_equation_system(str(xml_path))
    spec = map_pipe1d(es)

    SolverClass = _import_solver(variant_name)
    solver = SolverClass(bridge, spec, es=es)

    N = bridge.N
    n_steps = int(t_end / dt)

    # Initial conditions
    p = np.full(N, 7e6)
    alpha = np.full(N, 1e-6)
    h_l = np.full(N, 986.6e3)
    h_v = np.full(N, 2772.6e3)
    mdot = np.zeros(N + 1)

    # Gauge station cells
    gs_x = {"GS-1": 3.927, "GS-2": 3.769, "GS-3": 2.935, "GS-4": 2.024,
             "GS-5": 1.469, "GS-6": 0.914, "GS-7": 0.079}
    gs_cells = {name: min(int(x / spec.dx), N - 1) for name, x in gs_x.items()}

    # Record history at moderate intervals
    save_interval = 100  # every 5ms at dt=50us
    t_hist = []
    p_hist = {gs: [] for gs in gs_cells}
    alpha_hist = {gs: [] for gs in gs_cells}

    t0_wall = time_mod.perf_counter()
    crashed = False

    for step in range(n_steps):
        t = step * dt
        solver.time = t
        try:
            solver.step(p, alpha, h_l, h_v, mdot, dt)
        except Exception as e:
            print(f"  [{variant_name}] CRASHED at step {step} (t={t*1000:.1f}ms): {e}")
            crashed = True
            break

        if step % save_interval == 0:
            t_hist.append(t + dt)
            for gs, ci in gs_cells.items():
                p_hist[gs].append(p[ci])
                alpha_hist[gs].append(alpha[ci])

        # Safety: detect NaN
        if not np.all(np.isfinite(p)):
            print(f"  [{variant_name}] NaN in pressure at step {step} (t={t*1000:.1f}ms)")
            crashed = True
            break

    wall_time = time_mod.perf_counter() - t0_wall
    t_hist = np.array(t_hist)

    if crashed:
        return {"variant": variant_name, "crashed": True, "wall_time": wall_time,
                "crash_time_ms": t * 1000}

    # Load experimental data
    data_dir = OPAL_ROOT / "docs" / "validation" / "edwards" / "data"
    PSIA_TO_PA = 6894.76

    # Pressure MAPE per station
    exp_files = {
        "GS-1": "fig3-gs1.csv", "GS-2": "fig4-gs2.csv", "GS-3": "fig5-gs3.csv",
        "GS-4": "fig6-gs4.csv", "GS-5": "fig7-gs5.csv", "GS-6": "fig8-gs6.csv",
        "GS-7": "fig9-gs7.csv",
    }
    gs_mape = {}
    for gs_name, fname in exp_files.items():
        exp_path = data_dir / fname
        if not exp_path.exists():
            continue
        exp_data = np.loadtxt(str(exp_path), delimiter=",")
        t_exp, p_exp_Pa = exp_data[:, 0], exp_data[:, 1] * PSIA_TO_PA
        p_sim = np.array(p_hist[gs_name])
        p_interp = np.interp(t_exp, t_hist, p_sim)
        mask = p_exp_Pa > 1e5
        gs_mape[gs_name] = np.mean(np.abs(p_interp[mask] - p_exp_Pa[mask])
                                    / p_exp_Pa[mask]) * 100

    overall_mape = np.mean(list(gs_mape.values())) if gs_mape else float('nan')

    # Void fraction at GS-5
    void_exp_path = data_dir / "fig14-gs5.csv"
    void_mae = float('nan')
    void_onset_ms = float('nan')
    if void_exp_path.exists():
        void_data = np.loadtxt(str(void_exp_path), delimiter=",")
        t_void_exp = void_data[:, 0]
        alpha_void_exp = np.clip(void_data[:, 1], 0.0, 1.0)

        alpha_gs5 = np.array(alpha_hist["GS-5"])
        alpha_interp = np.interp(t_void_exp, t_hist, alpha_gs5)
        in_range = t_void_exp <= t_hist[-1]
        void_mae = np.mean(np.abs(alpha_interp[in_range] - alpha_void_exp[in_range]))

        # Void onset: first time alpha > 0.01
        onset_idx = np.argmax(alpha_gs5 > 0.01)
        void_onset_ms = t_hist[onset_idx] * 1000 if onset_idx > 0 else 999

    return {
        "variant": variant_name,
        "crashed": False,
        "overall_mape": overall_mape,
        "gs5_mape": gs_mape.get("GS-5", float('nan')),
        "gs1_mape": gs_mape.get("GS-1", float('nan')),
        "void_mae": void_mae,
        "void_onset_ms": void_onset_ms,
        "wall_time": wall_time,
        "gs_mape": gs_mape,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare void fraction solver variants")
    parser.add_argument("--variants", default="base,v1,v2,v3,v4,v5",
                        help="Comma-separated variant names to run")
    parser.add_argument("--dt", type=float, default=5e-5, help="Timestep [s]")
    parser.add_argument("--t-end", type=float, default=0.6, help="End time [s]")
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]

    print("=" * 80)
    print("Edwards Blowdown — Void Fraction Solver Variant Comparison")
    print(f"dt={args.dt*1e6:.0f}µs, t_end={args.t_end}s")
    print("=" * 80)

    results = []
    for v in variants:
        print(f"\nRunning variant: {v}...")
        try:
            r = run_variant(v, dt=args.dt, t_end=args.t_end)
            results.append(r)
            if r["crashed"]:
                print(f"  CRASHED at {r['crash_time_ms']:.1f}ms ({r['wall_time']:.1f}s)")
            else:
                print(f"  MAPE={r['overall_mape']:.1f}% "
                      f"GS5={r['gs5_mape']:.1f}% "
                      f"VoidMAE={r['void_mae']:.3f} "
                      f"Onset={r['void_onset_ms']:.0f}ms "
                      f"({r['wall_time']:.1f}s)")
        except Exception as e:
            print(f"  IMPORT ERROR: {e}")
            results.append({"variant": v, "crashed": True, "error": str(e)})

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Variant':>10s}  {'MAPE':>7s}  {'GS-1':>6s}  {'GS-5':>6s}  "
          f"{'VoidMAE':>8s}  {'Onset':>7s}  {'Time':>6s}  {'Status':>8s}")
    print("-" * 80)
    for r in results:
        if r.get("error"):
            print(f"{r['variant']:>10s}  {'---':>7s}  {'---':>6s}  {'---':>6s}  "
                  f"{'---':>8s}  {'---':>7s}  {'---':>6s}  {'IMPORT':>8s}")
        elif r["crashed"]:
            print(f"{r['variant']:>10s}  {'---':>7s}  {'---':>6s}  {'---':>6s}  "
                  f"{'---':>8s}  {'---':>7s}  {r['wall_time']:>5.1f}s  {'CRASH':>8s}")
        else:
            print(f"{r['variant']:>10s}  {r['overall_mape']:>6.1f}%  "
                  f"{r['gs1_mape']:>5.1f}%  {r['gs5_mape']:>5.1f}%  "
                  f"{r['void_mae']:>8.3f}  {r['void_onset_ms']:>6.0f}ms  "
                  f"{r['wall_time']:>5.1f}s  {'OK':>8s}")
    print("=" * 80)

    # RELAP5 reference
    print("\nRELAP5-3D reference: MAPE ~20%, Void MAE ~0.111, Onset ~10ms")
    print("Experiment: void onset at 9.5ms (GS-5)")


if __name__ == "__main__":
    main()
