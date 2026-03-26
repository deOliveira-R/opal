#!/usr/bin/env python3
"""
edwards_mesh_convergence.py — Mesh convergence study for Edwards blowdown.

Runs the 5-eq bridge validation at N=12, 24, 48, 96 with CFL-scaled timesteps.
All use Henry-Fauske + Modelica RampedBreak (canonical configuration).

Usage:
    python edwards_mesh_convergence.py [--save] [--plot]
"""

import sys
import json
import pathlib
import numpy as np

SOLVER_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SOLVER_ROOT))

from edwards_bridge_5eq_validation import run_validation, parse_args

OPAL_ROOT = SOLVER_ROOT.parent
RESULTS_DIR = OPAL_ROOT / "docs" / "validation" / "edwards" / "results" / "convergence"

# Mesh convergence configurations: (model_name, dt, N_expected)
# dt scales inversely with N to maintain CFL number
DT_BASE = 5e-5   # 50 µs at N=24
N_BASE = 24
CONFIGS = [
    ('hf_ramp_n12', DT_BASE * N_BASE / 12,  12),   # 100 µs
    ('hf_ramp',     DT_BASE,                 24),   # 50 µs (canonical)
    ('hf_ramp_n48', DT_BASE * N_BASE / 48,  48),   # 25 µs
    ('hf_ramp_n96', DT_BASE * N_BASE / 96,  96),   # 12.5 µs
]


def main():
    save = '--save' in sys.argv

    print("=" * 70)
    print("Edwards Blowdown — MESH CONVERGENCE STUDY")
    print("  Henry-Fauske + Modelica RampedBreak, CFL-scaled dt")
    print("=" * 70)

    results = []

    for model_name, dt, N_expected in CONFIGS:
        print(f"\n{'='*70}")
        print(f"  N={N_expected}, dt={dt*1e6:.1f} µs, model={model_name}")
        print(f"{'='*70}")

        # Build args for the validation driver
        argv = ['--model', model_name, '--dt', str(dt)]
        if save:
            argv.append('--save')
        args = parse_args(argv)

        try:
            overall, gs_errors, history, N, spec = run_validation(args)
            dx = spec.dx
            results.append({
                'N': N,
                'dx': dx,
                'dt': dt,
                'model': model_name,
                'overall_mape': round(overall, 2),
                'per_station': {k: round(v, 2) for k, v in gs_errors.items()},
                'n_steps': int(args.t_end / dt),
            })
        except Exception as e:
            print(f"\n  FAILED: {e}")
            results.append({
                'N': N_expected,
                'dx': 4.096 / N_expected,
                'dt': dt,
                'model': model_name,
                'overall_mape': float('nan'),
                'per_station': {},
                'error': str(e),
            })

    # ── Summary table ──
    print(f"\n\n{'='*70}")
    print("MESH CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'N':>5s} {'dx (m)':>10s} {'dt (µs)':>10s} {'Steps':>8s} {'MAPE':>8s} {'GS-1':>8s} {'GS-7':>8s}")
    print("-" * 65)
    for r in results:
        gs1 = r['per_station'].get('GS-1', float('nan'))
        gs7 = r['per_station'].get('GS-7', float('nan'))
        n_steps = r.get('n_steps', 0)
        print(f"{r['N']:5d} {r['dx']:10.4f} {r['dt']*1e6:10.1f} {n_steps:8d} "
              f"{r['overall_mape']:7.1f}% {gs1:7.1f}% {gs7:7.1f}%")

    # Check convergence: is MAPE decreasing or plateauing?
    valid = [r for r in results if not np.isnan(r['overall_mape'])]
    if len(valid) >= 2:
        mapes = [r['overall_mape'] for r in valid]
        if mapes[-1] <= mapes[0] + 2.0:
            print(f"\nConvergence: MAPE {'decreases' if mapes[-1] < mapes[0] else 'plateaus'} "
                  f"from {mapes[0]:.1f}% (N={valid[0]['N']}) to {mapes[-1]:.1f}% (N={valid[-1]['N']})")
        else:
            print(f"\nWARNING: MAPE increases from {mapes[0]:.1f}% to {mapes[-1]:.1f}% "
                  f"— possible stability issue at finer mesh")

    # ── Save convergence data ──
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        conv_path = RESULTS_DIR / "convergence_results.json"
        with open(conv_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {conv_path}")

        # Also save as npz for easy plotting
        npz_path = RESULTS_DIR / "convergence_results.npz"
        N_vals = np.array([r['N'] for r in valid])
        dx_vals = np.array([r['dx'] for r in valid])
        mape_vals = np.array([r['overall_mape'] for r in valid])
        np.savez(npz_path, N=N_vals, dx=dx_vals, mape=mape_vals)
        print(f"  Saved: {npz_path}")


if __name__ == '__main__':
    main()
