#!/usr/bin/env python3
"""
edwards_6eq_sweep.py — Darwinian sweep of 6-equation two-fluid solver variants.

Round 2: d_b sweep + BFL (mixture basis) + ALPHA_MAX + dGamma/dp diagonal only
         (Schur RHS proven to double-count with h_mix diagonal — excluded)

Usage:
    python edwards_6eq_sweep.py
"""

import sys
import json
import time as time_mod
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

RESULTS_DIR = OPAL_ROOT / "feasibility" / "results"

# ── Sweep configurations ──
# (bridge_tag, solver_kwargs, description)
SWEEP_CONFIGS = [
    # ── R1 reference ──
    ('baseline',  {'schur_rhs': False, 'dgamma_augment': False, 'break_form_loss': False, 'alpha_max': 0.95},
     'R1 reference (40.9%)'),

    # ── Alpha max ablation ──
    ('baseline',  {'schur_rhs': False, 'dgamma_augment': False, 'break_form_loss': False, 'alpha_max': 0.999},
     'α_max=0.999 only'),

    # ── dGamma/dp diagonal only (no Schur RHS) ──
    ('baseline',  {'schur_rhs': False, 'dgamma_augment': True, 'break_form_loss': False, 'alpha_max': 0.999},
     'dΓ/dp diagonal + α_max=0.999'),

    # ── BFL (mixture basis) ──
    ('baseline',  {'schur_rhs': False, 'dgamma_augment': False, 'break_form_loss': True, 'alpha_max': 0.999},
     'BFL (mixture) + α_max=0.999'),

    # ── Full combo: dGamma/dp + BFL + alpha_max ──
    ('baseline',  {'schur_rhs': False, 'dgamma_augment': True, 'break_form_loss': True, 'alpha_max': 0.999},
     'dΓ/dp + BFL + α_max=0.999'),

    # ── d_b sweep (stronger geometric HT) ──
    ('db1e4',     {'schur_rhs': False, 'dgamma_augment': False, 'break_form_loss': False, 'alpha_max': 0.999},
     'd_b=1e-4 (3x HT), α_max=0.999'),
    ('db5e5',     {'schur_rhs': False, 'dgamma_augment': False, 'break_form_loss': False, 'alpha_max': 0.999},
     'd_b=5e-5 (6x HT), α_max=0.999'),

    # ── d_b + dGamma/dp + BFL ──
    ('db1e4',     {'schur_rhs': False, 'dgamma_augment': True, 'break_form_loss': True, 'alpha_max': 0.999},
     'd_b=1e-4 + dΓ/dp + BFL'),
    ('db5e5',     {'schur_rhs': False, 'dgamma_augment': True, 'break_form_loss': True, 'alpha_max': 0.999},
     'd_b=5e-5 + dΓ/dp + BFL'),

    # ── J/L re-test with dGamma/dp + BFL ──
    ('jl_c0',     {'schur_rhs': False, 'dgamma_augment': True, 'break_form_loss': True, 'alpha_max': 0.999},
     'J/L C_tau=0 + dΓ/dp + BFL'),
    ('jl_c5',     {'schur_rhs': False, 'dgamma_augment': True, 'break_form_loss': True, 'alpha_max': 0.999},
     'J/L C_tau=5 + dΓ/dp + BFL'),
    ('jl_c10',    {'schur_rhs': False, 'dgamma_augment': True, 'break_form_loss': True, 'alpha_max': 0.999},
     'J/L C_tau=10 + dΓ/dp + BFL'),
]


def run_one_variant(bridge_tag, solver_kwargs, dt=5e-5, t_end=0.6):
    """Run Edwards blowdown with one 6-eq variant."""
    model_name = f'EdwardsTest_TwoFluid_{bridge_tag}'
    bridge_so = RESULTS_DIR / f"opal_bridge_{model_name}.so"
    info_json = RESULTS_DIR / f"{model_name}_info.json"
    xml_path = RESULTS_DIR / f"{model_name}.xml"

    if not bridge_so.exists() or not xml_path.exists():
        return None, {}, 0.0

    info = parse_info_json(info_json)
    bridge = OMEquationBridge(bridge_so, info)
    N = bridge.N

    es = load_equation_system(str(xml_path))
    spec = map_pipe1d(es)

    solver = BridgeTwoFluidSolver(bridge, spec, es=es, **solver_kwargs)

    import iapws
    p = np.full(N, 7e6)
    alpha = np.full(N, 1e-6)
    mdot_l = np.zeros(N + 1)
    mdot_v = np.zeros(N + 1)
    h_l = np.full(N, 986.6e3)
    h_v = np.full(N, iapws.IAPWS97(P=7.0, x=1).h * 1e3)

    n_steps = int(t_end / dt)

    from edwards_blowdown_data import edwards_blowdown
    gauge_stations = edwards_blowdown["gauge_stations"]
    gs_cells = {k: min(int(v["x_m"] / spec.dx), N - 1) for k, v in gauge_stations.items()}

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
            history.append((t, p.copy()))
            next_save_idx += 1

        solver.time = t
        try:
            solver.step(p, alpha, h_l, h_v, mdot_l, mdot_v, dt)
        except Exception:
            break
        t += dt

    history.append((t, p.copy()))
    wall_time = time_mod.perf_counter() - t_wall_start

    # Compute MAPE
    data_dir = OPAL_ROOT / "docs" / "validation" / "edwards" / "data"
    t_sim = np.array([rec[0] for rec in history])
    exp_files = {
        "GS-1": "fig3-gs1.csv", "GS-2": "fig4-gs2.csv", "GS-3": "fig5-gs3.csv",
        "GS-4": "fig6-gs4.csv", "GS-5": "fig7-gs5.csv", "GS-6": "fig8-gs6.csv",
        "GS-7": "fig9-gs7.csv",
    }
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
        for j in range(len(t_exp)):
            p_interp = np.interp(t_exp[j], t_sim, p_sim)
            if p_exp_MPa[j] > 0.1:
                errors.append(abs((p_interp - p_exp_MPa[j]) / p_exp_MPa[j] * 100))
        if errors:
            gs_errors[gs_name] = np.mean(errors)

    overall = np.mean(list(gs_errors.values())) if gs_errors else float('nan')
    return overall, gs_errors, wall_time


def main():
    print("=" * 90)
    print("6-EQUATION TWO-FLUID DARWINIAN SWEEP — Round 2")
    print("  h_mix diagonal + dΓ/dp augment + mixture BFL + α_max + d_b sweep")
    print("=" * 90)
    print()

    results = []

    for idx, (bridge_tag, kwargs, desc) in enumerate(SWEEP_CONFIGS):
        label = f"[{idx+1:2d}/{len(SWEEP_CONFIGS)}]"
        print(f"  {label} {desc:45s} ...", end="", flush=True)
        try:
            mape, per_station, wt = run_one_variant(bridge_tag, kwargs)
            if mape is not None:
                results.append({
                    'bridge': bridge_tag, 'desc': desc,
                    'mape': mape, 'per_station': per_station, 'wall_time': wt,
                })
                print(f"  MAPE={mape:5.1f}%  ({wt:.1f}s)")
            else:
                print(f"  SKIPPED (files missing)")
        except Exception as e:
            print(f"  FAILED: {e}")

    # Summary table
    print()
    print("=" * 90)
    print(f"{'#':>3s} {'MAPE':>7s} {'GS-1':>7s} {'GS-3':>7s} {'GS-5':>7s} {'GS-7':>7s} {'Time':>6s}  Description")
    print("-" * 90)

    for idx, r in enumerate(results):
        ps = r.get('per_station', {})
        mape = r.get('mape', float('nan'))
        wt = r.get('wall_time', 0)
        print(f"{idx+1:3d} {mape:6.1f}% "
              f"{ps.get('GS-1', float('nan')):6.1f}% "
              f"{ps.get('GS-3', float('nan')):6.1f}% "
              f"{ps.get('GS-5', float('nan')):6.1f}% "
              f"{ps.get('GS-7', float('nan')):6.1f}% "
              f"{wt:5.1f}s  {r['desc']}")

    print("-" * 90)
    print(f"    {'22.7%':>6s}                                          5-eq V24+AC10 (production)")
    print(f"    {'~20%':>6s}                                          RELAP5 (industry reference)")

    out_path = OPAL_ROOT / "docs" / "validation" / "edwards" / "results" / "6eq_sweep_r2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == '__main__':
    main()
