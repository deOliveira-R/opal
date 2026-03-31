#!/usr/bin/env python3
"""
edwards_jfnk_validation.py — Edwards blowdown with JFNK solver.

Jacobian-Free Newton-Krylov solver eliminates operator splitting between
pressure and energy equations. Uses V11 semi-implicit solver as warm start.

ALL physics from Modelica: metastable T_l/rho_l/rho_v, drift-flux phasic split
(C_0, V_gj), interfacial HT, Martinelli-Nelson friction, critical flow.
Solver provides ONLY: Newton iteration, GMRES, finite-difference JVP.

Usage:
    python edwards_jfnk_validation.py [options]

Options:
    --model {hf_ramp,...}     Model configuration (default: hf_ramp)
    --dt FLOAT                Timestep in seconds (default: 5e-5)
    --t-end FLOAT             End time in seconds (default: 0.6)
    --newton-tol FLOAT        Newton convergence tolerance (default: 1e-6)
    --newton-maxiter INT      Max Newton iterations per step (default: 10)
    --gmres-maxiter INT       Max GMRES iterations per Newton step (default: 30)
    --no-warmstart            Disable V11 warm start (use old state as initial guess)
    --save                    Save .npz + MAPE JSON to results directory
    --plot                    Generate plots after run
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
from partitioner.jfnk_solver import JFNKSolver
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d

# ── Model registry (same as edwards_bridge_5eq_validation.py) ──
RESULTS_DIR = OPAL_ROOT / "feasibility" / "results"
MODEL_REGISTRY = {
    'ramp': (
        'opal_bridge_EdwardsTest_DriftFlux_Ramp.so',
        'EdwardsTest_DriftFlux_Ramp_info.json',
        'EdwardsTest_DriftFlux_Ramp.xml',
    ),
    'hf_ramp': (
        'opal_bridge_EdwardsTest_DriftFlux_HF_Ramp.so',
        'EdwardsTest_DriftFlux_HF_Ramp_info.json',
        'EdwardsTest_DriftFlux_HF_Ramp.xml',
    ),
    'hf_ramp_flash': (
        'opal_bridge_EdwardsTest_DriftFlux_HF_Ramp_Flash.so',
        'EdwardsTest_DriftFlux_HF_Ramp_Flash_info.json',
        'EdwardsTest_DriftFlux_HF_Ramp_Flash.xml',
    ),
}
MODELICA_RAMP_MODELS = set(MODEL_REGISTRY.keys())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Edwards blowdown — JFNK solver (Newton-GMRES)")
    parser.add_argument('--model', choices=list(MODEL_REGISTRY.keys()),
                        default='hf_ramp', help='Model configuration')
    parser.add_argument('--dt', type=float, default=5e-5,
                        help='Timestep in seconds (default: 5e-5)')
    parser.add_argument('--t-end', type=float, default=0.6,
                        help='End time in seconds (default: 0.6)')
    parser.add_argument('--newton-tol', type=float, default=1e-6,
                        help='Newton convergence tolerance')
    parser.add_argument('--newton-maxiter', type=int, default=10,
                        help='Max Newton iterations per timestep')
    parser.add_argument('--gmres-maxiter', type=int, default=30,
                        help='Max GMRES iterations per Newton step')
    parser.add_argument('--no-warmstart', action='store_true',
                        help='Disable V11 warm start')
    parser.add_argument('--save', action='store_true',
                        help='Save .npz and MAPE JSON to results directory')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after run')
    return parser.parse_args(argv)


def run_validation(args):
    """Run Edwards blowdown validation with JFNK solver."""
    model_name = args.model
    dt = args.dt
    t_end = args.t_end

    # ── Resolve paths ──
    so_file, info_file, xml_file = MODEL_REGISTRY[model_name]
    bridge_so = RESULTS_DIR / so_file
    info_json = RESULTS_DIR / info_file
    edwards_xml = RESULTS_DIR / xml_file
    if not edwards_xml.exists():
        base_name = xml_file.replace('.xml', '_backEnd.xml')
        edwards_xml = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / base_name

    for path, name in [(bridge_so, "Bridge .so"), (info_json, "Info JSON")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    crit_flow = "Henry-Fauske" if 'hf' in model_name else "Ransom-Trapp"

    print("=" * 70)
    print("Edwards Blowdown — JFNK SOLVER (Newton-GMRES)")
    print(f"  Model: {model_name} | Critical flow: {crit_flow}")
    print(f"  Newton tol: {args.newton_tol}, maxiter: {args.newton_maxiter}")
    print(f"  GMRES maxiter: {args.gmres_maxiter}")
    print(f"  V11 warm start: {'ON' if not args.no_warmstart else 'OFF'}")
    print(f"  ALL physics from Modelica — solver provides ONLY Newton+GMRES")
    print("=" * 70)

    # ── Load model ──
    info = parse_info_json(info_json)
    bridge = OMEquationBridge(bridge_so, info)
    N = bridge.N

    if not edwards_xml.exists():
        raise FileNotFoundError(f"Edwards XML not found: {edwards_xml}")
    es = load_equation_system(str(edwards_xml))
    spec = map_pipe1d(es)

    solver = JFNKSolver(
        bridge, spec, es=es,
        newton_tol=args.newton_tol,
        newton_maxiter=args.newton_maxiter,
        gmres_maxiter=args.gmres_maxiter,
        use_v11_warmstart=not args.no_warmstart)

    print(f"\nModel: JFNK 5-eq drift-flux, N={N}")
    print(f"  {info.summary()}")
    print(f"  Bridge has mdot_v: {bridge.has('mdot_v')}, "
          f"mdot_l: {bridge.has('mdot_l')}")

    # ── Load experimental data ──
    from edwards_blowdown_data import edwards_blowdown
    import iapws

    # ── Initial conditions ──
    p_init = 7e6
    p = np.full(N, p_init)
    alpha = np.full(N, 1e-6)
    mdot = np.zeros(N + 1)
    h_l = np.full(N, 986.6e3)
    h_v_sat = iapws.IAPWS97(P=p_init / 1e6, x=1).h * 1e3
    h_v = np.full(N, h_v_sat)

    n_steps = int(t_end / dt)

    gauge_stations = edwards_blowdown["gauge_stations"]
    gs_cells = {}
    for name, gs in gauge_stations.items():
        gs_cells[name] = min(int(gs["x_m"] / spec.dx), N - 1)

    print(f"\nRunning {n_steps} steps (dt={dt*1e6:.0f}µs, t_end={t_end}s)...")
    print(f"{'step':>8s} {'t_ms':>8s} {'p_GS1':>10s} {'p_GS7':>10s} "
          f"{'a_max':>10s} {'Newton':>8s} {'GMRES':>10s} {'mdot_out':>12s}")

    save_times = np.concatenate([
        np.arange(0, 0.01, 0.0005),
        np.arange(0.01, 0.1, 0.005),
        np.arange(0.1, t_end + 0.01, 0.01),
    ])
    history = []
    next_save_idx = 0
    t = 0.0

    # Newton convergence tracking
    total_newton = 0
    total_gmres = 0
    total_bridge_evals = 0

    # ── Time integration ──
    t_wall_start = time_mod.perf_counter()

    for step_num in range(n_steps):
        while (next_save_idx < len(save_times) and
               t >= save_times[next_save_idx] - 0.5 * dt):
            history.append((t, p.copy(), alpha.copy(),
                            h_l.copy(), h_v.copy(), mdot.copy()))
            next_save_idx += 1

        solver.time = t
        solver.step(p, alpha, h_l, h_v, mdot, dt)
        t += dt

        total_newton += solver.newton_iters
        total_gmres += sum(solver.gmres_iters)
        total_bridge_evals += solver._n_bridge_evals

        if step_num % 2000 == 0 or step_num == n_steps - 1:
            p_gs1 = p[gs_cells["GS-1"]] / 1e6
            p_gs7 = p[gs_cells["GS-7"]] / 1e6
            a_max = np.max(alpha)
            n_iters = solver.newton_iters
            g_iters = solver.gmres_iters
            g_str = ','.join(str(g) for g in g_iters) if g_iters else '-'
            print(f"{step_num:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} "
                  f"{a_max:10.4f} {n_iters:8d} {g_str:>10s} {mdot[N]:12.3f}")

    history.append((t, p.copy(), alpha.copy(),
                    h_l.copy(), h_v.copy(), mdot.copy()))

    t_wall_end = time_mod.perf_counter()
    wall_time = t_wall_end - t_wall_start

    # ── Compare to experiment ──
    data_dir = OPAL_ROOT / "docs" / "validation" / "edwards" / "data"
    t_sim = np.array([rec[0] for rec in history])

    exp_files = {
        "GS-1": "fig3-gs1.csv", "GS-2": "fig4-gs2.csv",
        "GS-3": "fig5-gs3.csv", "GS-4": "fig6-gs4.csv",
        "GS-5": "fig7-gs5.csv", "GS-6": "fig8-gs6.csv",
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

    # ── Void fraction comparison at GS-5 ──
    void_exp_path = data_dir / "fig14-gs5.csv"
    void_mae = float('nan')
    void_onset_ms = float('nan')
    if void_exp_path.exists():
        void_data = np.loadtxt(void_exp_path, delimiter=",")
        t_void_exp = void_data[:, 0]
        alpha_void_exp = np.clip(void_data[:, 1], 0.0, 1.0)

        gs5_cell = gs_cells["GS-5"]
        alpha_sim_gs5 = np.array([rec[2][gs5_cell] for rec in history])

        mask = (t_void_exp >= 0) & (t_void_exp <= t_sim[-1])
        alpha_interp = np.interp(t_void_exp[mask], t_sim, alpha_sim_gs5)
        void_mae = np.mean(np.abs(alpha_interp - alpha_void_exp[mask]))

        for idx in range(len(t_sim)):
            if alpha_sim_gs5[idx] > 0.01:
                void_onset_ms = t_sim[idx] * 1000
                break

    overall = np.mean(list(gs_errors.values())) if gs_errors else float('nan')

    print(f"\n{'='*70}")
    print(f"Edwards Blowdown — JFNK SOLVER")
    print(f"  Model: {model_name} | Critical flow: {crit_flow}")
    print(f"  V11 warm start: {'ON' if not args.no_warmstart else 'OFF'}")
    print(f"{'='*70}")
    print(f"\nMAPE by station:")
    for gs_name in exp_files:
        if gs_name in gs_errors:
            print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): "
                  f"{gs_errors[gs_name]:.1f}%")
    print(f"\n  Overall MAPE: {overall:.1f}%")

    print(f"\nVoid fraction at GS-5:")
    print(f"  Void MAE:    {void_mae:.3f}")
    print(f"  Void onset:  {void_onset_ms:.1f} ms (experiment: 9.5 ms)")

    print(f"\nNewton-GMRES statistics:")
    avg_newton = total_newton / n_steps
    avg_gmres = total_gmres / max(total_newton, 1)
    print(f"  Total Newton iterations:  {total_newton} "
          f"(avg {avg_newton:.1f}/step)")
    print(f"  Total GMRES iterations:   {total_gmres} "
          f"(avg {avg_gmres:.1f}/Newton)")
    print(f"  Total bridge evaluations: {total_bridge_evals} "
          f"(avg {total_bridge_evals/n_steps:.0f}/step)")

    print(f"\nPerformance:")
    print(f"  Wall time: {wall_time:.1f}s ({n_steps/wall_time:.1f} steps/s, "
          f"{wall_time/n_steps*1e3:.1f} ms/step)")

    print(f"\nComparison:")
    print(f"  V11 (tau=4.5e-4, isentropic): 27.1% MAPE, "
          f"0.258 VoidMAE, 5ms onset")
    print(f"  JFNK:                         {overall:.1f}% MAPE, "
          f"{void_mae:.3f} VoidMAE, {void_onset_ms:.0f}ms onset")

    # ── Save results ──
    if args.save:
        save_dir = OPAL_ROOT / "docs" / "validation" / "edwards" / "results"
        save_dir = save_dir / "jfnk"
        save_dir.mkdir(parents=True, exist_ok=True)

        npz_path = save_dir / f"edwards_jfnk_{model_name}_N{N}.npz"
        np.savez(npz_path,
                 t=np.array([r[0] for r in history]),
                 p=np.array([r[1] for r in history]),
                 alpha=np.array([r[2] for r in history]),
                 h_l=np.array([r[3] for r in history]),
                 h_v=np.array([r[4] for r in history]),
                 mdot=np.array([r[5] for r in history]),
                 dx=np.array([spec.dx]),
                 N=np.array([N]),
                 model=np.array([model_name]),
                 dt=np.array([dt]),
                 t_end=np.array([t_end]),
                 solver=np.array(['jfnk']))
        print(f"\n  Saved: {npz_path}")

        mape_path = save_dir / f"mape_jfnk_{model_name}_N{N}.json"
        mape_data = {
            "solver": "jfnk",
            "model": model_name,
            "critical_flow": crit_flow,
            "N": N,
            "dt": dt,
            "t_end": t_end,
            "overall_mape": round(overall, 2),
            "per_station": {k: round(v, 2) for k, v in gs_errors.items()},
            "void_mae": round(void_mae, 3),
            "void_onset_ms": round(void_onset_ms, 1),
            "wall_time_s": round(wall_time, 2),
            "avg_newton_per_step": round(avg_newton, 2),
            "avg_gmres_per_newton": round(avg_gmres, 2),
            "total_bridge_evals": total_bridge_evals,
        }
        with open(mape_path, 'w') as f:
            json.dump(mape_data, f, indent=2)
        print(f"  Saved: {mape_path}")

    return overall, gs_errors, history, N, spec


if __name__ == '__main__':
    args = parse_args()
    run_validation(args)
