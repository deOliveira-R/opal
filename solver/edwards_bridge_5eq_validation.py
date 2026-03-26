#!/usr/bin/env python3
"""
edwards_bridge_5eq_validation.py — Edwards blowdown with 5-equation drift-flux
via the OM equation bridge (True Case 2).

ALL physics from Modelica: metastable T_l/rho_l/rho_v, drift-flux phasic split
(C_0, V_gj), interfacial HT, Martinelli-Nelson friction, critical flow.
Solver provides ONLY: Thomas algorithm + semi-implicit splitting.

Usage:
    python edwards_bridge_5eq_validation.py [options]

Options:
    --model {ramp,base,hf,hf_ramp}  Model configuration (default: ramp)
    --dt FLOAT                       Timestep in seconds (default: 5e-5)
    --t-end FLOAT                    End time in seconds (default: 0.6)
    --temp-profile                   Use axial temperature profile from experiment
    --save                           Save .npz + MAPE JSON to results directory
    --plot                           Generate plots after run

Models:
    ramp     Ransom-Trapp + Modelica RampedBreak (canonical, all physics in Modelica)
    hf_ramp  Henry-Fauske + Modelica RampedBreak (all physics in Modelica)
    base     Ransom-Trapp + Python-side C_d ramp (legacy)
    hf       Henry-Fauske + Python-side C_d ramp (legacy)
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
from partitioner.bridge_5eq_solver import BridgeDriftFluxSolver
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d

# ── Model registry ──
# Maps model name → (so_file, info_file, xml_file) under feasibility/results/
RESULTS_DIR = OPAL_ROOT / "feasibility" / "results"
MODEL_REGISTRY = {
    'ramp': (
        'opal_bridge_EdwardsTest_DriftFlux_Ramp.so',
        'EdwardsTest_DriftFlux_Ramp_info.json',
        'EdwardsTest_DriftFlux_Ramp.xml',
    ),
    'base': (
        'opal_bridge_EdwardsTest_DriftFlux.so',
        'EdwardsTest_DriftFlux_info.json',
        'EdwardsTest_DriftFlux.xml',
    ),
    'hf': (
        'opal_bridge_EdwardsTest_DriftFlux_HF.so',
        'EdwardsTest_DriftFlux_HF_info.json',
        'EdwardsTest_DriftFlux_HF.xml',
    ),
    'hf_ramp': (
        'opal_bridge_EdwardsTest_DriftFlux_HF_Ramp.so',
        'EdwardsTest_DriftFlux_HF_Ramp_info.json',
        'EdwardsTest_DriftFlux_HF_Ramp.xml',
    ),
    # Mesh convergence variants (HF + Ramp at different N)
    'hf_ramp_n12': (
        'opal_bridge_EdwardsTest_DriftFlux_HF_Ramp_N12.so',
        'EdwardsTest_DriftFlux_HF_Ramp_N12_info.json',
        'EdwardsTest_DriftFlux_HF_Ramp_N12.xml',
    ),
    'hf_ramp_n48': (
        'opal_bridge_EdwardsTest_DriftFlux_HF_Ramp_N48.so',
        'EdwardsTest_DriftFlux_HF_Ramp_N48_info.json',
        'EdwardsTest_DriftFlux_HF_Ramp_N48.xml',
    ),
    'hf_ramp_n96': (
        'opal_bridge_EdwardsTest_DriftFlux_HF_Ramp_N96.so',
        'EdwardsTest_DriftFlux_HF_Ramp_N96_info.json',
        'EdwardsTest_DriftFlux_HF_Ramp_N96.xml',
    ),
}

# Models where the break ramp is handled by Modelica RampedBreak
MODELICA_RAMP_MODELS = {'ramp', 'hf_ramp', 'hf_ramp_n12', 'hf_ramp_n48', 'hf_ramp_n96'}
# Models where the break ramp must be applied Python-side (legacy)
PYTHON_RAMP_MODELS = {'base', 'hf'}
# Break opening time for legacy Python ramp (Edwards glass disk ~1-2ms + geometry)
T_OPEN_LEGACY = 3.0e-3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Edwards blowdown — 5-eq drift-flux via OM bridge")
    parser.add_argument('--model', choices=list(MODEL_REGISTRY.keys()),
                        default='ramp', help='Model configuration (default: ramp)')
    parser.add_argument('--dt', type=float, default=5e-5,
                        help='Timestep in seconds (default: 5e-5)')
    parser.add_argument('--t-end', type=float, default=0.6,
                        help='End time in seconds (default: 0.6)')
    parser.add_argument('--temp-profile', action='store_true',
                        help='Use axial temperature profile from experiment')
    parser.add_argument('--save', action='store_true',
                        help='Save .npz and MAPE JSON to results directory')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots after run')
    return parser.parse_args(argv)


def run_validation(args):
    """Run Edwards blowdown validation. Returns (overall_mape, gs_errors, history)."""
    model_name = args.model
    dt = args.dt
    t_end = args.t_end

    # ── Resolve paths ──
    so_file, info_file, xml_file = MODEL_REGISTRY[model_name]
    bridge_so = RESULTS_DIR / so_file
    info_json = RESULTS_DIR / info_file
    edwards_xml = RESULTS_DIR / xml_file
    if not edwards_xml.exists():
        # Fall back to validation data dir for legacy XML
        base_name = xml_file.replace('.xml', '_backEnd.xml')
        edwards_xml = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / base_name

    for path, name in [(bridge_so, "Bridge .so"), (info_json, "Info JSON")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            mo_name = xml_file.replace('.xml', '')
            print(f"Run: python -c 'from partitioner.codegen.translate_model "
                  f"import translate_and_build; translate_and_build(\"{mo_name}\")'")
            sys.exit(1)

    ramp_source = "Modelica RampedBreak" if model_name in MODELICA_RAMP_MODELS else "Python C_d_factor"
    crit_flow = "Henry-Fauske" if 'hf' in model_name else "Ransom-Trapp"

    print("=" * 70)
    print("Edwards Blowdown — 5-EQ DRIFT-FLUX via OM Bridge (True Case 2)")
    print(f"  Model: {model_name} | Critical flow: {crit_flow} | Ramp: {ramp_source}")
    print(f"  ALL physics from Modelica — solver provides ONLY numerical methods")
    print("=" * 70)

    # ── Load model ──
    info = parse_info_json(info_json)
    bridge = OMEquationBridge(bridge_so, info)
    N = bridge.N

    if not edwards_xml.exists():
        raise FileNotFoundError(f"Edwards XML not found: {edwards_xml}")
    es = load_equation_system(str(edwards_xml))
    spec = map_pipe1d(es)

    solver = BridgeDriftFluxSolver(bridge, spec, es=es)

    print(f"\nModel: 5-eq drift-flux, N={N}")
    print(f"  {info.summary()}")
    print(f"  Bridge has mdot_v: {bridge.has('mdot_v')}, mdot_l: {bridge.has('mdot_l')}")
    has_cd_eff = bridge.has('C_d_eff')
    print(f"  Bridge has C_d_eff: {has_cd_eff}")

    # ── Load experimental data ──
    from edwards_blowdown_data import edwards_blowdown
    import iapws

    # ── Initial conditions ──
    p_init = 7e6
    p = np.full(N, p_init)
    alpha = np.full(N, 1e-6)
    mdot = np.zeros(N + 1)

    ic = edwards_blowdown["initial_conditions"]
    temp_profile = ic.get("temperature_profile", None)

    if args.temp_profile and temp_profile is not None:
        x_meas = np.array([pt[1] for pt in temp_profile])
        T_meas = np.array([pt[3] for pt in temp_profile])
        dx = spec.dx
        x_cells = np.array([(i + 0.5) * dx for i in range(N)])
        T_cells = np.interp(x_cells, x_meas, T_meas)
        h_l = np.array([iapws.IAPWS97(P=p_init/1e6, T=T_cells[i]).h * 1e3
                         for i in range(N)])
        print(f"  IC: temperature profile from experiment")
        print(f"    T range: {T_cells.min():.1f} - {T_cells.max():.1f} K")
        print(f"    h_l range: {h_l.min()/1e3:.1f} - {h_l.max()/1e3:.1f} kJ/kg")
    else:
        h_l = np.full(N, 986.6e3)
        print(f"  IC: isothermal T={ic['simplified_isothermal_K']:.1f} K")

    h_v_sat = iapws.IAPWS97(P=p_init/1e6, x=1).h * 1e3
    h_v = np.full(N, h_v_sat)

    n_steps = int(t_end / dt)

    gauge_stations = edwards_blowdown["gauge_stations"]
    gs_cells = {}
    for name, gs in gauge_stations.items():
        gs_cells[name] = min(int(gs["x_m"] / spec.dx), N - 1)

    print(f"\nRunning {n_steps} steps (dt={dt*1e6:.0f}µs, t_end={t_end}s)...")
    print(f"{'step':>8s} {'t_ms':>8s} {'p_GS1':>10s} {'p_GS7':>10s} "
          f"{'a_max':>10s} {'Gam_max':>10s} {'mdot_out':>12s}")

    save_times = np.concatenate([
        np.arange(0, 0.01, 0.0005),
        np.arange(0.01, 0.1, 0.005),
        np.arange(0.1, t_end + 0.01, 0.01),
    ])
    history = []
    next_save_idx = 0
    t = 0.0

    use_python_ramp = model_name in PYTHON_RAMP_MODELS

    # ── Time integration ──
    t_wall_start = time_mod.perf_counter()

    for step in range(n_steps):
        while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
            history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))
            next_save_idx += 1

        # Set simulation time so Modelica RampedBreak evaluates correctly
        solver.time = t

        # Legacy Python-side ramp for non-Ramp models
        if use_python_ramp:
            solver.C_d_factor = min(t / T_OPEN_LEGACY, 1.0) if T_OPEN_LEGACY > 0 else 1.0

        solver.step(p, alpha, h_l, h_v, mdot, dt)
        t += dt

        if step % 2000 == 0 or step == n_steps - 1:
            p_gs1 = p[gs_cells["GS-1"]] / 1e6
            p_gs7 = p[gs_cells["GS-7"]] / 1e6
            a_max = np.max(alpha)
            try:
                Gam = bridge.get('Gamma')
                Gam_max = np.max(Gam) if Gam is not None else 0.0
            except Exception:
                Gam_max = 0.0
            cd_str = ""
            if step < 100 and has_cd_eff:
                try:
                    cd_val = bridge.get('C_d_eff')
                    cd_str = f"  C_d_eff={cd_val[0]:.4f}" if cd_val is not None else ""
                except Exception:
                    pass
            print(f"{step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} "
                  f"{a_max:10.4f} {Gam_max:10.2f} {mdot[N]:12.3f}{cd_str}")

    history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))

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

    # ── Physical indicators ──
    alpha_final = history[-1][2]
    alpha_max_final = np.max(alpha_final)
    alpha_mid_time = history[len(history)//3][2] if len(history) > 3 else alpha_final
    alpha_max_mid = np.max(alpha_mid_time)

    overall = np.mean(list(gs_errors.values())) if gs_errors else float('nan')

    print(f"\n{'='*70}")
    print(f"Edwards Blowdown — 5-EQ DRIFT-FLUX via OM Bridge")
    print(f"  Model: {model_name} | Critical flow: {crit_flow} | Ramp: {ramp_source}")
    print(f"{'='*70}")
    print(f"\nMAPE by station:")
    for gs_name in exp_files:
        if gs_name in gs_errors:
            print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): {gs_errors[gs_name]:.1f}%")
    print(f"\n  Overall MAPE: {overall:.1f}%")

    print(f"\nPhysical indicators:")
    print(f"  alpha_max (mid-time):  {alpha_max_mid:.4f}")
    print(f"  alpha_max (final):     {alpha_max_final:.4f}")
    print(f"  mdot_out (final):      {history[-1][5][N]:.3f} kg/s")

    print(f"\nPerformance:")
    print(f"  Wall time: {wall_time:.2f}s ({n_steps/wall_time:.0f} steps/s, "
          f"{wall_time/n_steps*1e6:.1f} µs/step)")

    print(f"\nComparison:")
    print(f"  HEM + IAPWS (bridge):          81.0% MAPE")
    print(f"  5-eq Case 1 (Python, no flash): 79.8% MAPE")
    print(f"  5-eq Case 1 (Python, w/ fix):   30.0% MAPE")
    print(f"  5-eq Bridge (True Case 2):     {overall:.1f}% MAPE  <-- THIS RUN ({model_name})")
    print(f"\n  Architecture: Modelica .mo → OM translateModel → bridge .so → semi-implicit solver")

    # ── Save results ──
    if args.save:
        save_dir = OPAL_ROOT / "docs" / "validation" / "edwards" / "results" / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        npz_path = save_dir / f"edwards_{model_name}_N{N}.npz"
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
                 t_end=np.array([t_end]))
        print(f"\n  Saved: {npz_path}")

        mape_path = save_dir / f"mape_{model_name}_N{N}.json"
        mape_data = {
            "model": model_name,
            "critical_flow": crit_flow,
            "ramp_source": ramp_source,
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
