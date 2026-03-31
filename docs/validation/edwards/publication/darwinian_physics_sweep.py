#!/usr/bin/env python3
"""Darwinian physics sweep: 3 fixes, 8 combinations.

Fix 1: Acoustic choking limit on critical flow (use_acoustic_cf_limit=1)
Fix 2: Regime-dependent interfacial area+HTC (use_regime_iac=1)
Fix 4: Break area reduction 13% (C_d=0.531, C_d_final=0.531)

Compiles a model variant for each combination, runs Edwards blowdown
with V11 solver (tau_mix=4.5e-4, use_isentropic_a11=True), compares.
"""

import sys
import pathlib
import json
import time as time_mod
import tempfile
import numpy as np

# ============================================================================
# Paths
# ============================================================================
HERE = pathlib.Path(__file__).resolve().parent
OPAL_ROOT = HERE.parents[3]
SOLVER_ROOT = OPAL_ROOT / "solver"
EDWARDS_DIR = HERE.parent
DATA_DIR = EDWARDS_DIR / "data"
RESULTS_DIR = EDWARDS_DIR / "results"
PUB_DIR = HERE
FEASIBILITY_RESULTS = OPAL_ROOT / "feasibility" / "results"

# Add solver paths
sys.path.insert(0, str(SOLVER_ROOT / "two_phase"))
sys.path.insert(0, str(SOLVER_ROOT))
sys.path.insert(0, str(DATA_DIR))

# ============================================================================
# Constants
# ============================================================================
PSIA_TO_MPA = 6894.76 / 1e6

gs_x = {
    "GS-1": 3.927, "GS-2": 3.769, "GS-3": 2.935, "GS-4": 2.024,
    "GS-5": 1.469, "GS-6": 0.914, "GS-7": 0.079,
}
station_order = ["GS-1", "GS-2", "GS-3", "GS-4", "GS-5", "GS-6", "GS-7"]

exp_files = {
    "GS-1": "fig3-gs1.csv", "GS-2": "fig4-gs2.csv", "GS-3": "fig5-gs3.csv",
    "GS-4": "fig6-gs4.csv", "GS-5": "fig7-gs5.csv", "GS-6": "fig8-gs6.csv",
    "GS-7": "fig9-gs7.csv",
}

# 8 combinations: (label, fix1, fix2, fix4)
COMBOS = [
    ("baseline", False, False, False),
    ("F1",       True,  False, False),
    ("F2",       False, True,  False),
    ("F4",       False, False, True),
    ("F12",      True,  True,  False),
    ("F14",      True,  False, True),
    ("F24",      False, True,  True),
    ("F124",     True,  True,  True),
]


# ============================================================================
# Model generation
# ============================================================================
def make_model_string(label, fix1, fix2, fix4):
    """Generate Edwards test model with selected fixes."""
    cd = 0.531 if fix4 else 0.61
    regime = 1 if fix2 else 0
    acoustic = 1 if fix1 else 0
    # Use unique model name per variant to prevent OM caching
    model_name = f"EdwardsSweep_{label}"

    return model_name, f"""model {model_name}
  "Edwards blowdown: Darwinian sweep variant ({label})"
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.Pipe1D_DriftFlux pipe(
    redeclare package Medium = library.Media.Water,
    N=24, L=4.096, D=0.073, f_D=0.02,
    p_init=7e6, h_l_init=986.6e3, h_v_init=2772.6e3, alpha_init=1e-6,
    d_b=3e-4, C_0=1.0, alpha_nucleation=1e-3,
    use_relaxation=1, tau_flash=0.025, tau_flash_n=1, tau_flash_DT_ref=40.0,
    use_regime_iac={regime},
    use_critical_flow=true,
    critical_flow_model=2.0,
    C_d={cd}, x_ne=0.14, N_param=0.0, c_floor=10.0,
    use_acoustic_cf_limit={acoustic},
    use_two_phase_friction=true);
  library.Boundary.RampedBreak break_bc(
    p_back=101325.0,
    C_d_final={cd},
    t_open=0.003,
    h_set=986.6e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, break_bc.port);
  pipe.C_d_eff = break_bc.C_d;
end {model_name};
"""


# ============================================================================
# Compile a sweep variant
# ============================================================================
def compile_variant(label, fix1, fix2, fix4):
    """Compile a model variant and return (so_path, info_json_path).

    Writes the model string to a temp .mo file (to avoid OM loadString
    escaping issues with double-quoted annotation strings) and calls
    translate_and_build with model_file.
    """
    from partitioner.codegen.translate_model import translate_and_build

    model_name, model_str = make_model_string(label, fix1, fix2, fix4)

    # Write to temp .mo file — use unique model name to prevent OM caching
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"opal_sweep_{label}_"))
    mo_path = tmp_dir / f"{model_name}.mo"
    mo_path.write_text(model_str)

    print(f"\n--- Compiling variant '{label}' ---")
    print(f"  F1(acoustic)={fix1}, F2(regime)={fix2}, F4(Cd=0.531)={fix4}")
    print(f"  Model file: {mo_path}")

    so_path, info_json = translate_and_build(
        model_name,
        model_file=mo_path,
    )
    return so_path, info_json


# ============================================================================
# Run a single Edwards blowdown simulation
# ============================================================================
def run_edwards_sim(label, so_path, info_json, dt=5e-5, t_end=0.6):
    """Run Edwards blowdown with V11 solver. Returns metrics dict."""
    from partitioner.codegen.info_parser import parse_info_json
    from partitioner.codegen.equation_bridge import OMEquationBridge
    from partitioner.bridge_5eq_solver_v11_a12mod import BridgeDriftFluxSolver
    from partitioner.xml_reader import load_equation_system
    from partitioner.pipe1d_mapper import map_pipe1d

    tau_mix = 4.5e-4
    use_isentropic = True

    print(f"\n{'='*70}")
    print(f"Edwards Blowdown — combo: {label}")
    print(f"  tau_mix={tau_mix:.1e}, use_isentropic_a11={use_isentropic}")
    print(f"  dt={dt*1e6:.0f}us, t_end={t_end}s")
    print(f"{'='*70}")

    # Load model
    info = parse_info_json(info_json)
    bridge = OMEquationBridge(so_path, info)
    N = bridge.N

    # Use existing XML for equation system (geometry is the same for all combos)
    edwards_xml = FEASIBILITY_RESULTS / "EdwardsTest_DriftFlux_HF_Ramp_Flash.xml"
    if not edwards_xml.exists():
        raise FileNotFoundError(f"Edwards XML not found: {edwards_xml}")
    es = load_equation_system(str(edwards_xml))
    spec = map_pipe1d(es)

    # Create V11 solver
    solver = BridgeDriftFluxSolver(bridge, spec, es=es,
                                   tau_mix=tau_mix,
                                   use_isentropic_a11=use_isentropic)

    print(f"  Model loaded: N={N}, dx={spec.dx:.4f}m")

    # Initial conditions
    p_init = 7e6
    p = np.full(N, p_init)
    alpha = np.full(N, 1e-6)
    mdot = np.zeros(N + 1)
    h_l = np.full(N, 986.6e3)

    # h_v from IAPWS saturation
    try:
        import iapws
        h_v_sat = iapws.IAPWS97(P=p_init / 1e6, x=1).h * 1e3
    except ImportError:
        h_v_sat = 2772.6e3
    h_v = np.full(N, h_v_sat)

    n_steps = int(t_end / dt)

    # Gauge station cell indices
    gs_cells = {name: min(int(gs_x[name] / spec.dx), N - 1)
                for name in gs_x}

    # Save times (sparse sampling)
    save_times = np.concatenate([
        np.arange(0, 0.01, 0.0005),
        np.arange(0.01, 0.1, 0.005),
        np.arange(0.1, t_end + 0.01, 0.01),
    ])

    print(f"  Running {n_steps} steps...")
    print(f"  {'step':>8s} {'t_ms':>8s} {'p_GS1':>10s} {'p_GS7':>10s} "
          f"{'a_max':>10s} {'mdot_out':>12s}")

    history = []
    next_save_idx = 0
    t = 0.0

    t_wall_start = time_mod.perf_counter()

    for step in range(n_steps):
        # Save snapshots at sparse time points
        while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5 * dt:
            history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))
            next_save_idx += 1

        # Set simulation time so Modelica RampedBreak evaluates correctly
        solver.time = t

        # Step the solver
        solver.step(p, alpha, h_l, h_v, mdot, dt)
        t += dt

        # Progress logging
        if step % 2000 == 0 or step == n_steps - 1:
            p_gs1 = p[gs_cells["GS-1"]] / 1e6
            p_gs7 = p[gs_cells["GS-7"]] / 1e6
            a_max = np.max(alpha)
            print(f"  {step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} "
                  f"{a_max:10.4f} {mdot[N]:12.3f}")

    # Final snapshot
    history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))

    t_wall_end = time_mod.perf_counter()
    wall_time = t_wall_end - t_wall_start

    # ── Compute MAPE ──
    t_sim = np.array([rec[0] for rec in history])
    gs_errors = {}

    for gs_name, filename in exp_files.items():
        exp_path = DATA_DIR / filename
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

    overall_mape = np.mean(list(gs_errors.values())) if gs_errors else float('nan')

    # ── Void fraction comparison at GS-5 ──
    void_exp_path = DATA_DIR / "fig14-gs5.csv"
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

        # Void onset: first time alpha > 0.01
        for idx in range(len(t_sim)):
            if alpha_sim_gs5[idx] > 0.01:
                void_onset_ms = t_sim[idx] * 1000
                break

    # ── Print per-combo summary ──
    print(f"\n  Overall MAPE: {overall_mape:.1f}%")
    for gs in station_order:
        if gs in gs_errors:
            print(f"    {gs}: {gs_errors[gs]:.1f}%")
    print(f"  Void MAE (GS-5): {void_mae:.3f}")
    print(f"  Void onset: {void_onset_ms:.1f} ms")
    print(f"  Wall time: {wall_time:.2f}s ({n_steps/wall_time:.0f} steps/s)")

    return {
        "label": label,
        "overall_mape": overall_mape,
        "per_station": dict(gs_errors),
        "void_mae": void_mae,
        "onset_ms": void_onset_ms,
        "wall_time": wall_time,
    }


# ============================================================================
# Restore the production .so by recompiling the original model
# ============================================================================
def restore_production_model():
    """Recompile the original EdwardsTest_DriftFlux_HF_Ramp_Flash model."""
    from partitioner.codegen.translate_model import translate_and_build

    original_mo = OPAL_ROOT / "feasibility" / "models" / "EdwardsTest_DriftFlux_HF_Ramp_Flash.mo"
    if not original_mo.exists():
        print(f"WARNING: Cannot restore production model — {original_mo} not found")
        return

    print(f"\n{'='*70}")
    print("Restoring production model (EdwardsTest_DriftFlux_HF_Ramp_Flash)...")
    print(f"{'='*70}")
    try:
        translate_and_build("EdwardsTest_DriftFlux_HF_Ramp_Flash", model_file=original_mo)
        print("  Production model restored successfully.")
    except Exception as e:
        print(f"  WARNING: Failed to restore production model: {e}")


# ============================================================================
# Print comparison table
# ============================================================================
def print_results_table(results):
    """Print comprehensive comparison table."""
    print(f"\n{'='*100}")
    print(f"Darwinian Physics Sweep -- Edwards Blowdown")
    print(f"V11 solver, tau_mix=4.5e-4, isentropic A11")
    print(f"{'='*100}")

    header = (f"{'Combo':<10s} {'F1':>3s} {'F2':>3s} {'F4':>3s}   "
              f"{'MAPE':>6s}  {'GS-1':>6s} {'GS-2':>6s} {'GS-3':>6s} "
              f"{'GS-5':>6s} {'GS-7':>6s}  "
              f"{'VoidMAE':>8s} {'Onset':>7s} {'Time':>6s}")
    print(header)
    print("-" * 100)

    for combo_def, metrics in zip(COMBOS, results):
        label, f1, f2, f4 = combo_def

        f1_s = "X" if f1 else "-"
        f2_s = "X" if f2 else "-"
        f4_s = "X" if f4 else "-"

        mape = metrics["overall_mape"]
        gs1 = metrics["per_station"].get("GS-1", float('nan'))
        gs2 = metrics["per_station"].get("GS-2", float('nan'))
        gs3 = metrics["per_station"].get("GS-3", float('nan'))
        gs5 = metrics["per_station"].get("GS-5", float('nan'))
        gs7 = metrics["per_station"].get("GS-7", float('nan'))
        vmae = metrics["void_mae"]
        onset = metrics["onset_ms"]
        wtime = metrics["wall_time"]

        onset_s = f"{onset:.0f}ms" if not np.isnan(onset) else "N/A"

        print(f"{label:<10s} {f1_s:>3s} {f2_s:>3s} {f4_s:>3s}   "
              f"{mape:5.1f}%  {gs1:5.1f}% {gs2:5.1f}% {gs3:5.1f}% "
              f"{gs5:5.1f}% {gs7:5.1f}%  "
              f"{vmae:8.3f} {onset_s:>7s} {wtime:5.0f}s")

    print("=" * 100)

    # ── Individual contributions ──
    baseline_mape = results[0]["overall_mape"]
    baseline_vmae = results[0]["void_mae"]

    print(f"\nIndividual contributions (vs baseline MAPE={baseline_mape:.1f}%, VoidMAE={baseline_vmae:.3f}):")
    fix_labels = {
        "F1": "Fix 1 (acoustic limit)",
        "F2": "Fix 2 (regime IAC)",
        "F4": "Fix 4 (break area)",
    }
    individual_deltas = {}
    for combo_def, metrics in zip(COMBOS[1:4], results[1:4]):
        label = combo_def[0]
        delta_mape = metrics["overall_mape"] - baseline_mape
        delta_vmae = metrics["void_mae"] - baseline_vmae
        individual_deltas[label] = delta_mape
        sign_m = "+" if delta_mape >= 0 else ""
        sign_v = "+" if delta_vmae >= 0 else ""
        print(f"  {fix_labels[label]:<28s}: MAPE {sign_m}{delta_mape:.1f}%  "
              f"VoidMAE {sign_v}{delta_vmae:.3f}")

    # ── Synergy analysis ──
    print(f"\nSynergy analysis:")

    # F12 vs F1+F2 predicted
    f12_actual = results[4]["overall_mape"]
    f12_predicted = baseline_mape + individual_deltas.get("F1", 0) + individual_deltas.get("F2", 0)
    f12_synergy = f12_actual - f12_predicted
    print(f"  F12 actual={f12_actual:.1f}%  predicted(F1+F2)={f12_predicted:.1f}%  "
          f"synergy={f12_synergy:+.1f}%")

    # F14 vs F1+F4 predicted
    f14_actual = results[5]["overall_mape"]
    f14_predicted = baseline_mape + individual_deltas.get("F1", 0) + individual_deltas.get("F4", 0)
    f14_synergy = f14_actual - f14_predicted
    print(f"  F14 actual={f14_actual:.1f}%  predicted(F1+F4)={f14_predicted:.1f}%  "
          f"synergy={f14_synergy:+.1f}%")

    # F24 vs F2+F4 predicted
    f24_actual = results[6]["overall_mape"]
    f24_predicted = baseline_mape + individual_deltas.get("F2", 0) + individual_deltas.get("F4", 0)
    f24_synergy = f24_actual - f24_predicted
    print(f"  F24 actual={f24_actual:.1f}%  predicted(F2+F4)={f24_predicted:.1f}%  "
          f"synergy={f24_synergy:+.1f}%")

    # F124 vs sum of all individuals
    f124_actual = results[7]["overall_mape"]
    f124_predicted = baseline_mape + sum(individual_deltas.values())
    f124_synergy = f124_actual - f124_predicted
    print(f"  F124 actual={f124_actual:.1f}%  predicted(F1+F2+F4)={f124_predicted:.1f}%  "
          f"synergy={f124_synergy:+.1f}%")

    # Best combo
    best_idx = np.argmin([r["overall_mape"] for r in results])
    best = results[best_idx]
    print(f"\nBest MAPE:     {COMBOS[best_idx][0]} = {best['overall_mape']:.1f}%")

    best_void_idx = np.argmin([r["void_mae"] for r in results])
    best_void = results[best_void_idx]
    print(f"Best VoidMAE:  {COMBOS[best_void_idx][0]} = {best_void['void_mae']:.3f}")


# ============================================================================
# Main
# ============================================================================
def main():
    print(f"Darwinian Physics Sweep — 3 fixes, 8 combinations")
    print(f"OPAL root: {OPAL_ROOT}")
    print(f"Data dir:  {DATA_DIR}")
    print()

    # Verify experimental data exists
    missing = []
    for gs, fname in exp_files.items():
        if not (DATA_DIR / fname).exists():
            missing.append(fname)
    if not (DATA_DIR / "fig14-gs5.csv").exists():
        missing.append("fig14-gs5.csv")
    if missing:
        print(f"ERROR: Missing experimental data files: {missing}")
        sys.exit(1)

    # ── Phase 1: Compile all 8 variants ──
    print(f"\n{'#'*70}")
    print(f"# Phase 1: Compile 8 model variants")
    print(f"{'#'*70}")

    compiled = {}
    for label, f1, f2, f4 in COMBOS:
        try:
            so_path, info_json = compile_variant(label, f1, f2, f4)
            compiled[label] = (so_path, info_json)
            print(f"  OK: {label} -> {so_path}")
        except Exception as e:
            print(f"  FAILED: {label} -> {e}")
            compiled[label] = None

    n_compiled = sum(1 for v in compiled.values() if v is not None)
    print(f"\nCompiled {n_compiled}/{len(COMBOS)} variants successfully.")
    if n_compiled == 0:
        print("ERROR: No variants compiled. Exiting.")
        sys.exit(1)

    # ── Phase 2: Run simulations ──
    print(f"\n{'#'*70}")
    print(f"# Phase 2: Run 8 Edwards blowdown simulations")
    print(f"{'#'*70}")

    results = []
    for combo_def in COMBOS:
        label = combo_def[0]
        if compiled[label] is None:
            print(f"\n  SKIPPING {label} (compilation failed)")
            results.append({
                "label": label,
                "overall_mape": float('nan'),
                "per_station": {},
                "void_mae": float('nan'),
                "onset_ms": float('nan'),
                "wall_time": float('nan'),
            })
            continue

        so_path, info_json = compiled[label]
        try:
            metrics = run_edwards_sim(label, so_path, info_json)
            results.append(metrics)
        except Exception as e:
            print(f"\n  SIMULATION FAILED for {label}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "label": label,
                "overall_mape": float('nan'),
                "per_station": {},
                "void_mae": float('nan'),
                "onset_ms": float('nan'),
                "wall_time": float('nan'),
            })

    # ── Phase 3: Print results ──
    print_results_table(results)

    # ── Save results JSON ──
    json_out = PUB_DIR / "darwinian_sweep_results.json"
    json_data = {
        "description": "Darwinian physics sweep: 3 fixes, 8 combinations",
        "solver": "V11 (A12 moderation + isentropic A11)",
        "tau_mix": 4.5e-4,
        "dt": 5e-5,
        "t_end": 0.6,
        "combos": [],
    }
    for combo_def, metrics in zip(COMBOS, results):
        label, f1, f2, f4 = combo_def
        entry = {
            "label": label,
            "fix1_acoustic": f1,
            "fix2_regime": f2,
            "fix4_break_area": f4,
            "overall_mape": round(float(metrics["overall_mape"]), 2)
                if not np.isnan(metrics["overall_mape"]) else None,
            "per_station": {k: round(v, 2) for k, v in metrics["per_station"].items()},
            "void_mae": round(float(metrics["void_mae"]), 4)
                if not np.isnan(metrics["void_mae"]) else None,
            "onset_ms": round(float(metrics["onset_ms"]), 1)
                if not np.isnan(metrics["onset_ms"]) else None,
            "wall_time_s": round(float(metrics["wall_time"]), 2)
                if not np.isnan(metrics["wall_time"]) else None,
        }
        json_data["combos"].append(entry)

    with open(json_out, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to {json_out}")

    # ── Phase 4: Restore production model ──
    restore_production_model()

    print(f"\nDone.")


if __name__ == "__main__":
    main()
