#!/usr/bin/env python3
"""
generate_v11_publication.py — Generate V11 production results and publication
figures for Paper B (Edwards blowdown validation).

Runs the V11 block-coupled solver (A12 moderation + isentropic A11) at three
configurations, saves results as NPZ + MAPE JSON, generates 8 publication
figures, exports CSV data files, and updates results_tables.md.

ALL physics from Modelica — solver provides ONLY numerical methods.

Usage:
    python generate_v11_publication.py              # Full run (sims + figures)
    python generate_v11_publication.py --figures-only  # Figures from saved results
    python generate_v11_publication.py --sims-only     # Sims only, no figures
"""

import sys
import pathlib
import json
import time as time_mod
import argparse
import numpy as np

# ============================================================================
# Paths
# ============================================================================
HERE = pathlib.Path(__file__).resolve().parent
OPAL_ROOT = HERE.parents[3]
SOLVER_ROOT = OPAL_ROOT / "solver"
EDWARDS_DIR = HERE.parent
DATA_DIR = EDWARDS_DIR / "data"
RELAP_DIR = DATA_DIR / "relap5"
RESULTS_DIR = EDWARDS_DIR / "results"
PUB_DIR = HERE
EXPORT_DIR = PUB_DIR / "data_export"
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

# V11 configurations to run
V11_CONFIGS = {
    "production": {"tau_mix": 4.5e-4, "use_isentropic_a11": True},
    "balanced":   {"tau_mix": 4.0e-4, "use_isentropic_a11": True},
    "void_opt":   {"tau_mix": 2.5e-4, "use_isentropic_a11": True},
}

# Result path mapping
V11_RESULT_DIRS = {
    "production": "v11_production",
    "balanced":   "v11_balanced",
    "void_opt":   "v11_void_opt",
}
V11_NPZ_NAMES = {
    "production": "edwards_v11_prod_N24.npz",
    "balanced":   "edwards_v11_bal_N24.npz",
    "void_opt":   "edwards_v11_void_N24.npz",
}
V11_MAPE_NAMES = {
    "production": "mape_v11_prod_N24.json",
    "balanced":   "mape_v11_bal_N24.json",
    "void_opt":   "mape_v11_void_N24.json",
}

# Pareto frontier data (from tau_mix sweep with isentropic A11)
PARETO_POINTS = [
    {"tau": 3.0e-4, "mape": 52.7, "void_mae": 0.124, "label": None},
    {"tau": 3.5e-4, "mape": 36.3, "void_mae": 0.167, "label": None},
    {"tau": 4.0e-4, "mape": 27.6, "void_mae": 0.209, "label": "balanced"},
    {"tau": 4.5e-4, "mape": 27.1, "void_mae": 0.245, "label": "production"},
    {"tau": 5.0e-4, "mape": 31.0, "void_mae": 0.276, "label": None},
]
BASE_SCALAR_POINT = {"mape": 26.3, "void_mae": 0.146, "label": "h_mix (onset=140ms)"}
RELAP5_POINT = {"mape": 27.0, "void_mae": 0.111, "label": "RELAP5-3D"}

# Feature impact progression (for waterfall chart)
FEATURES = [
    ("3-eq HEM\n+ IAPWS",       81.0),
    ("+ metastable\nT_l",       30.0),
    ("+ extraction\nbridge",    36.0),
    ("+ implicit\nfriction",    28.4),
    ("+ Henry-\nFauske CF",     28.3),
    ("+ V11 block\n+ isentropic", 27.1),
]


# ============================================================================
# Part 1: Run V11 simulations
# ============================================================================
def run_v11_config(config_name, tau_mix, use_isentropic, dt=5e-5, t_end=0.6):
    """Run Edwards blowdown with V11 solver and save results.

    Returns metrics dict with MAPE, void MAE, onset time, wall time.
    """
    from partitioner.codegen.info_parser import parse_info_json
    from partitioner.codegen.equation_bridge import OMEquationBridge
    from partitioner.bridge_5eq_solver_v11_a12mod import BridgeDriftFluxSolver
    from partitioner.xml_reader import load_equation_system
    from partitioner.pipe1d_mapper import map_pipe1d

    print(f"\n{'='*70}")
    print(f"V11 Edwards Blowdown — config: {config_name}")
    print(f"  tau_mix={tau_mix:.1e}, use_isentropic_a11={use_isentropic}")
    print(f"  dt={dt*1e6:.0f}us, t_end={t_end}s")
    print(f"{'='*70}")

    # ── Resolve paths (hf_ramp_flash model) ──
    so_file = 'opal_bridge_EdwardsTest_DriftFlux_HF_Ramp_Flash.so'
    info_file = 'EdwardsTest_DriftFlux_HF_Ramp_Flash_info.json'
    xml_file = 'EdwardsTest_DriftFlux_HF_Ramp_Flash.xml'

    bridge_so = FEASIBILITY_RESULTS / so_file
    info_json = FEASIBILITY_RESULTS / info_file
    edwards_xml = FEASIBILITY_RESULTS / xml_file

    for path, name in [(bridge_so, "Bridge .so"), (info_json, "Info JSON")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    # ── Load model ──
    info = parse_info_json(info_json)
    bridge = OMEquationBridge(bridge_so, info)
    N = bridge.N

    if not edwards_xml.exists():
        raise FileNotFoundError(f"Edwards XML not found: {edwards_xml}")
    es = load_equation_system(str(edwards_xml))
    spec = map_pipe1d(es)

    # ── Create V11 solver ──
    solver = BridgeDriftFluxSolver(bridge, spec, es=es,
                                   tau_mix=tau_mix,
                                   use_isentropic_a11=use_isentropic)

    print(f"  Model loaded: N={N}, dx={spec.dx:.4f}m")
    print(f"  Bridge has mdot_v: {bridge.has('mdot_v')}, mdot_l: {bridge.has('mdot_l')}")

    # ── Initial conditions ──
    p_init = 7e6
    p = np.full(N, p_init)
    alpha = np.full(N, 1e-6)
    mdot = np.zeros(N + 1)
    h_l = np.full(N, 986.6e3)  # isothermal

    # h_v from IAPWS saturation
    try:
        import iapws
        h_v_sat = iapws.IAPWS97(P=p_init / 1e6, x=1).h * 1e3
    except ImportError:
        h_v_sat = 2772.6e3  # fallback
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

    # ── Print summary ──
    print(f"\n  Overall MAPE: {overall_mape:.1f}%")
    for gs in station_order:
        if gs in gs_errors:
            print(f"    {gs}: {gs_errors[gs]:.1f}%")
    print(f"  Void MAE (GS-5): {void_mae:.3f}")
    print(f"  Void onset: {void_onset_ms:.1f} ms")
    print(f"  Wall time: {wall_time:.2f}s ({n_steps/wall_time:.0f} steps/s)")

    # ── Save NPZ ──
    save_dir = RESULTS_DIR / V11_RESULT_DIRS[config_name]
    save_dir.mkdir(parents=True, exist_ok=True)

    npz_path = save_dir / V11_NPZ_NAMES[config_name]
    np.savez(npz_path,
             t=np.array([r[0] for r in history]),
             p=np.array([r[1] for r in history]),
             alpha=np.array([r[2] for r in history]),
             h_l=np.array([r[3] for r in history]),
             h_v=np.array([r[4] for r in history]),
             mdot=np.array([r[5] for r in history]),
             dx=np.array([spec.dx]),
             N=np.array([N]),
             model=np.array([config_name]),
             dt=np.array([dt]),
             t_end=np.array([t_end]))
    print(f"  Saved: {npz_path}")

    # ── Save MAPE JSON ──
    mape_path = save_dir / V11_MAPE_NAMES[config_name]
    mape_data = {
        "model": f"v11_{config_name}",
        "solver": "BridgeDriftFluxSolver (V11, A12 moderation)",
        "tau_mix": tau_mix,
        "use_isentropic_a11": use_isentropic,
        "critical_flow": "Henry-Fauske",
        "N": N,
        "dt": dt,
        "t_end": t_end,
        "overall_mape": round(overall_mape, 2),
        "per_station": {k: round(v, 2) for k, v in gs_errors.items()},
        "void_mae_gs5": round(float(void_mae), 4),
        "onset_ms": round(float(void_onset_ms), 1),
        "wall_time_s": round(wall_time, 2),
    }
    with open(mape_path, 'w') as f:
        json.dump(mape_data, f, indent=2)
    print(f"  Saved: {mape_path}")

    return mape_data


def run_all_v11_configs():
    """Run all three V11 configurations and return metrics dict."""
    results = {}
    for config_name, params in V11_CONFIGS.items():
        metrics = run_v11_config(
            config_name,
            tau_mix=params["tau_mix"],
            use_isentropic=params["use_isentropic_a11"],
        )
        results[config_name] = metrics
    return results


# ============================================================================
# Data loading helpers
# ============================================================================
def load_opal_results(npz_path):
    """Load OPAL results from npz. Returns dict with t, p, alpha, etc."""
    d = np.load(npz_path, allow_pickle=True)
    result = {k: d[k] for k in d.files}
    # Ensure dx and N are accessible as scalars
    result['dx_val'] = float(np.atleast_1d(result['dx'])[0])
    result['N_val'] = int(np.atleast_1d(result['N'])[0])
    # Compute gs_cells
    dx = result['dx_val']
    N = result['N_val']
    result['gs_cells'] = {gs: min(int(x / dx), N - 1) for gs, x in gs_x.items()}
    return result


def load_exp_pressure(gs_name):
    """Load experimental pressure. Returns (t_s, p_MPa)."""
    path = DATA_DIR / exp_files[gs_name]
    if not path.exists():
        return None, None
    data = np.loadtxt(path, delimiter=',')
    return data[:, 0], data[:, 1] * PSIA_TO_MPA


def load_relap5_pressure(gs_name):
    """Load RELAP5-3D pressure. Returns (t_s, p_MPa) or (None, None)."""
    relap_files = {
        "GS-1": "fig3-gs1-relap-modified.csv", "GS-2": "fig4-gs2-relap-modified.csv",
        "GS-3": "fig5-gs3-relap-modified.csv", "GS-4": "fig6-gs4-relap-modified.csv",
        "GS-5": "fig7-gs5-relap-modified.csv", "GS-6": "fig8-gs6-relap-modified.csv",
        "GS-7": "fig9-gs7-relap-modified.csv",
    }
    path = RELAP_DIR / relap_files[gs_name]
    if not path.exists():
        return None, None
    data = np.loadtxt(path, delimiter=',')
    return data[:, 0], data[:, 1] * PSIA_TO_MPA


def get_opal_pressure_at_gs(opal_data, gs_name):
    """Extract pressure time series at a gauge station from OPAL results."""
    cell = opal_data['gs_cells'][gs_name]
    return opal_data['t'], opal_data['p'][:, cell] / 1e6


def get_opal_void_at_gs5(opal_data):
    """Extract void fraction time series at GS-5."""
    cell = opal_data['gs_cells']["GS-5"]
    return opal_data['t'], np.clip(opal_data['alpha'][:, cell], 0, 1)


def compute_mape_pressure(t_sim, p_sim_MPa, t_exp, p_exp_MPa):
    """MAPE for pressure. Skip points with p_exp < 0.1 MPa."""
    errors = []
    for i in range(len(t_exp)):
        p_interp = np.interp(t_exp[i], t_sim, p_sim_MPa)
        if p_exp_MPa[i] > 0.1:
            errors.append(abs(p_interp - p_exp_MPa[i]) / p_exp_MPa[i] * 100)
    return np.mean(errors) if errors else np.nan


# ============================================================================
# Part 2: Publication figures
# ============================================================================
def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.2,
        'lines.markersize': 3,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    return plt


# Color palette (colorblind-safe)
OPAL_PROD = '#0072B2'
OPAL_BAL = '#009E73'
OPAL_BASE = '#56B4E9'
RELAP_COLOR = '#D55E00'
EXP_COLOR = '#000000'


def save_fig(fig, pub_dir, name):
    """Save figure in PDF and PNG."""
    for fmt in ['pdf', 'png']:
        fig.savefig(pub_dir / f'{name}.{fmt}')
    print(f"  Saved {name}.pdf/png")


def fig_pressure_all_stations(opal_prod, pub_dir, plt):
    """Figure 1: Pressure all 7 stations (4x2 grid, 7 panels + legend)."""
    print("\n--- Figure 1: Pressure all stations ---")

    fig, axes = plt.subplots(4, 2, figsize=(7.5, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    # Order: GS-7, GS-6, GS-5, GS-4, GS-3, GS-2, GS-1 (closed end to break)
    stations = ["GS-7", "GS-6", "GS-5", "GS-4", "GS-3", "GS-2", "GS-1"]
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)']

    for idx, (gs, label) in enumerate(zip(stations, labels)):
        ax = axes_flat[idx]

        # Experiment
        t_exp, p_exp = load_exp_pressure(gs)
        if t_exp is not None:
            ax.plot(t_exp, p_exp, 'o', color=EXP_COLOR, markersize=3,
                    markerfacecolor='none', label='Experiment', zorder=3)

        # RELAP5-3D
        t_rel, p_rel = load_relap5_pressure(gs)
        if t_rel is not None:
            ax.plot(t_rel, p_rel, '--', color=RELAP_COLOR, label='RELAP5-3D')

        # OPAL V11 production
        t_opal, p_opal = get_opal_pressure_at_gs(opal_prod, gs)
        ax.plot(t_opal, p_opal, '-', color=OPAL_PROD, label='OPAL V11')

        ax.set_xlim(0, 0.6)
        ax.set_ylim(0, 8)
        ax.set_title(f'{label} {gs} (x = {gs_x[gs]:.3f} m)', fontsize=10)
        if idx >= 6:
            ax.set_xlabel('Time [s]')
        if idx % 2 == 0:
            ax.set_ylabel('Pressure [MPa]')

    # 8th panel: shared legend
    axes_flat[7].axis('off')
    handles, labels_text = axes_flat[0].get_legend_handles_labels()
    axes_flat[7].legend(handles, labels_text, loc='center', fontsize=11, frameon=False)

    fig.tight_layout()
    save_fig(fig, pub_dir, 'fig_pressure_all_stations')
    plt.close(fig)


def fig_pressure_early_time(opal_prod, pub_dir, plt):
    """Figure 2: Early-time pressure (2x2, 0-50ms), 4 stations."""
    print("\n--- Figure 2: Pressure early time ---")

    early_stations = ["GS-7", "GS-5", "GS-3", "GS-1"]
    labels = ['(a)', '(b)', '(c)', '(d)']

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
    axes_flat = axes.flatten()

    for idx, (gs, label) in enumerate(zip(early_stations, labels)):
        ax = axes_flat[idx]

        # Experiment
        t_exp, p_exp = load_exp_pressure(gs)
        if t_exp is not None:
            mask = t_exp <= 0.05
            ax.plot(t_exp[mask], p_exp[mask], 'o', color=EXP_COLOR, markersize=3,
                    markerfacecolor='none', label='Experiment', zorder=3)

        # RELAP5-3D
        t_rel, p_rel = load_relap5_pressure(gs)
        if t_rel is not None:
            mask = t_rel <= 0.05
            ax.plot(t_rel[mask], p_rel[mask], '--', color=RELAP_COLOR, label='RELAP5-3D')

        # OPAL V11 production
        t_opal, p_opal = get_opal_pressure_at_gs(opal_prod, gs)
        mask = t_opal <= 0.05
        ax.plot(t_opal[mask], p_opal[mask], '-', color=OPAL_PROD, label='OPAL V11')

        ax.set_xlim(0, 0.05)
        ax.set_ylim(0, 8)
        ax.set_title(f'{label} {gs} (x = {gs_x[gs]:.3f} m)', fontsize=10)
        ax.set_xlabel('Time [s]')
        if idx % 2 == 0:
            ax.set_ylabel('Pressure [MPa]')
        if idx == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, pub_dir, 'fig_pressure_early_time')
    plt.close(fig)


def fig_void_fraction_gs5(opal_prod, opal_base, pub_dir, plt):
    """Figure 3: Void fraction at GS-5 (full time)."""
    print("\n--- Figure 3: Void fraction GS-5 ---")

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Experiment
    void_exp_path = DATA_DIR / "fig14-gs5.csv"
    if void_exp_path.exists():
        raw = np.loadtxt(void_exp_path, delimiter=",")
        t_exp, a_exp = raw[:, 0], np.clip(raw[:, 1], 0, 1)
        ax.plot(t_exp, a_exp, 'o', color=EXP_COLOR, markersize=3,
                markerfacecolor='none', label='Experiment')

    # RELAP5-3D (two models)
    relap_mod_path = RELAP_DIR / "fig14-gs5-relap-modified.csv"
    if relap_mod_path.exists():
        raw = np.loadtxt(relap_mod_path, delimiter=",")
        ax.plot(raw[:, 0], np.clip(raw[:, 1], 0, 1), '--', color=RELAP_COLOR,
                label='RELAP5-3D (modified)')

    relap_hf_path = RELAP_DIR / "fig14-gs5-relap-henry-fauske.csv"
    if relap_hf_path.exists():
        raw = np.loadtxt(relap_hf_path, delimiter=",")
        ax.plot(raw[:, 0], np.clip(raw[:, 1], 0, 1), '-.', color='#009E73',
                label='RELAP5-3D (HF)')

    # OPAL base scalar (shows stalled onset)
    if opal_base is not None:
        t_base, a_base = get_opal_void_at_gs5(opal_base)
        ax.plot(t_base, a_base, ':', color=OPAL_BASE, label='OPAL base (h_mix)')

    # OPAL V11 production
    t_prod, a_prod = get_opal_void_at_gs5(opal_prod)
    ax.plot(t_prod, a_prod, '-', color=OPAL_PROD, linewidth=1.5, label='OPAL V11')

    ax.set_xlim(0, 0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Void fraction [-]')
    ax.legend(loc='lower right', fontsize=8)

    fig.tight_layout()
    save_fig(fig, pub_dir, 'fig_void_fraction_gs5')
    plt.close(fig)


def fig_void_onset_detail(opal_prod, opal_base, pub_dir, plt):
    """Figure 4: Void onset detail (0-100ms zoom)."""
    print("\n--- Figure 4: Void onset detail ---")

    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Experiment
    void_exp_path = DATA_DIR / "fig14-gs5.csv"
    if void_exp_path.exists():
        raw = np.loadtxt(void_exp_path, delimiter=",")
        t_exp, a_exp = raw[:, 0], np.clip(raw[:, 1], 0, 1)
        mask = t_exp <= 0.1
        ax.plot(t_exp[mask], a_exp[mask], 'o', color=EXP_COLOR, markersize=4,
                markerfacecolor='none', label='Experiment')

    # RELAP5-3D (two models)
    relap_mod_path = RELAP_DIR / "fig14-gs5-relap-modified.csv"
    if relap_mod_path.exists():
        raw = np.loadtxt(relap_mod_path, delimiter=",")
        t_r, a_r = raw[:, 0], np.clip(raw[:, 1], 0, 1)
        mask = t_r <= 0.1
        ax.plot(t_r[mask], a_r[mask], '--', color=RELAP_COLOR,
                label='RELAP5-3D (modified)')

    relap_hf_path = RELAP_DIR / "fig14-gs5-relap-henry-fauske.csv"
    if relap_hf_path.exists():
        raw = np.loadtxt(relap_hf_path, delimiter=",")
        t_r, a_r = raw[:, 0], np.clip(raw[:, 1], 0, 1)
        mask = t_r <= 0.1
        ax.plot(t_r[mask], a_r[mask], '-.', color='#009E73',
                label='RELAP5-3D (HF)')

    # OPAL base scalar
    if opal_base is not None:
        t_base, a_base = get_opal_void_at_gs5(opal_base)
        mask = t_base <= 0.1
        ax.plot(t_base[mask], a_base[mask], ':', color=OPAL_BASE,
                label='OPAL base (h_mix)')

    # OPAL V11 production
    t_prod, a_prod = get_opal_void_at_gs5(opal_prod)
    mask = t_prod <= 0.1
    ax.plot(t_prod[mask], a_prod[mask], '-', color=OPAL_PROD, linewidth=1.5,
            label='OPAL V11')

    # Experimental onset marker
    ax.axvline(9.5e-3, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.annotate('Exp. onset\n(9.5 ms)', xy=(9.5e-3, 0.02), xytext=(0.025, 0.15),
                fontsize=8, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', linewidth=0.8))

    ax.set_xlim(0, 0.1)
    ax.set_ylim(-0.02, 0.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Void fraction [-]')
    ax.legend(loc='upper left', fontsize=7)

    fig.tight_layout()
    save_fig(fig, pub_dir, 'fig_void_onset_detail')
    plt.close(fig)


def fig_mape_comparison(opal_prod, pub_dir, plt):
    """Figure 5: MAPE comparison bar chart (OPAL V11 vs RELAP5-3D)."""
    print("\n--- Figure 5: MAPE comparison ---")

    import matplotlib.patches as mpatches

    # Compute OPAL V11 per-station MAPE
    opal_mape = {}
    for gs in station_order:
        t_opal, p_opal = get_opal_pressure_at_gs(opal_prod, gs)
        t_exp, p_exp = load_exp_pressure(gs)
        if t_exp is not None:
            opal_mape[gs] = compute_mape_pressure(t_opal, p_opal, t_exp, p_exp)

    # Compute RELAP5-3D per-station MAPE
    relap_mape = {}
    for gs in station_order:
        t_rel, p_rel = load_relap5_pressure(gs)
        t_exp, p_exp = load_exp_pressure(gs)
        if t_rel is not None and t_exp is not None:
            relap_mape[gs] = compute_mape_pressure(t_rel, p_rel, t_exp, p_exp)

    opal_overall = np.mean(list(opal_mape.values()))
    relap_overall = np.mean(list(relap_mape.values())) if relap_mape else np.nan

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(station_order))
    w = 0.35

    opal_vals = [opal_mape.get(gs, 0) for gs in station_order]
    relap_vals = [relap_mape.get(gs, 0) for gs in station_order]

    ax.bar(x - w / 2, opal_vals, w, color=OPAL_PROD, alpha=0.85, label='OPAL V11')
    ax.bar(x + w / 2, relap_vals, w, color=RELAP_COLOR, alpha=0.85, label='RELAP5-3D')

    ax.axhline(opal_overall, color=OPAL_PROD, linestyle=':', linewidth=1,
               label=f'OPAL avg ({opal_overall:.1f}%)')
    if not np.isnan(relap_overall):
        ax.axhline(relap_overall, color=RELAP_COLOR, linestyle=':', linewidth=1,
                   label=f'RELAP5 avg ({relap_overall:.1f}%)')

    ax.set_xticks(x)
    ax.set_xticklabels(station_order)
    ax.set_ylabel('MAPE [%]')
    ax.set_ylim(0, max(max(opal_vals), max(relap_vals)) * 1.2)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    save_fig(fig, pub_dir, 'fig_mape_comparison')
    plt.close(fig)

    return opal_mape, relap_mape, opal_overall, relap_overall


def fig_feature_impact(pub_dir, plt):
    """Figure 6: Feature impact waterfall chart."""
    print("\n--- Figure 6: Feature impact waterfall ---")

    import matplotlib.patches as mpatches

    feat_names = [f[0] for f in FEATURES]
    feat_mape = [f[1] for f in FEATURES]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(FEATURES))

    colors = []
    for i in range(len(feat_mape)):
        if i == 0:
            colors.append('#888888')
        elif feat_mape[i] < feat_mape[i - 1]:
            colors.append('#009E73')   # improvement = green
        elif feat_mape[i] > feat_mape[i - 1]:
            colors.append('#D55E00')   # regression = vermillion
        else:
            colors.append('#888888')

    ax.bar(x, feat_mape, color=colors, edgecolor='black', linewidth=0.5, width=0.7)

    # Annotate deltas
    for i in range(1, len(feat_mape)):
        delta = feat_mape[i] - feat_mape[i - 1]
        sign = '+' if delta > 0 else ''
        y_pos = max(feat_mape[i], feat_mape[i - 1]) + 1.5
        ax.annotate(f'{sign}{delta:.1f} pp', xy=(i, y_pos), ha='center', fontsize=7.5,
                    color=colors[i], fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, fontsize=8)
    ax.set_ylabel('MAPE [%]')
    ax.set_ylim(0, 95)

    green_patch = mpatches.Patch(color='#009E73', label='Improvement')
    red_patch = mpatches.Patch(color='#D55E00', label='Regression')
    ax.legend(handles=[green_patch, red_patch], fontsize=8)

    fig.tight_layout()
    save_fig(fig, pub_dir, 'fig_feature_impact')
    plt.close(fig)


def fig_mesh_convergence(pub_dir, plt):
    """Figure 7: Mesh convergence (two-panel: MAPE vs N, wall time vs N)."""
    print("\n--- Figure 7: Mesh convergence ---")

    mesh_data = [
        {"N": 12,  "mape": 23.1, "wall_time": 3.6},
        {"N": 24,  "mape": 27.1, "wall_time": 13.8},   # V11 production
        {"N": 48,  "mape": 39.9, "wall_time": 55.6},
        {"N": 96,  "mape": 49.9, "wall_time": 220.9},
    ]
    Ns = [m["N"] for m in mesh_data]
    mapes = [m["mape"] for m in mesh_data]
    wtimes = [m["wall_time"] for m in mesh_data]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(Ns, mapes, 'o-', color=OPAL_PROD, label='MAPE [%]', markersize=6)
    l2, = ax2.plot(Ns, wtimes, 's--', color=RELAP_COLOR, label='Wall time [s]', markersize=6)

    # Mark N=24 as production
    ax1.axvline(24, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax1.annotate('production\nmesh', xy=(24, 27.1), xytext=(40, 22),
                 fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'),
                 color='gray', ha='left')

    ax1.set_xlabel('Number of cells N')
    ax1.set_ylabel('MAPE [%]', color=OPAL_PROD)
    ax2.set_ylabel('Wall time [s]', color=RELAP_COLOR)
    ax1.tick_params(axis='y', labelcolor=OPAL_PROD)
    ax2.tick_params(axis='y', labelcolor=RELAP_COLOR)
    ax1.set_xticks(Ns)
    ax1.set_ylim(0, 60)
    ax2.set_ylim(0, 250)

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='upper left')

    fig.tight_layout()
    save_fig(fig, pub_dir, 'fig_mesh_convergence')
    plt.close(fig)


def fig_pareto_frontier(pub_dir, plt):
    """Figure 8: Pareto frontier (MAPE vs Void MAE)."""
    print("\n--- Figure 8: Pareto frontier ---")

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # V11 sweep points (connected)
    mapes = [pt["mape"] for pt in PARETO_POINTS]
    void_maes = [pt["void_mae"] for pt in PARETO_POINTS]
    ax.plot(mapes, void_maes, 'o-', color=OPAL_PROD, markersize=5,
            label='V11 sweep', zorder=3)

    # Label tau values
    for pt in PARETO_POINTS:
        tau_us = pt["tau"] * 1e6
        offset_x, offset_y = 1.5, 0.012
        if pt["label"] == "production":
            # Star marker for production
            ax.plot(pt["mape"], pt["void_mae"], '*', color=OPAL_PROD,
                    markersize=14, zorder=4)
            offset_x, offset_y = 2.0, -0.025
            ax.annotate(f'production\n($\\tau$={tau_us:.0f}$\\mu$s)',
                        xy=(pt["mape"], pt["void_mae"]),
                        xytext=(pt["mape"] + offset_x, pt["void_mae"] + offset_y),
                        fontsize=7, color=OPAL_PROD,
                        arrowprops=dict(arrowstyle='->', color=OPAL_PROD,
                                        linewidth=0.8))
        elif pt["label"] == "balanced":
            ax.annotate(f'balanced\n($\\tau$={tau_us:.0f}$\\mu$s)',
                        xy=(pt["mape"], pt["void_mae"]),
                        xytext=(pt["mape"] - 8, pt["void_mae"] + 0.02),
                        fontsize=7, color=OPAL_BAL,
                        arrowprops=dict(arrowstyle='->', color=OPAL_BAL,
                                        linewidth=0.8))
        else:
            ax.annotate(f'$\\tau$={tau_us:.0f}$\\mu$s',
                        xy=(pt["mape"], pt["void_mae"]),
                        xytext=(pt["mape"] + offset_x, pt["void_mae"] + offset_y),
                        fontsize=6.5, color='gray')

    # Base scalar point
    bs = BASE_SCALAR_POINT
    ax.plot(bs["mape"], bs["void_mae"], 's', color=OPAL_BASE, markersize=8,
            label='Base scalar (h_mix)', zorder=3)
    ax.annotate(bs["label"], xy=(bs["mape"], bs["void_mae"]),
                xytext=(bs["mape"] - 10, bs["void_mae"] - 0.03),
                fontsize=7, color=OPAL_BASE,
                arrowprops=dict(arrowstyle='->', color=OPAL_BASE, linewidth=0.8))

    # RELAP5-3D point
    rp = RELAP5_POINT
    ax.plot(rp["mape"], rp["void_mae"], 'D', color=RELAP_COLOR, markersize=8,
            label='RELAP5-3D', zorder=3)
    ax.annotate(rp["label"], xy=(rp["mape"], rp["void_mae"]),
                xytext=(rp["mape"] + 3, rp["void_mae"] - 0.02),
                fontsize=7, color=RELAP_COLOR)

    ax.set_xlabel('Pressure MAPE [%]')
    ax.set_ylabel('Void Fraction MAE [-]')
    ax.set_xlim(20, 60)
    ax.set_ylim(0.05, 0.35)
    ax.legend(fontsize=8, loc='upper right')

    # Add ideal corner annotation
    ax.annotate('ideal', xy=(20, 0.05), fontsize=8, color='gray', style='italic')

    fig.tight_layout()
    save_fig(fig, pub_dir, 'fig_pareto_frontier')
    plt.close(fig)


def generate_figures(pub_dir):
    """Generate all 8 publication figures."""
    plt = setup_matplotlib()

    # Load V11 production results
    prod_npz = RESULTS_DIR / V11_RESULT_DIRS["production"] / V11_NPZ_NAMES["production"]
    if not prod_npz.exists():
        print(f"ERROR: V11 production results not found at {prod_npz}")
        print("       Run simulations first (without --figures-only)")
        sys.exit(1)

    opal_prod = load_opal_results(prod_npz)

    # Load base scalar results (hf_ramp) for onset comparison
    base_npz = RESULTS_DIR / "hf_ramp" / "edwards_hf_ramp_N24.npz"
    opal_base = None
    if base_npz.exists():
        opal_base = load_opal_results(base_npz)
    else:
        print("  WARNING: Base scalar results not found, skipping base comparison")

    # Generate each figure
    fig_pressure_all_stations(opal_prod, pub_dir, plt)
    fig_pressure_early_time(opal_prod, pub_dir, plt)
    fig_void_fraction_gs5(opal_prod, opal_base, pub_dir, plt)
    fig_void_onset_detail(opal_prod, opal_base, pub_dir, plt)
    opal_mape, relap_mape, opal_overall, relap_overall = fig_mape_comparison(opal_prod, pub_dir, plt)
    fig_feature_impact(pub_dir, plt)
    fig_mesh_convergence(pub_dir, plt)
    fig_pareto_frontier(pub_dir, plt)

    return opal_mape, relap_mape, opal_overall, relap_overall


# ============================================================================
# Part 3: CSV export
# ============================================================================
def export_csvs(pub_dir):
    """Export CSV data files for reproducibility."""
    print("\n--- Exporting CSV data ---")

    export_dir = pub_dir / "data_export"
    export_dir.mkdir(exist_ok=True)

    # Load V11 production results
    prod_npz = RESULTS_DIR / V11_RESULT_DIRS["production"] / V11_NPZ_NAMES["production"]
    if not prod_npz.exists():
        print("  ERROR: V11 production results not found, skipping CSV export")
        return
    opal_prod = load_opal_results(prod_npz)

    # V11 pressure at each station
    for gs in station_order:
        t_opal, p_opal = get_opal_pressure_at_gs(opal_prod, gs)
        gs_tag = gs.lower().replace('-', '')
        out_csv = export_dir / f"opal_v11_pressure_{gs_tag}.csv"
        np.savetxt(out_csv, np.column_stack([t_opal, p_opal]),
                   header="time_s, pressure_MPa", delimiter=", ", fmt="%.6e")
        print(f"  Saved {out_csv.name}")

    # V11 void fraction at GS-5
    t_opal, a_opal = get_opal_void_at_gs5(opal_prod)
    out_csv = export_dir / "opal_v11_void_fraction_gs5.csv"
    np.savetxt(out_csv, np.column_stack([t_opal, a_opal]),
               header="time_s, void_fraction", delimiter=", ", fmt="%.6e")
    print(f"  Saved {out_csv.name}")

    # Base scalar void fraction at GS-5 (for onset comparison)
    base_npz = RESULTS_DIR / "hf_ramp" / "edwards_hf_ramp_N24.npz"
    if base_npz.exists():
        opal_base = load_opal_results(base_npz)
        t_base, a_base = get_opal_void_at_gs5(opal_base)
        out_csv = export_dir / "opal_base_void_fraction_gs5.csv"
        np.savetxt(out_csv, np.column_stack([t_base, a_base]),
                   header="time_s, void_fraction", delimiter=", ", fmt="%.6e")
        print(f"  Saved {out_csv.name}")

    # MAPE summary JSON
    mape_json = {}
    for config_name in V11_CONFIGS:
        mape_path = (RESULTS_DIR / V11_RESULT_DIRS[config_name]
                     / V11_MAPE_NAMES[config_name])
        if mape_path.exists():
            with open(mape_path) as f:
                mape_json[config_name] = json.load(f)

    if mape_json:
        with open(export_dir / "v11_mape_summary.json", "w") as f:
            json.dump(mape_json, f, indent=2)
        print("  Saved v11_mape_summary.json")


# ============================================================================
# Part 4: Results tables
# ============================================================================
def update_tables(pub_dir, opal_mape=None, relap_mape=None,
                  opal_overall=None, relap_overall=None):
    """Update results_tables.md with V11 production numbers."""
    print("\n--- Updating results_tables.md ---")

    # Load V11 production MAPE data
    prod_mape_path = (RESULTS_DIR / V11_RESULT_DIRS["production"]
                      / V11_MAPE_NAMES["production"])
    prod_data = None
    if prod_mape_path.exists():
        with open(prod_mape_path) as f:
            prod_data = json.load(f)

    # If MAPE values not passed from figures, compute from prod_data
    if opal_mape is None and prod_data is not None:
        opal_mape = prod_data["per_station"]
        opal_overall = prod_data["overall_mape"]

    if opal_mape is None:
        print("  ERROR: No MAPE data available, skipping table update")
        return

    # Compute RELAP5 MAPE if not passed
    if relap_mape is None:
        relap_mape = {}
        for gs in station_order:
            t_rel, p_rel = load_relap5_pressure(gs)
            t_exp, p_exp = load_exp_pressure(gs)
            if t_rel is not None and t_exp is not None:
                relap_mape[gs] = compute_mape_pressure(t_rel, p_rel, t_exp, p_exp)
        relap_overall = np.mean(list(relap_mape.values())) if relap_mape else np.nan

    lines = []
    lines.append("# Results Tables -- Edwards Blowdown Validation (V11 Production)\n")
    lines.append("Generated by `generate_v11_publication.py`.\n")

    # Table 1: Per-station MAPE (V11 production)
    lines.append("## Table 1: Per-Station MAPE -- OPAL V11 (N=24, tau_mix=450us)\n")
    lines.append("| Station | x [m] | MAPE [%] |")
    lines.append("|---|---:|---:|")
    for gs in station_order:
        mape_val = opal_mape.get(gs, np.nan)
        if isinstance(mape_val, (int, float)):
            lines.append(f"| {gs} | {gs_x[gs]:.3f} | {mape_val:.1f} |")
    lines.append(f"| **Overall** | | **{opal_overall:.1f}** |")

    # Table 2: V11 configurations comparison
    lines.append("\n## Table 2: V11 Configuration Comparison\n")
    lines.append("| Config | tau_mix [us] | MAPE [%] | Void MAE | Onset [ms] |")
    lines.append("|---|---:|---:|---:|---:|")
    for config_name in ["production", "balanced", "void_opt"]:
        mape_path = (RESULTS_DIR / V11_RESULT_DIRS[config_name]
                     / V11_MAPE_NAMES[config_name])
        if mape_path.exists():
            with open(mape_path) as f:
                d = json.load(f)
            tau_us = d["tau_mix"] * 1e6
            lines.append(f"| {config_name} | {tau_us:.0f} | {d['overall_mape']:.1f} "
                         f"| {d['void_mae_gs5']:.3f} | {d['onset_ms']:.1f} |")

    # Table 3: Code comparison OPAL V11 vs RELAP5-3D
    lines.append("\n## Table 3: Code Comparison -- Per-Station MAPE [%]\n")
    lines.append("| Station | x [m] | OPAL V11 | RELAP5-3D | Delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for gs in station_order:
        o_val = opal_mape.get(gs, np.nan)
        r_val = relap_mape.get(gs, np.nan)
        if isinstance(o_val, (int, float)) and isinstance(r_val, (int, float)):
            delta = o_val - r_val
            lines.append(f"| {gs} | {gs_x[gs]:.3f} | {o_val:.1f} | {r_val:.1f} | {delta:+.1f} |")
    if not np.isnan(opal_overall) and not np.isnan(relap_overall):
        lines.append(f"| **Overall** | | **{opal_overall:.1f}** | **{relap_overall:.1f}** | "
                     f"**{opal_overall - relap_overall:+.1f}** |")

    # Table 4: Feature impact progression (updated with V11)
    lines.append("\n## Table 4: Feature Impact Progression\n")
    lines.append("| Configuration | MAPE [%] | Delta [pp] |")
    lines.append("|---|---:|---:|")
    for i, (name, mape) in enumerate(FEATURES):
        name_flat = name.replace("\n", " ")
        delta = f"{mape - FEATURES[i - 1][1]:+.1f}" if i > 0 else "---"
        lines.append(f"| {name_flat} | {mape:.1f} | {delta} |")

    # Table 5: Mesh convergence
    lines.append("\n## Table 5: Mesh Convergence Study\n")
    lines.append("| N | MAPE [%] | Wall time [s] | Note |")
    lines.append("|---:|---:|---:|---|")
    mesh_rows = [
        (12,  23.1, 3.6,   "base scalar"),
        (24,  27.1, 13.8,  "V11 production"),
        (48,  39.9, 55.6,  "base scalar"),
        (96,  49.9, 220.9, "base scalar"),
    ]
    for n, mape, wt, note in mesh_rows:
        lines.append(f"| {n} | {mape:.1f} | {wt:.1f} | {note} |")

    # Table 6: Void fraction metrics
    lines.append("\n## Table 6: Void Fraction Metrics at GS-5\n")
    lines.append("| Code/Config | Void MAE [-] | Onset [ms] | Note |")
    lines.append("|---|---:|---:|---|")

    if prod_data:
        lines.append(f"| OPAL V11 (production) | {prod_data.get('void_mae_gs5', 'N/A')} "
                     f"| {prod_data.get('onset_ms', 'N/A')} | tau_mix=450us |")

    # Add RELAP5 void MAE if data available
    void_exp_path = DATA_DIR / "fig14-gs5.csv"
    if void_exp_path.exists():
        raw = np.loadtxt(void_exp_path, delimiter=",")
        t_exp, a_exp = raw[:, 0], np.clip(raw[:, 1], 0, 1)

        relap_mod_path = RELAP_DIR / "fig14-gs5-relap-modified.csv"
        if relap_mod_path.exists():
            raw = np.loadtxt(relap_mod_path, delimiter=",")
            t_r, a_r = raw[:, 0], np.clip(raw[:, 1], 0, 1)
            a_interp = np.interp(t_exp, t_r, a_r)
            mae_r = np.mean(np.abs(a_interp - a_exp))
            lines.append(f"| RELAP5-3D (modified) | {mae_r:.3f} | ~10 | Ransom-Trapp |")

        relap_hf_path = RELAP_DIR / "fig14-gs5-relap-henry-fauske.csv"
        if relap_hf_path.exists():
            raw = np.loadtxt(relap_hf_path, delimiter=",")
            t_r, a_r = raw[:, 0], np.clip(raw[:, 1], 0, 1)
            a_interp = np.interp(t_exp, t_r, a_r)
            mae_r = np.mean(np.abs(a_interp - a_exp))
            lines.append(f"| RELAP5-3D (HF) | {mae_r:.3f} | ~10 | Henry-Fauske |")

    lines.append(f"| OPAL base (h_mix) | 0.146 | ~140 | stalled onset |")
    lines.append(f"| Experiment | --- | 9.5 | reference |")

    # Table 7: Simulation setup
    lines.append("\n## Table 7: Simulation Setup\n")
    lines.append("| Parameter | OPAL V11 | RELAP5-3D |")
    lines.append("|---|---|---|")
    lines.append("| Code type | Modelica -> C -> Python solver | Fortran (monolithic) |")
    lines.append("| Solver | 2x2 block Thomas, A12 moderation | 6-eq two-fluid |")
    lines.append("| Mesh | 24 cells, dx = 0.171 m | 24 cells (modified model) |")
    lines.append("| Initial pressure | 7.0 MPa | 7.0 MPa |")
    lines.append("| Initial temperature | 502 K (subcooled, isothermal) | 502 K (subcooled) |")
    lines.append("| Critical flow model | Henry-Fauske (Modelica) | Ransom-Trapp / HF |")
    lines.append("| Timestep | 50 us (fixed, CFL-limited) | Adaptive |")
    lines.append("| Break opening | 3 ms linear ramp (Modelica) | 1 ms (modified) |")
    lines.append("| tau_mix | 450 us (A12 moderation) | N/A |")
    lines.append("| A11 compressibility | Isentropic (1/c^2) | Isenthalpic |")

    # Table 8: Pareto frontier data
    lines.append("\n## Table 8: Pareto Frontier (tau_mix Sweep)\n")
    lines.append("| tau_mix [us] | MAPE [%] | Void MAE [-] | Config |")
    lines.append("|---:|---:|---:|---|")
    for pt in PARETO_POINTS:
        tau_us = pt["tau"] * 1e6
        cfg = pt["label"] if pt["label"] else ""
        lines.append(f"| {tau_us:.0f} | {pt['mape']:.1f} | {pt['void_mae']:.3f} | {cfg} |")
    lines.append(f"| --- | {BASE_SCALAR_POINT['mape']:.1f} | "
                 f"{BASE_SCALAR_POINT['void_mae']:.3f} | base scalar (h_mix) |")
    lines.append(f"| --- | {RELAP5_POINT['mape']:.1f} | "
                 f"{RELAP5_POINT['void_mae']:.3f} | RELAP5-3D |")

    (pub_dir / "results_tables.md").write_text("\n".join(lines))
    print("  Saved results_tables.md")


# ============================================================================
# Main
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate V11 production results and publication figures")
    parser.add_argument('--figures-only', action='store_true',
                        help='Generate figures from saved results (skip simulations)')
    parser.add_argument('--sims-only', action='store_true',
                        help='Run simulations only (skip figures)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("OPAL V11 Publication Generator")
    print("  Edwards Blowdown Validation -- Paper B")
    print("  ALL physics from Modelica -- solver provides ONLY numerical methods")
    print("=" * 70)

    t_start = time_mod.perf_counter()

    # Create output directories
    PUB_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1: Run simulations
    if not args.figures_only:
        print("\n" + "=" * 70)
        print("PART 1: Running V11 simulations (3 configs)")
        print("=" * 70)
        sim_results = run_all_v11_configs()

        print("\n--- Simulation Summary ---")
        for name, metrics in sim_results.items():
            print(f"  {name}: MAPE={metrics['overall_mape']:.1f}%, "
                  f"VoidMAE={metrics['void_mae_gs5']:.3f}, "
                  f"onset={metrics['onset_ms']:.1f}ms, "
                  f"wall={metrics['wall_time_s']:.1f}s")

    if args.sims_only:
        t_end = time_mod.perf_counter()
        print(f"\n=== SIMULATIONS COMPLETE ({t_end - t_start:.1f}s) ===")
        return

    # Part 2: Generate figures
    print("\n" + "=" * 70)
    print("PART 2: Generating publication figures (8 figures)")
    print("=" * 70)
    opal_mape, relap_mape, opal_overall, relap_overall = generate_figures(PUB_DIR)

    # Part 3: Export CSVs
    print("\n" + "=" * 70)
    print("PART 3: Exporting CSV data")
    print("=" * 70)
    export_csvs(PUB_DIR)

    # Part 4: Update results tables
    print("\n" + "=" * 70)
    print("PART 4: Updating results tables")
    print("=" * 70)
    update_tables(PUB_DIR, opal_mape, relap_mape, opal_overall, relap_overall)

    t_end = time_mod.perf_counter()
    print(f"\n{'='*70}")
    print(f"ALL DONE in {t_end - t_start:.1f}s")
    print(f"Outputs in: {PUB_DIR}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
