#!/usr/bin/env python3
"""
plot_results.py — Edwards blowdown: plot simulation vs experiment for all gauge stations.

Generates:
  results/<solver>/pressure_all_stations.png  — 7-panel pressure comparison
  results/<solver>/pressure_per_station/      — individual station plots
  results/<solver>/void_fraction.png          — void fraction evolution (5-eq only)
  results/<solver>/error_summary.png          — MAPE bar chart
  results/<solver>/report.md                  — quantitative findings
"""

import sys
import pathlib
import numpy as np

OPAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OPAL_ROOT / "solver" / "two_phase"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available, skipping plots")

data_dir = pathlib.Path(__file__).parent / "data"
results_dir = pathlib.Path(__file__).parent / "results"

# ============================================================================
# Experimental data
# ============================================================================

exp_files = {
    "GS-1": "fig3-gs1.csv",
    "GS-2": "fig4-gs2.csv",
    "GS-3": "fig5-gs3.csv",
    "GS-4": "fig6-gs4.csv",
    "GS-5": "fig7-gs5.csv",
    "GS-6": "fig8-gs6.csv",
    "GS-7": "fig9-gs7.csv",
}

gs_x = {
    "GS-1": 3.927, "GS-2": 3.769, "GS-3": 2.935, "GS-4": 2.024,
    "GS-5": 1.469, "GS-6": 0.914, "GS-7": 0.079,
}

PSIA_TO_MPA = 6894.76 / 1e6

exp_data = {}
for gs_name, filename in exp_files.items():
    path = data_dir / filename
    if path.exists():
        raw = np.loadtxt(path, delimiter=",")
        exp_data[gs_name] = {
            "t_s": raw[:, 0],
            "p_MPa": raw[:, 1] * PSIA_TO_MPA,
        }


def load_solver_results(npz_path, solver_type):
    """Load solver results from .npz file."""
    d = np.load(npz_path, allow_pickle=True)
    t = d["t"]
    p = d["p"]  # (n_snap, N)
    N = p.shape[1]
    dx = float(d["dx"])

    # Map gauge stations to cell indices
    gs_cells = {}
    for gs_name, x_m in gs_x.items():
        gs_cells[gs_name] = min(int(x_m / dx), N - 1)

    result = {"t": t, "p": p, "N": N, "dx": dx, "gs_cells": gs_cells}

    if "alpha" in d:
        result["alpha"] = d["alpha"]
    if "mdot" in d:
        result["mdot"] = d["mdot"]
    if "H_i" in d:
        result["H_i"] = float(d["H_i"])

    return result


def compute_errors(sim, gs_name):
    """Compute error metrics for one gauge station."""
    if gs_name not in exp_data:
        return None

    t_exp = exp_data[gs_name]["t_s"]
    p_exp = exp_data[gs_name]["p_MPa"]

    cell = sim["gs_cells"][gs_name]
    t_sim = sim["t"]
    p_sim = sim["p"][:, cell] / 1e6

    errors = []
    early_errors = []  # t < 50 ms
    mid_errors = []    # 50 < t < 200 ms
    late_errors = []   # t > 200 ms

    for i in range(len(t_exp)):
        p_interp = np.interp(t_exp[i], t_sim, p_sim)
        if p_exp[i] > 0.1:
            err = (p_interp - p_exp[i]) / p_exp[i] * 100
            errors.append(abs(err))
            if t_exp[i] < 0.05:
                early_errors.append(abs(err))
            elif t_exp[i] < 0.2:
                mid_errors.append(abs(err))
            else:
                late_errors.append(abs(err))

    return {
        "mape": np.mean(errors) if errors else 0,
        "max_ape": np.max(errors) if errors else 0,
        "early_mape": np.mean(early_errors) if early_errors else 0,
        "mid_mape": np.mean(mid_errors) if mid_errors else 0,
        "late_mape": np.mean(late_errors) if late_errors else 0,
        "n_pts": len(errors),
    }


def plot_all_stations(sim, solver_name, out_dir):
    """7-panel pressure comparison plot."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    station_order = ["GS-7", "GS-6", "GS-5", "GS-4", "GS-3", "GS-2", "GS-1"]

    for idx, gs_name in enumerate(station_order):
        ax = axes[idx]
        cell = sim["gs_cells"][gs_name]
        t_sim = sim["t"] * 1e3  # ms
        p_sim = sim["p"][:, cell] / 1e6

        ax.plot(t_sim, p_sim, "b-", linewidth=1.2, label=solver_name)

        if gs_name in exp_data:
            t_exp = exp_data[gs_name]["t_s"] * 1e3
            p_exp = exp_data[gs_name]["p_MPa"]
            ax.plot(t_exp, p_exp, "ro", markersize=4, label="Experiment")

        ax.set_title(f"{gs_name}  (x = {gs_x[gs_name]:.3f} m)", fontsize=11)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Pressure [MPa]")
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 7.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide the 8th subplot
    axes[7].set_visible(False)

    fig.suptitle(f"Edwards-O'Brien Blowdown — {solver_name}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / "pressure_all_stations.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_early_time(sim, solver_name, out_dir):
    """Early-time detail (0-20 ms) showing wave propagation."""
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    station_order = ["GS-7", "GS-6", "GS-5", "GS-4", "GS-3", "GS-2", "GS-1"]

    for idx, gs_name in enumerate(station_order):
        ax = axes[idx]
        cell = sim["gs_cells"][gs_name]
        t_sim = sim["t"] * 1e3
        p_sim = sim["p"][:, cell] / 1e6

        ax.plot(t_sim, p_sim, "b-", linewidth=1.2, label=solver_name)

        if gs_name in exp_data:
            t_exp = exp_data[gs_name]["t_s"] * 1e3
            p_exp = exp_data[gs_name]["p_MPa"]
            mask = t_exp <= 20
            ax.plot(t_exp[mask], p_exp[mask], "ro", markersize=5, label="Experiment")

        ax.set_title(f"{gs_name}  (x = {gs_x[gs_name]:.3f} m)", fontsize=11)
        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Pressure [MPa]")
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 7.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[7].set_visible(False)

    fig.suptitle(f"Edwards Blowdown — Early Time — {solver_name}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / "pressure_early_time.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_void_fraction(sim, solver_name, out_dir):
    """Void fraction evolution at select stations (5-eq only)."""
    if not HAS_MPL or "alpha" not in sim:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    stations = ["GS-7", "GS-5", "GS-3", "GS-1"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for gs_name, color in zip(stations, colors):
        cell = sim["gs_cells"][gs_name]
        t_sim = sim["t"] * 1e3
        a_sim = sim["alpha"][:, cell]
        ax.plot(t_sim, a_sim, color=color, linewidth=1.2,
                label=f"{gs_name} (x={gs_x[gs_name]:.1f}m)")

    ax.set_xlabel("Time [ms]", fontsize=12)
    ax.set_ylabel("Void Fraction α [-]", fontsize=12)
    ax.set_title(f"Edwards Blowdown — Void Fraction — {solver_name}", fontsize=13)
    ax.set_xlim(0, 600)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    fig.tight_layout()
    out_path = out_dir / "void_fraction.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_error_summary(all_errors, solver_names, out_dir):
    """Bar chart comparing MAPE by station and time regime."""
    if not HAS_MPL:
        return

    stations = list(gs_x.keys())
    x = np.arange(len(stations))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall MAPE by station
    ax = axes[0]
    for i, (name, errors) in enumerate(zip(solver_names, all_errors)):
        mapes = [errors[gs]["mape"] for gs in stations]
        offset = (i - 0.5 * (len(solver_names) - 1)) * width
        ax.bar(x + offset, mapes, width, label=name, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stations, fontsize=9)
    ax.set_ylabel("MAPE [%]")
    ax.set_title("Mean Absolute % Error by Station")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # MAPE by time regime (averaged across stations)
    ax = axes[1]
    regimes = ["Early\n(< 50 ms)", "Mid\n(50-200 ms)", "Late\n(> 200 ms)"]
    for i, (name, errors) in enumerate(zip(solver_names, all_errors)):
        early = np.mean([errors[gs]["early_mape"] for gs in stations if errors[gs]["early_mape"] > 0])
        mid = np.mean([errors[gs]["mid_mape"] for gs in stations if errors[gs]["mid_mape"] > 0])
        late = np.mean([errors[gs]["late_mape"] for gs in stations if errors[gs]["late_mape"] > 0])
        rx = np.arange(3)
        offset = (i - 0.5 * (len(solver_names) - 1)) * width
        ax.bar(rx + offset, [early, mid, late], width, label=name, alpha=0.8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(regimes)
    ax.set_ylabel("MAPE [%]")
    ax.set_title("Error by Time Regime (avg. across stations)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "error_summary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def write_report(solver_name, solver_label, sim, errors, out_dir, extra_info=""):
    """Write a markdown report with quantitative findings."""

    lines = []
    lines.append(f"# Edwards-O'Brien Blowdown — {solver_label}")
    lines.append("")
    lines.append("## Problem")
    lines.append("")
    lines.append("NRC Standard Problem 1. Horizontal pipe (4.096 m, 0.073 m ID) filled")
    lines.append("with subcooled water at 7 MPa / 502 K, ruptured at one end. Duration 0.6 s.")
    lines.append("")
    lines.append(f"## Solver: {solver_label}")
    lines.append("")
    if extra_info:
        lines.append(extra_info)
        lines.append("")
    lines.append(f"- Mesh: {sim['N']} cells, dx = {sim['dx']:.4f} m")
    if "H_i" in sim:
        lines.append(f"- Interfacial HTC: H_i = {sim['H_i']:.0e} W/(m³·K)")
    lines.append("")

    # Error table
    lines.append("## Pressure Comparison — Mean Absolute Percent Error")
    lines.append("")
    lines.append("| Station | x [m] | Overall | Early (<50ms) | Mid (50-200ms) | Late (>200ms) |")
    lines.append("|---------|-------|---------|---------------|----------------|---------------|")
    for gs_name in gs_x:
        e = errors[gs_name]
        x = gs_x[gs_name]
        lines.append(
            f"| {gs_name} | {x:.3f} | {e['mape']:.1f}% | "
            f"{e['early_mape']:.1f}% | {e['mid_mape']:.1f}% | {e['late_mape']:.1f}% |"
        )
    overall = np.mean([e["mape"] for e in errors.values()])
    lines.append(f"| **Overall** | | **{overall:.1f}%** | | | |")
    lines.append("")

    # Qualitative assessment
    lines.append("## Figures")
    lines.append("")
    lines.append("- `pressure_all_stations.png` — Pressure at all 7 gauge stations (0-600 ms)")
    lines.append("- `pressure_early_time.png` — Early time detail (0-20 ms, wave propagation)")
    if "alpha" in sim:
        lines.append("- `void_fraction.png` — Void fraction evolution at 4 selected stations")
    lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved {report_path}")


# ============================================================================
# Main: generate results for each solver
# ============================================================================

solver_configs = []

# HEM v4 (if results exist)
hem_npz = data_dir / "edwards_results.npz"
if hem_npz.exists():
    solver_configs.append({
        "name": "hem_v4",
        "label": "HEM (3-equation, inertial momentum, Ransom-Trapp)",
        "npz": hem_npz,
        "info": ("3-equation homogeneous equilibrium model with inertial momentum\n"
                 "and Ransom-Trapp critical flow blend. Phase 2.5a, iteration v4."),
    })

# 5-eq v1
fiveq_npz = data_dir / "edwards_5eq_results.npz"
if fiveq_npz.exists():
    solver_configs.append({
        "name": "five_eq_v1",
        "label": "5-Equation Drift-Flux (non-equilibrium flashing)",
        "npz": fiveq_npz,
        "info": ("5-equation drift-flux model with separate liquid/vapor energy,\n"
                 "nucleation onset, and interfacial heat transfer.\n"
                 "Phase 3b, iteration v1."),
    })

all_errors = []
solver_names = []

for cfg in solver_configs:
    print(f"\nProcessing {cfg['label']}...")
    out_dir = results_dir / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = load_solver_results(cfg["npz"], cfg["name"])

    # Compute errors
    errors = {}
    for gs_name in gs_x:
        e = compute_errors(sim, gs_name)
        errors[gs_name] = e if e else {"mape": 0, "max_ape": 0,
                                        "early_mape": 0, "mid_mape": 0,
                                        "late_mape": 0, "n_pts": 0}

    all_errors.append(errors)
    solver_names.append(cfg["name"])

    # Plots
    plot_all_stations(sim, cfg["label"], out_dir)
    plot_early_time(sim, cfg["label"], out_dir)
    plot_void_fraction(sim, cfg["label"], out_dir)

    # Report
    write_report(cfg["name"], cfg["label"], sim, errors, out_dir, cfg["info"])

# Comparative error plot (if multiple solvers)
if len(all_errors) > 1:
    plot_error_summary(all_errors, solver_names, results_dir)

print("\nDone.")
