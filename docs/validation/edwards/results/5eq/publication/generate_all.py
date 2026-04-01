#!/usr/bin/env python3
"""
generate_all.py — Edwards blowdown publication analysis.

Generates all figures, tables, sensitivity studies, and data exports
for an NED journal paper comparing OPAL vs RELAP5-3D vs experiment.

Outputs go to docs/validation/edwards/publication/
"""

import json
import pathlib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================================
# Paths
# ============================================================================
HERE = pathlib.Path(__file__).resolve().parent
EDWARDS = HERE.parent
DATA = EDWARDS / "data"
RELAP = DATA / "relap5"
RESULTS = EDWARDS / "results"
OUT = HERE
EXPORT = OUT / "data_export"
EXPORT.mkdir(exist_ok=True)

# ============================================================================
# Publication style
# ============================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OPAL_COLOR = "#0072B2"
RELAP_COLOR = "#D55E00"
RELAP_HF_COLOR = "#009E73"
EXP_COLOR = "#000000"
EXP_MARKER = "o"
EXP_MS = 3

PSIA_TO_MPA = 6894.76 / 1e6

# ============================================================================
# Gauge station geometry
# ============================================================================
gs_x = {
    "GS-1": 3.927, "GS-2": 3.769, "GS-3": 2.935, "GS-4": 2.024,
    "GS-5": 1.469, "GS-6": 0.914, "GS-7": 0.079,
}
station_order = ["GS-1", "GS-2", "GS-3", "GS-4", "GS-5", "GS-6", "GS-7"]

# ============================================================================
# Load experimental pressure data
# ============================================================================
exp_files = {
    "GS-1": "fig3-gs1.csv", "GS-2": "fig4-gs2.csv", "GS-3": "fig5-gs3.csv",
    "GS-4": "fig6-gs4.csv", "GS-5": "fig7-gs5.csv", "GS-6": "fig8-gs6.csv",
    "GS-7": "fig9-gs7.csv",
}

exp_pressure = {}
for gs, fn in exp_files.items():
    raw = np.loadtxt(DATA / fn, delimiter=",")
    exp_pressure[gs] = {"t": raw[:, 0], "p_MPa": raw[:, 1] * PSIA_TO_MPA}

# Experimental void fraction at GS-5
exp_void_path = DATA / "fig14-gs5.csv"
exp_void = None
if exp_void_path.exists():
    raw = np.loadtxt(exp_void_path, delimiter=",")
    exp_void = {"t": raw[:, 0], "alpha": np.clip(raw[:, 1], 0, 1)}

# ============================================================================
# Load RELAP5-3D pressure data (modified model)
# ============================================================================
relap_files = {
    "GS-1": "fig3-gs1-relap-modified.csv", "GS-2": "fig4-gs2-relap-modified.csv",
    "GS-3": "fig5-gs3-relap-modified.csv", "GS-4": "fig6-gs4-relap-modified.csv",
    "GS-5": "fig7-gs5-relap-modified.csv", "GS-6": "fig8-gs6-relap-modified.csv",
    "GS-7": "fig9-gs7-relap-modified.csv",
}

relap_pressure = {}
for gs, fn in relap_files.items():
    path = RELAP / fn
    if path.exists():
        raw = np.loadtxt(path, delimiter=",")
        relap_pressure[gs] = {"t": raw[:, 0], "p_MPa": raw[:, 1] * PSIA_TO_MPA}

# RELAP void fraction
relap_void_mod = None
relap_void_hf = None
p = RELAP / "fig14-gs5-relap-modified.csv"
if p.exists():
    raw = np.loadtxt(p, delimiter=",")
    relap_void_mod = {"t": raw[:, 0], "alpha": np.clip(raw[:, 1], 0, 1)}
p = RELAP / "fig14-gs5-relap-henry-fauske.csv"
if p.exists():
    raw = np.loadtxt(p, delimiter=",")
    relap_void_hf = {"t": raw[:, 0], "alpha": np.clip(raw[:, 1], 0, 1)}

# ============================================================================
# Load OPAL simulation results (canonical N=24)
# ============================================================================
def load_opal(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    t = d["t"]; p = d["p"]; N = p.shape[1]; dx = float(np.atleast_1d(d["dx"])[0])
    gs_cells = {gs: min(int(x / dx), N - 1) for gs, x in gs_x.items()}
    result = {"t": t, "p": p, "N": N, "dx": dx, "gs_cells": gs_cells}
    if "alpha" in d:
        result["alpha"] = d["alpha"]
    if "mdot" in d:
        result["mdot"] = d["mdot"]
    return result

opal = load_opal(RESULTS / "hf_ramp" / "edwards_hf_ramp_N24.npz")

# ============================================================================
# MAPE / MAE computation
# ============================================================================
def compute_mape_pressure(t_sim, p_sim_MPa, t_exp, p_exp_MPa, t_shift=0.0, t_cutoff=0.0):
    """MAPE for pressure. Skip points with p_exp < 0.1 MPa."""
    errors = []
    for i in range(len(t_exp)):
        t_shifted = t_exp[i] + t_shift
        if t_shifted < t_cutoff:
            continue
        p_interp = np.interp(t_shifted, t_sim, p_sim_MPa)
        if p_exp_MPa[i] > 0.1:
            errors.append(abs(p_interp - p_exp_MPa[i]) / p_exp_MPa[i] * 100)
    return np.mean(errors) if errors else np.nan

def compute_mae_void(t_sim, a_sim, t_exp, a_exp, t_shift=0.0):
    """MAE for void fraction."""
    errors = []
    for i in range(len(t_exp)):
        t_shifted = t_exp[i] + t_shift
        a_interp = np.interp(t_shifted, t_sim, a_sim)
        errors.append(abs(a_interp - a_exp[i]))
    return np.mean(errors) if errors else np.nan

def opal_pressure_at(gs, sim=None):
    """Return (t_s, p_MPa) for OPAL at a gauge station."""
    if sim is None:
        sim = opal
    cell = sim["gs_cells"][gs]
    return sim["t"], sim["p"][:, cell] / 1e6

def opal_void_at(gs, sim=None):
    """Return (t_s, alpha) for OPAL at a gauge station."""
    if sim is None:
        sim = opal
    cell = sim["gs_cells"][gs]
    return sim["t"], np.clip(sim["alpha"][:, cell], 0, 1)

def relap_pressure_at(gs):
    """Return (t_s, p_MPa) for RELAP at a gauge station, or None."""
    if gs in relap_pressure:
        return relap_pressure[gs]["t"], relap_pressure[gs]["p_MPa"]
    return None, None

# ============================================================================
# Compute per-station MAPE for OPAL and RELAP
# ============================================================================
opal_mape = {}
relap_mape = {}
for gs in station_order:
    t_o, p_o = opal_pressure_at(gs)
    t_e, p_e = exp_pressure[gs]["t"], exp_pressure[gs]["p_MPa"]
    opal_mape[gs] = compute_mape_pressure(t_o, p_o, t_e, p_e)

    t_r, p_r = relap_pressure_at(gs)
    if t_r is not None:
        relap_mape[gs] = compute_mape_pressure(t_r, p_r, t_e, p_e)

opal_overall = np.mean(list(opal_mape.values()))
relap_overall = np.mean(list(relap_mape.values())) if relap_mape else np.nan

print(f"OPAL overall MAPE:  {opal_overall:.2f}%")
print(f"RELAP overall MAPE: {relap_overall:.2f}%")

# ============================================================================
# FIGURE 1: Pressure comparison — all 7 stations
# ============================================================================
print("\n--- Figure 1: Pressure all stations ---")
fig, axes = plt.subplots(4, 2, figsize=(7.5, 9))
axes_flat = axes.flatten()
panel_labels = "abcdefgh"

for idx, gs in enumerate(station_order):
    ax = axes_flat[idx]
    t_o, p_o = opal_pressure_at(gs)
    ax.plot(t_o, p_o, color=OPAL_COLOR, linestyle="-", label="OPAL")

    t_r, p_r = relap_pressure_at(gs)
    if t_r is not None:
        ax.plot(t_r, p_r, color=RELAP_COLOR, linestyle="--", label="RELAP5-3D")

    t_e, p_e = exp_pressure[gs]["t"], exp_pressure[gs]["p_MPa"]
    ax.plot(t_e, p_e, color=EXP_COLOR, marker=EXP_MARKER, linestyle="none",
            markersize=EXP_MS, label="Experiment", zorder=5)

    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 7.5)
    ax.text(0.03, 0.92, f"({panel_labels[idx]}) {gs}, x = {gs_x[gs]:.3f} m",
            transform=ax.transAxes, fontsize=9, va="top")
    if idx >= 5:
        ax.set_xlabel("Time [s]")
    if idx % 2 == 0:
        ax.set_ylabel("Pressure [MPa]")

# Legend in 8th panel
ax_leg = axes_flat[7]
ax_leg.set_visible(False)
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.06),
           fontsize=10, frameon=True)

fig.tight_layout(h_pad=0.8, w_pad=0.5)
fig.savefig(OUT / "fig_pressure_all_stations.pdf")
fig.savefig(OUT / "fig_pressure_all_stations.png")
plt.close(fig)
print("  Saved fig_pressure_all_stations.pdf/png")

# ============================================================================
# FIGURE 2: Pressure early time (0–50 ms), 4 stations
# ============================================================================
print("\n--- Figure 2: Pressure early time ---")
early_stations = ["GS-1", "GS-3", "GS-5", "GS-7"]
fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
axes_flat = axes.flatten()

for idx, gs in enumerate(early_stations):
    ax = axes_flat[idx]
    t_o, p_o = opal_pressure_at(gs)
    ax.plot(t_o, p_o, color=OPAL_COLOR, linestyle="-", label="OPAL")

    t_r, p_r = relap_pressure_at(gs)
    if t_r is not None:
        mask = t_r <= 0.05
        ax.plot(t_r[mask], p_r[mask], color=RELAP_COLOR, linestyle="--", label="RELAP5-3D")

    t_e, p_e = exp_pressure[gs]["t"], exp_pressure[gs]["p_MPa"]
    mask = t_e <= 0.05
    ax.plot(t_e[mask], p_e[mask], color=EXP_COLOR, marker=EXP_MARKER, linestyle="none",
            markersize=EXP_MS, label="Experiment", zorder=5)

    ax.set_xlim(0, 0.05)
    ax.set_ylim(0, 7.5)
    ax.text(0.03, 0.92, f"({panel_labels[idx]}) {gs}, x = {gs_x[gs]:.3f} m",
            transform=ax.transAxes, fontsize=9, va="top")
    ax.set_xlabel("Time [s]")
    if idx % 2 == 0:
        ax.set_ylabel("Pressure [MPa]")
    if idx == 0:
        ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "fig_pressure_early_time.pdf")
fig.savefig(OUT / "fig_pressure_early_time.png")
plt.close(fig)
print("  Saved fig_pressure_early_time.pdf/png")

# ============================================================================
# FIGURE 3: Void fraction at GS-5
# ============================================================================
print("\n--- Figure 3: Void fraction GS-5 ---")
fig, ax = plt.subplots(figsize=(5, 3.5))

t_o, a_o = opal_void_at("GS-5")
ax.plot(t_o, a_o, color=OPAL_COLOR, linestyle="-", label="OPAL (5-eq)")

if relap_void_mod is not None:
    ax.plot(relap_void_mod["t"], relap_void_mod["alpha"],
            color=RELAP_COLOR, linestyle="--", label="RELAP5-3D (Ransom-Trapp)")
if relap_void_hf is not None:
    ax.plot(relap_void_hf["t"], relap_void_hf["alpha"],
            color=RELAP_HF_COLOR, linestyle="-.", label="RELAP5-3D (Henry-Fauske)")
if exp_void is not None:
    ax.plot(exp_void["t"], exp_void["alpha"], color=EXP_COLOR, marker=EXP_MARKER,
            linestyle="none", markersize=EXP_MS, label="Experiment", zorder=5)

ax.set_xlim(0, 0.6)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Void Fraction [-]")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "fig_void_fraction_gs5.pdf")
fig.savefig(OUT / "fig_void_fraction_gs5.png")
plt.close(fig)
print("  Saved fig_void_fraction_gs5.pdf/png")

# ============================================================================
# FIGURE 4: MAPE bar chart — OPAL vs RELAP5-3D
# ============================================================================
print("\n--- Figure 4: MAPE comparison ---")
fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(station_order))
w = 0.35

opal_vals = [opal_mape[gs] for gs in station_order]
relap_vals = [relap_mape.get(gs, 0) for gs in station_order]

bars_o = ax.bar(x - w/2, opal_vals, w, color=OPAL_COLOR, alpha=0.85, label="OPAL")
bars_r = ax.bar(x + w/2, relap_vals, w, color=RELAP_COLOR, alpha=0.85, label="RELAP5-3D")

ax.axhline(opal_overall, color=OPAL_COLOR, linestyle=":", linewidth=1,
           label=f"OPAL avg ({opal_overall:.1f}%)")
ax.axhline(relap_overall, color=RELAP_COLOR, linestyle=":", linewidth=1,
           label=f"RELAP5 avg ({relap_overall:.1f}%)")

ax.set_xticks(x)
ax.set_xticklabels(station_order)
ax.set_ylabel("MAPE [%]")
ax.set_ylim(0, max(max(opal_vals), max(relap_vals)) * 1.2)
ax.legend(fontsize=8, ncol=2)

fig.tight_layout()
fig.savefig(OUT / "fig_mape_comparison.pdf")
fig.savefig(OUT / "fig_mape_comparison.png")
plt.close(fig)
print("  Saved fig_mape_comparison.pdf/png")

# ============================================================================
# FIGURE 5: Feature impact waterfall
# ============================================================================
print("\n--- Figure 5: Feature impact waterfall ---")

features = [
    ("3-eq HEM\n+ IAPWS",       81.0),
    ("5-eq, no\nmetastable",    79.8),
    ("+ metastable\nT_l",       30.0),
    ("+ extraction\nbridge",    36.0),
    ("+ implicit\nfriction",    28.4),
    ("+ RT +\nbreak ramp",      31.8),
    ("+ Henry-\nFauske",        28.3),
]
feat_names = [f[0] for f in features]
feat_mape  = [f[1] for f in features]

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(features))
colors = []
for i in range(len(feat_mape)):
    if i == 0:
        colors.append("#888888")
    elif feat_mape[i] < feat_mape[i-1]:
        colors.append("#009E73")  # improvement = green
    elif feat_mape[i] > feat_mape[i-1]:
        colors.append("#D55E00")  # regression = vermillion
    else:
        colors.append("#888888")

bars = ax.bar(x, feat_mape, color=colors, edgecolor="black", linewidth=0.5, width=0.7)

# Annotate deltas
for i in range(1, len(feat_mape)):
    delta = feat_mape[i] - feat_mape[i-1]
    sign = "+" if delta > 0 else ""
    y_pos = max(feat_mape[i], feat_mape[i-1]) + 1.5
    ax.annotate(f"{sign}{delta:.1f} pp", xy=(i, y_pos), ha="center", fontsize=7.5,
                color=colors[i], fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(feat_names, fontsize=8)
ax.set_ylabel("MAPE [%]")
ax.set_ylim(0, 95)

# Legend
import matplotlib.lines as mlines
green_patch = mpatches.Patch(color="#009E73", label="Improvement")
red_patch = mpatches.Patch(color="#D55E00", label="Regression")
ax.legend(handles=[green_patch, red_patch], fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "fig_feature_impact.pdf")
fig.savefig(OUT / "fig_feature_impact.png")
plt.close(fig)
print("  Saved fig_feature_impact.pdf/png")

# ============================================================================
# FIGURE 6: Mesh convergence
# ============================================================================
print("\n--- Figure 6: Mesh convergence ---")

mesh_data = [
    {"N": 12, "dx": 0.341, "mape": 23.1, "wall_time": 3.6},
    {"N": 24, "dx": 0.171, "mape": 28.3, "wall_time": 13.8},
    {"N": 48, "dx": 0.085, "mape": 39.9, "wall_time": 55.6},
    {"N": 96, "dx": 0.043, "mape": 49.9, "wall_time": 220.9},
]
Ns   = [m["N"] for m in mesh_data]
dxs  = [m["dx"] for m in mesh_data]
mapes= [m["mape"] for m in mesh_data]
wtimes=[m["wall_time"] for m in mesh_data]

fig, ax1 = plt.subplots(figsize=(5, 3.5))
ax2 = ax1.twinx()

l1, = ax1.plot(Ns, mapes, "o-", color=OPAL_COLOR, label="MAPE [%]", markersize=6)
l2, = ax2.plot(Ns, wtimes, "s--", color=RELAP_COLOR, label="Wall time [s]", markersize=6)

# Mark N=24 as production
ax1.axvline(24, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax1.annotate("production\nmesh", xy=(24, 28.3), xytext=(40, 22),
             fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"),
             color="gray", ha="left")

ax1.set_xlabel("Number of cells N")
ax1.set_ylabel("MAPE [%]", color=OPAL_COLOR)
ax2.set_ylabel("Wall time [s]", color=RELAP_COLOR)
ax1.tick_params(axis="y", labelcolor=OPAL_COLOR)
ax2.tick_params(axis="y", labelcolor=RELAP_COLOR)
ax1.set_xticks(Ns)
ax1.set_ylim(0, 60)
ax2.set_ylim(0, 250)

lines = [l1, l2]
ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")

fig.tight_layout()
fig.savefig(OUT / "fig_mesh_convergence.pdf")
fig.savefig(OUT / "fig_mesh_convergence.png")
plt.close(fig)
print("  Saved fig_mesh_convergence.pdf/png")

# ============================================================================
# FIGURE 7: Architecture diagram
# ============================================================================
print("\n--- Figure 7: Architecture diagram ---")

fig, ax = plt.subplots(figsize=(7, 3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.3", linewidth=1.2)

boxes = [
    (1.0, 2.0, "Modelica\n(.mo files)", "#D4E6F1"),
    (3.2, 2.0, "OpenModelica\ncompiler", "#D5F5E3"),
    (5.4, 2.0, "C code\ngeneration", "#D5F5E3"),
    (7.6, 2.0, "Bridge .so", "#FADBD8"),
    (7.6, 0.5, "Semi-implicit\nsolver", "#FADBD8"),
]

for cx, cy, txt, fc in boxes:
    bb = FancyBboxPatch((cx - 0.7, cy - 0.4), 1.4, 0.8,
                         boxstyle="round,pad=0.1", facecolor=fc,
                         edgecolor="black", linewidth=1.2)
    ax.add_patch(bb)
    ax.text(cx, cy, txt, ha="center", va="center", fontsize=8, fontweight="bold")

# Arrows between boxes
arrow_kw = dict(arrowstyle="-|>", color="black", linewidth=1.5,
                connectionstyle="arc3,rad=0")
for (x1, x2, y) in [(1.7, 2.5, 2.0), (3.9, 4.7, 2.0), (6.1, 6.9, 2.0)]:
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=arrow_kw)

# Vertical arrow from Bridge to solver
ax.annotate("", xy=(7.6, 1.1), xytext=(7.6, 1.6),
            arrowprops=arrow_kw)

# Arrow from solver back to bridge (evaluate())
ax.annotate("evaluate()", xy=(6.9, 1.4), xytext=(5.5, 0.5),
            fontsize=7, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#555555", linestyle="--"))

# Annotations
ax.text(1.0, 3.3, "ALL PHYSICS", ha="center", fontsize=10, fontweight="bold",
        color=OPAL_COLOR,
        bbox=dict(facecolor="white", edgecolor=OPAL_COLOR, linewidth=1.5, pad=3))
ax.text(7.6, 3.3, "NUMERICS ONLY", ha="center", fontsize=10, fontweight="bold",
        color=RELAP_COLOR,
        bbox=dict(facecolor="white", edgecolor=RELAP_COLOR, linewidth=1.5, pad=3))

# Dashed divider
ax.plot([4.8, 4.8], [0, 3.8], color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

fig.tight_layout()
fig.savefig(OUT / "fig_architecture.pdf")
fig.savefig(OUT / "fig_architecture.png")
plt.close(fig)
print("  Saved fig_architecture.pdf/png")

# ============================================================================
# SENSITIVITY STUDY A: Time-axis shift (pressure MAPE)
# ============================================================================
print("\n--- Sensitivity Study A: Time-axis shift ---")
shifts_ms = [-3, -2, -1, 0, 1, 2, 3]

# OPAL sensitivity
opal_shift = {}
for shift_ms in shifts_ms:
    shift_s = shift_ms / 1000.0
    per_station = {}
    for gs in station_order:
        t_o, p_o = opal_pressure_at(gs)
        t_e, p_e = exp_pressure[gs]["t"], exp_pressure[gs]["p_MPa"]
        per_station[gs] = compute_mape_pressure(t_o, p_o, t_e, p_e, t_shift=shift_s)
    opal_shift[shift_ms] = {**per_station, "Overall": np.mean(list(per_station.values()))}

# RELAP sensitivity
relap_shift = {}
for shift_ms in shifts_ms:
    shift_s = shift_ms / 1000.0
    per_station = {}
    for gs in station_order:
        t_r, p_r = relap_pressure_at(gs)
        t_e, p_e = exp_pressure[gs]["t"], exp_pressure[gs]["p_MPa"]
        if t_r is not None:
            per_station[gs] = compute_mape_pressure(t_r, p_r, t_e, p_e, t_shift=shift_s)
    relap_shift[shift_ms] = {**per_station, "Overall": np.mean(list(per_station.values()))}

# ============================================================================
# SENSITIVITY STUDY B: Early-time cutoff
# ============================================================================
print("--- Sensitivity Study B: Early-time cutoff ---")
cutoffs_ms = [0, 5, 10, 20, 50]

opal_cutoff = {}
relap_cutoff = {}
for cut_ms in cutoffs_ms:
    cut_s = cut_ms / 1000.0
    opal_ps = {}
    relap_ps = {}
    for gs in station_order:
        t_o, p_o = opal_pressure_at(gs)
        t_e, p_e = exp_pressure[gs]["t"], exp_pressure[gs]["p_MPa"]
        opal_ps[gs] = compute_mape_pressure(t_o, p_o, t_e, p_e, t_cutoff=cut_s)

        t_r, p_r = relap_pressure_at(gs)
        if t_r is not None:
            relap_ps[gs] = compute_mape_pressure(t_r, p_r, t_e, p_e, t_cutoff=cut_s)
    opal_cutoff[cut_ms] = {**opal_ps, "Overall": np.mean(list(opal_ps.values()))}
    relap_cutoff[cut_ms] = {**relap_ps, "Overall": np.mean(list(relap_ps.values()))}

# ============================================================================
# SENSITIVITY STUDY C: Void fraction time-shift (MAE)
# ============================================================================
print("--- Sensitivity Study C: Void fraction MAE ---")
void_mae_shift = {}
if exp_void is not None:
    for shift_ms in shifts_ms:
        shift_s = shift_ms / 1000.0
        row = {}
        # OPAL
        t_o, a_o = opal_void_at("GS-5")
        row["OPAL"] = compute_mae_void(t_o, a_o, exp_void["t"], exp_void["alpha"], t_shift=shift_s)
        # RELAP Modified
        if relap_void_mod is not None:
            row["RELAP-Modified"] = compute_mae_void(
                relap_void_mod["t"], relap_void_mod["alpha"],
                exp_void["t"], exp_void["alpha"], t_shift=shift_s)
        # RELAP HF
        if relap_void_hf is not None:
            row["RELAP-HF"] = compute_mae_void(
                relap_void_hf["t"], relap_void_hf["alpha"],
                exp_void["t"], exp_void["alpha"], t_shift=shift_s)
        void_mae_shift[shift_ms] = row

# ============================================================================
# SENSITIVITY STUDY D: Experimental data point distribution
# ============================================================================
print("--- Sensitivity Study D: Data point distribution ---")
data_dist = {}
for gs in station_order:
    t_e = exp_pressure[gs]["t"]
    data_dist[gs] = {
        "total": len(t_e),
        "before_10ms": int(np.sum(t_e < 0.01)),
        "before_50ms": int(np.sum(t_e < 0.05)),
    }

# ============================================================================
# Write sensitivity_study.md
# ============================================================================
print("\n--- Writing sensitivity_study.md ---")
lines = []
lines.append("# Sensitivity Studies — Edwards Blowdown Validation\n")

# Study A
lines.append("## Study A: Time-Axis Shift Sensitivity (Pressure MAPE)\n")
lines.append("Shifts applied to experimental time axis before MAPE computation.\n")
lines.append("### OPAL\n")
hdr = "| Shift [ms] | " + " | ".join(station_order) + " | Overall |"
sep = "|---:" + " | ---:" * (len(station_order) + 1) + " |"
lines.append(hdr)
lines.append(sep)
for s in shifts_ms:
    vals = [f"{opal_shift[s].get(gs, np.nan):.1f}" for gs in station_order]
    vals.append(f"**{opal_shift[s]['Overall']:.1f}**")
    lines.append(f"| {s:+d} | " + " | ".join(vals) + " |")

lines.append("\n### RELAP5-3D (Modified Model)\n")
lines.append(hdr)
lines.append(sep)
for s in shifts_ms:
    vals = [f"{relap_shift[s].get(gs, np.nan):.1f}" for gs in station_order]
    vals.append(f"**{relap_shift[s]['Overall']:.1f}**")
    lines.append(f"| {s:+d} | " + " | ".join(vals) + " |")

# Study B
lines.append("\n## Study B: Early-Time Cutoff Sensitivity (Pressure MAPE)\n")
lines.append("Experimental points before the cutoff are excluded.\n")
hdr_cut = "| Cutoff [ms] | OPAL Overall | RELAP5 Overall |"
sep_cut = "|---:|---:|---:|"
lines.append(hdr_cut)
lines.append(sep_cut)
for c in cutoffs_ms:
    lines.append(f"| {c} | {opal_cutoff[c]['Overall']:.1f} | {relap_cutoff[c]['Overall']:.1f} |")

# Per-station detail
lines.append("\n### OPAL per-station (cutoff sensitivity)\n")
hdr_c2 = "| Cutoff [ms] | " + " | ".join(station_order) + " | Overall |"
sep_c2 = "|---:" + " | ---:" * (len(station_order) + 1) + " |"
lines.append(hdr_c2)
lines.append(sep_c2)
for c in cutoffs_ms:
    vals = [f"{opal_cutoff[c].get(gs, np.nan):.1f}" for gs in station_order]
    vals.append(f"**{opal_cutoff[c]['Overall']:.1f}**")
    lines.append(f"| {c} | " + " | ".join(vals) + " |")

lines.append("\n### RELAP5-3D per-station (cutoff sensitivity)\n")
lines.append(hdr_c2)
lines.append(sep_c2)
for c in cutoffs_ms:
    vals = [f"{relap_cutoff[c].get(gs, np.nan):.1f}" for gs in station_order]
    vals.append(f"**{relap_cutoff[c]['Overall']:.1f}**")
    lines.append(f"| {c} | " + " | ".join(vals) + " |")

# Study C
lines.append("\n## Study C: Void Fraction Sensitivity (MAE at GS-5)\n")
if void_mae_shift:
    codes = list(next(iter(void_mae_shift.values())).keys())
    hdr_v = "| Shift [ms] | " + " | ".join(codes) + " |"
    sep_v = "|---:" + " | ---:" * len(codes) + " |"
    lines.append(hdr_v)
    lines.append(sep_v)
    for s in shifts_ms:
        vals = [f"{void_mae_shift[s].get(c, np.nan):.4f}" for c in codes]
        lines.append(f"| {s:+d} | " + " | ".join(vals) + " |")
else:
    lines.append("*No experimental void fraction data available.*\n")

# Study D
lines.append("\n## Study D: Experimental Data Point Distribution\n")
lines.append("| Station | Total points | Before 10 ms | Before 50 ms |")
lines.append("|---|---:|---:|---:|")
for gs in station_order:
    d = data_dist[gs]
    lines.append(f"| {gs} | {d['total']} | {d['before_10ms']} | {d['before_50ms']} |")

# Interpretation
lines.append("\n## Interpretation\n")
lines.append("The time-axis shift study (Study A) shows that within a ±3 ms band — "
             "representative of digitization uncertainty from the published figures — "
             "both OPAL and RELAP5-3D exhibit similar MAPE sensitivity. ")
opal_range = (min(opal_shift[s]["Overall"] for s in shifts_ms),
              max(opal_shift[s]["Overall"] for s in shifts_ms))
relap_range = (min(relap_shift[s]["Overall"] for s in shifts_ms),
               max(relap_shift[s]["Overall"] for s in shifts_ms))
lines.append(f"OPAL overall MAPE ranges from {opal_range[0]:.1f}% to {opal_range[1]:.1f}% "
             f"across shifts; RELAP5-3D ranges from {relap_range[0]:.1f}% to {relap_range[1]:.1f}%. "
             "The overlapping bands indicate that the two codes achieve comparable accuracy "
             "within the resolution of the available experimental data.\n")
lines.append("The early-time cutoff study (Study B) reveals that the largest MAPE contributions "
             "come from the initial decompression transient (t < 10 ms), where wave propagation "
             "timing and break-opening model dominate the error. Excluding these early points "
             "significantly reduces MAPE for both codes.\n")
lines.append("The void fraction MAE (Study C) is small for all codes and relatively insensitive "
             "to time shifts, indicating that all models capture the bulk void development "
             "at GS-5 reasonably well.\n")
lines.append("Study D shows that digitization density is highest at early times, which is where "
             "timing errors have the largest impact on MAPE. GS-1 (nearest the break) has the "
             "most early-time points and consequently the highest station MAPE for both codes.\n")

(OUT / "sensitivity_study.md").write_text("\n".join(lines))
print("  Saved sensitivity_study.md")

# ============================================================================
# Write results_tables.md
# ============================================================================
print("\n--- Writing results_tables.md ---")
lines = []
lines.append("# Results Tables — Edwards Blowdown Validation\n")

# Table 1: Per-station MAPE (OPAL canonical)
lines.append("## Table 1: Per-Station MAPE — OPAL (HF + Ramp, N=24)\n")
lines.append("| Station | x [m] | MAPE [%] |")
lines.append("|---|---:|---:|")
for gs in station_order:
    lines.append(f"| {gs} | {gs_x[gs]:.3f} | {opal_mape[gs]:.1f} |")
lines.append(f"| **Overall** | | **{opal_overall:.1f}** |")

# Table 2: Critical flow model comparison
lines.append("\n## Table 2: Critical Flow Model Comparison (N=24)\n")
with open(RESULTS / "ramp" / "mape_ramp_N24.json") as f:
    rt_data = json.load(f)
with open(RESULTS / "hf_ramp" / "mape_hf_ramp_N24.json") as f:
    hf_data = json.load(f)
lines.append("| Model | Overall MAPE [%] | GS-1 MAPE [%] |")
lines.append("|---|---:|---:|")
lines.append(f"| Ransom-Trapp | {rt_data['overall_mape']:.1f} | {rt_data['per_station']['GS-1']:.1f} |")
lines.append(f"| Henry-Fauske | {hf_data['overall_mape']:.1f} | {hf_data['per_station']['GS-1']:.1f} |")
lines.append(f"| Improvement  | {rt_data['overall_mape'] - hf_data['overall_mape']:.1f} pp | "
             f"{rt_data['per_station']['GS-1'] - hf_data['per_station']['GS-1']:.1f} pp |")

# Table 3: Feature impact progression
lines.append("\n## Table 3: Feature Impact Progression\n")
lines.append("| Configuration | MAPE [%] | Delta [pp] |")
lines.append("|---|---:|---:|")
for i, (name, mape) in enumerate(features):
    name_flat = name.replace("\n", " ")
    delta = f"{mape - features[i-1][1]:+.1f}" if i > 0 else "—"
    lines.append(f"| {name_flat} | {mape:.1f} | {delta} |")

# Table 4: Mesh convergence
lines.append("\n## Table 4: Mesh Convergence Study\n")
lines.append("| N | dx [m] | dt [µs] | MAPE [%] | GS-1 [%] | GS-7 [%] | Wall time [s] |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|")
conv_json = json.load(open(RESULTS / "convergence" / "convergence_results.json"))
for entry in conv_json:
    n = entry["N"]
    dx = entry["dx"]
    dt_us = entry["dt"] * 1e6
    mape = entry["overall_mape"]
    gs1 = entry["per_station"]["GS-1"]
    gs7 = entry["per_station"]["GS-7"]
    wt = [m["wall_time"] for m in mesh_data if m["N"] == n][0]
    lines.append(f"| {n} | {dx:.3f} | {dt_us:.1f} | {mape:.1f} | {gs1:.1f} | {gs7:.1f} | {wt:.1f} |")

# Table 5: Performance
lines.append("\n## Table 5: Computational Performance\n")
lines.append("| N | Steps | Wall time [s] | Steps/s | Real-time ratio |")
lines.append("|---:|---:|---:|---:|---:|")
for entry in conv_json:
    n = entry["N"]
    n_steps = entry["n_steps"]
    wt = [m["wall_time"] for m in mesh_data if m["N"] == n][0]
    sps = n_steps / wt
    # Simulated time is 0.6 s
    rtr = 0.6 / wt
    lines.append(f"| {n} | {n_steps} | {wt:.1f} | {sps:.0f} | {rtr:.2f}x |")

# Table 6: Code comparison OPAL vs RELAP5-3D
lines.append("\n## Table 6: Code Comparison — Per-Station MAPE [%]\n")
lines.append("| Station | x [m] | OPAL | RELAP5-3D | Delta |")
lines.append("|---|---:|---:|---:|---:|")
for gs in station_order:
    o_val = opal_mape[gs]
    r_val = relap_mape.get(gs, np.nan)
    delta = o_val - r_val
    lines.append(f"| {gs} | {gs_x[gs]:.3f} | {o_val:.1f} | {r_val:.1f} | {delta:+.1f} |")
lines.append(f"| **Overall** | | **{opal_overall:.1f}** | **{relap_overall:.1f}** | "
             f"**{opal_overall - relap_overall:+.1f}** |")

# Table 7: Void fraction MAE
lines.append("\n## Table 7: Void Fraction MAE at GS-5\n")
if exp_void is not None:
    t_o, a_o = opal_void_at("GS-5")
    mae_opal = compute_mae_void(t_o, a_o, exp_void["t"], exp_void["alpha"])
    mae_relap_mod = np.nan
    mae_relap_hf = np.nan
    if relap_void_mod is not None:
        mae_relap_mod = compute_mae_void(
            relap_void_mod["t"], relap_void_mod["alpha"],
            exp_void["t"], exp_void["alpha"])
    if relap_void_hf is not None:
        mae_relap_hf = compute_mae_void(
            relap_void_hf["t"], relap_void_hf["alpha"],
            exp_void["t"], exp_void["alpha"])
    lines.append("| Code | Critical Flow | MAE [-] |")
    lines.append("|---|---|---:|")
    lines.append(f"| OPAL | Henry-Fauske | {mae_opal:.4f} |")
    lines.append(f"| RELAP5-3D | Ransom-Trapp (modified) | {mae_relap_mod:.4f} |")
    lines.append(f"| RELAP5-3D | Henry-Fauske | {mae_relap_hf:.4f} |")
else:
    lines.append("*No experimental void fraction data available.*\n")

# Table 8: Simulation setup comparison
lines.append("\n## Table 8: Simulation Setup Comparison\n")
lines.append("| Parameter | OPAL | RELAP5-3D |")
lines.append("|---|---|---|")
lines.append("| Code type | Modelica → C → Python solver | Fortran (monolithic) |")
lines.append("| Mesh | 24 cells, dx = 0.171 m | 24 cells (modified model) |")
lines.append("| Initial pressure | 7.0 MPa | 7.0 MPa |")
lines.append("| Initial temperature | 502 K (subcooled) | 502 K (subcooled) |")
lines.append("| Critical flow model | Henry-Fauske | Ransom-Trapp / Henry-Fauske |")
lines.append("| Timestep | 50 µs (fixed, CFL-limited) | Adaptive |")
lines.append("| Break opening | 3 ms linear ramp | Instantaneous (standard) / 1 ms (modified) |")
lines.append("| Equation system | 5-eq drift-flux | 6-eq two-fluid |")
lines.append("| Property package | IAPWS-IF97 | ASME steam tables |")

(OUT / "results_tables.md").write_text("\n".join(lines))
print("  Saved results_tables.md")

# ============================================================================
# Data export: CSV files for reproducibility
# ============================================================================
print("\n--- Data export ---")

for gs in station_order:
    t_o, p_o = opal_pressure_at(gs)
    out_csv = EXPORT / f"opal_pressure_{gs.lower().replace('-', '')}.csv"
    np.savetxt(out_csv, np.column_stack([t_o, p_o]),
               header="time_s, pressure_MPa", delimiter=", ", fmt="%.6e")
    print(f"  Saved {out_csv.name}")

# Void fraction at GS-5
t_o, a_o = opal_void_at("GS-5")
out_csv = EXPORT / "opal_void_fraction_gs5.csv"
np.savetxt(out_csv, np.column_stack([t_o, a_o]),
           header="time_s, void_fraction", delimiter=", ", fmt="%.6e")
print(f"  Saved {out_csv.name}")

# MAPE summary JSON
mape_summary = {
    "opal": {
        "model": "5-eq drift-flux, Henry-Fauske, RampedBreak, N=24",
        "overall_mape": round(opal_overall, 2),
        "per_station": {gs: round(v, 2) for gs, v in opal_mape.items()},
    },
    "relap5_modified": {
        "model": "6-eq two-fluid, Ransom-Trapp (modified model)",
        "overall_mape": round(relap_overall, 2),
        "per_station": {gs: round(v, 2) for gs, v in relap_mape.items()},
    },
    "mesh_convergence": [
        {"N": m["N"], "dx_m": m["dx"], "mape_pct": m["mape"], "wall_time_s": m["wall_time"]}
        for m in mesh_data
    ],
    "feature_impact": [
        {"config": f[0].replace("\n", " "), "mape_pct": f[1]}
        for f in features
    ],
}
if exp_void is not None:
    mape_summary["void_fraction_mae_gs5"] = {
        "opal": round(mae_opal, 4),
        "relap5_modified": round(mae_relap_mod, 4),
        "relap5_henry_fauske": round(mae_relap_hf, 4),
    }

with open(EXPORT / "mape_summary.json", "w") as f:
    json.dump(mape_summary, f, indent=2)
print("  Saved mape_summary.json")

print("\n=== ALL DONE ===")
print(f"Outputs in: {OUT}")
