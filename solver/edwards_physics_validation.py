#!/usr/bin/env python3
"""
edwards_physics_validation.py — Edwards blowdown with progressive physics fixes.

Tracks the MAPE impact of each physics improvement honestly.
Uses the BridgeSolver with the equation bridge for all evaluation.
"""

import sys
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
from partitioner.bridge_solver import BridgeSolver
from edwards_blowdown_data import edwards_blowdown
import iapws

# ── Paths ──
XML = OPAL_ROOT / "docs/validation/edwards/data/EdwardsTest_IAPWS_CritFlow_backEnd.xml"
SO = OPAL_ROOT / "feasibility/results/opal_bridge_EdwardsTest_IAPWS_CritFlow.so"
INFO = OPAL_ROOT / "feasibility/results/EdwardsTest_IAPWS_CritFlow_info.json"

for p in [XML, SO, INFO]:
    if not p.exists():
        print(f"ERROR: {p.name} not found"); sys.exit(1)

es = load_equation_system(str(XML))
spec = map_pipe1d(es)
info = parse_info_json(INFO)
N = spec.N
dx = spec.dx

# ── Edwards experimental data ──
ic = edwards_blowdown["initial_conditions"]
p_init_MPa = ic["nominal_pressure_MPa"]
p_init = p_init_MPa * 1e6
T_simplified = ic["simplified_isothermal_K"]
T_profile_data = ic["temperature_profile"]  # (?, x_m, T_F, T_K) per point

gs = edwards_blowdown["gauge_stations"]
gs_cells = {name: min(int(g["x_m"] / dx), N - 1) for name, g in gs.items()}
gs_x = {name: g["x_m"] for name, g in gs.items()}

data_dir = OPAL_ROOT / "docs/validation/edwards/data"
exp_files = {
    "GS-1": "fig3-gs1.csv", "GS-2": "fig4-gs2.csv", "GS-3": "fig5-gs3.csv",
    "GS-4": "fig6-gs4.csv", "GS-5": "fig7-gs5.csv", "GS-6": "fig8-gs6.csv",
    "GS-7": "fig9-gs7.csv",
}
PSIA_TO_MPA = 6894.76 / 1e6


def compute_mape(history, t_sim):
    """Compute per-station and overall MAPE."""
    gs_err = {}
    for name, fn in exp_files.items():
        fp = data_dir / fn
        if not fp.exists():
            continue
        d = np.loadtxt(fp, delimiter=",")
        t_e, p_e = d[:, 0], d[:, 1] * PSIA_TO_MPA
        ci = gs_cells[name]
        p_s = np.array([r[1][ci] for r in history]) / 1e6
        errs = [abs((np.interp(t_e[i], t_sim, p_s) - p_e[i]) / p_e[i] * 100)
                for i in range(len(t_e)) if p_e[i] > 0.1]
        if errs:
            gs_err[name] = np.mean(errs)
    return gs_err


def run_edwards(p0, h0, label=""):
    """Run Edwards blowdown and return (history, t_sim, gs_errors)."""
    bridge = OMEquationBridge(SO, info)
    solver = BridgeSolver(bridge, spec, es=es)

    p = p0.copy()
    h = h0.copy()
    mdot = np.zeros(N + 1)

    dt = 5e-5
    t_end = 0.6
    n_steps = int(t_end / dt)

    save_times = np.concatenate([
        np.arange(0, 0.01, 0.0005),
        np.arange(0.01, 0.1, 0.005),
        np.arange(0.1, 0.61, 0.01),
    ])
    history = []
    ns = 0
    t = 0.0

    for step in range(n_steps):
        while ns < len(save_times) and t >= save_times[ns] - 0.5 * dt:
            history.append((t, p.copy()))
            ns += 1
        solver.step(p, h, mdot, dt)
        t += dt

    history.append((t, p.copy()))
    t_sim = np.array([r[0] for r in history])
    gs_err = compute_mape(history, t_sim)
    overall = np.mean(list(gs_err.values())) if gs_err else float("nan")

    if label:
        print(f"\n  {label}: Overall MAPE = {overall:.1f}%")
        for name in exp_files:
            if name in gs_err:
                print(f"    {name} (x={gs_x[name]:.1f}m): {gs_err[name]:.1f}%")

    return history, t_sim, gs_err, overall


# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Edwards Blowdown — Progressive Physics Fixes")
print("=" * 70)

# ── Baseline: isothermal IC ──
ref_isothermal = iapws.IAPWS97(P=p_init_MPa, T=T_simplified)
h_iso = ref_isothermal.h * 1e3
p0_iso = np.full(N, p_init)
h0_iso = np.full(N, h_iso)

print(f"\nModel: HEM + IAPWS + CritFlow, N={N}")
print(f"Baseline IC: isothermal T={T_simplified}K → h={h_iso/1e3:.1f} kJ/kg")

_, _, _, mape_iso = run_edwards(p0_iso, h0_iso, "Baseline (isothermal IC)")

# ── Fix 2: Temperature profile IC ──
# Interpolate experimental T(x) onto the N=24 mesh
x_exp = np.array([pt[1] for pt in T_profile_data])
T_exp = np.array([pt[3] for pt in T_profile_data])

# Cell centre positions (0-indexed)
x_cells = np.array([(i + 0.5) * dx for i in range(N)])

# Interpolate (extrapolate at boundaries if needed)
T_cells = np.interp(x_cells, x_exp, T_exp)

# Compute enthalpy for each cell
h_profile = np.zeros(N)
for i in range(N):
    ref = iapws.IAPWS97(P=p_init_MPa, T=float(T_cells[i]))
    h_profile[i] = ref.h * 1e3

print(f"\nFix 2 IC: temperature profile T={T_cells.min():.1f}-{T_cells.max():.1f}K")
print(f"  h range: {h_profile.min()/1e3:.1f}-{h_profile.max()/1e3:.1f} kJ/kg")

p0_profile = np.full(N, p_init)
_, _, _, mape_profile = run_edwards(p0_profile, h_profile,
                                     "Fix 2 (temperature profile IC)")

# ── Summary ──
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Baseline (isothermal):     {mape_iso:.1f}% MAPE")
print(f"  Fix 2 (temp profile):      {mape_profile:.1f}% MAPE")
delta = mape_iso - mape_profile
print(f"  Improvement:               {delta:+.1f} percentage points")
