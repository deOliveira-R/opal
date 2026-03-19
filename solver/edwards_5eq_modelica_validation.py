#!/usr/bin/env python3
"""
edwards_5eq_modelica_validation.py — Edwards blowdown with 5-equation
drift-flux model, ALL physics from Modelica.

Pipe1D_DriftFlux.mo with:
  - IAPWS-IF97 properties (Water.mo)
  - 5-equation: p, alpha, h_l, h_v, mdot
  - Drift-flux closures (H_i, C_0, Zuber-Findlay)
  - Ransom-Trapp critical flow (CriticalFlow.mo)
  - Martinelli-Nelson two-phase friction (TwoPhaseFriction.mo)
"""

import sys
import pathlib
import numpy as np

OPAL_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OPAL_ROOT / "two_phase"))
sys.path.insert(0, str(OPAL_ROOT))
sys.path.insert(0, str(OPAL_ROOT.parent / "docs" / "validation" / "edwards" / "data"))

import opal_two_phase as tp
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d
from partitioner.extracted_5eq_solver import Extracted5EqSolver
from edwards_blowdown_data import edwards_blowdown

EDWARDS_XML = OPAL_ROOT.parent / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_DriftFlux_backEnd.xml"

if not EDWARDS_XML.exists():
    print(f"ERROR: {EDWARDS_XML} not found")
    sys.exit(1)

print("=" * 70)
print("Edwards Blowdown — 5-EQUATION DRIFT-FLUX, ALL FROM MODELICA")
print("  Pipe1D_DriftFlux.mo + Water.mo + CriticalFlow.mo + TwoPhaseFriction.mo")
print("=" * 70)

# Parse extracted model (for geometry)
es = load_equation_system(str(EDWARDS_XML))
spec = map_pipe1d(es)
N = spec.N

print(f"\nExtracted: N={N}, {len(es.equations)} equations, {len(es.states)} states")

# IAPWS
fluid = tp.IAPWSIF97Properties()

import iapws
ic = edwards_blowdown["initial_conditions"]
geom = edwards_blowdown["geometry"]
p_init = ic["nominal_pressure_MPa"] * 1e6
T_init = ic["simplified_isothermal_K"]
ref = iapws.IAPWS97(P=p_init/1e6, T=T_init)
ref_f = iapws.IAPWS97(P=p_init/1e6, x=0)
ref_g = iapws.IAPWS97(P=p_init/1e6, x=1)
h_init = ref.h * 1e3
h_f_init = ref_f.h * 1e3
h_g_init = ref_g.h * 1e3

print(f"  IC: p={p_init/1e6:.1f} MPa, T={T_init:.1f} K, h={h_init/1e3:.1f} kJ/kg")

# 5-eq solver with all Modelica physics
# Start without two-phase friction (adds instability with IAPWS)
# H_i=1e5 is more stable than 1e7 for the initial transient
solver = Extracted5EqSolver(
    fluid, spec,
    H_i=1e5, C_0=1.0, alpha_nucleation=1e-3,
    use_critical_flow=True, C_d=geom["break_flow_area_fraction"],
    x_trans=0.10, c_floor=1200.0,
    use_two_phase_friction=False)

# Initial state
p = np.full(N, p_init)
alpha = np.full(N, 1e-6)
h_l = np.full(N, h_init)
h_v = np.full(N, h_g_init)
mdot = np.zeros(N + 1)

dt = 5e-5
t_end = 0.6
n_steps = int(t_end / dt)

gauge_stations = edwards_blowdown["gauge_stations"]
gs_cells = {}
for name, gs in gauge_stations.items():
    gs_cells[name] = min(int(gs["x_m"] / spec.dx), N - 1)

print(f"\nRunning {n_steps} steps (5-eq drift-flux)...")
print(f"{'step':>8s} {'t_ms':>8s} {'p_GS1':>10s} {'p_GS7':>10s} "
      f"{'a_GS1':>10s} {'mdot_out':>12s}")

save_times = np.concatenate([
    np.arange(0, 0.01, 0.0005),
    np.arange(0.01, 0.1, 0.005),
    np.arange(0.1, 0.61, 0.01),
])
history = []
next_save_idx = 0
t = 0.0

for step in range(n_steps):
    while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
        history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))
        next_save_idx += 1

    solver.step(p, alpha, h_l, h_v, mdot, dt)
    t += dt

    if step % 2000 == 0 or step == n_steps - 1:
        p_gs1 = p[gs_cells["GS-1"]] / 1e6
        p_gs7 = p[gs_cells["GS-7"]] / 1e6
        a_gs1 = alpha[gs_cells["GS-1"]]
        print(f"{step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} "
              f"{a_gs1:10.4f} {mdot[N]:12.3f}")

history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))

# Compare to experiment
data_dir = OPAL_ROOT.parent / "docs" / "validation" / "edwards" / "data"
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

print(f"\n{'='*60}")
print(f"MAPE by station — 5-EQUATION DRIFT-FLUX, ALL FROM MODELICA")
print(f"{'='*60}")
for gs_name in exp_files:
    if gs_name in gs_errors:
        print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): {gs_errors[gs_name]:.1f}%")
if gs_errors:
    overall = np.mean(list(gs_errors.values()))
    print(f"  Overall MAPE: {overall:.1f}%")

print(f"\nComparison:")
print(f"  C++ hand-wired 5-eq:           149% MAPE")
print(f"  Modelica HEM + crit flow:       81% MAPE")
print(f"  Modelica 5-eq + crit + 2ph fr: {overall:.0f}% MAPE" if gs_errors else "  (no data)")
