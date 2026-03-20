#!/usr/bin/env python3
"""
edwards_iapws_validation.py — Edwards blowdown with IAPWS through the
Modelica extraction pipeline. Compares against experimental data.

Architecture:
  Pipe1D.mo (redeclare Medium = Water) → OpenModelica extraction
  → equation classifier → Python semi-implicit solver
  → property evaluation via C++ IAPWSIF97Properties
  → Ransom-Trapp critical flow (C++ binding)
  → comparison against digitized experimental data

This is the first Modelica-driven validation against real experimental data.
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
from partitioner.equation_classifier import classify_equations
from partitioner.extracted_solver import ExtractedSemiImplicitSolver
from edwards_blowdown_data import edwards_blowdown

# ============================================================================
# Setup
# ============================================================================

EDWARDS_XML = OPAL_ROOT.parent / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_IAPWS_backEnd.xml"

if not EDWARDS_XML.exists():
    print(f"ERROR: {EDWARDS_XML} not found")
    print("Run: cd feasibility && ../external/venv/bin/python extract_edwards_iapws.py")
    sys.exit(1)

print("=" * 70)
print("Edwards Blowdown Validation — IAPWS-IF97 via Modelica Extraction")
print("=" * 70)

# Parse extracted model
es = load_equation_system(str(EDWARDS_XML))
spec = map_pipe1d(es)
cs = classify_equations(es, prefix=spec.prefix)

N = spec.N
print(f"\nExtracted: N={N}, dx={spec.dx:.4f}m, {len(es.equations)} equations")
print(f"  States: {len(es.states)} ({N} p + {N} h + {N-1} mdot)")

# Override outlet to atmospheric (extraction has PressureSource)
p_atm = 101325.0

# Edwards parameters (from validation data)
geom = edwards_blowdown["geometry"]
ic = edwards_blowdown["initial_conditions"]
C_d = geom["break_flow_area_fraction"]  # 0.87

# IAPWS initial conditions
import iapws
p_init = ic["nominal_pressure_MPa"] * 1e6
T_init = ic["simplified_isothermal_K"]
ref = iapws.IAPWS97(P=p_init/1e6, T=T_init)
h_init = ref.h * 1e3
ref_f = iapws.IAPWS97(P=p_init/1e6, x=0)
ref_g = iapws.IAPWS97(P=p_init/1e6, x=1)
h_f_init = ref_f.h * 1e3
h_g_init = ref_g.h * 1e3

print(f"  Initial: p={p_init/1e6:.1f} MPa, T={T_init:.1f} K, "
      f"h={h_init/1e3:.1f} kJ/kg, ρ={ref.rho:.1f} kg/m³")
print(f"  Break: C_d={C_d}, p_back={p_atm} Pa")

# ============================================================================
# Construct solver with IAPWS + critical flow
# ============================================================================

fluid = tp.IAPWSIF97Properties()
critical_flow = tp.RansomTrapp(fluid, x_trans=0.10, c_floor=10.0)

# Use the extraction-driven solver but with IAPWS properties
solver = ExtractedSemiImplicitSolver(cs, fluid, spec)

# Override initial conditions to IAPWS values (extraction has SimpleFluid defaults)
p = np.full(N, p_init)
h = np.full(N, h_init)
mdot = np.zeros(N + 1)

# Time integration parameters
dt = 5e-5
t_end = 0.6
n_steps = int(t_end / dt)

# Gauge stations
gauge_stations = edwards_blowdown["gauge_stations"]
gs_cells = {}
for name, gs in gauge_stations.items():
    gs_cells[name] = min(int(gs["x_m"] / spec.dx), N - 1)

print(f"\nTime: dt={dt*1e3:.3f}ms, t_end={t_end}s, {n_steps} steps")
print(f"Gauge stations: {dict((k, v) for k, v in gs_cells.items())}")

# ============================================================================
# Time integration with critical flow
# ============================================================================

print(f"\n{'step':>8s} {'t_ms':>8s} {'p_GS1':>10s} {'p_GS7':>10s} {'mdot_out':>12s} {'choked':>7s}")

save_times = np.concatenate([
    np.arange(0, 0.01, 0.0005),
    np.arange(0.01, 0.1, 0.005),
    np.arange(0.1, 0.61, 0.01),
])
history = []
next_save_idx = 0
t = 0.0

for step in range(n_steps):
    # Save snapshots
    while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
        history.append((t, p.copy(), h.copy(), mdot.copy()))
        next_save_idx += 1

    # Critical flow check at outlet
    last = N - 1
    fp_last = fluid.evaluate(p[last], h[last])
    pp_last = fluid.evaluate_phasic(max(p[last], fluid.p_min))

    cf_result = critical_flow.evaluate(
        p[last], h[last], fp_last.rho, fp_last.drho_dp_h,
        p_atm, spec.A_flow, C_d, mdot[N])

    # Semi-implicit step (pressure + momentum + energy)
    solver.step(p, h, mdot, dt)

    # Apply critical flow limiter to outlet face
    if cf_result.is_choked and mdot[N] > 0:
        mdot[N] = min(mdot[N], cf_result.mdot_crit)

    t += dt

    if step % 2000 == 0 or step == n_steps - 1:
        p_gs1 = p[gs_cells["GS-1"]] / 1e6
        p_gs7 = p[gs_cells["GS-7"]] / 1e6
        print(f"{step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} "
              f"{mdot[N]:12.3f} {'YES' if cf_result.is_choked else 'no':>7s}")

history.append((t, p.copy(), h.copy(), mdot.copy()))
print(f"\nComplete: {n_steps} steps, t_final = {t*1e3:.2f} ms")

# ============================================================================
# Compare to experimental data
# ============================================================================

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
print(f"MAPE by station (Modelica extraction + IAPWS + critical flow)")
print(f"{'='*60}")
for gs_name in exp_files:
    if gs_name in gs_errors:
        print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): {gs_errors[gs_name]:.1f}%")
if gs_errors:
    overall = np.mean(list(gs_errors.values()))
    print(f"  Overall: {overall:.1f}%")

print(f"\nReference: C++ solver (run_edwards_cpp.py): 149% MAPE")
print(f"Pipeline: Modelica Pipe1D (Water) → OM extraction → Python solver")
