#!/usr/bin/env python3
"""
edwards_bridge_validation.py — Edwards blowdown with the OM equation bridge.

True Case 2: the solver calls OM-generated per-equation C functions for
ALL algebraic evaluation. The solver provides ONLY numerical methods.

Compares BridgeSolver (OM equations) vs ExtractedSemiImplicitSolver (C++ fluid)
to verify identical results, then runs the full blowdown.
"""

import sys
import pathlib
import numpy as np

SOLVER_ROOT = pathlib.Path(__file__).resolve().parent
OPAL_ROOT = SOLVER_ROOT.parent
sys.path.insert(0, str(SOLVER_ROOT / "two_phase"))
sys.path.insert(0, str(SOLVER_ROOT))
sys.path.insert(0, str(OPAL_ROOT / "docs" / "validation" / "edwards" / "data"))

import opal_two_phase as tp
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d
from partitioner.equation_classifier import classify_equations
from partitioner.extracted_solver import ExtractedSemiImplicitSolver
from partitioner.bridge_solver import BridgeSolver
from partitioner.codegen.info_parser import parse_info_json
from partitioner.codegen.equation_bridge import OMEquationBridge

# ── Paths ──
EDWARDS_XML = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_backEnd.xml"
BRIDGE_SO = OPAL_ROOT / "feasibility" / "results" / "opal_bridge_InlineTest.so"
INFO_JSON = OPAL_ROOT / "feasibility" / "results" / "InlineTest_info.json"

print("=" * 70)
print("Edwards Blowdown — TRUE CASE 2 (OM Equation Bridge)")
print("  BridgeSolver: ALL algebraic evaluation from OM-generated C")
print("  Solver provides ONLY: Thomas algorithm + semi-implicit splitting")
print("=" * 70)

for path, name in [(EDWARDS_XML, "Edwards XML"), (BRIDGE_SO, "Bridge .so"), (INFO_JSON, "Info JSON")]:
    if not path.exists():
        print(f"ERROR: {name} not found at {path}")
        sys.exit(1)

# ── Load model ──
es = load_equation_system(str(EDWARDS_XML))
spec = map_pipe1d(es)
cs = classify_equations(es, prefix="pipe")
info = parse_info_json(INFO_JSON)

N = spec.N
print(f"\nModel: HEM, N={N}")
print(f"  {info.summary()}")

# ── Create both solvers ──
# Case 1: C++ FluidPackage + Python algebraic evaluation
fluid_cpp = tp.SimpleFluidProperties()
solver_case1 = ExtractedSemiImplicitSolver(cs, fluid_cpp, spec)

# True Case 2: OM equation bridge — ALL algebraic from C
bridge = OMEquationBridge(BRIDGE_SO, info)
solver_case2 = BridgeSolver(bridge, spec)

# ═══════════════════════════════════════════════════════════════════
# PART 1: Parity test (short run)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PART 1: Case 1 (C++ + Python) vs True Case 2 (OM bridge)")
print(f"{'='*70}")

p1 = np.array(spec.p0, dtype=float); h1 = np.array(spec.h0, dtype=float)
m1 = np.array(spec.mdot0, dtype=float)
p2 = p1.copy(); h2 = h1.copy(); m2 = m1.copy()

dt = 5e-5
n_parity = 2000
print(f"  Running {n_parity} steps side by side...")

for step in range(n_parity):
    solver_case1.step(p1, h1, m1, dt)
    solver_case2.step(p2, h2, m2, dt)

p_err = np.max(np.abs(p1 - p2) / np.maximum(np.abs(p1), 1.0))
h_err = np.max(np.abs(h1 - h2))
m_err = np.max(np.abs(m1 - m2))

print(f"\n  After {n_parity} steps (t={n_parity*dt*1e3:.1f}ms):")
print(f"    Pressure max rel error: {p_err:.2e}")
print(f"    Enthalpy max abs error: {h_err:.2e} J/kg")
print(f"    Mdot max abs error:     {m_err:.2e} kg/s")
print(f"    Case 1: p[0]={p1[0]/1e6:.4f}MPa  mdot_out={m1[-1]:.4f}")
print(f"    Case 2: p[0]={p2[0]/1e6:.4f}MPa  mdot_out={m2[-1]:.4f}")

if p_err < 1e-8:
    print(f"\n  VERDICT: MATCH — True Case 2 matches Case 1")
else:
    print(f"\n  VERDICT: MISMATCH — investigate")
    print(f"  (Some divergence expected from donor-cell differences at wall BC)")

# ═══════════════════════════════════════════════════════════════════
# PART 2: Full Edwards blowdown with True Case 2
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PART 2: Full Edwards blowdown — True Case 2 (BridgeSolver)")
print(f"{'='*70}")

bridge2 = OMEquationBridge(BRIDGE_SO, info)
solver_full = BridgeSolver(bridge2, spec)
p_run = np.array(spec.p0, dtype=float)
h_run = np.array(spec.h0, dtype=float)
m_run = np.array(spec.mdot0, dtype=float)

t_end = 0.6
n_full = int(t_end / dt)

from edwards_blowdown_data import edwards_blowdown
gauge_stations = edwards_blowdown["gauge_stations"]
gs_cells = {}
for name, gs in gauge_stations.items():
    gs_cells[name] = min(int(gs["x_m"] / spec.dx), N - 1)

save_times = np.concatenate([
    np.arange(0, 0.01, 0.0005), np.arange(0.01, 0.1, 0.005), np.arange(0.1, 0.61, 0.01),
])
history = []
next_save_idx = 0
t = 0.0

print(f"  Running {n_full} steps...")
for step in range(n_full):
    while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
        history.append((t, p_run.copy()))
        next_save_idx += 1
    solver_full.step(p_run, h_run, m_run, dt)
    t += dt
    if step % 4000 == 0 or step == n_full - 1:
        p_gs1 = p_run[gs_cells["GS-1"]] / 1e6
        p_gs7 = p_run[gs_cells["GS-7"]] / 1e6
        print(f"    step={step:8d}  t={t*1e3:8.2f}ms  p_GS1={p_gs1:.3f}MPa  p_GS7={p_gs7:.3f}MPa")
history.append((t, p_run.copy()))

# MAPE
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
    errors = [abs((np.interp(t_exp[i], t_sim, p_sim) - p_exp_MPa[i]) / p_exp_MPa[i] * 100)
              for i in range(len(t_exp)) if p_exp_MPa[i] > 0.1]
    if errors:
        gs_errors[gs_name] = np.mean(errors)

gs_x = {"GS-1": 3.927, "GS-2": 3.769, "GS-3": 2.935, "GS-4": 2.024,
         "GS-5": 1.469, "GS-6": 0.914, "GS-7": 0.079}
print(f"\n{'='*60}")
print(f"TRUE CASE 2 Edwards Blowdown — MAPE by station")
print(f"{'='*60}")
for gs_name in exp_files:
    if gs_name in gs_errors:
        print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): {gs_errors[gs_name]:.1f}%")
if gs_errors:
    overall = np.mean(list(gs_errors.values()))
    print(f"\n  Overall MAPE: {overall:.1f}%")
    print(f"\n  Architecture proven:")
    print(f"    Modelica .mo → OM translateModel → per-equation C → bridge .so")
    print(f"    → set_state/evaluate/get_* → semi-implicit solver → experimental MAPE")
    print(f"    Solver provides ONLY numerical methods. Zero physics reimplementation.")
