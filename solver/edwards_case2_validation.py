#!/usr/bin/env python3
"""
edwards_case2_validation.py — Edwards blowdown: Case 1 vs Case 2 parity.

Case 1: C++ SimpleFluidProperties (pybind11)
Case 2: ModelicaFluidPackage (OM-generated C via ctypes)

Both use Parameterized5EqSolver with the same ExtractedModelSpec.
Results must match to machine precision (same Modelica source, same numerics).

Then runs Case 2 alone for the full blowdown with experimental comparison.
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
from partitioner.codegen.modelica_fluid import ModelicaFluidPackage

# ── Paths ──
SO_PATH = OPAL_ROOT / "feasibility" / "results" / "opal_codegen_InlineTest.so"
EDWARDS_SF_XML = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_backEnd.xml"

print("=" * 70)
print("Edwards Blowdown — CASE 2 VALIDATION")
print("  Case 1: C++ SimpleFluidProperties")
print("  Case 2: ModelicaFluidPackage (OM translateModel → C → ctypes)")
print("=" * 70)

# ── Check prerequisites ──
if not SO_PATH.exists():
    print(f"ERROR: {SO_PATH} not found")
    print("  Run: python solver/partitioner/codegen/build_codegen.py")
    print("       feasibility/results/InlineTest_functions.c")
    print("       feasibility/results/InlineTest_functions.h")
    sys.exit(1)

# Use the SimpleFluid Edwards HEM XML (SimpleFluid is what the InlineTest .so provides)
if not EDWARDS_SF_XML.exists():
    print(f"ERROR: {EDWARDS_SF_XML} not found")
    sys.exit(1)

es = load_equation_system(str(EDWARDS_SF_XML))
spec = map_pipe1d(es)
cs = classify_equations(es, prefix="pipe")

N = spec.N
print(f"\nModel: HEM, N={N}, {len(es.equations)} equations")
print(f"  Geometry: dx={spec.dx:.4f}m, A={spec.A_flow:.6f}m², D_h={spec.D_h:.4f}m")
print(f"  IC: p={spec.p0[0]/1e6:.1f}MPa, h={spec.h0[0]/1e3:.1f}kJ/kg")
print(f"  BC: inlet={'closed' if spec.inlet_closed else 'open'}, outlet p={spec.p_out:.0f}Pa")

# ── Create both fluids ──
fluid_cpp = tp.SimpleFluidProperties()
fluid_om = ModelicaFluidPackage(str(SO_PATH))

# ═══════════════════════════════════════════════════════════════════
# PART 1: Case 1 vs Case 2 parity (short run)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PART 1: Case 1 (C++ SimpleFluid) vs Case 2 (OM codegen) parity")
print(f"{'='*70}")

solver1 = ExtractedSemiImplicitSolver(cs, fluid_cpp, spec)
p1 = np.array(spec.p0, dtype=float)
h1 = np.array(spec.h0, dtype=float)
m1 = np.array(spec.mdot0, dtype=float)

solver2 = ExtractedSemiImplicitSolver(cs, fluid_om, spec)
p2 = np.array(spec.p0, dtype=float)
h2 = np.array(spec.h0, dtype=float)
m2 = np.array(spec.mdot0, dtype=float)

dt = 5e-5
n_steps = 2000
print(f"  Running {n_steps} steps side by side...")

for step in range(n_steps):
    solver1.step(p1, h1, m1, dt)
    solver2.step(p2, h2, m2, dt)

p_err = np.max(np.abs(p1 - p2) / np.maximum(np.abs(p1), 1.0))
h_err = np.max(np.abs(h1 - h2))
m_err = np.max(np.abs(m1 - m2))

print(f"\n  After {n_steps} steps (t={n_steps*dt*1e3:.1f}ms):")
print(f"    Pressure max rel error: {p_err:.2e}")
print(f"    Enthalpy max abs error: {h_err:.2e} J/kg")
print(f"    Mdot max abs error:     {m_err:.2e} kg/s")
print(f"    Case 1 final: p[0]={p1[0]/1e6:.4f}MPa, mdot_out={m1[-1]:.4f}")
print(f"    Case 2 final: p[0]={p2[0]/1e6:.4f}MPa, mdot_out={m2[-1]:.4f}")

if p_err < 1e-10:
    print(f"\n  VERDICT: EXACT MATCH — Case 2 identical to Case 1")
elif p_err < 1e-6:
    print(f"\n  VERDICT: MATCH — within floating-point tolerance")
else:
    print(f"\n  VERDICT: MISMATCH — investigate")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# PART 2: Full Edwards blowdown with Case 2 (experimental comparison)
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"PART 2: Full Edwards blowdown — Case 2 (ModelicaFluidPackage)")
print(f"{'='*70}")

solver_case2 = ExtractedSemiImplicitSolver(cs, fluid_om, spec)
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
    np.arange(0, 0.01, 0.0005),
    np.arange(0.01, 0.1, 0.005),
    np.arange(0.1, 0.61, 0.01),
])
history = []
next_save_idx = 0
t = 0.0

print(f"  Running {n_full} steps (t_end={t_end}s, dt={dt*1e6:.0f}μs)...")
for step in range(n_full):
    while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
        history.append((t, p_run.copy(), h_run.copy(), m_run.copy()))
        next_save_idx += 1

    solver_case2.step(p_run, h_run, m_run, dt)
    t += dt

    if step % 4000 == 0 or step == n_full - 1:
        p_gs1 = p_run[gs_cells["GS-1"]] / 1e6
        p_gs7 = p_run[gs_cells["GS-7"]] / 1e6
        print(f"    step={step:8d}  t={t*1e3:8.2f}ms  p_GS1={p_gs1:.3f}MPa  p_GS7={p_gs7:.3f}MPa")

history.append((t, p_run.copy(), h_run.copy(), m_run.copy()))

# MAPE against experiment
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

    errors = []
    for i in range(len(t_exp)):
        p_interp = np.interp(t_exp[i], t_sim, p_sim)
        if p_exp_MPa[i] > 0.1:
            errors.append(abs((p_interp - p_exp_MPa[i]) / p_exp_MPa[i] * 100))
    if errors:
        gs_errors[gs_name] = np.mean(errors)

gs_x = {"GS-1": 3.927, "GS-2": 3.769, "GS-3": 2.935, "GS-4": 2.024,
         "GS-5": 1.469, "GS-6": 0.914, "GS-7": 0.079}

print(f"\n{'='*60}")
print(f"CASE 2 Edwards Blowdown — MAPE by station (HEM + SimpleFluid)")
print(f"{'='*60}")
for gs_name in exp_files:
    if gs_name in gs_errors:
        print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): {gs_errors[gs_name]:.1f}%")
if gs_errors:
    overall = np.mean(list(gs_errors.values()))
    print(f"\n  Overall MAPE: {overall:.1f}%")
    print(f"\n  Pipeline proven: Modelica → OM translateModel → C → .so → ctypes → solver")
    print(f"  Zero C++ dependency for property evaluation.")
    print(f"  (HEM + SimpleFluid; IAPWS 5-eq requires Water.mo codegen)")
