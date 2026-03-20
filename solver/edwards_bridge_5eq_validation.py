#!/usr/bin/env python3
"""
edwards_bridge_5eq_validation.py — Edwards blowdown with 5-equation drift-flux
via the OM equation bridge (True Case 2).

ALL physics from Modelica: metastable T_l/rho_l/rho_v, drift-flux phasic split
(C_0, V_gj), interfacial HT, Martinelli-Nelson friction, critical flow.
Solver provides ONLY: Thomas algorithm + semi-implicit splitting.
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
from partitioner.bridge_5eq_solver import BridgeDriftFluxSolver
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d

# ── Paths ──
BRIDGE_SO = OPAL_ROOT / "feasibility" / "results" / "opal_bridge_EdwardsTest_DriftFlux.so"
INFO_JSON = OPAL_ROOT / "feasibility" / "results" / "EdwardsTest_DriftFlux_info.json"
# Try freshly-extracted XML first, then fall back to validation data dir
EDWARDS_XML = OPAL_ROOT / "feasibility" / "results" / "EdwardsTest_DriftFlux.xml"
if not EDWARDS_XML.exists():
    EDWARDS_XML = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_DriftFlux_backEnd.xml"

for path, name in [(BRIDGE_SO, "Bridge .so"), (INFO_JSON, "Info JSON")]:
    if not path.exists():
        print(f"ERROR: {name} not found at {path}")
        print("Run: python -c 'from partitioner.codegen.translate_model import translate_and_build; translate_and_build(\"EdwardsTest_DriftFlux\")'")
        sys.exit(1)

print("=" * 70)
print("Edwards Blowdown — 5-EQ DRIFT-FLUX via OM Bridge (True Case 2)")
print("  ALL physics from Modelica (cp_f, drift-flux split, metastable, friction)")
print("  Solver provides ONLY numerical methods")
print("=" * 70)

# ── Load model ──
info = parse_info_json(INFO_JSON)
bridge = OMEquationBridge(BRIDGE_SO, info)
N = bridge.N

# Load spec + equation system for geometry AND parameter values
if EDWARDS_XML.exists():
    es = load_equation_system(str(EDWARDS_XML))
    spec = map_pipe1d(es)
else:
    raise FileNotFoundError(f"Edwards XML not found: {EDWARDS_XML}")

# Pass es so the bridge loads ALL parameter values (H_i, C_0, etc.) from Modelica
solver = BridgeDriftFluxSolver(bridge, spec, es=es)

print(f"\nModel: 5-eq drift-flux, N={N}")
print(f"  {info.summary()}")
print(f"  Bridge has mdot_v: {bridge.has('mdot_v')}, mdot_l: {bridge.has('mdot_l')}")

# ── Initial conditions (from EdwardsTest_DriftFlux.mo) ──
p = np.full(N, 7e6)
alpha = np.full(N, 1e-6)
h_l = np.full(N, 986.6e3)
h_v = np.full(N, 2772.6e3)
mdot = np.zeros(N + 1)

dt = 5e-5
t_end = 0.6
n_steps = int(t_end / dt)

from edwards_blowdown_data import edwards_blowdown
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
        a_max = np.max(alpha)
        # Get Gamma from bridge if available
        try:
            Gam = bridge.get('Gamma')
            Gam_max = np.max(Gam) if Gam is not None else 0.0
        except Exception:
            Gam_max = 0.0
        print(f"{step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} "
              f"{a_max:10.4f} {Gam_max:10.2f} {mdot[N]:12.3f}")

history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))

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

print(f"\n{'='*70}")
print(f"Edwards Blowdown — 5-EQ DRIFT-FLUX via OM Bridge")
print(f"{'='*70}")
print(f"\nMAPE by station:")
for gs_name in exp_files:
    if gs_name in gs_errors:
        print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): {gs_errors[gs_name]:.1f}%")
if gs_errors:
    overall = np.mean(list(gs_errors.values()))
    print(f"\n  Overall MAPE: {overall:.1f}%")

print(f"\nPhysical indicators:")
print(f"  alpha_max (mid-time):  {alpha_max_mid:.4f}")
print(f"  alpha_max (final):     {alpha_max_final:.4f}")
print(f"  mdot_out (final):      {history[-1][5][N]:.3f} kg/s")

print(f"\nComparison:")
print(f"  HEM + IAPWS (bridge):          81.0% MAPE")
print(f"  5-eq Case 1 (Python, no flash): 79.8% MAPE")
print(f"  5-eq Case 1 (Python, w/ fix):   30.0% MAPE")
if gs_errors:
    print(f"  5-eq Bridge (True Case 2):     {overall:.1f}% MAPE  <-- THIS RUN")
print(f"\n  Architecture: Modelica .mo → OM translateModel → bridge .so → semi-implicit solver")
