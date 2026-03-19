#!/usr/bin/env python3
"""
run_edwards_cpp.py — Edwards-O'Brien blowdown with all physics in C++.

This is the target architecture: a thin Python wrapper that constructs the
C++ solver with all strategies and calls step_5eq() once per timestep.
All physics (inertial momentum, critical flow, interfacial heat transfer,
nucleation, enthalpy bounds, pressure floor) is in the C++ solver.

Compare: run_edwards_5eq.py (hybrid Python pressure + C++ transport)
"""

import sys
import pathlib
import numpy as np

OPAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OPAL_ROOT / "solver" / "two_phase"))
sys.path.insert(0, str(pathlib.Path(__file__).parent / "data"))

import opal_two_phase as tp
from edwards_blowdown_data import edwards_blowdown
import iapws

# ============================================================================
# Problem setup
# ============================================================================

geom = edwards_blowdown["geometry"]
ic   = edwards_blowdown["initial_conditions"]

L       = geom["pipe_length_m"]
D       = geom["pipe_inner_diameter_m"]
A_flow  = geom["pipe_flow_area_m2"]
f_D     = 0.02
C_d     = geom["break_flow_area_fraction"]

N       = 24
dx      = L / N
D_h     = D
V_cell  = dx * A_flow

p_init  = ic["nominal_pressure_MPa"] * 1e6
T_init  = ic["simplified_isothermal_K"]
p_atm   = 101325.0

dt      = 5e-5
t_end   = 0.6
n_steps = int(t_end / dt)

# Initial conditions from IAPWS
ref   = iapws.IAPWS97(P=p_init / 1e6, T=T_init)
ref_f = iapws.IAPWS97(P=p_init / 1e6, x=0)
ref_g = iapws.IAPWS97(P=p_init / 1e6, x=1)
h_init   = ref.h * 1e3
h_f_init = ref_f.h * 1e3
h_g_init = ref_g.h * 1e3

print(f"Initial: p={p_init/1e6:.1f} MPa, T={T_init:.1f} K, "
      f"h={h_init/1e3:.1f} kJ/kg, rho={ref.rho:.1f} kg/m3")

# ============================================================================
# C++ solver construction — ALL strategies specified
# ============================================================================

H_i = 1e7

fluid    = tp.IAPWSIF97Properties()
closures = tp.DriftFluxClosures(H_i=H_i, C_0=1.0, alpha_nucleation=1e-3)
model    = tp.FiveEqModel(fluid, closures)
recon    = tp.DonorCell()
momentum = tp.InertialMomentum()

# Ransom-Trapp critical flow — uses C++ IAPWS phasic properties internally
critical_flow = tp.RansomTrapp(fluid, x_trans=0.10, c_floor=1200.0)

solver = tp.TwoPhaseSolver(N, dx, A_flow, D_h, f_D, fluid, recon, model,
                           momentum, critical_flow)

# ============================================================================
# Boundary conditions (BoundaryFace strategy objects)
# ============================================================================

bc_in  = tp.WallFace(h_init, h_g_init)
bc_out = tp.BreakFace(p_atm, C_d, h_init, h_g_init)

# ============================================================================
# State arrays
# ============================================================================

p     = np.full(N, p_init)
alpha = np.full(N, 1e-6)
h_l   = np.full(N, h_init)
h_v   = np.full(N, h_g_init)
mdot  = np.zeros(N + 1)

# Gauge stations
gauge_stations = edwards_blowdown["gauge_stations"]
gs_cells = {}
for name, gs in gauge_stations.items():
    gs_cells[name] = min(int(gs["x_m"] / dx), N - 1)

print(f"Mesh: {N} cells, dx={dx:.4f} m, dt={dt*1e3:.3f} ms, {n_steps} steps")
print(f"Solver: {model.name}, {momentum.name} momentum, "
      f"{critical_flow.name} critical flow, H_i={H_i:.0e}")

# ============================================================================
# Time integration — ONE LINE per timestep
# ============================================================================

save_times = np.concatenate([
    np.arange(0, 0.01, 0.0005),
    np.arange(0.01, 0.1, 0.005),
    np.arange(0.1, 0.61, 0.01),
])
history = []
next_save_idx = 0

print(f"\nRunning Edwards blowdown (full C++ solver)...")
print(f"{'step':>8s} {'t_ms':>8s} {'p_GS1':>10s} {'p_GS7':>10s} "
      f"{'alpha_GS1':>10s} {'mdot_break':>12s}")

t = 0.0
for step in range(n_steps):
    while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
        history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))
        next_save_idx += 1

    if step % 2000 == 0 or step == n_steps - 1:
        p_gs1 = p[gs_cells["GS-1"]] / 1e6
        p_gs7 = p[gs_cells["GS-7"]] / 1e6
        a_gs1 = alpha[gs_cells["GS-1"]]
        print(f"{step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} "
              f"{a_gs1:10.4f} {mdot[N]:12.3f}")

    # ── THE ONE LINE — all physics in C++ ──
    solver.step_bf(p, alpha, h_l, h_v, mdot, bc_in, bc_out, t, dt)

    t += dt

history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))
print(f"\nComplete: {n_steps} steps, t_final = {t*1e3:.2f} ms")

# ============================================================================
# Compare to experimental data
# ============================================================================

data_dir = pathlib.Path(__file__).parent / "data"
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

print(f"\n{'='*50}")
print(f"MAPE by station (full C++ solver)")
print(f"{'='*50}")
for gs_name in exp_files:
    if gs_name in gs_errors:
        print(f"  {gs_name} (x={gs_x[gs_name]:.1f}m): {gs_errors[gs_name]:.1f}%")
overall = np.mean(list(gs_errors.values()))
print(f"  Overall: {overall:.1f}%")

# Save
results_path = data_dir / "edwards_cpp_results.npz"
t_arr = np.array([rec[0] for rec in history])
p_arr = np.array([rec[1] for rec in history])
a_arr = np.array([rec[2] for rec in history])
hl_arr = np.array([rec[3] for rec in history])
hv_arr = np.array([rec[4] for rec in history])
mdot_arr = np.array([rec[5] for rec in history])
np.savez(results_path, t=t_arr, p=p_arr, alpha=a_arr,
         h_l=hl_arr, h_v=hv_arr, mdot=mdot_arr,
         dx=dx, N=N, dt=dt, H_i=H_i, gs_cells=dict(gs_cells))
print(f"\nResults saved to {results_path}")
