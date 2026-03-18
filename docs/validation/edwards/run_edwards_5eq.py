#!/usr/bin/env python3
"""
run_edwards_5eq.py — Edwards-O'Brien pipe blowdown with 5-equation model.

Same problem as run_edwards.py (HEM), but using the 5-equation drift-flux
model with separate liquid/vapor energy. The key new physics is non-equilibrium
flashing: liquid can be superheated (T_l > T_sat), and interfacial heat
transfer drives evaporation.

This should improve late-time depressurization where HEM gets stuck because
T = T_sat in two-phase (no superheat to drive flashing).

Uses the C++ FiveEqModel via step_5eq binding, with inertial momentum
handled in the pressure solve (same as HEM driver).
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
# Problem setup (identical to HEM driver)
# ============================================================================

geom = edwards_blowdown["geometry"]
ic   = edwards_blowdown["initial_conditions"]

L       = geom["pipe_length_m"]
D       = geom["pipe_inner_diameter_m"]
A_flow  = geom["pipe_flow_area_m2"]
f_D     = 0.02

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

# Properties
fluid = tp.SimpleFluidProperties()  # Will switch to IAPWS for final run
iapws_fluid = tp.IAPWSIF97Properties()

# Use IAPWS for the actual simulation
use_iapws = True
if use_iapws:
    sim_fluid = iapws_fluid
else:
    sim_fluid = fluid

# Initial enthalpy from IAPWS (p, T)
ref = iapws.IAPWS97(P=p_init / 1e6, T=T_init)
h_init = ref.h * 1e3
rho_init = ref.rho

# Saturation at initial pressure
ref_f = iapws.IAPWS97(P=p_init / 1e6, x=0)
ref_g = iapws.IAPWS97(P=p_init / 1e6, x=1)
h_f_init = ref_f.h * 1e3
h_g_init = ref_g.h * 1e3
T_sat_init = ref_f.T

print(f"Initial: p={p_init/1e6:.1f} MPa, T={T_init:.1f} K, T_sat={T_sat_init:.1f} K")
print(f"  h={h_init/1e3:.1f} kJ/kg, h_f={h_f_init/1e3:.1f} kJ/kg, "
      f"h_g={h_g_init/1e3:.1f} kJ/kg, rho={rho_init:.1f} kg/m3")
print(f"  Subcooling: {T_sat_init - T_init:.1f} K")

# 5-equation state arrays
p     = np.full(N, p_init)
alpha = np.full(N, 1e-6)       # tiny initial void (numerical seed)
h_l   = np.full(N, h_init)     # subcooled liquid
h_v   = np.full(N, h_g_init)   # saturated vapor
mdot  = np.zeros(N + 1)        # mdot[0] = 0 (closed end)

# Gauge stations
gauge_stations = edwards_blowdown["gauge_stations"]
gs_cells = {}
for name, gs in gauge_stations.items():
    gs_cells[name] = min(int(gs["x_m"] / dx), N - 1)

print(f"Mesh: {N} cells, dx={dx:.4f} m, dt={dt*1e3:.3f} ms, {n_steps} steps")

# ============================================================================
# Interfacial heat transfer coefficient
# ============================================================================
# H_i controls how fast non-equilibrium flashing occurs.
# Larger H_i → faster approach to thermal equilibrium → closer to HEM.
# For Edwards, we want moderate H_i to capture delayed flashing.
#
# Typical range: 1e4 (slow flashing) to 1e7 (near-equilibrium)
# Start with 1e5 and tune against experiment.
H_i = 1e7  # Large: rapid equilibration for rapid depressurization

# C++ 5-equation model for transport (nucleation, closures, enthalpy bounds)
# Uses SimpleFluid for phasic properties (IAPWS PhasicProperties not yet implemented)
closures = tp.DriftFluxClosures(H_i=H_i, C_0=1.0, alpha_nucleation=1e-3)
model = tp.FiveEqModel(fluid, closures)

# ============================================================================
# Semi-implicit solver with INERTIAL MOMENTUM (Python-level, like HEM driver)
# ============================================================================
# Python handles steps 1-5 (properties, friction, critical flow, pressure,
# inertial momentum). Step 6 (transport: void fraction + phasic enthalpies)
# is delegated to the C++ FiveEqModel via model.update_transport().

def step_edwards_5eq(p, alpha, h_l, h_v, mdot, dt):
    """One timestep: inertial momentum + 5-eq transport."""

    p_old = p.copy()
    mdot_old = mdot.copy()

    # 1. Evaluate mixture properties at old state
    # Clamp pressure to valid IAPWS range
    for i in range(N):
        p[i] = max(p[i], 700.0)  # above triple point (611 Pa)
    props = [sim_fluid.evaluate(p[i],
             (1 - alpha[i]) * h_l[i] + alpha[i] * h_v[i])
             for i in range(N)]

    # Face densities
    rho_face = np.zeros(N + 1)
    rho_face[0] = props[0].rho
    for i in range(1, N):
        rho_face[i] = 0.5 * (props[i-1].rho + props[i].rho)
    rho_face[N] = props[N-1].rho

    # 2. Friction per face
    fric = np.zeros(N + 1)
    for i in range(1, N + 1):
        if rho_face[i] > 0.01:
            fric[i] = (f_D * dx / (2 * D_h)
                       * abs(mdot_old[i]) * mdot_old[i]
                       / (rho_face[i] * A_flow**2))

    # 3. Critical flow (Ransom-Trapp, identical to HEM driver)
    C_d = geom["break_flow_area_fraction"]
    p_N = p_old[N-1]
    h_mix_N = (1 - alpha[N-1]) * h_l[N-1] + alpha[N-1] * h_v[N-1]
    rho_N = props[N-1].rho
    drho_dp_N = props[N-1].drho_dp_h

    try:
        _p_crit = max(p_N / 1e6, 0.01)
        if _p_crit > 22.0:
            _p_crit = 22.0  # below critical point
        ref_f = iapws.IAPWS97(P=_p_crit, x=0)
        ref_g = iapws.IAPWS97(P=_p_crit, x=1)
        rho_f_local = ref_f.rho
        h_f_local = ref_f.h * 1e3
        h_g_local = ref_g.h * 1e3
        h_fg_local = h_g_local - h_f_local
    except Exception:
        rho_f_local = rho_N
        h_f_local = h_mix_N
        h_g_local = h_mix_N + 1e6
        h_fg_local = 1e6

    if h_mix_N <= h_f_local:
        x_local = 0.0
    elif h_mix_N >= h_g_local:
        x_local = 1.0
    else:
        x_local = (h_mix_N - h_f_local) / h_fg_local

    dp_sub = max(p_N - p_atm, 0.0)
    G_sub = (2.0 * rho_f_local * dp_sub) ** 0.5

    if drho_dp_N > 0:
        c_hem = (1.0 / (rho_N * drho_dp_N)) ** 0.5
    else:
        c_hem = 1200.0
    G_hem = rho_N * c_hem

    x_trans = 0.10
    if x_local < x_trans:
        blend = x_local / x_trans
        G_crit = G_sub * (1.0 - blend) + G_hem * blend
    else:
        G_crit = G_hem
    G_crit = max(G_crit, G_hem)
    mdot_critical = C_d * A_flow * G_crit

    mdot_mom_estimate = (mdot_old[N]
                         + (dt * A_flow / dx) * (p_old[N-1] - p_atm)
                         - dt * fric[N])
    outlet_choked = (mdot_mom_estimate > mdot_critical) and (mdot_critical > 0)

    # 4. Implicit pressure solve with inertial momentum
    beta = dt * A_flow / dx

    a = np.zeros(N)
    b = np.zeros(N)
    c_arr = np.zeros(N)
    d = np.zeros(N)

    for i in range(N):
        alpha_i = V_cell * props[i].drho_dp_h / dt

        beta_left  = 0.0 if i == 0 else beta
        if i == N - 1 and outlet_choked:
            beta_right = 0.0
        else:
            beta_right = beta

        a[i] = -beta_left if i > 0 else 0.0
        c_arr[i] = -beta_right if i < N - 1 else 0.0
        b[i] = alpha_i + beta_left + beta_right
        d[i] = alpha_i * p_old[i]

        d[i] += (mdot_old[i] - mdot_old[i+1]) - dt * (fric[i] - fric[i+1])

        if i == N - 1:
            if outlet_choked:
                d[i] += (mdot_old[N] - mdot_critical)
            else:
                d[i] += beta_right * p_atm

    # Thomas solve
    c_p = np.zeros(N)
    d_p = np.zeros(N)
    c_p[0] = c_arr[0] / b[0]
    d_p[0] = d[0] / b[0]
    for i in range(1, N):
        denom = b[i] - a[i] * c_p[i-1]
        c_p[i] = c_arr[i] / denom
        d_p[i] = (d[i] - a[i] * d_p[i-1]) / denom
    p[N-1] = d_p[N-1]
    for i in range(N-2, -1, -1):
        p[i] = d_p[i] - c_p[i] * p[i+1]

    # 5. Update flows (inertial momentum)
    mdot[0] = 0.0
    for i in range(1, N):
        mdot[i] = mdot_old[i] + beta * (p[i-1] - p[i]) - dt * fric[i]

    mdot_momentum = mdot_old[N] + beta * (p[N-1] - p_atm) - dt * fric[N]
    if mdot_momentum > 0:
        mdot[N] = min(mdot_momentum, mdot_critical)
    else:
        mdot[N] = mdot_momentum

    # 6. Transport update — delegated to C++ FiveEqModel.
    # The C++ update_transport handles nucleation onset, interfacial area,
    # drift-flux phasic split, phasic energy with enthalpy bounds, and
    # phase reappearance enthalpy reset. No Python reimplementation.
    # Transport BC for the FiveEqModel (still uses legacy struct internally)
    bc_5eq = tp.BoundaryConditions()
    bc_5eq.p_in = p_init
    bc_5eq.p_out = p_atm
    bc_5eq.h_in = h_init
    bc_5eq.h_l_in = h_init
    bc_5eq.h_v_in = h_g_init

    model.update_transport(
        p, p_old, alpha, h_l, h_v, mdot, bc_5eq,
        N, dx, A_flow, D_h, f_D, dt)

    return p, alpha, h_l, h_v, mdot


# ============================================================================
# Time integration
# ============================================================================

save_times = np.concatenate([
    np.arange(0, 0.01, 0.0005),
    np.arange(0.01, 0.1, 0.005),
    np.arange(0.1, 0.61, 0.01),
])
history = []
next_save_idx = 0

print(f"\nRunning Edwards blowdown (5-equation drift-flux, H_i={H_i:.0e})...")
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

    p, alpha, h_l, h_v, mdot = step_edwards_5eq(p, alpha, h_l, h_v, mdot, dt)
    t += dt

history.append((t, p.copy(), alpha.copy(), h_l.copy(), h_v.copy(), mdot.copy()))
print(f"\nComplete: {n_steps} steps, t_final = {t*1e3:.2f} ms")

# ============================================================================
# Compare to experimental data — all 7 gauge stations
# ============================================================================

data_dir = pathlib.Path(__file__).parent / "data"
t_sim = np.array([rec[0] for rec in history])

# Figure → gauge station file mapping
exp_files = {
    "GS-1": "fig3-gs1.csv",
    "GS-2": "fig4-gs2.csv",
    "GS-3": "fig5-gs3.csv",
    "GS-4": "fig6-gs4.csv",
    "GS-5": "fig7-gs5.csv",
    "GS-6": "fig8-gs6.csv",
    "GS-7": "fig9-gs7.csv",
}

gs_errors = {}  # store per-station error metrics

for gs_name, filename in exp_files.items():
    exp_path = data_dir / filename
    if not exp_path.exists():
        print(f"\n  SKIP {gs_name} ({filename} not found)")
        continue

    exp_data = np.loadtxt(exp_path, delimiter=",")
    t_exp = exp_data[:, 0]
    p_exp_psia = exp_data[:, 1]
    p_exp_MPa = p_exp_psia * 6894.76 / 1e6

    cell_idx = gs_cells[gs_name]
    p_sim = np.array([rec[1][cell_idx] for rec in history]) / 1e6
    a_sim = np.array([rec[2][cell_idx] for rec in history])

    print(f"\n{'='*70}")
    print(f"COMPARISON: {gs_name} Pressure (5-eq vs experiment)")
    print(f"{'='*70}")
    print(f"{'t_ms':>10s} {'p_exp':>10s} {'p_5eq':>10s} {'err_%':>8s} {'alpha':>8s}")

    errors = []
    for i in range(len(t_exp)):
        p_interp = np.interp(t_exp[i], t_sim, p_sim)
        a_interp = np.interp(t_exp[i], t_sim, a_sim)
        if p_exp_MPa[i] > 0.1:
            err_pct = (p_interp - p_exp_MPa[i]) / p_exp_MPa[i] * 100
            errors.append(err_pct)
        else:
            err_pct = float('nan')
        print(f"{t_exp[i]*1e3:10.2f} {p_exp_MPa[i]:10.3f} {p_interp:10.3f} "
              f"{err_pct:8.1f} {a_interp:8.4f}")

    if errors:
        gs_errors[gs_name] = {
            'mean': np.mean(np.abs(errors)),
            'max': np.max(np.abs(errors)),
            'n': len(errors),
        }

# Summary across all stations
print(f"\n{'='*70}")
print(f"SUMMARY: Mean Absolute Percent Error by Gauge Station")
print(f"{'='*70}")
print(f"{'Station':>8s} {'x_m':>6s} {'MAPE':>8s} {'MaxAPE':>8s} {'pts':>5s}")
gs_x = {"GS-1": 3.927, "GS-2": 3.769, "GS-3": 2.935, "GS-4": 2.024,
         "GS-5": 1.469, "GS-6": 0.914, "GS-7": 0.079}
for gs_name in exp_files:
    if gs_name in gs_errors:
        e = gs_errors[gs_name]
        print(f"{gs_name:>8s} {gs_x[gs_name]:6.3f} {e['mean']:7.1f}% {e['max']:7.1f}% {e['n']:5d}")
overall_mape = np.mean([e['mean'] for e in gs_errors.values()])
print(f"{'Overall':>8s} {'':>6s} {overall_mape:7.1f}%")

# Save results
results_path = data_dir / "edwards_5eq_results.npz"
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
