#!/usr/bin/env python3
"""
run_edwards.py — Edwards-O'Brien pipe blowdown simulation.

Pure Python semi-implicit solver with INERTIAL MOMENTUM, using the
Pipe1D equation structure extracted from Modelica. Compares against
digitized experimental data from fig3.csv (GS-1 pressure).

Incremental approach: start simple, add physics features one at a time,
compare to data after each addition.
"""

import sys
import pathlib
import numpy as np

OPAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OPAL_ROOT / "solver" / "two_phase"))
sys.path.insert(0, str(pathlib.Path(__file__).parent / "data"))

import opal_two_phase as tp
from edwards_blowdown_data import edwards_blowdown

# ============================================================================
# Problem setup
# ============================================================================

geom = edwards_blowdown["geometry"]
ic   = edwards_blowdown["initial_conditions"]

L       = geom["pipe_length_m"]           # 4.096 m
D       = geom["pipe_inner_diameter_m"]   # 0.073 m
A_flow  = geom["pipe_flow_area_m2"]       # 4.185e-3 m^2
f_D     = 0.02                            # Darcy friction factor

N       = 24                              # cells
dx      = L / N
D_h     = D
V_cell  = dx * A_flow

p_init  = ic["nominal_pressure_MPa"] * 1e6   # 7.0 MPa
T_init  = ic["simplified_isothermal_K"]       # 502.2 K
p_atm   = 101325.0

dt      = 5e-5                             # 0.05 ms
t_end   = 0.6
n_steps = int(t_end / dt)

# Properties
fluid = tp.IAPWSIF97Properties()

# Initial enthalpy from (p, T)
import iapws
ref = iapws.IAPWS97(P=p_init/1e6, T=T_init)
h_init = ref.h * 1e3
print(f"Initial: p={p_init/1e6:.1f} MPa, T={T_init:.1f} K, "
      f"h={h_init/1e3:.1f} kJ/kg, rho={ref.rho:.1f} kg/m3")

# State arrays
p    = np.full(N, p_init)
h    = np.full(N, h_init)
mdot = np.zeros(N + 1)  # mdot[0] = 0 always (closed end)

# Gauge stations
gauge_stations = edwards_blowdown["gauge_stations"]
gs_cells = {}
for name, gs in gauge_stations.items():
    gs_cells[name] = min(int(gs["x_m"] / dx), N - 1)

print(f"Mesh: {N} cells, dx={dx:.4f} m, dt={dt*1e3:.3f} ms, {n_steps} steps")

# ============================================================================
# Semi-implicit solver with INERTIAL MOMENTUM
# ============================================================================

def step_edwards(p, h, mdot, dt):
    """One semi-implicit timestep with inertial momentum."""

    p_old = p.copy()
    mdot_old = mdot.copy()

    # 1. Evaluate properties at old state
    props = [fluid.evaluate(p[i], h[i]) for i in range(N)]

    # Face densities (arithmetic average)
    rho_face = np.zeros(N + 1)
    rho_face[0] = props[0].rho  # closed end
    for i in range(1, N):
        rho_face[i] = 0.5 * (props[i-1].rho + props[i].rho)
    rho_face[N] = props[N-1].rho  # outlet

    # 2. Friction terms per face (old-time)
    fric = np.zeros(N + 1)
    for i in range(1, N + 1):
        fric[i] = f_D * dx / (2 * D_h) * abs(mdot_old[i]) * mdot_old[i] / (rho_face[i] * A_flow**2)

    # 3. Check if outlet will be choked (using old-time properties)
    # Use a simple critical flow model that blends between subcooled
    # (frozen) and HEM (equilibrium) based on local void fraction.
    # At very low quality, HEM underestimates the sound speed severely.
    rho_N = props[N-1].rho
    drho_dp_N = props[N-1].drho_dp_h
    if drho_dp_N > 0:
        c_sound_hem = (1.0 / (rho_N * drho_dp_N)) ** 0.5
    else:
        c_sound_hem = 1200.0

    # Subcooled sound speed (frozen): use single-phase compressibility
    # For water at typical conditions: c ~ 1000-1200 m/s
    # Approximate: c_frozen = min(1200, c_hem) but at least 50 m/s
    c_sound = max(c_sound_hem, 50.0)  # floor at 50 m/s to avoid HEM collapse

    C_d = geom["break_flow_area_fraction"]
    mdot_critical = C_d * A_flow * rho_N * c_sound

    # Will the momentum equation give more than critical?
    mdot_mom_estimate = mdot_old[N] + (dt * A_flow / dx) * (p_old[N-1] - p_atm) - dt * fric[N]
    outlet_choked = (mdot_mom_estimate > mdot_critical) and (mdot_critical > 0)

    # 3b. Implicit pressure solve with inertial momentum coupling
    beta = dt * A_flow / dx

    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    for i in range(N):
        alpha_i = V_cell * props[i].drho_dp_h / dt

        # Wall BC at face 0: no left connection for cell 0
        beta_left  = 0.0 if i == 0 else beta

        # Choked outlet: face N decouples from downstream pressure
        if i == N - 1 and outlet_choked:
            beta_right = 0.0  # no pressure coupling through choked face
        else:
            beta_right = beta

        a[i] = -beta_left if i > 0 else 0.0
        c[i] = -beta_right if i < N - 1 else 0.0
        b[i] = alpha_i + beta_left + beta_right
        d[i] = alpha_i * p_old[i]

        # RHS: old-time mass flux imbalance + friction correction
        d[i] += (mdot_old[i] - mdot_old[i+1]) - dt * (fric[i] - fric[i+1])

        # Boundary terms
        if i == N - 1:
            if outlet_choked:
                # Choked: outlet flux is fixed at mdot_critical
                # Mass balance for last cell: accumulation = mdot[N-1] - mdot_critical
                # The mdot_old[N] in the RHS should be replaced by mdot_critical
                d[i] += (mdot_old[N] - mdot_critical)  # correction for choked flow
            else:
                d[i] += beta_right * p_atm

    # Thomas algorithm
    c_p = np.zeros(N)
    d_p = np.zeros(N)
    c_p[0] = c[0] / b[0]
    d_p[0] = d[0] / b[0]
    for i in range(1, N):
        denom = b[i] - a[i] * c_p[i-1]
        c_p[i] = c[i] / denom
        d_p[i] = (d[i] - a[i] * d_p[i-1]) / denom
    p[N-1] = d_p[N-1]
    for i in range(N-2, -1, -1):
        p[i] = d_p[i] - c_p[i] * p[i+1]

    # 4. Update flows (time-advanced momentum)
    mdot[0] = 0.0  # wall BC

    for i in range(1, N):
        mdot[i] = mdot_old[i] + beta * (p[i-1] - p[i]) - dt * fric[i]

    # Outlet face — with critical flow limiter
    mdot_momentum = mdot_old[N] + beta * (p[N-1] - p_atm) - dt * fric[N]

    # Critical flow (same model as pressure solve)
    rho_N = props[N-1].rho
    drho_dp_N = props[N-1].drho_dp_h
    if drho_dp_N > 0:
        c_sound_hem = (1.0 / (rho_N * drho_dp_N)) ** 0.5
    else:
        c_sound_hem = 1200.0
    c_sound = max(c_sound_hem, 50.0)
    C_d = geom["break_flow_area_fraction"]
    mdot_critical = C_d * A_flow * rho_N * c_sound

    # Critical flow only limits positive outflow (not inflow)
    if mdot_momentum > 0:
        mdot[N] = min(mdot_momentum, mdot_critical)
    else:
        mdot[N] = mdot_momentum

    # 5. Enthalpy update (donor-cell, forward Euler)
    for i in range(N):
        rho_i = props[i].rho

        # Donor-cell face enthalpies
        if mdot[i] >= 0:
            h_face_in = h[i-1] if i > 0 else h[0]  # wall: use cell 0
        else:
            h_face_in = h[i]

        if mdot[i+1] >= 0:
            h_face_out = h[i]
        else:
            h_face_out = h[i+1] if i < N-1 else h[i]

        flux = mdot[i] * (h_face_in - h[i]) - mdot[i+1] * (h_face_out - h[i])
        p_work = V_cell * (p[i] - p_old[i]) / dt

        h[i] = h[i] + dt / (rho_i * V_cell) * (flux + p_work)

    return p, h, mdot

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

print(f"\nRunning Edwards blowdown (inertial momentum)...")
print(f"{'step':>8s} {'t_ms':>8s} {'p_GS1':>10s} {'p_GS7':>10s} {'mdot_break':>12s}")

t = 0.0
for step in range(n_steps):
    while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
        history.append((t, p.copy(), h.copy(), mdot.copy()))
        next_save_idx += 1

    if step % 2000 == 0 or step == n_steps - 1:
        p_gs1 = p[gs_cells["GS-1"]] / 1e6
        p_gs7 = p[gs_cells["GS-7"]] / 1e6
        print(f"{step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} {mdot[N]:12.3f}")

    p, h, mdot = step_edwards(p, h, mdot, dt)
    t += dt

history.append((t, p.copy(), h.copy(), mdot.copy()))

print(f"\nComplete: {n_steps} steps, t_final = {t*1e3:.2f} ms")

# ============================================================================
# Compare to experimental data
# ============================================================================

data_dir = pathlib.Path(__file__).parent / "data"
fig3_path = data_dir / "fig3.csv"

if fig3_path.exists():
    exp_data = np.loadtxt(fig3_path, delimiter=",")
    t_exp = exp_data[:, 0]
    p_exp_psia = exp_data[:, 1]
    p_exp_MPa = p_exp_psia * 6894.76 / 1e6

    gs1_cell = gs_cells["GS-1"]
    t_sim = np.array([h[0] for h in history])
    p_sim_gs1 = np.array([h[1][gs1_cell] for h in history]) / 1e6

    print(f"\n{'='*60}")
    print(f"COMPARISON: GS-1 Pressure (simulation vs experiment)")
    print(f"{'='*60}")
    print(f"{'t_ms':>10s} {'p_exp':>10s} {'p_sim':>10s} {'err_%':>8s}")

    for i in range(len(t_exp)):
        p_interp = np.interp(t_exp[i], t_sim, p_sim_gs1)
        if p_exp_MPa[i] > 0.1:
            err_pct = (p_interp - p_exp_MPa[i]) / p_exp_MPa[i] * 100
        else:
            err_pct = float('nan')
        print(f"{t_exp[i]*1e3:10.2f} {p_exp_MPa[i]:10.3f} {p_interp:10.3f} {err_pct:8.1f}")

# Save results
results_path = data_dir / "edwards_results.npz"
t_arr = np.array([h[0] for h in history])
p_arr = np.array([h[1] for h in history])
h_arr = np.array([h[2] for h in history])
mdot_arr = np.array([h[3] for h in history])
np.savez(results_path, t=t_arr, p=p_arr, h=h_arr, mdot=mdot_arr,
         dx=dx, N=N, dt=dt, gs_cells=dict(gs_cells))
print(f"\nResults saved to {results_path}")
