"""
run_edwards.py — Edwards-O'Brien pipe blowdown simulation.

Pure Python semi-implicit solver with wall BC at the closed end,
using the C++ IAPWS-IF97 property package for steam tables.
Compares against digitized experimental data from fig3.csv (GS-1 pressure).

This is a VALIDATION driver — it does not modify the verified solver.
"""

import sys
import os
import pathlib
import numpy as np

# Add the two-phase solver module to path
OPAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(OPAL_ROOT / "solver" / "two_phase"))
import opal_two_phase as tp

# Load problem data
from edwards_blowdown_data import edwards_blowdown

# ============================================================================
# Problem setup from the Edwards data
# ============================================================================

geom = edwards_blowdown["geometry"]
ic   = edwards_blowdown["initial_conditions"]

L       = geom["pipe_length_m"]           # 4.096 m
D       = geom["pipe_inner_diameter_m"]   # 0.073 m
A_flow  = geom["pipe_flow_area_m2"]       # 4.185e-3 m^2
f_D     = 0.02                            # Darcy friction factor (smooth pipe)

N       = 24                              # cells (standard nodalization)
dx      = L / N
D_h     = D                               # hydraulic diameter = pipe diameter
V_cell  = dx * A_flow

p_init  = ic["nominal_pressure_MPa"] * 1e6   # 7.0 MPa
T_init  = ic["simplified_isothermal_K"]       # 502.2 K (simplified)
p_atm   = 101325.0                            # atmospheric (break BC)

dt      = 5e-5                             # 0.05 ms (CFL for acoustic: dx/c ~ 0.17/1200 ~ 0.14 ms)
t_end   = 0.6                             # 0.6 s transient
n_steps = int(t_end / dt)

# Property evaluator
fluid = tp.IAPWSIF97Properties()

# ============================================================================
# Initial conditions
# ============================================================================

# Compute initial enthalpy from (p, T) using IAPWS
# We need h = h(p, T). Use the evaluate interface at a known T:
# Evaluate at a slightly different h to get T, then iterate to find h(p,T).
# Simpler: use iapws Python package for initial enthalpy.
try:
    import iapws
    ref = iapws.IAPWS97(P=p_init/1e6, T=T_init)
    h_init = ref.h * 1e3  # kJ/kg -> J/kg
    print(f"Initial conditions: p={p_init/1e6:.1f} MPa, T={T_init:.1f} K, "
          f"h={h_init/1e3:.1f} kJ/kg, rho={ref.rho:.1f} kg/m3")
except ImportError:
    # Fallback: approximate enthalpy for subcooled water at 7 MPa, 502 K
    h_init = 980.0e3  # ~980 kJ/kg (subcooled liquid at 7 MPa, 502 K)
    print(f"Initial conditions (approx): p={p_init/1e6:.1f} MPa, T={T_init:.1f} K, "
          f"h={h_init/1e3:.1f} kJ/kg")

# Break enthalpy (for outflow BC — donor cell uses last cell's enthalpy)
h_break = h_init  # initial value, not used if flow is outward

# State arrays
p    = np.full(N, p_init)
h    = np.full(N, h_init)
mdot = np.zeros(N + 1)  # face mass flows

# ============================================================================
# Gauge station cell indices (for output)
# ============================================================================

gauge_stations = edwards_blowdown["gauge_stations"]
gs_cells = {}
for name, gs in gauge_stations.items():
    cell_idx = min(int(gs["x_m"] / dx), N - 1)
    gs_cells[name] = cell_idx

print(f"\nMesh: {N} cells, dx={dx:.4f} m, dt={dt*1e3:.3f} ms, {n_steps} steps")
print(f"Gauge stations → cell indices: { {k: v for k, v in gs_cells.items()} }")

# ============================================================================
# Semi-implicit solver with wall BC
# ============================================================================

def evaluate_all_properties(p, h):
    """Evaluate IAPWS properties at all cells."""
    props = []
    for i in range(N):
        props.append(fluid.evaluate(p[i], h[i]))
    return props

def compute_R_face(props):
    """Face resistances. Face 0 = wall (infinite R), Face N = break."""
    geom_coeff = f_D * dx / (2.0 * D_h * A_flow**2)
    R = np.zeros(N + 1)

    # Face 0: WALL (closed end) — infinite resistance, no flow
    R[0] = 1e30

    # Interior faces: arithmetic average density
    for i in range(1, N):
        rho_face = 0.5 * (props[i-1].rho + props[i].rho)
        R[i] = geom_coeff / rho_face

    # Face N: break (open end) — use last cell density
    R[N] = geom_coeff / props[N-1].rho

    return R

def pressure_solve(p_old, props, R, dt):
    """Implicit pressure solve with wall BC at face 0, pressure BC at face N."""
    p_new = np.zeros(N)

    # Build tridiagonal: a[i]*p[i-1] + b[i]*p[i] + c[i]*p[i+1] = d[i]
    a = np.zeros(N)
    b = np.zeros(N)
    c = np.zeros(N)
    d = np.zeros(N)

    for i in range(N):
        alpha = V_cell * props[i].drho_dp_h / dt

        # Wall BC: face 0 has no connection (inv_R_left = 0 for cell 0)
        inv_R_left  = 0.0 if i == 0 else 1.0 / R[i]
        inv_R_right = 1.0 / R[i + 1]

        a[i] = -inv_R_left if i > 0 else 0.0
        c[i] = -inv_R_right if i < N - 1 else 0.0
        b[i] = alpha + inv_R_left + inv_R_right
        d[i] = alpha * p_old[i]

        # Pressure BC at break (face N, cell N-1)
        if i == N - 1:
            d[i] += p_atm * inv_R_right

    # Thomas algorithm
    c_prime = np.zeros(N)
    d_prime = np.zeros(N)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, N):
        denom = b[i] - a[i] * c_prime[i-1]
        c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

    p_new[N-1] = d_prime[N-1]
    for i in range(N-2, -1, -1):
        p_new[i] = d_prime[i] - c_prime[i] * p_new[i+1]

    return p_new

def update_flows(p, R):
    """Algebraic flow update. Wall at face 0, pressure BC at face N."""
    mdot_new = np.zeros(N + 1)
    mdot_new[0] = 0.0  # wall BC
    for i in range(1, N):
        mdot_new[i] = (p[i-1] - p[i]) / R[i]
    mdot_new[N] = (p[N-1] - p_atm) / R[N]
    return mdot_new

def update_enthalpy(h_old, p_new, p_old, mdot, props, dt):
    """Explicit enthalpy update (donor-cell upwind, forward Euler)."""
    h_new = np.copy(h_old)

    for i in range(N):
        rho_i = props[i].rho

        # Inlet face enthalpy (face i) — donor cell
        if mdot[i] >= 0.0:
            h_face_in = h_old[i-1] if i > 0 else h_old[0]  # wall: use cell 0
        else:
            h_face_in = h_old[i]

        # Outlet face enthalpy (face i+1) — donor cell
        if mdot[i+1] >= 0.0:
            h_face_out = h_old[i]
        else:
            h_face_out = h_old[i+1] if i < N-1 else h_old[i]

        flux = mdot[i] * (h_face_in - h_old[i]) - mdot[i+1] * (h_face_out - h_old[i])
        p_work = V_cell * (p_new[i] - p_old[i]) / dt

        h_new[i] = h_old[i] + dt / (rho_i * V_cell) * (flux + p_work)

    return h_new

# ============================================================================
# Time integration
# ============================================================================

# Output storage: save at selected times
save_times = np.concatenate([
    np.arange(0, 0.01, 0.001),     # fine resolution for first 10 ms
    np.arange(0.01, 0.1, 0.005),   # medium for 10-100 ms
    np.arange(0.1, 0.61, 0.01),    # coarse for 100-600 ms
])
history = []  # list of (t, p_array, h_array, mdot_array)
next_save_idx = 0

print(f"\nRunning Edwards blowdown simulation...")
print(f"{'step':>8s} {'t_ms':>8s} {'p_GS1_MPa':>10s} {'p_GS7_MPa':>10s} {'mdot_break':>12s}")

t = 0.0
for step in range(n_steps):
    # Save snapshot
    while next_save_idx < len(save_times) and t >= save_times[next_save_idx] - 0.5*dt:
        history.append((t, p.copy(), h.copy(), mdot.copy()))
        next_save_idx += 1

    # Print progress
    if step % 2000 == 0 or step == n_steps - 1:
        p_gs1 = p[gs_cells["GS-1"]] / 1e6
        p_gs7 = p[gs_cells["GS-7"]] / 1e6
        print(f"{step:8d} {t*1e3:8.2f} {p_gs1:10.3f} {p_gs7:10.3f} {mdot[N]:12.3f}")

    # --- Semi-implicit step ---
    p_old = p.copy()
    props = evaluate_all_properties(p, h)
    R = compute_R_face(props)

    p = pressure_solve(p_old, props, R, dt)
    mdot = update_flows(p, R)
    h = update_enthalpy(h, p, p_old, mdot, props, dt)

    t += dt

# Save final state
history.append((t, p.copy(), h.copy(), mdot.copy()))

print(f"\nSimulation complete: {n_steps} steps, t_final = {t*1e3:.2f} ms")

# ============================================================================
# Load experimental data and compare
# ============================================================================

data_dir = pathlib.Path(__file__).parent / "data"
fig3_path = data_dir / "fig3.csv"

if fig3_path.exists():
    # fig3.csv: time [s], pressure [psia] at GS-1
    exp_data = np.loadtxt(fig3_path, delimiter=",")
    t_exp = exp_data[:, 0]
    p_exp_psia = exp_data[:, 1]
    p_exp_MPa = p_exp_psia * 6894.76 / 1e6  # psia -> MPa

    # Extract simulation pressure at GS-1 at matching times
    gs1_cell = gs_cells["GS-1"]
    t_sim = np.array([h[0] for h in history])
    p_sim_gs1 = np.array([h[1][gs1_cell] for h in history]) / 1e6  # Pa -> MPa

    print(f"\n{'='*60}")
    print(f"COMPARISON: GS-1 Pressure (simulation vs experiment)")
    print(f"{'='*60}")
    print(f"{'t_exp_ms':>10s} {'p_exp_MPa':>10s} {'p_sim_MPa':>10s} {'err_%':>8s}")

    for i in range(len(t_exp)):
        # Interpolate simulation to experimental time
        p_interp = np.interp(t_exp[i], t_sim, p_sim_gs1)
        if p_exp_MPa[i] > 0.1:
            err_pct = (p_interp - p_exp_MPa[i]) / p_exp_MPa[i] * 100
        else:
            err_pct = float('nan')
        print(f"{t_exp[i]*1e3:10.2f} {p_exp_MPa[i]:10.3f} {p_interp:10.3f} {err_pct:8.1f}")
else:
    print("\nfig3.csv not found — skipping experimental comparison")

# ============================================================================
# Save results for plotting
# ============================================================================

results_path = data_dir / "edwards_results.npz"
t_arr = np.array([h[0] for h in history])
p_arr = np.array([h[1] for h in history])
h_arr = np.array([h[2] for h in history])
mdot_arr = np.array([h[3] for h in history])

np.savez(results_path,
         t=t_arr, p=p_arr, h=h_arr, mdot=mdot_arr,
         dx=dx, N=N, dt=dt, gs_cells=gs_cells)
print(f"\nResults saved to {results_path}")
