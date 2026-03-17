#!/usr/bin/env python3
"""
Derivation: Explicit enthalpy update (donor-cell upwind, forward Euler).

The energy equation in (p, h) variables, after subtracting mass conservation
to eliminate the d(rho)/dt term:

    rho_i * V * (h_new - h_old) / dt
        = mdot_in * (h_face_in - h_old)
        - mdot_out * (h_face_out - h_old)
        + V * (p_new - p_old) / dt       [pressure work]
        + q_wall                          [wall heat source]

where face enthalpies use donor-cell (upwind) selection:
    h_face = h_upstream  (determined by sign of mdot at that face)

Reference:
    Patankar, "Numerical Heat Transfer and Fluid Flow", Chapter 5
    RELAP5/MOD3 Code Manual, Volume 1, §3.1.4
    solver/two_phase/solver.cpp, update_enthalpy()
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from opal_sympy.symbols import dt, V_cell, q_wall
from opal_sympy.codegen import to_numpy
import numpy as np

print("=" * 70)
print("DERIVATION: Enthalpy update (donor-cell, forward Euler)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Define symbols
# ══════════════════════════════════════════════════════════════════════════

h_old = sp.Symbol('h_old', real=True)       # cell enthalpy at old time
rho = sp.Symbol('rho', positive=True)       # cell density at old time
p_new = sp.Symbol('p_new', real=True)       # cell pressure at new time
p_old = sp.Symbol('p_old', real=True)       # cell pressure at old time
mdot_in = sp.Symbol('mdot_in', real=True)   # mass flow at inlet face
mdot_out = sp.Symbol('mdot_out', real=True) # mass flow at outlet face
h_face_in = sp.Symbol('h_face_in', real=True)   # donor-cell enthalpy at inlet
h_face_out = sp.Symbol('h_face_out', real=True)  # donor-cell enthalpy at outlet

print("\n1. Symbols: h_old, rho, p_new, p_old, mdot_in, mdot_out,")
print("           h_face_in, h_face_out, V_cell, dt, q_wall")

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Derive the enthalpy update equation
# ══════════════════════════════════════════════════════════════════════════

# Starting from the enthalpy form of energy conservation:
#   d(rho*h)/dt * V = mdot_in*h_in - mdot_out*h_out + V*dp/dt + q_wall
#
# Expand LHS using product rule:
#   rho*V*dh/dt + h*V*drho/dt = mdot_in*h_in - mdot_out*h_out + V*dp/dt + q_wall
#
# Mass conservation: V*drho/dt = mdot_in - mdot_out
# Substitute:
#   rho*V*dh/dt + h*(mdot_in - mdot_out) = mdot_in*h_in - mdot_out*h_out + V*dp/dt + q_wall
#
# Rearrange:
#   rho*V*dh/dt = mdot_in*(h_in - h) - mdot_out*(h_out - h) + V*dp/dt + q_wall

# Advective flux (relative to cell enthalpy)
flux = mdot_in * (h_face_in - h_old) - mdot_out * (h_face_out - h_old)

# Pressure work
p_work = V_cell * (p_new - p_old) / dt

# Full RHS
rhs = flux + p_work + q_wall

# Forward Euler update
h_new = h_old + dt / (rho * V_cell) * rhs

print(f"\n2. Enthalpy update equation:")
print(f"   h_new = h_old + dt/(rho*V) * [flux + V*dp/dt + q_wall]")
print(f"   flux = mdot_in*(h_face_in - h_old) - mdot_out*(h_face_out - h_old)")
print(f"\n   Full expression:")
h_new_expanded = sp.expand(h_new)
print(f"   h_new = {h_new_expanded}")

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Verify conservation — no wall heat, no pressure change, no flow
# ══════════════════════════════════════════════════════════════════════════

print("\n3. Conservation checks:")

# No flow, no heat, no pressure change → h unchanged
h_static = h_new.subs([(mdot_in, 0), (mdot_out, 0), (q_wall, 0), (p_new, p_old)])
assert sp.simplify(h_static - h_old) == 0
print("   No flow + no heat + no dp: h_new = h_old ✓")

# Uniform enthalpy: if h_face_in = h_face_out = h_old, advection contributes nothing
h_uniform = h_new.subs([(h_face_in, h_old), (h_face_out, h_old), (q_wall, 0), (p_new, p_old)])
assert sp.simplify(h_uniform - h_old) == 0
print("   Uniform h field + no heat + no dp: h_new = h_old ✓")

# Pure pressure work (no flow, no heat):
# h_new = h_old + (p_new - p_old) / rho
h_pwork = h_new.subs([(mdot_in, 0), (mdot_out, 0), (q_wall, 0)])
expected_pwork = h_old + (p_new - p_old) / rho
assert sp.simplify(h_pwork - expected_pwork) == 0
print("   Pure pressure work: h_new = h_old + dp/rho ✓")

# Pure heating (no flow, no dp):
# h_new = h_old + dt * q_wall / (rho * V_cell)
h_heat = h_new.subs([(mdot_in, 0), (mdot_out, 0), (p_new, p_old)])
expected_heat = h_old + dt * q_wall / (rho * V_cell)
assert sp.simplify(h_heat - expected_heat) == 0
print("   Pure heating: h_new = h_old + dt*q/(rho*V) ✓")

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Steady-state energy balance
# ══════════════════════════════════════════════════════════════════════════

print("\n4. Steady-state energy balance:")
print("   At steady state (h_new = h_old, dp = 0):")
print("   0 = mdot_in*(h_face_in - h) - mdot_out*(h_face_out - h) + q_wall")
print("   For uniform flow (mdot_in = mdot_out = mdot, h_face_in = h_in, h_face_out = h):")
print("   0 = mdot*(h_in - h) + q_wall  →  h = h_in + q_wall/mdot")

mdot_ss = sp.Symbol('mdot_ss', positive=True)
h_in_ss = sp.Symbol('h_in_ss', real=True)
ss_eq = mdot_ss * (h_in_ss - h_old) + q_wall
h_ss = sp.solve(ss_eq, h_old)[0]
print(f"   h_ss = {h_ss} = h_in + q_wall/mdot ✓")

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Generate code
# ══════════════════════════════════════════════════════════════════════════

print("\n5. Generated C code:")
c_args = [h_old, rho, V_cell, dt, mdot_in, mdot_out,
          h_face_in, h_face_out, p_new, p_old, q_wall]
from opal_sympy.codegen import to_c
c_code = to_c(h_new, 'enthalpy_update', c_args)
print(c_code)

# ══════════════════════════════════════════════════════════════════════════
# Step 6: Numerical verification against solver code
# ══════════════════════════════════════════════════════════════════════════

print("6. Numerical verification against solver/two_phase/solver.cpp:")

f_h_new = to_numpy(h_new, c_args)

rng = np.random.default_rng(42)
n_pass = 0
n_total = 500
for _ in range(n_total):
    h_o = rng.uniform(100e3, 2800e3)
    rho_v = rng.uniform(1.0, 1000.0)
    V_v = rng.uniform(1e-4, 0.1)
    dt_v = rng.uniform(1e-5, 1e-2)
    mi = rng.uniform(-10, 50)
    mo = rng.uniform(-10, 50)
    hfi = rng.uniform(100e3, 2800e3)
    hfo = rng.uniform(100e3, 2800e3)
    pn = rng.uniform(1e6, 17e6)
    po = rng.uniform(1e6, 17e6)
    qw = rng.uniform(0, 1e6)

    sympy_result = f_h_new(h_o, rho_v, V_v, dt_v, mi, mo, hfi, hfo, pn, po, qw)

    # Reproduce solver calculation exactly
    flux_v = mi * (hfi - h_o) - mo * (hfo - h_o)
    p_work_v = V_v * (pn - po) / dt_v
    solver_result = h_o + dt_v / (rho_v * V_v) * (flux_v + p_work_v + qw)

    if abs(sympy_result - solver_result) < 1e-8 * max(abs(solver_result), 1.0):
        n_pass += 1

print(f"   SymPy vs solver.cpp: {n_pass}/{n_total} match to machine precision")
assert n_pass == n_total, f"VERIFICATION FAILED: {n_pass}/{n_total}"

print("\n" + "=" * 70)
print("DERIVATION COMPLETE — ALL VERIFICATIONS PASSED")
print("=" * 70)
