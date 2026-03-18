#!/usr/bin/env python3
"""
Derivation: 5-equation drift-flux pressure linearization (1D)

The 5-equation model solves:
  - Liquid mass:  ∂/∂t[(1-α)ρ_l] + ∂/∂z[(1-α)ρ_l·v_l] = -Γ
  - Vapor mass:   ∂/∂t[α·ρ_v]    + ∂/∂z[α·ρ_v·v_v]    = +Γ
  - Mixture momentum (algebraic with drift-flux slip)
  - Liquid energy (separate)
  - Vapor energy (separate)

The pressure equation comes from summing the two phasic mass equations
(Γ cancels) to get mixture mass conservation, then linearizing around
old pressure.

State variables: (P, α, h_l, h_v, mdot_m)

Reference: RELAP5/MOD3 Code Manual, Volume 1, Section 3.1
           Ishii & Hibiki, "Thermo-Fluid Dynamics of Two-Phase Flow"
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from opal_sympy.symbols import (
    P, P_old, alpha, rho_l, rho_v, dt, V_cell, A_flow
)
from opal_sympy.thermo import (
    drho_l_dP_h, drho_v_dP_h,
    mixture_density, mixture_density_derivative_P,
)
from opal_sympy.codegen import to_c, to_numpy
from opal_sympy.conservation import verify_mass_conservation

print("=" * 70)
print("DERIVATION: 5-equation pressure linearization (1D)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Mixture density and its pressure derivative
# ══════════════════════════════════════════════════════════════════════════

rho_m = mixture_density(alpha, rho_l, rho_v)
drho_m_dP = mixture_density_derivative_P(alpha)

print("\n1. Mixture density:")
print(f"   ρ_m = {rho_m}")
print(f"   ∂ρ_m/∂P|_{{h,α}} = {drho_m_dP}")

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Discrete accumulation (same form as HEM)
# ══════════════════════════════════════════════════════════════════════════

rho_m_old = sp.Symbol('rho_m_old', positive=True)
rho_m_linearized = rho_m_old + drho_m_dP * (P - P_old)
accumulation = V_cell / dt * (rho_m_linearized - rho_m_old)
accumulation = sp.expand(accumulation)

expected = V_cell / dt * drho_m_dP * (P - P_old)
assert sp.expand(accumulation - expected) == 0

print("\n2. Accumulation (linearized):")
print(f"   V/Δt · ∂ρ_m/∂P · (P - P_old)")
print("   ✓ Same tridiagonal structure as HEM")

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Pressure matrix diagonal
# ══════════════════════════════════════════════════════════════════════════

a_diag = V_cell / dt * drho_m_dP

print("\n3. Pressure matrix diagonal:")
print(f"   a_ii = V/Δt · [(1-α)·∂ρ_l/∂P + α·∂ρ_v/∂P]")

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Void fraction transport (explicit)
# ══════════════════════════════════════════════════════════════════════════

alpha_old = sp.Symbol('alpha_old', nonnegative=True)
rho_v_old = sp.Symbol('rho_v_old', positive=True)
rho_v_new = sp.Symbol('rho_v_new', positive=True)
Gamma = sp.Symbol('Gamma', real=True)
flux_v_net = sp.Symbol('flux_v_net', real=True)

alpha_rho_v_old = alpha_old * rho_v_old
alpha_rho_v_new = alpha_rho_v_old + dt / V_cell * (flux_v_net + V_cell * Gamma)
alpha_new = alpha_rho_v_new / rho_v_new

print("\n4. Void fraction transport (explicit):")
print(f"   (αρ_v)^{{n+1}} = (αρ_v)^n + Δt/V · [flux_v_net + V·Γ]")
print(f"   α^{{n+1}} = (αρ_v)^{{n+1}} / ρ_v(P^{{n+1}}, h_v)")

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Drift-flux slip model — phasic velocity split
# ══════════════════════════════════════════════════════════════════════════
# Given mixture mass flux G_m = mdot_m/A, find (v_l, v_v) via drift-flux.
#
# Drift-flux: v_v = C_0·j + V_gj
# where j = volumetric flux = (1-α)v_l + α·v_v
#
# Key relation (Ishii & Hibiki, Eq. 6.1-3):
#   G_m = j·[ρ_l + α·C_0·(ρ_v - ρ_l)] + α·V_gj·(ρ_v - ρ_l)
#
# Solving for j:
#   j = [G_m - α·V_gj·(ρ_v - ρ_l)] / [ρ_l + α·C_0·(ρ_v - ρ_l)]

C_0 = sp.Symbol('C_0', positive=True)
V_gj = sp.Symbol('V_gj', real=True)
G_m = sp.Symbol('G_m', real=True)  # mixture mass flux [kg/m²/s]

# Denominator of j expression
rho_eff = rho_l + alpha * C_0 * (rho_v - rho_l)

j = (G_m - alpha * V_gj * (rho_v - rho_l)) / rho_eff

v_v_drift = C_0 * j + V_gj
v_l_drift = (j - alpha * v_v_drift) / (1 - alpha)

# Phasic mass fluxes
G_v = alpha * rho_v * v_v_drift
G_l = (1 - alpha) * rho_l * v_l_drift

print("\n5. Drift-flux phasic velocities:")
print(f"   j = [G_m - α·V_gj·(ρ_v - ρ_l)] / [ρ_l + α·C_0·(ρ_v - ρ_l)]")
print(f"   v_v = C_0·j + V_gj")
print(f"   v_l = [j - α·v_v] / (1-α)")

# Verify: G_l + G_v = G_m
G_sum = sp.simplify(sp.expand(G_l + G_v))
residual_flux = sp.simplify(G_sum - G_m)
print(f"\n   Symbolic G_l + G_v - G_m = {residual_flux}")

# Numerical verification of flux conservation
f_resid = to_numpy(
    sp.expand(G_l + G_v - G_m),
    [alpha, rho_l, rho_v, G_m, C_0, V_gj]
)

import numpy as np
rng = np.random.default_rng(42)
max_err = 0.0
for _ in range(500):
    a_val = rng.uniform(0.01, 0.99)
    rl_val = rng.uniform(500, 1000)
    rv_val = rng.uniform(1, 100)
    Gm_val = rng.uniform(-5000, 5000)
    C0_val = rng.uniform(1.0, 1.3)
    Vgj_val = rng.uniform(-2, 2)
    err = abs(f_resid(a_val, rl_val, rv_val, Gm_val, C0_val, Vgj_val))
    max_err = max(max_err, err)

print(f"   G_l + G_v = G_m? max|error| = {max_err:.2e}")
assert max_err < 1e-8, f"Phasic flux sum FAILED: max error = {max_err}"
print("   ✓ Phasic fluxes sum to mixture mass flux (500 random states)")

# ══════════════════════════════════════════════════════════════════════════
# Step 6: HEM limit verification
# ══════════════════════════════════════════════════════════════════════════

# When C_0 = 1 and V_gj = 0 (no slip), v_l = v_v = j = G_m/ρ_m
j_no_slip = j.subs([(C_0, 1), (V_gj, 0)])
j_no_slip = sp.simplify(j_no_slip)
rho_m_sym = (1 - alpha) * rho_l + alpha * rho_v
j_expected = G_m / rho_m_sym
residual_hem = sp.simplify(j_no_slip - j_expected)
print(f"\n6. HEM limit (C_0=1, V_gj=0):")
print(f"   j = G_m/ρ_m? residual = {residual_hem}")
assert residual_hem == 0, "HEM limit FAILED"
print("   ✓ Drift-flux reduces to HEM when C_0=1, V_gj=0")

# ══════════════════════════════════════════════════════════════════════════
# Step 7: Generate C code
# ══════════════════════════════════════════════════════════════════════════

print("\n7. Generated C code:")
c_code = to_c(a_diag, 'pressure_diagonal_5eq',
    [V_cell, dt, alpha, drho_l_dP_h, drho_v_dP_h])
print(c_code)

# ══════════════════════════════════════════════════════════════════════════
# Step 8: Numerical verification — pressure diagonal
# ══════════════════════════════════════════════════════════════════════════

print("\n8. Numerical verification:")

f_diag = to_numpy(a_diag, [V_cell, dt, alpha, drho_l_dP_h, drho_v_dP_h])

n_pass = 0
n_total = 200
for _ in range(n_total):
    V_val = rng.uniform(0.001, 0.1)
    dt_val = rng.uniform(0.001, 0.1)
    alpha_val = rng.uniform(0.0, 1.0)
    drho_l_val = rng.uniform(1e-7, 1e-5)
    drho_v_val = rng.uniform(1e-6, 1e-4)

    computed = f_diag(V_val, dt_val, alpha_val, drho_l_val, drho_v_val)
    expected_val = V_val / dt_val * (
        (1 - alpha_val) * drho_l_val + alpha_val * drho_v_val
    )
    if abs(computed - expected_val) < 1e-10 * max(abs(expected_val), 1e-20):
        n_pass += 1

print(f"   Pressure diagonal: {n_pass}/{n_total} passed")
assert n_pass == n_total, "Pressure diagonal verification FAILED"

# HEM limit: alpha=0 → V/dt * drho_l_dP_h
f_hem_limit = f_diag(0.05, 0.01, 0.0, 3e-6, 5e-5)
f_hem_expected = 0.05 / 0.01 * 3e-6
assert abs(f_hem_limit - f_hem_expected) < 1e-15
print("   HEM limit (α=0): PASS")

# Conservation
assert verify_mass_conservation()
print("   Mixture mass conservation (Γ cancels): PASS")

print("\n" + "=" * 70)
print("DERIVATION COMPLETE — ALL VERIFICATIONS PASSED")
print("=" * 70)
