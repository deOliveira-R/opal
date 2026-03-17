#!/usr/bin/env python3
"""
Derivation: Semi-implicit pressure equation linearization (1D, two-phase)

This is the core equation of the RELAP5-heritage semi-implicit solver.
It produces the pressure matrix that is solved implicitly each time step.

Physics:
    Start from mixture mass conservation:
        ∂/∂t[ρ_m] + ∂/∂z[G_m] = 0
    where ρ_m = (1-α)ρ_l + αρ_v and G_m = (1-α)ρ_l·v_l + αρ_v·v_v

    Linearize ρ_m around old pressure:
        ρ_m(P) ≈ ρ_m(P_old) + (∂ρ_m/∂P)|_h · (P - P_old)

    Discretize in time (backward Euler for accumulation) and space
    (donor-cell for mass flux). The result is a tridiagonal system for P.

Reference: RELAP5/MOD3 Code Manual, Volume 1, Section 3.1
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
from opal_sympy.symbols import (
    P, P_old, alpha, rho_l, rho_v, v_l, v_v,
    dt, dx, V_cell, A_flow, i
)
from opal_sympy.thermo import (
    drho_l_dP_h, drho_v_dP_h,
    mixture_density, mixture_density_derivative_P,
    linearize_around,
)
from opal_sympy.stencil import center_1d, east_1d, west_1d
from opal_sympy.codegen import to_modelica, to_c, to_numpy
from opal_sympy.verify import check_conservation

print("=" * 70)
print("DERIVATION: Semi-implicit pressure linearization (1D)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Define mixture density and its pressure derivative
# ══════════════════════════════════════════════════════════════════════════

rho_m = mixture_density(alpha, rho_l, rho_v)
drho_m_dP = mixture_density_derivative_P(alpha)

print("\n1. Mixture density:")
print(f"   ρ_m = {rho_m}")
print(f"   ∂ρ_m/∂P|_h = {drho_m_dP}")

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Linearize mixture density around P_old
# ══════════════════════════════════════════════════════════════════════════

# ρ_m(P) ≈ ρ_m(P_old) + (∂ρ_m/∂P)(P - P_old)
# We define ρ_m_old as the value at P_old (known from previous time step)
rho_m_old = sp.Symbol('rho_m_old', positive=True)

rho_m_linearized = rho_m_old + drho_m_dP * (P - P_old)

print("\n2. Linearized mixture density:")
print(f"   ρ_m(P) ≈ {rho_m_linearized}")

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Discrete mass conservation (cell i)
# ══════════════════════════════════════════════════════════════════════════

# Accumulation: V_cell * (ρ_m^{n+1} - ρ_m^n) / dt
# Using linearized ρ_m^{n+1}:
accumulation = V_cell / dt * (rho_m_linearized - rho_m_old)

# Simplify: the rho_m_old terms cancel, leaving only the pressure correction
accumulation_simplified = sp.expand(accumulation)

print("\n3. Accumulation term (after cancellation):")
print(f"   V/Δt · (ρ_m^{'{n+1}'} - ρ_m^n) = {accumulation_simplified}")

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Verify the accumulation simplifies to pressure-only
# ══════════════════════════════════════════════════════════════════════════

# Should be: V_cell/dt * drho_m_dP * (P - P_old)
expected_accumulation = V_cell / dt * drho_m_dP * (P - P_old)
residual = sp.expand(accumulation_simplified - expected_accumulation)
assert residual == 0, f"Accumulation mismatch: residual = {residual}"
print("   ✓ Verified: accumulation = V/Δt · (∂ρ_m/∂P)(P - P_old)")

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Mass flux at faces (donor cell, linearized)
# ══════════════════════════════════════════════════════════════════════════

# At east face of cell i, the mass flux is:
#   G_east = (1-α_d)ρ_{l,d}·v_l + α_d·ρ_{v,d}·v_v
# where d = donor cell (upwind).
#
# In the semi-implicit scheme, velocities are linearized in pressure:
#   v^{n+1} ≈ v^n + (∂v/∂P_i)(P_i - P_i_old) + (∂v/∂P_{i+1})(P_{i+1} - P_{i+1}_old)
#
# This creates the off-diagonal pressure matrix entries.

# For now, define the pressure coefficient at east face abstractly:
# The full momentum linearization is in a separate derivation script.
dG_dP_i = sp.Symbol('dG_dP_i', real=True)       # ∂G_east/∂P_i
dG_dP_ip1 = sp.Symbol('dG_dP_ip1', real=True)   # ∂G_east/∂P_{i+1}

P_i = sp.IndexedBase('P')
P_old_i = sp.IndexedBase('P_old')

flux_east_linearized = (
    dG_dP_i * (P_i[i] - P_old_i[i])
    + dG_dP_ip1 * (P_i[i + 1] - P_old_i[i + 1])
)

print("\n5. Linearized mass flux at east face:")
print(f"   ΔG_east = {flux_east_linearized}")

# ══════════════════════════════════════════════════════════════════════════
# Step 6: Assemble pressure equation for cell i
# ══════════════════════════════════════════════════════════════════════════

# The discrete conservation equation:
#   accumulation + A_flow * (G_east - G_west) = 0
#
# After linearization, this gives:
#   a_{i-1} · δP_{i-1} + a_i · δP_i + a_{i+1} · δP_{i+1} = b_i
#
# where δP = P - P_old

# Diagonal coefficient (from accumulation + flux dependencies on P_i)
a_diag = V_cell / dt * drho_m_dP

print("\n6. Pressure matrix diagonal (from accumulation only):")
print(f"   a_ii = {a_diag}")
print("   (Off-diagonal terms come from momentum linearization — see")
print("    derivations/semi_implicit_momentum.py)")

# ══════════════════════════════════════════════════════════════════════════
# Step 7: Generate code
# ══════════════════════════════════════════════════════════════════════════

print("\n7. Generated Modelica code:")
modelica_code = to_modelica(a_diag, varname='a_diag[i]',
    comment='Pressure matrix diagonal (accumulation contribution)')
print(modelica_code)

print("\n   Generated C code:")
c_code = to_c(a_diag, 'pressure_diagonal_accum',
    [V_cell, dt, alpha, drho_l_dP_h, drho_v_dP_h])
print(c_code)

# ══════════════════════════════════════════════════════════════════════════
# Step 8: Numerical verification
# ══════════════════════════════════════════════════════════════════════════

print("8. Numerical verification:")

# Verify that the linearized accumulation integrates correctly:
# For constant drho_m_dP and a known pressure change ΔP,
# the mass change should be V_cell * drho_m_dP * ΔP
f_accum = to_numpy(accumulation_simplified, [V_cell, dt, alpha, drho_l_dP_h, drho_v_dP_h, P, P_old])

import numpy as np
rng = np.random.default_rng(42)
n_pass = 0
n_total = 100
for _ in range(n_total):
    V_val = rng.uniform(0.001, 0.1)
    dt_val = rng.uniform(0.001, 0.1)
    alpha_val = rng.uniform(0.0, 1.0)
    drho_l_val = rng.uniform(1e-7, 1e-5)
    drho_v_val = rng.uniform(1e-6, 1e-4)
    P_val = rng.uniform(1e6, 17e6)
    P_old_val = rng.uniform(1e6, 17e6)

    computed = f_accum(V_val, dt_val, alpha_val, drho_l_val, drho_v_val, P_val, P_old_val)
    expected_val = V_val / dt_val * (
        (1 - alpha_val) * drho_l_val + alpha_val * drho_v_val
    ) * (P_val - P_old_val)

    if abs(computed - expected_val) < 1e-10 * max(abs(expected_val), 1e-20):
        n_pass += 1

print(f"   Accumulation consistency: {n_pass}/{n_total} passed")
assert n_pass == n_total, "Numerical verification FAILED"

# Verify conservation: Gamma cancels in mixture
from opal_sympy.conservation import verify_mass_conservation
assert verify_mass_conservation()
print("   Mixture mass conservation (Γ cancels): PASS")

print("\n" + "=" * 70)
print("DERIVATION COMPLETE — ALL VERIFICATIONS PASSED")
print("=" * 70)
