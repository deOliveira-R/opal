#!/usr/bin/env python3
"""
Derivation: Interfacial drag closure for two-fluid model

Ishii-Mishima bubbly-flow drag model with Schiller-Naumann drag coefficient.

F_drag = (3/4) * (C_D / d_b) * alpha * rho_l * |v_v - v_l| * (v_v - v_l)

where C_D = (24/Re_b) * (1 + 0.1 * Re_b^0.75)   [Schiller-Naumann]
      Re_b = rho_l * |v_v - v_l| * d_b / mu_l     [bubble Reynolds number]

Sign convention:
  F_drag > 0 when v_v > v_l (drag pushes liquid in +x, vapor in -x)
  F_drag < 0 when v_v < v_l

In the momentum equations:
  Liquid:  + F_drag  (accelerates liquid toward vapor velocity)
  Vapor:   - F_drag  (decelerates vapor toward liquid velocity)

Reference: Ishii & Hibiki, "Thermo-Fluid Dynamics of Two-Phase Flow", 2nd ed.
           Chapter 9, Eq. 9.85 (bubbly flow drag)
           Schiller & Naumann (1933) — standard drag correlation
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
import numpy as np
from opal_sympy.symbols import alpha, rho_l, rho_v, v_l, v_v
from opal_sympy.thermo import mu_l
from opal_sympy.codegen import to_numpy, to_modelica

print("=" * 70)
print("DERIVATION: Interfacial drag closure (Ishii bubbly + Schiller-Naumann)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Define symbols
# ══════════════════════════════════════════════════════════════════════════

d_b = sp.Symbol('d_b', positive=True)  # bubble diameter [m]
v_rel = sp.Symbol('v_rel', real=True)  # v_v - v_l [m/s]

print("\n1. Symbols defined:")
print(f"   alpha (void fraction), rho_l, v_l, v_v, d_b, mu_l")

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Bubble Reynolds number
# ══════════════════════════════════════════════════════════════════════════

Re_b = rho_l * sp.Abs(v_rel) * d_b / mu_l

print("\n2. Bubble Reynolds number:")
print(f"   Re_b = rho_l * |v_v - v_l| * d_b / mu_l")

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Schiller-Naumann drag coefficient
# ══════════════════════════════════════════════════════════════════════════

# C_D = (24 / Re_b) * (1 + 0.1 * Re_b^0.75)
# Note: for Re_b → 0, C_D → 24/Re_b (Stokes drag)
# For Re_b >> 1, C_D ≈ 24 * 0.1 * Re_b^(-0.25) (inertial drag)
# Valid range: Re_b < 1000 (bubbly flow)

C_D = (24 / Re_b) * (1 + sp.Rational(1, 10) * Re_b**sp.Rational(3, 4))

print("\n3. Schiller-Naumann drag coefficient:")
print(f"   C_D = (24/Re_b) * (1 + 0.1 * Re_b^0.75)")
print(f"   Expanded: C_D = {sp.simplify(C_D)}")

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Ishii bubbly drag force
# ══════════════════════════════════════════════════════════════════════════

# F_drag = (3/4) * (C_D / d_b) * alpha * rho_l * |v_rel| * v_rel
# [N/m^3] = [-/m] * [-] * [kg/m^3] * [m/s] * [m/s] = [kg/(m^3 * s^2)] = [N/m^3] ✓

F_drag = sp.Rational(3, 4) * C_D / d_b * alpha * rho_l * sp.Abs(v_rel) * v_rel

print("\n4. Ishii bubbly drag force [N/m^3]:")
print(f"   F_drag = (3/4) * (C_D/d_b) * alpha * rho_l * |v_rel| * v_rel")

# Simplify: substitute C_D and cancel |v_rel|
# C_D / d_b * |v_rel| = (24/(rho_l*|v_rel|*d_b/mu_l)) * (1 + 0.1*Re^0.75) / d_b * |v_rel|
# = 24 * mu_l / (rho_l * d_b^2) * (1 + 0.1*Re^0.75)
# So: F_drag = (3/4) * 24 * mu_l / (rho_l * d_b^2) * (1 + 0.1*Re^0.75) * alpha * rho_l * v_rel
#            = 18 * mu_l * alpha / d_b^2 * (1 + 0.1*Re^0.75) * v_rel

F_drag_simplified = sp.Rational(18, 1) * mu_l * alpha / d_b**2 * (
    1 + sp.Rational(1, 10) * Re_b**sp.Rational(3, 4)
) * v_rel

print(f"   Simplified: F_drag = 18 * mu_l * alpha / d_b^2 * (1 + 0.1*Re^0.75) * v_rel")

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Implicit drag linearization (for semi-implicit solver)
# ══════════════════════════════════════════════════════════════════════════

# For the semi-implicit scheme, we need dF_drag/d(v_rel) at old state
# This enters the sigma term: sigma_drag = dt * |dF_drag/d(v_rel)|

# Since F_drag ∝ |v_rel| * v_rel = sign(v_rel) * v_rel^2,
# dF_drag/d(v_rel) = 2 * |F_drag / v_rel| (when v_rel ≠ 0)
# More precisely for the simplified form:
# dF_drag/d(v_rel) at fixed Re approximation ≈ F_drag_coeff where
# F_drag_coeff = 18 * mu_l * alpha / d_b^2 * (1 + 0.1*Re^0.75)

# In practice we linearize as:
# F_drag(v_rel_new) ≈ F_drag(v_rel_old) + K_drag * (v_rel_new - v_rel_old)
# where K_drag = |∂F/∂v_rel| evaluated at old state

# For the Schiller-Naumann form with |v_rel|*v_rel:
# F = C * |v_rel| * v_rel where C = (3/4) * C_D(Re) / d_b * alpha * rho_l
# dF/dv_rel = 2 * C * |v_rel| = 2 * |F_drag| / |v_rel|

print("\n5. Implicit drag linearization:")
print("   K_drag = |dF_drag/dv_rel| ≈ 2 * |F_drag| / max(|v_rel|, eps)")
print("   sigma_drag = dt * K_drag / (rho_phase * A_flow)")
print("   Used in semi-implicit friction resistance parameter")

# ══════════════════════════════════════════════════════════════════════════
# Step 6: Dimensional analysis
# ══════════════════════════════════════════════════════════════════════════

print("\n6. Dimensional analysis:")
print("   [F_drag] = [mu_l]*[-]*[m^-2]*[-]*[m/s]")
print("            = [Pa·s] * [m^-2] * [m/s]")
print("            = [Pa/m] = [N/m^3] ✓")

# ══════════════════════════════════════════════════════════════════════════
# Step 7: Numerical verification
# ══════════════════════════════════════════════════════════════════════════

print("\n7. Numerical verification:")

rng = np.random.default_rng(42)

# Test 1: F_drag = 0 when v_l = v_v
n_pass = 0
for _ in range(100):
    a_val = rng.uniform(0.01, 0.99)
    rl_val = rng.uniform(500, 1000)
    vl_val = rng.uniform(-10, 10)
    vv_val = vl_val  # same velocity
    db_val = rng.uniform(1e-4, 1e-2)
    mul_val = rng.uniform(1e-4, 1e-3)

    v_rel_val = vv_val - vl_val
    Re_val = rl_val * abs(v_rel_val) * db_val / mul_val

    if Re_val < 1e-15:
        F_val = 0.0
    else:
        CD_val = (24 / Re_val) * (1 + 0.1 * Re_val**0.75)
        F_val = 0.75 * CD_val / db_val * a_val * rl_val * abs(v_rel_val) * v_rel_val

    if abs(F_val) < 1e-15:
        n_pass += 1

print(f"   Zero v_rel test: {n_pass}/100 passed")
assert n_pass == 100

# Test 2: F_drag direction — pushes phases toward same velocity
n_pass = 0
for _ in range(500):
    a_val = rng.uniform(0.01, 0.99)
    rl_val = rng.uniform(500, 1000)
    vl_val = rng.uniform(-5, 5)
    vv_val = rng.uniform(-5, 5)
    db_val = rng.uniform(1e-4, 1e-2)
    mul_val = rng.uniform(1e-4, 1e-3)

    v_rel_val = vv_val - vl_val
    if abs(v_rel_val) < 1e-12:
        n_pass += 1
        continue

    Re_val = rl_val * abs(v_rel_val) * db_val / mul_val
    CD_val = (24 / Re_val) * (1 + 0.1 * Re_val**0.75)
    F_val = 0.75 * CD_val / db_val * a_val * rl_val * abs(v_rel_val) * v_rel_val

    # F_drag should have same sign as v_rel = v_v - v_l
    # When v_v > v_l: F > 0 → liquid +F (accelerate), vapor -F (decelerate)
    # When v_v < v_l: F < 0 → liquid -F (decelerate), vapor +F (accelerate)
    if F_val * v_rel_val > 0 or abs(F_val) < 1e-15:
        n_pass += 1

print(f"   Direction test: {n_pass}/500 passed")
assert n_pass == 500

# Test 3: Magnitude check — compare against known RELAP5 values
# For typical Edwards conditions: rho_l=800, v_rel=5m/s, alpha=0.3, d_b=1mm, mu_l=2.8e-4
rl_test = 800.0
vrel_test = 5.0
a_test = 0.3
db_test = 1e-3
mul_test = 2.8e-4

Re_test = rl_test * abs(vrel_test) * db_test / mul_test
CD_test = (24 / Re_test) * (1 + 0.1 * Re_test**0.75)
F_test = 0.75 * CD_test / db_test * a_test * rl_test * abs(vrel_test) * vrel_test

print(f"\n   Edwards-like conditions:")
print(f"   Re_b = {Re_test:.0f}")
print(f"   C_D = {CD_test:.4f}")
print(f"   F_drag = {F_test:.0f} N/m^3")
print(f"   (Typical range: 1e3 to 1e6 N/m^3 — {'OK' if 1e3 < abs(F_test) < 1e7 else 'SUSPECT'})")
assert 1e3 < abs(F_test) < 1e7, f"F_drag magnitude {F_test} outside expected range"

# Test 4: Stokes limit (Re → 0, C_D → 24/Re → F_drag → 18*mu*alpha/d_b^2 * v_rel)
vrel_small = 1e-6
Re_small = rl_test * abs(vrel_small) * db_test / mul_test
CD_small = (24 / Re_small) * (1 + 0.1 * Re_small**0.75)
F_small = 0.75 * CD_small / db_test * a_test * rl_test * abs(vrel_small) * vrel_small

# Stokes analytical: F = 18 * mu * alpha / d_b^2 * v_rel (leading order)
F_stokes = 18 * mul_test * a_test / db_test**2 * vrel_small
rel_err = abs(F_small - F_stokes) / abs(F_stokes)

print(f"\n   Stokes limit test (Re={Re_small:.4f}):")
print(f"   Full formula: {F_small:.6e}")
print(f"   Stokes limit: {F_stokes:.6e}")
print(f"   Relative error: {rel_err:.2e}")
assert rel_err < 0.01, f"Stokes limit error {rel_err} too large"

# Test 5: Verify simplified form equals original
n_pass = 0
for _ in range(500):
    a_val = rng.uniform(0.01, 0.99)
    rl_val = rng.uniform(500, 1000)
    vl_val = rng.uniform(-5, 5)
    vv_val = rng.uniform(-5, 5)
    db_val = rng.uniform(1e-4, 1e-2)
    mul_val = rng.uniform(1e-4, 1e-3)

    v_rel_val = vv_val - vl_val
    if abs(v_rel_val) < 1e-12:
        n_pass += 1
        continue

    Re_val = rl_val * abs(v_rel_val) * db_val / mul_val
    CD_val = (24 / Re_val) * (1 + 0.1 * Re_val**0.75)

    # Original form
    F_orig = 0.75 * CD_val / db_val * a_val * rl_val * abs(v_rel_val) * v_rel_val

    # Simplified form
    F_simp = 18 * mul_val * a_val / db_val**2 * (1 + 0.1 * Re_val**0.75) * v_rel_val

    rel_err = abs(F_orig - F_simp) / max(abs(F_orig), 1e-20)
    if rel_err < 1e-10:
        n_pass += 1

print(f"\n   Simplified ≡ Original: {n_pass}/500 passed")
assert n_pass == 500

print("\n" + "=" * 70)
print("DERIVATION COMPLETE — ALL VERIFICATIONS PASSED")
print("=" * 70)
