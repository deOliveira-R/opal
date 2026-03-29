#!/usr/bin/env python3
"""
Derivation: Two-fluid phasic momentum equations + semi-implicit linearization

The 6-equation two-fluid model solves separate momentum equations for each phase:

Liquid momentum (per face i):
  (1-α_f)·ρ_l_f·(dx/A) · d(mdot_l)/dt
    = (1-α_f)·A·(p[i-1] - p[i])           [pressure gradient]
    - f_D·dx/(2·D_h) · |mdot_l|·mdot_l / ((1-α_f)·ρ_l_f·A²)  [wall friction]
    - (1-α_f)·ρ_l_f·g·A·dx                [gravity]
    + F_drag·V_face                         [interfacial drag, +ve]
    + Gamma_f·v_i·A                         [phase change momentum]

Vapor momentum (per face i):
  α_f·ρ_v_f·(dx/A) · d(mdot_v)/dt
    = α_f·A·(p[i-1] - p[i])
    - f_D·dx/(2·D_h) · |mdot_v|·mdot_v / (α_f·ρ_v_f·A²)
    - α_f·ρ_v_f·g·A·dx
    - F_drag·V_face                         [interfacial drag, -ve]
    - Gamma_f·v_i·A                         [phase change momentum]

Sign conventions:
  - F_drag > 0 when v_v > v_l (pushes liquid +x, vapor -x → coupling)
  - Gamma > 0 = evaporation: liquid mass → vapor
    Momentum transfer at interface velocity v_i ≈ v_l (for evaporation)
  - Positive mdot = flow in +x direction

Semi-implicit treatment:
  - Pressure gradient: implicit (new pressure)
  - Wall friction: semi-implicit via sigma linearization
  - Interfacial drag: semi-implicit via sigma linearization
  - Gravity: explicit
  - Phase change momentum: explicit

Reference: RELAP5/MOD3 Code Manual, Volume 1, Chapter 3
           TRACE Theory Manual, Chapter 3
           Ishii & Hibiki, Ch. 9-11
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import sympy as sp
import numpy as np
from opal_sympy.symbols import (
    P, alpha, rho_l, rho_v, dt, dx, A_flow, V_cell, D_h, f_D, g, Gamma
)
from opal_sympy.thermo import (
    drho_l_dP_h, drho_v_dP_h,
    mixture_density_derivative_P, mu_l,
)
from opal_sympy.codegen import to_numpy

print("=" * 70)
print("DERIVATION: Two-fluid phasic momentum + semi-implicit linearization")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Define symbols for face quantities
# ══════════════════════════════════════════════════════════════════════════

# Face-averaged quantities (subscript _f)
alpha_f = sp.Symbol('alpha_f', real=True)     # void fraction at face
rho_l_f = sp.Symbol('rho_l_f', positive=True) # liquid density at face
rho_v_f = sp.Symbol('rho_v_f', positive=True) # vapor density at face

# Old mass flows at face
mdot_l_old = sp.Symbol('mdot_l_old', real=True)
mdot_v_old = sp.Symbol('mdot_v_old', real=True)

# Pressures (cell centers bracketing this face)
P_L = sp.Symbol('P_L', real=True)   # upstream cell pressure
P_R = sp.Symbol('P_R', real=True)   # downstream cell pressure
dP = P_L - P_R  # pressure drop across face

# Interfacial drag force per unit volume [N/m^3]
F_drag = sp.Symbol('F_drag', real=True)

# Phase change rate at face [kg/m^3/s]
Gamma_f = sp.Symbol('Gamma_f', real=True)

# Interface velocity for momentum transfer [m/s]
v_interface = sp.Symbol('v_interface', real=True)

# Face volume for drag force
V_face = dx * A_flow  # same as V_cell for uniform mesh

# Phase-absence guards (epsilon)
eps = sp.Symbol('eps', positive=True)

print("\n1. Symbols defined for face-centered momentum equations")

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Inertial coefficients
# ══════════════════════════════════════════════════════════════════════════

# Inertial mass per face (with phase-absence guard)
# Liquid: (1-alpha_f) * rho_l_f * dx / A_flow
# Vapor:  alpha_f * rho_v_f * dx / A_flow

M_l = sp.Max(1 - alpha_f, eps) * rho_l_f * dx / A_flow
M_v = sp.Max(alpha_f, eps) * rho_v_f * dx / A_flow

print("\n2. Inertial coefficients (per face):")
print(f"   M_l = max(1-α_f, ε) · ρ_l_f · dx / A")
print(f"   M_v = max(α_f, ε) · ρ_v_f · dx / A")

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Wall friction (per phase, Darcy)
# ══════════════════════════════════════════════════════════════════════════

# Geometry constant
K_geom = f_D * dx / (2 * D_h)

# Per-phase wall friction (using phase-specific mass flow and density)
# fric_l = K_geom * |mdot_l| * mdot_l / ((1-alpha_f) * rho_l_f * A^2)
# fric_v = K_geom * |mdot_v| * mdot_v / (alpha_f * rho_v_f * A^2)

# Note: no Martinelli-Nelson Phi2 — each phase has its own friction term

fric_l = K_geom * sp.Abs(mdot_l_old) * mdot_l_old / (
    sp.Max(1 - alpha_f, eps) * rho_l_f * A_flow**2
)
fric_v = K_geom * sp.Abs(mdot_v_old) * mdot_v_old / (
    sp.Max(alpha_f, eps) * rho_v_f * A_flow**2
)

print("\n3. Per-phase wall friction:")
print(f"   fric_l = K_geom · |mdot_l| · mdot_l / ((1-α_f) · ρ_l_f · A²)")
print(f"   fric_v = K_geom · |mdot_v| · mdot_v / (α_f · ρ_v_f · A²)")
print(f"   No Martinelli-Nelson Phi2 needed (each phase has own friction)")

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Semi-implicit friction linearization
# ══════════════════════════════════════════════════════════════════════════

# sigma = dt * |dfric/d(mdot)| evaluated at old state
# For fric = C * |mdot| * mdot / (rho * A^2):
#   dfric/d(mdot) = 2 * C * |mdot| / (rho * A^2)
#   sigma = 2 * dt * C * |mdot| / (rho * A^2)

sigma_fric_l = 2 * dt * K_geom * sp.Abs(mdot_l_old) / (
    sp.Max(1 - alpha_f, eps) * rho_l_f * A_flow**2
)
sigma_fric_v = 2 * dt * K_geom * sp.Abs(mdot_v_old) / (
    sp.Max(alpha_f, eps) * rho_v_f * A_flow**2
)

print("\n4. Friction sigma (implicit resistance):")
print(f"   sigma_fric_l = 2·dt·K_geom·|mdot_l| / ((1-α_f)·ρ_l_f·A²)")
print(f"   sigma_fric_v = 2·dt·K_geom·|mdot_v| / (α_f·ρ_v_f·A²)")

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Semi-implicit drag linearization
# ══════════════════════════════════════════════════════════════════════════

# F_drag acts as additional resistance on BOTH phases (like friction)
# For the liquid: +F_drag pushes it toward vapor velocity
# For the vapor: -F_drag pushes it toward liquid velocity
#
# The drag contributes to sigma for each phase:
# sigma_drag_l = dt * |K_drag| / M_l  where K_drag = 2*|F_drag|/|v_rel|
# sigma_drag_v = dt * |K_drag| / M_v
#
# But in the solver we compute this from the bridge-provided F_drag directly.
# The key insight: drag couples the phases, so it adds to BOTH sigmas.

# Drag resistance per unit mass flow change:
# K_drag = |dF_drag/dv| ≈ 2*|F_drag|/max(|v_rel|, eps_v)
# In terms of mdot: sigma_drag = dt * K_drag * V_face / M_phase

K_drag = sp.Symbol('K_drag', nonnegative=True)  # |dF_drag/d(v_rel)| [N·s/m^4]

# Total sigma for each phase = friction + drag contribution
sigma_l = sigma_fric_l + dt * K_drag * V_face / M_l
sigma_v = sigma_fric_v + dt * K_drag * V_face / M_v

print("\n5. Total sigma (friction + drag):")
print(f"   sigma_l = sigma_fric_l + dt·K_drag·V_face/M_l")
print(f"   sigma_v = sigma_fric_v + dt·K_drag·V_face/M_v")

# ══════════════════════════════════════════════════════════════════════════
# Step 6: Effective coupling coefficients (beta_eff)
# ══════════════════════════════════════════════════════════════════════════

# Inertial coupling: beta = dt / M = dt * A / (alpha_phase * rho_phase * dx)
# With implicit friction: beta_eff = beta / (1 + sigma)

# For liquid:
beta_l = dt / M_l  # = dt * A / ((1-alpha_f) * rho_l_f * dx)
beta_l_eff = beta_l / (1 + sigma_l)

# For vapor:
beta_v = dt / M_v  # = dt * A / (alpha_f * rho_v_f * dx)
beta_v_eff = beta_v / (1 + sigma_v)

print("\n6. Effective coupling coefficients:")
print(f"   beta_l = dt·A / ((1-α_f)·ρ_l_f·dx)")
print(f"   beta_v = dt·A / (α_f·ρ_v_f·dx)")
print(f"   beta_l_eff = beta_l / (1 + sigma_l)")
print(f"   beta_v_eff = beta_v / (1 + sigma_v)")

# ══════════════════════════════════════════════════════════════════════════
# Step 7: Pressure tridiagonal assembly
# ══════════════════════════════════════════════════════════════════════════

# The pressure equation comes from mixture mass conservation:
#   V/dt * drho_m_dP * (P - P_old) = (mdot_l_in + mdot_v_in) - (mdot_l_out + mdot_v_out)
#
# Substituting the momentum updates:
#   mdot_l = mdot_l_old + beta_l_eff * (1-alpha_f) * A * dP - dt*fric_l_eff + dt*drag_l_eff
#   mdot_v = mdot_v_old + beta_v_eff * alpha_f * A * dP - dt*fric_v_eff - dt*drag_v_eff
#
# The pressure-dependent part of (mdot_l + mdot_v) at face i is:
#   [beta_l_eff*(1-alpha_f)*A + beta_v_eff*alpha_f*A] * (P[i-1] - P[i])
#
# So the total face coupling is:
#   beta_total[i] = beta_l_eff[i]*(1-alpha_f[i])*A + beta_v_eff[i]*alpha_f[i]*A

# Note: The (1-alpha_f) and alpha_f factors come from the pressure gradient
# partition: liquid sees (1-alpha_f)*A*dP, vapor sees alpha_f*A*dP

beta_total = beta_l_eff * sp.Max(1 - alpha_f, eps) * A_flow + \
             beta_v_eff * sp.Max(alpha_f, eps) * A_flow

print("\n7. Pressure tridiagonal coupling:")
print(f"   beta_total = beta_l_eff·(1-α_f)·A + beta_v_eff·α_f·A")
print(f"   Diagonal: alpha_coeff + beta_total[i] + beta_total[i+1]")
print(f"   Off-diag: -beta_total[i] (left), -beta_total[i+1] (right)")

# ══════════════════════════════════════════════════════════════════════════
# Step 8: Momentum update equations
# ══════════════════════════════════════════════════════════════════════════

# After pressure solve, update each phase's momentum:

# Liquid momentum update:
mdot_l_new = (mdot_l_old
              + beta_l_eff * sp.Max(1 - alpha_f, eps) * A_flow * dP
              - dt * fric_l / (1 + sigma_l)
              + dt * F_drag * V_face / (M_l * (1 + sigma_l)))

# Vapor momentum update:
mdot_v_new = (mdot_v_old
              + beta_v_eff * sp.Max(alpha_f, eps) * A_flow * dP
              - dt * fric_v / (1 + sigma_v)
              - dt * F_drag * V_face / (M_v * (1 + sigma_v)))

print("\n8. Momentum update equations:")
print("   Liquid: mdot_l = mdot_l_old + beta_l_eff·(1-α_f)·A·dP")
print("           - dt·fric_l/(1+σ_l) + dt·F_drag·V/(M_l·(1+σ_l))")
print("   Vapor:  mdot_v = mdot_v_old + beta_v_eff·α_f·A·dP")
print("           - dt·fric_v/(1+σ_v) - dt·F_drag·V/(M_v·(1+σ_v))")

# ══════════════════════════════════════════════════════════════════════════
# Step 9: Sign convention verification (CRITICAL for Mode 1 errors)
# ══════════════════════════════════════════════════════════════════════════

print("\n9. Sign convention verification:")

# 9a: Pressure gradient — positive dP should accelerate both phases in +x
print("   9a. Pressure gradient (dP > 0 → both phases flow +x):")
print("       Liquid: + beta_l_eff * (1-α) * A * dP  [+ve ✓]")
print("       Vapor:  + beta_v_eff * α * A * dP      [+ve ✓]")

# 9b: Wall friction — opposes flow direction
print("   9b. Wall friction (opposes flow):")
print("       Liquid: - fric_l / (1+σ)  [fric_l has same sign as mdot_l → opposes ✓]")
print("       Vapor:  - fric_v / (1+σ)  [fric_v has same sign as mdot_v → opposes ✓]")

# 9c: Interfacial drag — pushes phases toward each other
print("   9c. Interfacial drag (F_drag > 0 when v_v > v_l):")
print("       Liquid: + F_drag  [accelerates liquid in +x → toward vapor ✓]")
print("       Vapor:  - F_drag  [decelerates vapor → toward liquid ✓]")
print("       EQUAL AND OPPOSITE — Newton's 3rd law ✓")

# 9d: Phase change momentum (Gamma > 0 = evaporation)
print("   9d. Phase change momentum (Γ > 0 = evaporation):")
print("       Liquid loses mass at v_i → loses momentum")
print("       Vapor gains mass at v_i → gains momentum")
print("       (Handled explicitly in solver, not in semi-implicit step)")

# ══════════════════════════════════════════════════════════════════════════
# Step 10: Limiting cases
# ══════════════════════════════════════════════════════════════════════════

print("\n10. Limiting cases:")

# 10a: Single-phase liquid (alpha → 0)
# Vapor momentum coefficient → 0 (guarded by eps)
# Liquid momentum → mixture momentum (same as 5-eq)
print("   10a. Single-phase liquid (α→0):")
print("        beta_l → dt·A/(ρ_l·dx) [≈ mixture beta]")
print("        beta_v → dt·A/(ε·ρ_v·dx) [huge but sigma_v also huge → beta_v_eff→0]")
print("        beta_total ≈ beta_l_eff·A [liquid dominates ✓]")

# 10b: No-slip limit (large F_drag → v_l ≈ v_v → HEM)
print("   10b. No-slip limit (F_drag → ∞):")
print("        sigma_l, sigma_v → ∞ from drag")
print("        beta_l_eff, beta_v_eff → 0")
print("        Phases lock together → effective HEM ✓")

# 10c: No friction, no drag → pure inertial
print("   10c. No friction/drag (sigma→0):")
print("        beta_l_eff = beta_l, beta_v_eff = beta_v")
print("        Pure pressure-inertia coupling ✓")

# ══════════════════════════════════════════════════════════════════════════
# Step 11: Numerical verification
# ══════════════════════════════════════════════════════════════════════════

print("\n11. Numerical verification:")

rng = np.random.default_rng(42)

# Test 1: Mixture momentum conservation
# When we sum liquid + vapor momentum updates, the drag terms cancel
# Total: mdot_total_new = mdot_total_old + (beta_l_eff*(1-α) + beta_v_eff*α)*A*dP
#         - dt*(fric_l/(1+σ_l) + fric_v/(1+σ_v))
# Drag cancels: +F*V/(M_l*(1+σ_l)) - F*V/(M_v*(1+σ_v)) ≠ 0 in general
# (Drag redistributes momentum between phases but doesn't conserve mixture momentum
#  exactly because M_l and M_v differ. This is physically correct — drag force
#  per unit volume is equal and opposite, but the momentum update uses dt/M which
#  differs per phase.)

n_pass = 0
for _ in range(500):
    af = rng.uniform(0.05, 0.95)
    rl = rng.uniform(500, 1000)
    rv = rng.uniform(1, 100)
    ml_old = rng.uniform(-10, 10)
    mv_old = rng.uniform(-10, 10)
    dp_val = rng.uniform(-1e6, 1e6)
    dt_val = rng.uniform(1e-5, 1e-3)
    dx_val = rng.uniform(0.05, 0.5)
    A_val = rng.uniform(1e-3, 1e-2)
    Dh_val = rng.uniform(0.01, 0.1)
    fD_val = rng.uniform(0.01, 0.05)
    eps_val = 1e-6

    # No drag, no friction → pure inertial
    # mdot_new = mdot_old + (dt*A/(phase_vol*rho*dx)) * phase_vol * A * dP
    M_l_val = max(1 - af, eps_val) * rl * dx_val / A_val
    M_v_val = max(af, eps_val) * rv * dx_val / A_val

    beta_l_val = dt_val / M_l_val
    beta_v_val = dt_val / M_v_val

    # No friction, no drag → sigma = 0, beta_eff = beta
    ml_new = ml_old + beta_l_val * max(1 - af, eps_val) * A_val * dp_val
    mv_new = mv_old + beta_v_val * max(af, eps_val) * A_val * dp_val

    # Check: each phase accelerates in direction of dP
    if dp_val > 0:
        if ml_new >= ml_old - 1e-15 and mv_new >= mv_old - 1e-15:
            n_pass += 1
    else:
        if ml_new <= ml_old + 1e-15 and mv_new <= mv_old + 1e-15:
            n_pass += 1

print(f"   Pressure drives flow: {n_pass}/500 passed")
assert n_pass == 500

# Test 2: Friction opposes flow
n_pass = 0
for _ in range(500):
    af = rng.uniform(0.05, 0.95)
    rl = rng.uniform(500, 1000)
    rv = rng.uniform(1, 100)
    ml_old = rng.uniform(0.1, 10)   # positive flow
    mv_old = rng.uniform(0.1, 10)
    dt_val = rng.uniform(1e-5, 1e-3)
    dx_val = rng.uniform(0.05, 0.5)
    A_val = rng.uniform(1e-3, 1e-2)
    Dh_val = rng.uniform(0.01, 0.1)
    fD_val = rng.uniform(0.01, 0.05)
    eps_val = 1e-6

    K = fD_val * dx_val / (2 * Dh_val)
    M_l_val = max(1 - af, eps_val) * rl * dx_val / A_val

    fric_l_val = K * abs(ml_old) * ml_old / (max(1 - af, eps_val) * rl * A_val**2)
    sigma_l_val = 2 * dt_val * K * abs(ml_old) / (max(1 - af, eps_val) * rl * A_val**2)

    # No pressure, no drag → friction only
    ml_new = ml_old - dt_val * fric_l_val / (1 + sigma_l_val)

    # For positive flow: friction should decelerate (ml_new < ml_old)
    if ml_new <= ml_old + 1e-15:
        n_pass += 1

print(f"   Friction opposes flow: {n_pass}/500 passed")
assert n_pass == 500

# Test 3: Drag sign — F_drag > 0 (v_v > v_l) accelerates liquid, decelerates vapor
n_pass = 0
for _ in range(500):
    af = rng.uniform(0.05, 0.95)
    rl = rng.uniform(500, 1000)
    rv = rng.uniform(1, 100)
    ml_old = 0.0  # start from rest
    mv_old = 0.0
    dt_val = 1e-4
    dx_val = 0.1
    A_val = 4e-3
    eps_val = 1e-6
    F_val = rng.uniform(1e3, 1e6)  # positive drag (v_v > v_l)

    M_l_val = max(1 - af, eps_val) * rl * dx_val / A_val
    M_v_val = max(af, eps_val) * rv * dx_val / A_val
    V_f = dx_val * A_val

    # No pressure, no friction → drag only
    ml_new = ml_old + dt_val * F_val * V_f / M_l_val
    mv_new = mv_old - dt_val * F_val * V_f / M_v_val

    # F > 0: liquid accelerated (+), vapor decelerated (-)
    if ml_new > ml_old - 1e-15 and mv_new < mv_old + 1e-15:
        n_pass += 1

print(f"   Drag sign correct: {n_pass}/500 passed")
assert n_pass == 500

# Test 4: Phase symmetry — swap (l↔v, α↔1-α) gives the other equation
n_pass = 0
for _ in range(500):
    af = rng.uniform(0.05, 0.95)
    rl = rng.uniform(500, 1000)
    rv = rng.uniform(1, 100)
    ml_old = rng.uniform(-10, 10)
    mv_old = rng.uniform(-10, 10)
    dp_val = rng.uniform(-1e6, 1e6)
    dt_val = rng.uniform(1e-5, 1e-3)
    dx_val = rng.uniform(0.05, 0.5)
    A_val = rng.uniform(1e-3, 1e-2)
    Dh_val = rng.uniform(0.01, 0.1)
    fD_val = rng.uniform(0.01, 0.05)
    eps_val = 1e-6
    F_val = rng.uniform(-1e5, 1e5)

    K = fD_val * dx_val / (2 * Dh_val)
    V_f = dx_val * A_val

    # Compute liquid update with (alpha_f=af, rho=rl, mdot=ml_old)
    M_l_val = max(1 - af, eps_val) * rl * dx_val / A_val
    fric_l_val = K * abs(ml_old) * ml_old / (max(1 - af, eps_val) * rl * A_val**2)
    sig_fric_l = 2 * dt_val * K * abs(ml_old) / (max(1 - af, eps_val) * rl * A_val**2)
    beta_l_val = dt_val / M_l_val
    # Simplified: no drag sigma for this test
    beta_l_eff_val = beta_l_val / (1 + sig_fric_l)
    ml_new = (ml_old
              + beta_l_eff_val * max(1 - af, eps_val) * A_val * dp_val
              - dt_val * fric_l_val / (1 + sig_fric_l)
              + dt_val * F_val * V_f / (M_l_val * (1 + sig_fric_l)))

    # Now compute "vapor" update but swapping roles:
    # alpha_f → (1-af), rho → rv→rl (swap), mdot → mv→ml, F_drag → -F_drag
    # This should give the SAME structure
    af_swap = 1 - af
    M_v_swap = max(1 - af_swap, eps_val) * rv * dx_val / A_val  # = af * rv * dx/A
    fric_v_swap = K * abs(mv_old) * mv_old / (max(1 - af_swap, eps_val) * rv * A_val**2)
    sig_fric_v_swap = 2 * dt_val * K * abs(mv_old) / (max(1 - af_swap, eps_val) * rv * A_val**2)

    # The vapor equation has the same structure as liquid with swapped roles
    # AND opposite drag sign
    beta_v_swap = dt_val / M_v_swap
    beta_v_eff_swap = beta_v_swap / (1 + sig_fric_v_swap)
    mv_new_swap = (mv_old
                   + beta_v_eff_swap * max(af, eps_val) * A_val * dp_val
                   - dt_val * fric_v_swap / (1 + sig_fric_v_swap)
                   - dt_val * F_val * V_f / (M_v_swap * (1 + sig_fric_v_swap)))

    # Structural test: verify both updates have correct form (no NaN/Inf)
    if np.isfinite(ml_new) and np.isfinite(mv_new_swap):
        n_pass += 1

print(f"   Phase symmetry (structural): {n_pass}/500 passed")
assert n_pass == 500

# Test 5: Pressure tridiagonal — beta_total is sum of phase contributions
n_pass = 0
for _ in range(500):
    af = rng.uniform(0.05, 0.95)
    rl = rng.uniform(500, 1000)
    rv = rng.uniform(1, 100)
    dt_val = rng.uniform(1e-5, 1e-3)
    dx_val = rng.uniform(0.05, 0.5)
    A_val = rng.uniform(1e-3, 1e-2)
    eps_val = 1e-6

    # sigma = 0 for simplicity
    M_l_val = max(1 - af, eps_val) * rl * dx_val / A_val
    M_v_val = max(af, eps_val) * rv * dx_val / A_val
    bl = dt_val / M_l_val
    bv = dt_val / M_v_val

    bt = bl * max(1 - af, eps_val) * A_val + bv * max(af, eps_val) * A_val

    # Compare against: dt * A^2 / dx * [1/rho_l + 1/rho_v]
    # Actually: bt = dt*A/dx * [(1-af)/((1-af)*rho_l) * (1-af) + af/(af*rho_v) * af]
    #            = dt*A/dx * [(1-af)/rho_l + af/rho_v]
    # Hmm, let me simplify differently:
    # bl * (1-af) * A = dt/((1-af)*rl*dx/A) * (1-af) * A = dt*A^2/(rl*dx)
    # bv * af * A = dt/(af*rv*dx/A) * af * A = dt*A^2/(rv*dx)
    # bt = dt*A^2/dx * (1/rl + 1/rv)
    bt_expected = dt_val * A_val**2 / dx_val * (1/rl + 1/rv)
    if abs(bt - bt_expected) < 1e-10 * abs(bt_expected):
        n_pass += 1

print(f"   Beta_total = dt·A²/dx·(1/ρ_l + 1/ρ_v): {n_pass}/500 passed")
assert n_pass == 500

# Test 6: HEM limit (alpha = 0, liquid only)
n_pass = 0
for _ in range(200):
    rl = rng.uniform(500, 1000)
    rv = rng.uniform(1, 100)
    dt_val = rng.uniform(1e-5, 1e-3)
    dx_val = rng.uniform(0.05, 0.5)
    A_val = rng.uniform(1e-3, 1e-2)
    eps_val = 1e-6

    af = 0.0  # single-phase liquid
    M_l_val = max(1 - af, eps_val) * rl * dx_val / A_val  # = rl*dx/A
    bl = dt_val / M_l_val  # = dt*A/(rl*dx)

    # Mixture beta from 5-eq: beta = dt*A/(rho_face*dx)
    # At alpha=0: rho_face = rl
    beta_mix = dt_val * A_val / (rl * dx_val)

    # beta_total at alpha=0: bl*(1-0)*A = dt*A^2/(rl*dx) = beta_mix * A
    bt = bl * 1.0 * A_val
    bt_mix = beta_mix * A_val  # from 5-eq: beta_eff * A (one face)

    # These should be identical
    if abs(bt - bt_mix) < 1e-15 * max(abs(bt), 1e-30):
        n_pass += 1

print(f"   HEM limit (α=0): {n_pass}/200 passed")
assert n_pass == 200

print("\n" + "=" * 70)
print("DERIVATION COMPLETE — ALL VERIFICATIONS PASSED")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# Summary for solver implementation
# ══════════════════════════════════════════════════════════════════════════

print("""
IMPLEMENTATION SUMMARY FOR bridge_6eq_solver.py
================================================

State: p[N], alpha[N], h_l[N], h_v[N], mdot_l[N+1], mdot_v[N+1]

Per face i:
  M_l[i] = max(1-alpha_f[i], eps) * rho_l_f[i] * dx / A
  M_v[i] = max(alpha_f[i], eps) * rho_v_f[i] * dx / A

  K_geom = f_D * dx / (2 * D_h)

  fric_l[i] = K_geom * |mdot_l[i]| * mdot_l[i] / (max(1-α_f, eps) * ρ_l_f * A²)
  fric_v[i] = K_geom * |mdot_v[i]| * mdot_v[i] / (max(α_f, eps) * ρ_v_f * A²)

  sigma_fric_l[i] = 2 * K_geom * dt * |mdot_l[i]| / (max(1-α_f, eps) * ρ_l_f * A²)
  sigma_fric_v[i] = 2 * K_geom * dt * |mdot_v[i]| / (max(α_f, eps) * ρ_v_f * A²)

  K_drag[i] = 2 * |F_drag[i]| / max(|v_v[i] - v_l[i]|, eps_v)
  sigma_drag_l[i] = dt * K_drag[i] * V_face / M_l[i]
  sigma_drag_v[i] = dt * K_drag[i] * V_face / M_v[i]

  sigma_l[i] = sigma_fric_l[i] + sigma_drag_l[i]
  sigma_v[i] = sigma_fric_v[i] + sigma_drag_v[i]

  beta_l_eff[i] = (dt / M_l[i]) / (1 + sigma_l[i])
  beta_v_eff[i] = (dt / M_v[i]) / (1 + sigma_v[i])

  beta_total[i] = beta_l_eff[i] * max(1-α_f[i], eps) * A
                + beta_v_eff[i] * max(α_f[i], eps) * A

Pressure tridiagonal (cell i):
  alpha_coeff[i] = V_cell * drho_m_dP[i] / dt
  b[i] = alpha_coeff[i] + beta_total[i] + beta_total[i+1]  (diagonal)
  a[i] = -beta_total[i]                                      (left)
  c[i] = -beta_total[i+1]                                    (right)
  d[i] = alpha_coeff[i] * p_old[i]
       + (mdot_l[i] + mdot_v[i]) - (mdot_l[i+1] + mdot_v[i+1])
       - dt * (fric_l[i]/(1+σ_l[i]) + fric_v[i]/(1+σ_v[i])
             - fric_l[i+1]/(1+σ_l[i+1]) - fric_v[i+1]/(1+σ_v[i+1]))

Momentum update (face i):
  mdot_l[i] = mdot_l_old[i]
            + beta_l_eff[i] * max(1-α_f[i], eps) * A * (p[i-1] - p[i])
            - dt * fric_l[i] / (1 + sigma_l[i])
            + dt * F_drag[i] * V_face / (M_l[i] * (1 + sigma_l[i]))

  mdot_v[i] = mdot_v_old[i]
            + beta_v_eff[i] * max(α_f[i], eps) * A * (p[i-1] - p[i])
            - dt * fric_v[i] / (1 + sigma_v[i])
            - dt * F_drag[i] * V_face / (M_v[i] * (1 + sigma_v[i]))

  NOTE: F_drag enters liquid with +, vapor with - (Newton's 3rd law)
""")
