#!/usr/bin/env python3
"""
Derivation: Coupled phasic momentum solve for the 6-equation two-fluid model

RELAP5-style 2x2 block coupling per face. Instead of independent per-phase
momentum updates with drag-through-sigma, solve a coupled 2x2 system that
includes explicit drag correction stabilized by diagonal drag sigma.

At each interior face i, the semi-implicit phasic momentum with linearized
friction and drag gives:

  a_ll * Delta_l - sigma_drag_v * Delta_v = R_l
 -sigma_drag_l * Delta_l + a_vv * Delta_v = R_v

where:
  Delta_k = mdot_k_new - mdot_k_old
  a_ll = 1 + sigma_fric_l + sigma_drag_l  (+ phase-absence boost)
  a_vv = 1 + sigma_fric_v + sigma_drag_v  (+ phase-absence boost)
  sigma_drag_l = dt * K_drag * dx / (alpha_l * rho_l)
  sigma_drag_v = dt * K_drag * dx / (alpha_v * rho_v)
  R_l = beta*(1-alpha)*dp - dt*fric_l + dt*F_drag_old*V_face
  R_v = beta*alpha*dp - dt*fric_v - dt*F_drag_old*V_face

Solve via Cramer's rule (no iteration):
  det = a_ll*a_vv - sigma_drag_l*sigma_drag_v
  Delta_l = (R_l*a_vv + sigma_drag_v*R_v) / det
  Delta_v = (a_ll*R_v + sigma_drag_l*R_l) / det

Pressure tridiagonal coupling:
  beta_total = d(Delta_l + Delta_v)/d(dp)
             = beta * [(1-alpha)*a_vv + alpha*sigma_drag_v
                      + alpha*a_ll + (1-alpha)*sigma_drag_l] / det

Sign convention:
  F_drag > 0 when v_v > v_l (accelerates liquid, decelerates vapor)
  +F_drag*V in liquid RHS, -F_drag*V in vapor RHS

Reference: RELAP5/MOD3 Code Manual, Volume II, Section 2.2
           Current OPAL: solver/partitioner/bridge_6eq_solver.py
           Derivation: docs/math/derivations/two_fluid_momentum.py
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

print("=" * 70)
print("DERIVATION: Coupled phasic momentum (2x2 block solve per face)")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Define the semi-implicit framework
# ══════════════════════════════════════════════════════════════════════════

print("\n1. Semi-implicit framework (geometric beta):")
print("   beta = dt * A / dx  (geometric, removes acoustic CFL)")
print("   sigma_fric_k = dt * d(fric_k)/d(mdot_k)")
print("   beta_eff_k = beta / (1 + sigma_k)")

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Drag linearization
# ══════════════════════════════════════════════════════════════════════════

print("\n2. Drag linearization:")
print("   F_drag(v_rel_new) ~ F_drag_old + K_drag * (v_rel_new - v_rel_old)")
print("   K_drag = 2 * |F_drag_old| / max(|v_rel_old|, eps)")
print("")
print("   v_rel = v_v - v_l = mdot_v/(alpha_v*rho_v*A) - mdot_l/(alpha_l*rho_l*A)")
print("   dv_rel/d(mdot_l) = -C_l = -1/(alpha_l*rho_l*A)")
print("   dv_rel/d(mdot_v) = +C_v = +1/(alpha_v*rho_v*A)")

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Per-phase drag sigma
# ══════════════════════════════════════════════════════════════════════════

print("\n3. Per-phase drag sigma:")
print("   Drag force on liquid: +F_drag * V_face")
print("   d(F_drag*V)/d(mdot_l) = K_drag * (-C_l) * V = -K_drag * dx / (alpha_l * rho_l)")
print("   sigma_drag_l = dt * |d(F_drag*V)/d(mdot_l)| = dt * K_drag * dx / (alpha_l * rho_l)")
print("   sigma_drag_v = dt * K_drag * dx / (alpha_v * rho_v)")
print("")
print("   Cross-coupling: d(F_drag*V)/d(mdot_v) = K_drag * C_v * V = K_drag * dx / (alpha_v * rho_v)")
print("   => sigma_cross_lv = sigma_drag_v  (drag on liquid from vapor velocity change)")
print("   => sigma_cross_vl = sigma_drag_l  (drag on vapor from liquid velocity change)")

# ══════════════════════════════════════════════════════════════════════════
# Step 4: 2x2 matrix assembly
# ══════════════════════════════════════════════════════════════════════════

print("\n4. 2x2 block system per face:")
print("   [a_ll   -sigma_drag_v] [Delta_l]   [R_l]")
print("   [-sigma_drag_l   a_vv] [Delta_v] = [R_v]")
print("")
print("   a_ll = 1 + sigma_fric_l + sigma_drag_l  (+ absence boost)")
print("   a_vv = 1 + sigma_fric_v + sigma_drag_v  (+ absence boost)")
print("   R_l = beta*(1-alpha)*dp - dt*fric_l + dt*F_drag_old*V_face")
print("   R_v = beta*alpha*dp - dt*fric_v - dt*F_drag_old*V_face")

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Cramer's rule solution
# ══════════════════════════════════════════════════════════════════════════

print("\n5. Cramer's rule (2x2 solve):")
print("   det = a_ll * a_vv - sigma_drag_l * sigma_drag_v")
print("   Delta_l = (R_l * a_vv + sigma_drag_v * R_v) / det")
print("   Delta_v = (a_ll * R_v + sigma_drag_l * R_l) / det")

# Expand determinant to show it's always positive
print("\n   Determinant expansion:")
print("   det = (1+sf_l+sd_l)(1+sf_v+sd_v) - sd_l*sd_v")
print("       = (1+sf_l)(1+sf_v) + (1+sf_l)*sd_v + sd_l*(1+sf_v)")
print("   Note: sd_l*sd_v terms cancel! det is sum of positive terms => det > 0 always")

# ══════════════════════════════════════════════════════════════════════════
# Step 6: beta_total for pressure tridiagonal
# ══════════════════════════════════════════════════════════════════════════

print("\n6. Pressure tridiagonal coupling:")
print("   d(R_l)/d(dp) = beta*(1-alpha)")
print("   d(R_v)/d(dp) = beta*alpha")
print("")
print("   d(Delta_l)/d(dp) = [beta*(1-alpha)*a_vv + beta*alpha*sigma_drag_v] / det")
print("   d(Delta_v)/d(dp) = [beta*alpha*a_ll + beta*(1-alpha)*sigma_drag_l] / det")
print("")
print("   beta_total = d(Delta_l + Delta_v)/d(dp)")
print("             = beta * [(1-alpha)*(a_vv + sigma_drag_l)")
print("                     + alpha*(a_ll + sigma_drag_v)] / det")

# ══════════════════════════════════════════════════════════════════════════
# Step 7: Explicit correction for pressure tridiagonal RHS
# ══════════════════════════════════════════════════════════════════════════

print("\n7. Explicit correction (dp=0 contribution):")
print("   R_l_0 = -dt*fric_l + dt*F_drag*V")
print("   R_v_0 = -dt*fric_v - dt*F_drag*V")
print("   corr = Delta_l(dp=0) + Delta_v(dp=0)")
print("        = [(a_vv+sd_l)*R_l_0 + (a_ll+sd_v)*R_v_0] / det")

# ══════════════════════════════════════════════════════════════════════════
# Step 8: Sign convention verification
# ══════════════════════════════════════════════════════════════════════════

print("\n8. Sign convention verification:")
print("   8a. Pressure gradient (dp > 0, p_L > p_R):")
print("       R_l contribution: +beta*(1-alpha)*dp  [+ve, liquid flows +x] ✓")
print("       R_v contribution: +beta*alpha*dp      [+ve, vapor flows +x] ✓")
print("   8b. Wall friction (opposes flow):")
print("       R_l contribution: -dt*fric_l  [opposes liquid flow] ✓")
print("       R_v contribution: -dt*fric_v  [opposes vapor flow] ✓")
print("   8c. Interfacial drag (F_drag > 0 when v_v > v_l):")
print("       R_l contribution: +dt*F_drag*V  [accelerates liquid toward vapor] ✓")
print("       R_v contribution: -dt*F_drag*V  [decelerates vapor toward liquid] ✓")
print("       Sum R_l+R_v drag terms: +dt*FV - dt*FV = 0  [Newton's 3rd law] ✓")

# ══════════════════════════════════════════════════════════════════════════
# Step 9: Numerical verification
# ══════════════════════════════════════════════════════════════════════════

print("\n9. Numerical verification:")

rng = np.random.default_rng(42)


def coupled_solve(alpha_l, alpha_v, rho_l, rho_v, mdot_l_old, mdot_v_old,
                  dp, fric_l, fric_v, F_drag, K_drag, V_face, dt, beta, dx):
    """Solve the 2x2 coupled system. Returns (mdot_l_new, mdot_v_new)."""
    EPS = 1e-6
    A_flow = V_face / dx  # recover A from V=dx*A

    alpha_l_f = max(alpha_l, EPS)
    alpha_v_f = max(alpha_v, EPS)
    rho_l_f = max(rho_l, 0.01)
    rho_v_f = max(rho_v, 0.01)

    # Friction sigma
    K_geom = 0.175  # f_D*dx/(2*D_h), typical
    sigma_fric_l = 2 * dt * K_geom * abs(mdot_l_old) / (alpha_l_f * rho_l_f * A_flow**2)
    sigma_fric_v = 2 * dt * K_geom * abs(mdot_v_old) / (alpha_v_f * rho_v_f * A_flow**2)

    # Drag sigma
    sigma_drag_l = dt * K_drag * dx / (alpha_l_f * rho_l_f)
    sigma_drag_v = dt * K_drag * dx / (alpha_v_f * rho_v_f)

    # Matrix
    a_ll = 1 + sigma_fric_l + sigma_drag_l
    a_vv = 1 + sigma_fric_v + sigma_drag_v
    det = a_ll * a_vv - sigma_drag_l * sigma_drag_v

    # RHS
    R_l = beta * alpha_l_f * dp - dt * fric_l + dt * F_drag * V_face
    R_v = beta * alpha_v_f * dp - dt * fric_v - dt * F_drag * V_face

    # Cramer
    Delta_l = (R_l * a_vv + sigma_drag_v * R_v) / det
    Delta_v = (a_ll * R_v + sigma_drag_l * R_l) / det

    return mdot_l_old + Delta_l, mdot_v_old + Delta_v, det, a_ll, a_vv


def uncoupled_solve(alpha_l, alpha_v, rho_l, rho_v, mdot_l_old, mdot_v_old,
                    dp, fric_l, fric_v, dt, beta):
    """Current uncoupled scheme (no drag correction). For comparison."""
    EPS = 1e-6
    alpha_l_f = max(alpha_l, EPS)
    alpha_v_f = max(alpha_v, EPS)
    rho_l_f = max(rho_l, 0.01)
    rho_v_f = max(rho_v, 0.01)

    K_geom = 0.175
    A_flow = 7.854e-5  # hardcoded for test
    sigma_fric_l = 2 * dt * K_geom * abs(mdot_l_old) / (alpha_l_f * rho_l_f * A_flow**2)
    sigma_fric_v = 2 * dt * K_geom * abs(mdot_v_old) / (alpha_v_f * rho_v_f * A_flow**2)

    beta_l_eff = beta / (1 + sigma_fric_l)
    beta_v_eff = beta / (1 + sigma_fric_v)

    mdot_l_new = mdot_l_old + beta_l_eff * alpha_l_f * dp - dt * fric_l / (1 + sigma_fric_l)
    mdot_v_new = mdot_v_old + beta_v_eff * alpha_v_f * dp - dt * fric_v / (1 + sigma_fric_v)
    return mdot_l_new, mdot_v_new


# --- Test 1: K_drag = 0 recovers uncoupled scheme ---
n_pass = 0
for _ in range(500):
    alpha_v = rng.uniform(0.05, 0.95)
    alpha_l = 1 - alpha_v
    rho_l = rng.uniform(500, 1000)
    rho_v = rng.uniform(10, 100)
    mdot_l = rng.uniform(-2, 2)
    mdot_v = rng.uniform(-2, 2)
    dp = rng.uniform(-1e6, 1e6)
    fric_l = rng.uniform(0, 1e5) * np.sign(mdot_l)
    fric_v = rng.uniform(0, 1e4) * np.sign(mdot_v)
    dt_val = 5e-5
    dx_val = 0.175
    A_flow = 7.854e-5
    beta_val = dt_val * A_flow / dx_val
    V_face = dx_val * A_flow

    # Coupled with K_drag = 0
    ml_c, mv_c, _, _, _ = coupled_solve(
        alpha_l, alpha_v, rho_l, rho_v, mdot_l, mdot_v,
        dp, fric_l, fric_v, F_drag=0.0, K_drag=0.0,
        V_face=V_face, dt=dt_val, beta=beta_val, dx=dx_val)

    # Uncoupled
    ml_u, mv_u = uncoupled_solve(
        alpha_l, alpha_v, rho_l, rho_v, mdot_l, mdot_v,
        dp, fric_l, fric_v, dt=dt_val, beta=beta_val)

    err_l = abs(ml_c - ml_u) / max(abs(ml_u), 1e-20)
    err_v = abs(mv_c - mv_u) / max(abs(mv_u), 1e-20)
    if err_l < 1e-10 and err_v < 1e-10:
        n_pass += 1

print(f"   Test 1 (K_drag=0 recovery): {n_pass}/500 passed")
assert n_pass == 500, f"K_drag=0 recovery failed: {n_pass}/500"


# --- Test 2: K_drag -> inf recovers mixture momentum ---
n_pass = 0
for _ in range(500):
    alpha_v = rng.uniform(0.1, 0.9)
    alpha_l = 1 - alpha_v
    rho_l = rng.uniform(500, 1000)
    rho_v = rng.uniform(10, 100)
    mdot_l = rng.uniform(-2, 2)
    mdot_v = rng.uniform(-2, 2)
    dp = rng.uniform(-1e6, 1e6)
    fric_l = rng.uniform(0, 1e5) * np.sign(mdot_l) if abs(mdot_l) > 0.01 else 0.0
    fric_v = rng.uniform(0, 1e4) * np.sign(mdot_v) if abs(mdot_v) > 0.01 else 0.0
    dt_val = 5e-5
    dx_val = 0.175
    A_flow = 7.854e-5
    beta_val = dt_val * A_flow / dx_val
    V_face = dx_val * A_flow

    # Coupled with very large K_drag
    K_drag_inf = 1e15
    ml_c, mv_c, det_val, _, _ = coupled_solve(
        alpha_l, alpha_v, rho_l, rho_v, mdot_l, mdot_v,
        dp, fric_l, fric_v, F_drag=0.0, K_drag=K_drag_inf,
        V_face=V_face, dt=dt_val, beta=beta_val, dx=dx_val)

    # At infinite drag, phases lock together: Delta_v/Delta_l = sigma_drag_v/sigma_drag_l
    # Which means velocity changes match: the relative velocity change -> 0
    # Check: v_rel change should be near zero
    v_l_old = mdot_l / (alpha_l * rho_l * A_flow)
    v_v_old = mdot_v / (alpha_v * rho_v * A_flow)
    v_l_new = ml_c / (alpha_l * rho_l * A_flow)
    v_v_new = mv_c / (alpha_v * rho_v * A_flow)

    dv_rel = (v_v_new - v_l_new) - (v_v_old - v_l_old)
    v_scale = max(abs(v_v_old - v_l_old), abs(v_l_old), abs(v_v_old), 1.0)

    if abs(dv_rel) / v_scale < 1e-4:  # relative velocity change -> 0
        n_pass += 1

print(f"   Test 2 (K_drag->inf, phases lock): {n_pass}/500 passed")
assert n_pass == 500, f"K_drag->inf test failed: {n_pass}/500"


# --- Test 3: Drag cancels in mixture sum (Newton's 3rd) ---
# With symmetric friction (mdot_old = 0 => sigma_fric = 0 for both), drag alone
# conserves total momentum because R_l + R_v = 0 and the matrix is symmetric
# in the drag terms. With asymmetric friction sigma, drag still cancels in R_l+R_v
# but the total increment (1+sf_l)*Delta_l + (1+sf_v)*Delta_v = R_l+R_v holds.
n_pass = 0
for _ in range(500):
    alpha_v = rng.uniform(0.05, 0.95)
    alpha_l = 1 - alpha_v
    rho_l = rng.uniform(500, 1000)
    rho_v = rng.uniform(10, 100)
    F_drag_val = rng.uniform(-1e5, 1e5)
    K_drag_val = rng.uniform(1e3, 1e6)
    dt_val = 5e-5
    dx_val = 0.175
    A_flow = 7.854e-5
    beta_val = dt_val * A_flow / dx_val
    V_face = dx_val * A_flow

    # At dp=0, fric=0, mdot_old=0: sigma_fric=0, drag-only case
    # R_l = +dt*F*V, R_v = -dt*F*V, so R_l+R_v = 0
    # With sf=0: Delta_l + Delta_v should be exactly 0
    ml_d, mv_d, _, _, _ = coupled_solve(
        alpha_l, alpha_v, rho_l, rho_v, 0.0, 0.0,
        dp=0.0, fric_l=0.0, fric_v=0.0, F_drag=F_drag_val, K_drag=K_drag_val,
        V_face=V_face, dt=dt_val, beta=beta_val, dx=dx_val)

    total_change = (ml_d + mv_d) - 0.0  # mdot_old_total = 0
    if abs(total_change) < 1e-15:
        n_pass += 1

print(f"   Test 3 (Newton's 3rd, drag conserves total): {n_pass}/500 passed")
assert n_pass == 500, f"Newton's 3rd test failed: {n_pass}/500"

# --- Test 3b: Mixture momentum constraint with friction ---
# Even with asymmetric friction, the sum equation must hold:
# (1+sf_l)*Delta_l + (1+sf_v)*Delta_v = R_l + R_v = beta*dp - dt*(fric_l+fric_v)
n_pass = 0
for _ in range(500):
    alpha_v = rng.uniform(0.05, 0.95)
    alpha_l = 1 - alpha_v
    rho_l = rng.uniform(500, 1000)
    rho_v = rng.uniform(10, 100)
    mdot_l = rng.uniform(-2, 2)
    mdot_v = rng.uniform(-2, 2)
    dp = rng.uniform(-1e6, 1e6)
    fric_l = rng.uniform(0, 1e5) * np.sign(mdot_l) if abs(mdot_l) > 0.01 else 0.0
    fric_v = rng.uniform(0, 1e4) * np.sign(mdot_v) if abs(mdot_v) > 0.01 else 0.0
    F_drag_val = rng.uniform(-1e5, 1e5)
    K_drag_val = rng.uniform(1e3, 1e6)
    dt_val = 5e-5
    dx_val = 0.175
    A_flow = 7.854e-5
    beta_val = dt_val * A_flow / dx_val
    V_face = dx_val * A_flow

    ml_new, mv_new, _, a_ll_val, a_vv_val = coupled_solve(
        alpha_l, alpha_v, rho_l, rho_v, mdot_l, mdot_v,
        dp, fric_l, fric_v, F_drag_val, K_drag_val,
        V_face=V_face, dt=dt_val, beta=beta_val, dx=dx_val)

    # Compute sigma_fric from the solver's logic
    EPS = 1e-6
    K_geom = 0.175
    sf_l = 2 * dt_val * K_geom * abs(mdot_l) / (max(alpha_l, EPS) * max(rho_l, 0.01) * A_flow**2)
    sf_v = 2 * dt_val * K_geom * abs(mdot_v) / (max(alpha_v, EPS) * max(rho_v, 0.01) * A_flow**2)

    Delta_l = ml_new - mdot_l
    Delta_v = mv_new - mdot_v

    lhs = (1 + sf_l) * Delta_l + (1 + sf_v) * Delta_v
    rhs = beta_val * (max(alpha_l, EPS) + max(alpha_v, EPS)) * dp - dt_val * (fric_l + fric_v)
    # Note: R_l+R_v = beta*(alpha_l+alpha_v)*dp - dt*(fric_l+fric_v) + dt*F*V - dt*F*V
    rhs_correct = beta_val * max(alpha_l, EPS) * dp + beta_val * max(alpha_v, EPS) * dp \
                  - dt_val * fric_l - dt_val * fric_v

    err = abs(lhs - rhs_correct) / max(abs(rhs_correct), 1e-20)
    if err < 1e-8:
        n_pass += 1

print(f"   Test 3b (mixture momentum constraint): {n_pass}/500 passed")
assert n_pass == 500, f"Mixture momentum test failed: {n_pass}/500"


# --- Test 4: Sign test — F_drag > 0 accelerates liquid, decelerates vapor ---
n_pass = 0
for _ in range(500):
    alpha_v = rng.uniform(0.1, 0.9)
    alpha_l = 1 - alpha_v
    rho_l = rng.uniform(500, 1000)
    rho_v = rng.uniform(10, 100)
    mdot_l = 0.5  # positive flow
    mdot_v = 0.5
    dt_val = 5e-5
    dx_val = 0.175
    A_flow = 7.854e-5
    beta_val = dt_val * A_flow / dx_val
    V_face = dx_val * A_flow
    F_drag_val = rng.uniform(1e3, 1e6)  # positive F_drag (v_v > v_l)
    K_drag_val = 2 * F_drag_val  # from linearization

    # With positive drag
    ml_d, mv_d, _, _, _ = coupled_solve(
        alpha_l, alpha_v, rho_l, rho_v, mdot_l, mdot_v,
        dp=0.0, fric_l=0.0, fric_v=0.0,
        F_drag=F_drag_val, K_drag=K_drag_val,
        V_face=V_face, dt=dt_val, beta=beta_val, dx=dx_val)

    # Without drag (baseline)
    ml_0, mv_0, _, _, _ = coupled_solve(
        alpha_l, alpha_v, rho_l, rho_v, mdot_l, mdot_v,
        dp=0.0, fric_l=0.0, fric_v=0.0,
        F_drag=0.0, K_drag=0.0,
        V_face=V_face, dt=dt_val, beta=beta_val, dx=dx_val)

    # F_drag > 0: liquid should be pushed forward (+), vapor backward (-)
    liquid_boost = (ml_d - ml_0)
    vapor_retard = (mv_d - mv_0)

    if liquid_boost > -1e-15 and vapor_retard < 1e-15:
        n_pass += 1

print(f"   Test 4 (sign: F>0 accel liquid, decel vapor): {n_pass}/500 passed")
assert n_pass == 500, f"Sign test failed: {n_pass}/500"


# --- Test 5: beta_total is always positive ---
n_pass = 0
for _ in range(500):
    alpha_v = rng.uniform(0.01, 0.99)
    alpha_l = 1 - alpha_v
    rho_l = rng.uniform(500, 1000)
    rho_v = rng.uniform(1, 100)
    K_drag_val = rng.uniform(0, 1e8)
    dt_val = 5e-5
    dx_val = 0.175
    A_flow = 7.854e-5
    beta_val = dt_val * A_flow / dx_val

    EPS = 1e-6
    al = max(alpha_l, EPS)
    av = max(alpha_v, EPS)
    rl = max(rho_l, 0.01)
    rv = max(rho_v, 0.01)

    sigma_drag_l = dt_val * K_drag_val * dx_val / (al * rl)
    sigma_drag_v = dt_val * K_drag_val * dx_val / (av * rv)

    # Use small friction sigma for generality
    sigma_fric_l = rng.uniform(0, 10)
    sigma_fric_v = rng.uniform(0, 10)

    a_ll = 1 + sigma_fric_l + sigma_drag_l
    a_vv = 1 + sigma_fric_v + sigma_drag_v
    det = a_ll * a_vv - sigma_drag_l * sigma_drag_v

    beta_total = beta_val * (al * (a_vv + sigma_drag_l)
                           + av * (a_ll + sigma_drag_v)) / det

    if beta_total > 0 and np.isfinite(beta_total):
        n_pass += 1

print(f"   Test 5 (beta_total > 0): {n_pass}/500 passed")
assert n_pass == 500, f"beta_total positivity test failed: {n_pass}/500"


# --- Test 6: Determinant expansion identity ---
n_pass = 0
for _ in range(500):
    sf_l = rng.uniform(0, 100)
    sf_v = rng.uniform(0, 100)
    sd_l = rng.uniform(0, 100)
    sd_v = rng.uniform(0, 100)

    a_ll = 1 + sf_l + sd_l
    a_vv = 1 + sf_v + sd_v
    det_full = a_ll * a_vv - sd_l * sd_v
    det_expanded = (1 + sf_l) * (1 + sf_v) + (1 + sf_l) * sd_v + sd_l * (1 + sf_v)

    err = abs(det_full - det_expanded) / max(abs(det_full), 1e-20)
    if err < 1e-10:
        n_pass += 1

print(f"   Test 6 (determinant identity): {n_pass}/500 passed")
assert n_pass == 500, f"Determinant identity test failed: {n_pass}/500"


# --- Test 7: Edwards-like conditions magnitude check ---
print("\n   Edwards-like conditions:")

# Near break: alpha=0.8, high relative velocity
alpha_v_test = 0.8
alpha_l_test = 0.2
rho_l_test = 800.0
rho_v_test = 50.0
mdot_l_test = 0.5
mdot_v_test = 0.3
dt_test = 5e-5
dx_test = 0.175
A_test = 7.854e-5
V_test = dx_test * A_test
beta_test = dt_test * A_test / dx_test

v_l_test = mdot_l_test / (alpha_l_test * rho_l_test * A_test)
v_v_test = mdot_v_test / (alpha_v_test * rho_v_test * A_test)
v_rel_test = v_v_test - v_l_test

print(f"   alpha_v={alpha_v_test}, rho_l={rho_l_test}, rho_v={rho_v_test}")
print(f"   v_l={v_l_test:.1f} m/s, v_v={v_v_test:.1f} m/s, v_rel={v_rel_test:.1f} m/s")

# Compute F_drag at these conditions
mu_test = 2.8e-4
d_b_test = 1e-3
Re_test = rho_l_test * abs(v_rel_test) * d_b_test / mu_test
CD_test = (24 / Re_test) * (1 + 0.1 * Re_test**0.75)
F_drag_test = 0.75 * CD_test / d_b_test * alpha_v_test * rho_l_test * abs(v_rel_test) * v_rel_test
K_drag_test = 2 * abs(F_drag_test) / max(abs(v_rel_test), 1e-6)

sigma_drag_l_test = dt_test * K_drag_test * dx_test / (alpha_l_test * rho_l_test)
sigma_drag_v_test = dt_test * K_drag_test * dx_test / (alpha_v_test * rho_v_test)

print(f"   F_drag = {F_drag_test:.0f} N/m^3")
print(f"   K_drag = {K_drag_test:.0f}")
print(f"   sigma_drag_l = {sigma_drag_l_test:.4f}")
print(f"   sigma_drag_v = {sigma_drag_v_test:.4f}")
print(f"   dt*F_drag*V = {dt_test * F_drag_test * V_test:.6f} kg/s")

# Solve with drag vs without
ml_d, mv_d, det_d, _, _ = coupled_solve(
    alpha_l_test, alpha_v_test, rho_l_test, rho_v_test,
    mdot_l_test, mdot_v_test,
    dp=1e6, fric_l=100.0, fric_v=10.0,
    F_drag=F_drag_test, K_drag=K_drag_test,
    V_face=V_test, dt=dt_test, beta=beta_test, dx=dx_test)

ml_0, mv_0 = uncoupled_solve(
    alpha_l_test, alpha_v_test, rho_l_test, rho_v_test,
    mdot_l_test, mdot_v_test,
    dp=1e6, fric_l=100.0, fric_v=10.0,
    dt=dt_test, beta=beta_test)

print(f"\n   Coupled:   mdot_l={ml_d:.6f}, mdot_v={mv_d:.6f}")
print(f"   Uncoupled: mdot_l={ml_0:.6f}, mdot_v={mv_0:.6f}")
print(f"   Drag effect on liquid: {(ml_d-ml_0):.6e}")
print(f"   Drag effect on vapor:  {(mv_d-mv_0):.6e}")

# Verify total momentum is consistent
total_coupled = ml_d + mv_d
total_uncoupled = ml_0 + mv_0
print(f"   Total coupled:   {total_coupled:.6f}")
print(f"   Total uncoupled: {total_uncoupled:.6f}")

print("\n" + "=" * 70)
print("DERIVATION COMPLETE — ALL VERIFICATIONS PASSED")
print("=" * 70)
