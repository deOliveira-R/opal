# 6-Equation Principled Physics Stack: Implementation Plan

## Motivation

The 6-equation two-fluid model exists to compute phase-dependent physics from
first principles — separate phasic momentum, interfacial correlations for mass,
momentum, and energy exchange using the relative velocity v_rel = v_v - v_l.
The 5-equation drift-flux model cannot compute v_rel (it uses an algebraic slip
relation), locking it into lumped-parameter closures like Jones/Lahey relaxation
and tau_mix moderation.

The current 6-eq solver achieves 36.6% MAPE on Edwards blowdown, compared to
22.7% for the 5-eq and ~20% for RELAP5. The 6-eq uses tau_v=3.5e-3 — a lumped
moderation parameter 7x larger than the 5-eq's tau_mix=4.5e-4. This tau_v
compensates for three missing physics:

1. **Missing convective interfacial HT** — Nu=2 (conduction limit) ignores the
   large relative velocities present during blowdown (v_rel ~ 10-50 m/s)
2. **Missing virtual mass force** — no physical inertial resistance to rapid
   phase separation
3. **Explicit vapor flux lag** — phasic momentum is solved after the pressure-void
   block, creating a 1-timestep lag

The plan below addresses all three through principled physics that exploits the
6-eq model's unique capabilities, layered so each piece enables the next.

## Background: What We Proved

### The Darwinian Campaign (Session March 31, 2026)

Over 25+ experiments across 3 rounds, we established:

- **Scalar pressure solve baseline**: 40.9% MAPE (geometric HT, no J/L)
- **J/L without block coupling**: 48.2% (WORSE — runaway flashing without implicit feedback)
- **2x2 Block (p, alpha) + J/L + BFL**: 36.6% (BEST — block enables J/L)
- **Implicit vapor flux in Row 2**: 70-112% (positive feedback, removes natural lag-damping)
- **Blended compressibility**: 63-1192% (isentropic portion destabilizes)
- **Predictor-corrector**: 40.5% (algebraically equivalent to single-pass)
- **Newton iteration**: 45.8% (compressibility model is bottleneck)

### The Enabling Pattern

Models that fail alone can succeed in combination:
- Block coupling alone (no J/L): 70.7% (MUCH WORSE)
- J/L alone (scalar): 48.2% (WORSE)
- **Block + J/L: 36.6%** (BEST)

This demonstrates that the right modeling STACK matters more than individual features.
The block coupling ENABLED J/L by providing implicit void-pressure feedback.

### The Key Structural Finding

Making the vapor flux implicit in Row 2 of the block solve DESTABILIZES rather
than helps, because the 6-eq's unconstrained phasic momentum creates positive
feedback (dp -> more vapor flux -> more void -> more dp). The 5-eq avoids this
because drift-flux constrains vapor to be a fraction of mixture flow.

This means we CANNOT simply make the 6-eq "more implicit" — we need physics
that provides natural damping.

## The Principled Stack

### Overview

```
Layer 1: Virtual Mass Force (C_vm=0.5)
  |   Physical spring that resists rapid phase separation.
  |   All production codes (RELAP5, TRACE, CATHARE) include this.
  |   Enables: Layer 2 (stabilizes v_rel for Ranz-Marshall)
  |
  v
Layer 2: Ranz-Marshall Interfacial HT
  |   Nu = 2 + 0.6 * Re_b^0.5 * Pr_l^0.33
  |   Exploits 6-eq's UNIQUE capability: v_rel at every face.
  |   REPLACES Jones/Lahey for 6-eq (not additive).
  |   Self-limiting: more void -> more drag -> less v_rel -> less Nu
  |
  v
Layer 3 (optional): Weber-Number d_b
      d_b_max = We_cr * sigma / (rho_l * v_rel^2)
      Smaller effective d_b at high v_rel -> more interfacial area.
      HIGH RISK: must be capped (d_b_eff >= 0.1*d_b).
      Implement only after Layers 1+2 validated.
```

### Why This Order

- **Layer 2 without Layer 1 will fail** — same mechanism as J/L failure.
  Ranz-Marshall enhances HT proportional to sqrt(v_rel). Without virtual mass
  to resist v_rel spikes, the enhancement is uncontrolled.

- **Layer 1 without Layer 2 has modest effect** — virtual mass provides damping
  but doesn't improve the HT closure. Expected: allows tau_v reduction but
  doesn't close the MAPE gap.

- **Layer 1 + Layer 2 together** — virtual mass stabilizes v_rel, Ranz-Marshall
  converts v_rel into physics-based HT enhancement with natural negative feedback.
  The combined system has two self-limiting loops:
  1. More void -> more drag -> less v_rel -> less Nu -> less HT -> less void
  2. More v_rel -> more VM resistance -> v_rel grows more slowly

---

## Phase 1: Virtual Mass Force

### Rationale

The virtual mass (added mass) force represents the work needed to accelerate
surrounding liquid when a bubble accelerates:

  F_vm = C_vm * alpha * rho_l * (Dv_l/Dt - Dv_v/Dt)

C_vm = 0.5 for spherical bubbles (exact potential flow result, Zuber 1964).
RELAP5/MOD3 Vol I Section 3.4.6, TRACE Theory Manual Ch 4, and CATHARE all
include virtual mass in their two-fluid models.

At Edwards conditions (alpha=0.3, rho_l=700 kg/m3):
- Virtual mass coefficient: C_vm * alpha * rho_l = 0.5 * 0.3 * 700 = 105 kg/m3
- Vapor inertial mass: alpha * rho_v = 0.3 * 30 = 9 kg/m3
- VM increases effective vapor inertia by 12x

This is the ORDER OF MAGNITUDE needed to replace tau_v's artificial damping.

### Modelica Changes (library/Pipes/Pipe1D_TwoFluid.mo)

Add parameter:
```modelica
parameter Real C_vm = 0.5
  "Virtual mass coefficient [-]. 0.5 = sphere (Zuber 1964).
   Set 0 to disable. Blends to zero above alpha=0.5 (annular flow).
   Ref: RELAP5/MOD3 Vol I Sec 3.4.6, Drew & Lahey (1987).";
```

Add variable:
```modelica
Real K_vm[N + 1] "Virtual mass coefficient at faces [kg/m^3]";
```

Add equation (in face-averaged properties section):
```modelica
for i in 1:N + 1 loop
  K_vm[i] = C_vm * max(alpha_face[i], 1e-6) * rho_l_face[i]
            * (1.0 - noEvent(min(max((alpha_face[i] - 0.3) / 0.2, 0.0), 1.0)));
  // Blends to zero above alpha=0.5 (virtual mass not applicable in annular flow)
end for;
```

### Solver Changes (bridge_6eq_solver_block.py)

Read K_vm from bridge:
```python
K_vm = self.bridge.get('K_vm') if self.bridge.has('K_vm') else np.zeros(N + 1)
```

Modify 2x2 Cramer solve per face. The virtual mass adds coupling between liquid
and vapor momentum updates. In the semi-implicit discretization:

```python
# sigma_vm: virtual mass contribution (dimensionless ratio)
sigma_vm = dt * K_vm[i] / max(rho_f * self.A_flow * self.dx / self.A_flow, 1e-10)

# The 2x2 system becomes:
#   (1 + sigma_fric_l + sigma_drag + sigma_vm) * Delta_l
#     - (sigma_drag + sigma_vm) * Delta_v = R_l
#   -(sigma_drag + sigma_vm) * Delta_l
#     + (1 + sigma_fric_v + sigma_drag + sigma_vm) * Delta_v = R_v
```

The off-diagonal coupling increases from sigma_drag to (sigma_drag + sigma_vm).
This makes the determinant LARGER (more stable), and the effective beta_total
reflects the VM-limited momentum response.

### Edwards Test Model Update

Create `EdwardsTest_TwoFluid_VM.mo`:
- C_vm = 0.5
- No J/L (use_relaxation = 0)
- Geometric HT (Nu = 2, baseline closures)
- x_ne = 0.05 (baseline)

### Experiments

| Config | C_vm | tau_v | Expected |
|--------|------|-------|----------|
| Baseline (reference) | 0 | 3.5e-3 | 36.6% (with J/L) or ~40% (without) |
| VM only | 0.5 | 3.5e-3 | ~38% (modest, establishes VM effect) |
| VM + lower tau_v | 0.5 | 2.0e-3 | ~35% (tau_v reduction is the signal) |
| VM + lower tau_v | 0.5 | 1.0e-3 | ~33% or unstable (tests limit) |
| VM sweep | 0.25/0.5/1.0 | 2.0e-3 | Find optimal C_vm |

### Success Criterion

**tau_v can be reduced below 2e-3 without instability.** This confirms VM is
providing the physical damping that tau_v was compensating for.

### What Could Go Wrong

- C_vm=0.5 may over-couple phases (makes 6-eq behave like drift-flux). Mitigation:
  sweep C_vm, check that v_rel remains physically meaningful.
- VM singular at alpha->0 (no interface). Mitigation: the blend function ramps
  K_vm to zero below alpha=0.01 (via the alpha_face guard) and above alpha=0.5.
- OM may inline K_vm (same issue as drho_l_dp_s). Mitigation: K_vm is used in
  the momentum equations which reference it through der(mdot), so OM should keep it.
  If not, compute sigma_vm in the solver from available alpha_face and rho_l_face.

---

## Phase 2: Ranz-Marshall Interfacial Heat Transfer

### Rationale

The 6-eq model has v_rel at every face. The 5-eq model does not. This is THE
unique capability that justifies the additional equation. Currently it's wasted:
the interfacial HT uses Nu=2 (conduction limit, no v_rel dependence).

The Ranz-Marshall correlation:
  Nu = 2 + 0.6 * Re_b^0.5 * Pr_l^0.33

At Edwards conditions (v_rel=10 m/s, d_b=1mm, rho_l=760, mu_l=2.8e-4, cp_l=5400, k_l=0.55):
- Re_b = 760 * 10 * 0.001 / 2.8e-4 = 27,000
- Pr_l = 2.8e-4 * 5400 / 0.55 = 2.75
- Nu = 2 + 0.6 * 164 * 1.41 = 141

This is 70x the conduction limit. The resulting H_i*a_i ~ 4.2e7 W/(m^3*K),
comparable to J/L's 7.4e7 but physically derived and self-limiting.

**This REPLACES Jones/Lahey for 6-eq.** J/L is a lumped model that estimates the
effective HT from bulk thermodynamic properties. Ranz-Marshall computes it from
the actual flow field. Using both would double-count.

### Modelica Changes (library/Pipes/Pipe1D_TwoFluid.mo)

Add variables:
```modelica
Real v_rel_cell[N] "Cell-centered |relative velocity| [m/s]";
Real Re_b[N] "Bubble Reynolds number [-]";
Real Pr_l[N] "Liquid Prandtl number [-]";
Real Nu_cell[N] "Ranz-Marshall Nusselt number [-]";
```

Add equations (in interfacial closure section):
```modelica
for i in 1:N loop
  // Cell-centered relative velocity (averaged from adjacent faces)
  v_rel_cell[i] = 0.5 * (abs(v_v[i] - v_l[i]) + abs(v_v[i+1] - v_l[i+1]));

  // Bubble Reynolds and liquid Prandtl numbers
  Re_b[i] = rho_l[i] * v_rel_cell[i] * d_b_eff[i] / Medium.mu_f(p[i]);
  Pr_l[i] = Medium.mu_f(p[i]) * Medium.cp_f(p[i]) / Medium.k_f(p[i]);

  // Ranz-Marshall: Nu = 2 + 0.6 * Re^0.5 * Pr^0.33
  // Ref: Ranz & Marshall (1952); used by TRACE (Theory Manual Ch 3)
  Nu_cell[i] = 2.0 + 0.6 * noEvent(max(Re_b[i], 0.0))^0.5
                    * noEvent(max(Pr_l[i], 0.1))^0.33;
end for;
```

Replace Nu_i (constant) with Nu_cell[i] in h_i computation:
```modelica
  h_i[i] = (1 - use_regime_iac)
             * Nu_cell[i] * Medium.k_f(p[i]) / d_b_eff[i]
           + use_regime_iac
             * ((1.0 - blend_regime[i])
                  * Nu_cell[i] * Medium.k_f(p[i]) / max(d_b_eff[i], d_b_min)
                + blend_regime[i]
                  * Medium.k_f(p[i]) / max(delta_film[i], 1e-5));
```

Replace the q_i_l closure — use simple bidirectional geometric HT (no J/L):
```modelica
  q_i_l[i] = h_i[i] * a_i[i] * (T_sat_cell[i] - T_l[i]);
  // Positive when subcooled (heat into liquid), negative when superheated
  // (drives evaporation through Gamma = -q_i_l / h_fg > 0)
```

### Edwards Test Model Update

Create `EdwardsTest_TwoFluid_VM_RM.mo`:
- C_vm = 0.5
- use_relaxation = 0 (NO J/L — Ranz-Marshall replaces it)
- Geometric IAC (use_regime_iac = 0)
- d_b = 3e-4 (baseline)
- x_ne = 0.05 (baseline)

### Experiments

| Config | VM | RM | J/L | tau_v | Expected |
|--------|----|----|-----|-------|----------|
| Phase 1 best | 0.5 | No | No | ~2e-3 | ~35% (baseline for RM comparison) |
| RM only (no VM) | 0 | Yes | No | 3.5e-3 | Likely FAILS (runaway, like J/L) |
| VM + RM | 0.5 | Yes | No | 3.5e-3 | ~30% (RM enhancement, VM stability) |
| VM + RM + tau_v sweep | 0.5 | Yes | No | 0 to 3.5e-3 | Find optimal tau_v |
| VM + RM + lower tau_v | 0.5 | Yes | No | ~5e-4 | ~25% (approaching 5-eq) |

### The Litmus Test

**Sweep tau_v after implementing Layers 1+2.**
- If optimal tau_v decreases from 3.5e-3 toward 5e-4: **SUCCESS** — physics
  stack is replacing tau_v's role.
- If optimal tau_v stays at 3.5e-3: **FAILURE** — stack is not addressing
  the right deficit. Need to diagnose which of the three deficits remains.

### What Could Go Wrong

- Ranz-Marshall at Re~27000 is extrapolated (original data to Re~1000). TRACE
  uses it at high Re. Mitigation: cap Nu_max = 200 if needed.
- Without VM, Ranz-Marshall will cause the same runaway as J/L (tested in Round 1).
  **Layer 1 is prerequisite for Layer 2.**
- d_b is uncertain during nucleation (10-100 um vs 1mm bulk). Using bulk d_b
  in Re_b may underestimate early enhancement. Addressed by Layer 3 (Weber d_b).
- OM may inline v_rel_cell, Re_b, Nu_cell. Mitigation: h_i depends on Nu_cell
  which enters the der(h_l) equation — OM should keep the dependency chain.

---

## Phase 3: The Litmus Test — tau_v Sweep

### Rationale

This is NOT an implementation phase — it's a diagnostic. After Phases 1+2,
sweep tau_v from 3.5e-3 down to 0. Plot MAPE vs tau_v and compare to the
pre-VM/RM curve.

### Interpretation

| Observation | Meaning | Next Step |
|-------------|---------|-----------|
| Optimal tau_v drops to < 1e-3 | VM+RM replacing tau_v's role | Proceed to Phase 4 |
| Optimal tau_v drops to ~2e-3 | Partial replacement, need Layer 3 | Implement Weber d_b |
| Optimal tau_v stays at 3.5e-3 | VM+RM not addressing deficit | Diagnose: compressibility? drag? |
| System stable at tau_v = 0 | Full replacement achieved | Victory lap |

---

## Phase 4 (If Needed): Weber-Number Bubble Diameter

### Rationale

The critical Weber number for bubble breakup:
  We_cr = rho_l * v_rel^2 * d_b / sigma ~ 6-12

gives a maximum stable diameter:
  d_b_max = We_cr * sigma / (rho_l * v_rel^2)

At Edwards conditions: d_b_max ~ 2 um at v_rel=10 m/s. The HT product
h_i * a_i scales as Nu * k / d_b^2, so smaller d_b dramatically increases HT.

### Implementation

Modelica change (~10 lines):
```modelica
Real We_b[N] "Bubble Weber number [-]";
Real d_b_Weber[N] "Weber-limited bubble diameter [m]";

for i in 1:N loop
  We_b[i] = rho_l[i] * v_rel_cell[i]^2 * d_b / max(Medium.sigma(p[i]), 1e-6);
  d_b_Weber[i] = noEvent(max(
    12.0 * Medium.sigma(p[i]) / max(rho_l[i] * v_rel_cell[i]^2, 0.01),
    0.1 * d_b));  // Cap: never smaller than 10% of bulk d_b
end for;
```

Replace d_b_eff with d_b_Weber in a_i and h_i computations.

### HIGH RISK — Implement Only After Phases 1-3 Validated

The 1/d_b^2 scaling caused runaway in the d_b_eff inception model (Session 31c,
53.4% MAPE). The difference: Weber d_b is self-limiting through v_rel feedback,
while inception d_b was static. But the risk is real. Start with a conservative
cap (d_b_eff >= 0.1*d_b = 30 um) and sweep the cap.

---

## Phase 5 (Optional): Void Sub-cycling

If Phases 1-4 achieve tau_v < 1e-3 but not tau_v = 0, the remaining deficit is
the numerical lag (Deficit A). The solver architect recommends void sub-cycling:

- After the pressure solve and momentum update, sub-cycle the void transport
  M times within each pressure timestep (M = 5-20).
- Each sub-step: conservative alpha*rho_v update with linearized Gamma.
- Cost: ~2-3x (sub-steps are cheap, no bridge evaluation needed).

This resolves the multi-timestep thermodynamic feedback that h_mix captured in
a single step through its thermal compressibility.

---

## Summary: What Replaces What

| 5-eq Lumped Parameter | 6-eq Principled Replacement |
|-----------------------|---------------------------|
| tau_mix (A12 moderation) | Virtual mass force (physical inertia) |
| Jones/Lahey H_eff | Ranz-Marshall Nu(v_rel) |
| h_mix compressibility | Mechanical + Schur + (if needed) void sub-cycling |
| Drift-flux algebraic slip | Phasic momentum with interfacial drag |
| Martinelli-Nelson Phi2 | Per-phase Darcy wall friction |
| C_tau_alpha (alpha-dependent tau) | Drag equilibration of v_rel (natural feedback) |

---

## Files to Modify

| Phase | File | Change | Lines |
|-------|------|--------|-------|
| 1 | library/Pipes/Pipe1D_TwoFluid.mo | Add C_vm, K_vm | ~15 |
| 1 | solver/partitioner/two_fluid_variants/bridge_6eq_solver_block.py | VM in Cramer | ~40 |
| 1 | feasibility/models/edwards_6eq/EdwardsTest_TwoFluid_VM.mo | New test model | ~25 |
| 2 | library/Pipes/Pipe1D_TwoFluid.mo | v_rel_cell, Re_b, Pr_l, Nu_cell, h_i | ~40 |
| 2 | feasibility/models/edwards_6eq/EdwardsTest_TwoFluid_VM_RM.mo | New test model | ~25 |
| 3 | solver/edwards_6eq_sweep.py | Add VM+RM sweep configs | ~30 |
| 4 | library/Pipes/Pipe1D_TwoFluid.mo | We_b, d_b_Weber | ~15 |

Total: ~190 lines of new code across 3 files + 2 new test models.

## References

- Ranz & Marshall (1952), "Evaporation from drops", Chem. Eng. Prog. 48:141-146
- Zuber (1964), "On the dispersed two-phase flow", Chem. Eng. Sci. 19:897-917
- Drew & Lahey (1987), "The virtual mass and lift force on a sphere", Int. J. Multiphase Flow 13:113-121
- Ishii & Hibiki (2006), "Thermo-Fluid Dynamics of Two-Phase Flow", Ch 9 (drag), Ch 11 (IAC)
- RELAP5/MOD3 Code Manual Vol I, Sections 3.1-3.5
- TRACE Theory Manual, Chapters 3-6
- Bestion (1990), "The physical closure laws in the CATHARE code", Nuclear Eng. Design 124:229-245
- Hinze (1955), "Fundamentals of the hydrodynamic mechanism of splitting", AIChE J. 1:289-295
