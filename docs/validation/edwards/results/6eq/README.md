# 6-Equation Two-Fluid Edwards Blowdown Results

## Overview

The 6-equation two-fluid model solves separate phasic momentum equations for liquid
and vapor, replacing the drift-flux algebraic slip of the 5-equation model. This
gives vapor its own physical inertia and enables first-principles interfacial
correlations using the relative velocity v_rel = v_v - v_l.

**Current best**: 36.6% MAPE (2x2 block solve + J/L + BFL, tau_v=3.5e-3)
**5-eq reference**: 22.7% MAPE (V24 + AsymCond C_tau_alpha=10)
**RELAP5 reference**: ~20% MAPE

## Model Architecture

```
Modelica: Pipe1D_TwoFluid.mo (6 conservation equations)
  - Mixture mass conservation (pressure linearization)
  - Void fraction transport (vapor mass balance + Gamma)
  - Liquid momentum (inertial + pressure + friction + drag)
  - Vapor momentum (inertial + pressure + friction - drag)
  - Liquid energy (phasic, with interfacial HT)
  - Vapor energy (phasic, with interfacial HT)

Closures (from Modelica):
  - Interfacial drag: Ishii bubbly (Schiller-Naumann C_D)
  - Interfacial HT: geometric (Nu=2) or Jones/Lahey relaxation
  - Per-phase wall friction: Darcy (f_D=0.02, no Phi2)
  - Critical flow: Henry-Fauske (N=0, sharp orifice)
  - IAPWS-IF97 properties (Regions 1, 2, 4)

Solver: bridge_6eq_solver_block.py
  - 2x2 block Thomas for (pressure, void fraction)
  - 2x2 Cramer per face for (mdot_l, mdot_v)
  - Explicit donor-cell for phasic energy
  - Break form loss (mixture basis, per RELAP5)
```

## Darwinian Campaign Summary

### Round 1: Closure Sweep (8 Modelica variants, scalar pressure solve)

Tested Jones/Lahey flashing, C_tau_alpha, bubble diameter, regime drag.
All closures transplanted from 5-eq production model.

| Variant | MAPE | Key Finding |
|---------|------|-------------|
| **baseline (no J/L, x_ne=0.05)** | **40.9%** | Best scalar solve |
| xne14 (no J/L, x_ne=0.14) | 44.1% | x_ne=0.14 hurts GS-1 |
| jl_c0 (J/L, C_tau=0) | 44.0% | J/L barely helps |
| jl_c5 (J/L, C_tau=5) | 43.8% | GS-1 → 70% (destroyed) |
| jl_c10 (J/L, C_tau=10) | 48.2% | Worse than baseline |
| jl_c10_slow (tau=0.050) | 49.3% | Slower flash doesn't help |
| jl_c10_fast (tau=0.005) | 50.7% | Faster flash even worse |
| jl_c10_regime (drag=2) | 73.2% | Regime drag catastrophic |

**Key finding**: J/L makes the 6-eq scalar solver WORSE because there's no
implicit void-pressure coupling to stabilize the flashing feedback loop.

### Round 2: Solver Architecture (3 approaches, parallel implementation)

Tested three competing approaches to add implicit void-pressure coupling.

| Approach | Best MAPE | Verdict |
|----------|-----------|---------|
| **B: 2x2 Block (p, alpha)** | **36.6%** | Winner (tau_v=3.5e-3) |
| A: Predictor-Corrector | 40.5% | Dead end |
| C: Newton Iteration | 45.8% | Negative result |

**Key finding**: The block solve provides implicit void-pressure coupling that
makes J/L safe. J/L went from 48.2% (scalar) to 36.6% (block + BFL).

### Round 3: Technique Refinements

Tested implicit vapor flux substitution, blended compressibility, and
various tau_v sweeps.

| Technique | Result | Why |
|-----------|--------|-----|
| Implicit vapor flux in Row 2 | 70-112% | Positive feedback from unconstrained phasic momentum |
| Blended compressibility | 63-1192% | Isentropic portion destabilizes |
| Block + J/L + BFL | **36.6%** | Confirmed as current best |

## Dead-End Map (DO NOT Revisit)

| Technique | MAPE | Root Cause |
|-----------|------|------------|
| h_mix compressibility (scalar) | 40.9% | Blocks wave propagation in two-phase |
| Pure isentropic (scalar) | 86.7% | No void-pressure feedback |
| Schur RHS on h_mix diagonal | 600-1000% | Double-counts void coupling |
| Full Schur (isenthalpic + RHS) | 112-199% | Operator splitting lag overshoots |
| Implicit vapor flux in Row 2 | 70-112% | Removes natural lag-damping |
| Blended compressibility (alpha-dependent A11) | 63-1192% | Isentropic portion destabilizes |
| Predictor-corrector iteration | 40.5% | Can't fix structural incompressibility |
| Newton iteration | 45.8% | Compressibility model is bottleneck |
| Regime map drag | 73.2% | Transition discontinuities |
| d_b=1e-4 or 5e-5 (stronger HT) | 76-393% | Runaway void growth without coupling |

## Block Solver tau_v Sweep (Best Configuration)

Model: EdwardsTest_TwoFluid_HF_Ramp (J/L + C_tau=10 + BFL)

| tau_v [s] | MAPE | GS-1 | GS-2 | GS-3 | GS-4 | GS-5 | GS-6 | GS-7 |
|-----------|------|------|------|------|------|------|------|------|
| 2.0e-3 | 41.7% | 38.5 | 50.4 | 39.5 | 34.3 | 45.4 | 43.9 | 39.9 |
| 2.5e-3 | 40.1% | 34.9 | 44.9 | 36.0 | 35.6 | 45.6 | 43.9 | 39.5 |
| 3.0e-3 | 37.5% | 34.5 | 40.6 | 34.3 | 34.1 | 41.6 | 40.5 | 37.1 |
| **3.5e-3** | **36.6%** | 35.1 | 39.3 | 33.9 | 33.0 | 40.1 | 38.7 | 36.3 |
| 4.0e-3 | 37.7% | 36.6 | 38.4 | 33.4 | 34.4 | 42.2 | 40.5 | 38.0 |
| 4.5e-3 | 37.6% | 39.6 | 36.7 | 32.4 | 35.7 | 41.6 | 39.3 | 37.7 |
| 5.0e-3 | 39.7% | 42.9 | 39.0 | 36.0 | 37.7 | 42.5 | 41.2 | 38.8 |

## Structural Findings

### 1. Phasic Momentum Does NOT Replace tau Moderation

The original hypothesis — that giving vapor its own momentum equation would provide
natural inertia, eliminating tau_mix — is **disproven**. The 6-eq requires 7x MORE
moderation (tau_v=3.5e-3 vs tau_mix=4.5e-4) because explicit phasic vapor flux is
inconsistent with the implicit pressure solve.

### 2. J/L Requires Block Coupling

Jones/Lahey relaxation drives aggressive flashing (H_eff ~ 10^7 W/(m^3*K) vs
geometric H_i*a_i ~ 6*10^5). Without implicit void-pressure coupling, this
overwhelms the pressure equation. The 2x2 block provides the needed coupling.

### 3. h_mix Compressibility vs Isentropic

The 5-eq uses h_mix compressibility as a proxy for implicit void-pressure coupling.
The 6-eq block uses isentropic phasic compressibility + explicit A12 coupling.
Both approximate the same Schur complement, but through different mechanisms.

### 4. The Enabling Pattern

Models that fail alone can succeed in combination:
- J/L alone (scalar): 48.2% (WORSE)
- Block alone (no J/L): 70.7% (MUCH WORSE)
- **J/L + Block: 36.6%** (BEST)

This demonstrates that the darwinian game must test COMBINATIONS, not just
individual features. The block coupling ENABLES J/L by providing the implicit
feedback that prevents flashing runaway.

## Files

### Solvers
- `solver/partitioner/two_fluid_variants/bridge_6eq_solver.py` — Scalar (R1 baseline)
- `solver/partitioner/two_fluid_variants/bridge_6eq_solver_block.py` — Block (current best)
- `solver/partitioner/two_fluid_variants/bridge_6eq_solver_pc.py` — Predictor-corrector (dead end)
- `solver/partitioner/two_fluid_variants/bridge_6eq_solver_newton.py` — Newton (dead end)

### Modelica Models
- `library/Pipes/Pipe1D_TwoFluid.mo` — 6-eq model (all closures ported from 5-eq)
- `feasibility/models/edwards_6eq/` — 10 Edwards test model variants

### Validation
- `solver/edwards_bridge_6eq_validation.py` — Single-run validation driver
- `solver/edwards_6eq_sweep.py` — Darwinian sweep driver

### Results
- `docs/validation/edwards/results/6eq/round1_closure_sweep.json` — 8 closure variants
- `docs/validation/edwards/results/6eq/round2_solver_sweep.json` — 12 solver experiments

## Next Steps

The remaining 14pp gap to 5-eq (36.6% → 22.7%) and 16pp to RELAP5 (~20%) likely
requires principled physics that exploits the 6-eq model's unique capabilities
(v_rel, separate phasic momentum) rather than replicating 5-eq patterns. Areas
under investigation:
- Ranz-Marshall Nu enhancement using v_rel
- Weber-number-dependent bubble breakup
- Virtual mass force (physical inertia for phase separation)
- Wall-to-fluid heat transfer
- Reducing/eliminating tau_v through physical drag coupling
