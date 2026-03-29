# 6-Equation Two-Fluid Model: Implementation Plan

## Current State (2026-03-28, updated after Session 2)

### Key Finding from Session 2

**The interfacial heat transfer model is the bottleneck, not drag or momentum coupling.**

Both the physics reviewer and solver architect independently concluded:
- The geometric HT model (`H_i = h_i * a_i = 6000 W/(m³·K)`) is **250x too weak** for rapid depressurization flashing
- Both OPAL 5-eq and 6-eq nucleate ~100ms late at GS-5 (experiment starts voiding at t=50ms)
- Void fraction MAE: OPAL 5-eq = 0.226, OPAL 6-eq = 0.221, RELAP5 = 0.111
- The 6-eq coupled momentum framework is algebraically correct (40/40 L0 tests) but cannot fix a closure deficiency
- The 6-eq model's advantage (phasic momentum) matters for counter-current flow, not co-current blowdown

### What Was Built in Session 2

| Component | File | Status |
|-----------|------|--------|
| Coupled momentum derivation | `docs/math/derivations/two_fluid_coupled_momentum.py` | Complete, 7 tests pass |
| Regime drag derivation | `docs/math/derivations/drag_regime_map.py` | Complete, 8 tests pass |
| 2x2 Cramer block solve | `solver/partitioner/bridge_6eq_solver.py` | Complete (sigma-only, no explicit drag) |
| Regime map drag | `library/Numerics/InterfacialDrag.mo` → `regime_map_drag()` | Complete (bubbly+slug+annular) |
| C_D cap at 0.44 | `library/Numerics/InterfacialDrag.mo` → both drag functions | Complete |
| Selectable drag model | `library/Pipes/Pipe1D_TwoFluid.mo` → `drag_model` parameter | Complete (1=ishii, 2=regime) |
| Level 0 QA tests | `solver/tests/test_bridge_6eq.py` | 40 tests, all pass |

### MAPE Results from Session 2

| Configuration | Overall MAPE | Notes |
|---|---|---|
| 6-eq baseline (sigma-only bubbly) | 39.5% | Session 1 result |
| + coupled momentum + explicit drag | 42.3% | Explicit drag too strong near break |
| + regime drag + explicit drag | 45.7% | Slug→annular cliff adds noise |
| + no explicit drag, per-phase sigma | 104.7% | Per-phase sigma too weak (0.003) |
| + no explicit drag, mixture sigma | 42.6% | GS-5 improved to 25.2%, others mixed |

**Conclusion: None of the momentum/drag improvements beat the 39.5% baseline.** The binding constraint is the interfacial HT closure, not the momentum coupling.

### What Exists

The 6-equation two-fluid model is **functional end-to-end** with all components in place:

| Component | File | Status |
|-----------|------|--------|
| SymPy momentum derivation | `docs/math/derivations/two_fluid_momentum.py` | Complete, 2700+ tests pass |
| SymPy drag derivation | `docs/math/derivations/interfacial_drag.py` | Complete, all tests pass |
| Interfacial drag closure | `library/Numerics/InterfacialDrag.mo` | Complete (Ishii-Zuber bubbly) |
| 6-eq Modelica model | `library/Pipes/Pipe1D_TwoFluid.mo` | Complete (~300 lines) |
| Edwards test case | `feasibility/models/EdwardsTest_TwoFluid_HF_Ramp.mo` | Complete |
| Compiled bridge | `feasibility/results/opal_bridge_EdwardsTest_TwoFluid_HF_Ramp.so` | 146KB, compiles in 7.5s |
| Bridge pipeline extensions | `solver/partitioner/codegen/equation_bridge.py` | Modified: mdot_l/mdot_v state support |
| Semi-implicit solver | `solver/partitioner/bridge_6eq_solver.py` | Complete (BridgeTwoFluidSolver) |
| Validation driver | `solver/edwards_bridge_6eq_validation.py` | Complete |
| Results | `docs/validation/edwards/results/six_eq_hf_ramp/` | .npz + MAPE JSON saved |

### Benchmark Results

**Edwards blowdown, N=24, dt=50µs, Henry-Fauske + RampedBreak:**

| Station | 5-eq Drift-Flux | 6-eq Two-Fluid | Delta |
|---------|----------------|----------------|-------|
| GS-1 (x=3.9m, near break) | 43.7% | **17.3%** | -26.4 pp |
| GS-2 (x=3.8m) | 26.5% | **15.3%** | -11.2 pp |
| GS-3 (x=2.9m) | 23.1% | 33.0% | +9.9 pp |
| GS-4 (x=2.0m) | 27.5% | 47.8% | +20.3 pp |
| GS-5 (x=1.5m) | 22.4% | 57.1% | +34.7 pp |
| GS-6 (x=0.9m) | 30.4% | 62.0% | +31.6 pp |
| GS-7 (x=0.1m, far end) | 24.6% | 43.9% | +19.3 pp |
| **Overall** | **28.3%** | **39.5%** | **+11.2 pp** |

The 6-eq model **improves** near the break (GS-1, GS-2) but **degrades** at far stations (GS-4 through GS-7). Wall time is comparable (14.7s vs 13.8s).

---

## Lessons Learned

### Lesson 1: Semi-implicit beta is geometric, not physical

The SymPy derivation correctly derives `beta_k = dt / M_k = dt*A / (alpha_k * rho_k * dx)` as the physical inertial coupling. But this makes beta_total ~180,000x smaller than the compressibility term alpha_coeff, effectively decoupling the pressure tridiagonal. No pressure waves propagate.

The 5-eq solver uses `beta = dt*A/dx` — a **geometric** coupling that intentionally omits density. This is the standard semi-implicit design (RELAP5, TRACE): decouple pressure coupling from density to remove the acoustic CFL constraint. Density enters through:
- `drho_dp` in the diagonal (compressibility)
- `sigma` in the friction/drag terms (resistance)

**Rule: The semi-implicit pressure coupling beta is a numerical regularization, not a physical coefficient. Use the same geometric `beta = dt*A/dx` for any model (3-eq, 5-eq, 6-eq).**

### Lesson 2: Explicit drag requires coupled-matrix solve

The interfacial drag force `F_drag` enters liquid momentum with + and vapor with −. The physical correction per step is `dt * F_drag * V / (M_k * (1+sigma_k))`, which requires the inertial mass M_k.

This correction is **unstable** when combined with geometric beta because:
- Geometric beta has no M_k in it (by design)
- Drag correction has M_k in the denominator (by physics)
- When M_k is moderate and sigma_drag is small (moderate alpha, moderate v_rel), the correction accumulates without the sigma damping catching it

The RELAP5 solution: solve a 2×2 coupled system per face that simultaneously determines liquid and vapor momentum changes. The inter-phase drag appears as off-diagonal terms in the block matrix, naturally stabilizing the coupling.

**Current state: drag enters ONLY through sigma (implicit damping). This provides first-order coupling — phases resist relative motion but no directed restoring force pushes them toward each other.**

### Lesson 3: Phase absence needs sigma boost, not momentum zeroing

When a phase is nearly absent (alpha_k < threshold), its inertial mass M_k → 0. Three approaches were tried:
1. **Explicit ramp** `mdot_k *= alpha_k / threshold` — too aggressive, creates "vapor lock"
2. **Hard alpha cap** at 0.95 — prevents full phase disappearance, artificial but stable
3. **Sigma boost** `sigma_k += 200 * (threshold - alpha_k) / threshold` — smoothly drives beta_eff → 0

The sigma boost (option 3) is the best: it lets the pressure tridiagonal naturally reduce the absent phase's contribution without creating discontinuities. Combined with the alpha cap at 0.95 for safety.

### Lesson 4: SymPy derivations were correct, implementation deviated

The derivation scripts (`two_fluid_momentum.py`, `interfacial_drag.py`) passed all verification tests and were never wrong. The implementation deviated during debugging:
- First version matched derivation → blew up (explicit drag singularity)
- "Fix" switched to geometric beta → correct for semi-implicit but no longer matches derivation
- Removed explicit drag → stable but incomplete coupling

**The derivation is the physics truth. The solver is a numerical approximation of the physics. They are deliberately different.**

---

## Architecture Diagram

```
Pipe1D_TwoFluid.mo          InterfacialDrag.mo
  (6 conservation eqs)        (Ishii-Zuber bubbly)
         |                          |
         └──────────┬───────────────┘
                    │
          OM translateModel
                    │
                    ▼
    opal_bridge_EdwardsTest_TwoFluid_HF_Ramp.so
                    │
                    ▼
          OMEquationBridge (Python ↔ C)
            set_state(p, alpha, h_l, h_v, mdot_l, mdot_v)
            evaluate()
            get('F_drag', 'fric_l', 'fric_v', 'v_l', 'v_v', ...)
                    │
                    ▼
          BridgeTwoFluidSolver
            ┌───────────────────────────────┐
            │ 1. Evaluate physics (bridge)  │
            │ 2. Compute sigma_l, sigma_v   │
            │    (friction + drag + absence)│
            │ 3. Pressure tridiagonal       │
            │ 4. Thomas algorithm           │
            │ 5. Liquid momentum update     │
            │ 6. Vapor momentum update      │
            │ 7. Critical flow limiter      │
            │ 8. Void fraction transport    │
            │ 9. Phasic energy updates      │
            └───────────────────────────────┘
                    │
                    ▼
            edwards_bridge_6eq_validation.py
              → .npz results + MAPE JSON
```

---

## Task List for Next Session (Revised after Session 2)

### NEW Priority: Flashing Inception Model (CRITICAL — actual bottleneck)

**Goal:** Replace the constant-Nu geometric interfacial HT with a pressure-dependent flashing model.

**Background:** The current model uses `Nu_i = 2` (conduction limit) and `d_b = 1mm`, giving `H_i = 6000 W/(m³·K)`. Rapid depressurization flashing requires ~2,000,000 W/(m³·K) — a 250x shortfall. Both 5-eq and 6-eq models nucleate 100ms late at GS-5.

**Options (increasing physics fidelity):**
1. **Smaller d_b at nucleation** (`d_b_flash = 0.1mm` when alpha < 0.01) → 100x enhancement
2. **Ranz-Marshall Nu at finite Re** (Nu >> 2 when v_rel > 0) → Nu ~ 90 at Edwards conditions
3. **Jones (1982) flashing model** (RELAP5 approach) → `H_i ~ 10^8 at early times`

**Implementation:** All in Modelica (cardinal rule). Modify `Pipe1D_TwoFluid.mo` and/or `Pipe1D_DriftFlux.mo` interfacial closure section. Benefits both 5-eq and 6-eq models.

**Verification:** Void fraction MAE at GS-5 should drop from 0.22 toward RELAP5's 0.11.

---

### COMPLETED: Task 1 — Coupled Phasic Momentum Matrix

**Status:** DONE. 2×2 Cramer block solve, sigma-only drag (no explicit correction),
40 L0 + 17 swap-detection tests pass. Algebraically correct but does not improve
Edwards MAPE because drag coupling is not the binding constraint.

**Files:** `solver/partitioner/bridge_6eq_solver.py`, `docs/math/derivations/two_fluid_coupled_momentum.py`

### COMPLETED: Task 2 — Flow Regime-Dependent Drag

**Status:** DONE. `regime_map_drag()` in `InterfacialDrag.mo` with bubbly→slug→annular
blending, C_D cap at 0.44, widened transition bands [0.20-0.40, 0.50-0.80].
Selectable via `drag_model` parameter (1=ishii, 2=regime_map). Not active by default
because it worsened Edwards MAPE (regime transitions not relevant for co-current blowdown).

**Files:** `library/Numerics/InterfacialDrag.mo`, `library/Pipes/Pipe1D_TwoFluid.mo`,
`docs/math/derivations/drag_regime_map.py`

### COMPLETED: Task 6 — Level 0 QA Tests

**Status:** DONE. 57 tests in `solver/tests/test_bridge_6eq.py`:
- 8 block solve tests (K_drag=0 recovery, K_drag→∞, Newton's 3rd, sign, determinant)
- 4 pressure tridiagonal tests (beta_total consistency, positivity, single-cell, 5-eq reduction)
- 12 regime drag tests (regime centers, C_D cap, continuity, sign, limits, hand calcs)
- 4 phase absence tests (sigma boost, beta_total bounds)
- 4 void transport tests (conservation, nucleation floor, evaporation/condensation)
- 4 energy tests (no-source, pressure work sign, interfacial HT sign, phase change)
- 4 structural tests (v_rel decrease, mixture momentum, symmetry)
- 17 swap-detection tests (all 8 plausible l/v swaps caught, including masked sd_l/sd_v)

### Task 3: Virtual Mass Force (PARKED — low priority)

Deferred. Solver architect estimates <1% impact on Edwards. The coupled matrix is ready
for it if needed for counter-current flow problems.

### Task 4: Remove Alpha Cap (LOW — can attempt cautiously)

Replace `ALPHA_MAX = 0.95` with `0.99`. The sigma-boost phase-absence treatment
is already in place. Expected to reduce late-time void MAE at GS-5 but not change
pressure MAPE significantly.

### Task 5: Void Fraction Analysis at GS-5 (COMPLETED — baseline established)

**Status:** Analysis done. Results:

| Model | Void MAE at GS-5 |
|-------|-------------------|
| RELAP5-Modified | 0.111 |
| RELAP5-HF | 0.144 |
| OPAL 5-eq | 0.226 |
| OPAL 6-eq | 0.221 |

The 2x improvement gap vs RELAP5 is entirely due to the interfacial HT closure (Nu=2, 250x too weak).

---

## Testing Gaps for Next Session

### L2 Integration Test (HIGH — closes formula↔solver gap)

The 57 L0 tests verify the mathematical formulas are correct but cannot call the
solver's `step()` method directly (requires OM bridge). If the solver's code drifts
from the tested helper functions, no test catches it.

**Needed:** An integration test that:
1. Constructs a simple 2-3 cell problem with known analytical pressure/momentum solution
2. Runs the actual `BridgeTwoFluidSolver.step()` through the bridge
3. Compares against the analytical solution
4. Uses an asymmetric state where each of the 8 plausible l/v swaps would produce
   a detectably different result

This requires a compiled bridge .so but can use a trivial Modelica model (e.g., SimpleFluid).

### Per-Phase Sigma Regression Test (MEDIUM — future-proofs)

Currently `sigma_drag_l == sigma_drag_v` (mixture normalization). If per-phase
normalization is ever re-enabled, 3 of 8 swap categories become live bugs unless
the Cramer indices are correct. The 17 swap-detection tests cover this, but they
test helper functions, not the solver code directly.

**Needed:** When per-phase sigma is activated, an L1 integration test that verifies
the solver produces different (correct) results for an asymmetric state.

---

## L0 Errors Detected in Session 2

Catalog of errors found and corrected by the L0 methodology during this session:

### Error 1: Explicit Drag Correction Destabilizes Pressure Solve (PHYSICS)

**What:** The RHS vectors `R_l` and `R_v` included `±dt*F_drag*V_face` explicit
drag correction terms. At near-break conditions (alpha=0.6, v_rel=50 m/s),
`dt*F*V = 3.0 kg/s` was 3.1x larger than the pressure term `beta*(1-α)*dp = 0.98 kg/s`,
overwhelming the semi-implicit pressure solve.

**Detection:** Physics reviewer Check 5d (combined pressure + drag interaction).
The `test_no_explicit_drag_in_rhs_isolation` L0 test was designed as a bug detector
for this — it verifies R_l and R_v are independent of F_drag.

**Fix:** Removed explicit drag from RHS. Drag enters ONLY through sigma terms in
the 2×2 matrix diagonal and cross-coupling.

**Category:** Sign/magnitude — explicit correction had correct sign but uncontrolled
magnitude relative to the semi-implicit pressure coupling.

### Error 2: Per-Phase Sigma Drag Too Weak by 100,000x (NORMALIZATION)

**What:** The physically-derived per-phase sigma `sigma_drag_l = dt*K*dx/(α_l*ρ_l)`
gives sigma ≈ 0.003, while the semi-implicit scheme requires sigma >> 1 for adequate
phase coupling. The old mixture normalization `2*dt*K*V/(ρ_m*A²*dx)` gives sigma ≈ 220.

**Detection:** Edwards validation — 104.7% MAPE with per-phase sigma (rarefaction
wave stalled). Solver architect quantified the 100,000x ratio.

**Fix:** Reverted to mixture sigma normalization. The per-phase derivation is
mathematically correct for the continuous equations but inconsistent with the
geometric-beta semi-implicit framework where beta = dt*A/dx has no density.

**Category:** Convention drift — correct derivation applied to wrong framework.
The semi-implicit scheme's "sigma" is a numerical damping ratio, not a physical
linearization coefficient.

### Error 3: Slug-to-Annular Drag Cliff (REGIME TRANSITION)

**What:** Drag drops 600x between alpha=0.60 (slug, F=84M N/m³) and alpha=0.70
(annular, F=135k N/m³) with the original transition band [0.60, 0.70]. During
Edwards blowdown, void sweeps rapidly through this range, injecting numerical noise.

**Detection:** Physics reviewer Check 6b. The `test_transition_continuity` L0 test
verified C0 continuity (no jumps) but did not flag the 600x magnitude change —
this is a physics issue, not a numerical one.

**Fix:** Widened transition bands to [0.50, 0.80] and [0.20, 0.40], reducing the
gradient. The `test_widened_transition_bands` L0 test validates the new band parameters.

**Category:** Plausible substitution — the linear blending was smooth (C0) but the
magnitude ratio between regimes was not checked. A "magnitude ratio at transition
boundaries" test would have caught this.

### Error 4: C_D Extrapolated 100x Beyond Validity (CORRELATION RANGE)

**What:** Schiller-Naumann C_D = (24/Re)(1 + 0.1Re^0.75) was applied at Re up to
10^5, far beyond its validity range of Re < 800. At high Re, the formula gives
C_D = 0.13 vs the standard Newton regime value of 0.44.

**Detection:** Physics reviewer Check 6a. The `test_cd_cap_at_high_re` L0 test
documents the behavior and verifies the cap.

**Fix:** Added `C_D = min(C_D_SN, 0.44)` in both `ishii_drag` and `regime_map_drag`.

**Category:** Regime applicability — correlation used outside documented validity
without a cap or warning.

---

## Execution Order (Revised)

```
Flashing model (CRITICAL) ──→ Validate on 5-eq ──→ Port to 6-eq
                                                        │
Alpha cap removal (LOW)    ─────────────────────────→   │
                                                        ▼
L2 integration test ──→ Per-phase sigma regression   Void comparison
```

The flashing model is the single highest-impact physics gap in OPAL.
It benefits both 5-eq and 6-eq models. Validate on 5-eq first (stable
solver, no confounding variables), then port to 6-eq.

---

## Session 3: Flashing Inception Model (2026-03-29)

### What Was Built

| Component | File | Status |
|-----------|------|--------|
| Flashing inception d_b_eff | `library/Pipes/Pipe1D_DriftFlux.mo` → `use_inception` | Complete, parked (runaway) |
| Jones/Lahey relaxation | `library/Pipes/Pipe1D_DriftFlux.mo` → `use_relaxation` | Complete, validated |
| Semi-implicit void coupling | `solver/partitioner/bridge_5eq_solver.py` | Complete |
| OM GreaterEq codegen fix | `solver/partitioner/codegen/bridge_codegen.py` | Complete |
| Edwards flash test case | `feasibility/models/EdwardsTest_DriftFlux_HF_Ramp_Flash.mo` | Complete |
| L0 QA tests | `solver/tests/test_flash_inception.py` | 35 tests, all pass |

### Results Summary

| Configuration | Pressure MAPE | Void MAE (GS-5) | Void Onset (GS-5) | Notes |
|---|---|---|---|---|
| 5-eq baseline (old solver) | 28.3% | 0.226 | ~220 ms | Previous canonical |
| **5-eq baseline (void coupling)** | **21.6%** | 0.264 | ~220 ms | Semi-implicit coupling alone |
| **5-eq + Jones/Lahey relaxation** | **23.0%** | **0.140** | ~170 ms | tau=0.025s, x_ne=0.14 |
| 5-eq + phasic drho_dp (reverted) | 75.7% | 0.466 | **6.5 ms** | Correct onset, pressure overshoots |
| d_b_eff inception (parked) | 53.4% | — | — | Runaway evaporation |
| RELAP5-Modified | ~27% | 0.111 | — | Reference |
| Experiment | — | — | ~10 ms | — |

### Key Findings

**1. Enhanced H_i alone worsens MAPE — the solver coupling was the bottleneck.**

Every attempt to increase the interfacial HT coefficient without solver changes made MAPE
worse. The root cause: the explicit void transport runs ahead of the implicit pressure solve.
At baseline Gamma, the coupling stability ratio is ~0.02 (safe). At >50x enhancement,
it exceeds 1.0 (unstable). The semi-implicit void-pressure coupling (linearized dGamma/dp
on the pressure diagonal) restores stability.

**2. The semi-implicit coupling improves the baseline by 7 percentage points.**

Even without enhanced flashing, adding the void-pressure coupling improved MAPE from 28.3%
to 21.4% — beating RELAP5's ~27% pressure prediction. The improvement is concentrated at
interior stations (GS-3 through GS-7), consistent with better pressure-void consistency.

**3. The Jones/Lahey relaxation model halves void MAE at GS-5.**

With `use_relaxation=1, tau_flash=0.025s, x_ne=0.14`:
- Void MAE: 0.274 → 0.145 (approaching RELAP5's 0.111)
- Alpha at 200ms: 0.397 vs experiment 0.35 (good match)
- Void onset still late: 170ms vs experiment ~10ms
- Pressure MAPE trade-off: +1.6% (21.4% → 23.0%)

**4. d_b_eff inception model causes runaway — parked.**

The geometric d_b_eff approach (use_inception=1) enhances H_i by 100x at nucleation but
has no inertial rate limiter. H_i grows with alpha through a_i, creating positive feedback.
53.4% MAPE vs 28.3% baseline. The Jones/Lahey model avoids this because H_relax =
alpha*(1-alpha)*rho_l*cp_f/tau is self-limiting at both alpha limits.

**5. OM CSE bug discovered and workaround found.**

OpenModelica pre-evaluates parameter expressions like `min(max(flash_model-1, 0), 1)` into
CSE (Common Subexpression Elimination) parameters ($cseN) with value=None in the XML.
The bridge initializes these to 0, making parameter-derived switches inoperable. Workaround:
use direct control parameters (use_inception, use_relaxation) instead of computing weights
from a selector parameter. This avoids the CSE path entirely.

**6. OM GreaterEq codegen gap fixed.**

When Modelica code uses `if param >= 1 then ... else ...`, OM generates bare `GreaterEq()`
C function calls that the bridge codegen didn't define. Fixed by adding comparison function
macros (#define GreaterEq, Greater, LessEq, Less) to the bridge C header.

### Errors Detected by L0 Methodology

**Error 1: Bidirectional Relaxation Overwhelms Condensation (SIGN/MAGNITUDE)**
First implementation applied relaxation H_eff to BOTH condensation and evaporation
directions. At subcooled conditions, H_relax is 10^6x geometric, pumping massive heat
into liquid (Gamma=-3392 at step 0). Fixed by splitting q_i_l into condensation (geometric)
and evaporation (relaxation) components using max(T_sat-T_l, 0) and max(T_l-T_sat, 0).

**Error 2: Explicit Void Source Wrong Sign (SIGN)**
First solver fix attempt added V*(rho_v-rho_l)/rho_v*Gamma to pressure RHS (negative
during evaporation). Should have been -V*(rho_v-rho_l)/rho_v*Gamma = +V*(rho_l-rho_v)/rho_v*Gamma
(positive, opposing depressurization). Wrong sign caused 597.5% MAPE. Corrected by switching
to semi-implicit diagonal treatment instead of explicit RHS.

**Error 3: OM CSE Silently Zeroes Parameter Switches (INFRASTRUCTURE)**
flash_model=2 in the .mo file was correctly compiled (XML shows 2.0), but the derived
CSE parameter $cse58 = min(max(flash_model-1, 0), 1) had value=None in XML, defaulting
to 0.0 in the bridge. Result: relaxation model completely inactive, simulation identical
to baseline. Detected by comparing trace output against known-different expected behavior.

### drho_dp Investigation: Why Void Onset Is Late

**Problem:** Void onset at GS-5 is 170ms in simulation vs ~10ms in experiment.

**Root cause:** `drho_dp(p, h_mix)` jumps 2400x when h_mix crosses h_f during
depressurization. At p ≈ 2.85 MPa, the mixture enthalpy crosses from Region 1
(subcooled, drho_dp ~ 5e-7) to Region 4 (two-phase, drho_dp ~ 1.2e-3). The
pressure tridiagonal diagonal (alpha_coeff = V*drho_dp/dt) increases 2400x,
freezing the pressure at ~2.7 MPa for 150ms.

**Attempted fix:** Phasic drho_dp evaluated at capped phasic enthalpies:
```
drho_dp = (1-alpha) * drho_dp_h(p, min(h_l, h_f-100))
        + alpha * drho_dp_h(p, max(h_v, h_g+100))
```
**Result:** Void onset correct (6.5ms!) but pressure overshoots to atmospheric
(75.7% MAPE). Pure mechanical compressibility (~5e-7) is 1000x too small for
the semi-implicit scheme — the rarefaction wave amplitude is unbounded.

**Also found:** IAPWS `region_ph` uses strict inequality `h < h_f` for Region 1.
At `h = h_f` exactly, evaluation falls into Region 4 (two-phase). Requires a
margin (h_f - 100) to stay in Region 1.

**Physics reviewer conclusion (correcting initial diagnosis):**

`drho_dp(p, h_mix)` is the **correct** effective compressibility for the semi-implicit
scheme — NOT a compensating error or double-counting. When enthalpies are frozen
during the pressure solve, the mixture-enthalpy parameterization anticipates the
density change that void growth will cause. The experimental data at GS-5 **confirms**
the pressure stall is physical: depressurization rate drops 1000x at the saturation
crossing (~2.85 MPa), matching the thermal compressibility jump.

Phasic drho_dp is correct only with fully implicit void-pressure coupling (RELAP5-style
block matrix where dalpha/dp enters the pressure diagonal without the dt multiplier).

**Solver architect evaluated 7 intermediate fixes — none work:**

| Fix | Why It Fails |
|-----|-------------|
| Alpha-blended drho_dp | At critical transition, alpha ~ 0, blend = pure phasic |
| Cap mixture drho_dp | Tuning parameter with no physics derivation |
| Smooth R1/R4 transition | No blending width gives the right time constant |
| Saturation-line drho_dp (drho_f/dp) | Negative sign — ill-conditions the tridiagonal |
| Saturation-tracking compressibility | Makes the problem worse (reduces effective drho_dp) |
| Energy coupling correction | Wrong direction (reduces drho_dp by 30%) |
| Existing dGamma/dp diagonal | Correct but negligible (4 orders of magnitude too small) |

### L0 Tests Written: 42 tests in `solver/tests/test_flash_inception.py`

| Category | Tests | Coverage |
|----------|-------|----------|
| d_b_eff inception model | 6 | Passthrough, nucleation, bulk, monotonicity, continuity, mid-ramp |
| Baseline equivalence (use_relaxation=0) | 4 | Subcooled, superheated, equilibrium, sweep |
| Relaxation model signs & magnitudes | 9 | H_relax literal, ratio, alpha-independence, self-limiting |
| Nucleation onset interaction | 1 | Floor + relaxation |
| Interface energy balance | 10 | 5 states × 2 models |
| Void-pressure coupling | 7 | Sign, scaling, hand calc with literal |
| Depressurization flashing (Gap 4) | 4 | Superheat, Gamma sign, hard-coded literal, condensation preserved |
| Small-superheat realizability | 1 | Gamma at 0.001 K superheat |

### Errors Detected by L0 Methodology

**Error 1: Bidirectional Relaxation Overwhelms Condensation (SIGN/MAGNITUDE)**
First implementation applied relaxation H_eff to BOTH condensation and evaporation
directions. At subcooled conditions, H_relax is 10^6x geometric, pumping massive heat
into liquid (Gamma=-3392 at step 0). Fixed by splitting q_i_l into condensation (geometric)
and evaporation (relaxation) components using max(T_sat-T_l, 0) and max(T_l-T_sat, 0).

**Error 2: Explicit Void Source Wrong Sign (SIGN)**
First solver fix attempt added V*(rho_v-rho_l)/rho_v*Gamma to pressure RHS (negative
during evaporation). Correct sign is positive (evaporation opposes depressurization in
rigid pipe). Wrong sign caused 597.5% MAPE. Corrected by switching to semi-implicit
diagonal treatment instead of explicit RHS.

**Error 3: OM CSE Silently Zeroes Parameter Switches (INFRASTRUCTURE)**
flash_model=2 in the .mo file was correctly compiled (XML shows 2.0), but the derived
CSE parameter $cse58 = min(max(flash_model-1, 0), 1) had value=None in XML, defaulting
to 0.0 in the bridge. Result: relaxation model completely inactive, simulation identical
to baseline. Detected by comparing trace output against known-different expected behavior.

**Error 4: d_b_eff Inception Runaway (PHYSICS/RATE-LIMITING)**
Geometric d_b_eff model enhances H_i by 100x at nucleation but has no inertial rate
limiter. Bubble growth is instantaneous and a_i grows with alpha, creating positive
feedback. 53.4% MAPE vs 28.3% baseline. Parked in favor of Jones/Lahey relaxation.

**Error 5: OM Bare GreaterEq() Codegen Gap (INFRASTRUCTURE)**
OM generates bare GreaterEq() C function calls for `if param >= 1 then...else...`.
Bridge codegen only handled relationhysteresis() wrappers, not bare calls. Fixed by
adding #define macros for GreaterEq/Greater/LessEq/Less.

**Error 6: IAPWS Region Boundary in drho_dp_h (INFRASTRUCTURE)**
min(h_l, h_sat_l) passes h = h_f exactly. Water.mo's region_ph uses strict inequality
(h < h_f), so h = h_f evaluates in two-phase Region 4 — returning the giant drho_dp
we tried to avoid. Requires margin (h_f - 100) to stay in Region 1.

---

## Roadmap for Next Session

### Current State (2026-03-29, end of Session 3)

- **930 tests pass** (888 pre-existing + 42 new flash inception)
- **Pressure MAPE: 21.6%** (baseline with semi-implicit coupling, beats RELAP5 ~27%)
- **Void MAE at GS-5: 0.140** (with Jones/Lahey, approaching RELAP5 0.111)
- **Void onset at GS-5: 170ms** (experiment: ~10ms — known limitation from drho_dp stall)

### Priority 1: Predictor-Corrector Pressure Solve (HIGH — unlocks phasic drho_dp)

The solver architect identified this as the cheapest path to fixing the void onset delay.
Run two Thomas solves per timestep:
1. First solve with h_mix drho_dp (current, provides thermal damping)
2. Update void + energy explicitly
3. Re-evaluate drho_dp with the new state
4. Second Thomas solve (corrects for the void change)

This effectively halves the "lag" between pressure and void without requiring a full
block-coupled matrix. Estimated implementation: 20-30 lines in `bridge_5eq_solver.py`.
Expected result: void onset moves from 170ms toward 30-50ms.

### Priority 2: Port Flashing Model to 6-eq (MEDIUM)

Apply the same changes to the 6-eq two-fluid model:
- `library/Pipes/Pipe1D_TwoFluid.mo`: add use_relaxation, tau_flash, split q_i_l
- `solver/partitioner/bridge_6eq_solver.py`: add semi-implicit void coupling + T_l/T_sat reads
- `feasibility/models/EdwardsTest_TwoFluid_HF_Ramp_Flash.mo`: new test case

The 6-eq model has v_rel directly available, enabling Ranz-Marshall Nu as an additional
enhancement (not possible in 5-eq where v_rel is algebraic, not explicit).

### Priority 3: Block-Coupled Pressure Solve (HIGH — major infrastructure)

Replace the scalar N×N pressure tridiagonal with a block-coupled system that
simultaneously solves p, alpha (and optionally h_l, h_v) per cell. This provides:
- Correct phasic drho_dp (no thermal compressibility double-counting)
- Implicit void-pressure coupling (dalpha/dp on LHS, not proportional to dt)
- Correct void onset timing (~10ms)

This is a significant solver architecture change (~200-300 lines). The current Thomas
algorithm becomes a block Thomas algorithm with 2×2 or 3×3 blocks per cell.

### Priority 4: tau_flash Sensitivity Study (LOW)

tau_flash=0.025 gives ~2x geometric enhancement at Edwards conditions. Systematic sweep
of tau_flash = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1] to map the pressure-void trade-off.
This requires multiple recompilations (one per tau value) unless a runtime parameter
override is implemented in the bridge.

### Key Files Modified in Session 3

| File | What Changed |
|------|-------------|
| `library/Pipes/Pipe1D_DriftFlux.mo` | use_inception, use_relaxation, tau_flash, d_b_eff, split q_i_l, phasic drho_dp comment |
| `solver/partitioner/bridge_5eq_solver.py` | Semi-implicit void coupling (dGamma/dp diagonal), T_l/T_sat reads, removed dead drho_dh |
| `solver/partitioner/codegen/bridge_codegen.py` | GreaterEq/Greater/LessEq/Less C macros |
| `solver/edwards_bridge_5eq_validation.py` | hf_ramp_flash model registry |
| `feasibility/models/EdwardsTest_DriftFlux_HF_Ramp_Flash.mo` | Created (use_relaxation=1, tau=0.025, x_ne=0.14) |
| `solver/tests/test_flash_inception.py` | Created (42 L0 tests) |

---

## References

- RELAP5/MOD3 Code Manual, Volume I, Ch 3 — two-fluid field equations
- RELAP5/MOD3 Code Manual, Volume II, §2.2 — coupled phasic momentum solve
- TRACE Theory Manual, Ch 3 — two-fluid momentum with interfacial drag
- Ishii & Hibiki, Ch 9 — drag closures, regime transitions
- Wallis (1969) — annular flow correlations
- Current OPAL files: see Architecture Diagram above
