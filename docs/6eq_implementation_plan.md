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

## References

- RELAP5/MOD3 Code Manual, Volume I, Ch 3 — two-fluid field equations
- RELAP5/MOD3 Code Manual, Volume II, §2.2 — coupled phasic momentum solve
- TRACE Theory Manual, Ch 3 — two-fluid momentum with interfacial drag
- Ishii & Hibiki, Ch 9 — drag closures, regime transitions
- Wallis (1969) — annular flow correlations
- Current OPAL files: see Architecture Diagram above
