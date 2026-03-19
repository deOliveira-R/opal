# QA Methodology for AI-Generated Numerical Code

> **LIVING DOCUMENT** — This methodology is actively used for all OPAL code.
> Originally written for the C++ solver; equally applicable to Modelica models
> and the Python extraction pipeline. The QA agent (session 2026-03-19) caught
> a V_gj variable swap (sigma→rho_f, AI failure mode #2) in Pipe1D_DriftFlux.mo
> using this methodology.

Date: 2026-03-18
Context: Post-mortem after two bugs escaped 32 tests in the OPAL 5-equation solver.
Author: QA Agent (self-evaluation)

---

## 1. Assessment of the Current Test Suite (74 new tests)

### 1.1 Strengths

The P0 and P1 test suites represent a genuine paradigm shift from the original 32 tests. The key improvements:

**Direct term isolation.** `TestClosureSignConvention` calls `DriftFluxClosures::compute()` with a hand-constructed `InterfacialState` and checks the sign of every output. This is the correct granularity. If the sign of `q_i_l` were flipped back to `(T_l - T_sat)`, tests `test_superheated_liquid_q_i_l_negative` and `test_subcooled_liquid_q_i_l_positive` would fail immediately — not after 10,000 steps of an integration test, but on the first assertion.

**Hand-calculated reference values.** `TestClosureGammaMagnitude::test_gamma_exact_value` computes `a_i`, `q_i_l`, and `Gamma` step-by-step from known inputs and compares at `rel=1e-10`. This is verification at its best: the test IS the derivation. If any coefficient, sign, or variable is wrong, the exact expected value will disagree.

**Algebraic identities.** `TestClosureEnergyBalance` verifies `q_i_l + q_i_v + Gamma*(h_v - h_l) = 0` at six different state points. This identity is enforced by construction in the code (line 138 of closures.hpp), so the test verifies the construction is correct. The parametric sweep across subcooled, superheated, near-saturation, high void, low void, and nucleation states is thorough.

**Per-step invariant checking.** `TestPostStepInvariants._run_and_check()` asserts finiteness, pressure bounds, alpha bounds, enthalpy bounds, and density positivity after EVERY timestep for 500-1000 steps. This is the single most valuable addition. The original tests only checked the final state, which means a transient violation (h_v dropping below h_sat_v at step 47 before recovering by step 200) would be invisible.

**Pressure sweep.** `TestPressureSweep` runs the solver at 7 different pressures from 1 MPa to 20 MPa. AI-generated code frequently works at the development pressure (10 MPa) but fails elsewhere. SimpleFluid properties are linear, so this primarily tests numerical stability across the pressure range rather than property accuracy — which is exactly the right thing for a verification test.

### 1.2 Weaknesses and False Security Risks

**W1: The closure tests verify the closure in isolation, but the energy equation applies closure outputs with additional algebra that is NOT individually tested.**

Specifically, in `five_eq_model.cpp` lines 439-446, the liquid energy update is:

```cpp
h_l_new = h_l_old + dt/m_l * (flux_l + p_work_l + q_wall_l + qi_l + phase_l)
```

where:
- `flux_l = mdot_l_in * (h_face_in - h_l_old) - mdot_l_out * (h_face_out - h_l_old)`
- `p_work_l = (1 - al) * V * dp_dt`
- `q_wall_l = q_total * (1 - al)`
- `qi_l = cr.q_i_l * V`
- `phase_l = -cr.Gamma * h_l_old * V`

The P0 tests verify that `cr.q_i_l` has the correct sign and magnitude. But no test verifies that `qi_l = cr.q_i_l * V` is applied with the correct sign in the energy equation. No test verifies that `phase_l = -cr.Gamma * h_l_old * V` has the correct sign (the negative is crucial: evaporation removes liquid mass, so the enthalpy of departing mass must be subtracted). An AI could write `phase_l = cr.Gamma * h_l_old * V` (missing negation) and the closure tests would still pass — only the integration tests in `TestPostStepInvariants` might catch it, and only if the effect is large enough to violate bounds.

**W2: The vapor energy equation has the symmetric problem.** Line 477: `phase_v = cr.Gamma * h_v_old * V` (positive because evaporation adds vapor mass carrying enthalpy h_v). If this sign were flipped, the closure tests would pass, the energy balance identity would pass, and only the integration tests might detect it through bounds violations.

**W3: `TestSuperheatedEnthalpyDirection` checks direction but not mechanism.** The test verifies that h_l decreases when liquid is superheated, which is good. But the test runs the full solver, so the decrease could come from advection washing out the high-enthalpy fluid rather than from interfacial heat transfer. To truly isolate the interfacial mechanism, the test should use equal pressures at inlet and outlet (it does: `p_in = p_out = 10e6`) AND check that the h_l decrease is consistent with the interfacial heat transfer rate. The current test does not do this second check.

**W4: `TestMixtureConservation::test_steady_state_uniform_flow` uses a 5% tolerance.** At steady state with a semi-implicit scheme, mass flow should be uniform to within truncation error of the pressure solve (machine precision for a tridiagonal). 5% is generous enough to mask a subtle leak.

**W5: No test isolates the advective flux sign.** The advective flux formula `mdot_l_in * (h_face_in - h_l_old) - mdot_l_out * (h_face_out - h_l_old)` has a critical sign structure: inlet flux adds energy, outlet flux removes it. If the minus sign between the two terms were dropped (making it a sum), the test suite would not catch it directly — only through bounds violations in multi-step integration tests. This is a classic AI error: both terms are structurally similar, and an AI generating the second by analogy with the first might forget to negate it.

**W6: No test verifies the pressure work sign.** `p_work_l = (1 - al) * V * dp_dt`. When pressure rises (dp_dt > 0), compression work adds energy to the fluid. When pressure drops, expansion cools it. No test checks this individually. A sign flip here would be compensated by the pressure solve feedback (pressure adjusts to maintain the mass balance), making it nearly invisible in integration tests.

**W7: The IAPWS integration tests (`TestIAPWSAtSolverStates`) check invariants every step for `test_iapws_5eq_invariants` but only every 100 steps for `test_iapws_5eq_depressurization`.** A transient violation between steps 101 and 199 would be missed. This is likely a performance trade-off, but it creates a gap.

### 1.3 Tests That Would Survive a Sign Flip (False Security)

I evaluated each test class for its ability to detect a specific sign flip reintroduced into the code:

| Test | Would detect q_i_l sign flip? | Would detect h_sat_l/h_sat_v swap? |
|------|-----|------|
| TestClosureSignConvention | YES | No (doesn't test clamping) |
| TestClosureEnergyBalance | YES (identity would break) | No |
| TestClosureGammaMagnitude | YES (exact value) | No |
| TestVaporEnthalpyFloor | No (tests clamping, not closure) | YES |
| TestVaporDensityPositivity | Indirect (may trigger via enthalpy drift) | YES (direct consequence) |
| TestLiquidEnthalpyBounds | No (tests clamping) | No |
| TestSuperheatedEnthalpyDirection | YES (direction reverses) | No |
| TestInterfacialArea | YES (extracted from Gamma) | No |
| TestPhasicFluxSplit | No (drift-flux, not closures) | No |
| TestPostStepInvariants | MAYBE (depends on magnitude) | YES (density goes negative) |

If someone introduced a sign error in the **phase change enthalpy term** (`phase_l = -cr.Gamma * h_l_old * V` -> `phase_l = cr.Gamma * h_l_old * V`), NONE of the P0 closure tests would catch it. The P1 post-step invariant tests MIGHT catch it if the effect is large enough to violate h_l bounds, but with small Gamma and large m_l, the per-step perturbation could be within tolerance.

### 1.4 Classification

Per the Verification-Validation-Benchmarking hierarchy:

- **P0 closure tests (TestClosureSignConvention, TestClosureEnergyBalance, TestClosureGammaMagnitude):** These are pure **verification** — mathematical correctness of the closure formula against hand-derived values. Correctly use no fluid properties (just hand-set InterfacialState). Grade: A.

- **P0 solver bounds tests (TestVaporEnthalpyFloor, TestVaporDensityPositivity, TestLiquidEnthalpyBounds):** These are **integration tests** (solver + fluid properties). They verify the solver's defensive clamping rather than the transport equations. Use SimpleFluid, which is acceptable for this purpose. Grade: B (would be A if they also verified the clamp VALUE, not just that clamps exist).

- **P1 drift-flux tests (TestInterfacialArea, TestNucleationOnset, TestDriftFluxVgj):** Pure **verification** with hand-calculated reference values. Grade: A.

- **P1 phasic flux split (TestPhasicFluxSplit):** Pure **verification** — algebraic identity (sum = total) plus hand-calculated no-slip limit. Grade: A.

- **P1 invariant tests (TestPostStepInvariants):** **Integration tests** — run full solver and check invariants. Use SimpleFluid. Grade: B+ (good invariant selection, but per-step checking is expensive enough that coverage is limited to 500-1000 steps).

- **P1 IAPWS tests (TestIAPWSAtSolverStates):** **Integration tests** combining solver + IAPWS-IF97 properties. These are NOT verification (cannot distinguish solver error from property error). Grade: B (correctly classified as integration, good invariant checks, but insufficient step coverage in depressurization test).

---

## 2. Self-Evaluation: Would the QA Agent Have Caught These Bugs?

### 2.1 Honest Answer: No.

If I had been asked to review the original code with my existing methodology, I would have:

1. **Classified the test suite correctly** — I would have noted that the 32 tests were mostly integration tests (solver + properties) and lacked unit-level verification of individual terms.

2. **Demanded conservation checks** — I would have asked "does mass and energy balance to machine precision?" But the operator-split scheme has O(dt) splitting error, so exact conservation would fail. I would have accepted "conservation to truncation order" and missed the sign bug, because the sign bug does not violate global conservation (it merely transfers energy in the wrong direction between the interface and the liquid).

3. **Demanded convergence rate tests** — I would have asked for grid refinement studies. These exist for the HEM model but not for the 5-equation model. A convergence rate test MIGHT have revealed the sign bug if it prevented the scheme from converging at the expected rate, but this is speculative.

4. **NOT demanded per-term sign verification.** My methodology says "check conservation, check convergence rates, check analytical solutions." It does NOT say "for every term in the discretized equation, verify the sign against a hand calculation." This is the critical gap.

5. **NOT checked bounds after every timestep.** My methodology says "verify conservation" but not "verify physical realizability at every intermediate state." The h_v < h_sat_v bug is a realizability violation, not a conservation violation.

### 2.2 Why My Methodology Failed

My system prompt is built around the traditional V&V hierarchy for human-written code. It assumes that bugs are primarily of these types:
- Discretization errors (wrong order of accuracy) -- caught by convergence tests
- Conservation violations (missing terms) -- caught by conservation tests
- Wrong equations (bad physics) -- caught by validation against experiments

It does NOT account for the dominant AI failure mode: **plausible substitution errors** that preserve the qualitative structure of the code while corrupting specific terms. These errors:

- Do not violate conservation (the energy balance identity `q_i_l + q_i_v + Gamma*(h_v - h_l) = 0` was enforced by construction, so the sign bug still conserved energy at the interface)
- Do not change convergence order (the scheme is still first-order with the wrong sign)
- Do not produce obviously wrong qualitative behavior (void fraction still grew, just at the wrong rate)
- ARE visible only when you compute the specific term's value from hand-traceable inputs

### 2.3 What Must Change

The QA agent methodology must add a new verification layer between "check conservation" and "check convergence": **term-by-term sign and magnitude verification**.

For every term in every discretized equation:
1. Define the physical meaning of the term (heat INTO liquid, mass flux OUT of cell, etc.)
2. Construct an input state where the term's sign and magnitude are analytically predictable
3. Extract the term's value (either by exposing it in the bindings, or by constructing a test where only that term is active)
4. Compare against the hand calculation

This is not standard V&V methodology. It is specific to AI-generated code, where the dominant failure mode is not "wrong algorithm" but "right algorithm with a sign flip in one term."

---

## 3. Missing Tests: Specific Gaps in the Current Suite

### 3.1 Critical Missing Tests (Would Catch Real Bugs)

**M1: Phase change enthalpy term sign (liquid energy equation)**
```
File: five_eq_model.cpp, line 442
Term: phase_l = -cr.Gamma * h_l_old * V
Bug: AI writes positive instead of negative

Test specification:
  Setup: N=1 cell, alpha=0.3, h_l > h_sat_l (superheated), zero flow (p_in=p_out),
         zero wall heat, known Gamma from closure.
  After one step:
    dh_l should include term: dt/m_l * (-Gamma * h_l * V)
    Sign: Gamma > 0 (evaporation), so phase_l < 0 (mass leaving carries enthalpy away)
    h_l should decrease by an amount consistent with phase_l magnitude.
  How to isolate: set H_i very large so phase change term dominates advection and p_work.
```

**M2: Phase change enthalpy term sign (vapor energy equation)**
```
File: five_eq_model.cpp, line 477
Term: phase_v = cr.Gamma * h_v_old * V
Bug: AI writes negative instead of positive

Test specification:
  Setup: Same as M1 but check h_v.
  After one step:
    dh_v should include term: dt/m_v * (+Gamma * h_v * V)
    Sign: Gamma > 0 (evaporation), so phase_v > 0 (mass arriving carries enthalpy)
    h_v should stay near h_sat_v or increase.
```

**M3: Advective flux sign structure**
```
File: five_eq_model.cpp, lines 439-440
Term: flux_l = mdot_l_in * (h_face_in - h_l_old) - mdot_l_out * (h_face_out - h_l_old)
Bug: AI writes + instead of - between terms

Test specification:
  Setup: N=3 cells, uniform h_l, positive steady flow, no phase change, no wall heat.
  Expected: flux_l = 0 for each cell (uniform enthalpy, no gradient).
  Then: set h_l[0] = h_l_base + 100e3, run one step.
  Cell 1 should gain energy (hot fluid advected in from cell 0).
  Cell 0 should lose energy (hot fluid advected out to cell 1).
  The SIGN of dh_l for cell 0 should be negative, cell 1 should be positive.
```

**M4: Pressure work sign**
```
File: five_eq_model.cpp, line 441
Term: p_work_l = (1 - al) * V * dp_dt
Bug: AI writes -(1-al)*V*dp_dt or al*V*dp_dt

Test specification:
  Setup: N=1 cell, no flow, no phase change, no wall heat.
  Step 1: set p_new = p_old + dp (pressurization).
  h_l should increase (compression adds energy).
  Step 2: set p_new = p_old - dp (depressurization).
  h_l should decrease (expansion removes energy).
  Magnitude check: dh_l approx = dt/m_l * (1-al) * V * dp/dt.
  Note: since pressure is solved implicitly, this requires careful setup
  (perhaps using WALL+PRESSURE BCs to control dp directly).
```

**M5: Wall heat split proportionality**
```
File: five_eq_model.cpp, lines 413-414
Term: q_wall_l = q_total * (1 - al), q_wall_v = q_total * al
Bug: AI swaps (1-al) and al

Test specification:
  Setup: N=1, alpha=0.1, q_wall=1e6, no flow, no phase change.
  After one step: liquid gains 90% of wall heat, vapor gains 10%.
  Then: alpha=0.9, same q_wall.
  After one step: liquid gains 10%, vapor gains 90%.
  Check: sign of dh_l and dh_v, and relative magnitude.
```

### 3.2 Momentum Equation Gaps (momentum.hpp)

**M6: Inertial momentum friction sign**
```
File: momentum.hpp, line 283
Term: fric[i] = geom * |mdot| * mdot / (rho_face * A^2)
Bug: friction has wrong sign convention, or missing |mdot| (making it signed)

Test specification:
  Setup: Known positive flow, known density.
  fric should be positive (opposes flow direction).
  Then: negative flow.
  fric should be negative (opposes flow, which is now negative).
  Key check: friction OPPOSES flow direction, so fric has same sign as mdot.
```

**M7: Inertial momentum pressure update direction**
```
File: momentum.hpp, line 248-249
Term: mdot[i] = mdot_old[i] + beta*(p[i-1] - p[i]) - dt*fric[i]
Bug: AI writes p[i] - p[i-1] (pressure gradient reversed)

Test specification:
  Setup: p[0] = 10 MPa, p[1] = 9 MPa, zero initial flow, zero friction.
  After one step: mdot[1] should be positive (flow from high to low pressure).
  Then: p[0] = 9 MPa, p[1] = 10 MPa.
  After one step: mdot[1] should be negative.
```

**M8: Inertial momentum friction RHS in pressure equation**
```
File: momentum.hpp, lines 199-200
Term: tri.d[i] += (mdot[i] - mdot[i+1]) - dt*(fric[i] - fric[i+1])
Bug: Wrong sign on friction correction, or fric[i] and fric[i+1] swapped

Test specification:
  Setup: Known steady-state with friction. Assemble pressure matrix.
  Verify that the tridiagonal system produces the same pressure field as
  the algebraic momentum at steady state (they must agree because
  inertial momentum reduces to algebraic at steady state).
```

**M9: Critical flow choke detection direction**
```
File: momentum.hpp, line 260
Term: if (cf && cf->is_choked && mdot_momentum > 0)
Bug: Missing mdot > 0 check, so reverse flow gets limited too

Test specification:
  Setup: High upstream pressure, low downstream. Choked outlet.
  Verify mdot[N] = mdot_crit (limited).
  Then: Reverse the pressures (downstream > upstream).
  Verify flow is NOT limited (choke only applies to forward flow).
```

### 3.3 Critical Flow Gaps (critical_flow.hpp)

**M10: Ransom-Trapp quality calculation**
```
File: critical_flow.hpp, lines 113-120
Term: x_local = (h_mix - h_f) / h_fg
Bug: AI uses h_g instead of h_f in numerator, or h_fg is (h_f - h_g)

Test specification:
  Setup: p=7 MPa, h_mix at saturation liquid (x=0), saturation vapor (x=1),
         and midway (x=0.5).
  For each: verify x_local matches expected value.
  Then: h_mix below h_f (subcooled): x=0.
  Then: h_mix above h_g (superheated): x=1.
```

**M11: Ransom-Trapp blend formula**
```
File: critical_flow.hpp, lines 137-143
Term: G_crit = G_sub*(1-blend) + G_hem*blend where blend = x/x_trans
Bug: blend = 1 - x/x_trans (inverted), or G_sub and G_hem swapped

Test specification:
  At x=0: G_crit should be G_sub (all subcooled).
  At x=x_trans: G_crit should be G_hem (fully two-phase).
  At x=x_trans/2: G_crit should be midway between G_sub and G_hem.
  Verify monotonic transition.
```

**M12: Ransom-Trapp Bernoulli formula**
```
File: critical_flow.hpp, line 124
Term: G_sub = sqrt(2 * rho_f * dp)
Bug: Missing factor of 2, or uses rho_v instead of rho_f

Test specification:
  Setup: Known p_cell, p_back, rho_f.
  Compute G_sub by hand: sqrt(2 * rho_f * (p_cell - p_back)).
  Verify against code output.
```

**M13: Ransom-Trapp HEM sound speed**
```
File: critical_flow.hpp, lines 128-131
Term: c_hem = sqrt(1 / (rho * drho_dp_h))
Bug: Missing 1/ (writes rho*drho), or sqrt of wrong quantity

Test specification:
  Setup: Known rho and drho_dp_h.
  c_hem = sqrt(1/(rho*drho_dp_h)) by hand.
  G_hem = rho * c_hem.
  Verify against code output.
```

### 3.4 Solver Orchestration Gaps (solver.cpp)

**M14: Face density computation at boundaries**
```
File: solver.cpp, lines 91-110
Term: rho_face[0] = 0.5*(rho_in + props[0].rho) for pressure BC
      rho_face[0] = props[0].rho for wall BC
Bug: Wall BC uses wrong cell (props[N-1] instead of props[0]),
     or averaging uses wrong index

Test specification:
  Setup: WALL BC at inlet. Verify rho_face[0] = props[0].rho.
  Setup: PRESSURE BC at inlet. Verify rho_face[0] = average of inlet and cell 0.
  Verify rho_face[N] = props[N-1].rho (outlet always uses last cell, not average).
```

**M15: Mixture enthalpy for property evaluation**
```
File: five_eq_model.cpp, lines 165-174
Term: h_mix = (1-al)*h_l + al*h_v
Bug: AI writes al*h_l + (1-al)*h_v (swapped)

Test specification:
  Setup: alpha=0.9, h_l=800 kJ/kg, h_v=2800 kJ/kg.
  Expected h_mix = 0.1*800 + 0.9*2800 = 2600 kJ/kg.
  If swapped: h_mix = 0.9*800 + 0.1*2800 = 1000 kJ/kg (very wrong).
  This could be tested by exposing evaluate_properties or by checking that
  the mixture density at a known (p, alpha, h_l, h_v) state matches
  the density at (p, h_mix) where h_mix is computed correctly.
```

### 3.5 Reconstruction Gaps (reconstruction.hpp)

**M16: MUSCL gradient ratio for negative flow**
```
File: reconstruction.hpp, lines 72-78 (MUSCL_Minmod)
Term: For negative flow (upwind = cell_R): r = (cell_R - cell_RR) / (cell_L - cell_R)
Bug: AI uses (cell_R - cell_RR) / (cell_R - cell_L) — sign flip in denominator

Test specification:
  Setup: cell_LL=1, cell_L=2, cell_R=3, cell_RR=4, mdot=-1.
  Expected: delta = L-R = -1, r = (R-RR)/delta = (3-4)/(-1) = 1.
  phi_minmod(1) = 1. face = R + 0.5*1*(-1) = 2.5. (linear interpolation)
  Verify face_value matches 2.5.
  Then: cell_LL=1, cell_L=2, cell_R=3, cell_RR=2 (non-monotone).
  r = (3-2)/(-1) = -1. phi_minmod(-1) = 0. face = 3.0 (donor cell).
  Verify face_value matches 3.0.
```

---

## 4. Principles: QA Methodology for AI-Generated Numerical Code

### 4.1 The Core Problem

Large language models produce code by pattern completion. They have seen thousands of examples of finite-volume energy equations, and they can write one that LOOKS correct. But pattern completion makes characteristic errors:

1. **Sign flips.** `(a - b)` vs `(b - a)`. Both are valid subtraction patterns. The LLM picks based on which it has seen more often in similar contexts, not based on physical reasoning.

2. **Variable swaps.** `h_sat_l` vs `h_sat_v`, `rho_l` vs `rho_v`. These are structurally identical tokens that appear in similar positions. The LLM may select the wrong one by analogy with a slightly different equation.

3. **Missing negation.** `Gamma = q / h_fg` vs `Gamma = -q / h_fg`. The negative sign carries critical physical meaning (evaporation vs. condensation direction) but is a single token.

4. **Factor errors.** Missing `2*`, missing area, missing volume. The LLM may omit a factor that it considers "implied" from a different formulation.

5. **Index errors.** `face[i]` vs `face[i+1]`, `cell[i-1]` vs `cell[i]`. Staggered-mesh indexing is notoriously error-prone even for humans.

6. **Convention drift.** The LLM defines `q_i_l = heat INTO liquid` in the closure, then uses `q_i_l` in the energy equation as though it means `heat FROM liquid`. The definition and usage are in different functions, possibly generated in different conversations.

These errors compile, run, produce finite numbers, and even pass coarse integration tests. They are the AI equivalent of the "off-by-one" errors that plague human programmers — but harder to find because they don't crash.

### 4.2 The Verification Hierarchy for AI Code

The traditional hierarchy (Verification -> Validation -> Production) must be extended with a new bottom layer:

```
Level 0: Term Verification      (every term, every sign, every factor)
Level 1: Equation Verification  (conservation, convergence, analytical solutions)
Level 2: Integration Testing    (solver + properties, multi-physics coupling)
Level 3: Validation             (comparison against experiments)
Level 4: Production Use
```

Level 0 is new. It does not exist in traditional V&V because human engineers are assumed to get individual terms right (or at least to debug them by reading the code). For AI-generated code, Level 0 cannot be skipped.

### 4.3 Level 0: Term Verification Rules

For every discretized equation in the solver:

**Rule T1: Enumerate all terms.** Write down every term in the discretized equation, with its physical meaning and expected sign for a reference state.

Example for the liquid energy equation:
```
h_l_new = h_l_old + dt/m_l * (
    flux_l    : advective enthalpy transport     [sign depends on flow direction]
  + p_work_l  : compression/expansion work        [positive when dp > 0]
  + q_wall_l  : wall heat to liquid               [positive when q_wall > 0]
  + qi_l      : interfacial heat to liquid         [negative when T_l > T_sat]
  + phase_l   : enthalpy carried by phase change   [negative for evaporation]
)
```

**Rule T2: For each term, write a test that isolates it.** Construct a state where only that term is active (set all other terms to zero by choosing appropriate boundary conditions, initial conditions, and closure parameters).

**Rule T3: Verify both sign and magnitude.** A sign-only test (`assert dh_l < 0`) is necessary but not sufficient. Compute the expected magnitude from the known inputs and compare at tight tolerance.

**Rule T4: Verify the term at both polarities.** If a term can be positive or negative (like pressure work, which depends on dp/dt sign), test both cases. An AI might get one polarity right and the other wrong.

**Rule T5: Verify index consistency at faces.** For any term that uses face values (advective flux, face density, face enthalpy), verify that face `i` corresponds to the left boundary of cell `i` and face `i+1` to the right. Test this with a non-uniform profile where getting the index wrong produces a detectably different answer.

### 4.4 Invariant Checking Rules

**Rule I1: Check physical realizability after every timestep, not just at the end.** Density > 0, 0 <= alpha <= 1, h_v >= h_sat_v, h_l <= h_sat_v, p > 0.

**Rule I2: Check algebraic identities at every intermediate state.** If the closure enforces `q_i_l + q_i_v + Gamma*(h_v - h_l) = 0`, verify this at every cell, every step, not just for isolated closure tests.

**Rule I3: Verify that defensive clamps use the correct bound value.** If the code clamps h_v to `[h_sat_v, h_max]`, test that the lower bound is indeed h_sat_v and not h_sat_l. This requires setting up a state that hits the clamp and verifying the clamped value.

### 4.5 Anti-Patterns to Watch For

**Anti-pattern 1: "It produces reasonable numbers."** A sign-flipped term can produce reasonable numbers if it is small relative to other terms. The q_i_l sign bug survived because interfacial heat transfer was typically a small correction to the advective flux. Test individual terms in isolation, not through emergent behavior.

**Anti-pattern 2: "The integration test passes."** An integration test exercises the code path but does not verify correctness. Two compensating errors can produce a passing integration test. Always demand term-level verification before integration testing.

**Anti-pattern 3: "It matches the oracle."** Benchmarking against another code establishes equivalence, not correctness. If both codes have the same sign convention error (common when the AI learned from the oracle's documentation), they will agree and both be wrong.

**Anti-pattern 4: "Conservation holds."** Conservation is necessary but not sufficient. The q_i_l sign bug conserved energy at the interface (the identity was enforced by construction). Conservation checks catch missing terms but not sign-flipped terms.

**Anti-pattern 5: "The convergence rate is correct."** Convergence rate tests catch discretization errors but not physics errors. A sign-flipped source term still converges at first order — it just converges to the wrong answer.

### 4.6 Review Checklist for AI-Generated Numerical Code

Before accepting any AI-generated solver code:

- [ ] Every term in every discretized equation has a dedicated sign+magnitude test
- [ ] Every term has been tested at both polarities (positive and negative)
- [ ] Every face-indexed quantity has been tested with a non-uniform profile
- [ ] Every clamp/floor/ceiling uses the correct bound variable (not a similar-named one)
- [ ] Physical realizability is checked after every timestep in at least one test
- [ ] Algebraic identities are verified across a sweep of states
- [ ] The code has been reviewed for the 6 AI failure modes listed in Section 4.1
- [ ] Conservation tests exist but are NOT the only verification
- [ ] At least one test uses analytically tractable fluid (SimpleFluid), NOT IAPWS
- [ ] Any test using IAPWS is classified as integration, not verification

### 4.7 Additions to the QA Agent System Prompt

The following principles should be added to the QA agent's methodology:

```
## AI Code Verification (Level 0)

When reviewing AI-generated numerical code:

1. **Demand term-level tests.** For every term in every discretized equation,
   there must be a test that isolates that term and verifies its sign and
   magnitude against a hand calculation. "The integration test passes" is
   not sufficient evidence.

2. **Check for the six AI failure modes.** For every function, ask:
   - Could a sign be flipped? Is there a test that would catch it?
   - Could two similar variables be swapped? Is there a test?
   - Could a negation be missing? Is there a test?
   - Could a factor (2, pi, area, volume) be missing? Is there a test?
   - Could a face index be off by one? Is there a test?
   - Could a convention defined in one place be used differently elsewhere?

3. **Require per-step realizability checks.** At least one integration test
   must check physical bounds (density > 0, 0 <= alpha <= 1, enthalpy within
   range) after EVERY timestep, not just at the final time.

4. **Verify clamp values, not just clamp existence.** If the code clamps h_v
   to [floor, ceiling], verify that floor = h_sat_v (not h_sat_l). The
   existence of a clamp does not prove it uses the right value.

5. **Do not trust emergent behavior tests alone.** A test that checks "void
   fraction grows" can pass even when the growth mechanism has the wrong sign,
   if other terms compensate. Always pair emergent tests with term-level tests.
```

---

## 5. Summary of Findings

### What the test suite gets right
- Direct closure verification with hand-calculated values (would have caught Bug 1)
- Per-step invariant checking with density positivity (would have caught Bug 2)
- Parametric sweeps across alpha and pressure ranges
- Correct V&V classification (SimpleFluid for verification, IAPWS for integration)

### What the test suite still misses
- Term-level verification of the energy equation assembly (5 terms, none individually tested)
- Sign verification of the momentum equation pressure gradient and friction
- All of critical_flow.hpp (Ransom-Trapp: quality calculation, blend formula, Bernoulli, HEM sound speed)
- Face density computation at boundaries in solver.cpp
- Mixture enthalpy calculation for property evaluation
- MUSCL reconstruction for negative flow direction

### What the QA agent must change
- Add "Level 0: Term Verification" to the verification hierarchy
- Demand per-term sign+magnitude tests for all AI-generated code
- Check for the 6 AI failure modes (sign flip, variable swap, missing negation, factor error, index error, convention drift) during every code review
- Require per-step realizability checks, not just end-state checks
- Never accept "the integration test passes" as sufficient verification for AI-generated code

### Priority for next implementation
1. M1, M2 (phase change enthalpy sign in energy equation) -- highest risk, zero coverage
2. M3, M4 (advective flux sign, pressure work sign) -- zero coverage for critical terms
3. M5 (wall heat split) -- simple but untested
4. M6-M9 (momentum equation terms) -- complex code, zero term-level coverage
5. M10-M13 (Ransom-Trapp critical flow) -- entirely untested at term level
6. M14-M16 (solver orchestration, reconstruction) -- secondary risk

---

## Files Referenced

- `/Users/rodrigo/git/OPAL/solver/tests/test_p0_closures_energy.py` -- 23 P0 tests (all passing)
- `/Users/rodrigo/git/OPAL/solver/tests/test_p1_term_verification.py` -- 51 P1 tests (all passing)
- `/Users/rodrigo/git/OPAL/solver/tests/COVERAGE_GAPS.md` -- gap analysis document
- `/Users/rodrigo/git/OPAL/solver/two_phase/closures.hpp` -- interfacial closures (fixed)
- `/Users/rodrigo/git/OPAL/solver/two_phase/five_eq_model.cpp` -- 5-eq transport (fixed)
- `/Users/rodrigo/git/OPAL/solver/two_phase/momentum.hpp` -- momentum models (untested for AI errors)
- `/Users/rodrigo/git/OPAL/solver/two_phase/solver.cpp` -- solver orchestration (partially tested)
- `/Users/rodrigo/git/OPAL/solver/two_phase/critical_flow.hpp` -- Ransom-Trapp (untested at term level)
- `/Users/rodrigo/git/OPAL/solver/two_phase/reconstruction.hpp` -- MUSCL (untested for negative flow)
