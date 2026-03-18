# OPAL 5-Equation Solver Test Coverage Gap Analysis

Date: 2026-03-18
Triggered by: Two escaped bugs (closure sign convention, vapor enthalpy floor)

## Executive Summary

Two bugs survived to production because existing tests only checked *qualitative*
behavior (does void grow? does enthalpy increase?) without checking *quantitative
correctness* of individual terms. The closure sign bug persisted because no test
verified the DIRECTION of interfacial heat transfer. The enthalpy floor bug persisted
because no test verified that vapor enthalpy stays above h_sat_v or that rho_vapor
returns positive values for all solver states.

The existing 32 tests in `test_five_eq.py` plus 26 in `test_two_phase.py` cover
the "happy path" well but leave dangerous gaps in:
- Sign conventions of individual closure terms
- Enthalpy/density bounds after each transport update
- Property validity at solver-computed state points
- Individual term verification in the energy equation
- Edge-of-validity inputs to IAPWS

---

## P0: Tests That Would Have Caught the Two Escaped Bugs

These are the highest priority. Each test below is a direct consequence of
a bug that escaped.

### P0-1: Closure sign convention -- q_i_l direction

**Bug:** `q_i_l = H_i * a_i * (T_l - T_sat)` had wrong sign. When liquid is
superheated (T_l > T_sat), heat should LEAVE the liquid (q_i_l < 0), but
the old code ADDED heat to the liquid.

**What to test:** Call `DriftFluxClosures::compute()` directly with a constructed
`InterfacialState` and verify the SIGN of each output.

```
Setup:
  InterfacialState with T_l = T_sat + 10 K (superheated liquid)
  All other fields at reasonable saturation values
Assert:
  q_i_l < 0  (heat LEAVES superheated liquid)
  Gamma > 0  (evaporation when liquid is superheated)

Setup:
  InterfacialState with T_l = T_sat - 10 K (subcooled liquid)
Assert:
  q_i_l > 0  (heat ENTERS subcooled liquid)
  Gamma < 0  (condensation when liquid is subcooled)

Setup:
  InterfacialState with T_l = T_sat exactly
Assert:
  q_i_l == 0
  Gamma == 0
```

**Existing coverage:** TestVoidFractionEvolution::test_superheated_flashing
checks that alpha grows, but does NOT check q_i_l sign. The test would pass
even with the wrong sign if other terms compensated.

**Classification:** Verification (unit test of a formula, no physics needed)

---

### P0-2: Closure energy balance identity

**Bug context:** The interface energy balance `q_i_l + q_i_v + Gamma*(h_v - h_l) = 0`
must hold exactly (it's enforced by construction on line 138 of closures.hpp).

**What to test:**

```
Setup:
  Several InterfacialState configurations:
  - Superheated liquid (T_l > T_sat), alpha = 0.3
  - Subcooled liquid (T_l < T_sat), alpha = 0.3
  - Near-saturation (T_l ~ T_sat), alpha = 0.5
  - High void (alpha = 0.95)
  - Low void (alpha = 0.01)
Assert for each:
  |q_i_l + q_i_v + Gamma * (h_v - h_l)| < 1e-10 * max(|terms|)
```

**Existing coverage:** NONE. No test verifies the interfacial energy balance identity.

---

### P0-3: Closure Gamma magnitude (evaporation rate)

**What to test:** Gamma = -q_i_l / h_fg. Verify the magnitude is physically
reasonable and matches the formula exactly.

```
Setup:
  H_i = 1e5, T_l = T_sat + 10 K, alpha = 0.3, h_fg = 2e6
  a_i = max(4*0.3*0.7, 0.3) = max(0.84, 0.3) = 0.84
  q_i_l = H_i * a_i * (T_sat - T_l) = 1e5 * 0.84 * (-10) = -840000
  Gamma = -(-840000) / 2e6 = 0.42
Assert:
  Gamma == pytest.approx(0.42, rel=1e-10)
  q_i_l == pytest.approx(-840000, rel=1e-10)
```

**Existing coverage:** NONE. No test verifies Gamma magnitude against hand calculation.

---

### P0-4: Vapor enthalpy floor at h_sat_v (not h_sat_l)

**Bug:** `h_v` was clamped to `[h_sat_l, 4 MJ/kg]` instead of `[h_sat_v, 4 MJ/kg]`.
When h_v dropped below h_sat_v, IAPWS Region 2 returned negative density.

**What to test:**

```
Setup:
  Run one timestep with conditions that drive h_v downward
  (e.g., strong condensation: subcooled liquid with two-phase present)
Assert:
  h_v[i] >= h_sat_v(p[i]) for all cells, after every timestep

Setup:
  Initialize h_v = h_sat_v - 10000 (below floor) and run one step
Assert:
  After step, h_v >= h_sat_v (floor enforced)
```

**Existing coverage:** NONE. No test checks the enthalpy floor value. The test
TestPhaseReappearance::test_liquid_enthalpy_reset_on_reappearance checks finiteness
but not the actual bound.

---

### P0-5: Vapor density positivity

**Bug consequence:** h_v < h_sat_v caused rho_vapor to return negative values from
IAPWS Region 2 (which is undefined for h < h_g).

**What to test:**

```
Setup:
  Run a multi-step simulation with conditions that exercise two-phase transport
  (heated channel approaching boiling, blowdown, condensation)
Assert after EVERY timestep:
  rho_vapor(p[i], h_v[i]) > 0 for all cells with alpha > 0
  rho_liquid(p[i], h_l[i]) > 0 for all cells with alpha < 1

Property-level test:
  IAPWSIF97Properties::rho_vapor(p, h) for h < h_g(p)
  Should document that this returns garbage (negative or huge values)
  Verifies that the caller MUST enforce h_v >= h_sat_v before calling
```

**Existing coverage:** NONE. No test checks density positivity during simulation.

---

### P0-6: Liquid enthalpy bounds

**What to test:** After transport update, `h_l[i]` is clamped to `[1e4, h_sat_v(p[i])]`.
The upper bound `h_sat_v` is correct (liquid can be superheated up to steam enthalpy).
The lower bound `1e4` prevents unphysical sub-freezing.

```
Setup:
  Initialize h_l just above h_min = 1e4, run with strong cooling (q_wall < 0)
Assert:
  h_l[i] >= 1e4 for all i (floor holds)

Setup:
  Initialize h_l near h_sat_v, run with strong heating
Assert:
  h_l[i] <= h_sat_v(p[i]) (ceiling holds)
```

**Existing coverage:** NONE.

---

### P0-7: Superheated liquid enthalpy DECREASES toward saturation

**Bug manifestation:** With the sign bug, superheated liquid enthalpy INCREASED
instead of decreasing. A test checking this direction would have caught the bug.

```
Setup:
  N=5, uniform h_l = 850e3 (above h_sat_l=800e3), alpha=0.1, no wall heat
  H_i = 1e6 (strong interfacial HT), run 100 steps
Assert:
  h_l values have DECREASED (moved toward saturation)
  h_l < 850e3 for most cells (unless pinned by advection)
```

**Existing coverage:** TestVoidFractionEvolution::test_superheated_flashing checks
void growth but NOT enthalpy direction. The enthalpy could increase (wrong) while
void still grows (from other terms), so the test is not sufficient.

---

## P1: Conservation Laws, Physical Invariants, Limiting Cases

### P1-1: Per-cell mixture mass conservation (5-eq, per-timestep)

**What to test:** For the 5-equation model, the mixture mass balance is:
`d/dt[(1-alpha)*rho_l + alpha*rho_v] * V = mdot_in - mdot_out`

This is already tested in `TestTransientMassConservation5Eq::test_mass_balance_per_step`
but with loose tolerance (10% of flux). Should be tightened.

**Existing coverage:** Partial. Tolerance is too loose to catch small conservation violations.

---

### P1-2: Per-cell phasic mass conservation (5-eq)

**What to test:** The VAPOR mass equation specifically:
`d/dt[alpha*rho_v] * V = mdot_v_in - mdot_v_out + Gamma * V`

No test verifies the phasic (not mixture) mass balance. This would require
exposing phasic fluxes from the solver.

**Existing coverage:** NONE. This is a fundamental gap. The mixture mass balance
can be satisfied while the phasic split is wrong.

---

### P1-3: Mixture energy conservation (5-eq, quantitative)

**What to test:**
`d/dt[sum_k (alpha_k * rho_k * h_k)] * V = mdot*h boundary flux + q_wall + dp/dt*V`

The existing `TestTransientEnergyConservation5Eq::test_mixture_energy_balance` only
checks that `|dE| < 10 * |q_total| + 1e-3`. This is not a conservation test -- it's a
"not wildly wrong" test. A real conservation test would compute ALL terms on the RHS
and compare to the LHS.

**Existing coverage:** Effectively NONE (tolerance is meaningless).

---

### P1-4: Interfacial mass + energy balance consistency

**What to test:** Gamma from the closures drives both phasic mass and phasic
energy. The evaporated mass carries enthalpy. Verify that:
- Mass added to vapor (Gamma * V * dt) produces the correct change in alpha
- Energy removed from liquid (q_i_l * V * dt) matches the enthalpy change in liquid
- Energy added to vapor (q_i_v * V * dt + Gamma * h_v * V * dt) is consistent

**Existing coverage:** NONE.

---

### P1-5: Drift-flux phasic velocity split -- mdot_l + mdot_v = mdot_m

**What to test:** The function `split_phasic_flux` must return (mdot_l, mdot_v) that
sum to mdot_m. Test directly.

```
Setup:
  mdot_m = 5.0, alpha = 0.3, rho_l = 750, rho_v = 40, C_0 = 1.13, V_gj = 0.2, A = 0.01
  Call split_phasic_flux
Assert:
  mdot_l + mdot_v == pytest.approx(mdot_m, rel=1e-12)
```

Test over a sweep of alpha from 0.001 to 0.999.

**Existing coverage:** TestDriftFluxSplit::test_phasic_flux_sums_to_mixture runs
the full solver but does NOT call split_phasic_flux directly or verify the sum.
It only checks that the state "remains finite" and "mixture flow should be positive."

---

### P1-6: No-slip limit (C_0=1, V_gj=0) -- v_l = v_v = j

**What to test:** When there is no drift (C_0=1, V_gj=0), both phases must
move at the mixture volumetric flux velocity.

```
Setup:
  Call split_phasic_flux with C_0=1, V_gj=0, various alpha values
Assert:
  v_l = v_v = j = mdot_m / (rho_m * A)
  mdot_l = (1-alpha) * rho_l * v_l * A
  mdot_v = alpha * rho_v * v_v * A
  mdot_l + mdot_v = mdot_m exactly
```

**Existing coverage:** TestDriftFluxSplit::test_no_slip_phasic_split only checks
that the simulation "is stable" -- does not verify v_l = v_v.

---

### P1-7: Pressure bounds enforced by solver

**What to test:** solver.cpp lines 192-196: `p[i] = clamp(p[i], 700.0, 21.0e6)`.

```
Setup:
  Extreme boundary conditions that would drive pressure outside bounds
  e.g., p_in = 21.5e6, p_out = 21.0e6 (above ceiling)
Assert:
  p[i] <= 21.0e6 for all cells
```

**Existing coverage:** NONE. No test verifies the pressure clamp behavior.

---

### P1-8: Alpha bounds enforced by transport update

**What to test:** five_eq_model.cpp line 395: `alpha_new = clamp(alpha_new, 0.0, 1.0)`.

```
Setup:
  Extreme conditions: very strong heating to drive massive void generation
  Run for enough steps to push alpha toward 1
Assert:
  0 <= alpha[i] <= 1 for all cells, for all timesteps
```

**Existing coverage:** NONE explicitly. Several integration tests would likely fail
if this broke, but no test directly verifies the clamp.

---

### P1-9: Vapor density floor in void fraction update

**What to test:** five_eq_model.cpp lines 390-392:
`rv_new = max(rv_new, max(0.01 * rho_v_sat, 0.01))`.

This floor prevents division by near-zero vapor density in `alpha = (alpha*rho_v) / rv_new`.

```
Setup:
  Very low pressure where rho_v_sat is small
  Run with conditions that compute rv_new near the floor
Assert:
  rv_new used in alpha computation >= floor value
  No division-by-zero or extreme alpha values
```

**Existing coverage:** NONE.

---

### P1-10: Nucleation floor enforcement

**What to test:** five_eq_model.cpp lines 400-402: when `Gamma > 0`, enforce
`alpha >= 1e-3`. Also closures.hpp lines 114-116: when `T_l > T_sat` and
`alpha < alpha_nucleation`, set `alpha_eff = alpha_nucleation`.

```
Setup:
  Superheated liquid (h_l > h_sat_l), alpha = 0.0 initially
Assert:
  After one step with evaporating closure, alpha >= 1e-3 in affected cells
```

**Existing coverage:** TestNucleation::test_nucleation_creates_void_from_superheat
checks that void appears but not that it respects the exact floor value.

---

### P1-11: Phase-absent enthalpy reset

**What to test:** five_eq_model.cpp lines 424 and 459-460: when phase mass is
negligible (m_l <= 1e-12 or m_v <= 1e-12), enthalpy resets to saturation.

```
Setup:
  alpha = 0.0 everywhere (no vapor) → m_v ~ 0
Assert:
  h_v[i] == h_sat_v(p[i]) (reset to saturation)

Setup:
  alpha = 1.0 everywhere (no liquid) → m_l ~ 0
Assert:
  h_l[i] == h_sat_l(p[i]) (reset to saturation)
```

**Existing coverage:** TestPhaseReappearance checks finiteness but not exact saturation reset.

---

### P1-12: Wall heat split proportional to phase fraction

**What to test:** five_eq_model.cpp lines 412-414:
`q_wall_l = q_total * (1 - alpha)`, `q_wall_v = q_total * alpha`.

```
Setup:
  alpha = 0.3, q_wall = 1000 W
Assert:
  q_wall_l = 700, q_wall_v = 300

Setup:
  alpha = 0.0 (all liquid)
Assert:
  q_wall_l = q_wall, q_wall_v = 0

Setup:
  alpha = 1.0 (all vapor)
Assert:
  q_wall_l = 0, q_wall_v = q_wall
```

This requires exposing intermediate values or doing a controlled single-cell test.

**Existing coverage:** NONE directly (integrated into energy tests but never isolated).

---

## P2: Numerical Stability, Edge Cases, Error Handling

### P2-1: DriftFlux V_gj limiting at extreme alpha

**What to test:** closures.hpp lines 166-167: V_gj scaled by `4*alpha*(1-alpha)`.
At alpha=0 and alpha=1, V_gj should be exactly zero.

```
Setup:
  alpha = 0.0: V_gj should be 0
  alpha = 1.0: V_gj should be 0
  alpha = 0.5: V_gj should be maximum
Assert:
  V_gj(alpha=0) == 0
  V_gj(alpha=1) == 0
  V_gj(alpha=0.5) > V_gj(alpha=0.1)
```

**Existing coverage:** NONE.

---

### P2-2: DriftFlux V_gj at extreme density ratios

**What to test:** closures.hpp lines 150-153: drho clamped to 0.01,
rho_l^2 clamped to 1.0.

```
Setup:
  rho_l = rho_v (near critical point): drho ≈ 0
Assert:
  V_gj is small but finite (clamped drho = 0.01)

Setup:
  rho_l = 0 (should not happen but test defensive code)
Assert:
  No division by zero, V_gj is finite
```

**Existing coverage:** NONE.

---

### P2-3: Interfacial area a_i formula

**What to test:** closures.hpp line 125:
`a_i = max(4*alpha*(1-alpha), alpha)`.

```
Assert:
  a_i(0.0) = 0.0
  a_i(0.001) = max(0.003996, 0.001) = 0.003996
  a_i(0.5) = max(1.0, 0.5) = 1.0
  a_i(0.99) = max(0.0396, 0.99) = 0.99
  a_i(1.0) = max(0.0, 1.0) = 1.0
  Always >= 0
```

**Existing coverage:** NONE.

---

### P2-4: h_fg floor near critical point

**What to test:** closures.hpp line 101: `if (h_fg < 1.0) h_fg = 1.0`.

```
Setup:
  h_sat_l and h_sat_v nearly equal (near critical point)
Assert:
  Gamma remains finite (no division by zero)
```

**Existing coverage:** NONE.

---

### P2-5: split_phasic_flux at extreme alpha

**What to test:** five_eq_model.cpp lines 270-278:
- `alpha < 1e-6`: return (mdot_m, 0)
- `alpha > 1-1e-6`: return (0, mdot_m)

```
Setup:
  alpha_face = 0 → mdot_l = mdot_m, mdot_v = 0
  alpha_face = 1 → mdot_l = 0, mdot_v = mdot_m
  alpha_face = 1e-7 → same as alpha=0 branch
  alpha_face = 1-1e-7 → same as alpha=1 branch
Assert:
  Exact values as above
  mdot_l + mdot_v == mdot_m in all cases
```

**Existing coverage:** NONE.

---

### P2-6: split_phasic_flux rho_eff guard

**What to test:** five_eq_model.cpp lines 286-288: when `|rho_eff| < 1.0`,
it's replaced by `copysign(1.0, rho_eff)`.

```
Setup:
  Choose alpha, C_0, rho_l, rho_v such that rho_eff ~ 0
Assert:
  No division by zero, finite outputs
```

**Existing coverage:** NONE.

---

### P2-7: IAPWS rho_vapor with h_v < h_g(p) -- invalid input behavior

**What to test:** Calling `IAPWSIF97Properties::rho_vapor(p, h_v)` when
`h_v < h_g(p)` invokes `T_ph_R2(p, h_v)` which attempts Newton iteration
in Region 2 with an out-of-range input.

```
Setup:
  p = 7e6
  h_v = h_g(7e6) - 100e3  (100 kJ/kg below saturation)
  Call rho_vapor(p, h_v)
Assert:
  Document the return value (negative? huge? NaN?)
  This test SHOULD fail -- the result is non-physical
  Purpose: prove the solver MUST prevent this input
```

**Existing coverage:** NONE. test_iapws_cpp.py only tests valid inputs.

---

### P2-8: IAPWS rho_liquid with h_l > h_f(p) -- subcooled to two-phase

**What to test:** Calling `rho_liquid(p, h_l)` with `h_l > h_f(p)` uses
Region 1 Newton iteration, which may diverge or return wrong values.

```
Setup:
  p = 7e6, h_l = h_f(7e6) + 50e3 (above saturation)
Assert:
  Document behavior -- rho_liquid should still be callable
  (liquid can be superheated; T_ph_R1 stays in [273, 623])
```

**Existing coverage:** NONE.

---

### P2-9: SimpleFluid rho_vapor can return negative values

**What to test:** `rho_g(p) - A_G * (h_v - h_g(p))`. If h_v >> h_g(p),
this becomes negative.

```
Setup:
  p = 10e6, h_v = h_g(10e6) + rho_g(10e6)/A_G + 1  (just past zero crossing)
  = 2800e3 + 40/2e-5 + 1 = 2800e3 + 2e6 + 1 = 4.8e6
Assert:
  rho_vapor returns negative
  Purpose: verify floor is needed in production
```

**Existing coverage:** NONE.

---

### P2-10: Thomas algorithm numerical stability

**What to test:** solver.cpp solve_tridiagonal: verify diagonal dominance of
the tridiagonal system (necessary for Thomas algorithm stability).

```
Setup:
  Various conditions (subcooled, two-phase, high flow, low flow)
  After assemble_pressure_system, check:
Assert:
  |b[i]| >= |a[i]| + |c[i]| for all i (diagonal dominance)
  b[i] > 0 for all i
```

**Existing coverage:** NONE. The linearized mass conservation test indirectly
validates the solve but never checks diagonal dominance.

---

### P2-11: CFL condition check

**What to test:** solver.cpp lines 204-221: CFL warning is issued when
`dt > rho * V / mdot`.

```
Setup:
  High flow rate, large dt that violates CFL
Assert:
  Warning is issued (stderr capture)
  Solution still runs (CFL violation is warning, not error)
```

**Existing coverage:** NONE.

---

### P2-12: Zero flow rate edge case

**What to test:** p_in = p_out (no driving pressure).

```
Setup:
  p_in = p_out = 10e6, no wall heat
Assert:
  mdot = 0 everywhere
  All state variables remain finite
  No division by zero in enthalpy transport
```

**Existing coverage:** NONE.

---

### P2-13: Very small alpha (nucleation onset)

**What to test:** alpha = 1e-10, with non-zero vapor. Ensure numerical
stability of void fraction update.

```
Setup:
  alpha = 1e-10, h_l at saturation, h_v = h_sat_v, no heat
Assert:
  alpha remains >= 0
  No floating point catastrophic cancellation
  rho_v computation is stable
```

**Existing coverage:** NONE directly. TestNucleation exercises small alpha indirectly.

---

### P2-14: Pressure work term sign and magnitude

**What to test:** five_eq_model.cpp line 441:
`p_work_l = (1 - alpha) * V * dp/dt`. Sign should match physical expectation:
rising pressure compresses fluid, adding work to it.

```
Setup:
  dp > 0 (pressurizing): p_work should be positive (adds energy)
  dp < 0 (depressurizing): p_work should be negative
Assert:
  p_work_l has correct sign in both cases
  Magnitude matches (1-alpha) * V * (p_new - p_old)/dt
```

**Existing coverage:** NONE directly.

---

### P2-15: Phasic enthalpy advection flux sign

**What to test:** five_eq_model.cpp lines 439-440:
```
flux_l = mdot_l_in * (h_face_in - h_l_old) - mdot_l_out * (h_face_out - h_l_old)
```

With positive flow and uniform enthalpy, flux should be zero.
With positive flow and h_in > h_l_old, flux should add energy.

```
Setup:
  Uniform enthalpy, positive flow
Assert:
  flux_l ≈ 0 (within reconstruction accuracy)

Setup:
  h at inlet > h in cells, positive flow
Assert:
  flux_l > 0 (enthalpy is being advected in)
```

**Existing coverage:** NONE as isolated test.

---

## P3: Integration Tests Combining Multiple Components

### P3-1: 5-equation solver with IAPWS-IF97 properties

**What to test:** Full solver with IAPWS-IF97 (not SimpleFluid). This is an
INTEGRATION test that exercises property-solver coupling.

```
Setup:
  N=10, subcooled water at 7 MPa, heated channel
Assert:
  Reaches steady state (finite, monotonic enthalpy)
  Mass conservation within 1% of flux
  Enthalpy exits two-phase region if sufficiently heated
```

**Existing coverage:** test_iapws_cpp.py::TestIAPWSSimpleFluidCrossCheck only runs
HEM with IAPWS, not the 5-equation model.

---

### P3-2: 5-equation with inertial momentum

**What to test:** FiveEqModel + InertialMomentum (not AlgebraicMomentum).
The inertial momentum model is only tested in the Edwards blowdown context;
never tested with the 5-eq model in isolation.

```
Setup:
  Standard subcooled channel, InertialMomentum, 5-eq model
Assert:
  Reaches same steady state as algebraic momentum
  Pressure wave propagation during transient
```

**Existing coverage:** NONE for 5-eq + inertial combination.

---

### P3-3: 5-equation with critical flow

**What to test:** FiveEqModel + InertialMomentum + RansomTrapp critical flow.

```
Setup:
  Break BC at outlet, high upstream pressure
  Two-phase conditions at break
Assert:
  Flow rate limited to critical
  is_choked flag set correctly
```

**Existing coverage:** NONE for 5-eq + critical flow combination.

---

### P3-4: MUSCL reconstruction with 5-equation two-phase flow

**What to test:** MUSCL with two-phase (not just subcooled). The existing
TestMUSCL5Eq tests only run subcooled liquid (alpha=0).

```
Setup:
  N=10, heated channel transitioning from subcooled to two-phase
  MUSCL_Minmod reconstruction
Assert:
  No overshoot in enthalpy at phase boundary
  TVD property maintained for void fraction profile
  h_l and h_v both finite
```

**Existing coverage:** TestMUSCL5Eq only tests subcooled flow with 5-eq.

---

### P3-5: Depressurization transient (blowdown-like)

**What to test:** Start at high pressure, suddenly reduce p_out. The rapid
depressurization causes flashing throughout the pipe.

```
Setup:
  Initially steady at p=15 MPa, subcooled
  At t=0, reduce p_out to 1 MPa
Assert:
  Solution remains finite
  Pressure decreases monotonically toward outlet
  Void fraction increases as expected from flashing
  h_v stays above h_sat_v at each local pressure
  rho_vapor stays positive everywhere
```

**Existing coverage:** NONE (Edwards blowdown uses HEM model, not 5-eq).

---

### P3-6: Condensation transient (subcooled inlet into steam-filled pipe)

**What to test:** Pipe initially full of steam, subcooled liquid injected at inlet.
Condensation front should propagate downstream.

```
Setup:
  alpha_init = 0.99, h_l = h_sat_l, h_v = h_sat_v
  Subcooled inlet: h_in = h_sat_l - 100e3, alpha_in = 0.0
Assert:
  Void fraction decreases from inlet to outlet over time
  No negative densities during condensation
  Energy balance maintained
```

**Existing coverage:** TestVoidFractionEvolution::test_subcooled_condensation is
qualitative only (checks mean alpha decreased, not the transient profile).

---

### P3-7: Full parameter sweep over (pressure, quality, flow rate)

**What to test:** Systematic sweep to find unstable regions.

```
Sweep:
  Pressure: [1, 3, 7, 10, 15] MPa
  Quality at inlet: [0, 0.01, 0.1, 0.5, 0.9, 0.99]
  dp: [0.1, 0.5, 1.0] MPa (driving pressure difference)
Assert for each combination:
  Solution remains finite after 1000 steps
  alpha in [0, 1]
  Density positive everywhere
  No NaN/Inf in any state variable
```

**Existing coverage:** NONE. Tests use only 10 MPa and a handful of conditions.

---

### P3-8: Boundary condition types (WALL, BREAK)

**What to test:** BoundaryConditions with BCType::WALL (closed end) and
BCType::BREAK (with critical flow). Only tested via Edwards blowdown.

```
Setup:
  WALL at inlet, PRESSURE at outlet
Assert:
  mdot[0] = 0 always
  Fluid drains from pipe

Setup:
  PRESSURE at inlet, BREAK at outlet
Assert:
  Critical flow limiter activates when appropriate
```

**Existing coverage:** Only tested through Edwards blowdown (HEM), never with 5-eq.

---

## Summary Table

| ID | Priority | Component | What's Missing | Would Have Caught Bug? |
|----|----------|-----------|----------------|----------------------|
| P0-1 | P0 | closures | q_i_l sign convention | YES (Bug 1) |
| P0-2 | P0 | closures | Energy balance identity | YES (Bug 1) |
| P0-3 | P0 | closures | Gamma magnitude vs hand calc | YES (Bug 1) |
| P0-4 | P0 | five_eq_model | h_v floor at h_sat_v | YES (Bug 2) |
| P0-5 | P0 | five_eq_model | rho_vapor positivity | YES (Bug 2) |
| P0-6 | P0 | five_eq_model | h_l bounds enforcement | Partially |
| P0-7 | P0 | five_eq_model | Superheated h_l decreases | YES (Bug 1) |
| P1-1 | P1 | five_eq_model | Per-cell mass conservation (tight tol) | No |
| P1-2 | P1 | five_eq_model | Per-cell PHASIC mass conservation | No |
| P1-3 | P1 | five_eq_model | Quantitative energy conservation | No |
| P1-4 | P1 | five_eq+closures | Interfacial mass/energy consistency | No |
| P1-5 | P1 | five_eq_model | split_phasic_flux sum = mdot_m | No |
| P1-6 | P1 | five_eq_model | No-slip limit v_l = v_v | No |
| P1-7 | P1 | solver | Pressure bounds enforcement | No |
| P1-8 | P1 | five_eq_model | Alpha bounds enforcement | No |
| P1-9 | P1 | five_eq_model | Vapor density floor | No |
| P1-10 | P1 | five_eq+closures | Nucleation floor exact value | No |
| P1-11 | P1 | five_eq_model | Phase-absent enthalpy reset exact | No |
| P1-12 | P1 | five_eq_model | Wall heat split by phase fraction | No |
| P2-1 | P2 | closures | V_gj limiting at alpha=0,1 | No |
| P2-2 | P2 | closures | V_gj near critical density | No |
| P2-3 | P2 | closures | a_i formula exact values | No |
| P2-4 | P2 | closures | h_fg floor near critical | No |
| P2-5 | P2 | five_eq_model | split_phasic_flux extreme alpha | No |
| P2-6 | P2 | five_eq_model | split_phasic_flux rho_eff guard | No |
| P2-7 | P2 | iapws97 | rho_vapor invalid input behavior | No |
| P2-8 | P2 | iapws97 | rho_liquid above saturation | No |
| P2-9 | P2 | simple_fluid | rho_vapor negative for extreme h_v | No |
| P2-10 | P2 | solver | Thomas algorithm diagonal dominance | No |
| P2-11 | P2 | solver | CFL warning issued | No |
| P2-12 | P2 | solver | Zero flow rate edge case | No |
| P2-13 | P2 | five_eq_model | Very small alpha stability | No |
| P2-14 | P2 | five_eq_model | Pressure work term sign | No |
| P2-15 | P2 | five_eq_model | Advection flux sign | No |
| P3-1 | P3 | solver+iapws | 5-eq with IAPWS properties | No |
| P3-2 | P3 | solver+momentum | 5-eq with inertial momentum | No |
| P3-3 | P3 | solver+critical | 5-eq with critical flow | No |
| P3-4 | P3 | solver+muscl | MUSCL with two-phase flow | No |
| P3-5 | P3 | solver | Depressurization transient | No |
| P3-6 | P3 | solver | Condensation transient | No |
| P3-7 | P3 | solver | Parameter sweep stability | No |
| P3-8 | P3 | solver | WALL/BREAK BC types with 5-eq | No |

## Root Cause Analysis

The two bugs escaped because the test suite relied on **emergent behavior** (void
fraction grows, enthalpy increases) rather than **direct verification of individual
terms**. Specifically:

1. **No unit tests for closure formulas.** The DriftFluxClosures::compute() function
   is never called directly in any test. It is only exercised through the full solver,
   which makes it impossible to isolate sign errors from magnitude errors.

2. **No bounds-checking invariant tests.** After every timestep, the solver enforces
   bounds on h_l, h_v, alpha, and pressure. No test verifies these bounds are correct
   or that they hold.

3. **No property-validity tests.** No test ever checks that rho_vapor(p, h_v) returns
   a positive number for solver-computed states. The IAPWS tests in test_iapws_cpp.py
   only test known-good inputs.

4. **Qualitative vs quantitative assertions.** Tests like "alpha grows" or "enthalpy
   increases" can pass even when individual terms have the wrong sign, because the
   system has enough redundancy that other terms compensate partially.

## Recommended Implementation Order

1. **P0-1 through P0-7** (7 tests): Direct verification of the two fixed bugs.
   These should be written immediately and must pass before any further development.

2. **P1-5, P1-6** (phasic split): Requires exposing `split_phasic_flux` to Python.
   Fundamental property of the drift-flux model.

3. **P1-2, P1-3, P1-4** (conservation): Requires careful accounting. May need
   intermediate value exposure from C++.

4. **P2-1 through P2-6** (edge cases): Direct tests of guard clauses and clamps.
   Requires exposing closure and split functions to Python bindings.

5. **P3-1 through P3-8** (integration): Full solver tests with various configurations.
   Longest to run but highest coverage payoff.

## Classification Note

All tests in this document that use SimpleFluid are **verification** tests (mathematical
correctness of the numerical method with analytically tractable properties). Tests that
use IAPWS-IF97 with the solver are **integration** tests (property + solver coupling).
Neither type is validation (comparison against experimental data with acceptance criteria).
Validation requires experimental data and is out of scope for this gap analysis.

---

Files referenced:
- `/Users/rodrigo/git/OPAL/solver/two_phase/closures.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/five_eq_model.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/five_eq_model.cpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/flow_model.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/solver_state.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/boundary_conditions.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/phasic_properties.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/simple_fluid.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/iapws97.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/momentum.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/solver.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/solver.cpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/critical_flow.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/reconstruction.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/fluid_package.hpp`
- `/Users/rodrigo/git/OPAL/solver/tests/test_five_eq.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_two_phase.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_iapws_cpp.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_muscl.py`
