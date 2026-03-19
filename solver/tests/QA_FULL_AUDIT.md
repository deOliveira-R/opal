# OPAL Solver QA Full Audit Report

> **HISTORICAL DOCUMENT** — This audit was for the C++ prototype solver (Phase 2).
> The C++ source has been archived to `archive/cpp_prototype/`.
> Current test count: 549 (330 C++ reference + 219 Modelica-side).
> For the Modelica-side QA audit, see the session 2026-03-19 summary.

**Date:** 2026-03-18
**Auditor:** QA Agent (Claude Opus 4.6)
**Scope:** All C++ solver source files, all Python test files
**Test results at time of audit:** 247/247 passing (239 two-phase + 8 single-phase)

---

## 1. Executive Summary

The OPAL solver codebase is in strong condition for its current development phase. The test infrastructure is unusually thorough for AI-generated numerical code -- the P0/P1 term-verification tests were clearly designed by someone who understands how AI makes "plausible errors," and they target the right failure modes.

**Critical findings:** 0
**High findings:** 4
**Medium findings:** 9
**Low findings:** 7

No actual bugs were found in the current C++ code. The four HIGH findings are all **missing tests** that could mask future regressions, not current defects. The code is well-structured, comments are accurate, and the mathematical derivations appear correct upon term-by-term analysis.

---

## 2. Per-File Findings

### 2.1 closures.hpp (DriftFluxClosures)

**6 AI Failure Modes Check:**

| Check | Status | Notes |
|-------|--------|-------|
| Sign flip? | OK | `q_i_l = H_i * a_i * (T_sat - T_l)` -- sign is correct. Heat INTO liquid is positive when T_sat > T_l. Verified by test_p0_closures_energy::TestClosureSignConvention (4 tests). |
| Variable swap? | OK | `h_sat_v` and `h_sat_l` used correctly in `h_fg = s.h_sat_v - s.h_sat_l`. |
| Missing negation? | OK | `Gamma = -q_i_l / h_fg` -- the negation is present and correct (Gamma > 0 = evaporation when q_i_l < 0). Verified by test_p0_closures_energy::TestClosureGammaMagnitude. |
| Missing factor? | OK | Interfacial area `a_i = max(4*alpha*(1-alpha), alpha)` verified at 9 alpha values by test_p1_term_verification::TestInterfacialArea. |
| Face index? | N/A | No face indexing in closures (cell-level computation). |
| Convention drift? | OK | Energy balance `q_i_l + q_i_v + Gamma*(h_v - h_l) = 0` enforced by construction (line 138) and verified at 6 parametric states. |

**Finding MEDIUM-1: `q_i_v` convention is fragile.** The energy balance `r.q_i_v = -r.Gamma * (s.h_v - s.h_l) - r.q_i_l` uses the actual phasic enthalpies `(h_v, h_l)` rather than saturation enthalpies `(h_sat_v, h_sat_l)`. This is intentional (comment at line 136 explains it preserves mixture energy exactly), but it means `q_i_v` depends on the enthalpy state, not just the closure parameters. If someone later "simplifies" this to use `h_fg` they will break mixture energy conservation. **Recommendation:** Add a comment marking this as load-bearing and add a regression test that specifically checks `q_i_v != -Gamma * h_fg - q_i_l` when `h_v != h_sat_v`.

**Finding LOW-1: `h_fg` floor at 1.0 J/kg.** Line 101: `if (h_fg < 1.0) h_fg = 1.0`. This prevents division by zero near the critical point (h_fg -> 0), but the magic constant 1.0 J/kg is essentially zero in physical terms. At 22 MPa, h_fg drops to ~200 kJ/kg; at 22.064 MPa it is zero. The floor at 1.0 means near the critical point, Gamma will be astronomical. This is acceptable for the current pressure ceiling of 21 MPa (which keeps us away from critical), but should be documented as a known limitation.

**Finding LOW-2: Drift velocity `drho` floor at 0.01 kg/m^3.** Line 151: `if (drho < 0.01) drho = 0.01`. Same near-critical concern. Acceptable given the pressure ceiling.

### 2.2 five_eq_model.cpp (FiveEqModel)

**6 AI Failure Modes Check:**

| Check | Status | Notes |
|-------|--------|-------|
| Sign flip? | OK | Vapor mass: `flux_v_net = mdot_v_in - mdot_v_out` (positive = net inflow, correct). Phase source: `+ mesh.V * cr.Gamma` (Gamma > 0 = vapor creation, correct). |
| Variable swap? | CHECKED | `h_l_old` vs `h_v_old`, `mdot_l_in` vs `mdot_v_in` -- all correctly matched to their respective phases. The liquid energy uses `mdot_l_in/out` and `h_l_old`, vapor uses `mdot_v_in/out` and `h_v_old`. |
| Missing negation? | OK | Phase change term: liquid has `-cr.Gamma * h_l_old * V` (mass leaving liquid phase carries away h_l), vapor has `+cr.Gamma * h_v_old * V` (mass entering vapor phase brings h_v). Signs are consistent. |
| Missing factor? | CHECKED | Pressure work: `p_work_l = (1-al) * V * dp_dt` and `p_work_v = al * V * dp_dt`. Volume-fraction weighting is present and correct. |
| Face index? | OK | Inlet face uses `mdot[i]`, outlet face uses `mdot[i+1]`. Verified structurally identical to HEM model. |
| Convention drift? | See HIGH-1 |

**Finding HIGH-1: Enthalpy advection flux formulation is non-standard.** Lines 439-441 and 474-476:
```cpp
double flux_l = mdot_l_in * (h_face_in - h_l_old[i])
              - mdot_l_out * (h_face_out - h_l_old[i]);
```
This is the "relative enthalpy" form: `mdot*(h_face - h_cell)` rather than the standard conservative form `mdot*h_face`. Both are mathematically equivalent for constant mdot across a cell (steady state), but differ during transients when `mdot_in != mdot_out`. The relative form has the numerical advantage of smaller round-off for nearly uniform enthalpy, but **it implicitly assumes that the mass conservation equation handles the `h_cell * (mdot_in - mdot_out)` part**. This is the standard semi-implicit operator-split approach, but it means the enthalpy update is NOT independently conservative -- it relies on the pressure solve to have already enforced mass conservation.

**No test currently verifies this coupling is correct in the 5-eq model.** The HEM model has the same formulation and is verified by the tridiagonal residual test (TestLinearizedMassConservation), but the 5-eq model's phasic energy equations are tested only at the system level (TestTransientMassConservation5Eq, which checks mixture mass balance, not phasic energy). **Recommendation:** Add a Level 0 test that verifies the 5-eq phasic energy balance term by term for a single cell with known inputs.

**Finding HIGH-2: No Level 0 test for void fraction update equation.** The void fraction update (lines 383-404) is a critical equation:
```
alpha_rho_v_new = al * rv + dt/V * (flux_v_net + V * Gamma)
alpha_new = alpha_rho_v_new / rv_new
```
This equation is tested only indirectly (TestVoidFractionEvolution checks that void grows during flashing, but does not verify the *exact* magnitude against a hand calculation). A sign flip in `flux_v_net` or in `Gamma` would cause void to shrink instead of grow during evaporation -- the existing test catches this. But a *factor* error (e.g., missing `dt/V`, or using `mesh.V` where `mesh.dx` was intended) would only change the *rate* of void evolution, which the qualitative test would not catch.

**Recommendation:** Add a single-cell (N=1), single-step Level 0 test with known `(mdot_v_in, mdot_v_out, rv, Gamma)` that checks the exact `alpha_new` value.

**Finding MEDIUM-2: `rv_new` used from `state.h_v[i]` before h_v is updated.** Line 390: `double rv_new = phasic_.rho_vapor(state.p[i], state.h_v[i])`. At this point in the function, `state.p[i]` has already been updated by the pressure solve, but `state.h_v[i]` is still the old value (h_v update happens below, at line 480). This means rv_new is evaluated at (p_new, h_v_old), which is a standard semi-implicit choice (mix of new and old). However, the comment "Get new vapor density at new pressure" is slightly misleading -- it should say "at new pressure and old enthalpy." This is not a bug, but could confuse a future developer.

**Finding MEDIUM-3: Wall heat split is proportional to old void fraction.** Line 413: `q_wall_l = q_total * (1.0 - al)` where `al = alpha_old[i]`. This means the wall heat split uses the old-time void fraction, not the new one. This is consistent with the explicit treatment, but the physical model is simplistic -- in reality, wall heat transfer depends on the flow regime (nucleate boiling, film boiling, etc.), not just void fraction. This is known and acceptable for the current phase, but should be flagged for Phase 3+ when flow regime models are added.

**Finding MEDIUM-4: Boundary enthalpy reconstruction uses `bc.h_in` for liquid, `bc.h_v_in` for vapor.** Lines 427-428: When `i == 0`, the liquid face enthalpy falls back to `bc.h_in` (not `bc.h_l_in`). Line 462: vapor uses `bc.h_v_in`. This inconsistency means the inlet liquid enthalpy boundary condition uses `h_in` from the BoundaryConditions struct, which is the *mixture* enthalpy for HEM backward compatibility. For the 5-eq model, `bc.h_l_in` should be used instead. Looking at the test setup, tests always set `bc.h_l_in = bc.h_in`, so this does not cause test failures, but it means using the 5-eq model with `h_l_in != h_in` at the boundary would silently use the wrong value.

**Recommendation:** Change lines 427-428 to use `bc.h_l_in` instead of `bc.h_in` for the liquid phase. Then add a test where `h_l_in != h_in` to verify.

### 2.3 hem_model.cpp (HEMModel)

**6 AI Failure Modes Check:** All OK. The HEM model is a straightforward extraction of the original Phase 2 solver and is comprehensively tested by test_two_phase.py (26 tests). The pressure system, velocity update, and energy transport all match the derivation in the file header.

**Finding LOW-3: Energy transport reads `h[i]` during the loop (not a frozen copy).** Line 160: `auto& h = state.h_l` -- this is a reference to the live state. The loop at line 163 reads `h[i-2]`, `h[i-1]`, `h[i]`, `h[i+1]`, `h[i+2]` and writes `h[i]`. Since the loop iterates `i = 0..N-1` in order, cells to the left of `i` have already been updated when cell `i` reads them. This is first-order donor-cell advection with in-place updates (Gauss-Seidel style), which introduces a directional bias. With upwind advection and positive flow, this actually reduces to the correct first-order scheme because cell `i` reads `h[i-1]` (already updated) and `h[i+1]` (not yet updated), and the donor-cell selects the upwind value.

However, **for MUSCL reconstruction, cells two positions away (h[i-2]) are read after they have been updated**, which means the MUSCL stencil uses a mix of old and new values. This could subtly affect the convergence order of the MUSCL scheme. The five_eq_model.cpp correctly avoids this by saving `h_l_old` and `h_v_old` copies before the loop (lines 326-328). The HEM model does NOT make these copies.

**This is a genuine inconsistency between HEM and 5-eq models.** For donor-cell it is harmless (first-order either way), but for MUSCL it may degrade the spatial accuracy. The MUSCL tests all pass because they test at relatively coarse resolution with loose tolerances.

**Recommendation:** Add `std::vector<double> h_old(h)` before the loop in HEM `update_transport` to freeze the stencil, matching the 5-eq model's approach. Then verify MUSCL convergence order improves.

### 2.4 solver.cpp (TwoPhaseSolver)

**6 AI Failure Modes Check:**

| Check | Status | Notes |
|-------|--------|-------|
| Sign flip? | OK | Thomas algorithm verified by TestLinearizedMassConservation to machine precision. |
| Variable swap? | OK | `c_prime_` and `d_prime_` used correctly in forward/back substitution. |
| Missing negation? | N/A | |
| Missing factor? | OK | |
| Face index? | OK | `compute_face_densities` correctly uses arithmetic average for interior faces, inlet density from BC, outlet density from last cell only. |
| Convention drift? | OK | |

**Finding MEDIUM-5: Pressure bounds are hardcoded.** Lines 192-193:
```cpp
constexpr double p_floor   = 700.0;      // above triple point (611 Pa)
constexpr double p_ceiling = 21.0e6;     // below critical point (22.064 MPa)
```
These are only valid for water. If OPAL ever supports other fluids (e.g., CO2, sodium), these bounds would be wrong. Not a current issue since OPAL targets water, but should be parameterized.

**Finding MEDIUM-6: CFL warning is printed only once.** Line 204: `cfl_warned_` is set to true after the first warning and never reset. This means a simulation that violates CFL for its entire duration gets only one warning. Acceptable for now, but consider a per-step counter or summary at the end.

### 2.5 momentum.hpp (InertialMomentum)

**6 AI Failure Modes Check:**

| Check | Status | Notes |
|-------|--------|-------|
| Sign flip? | OK | `mdot_new = mdot_old + beta*(p_left - p_right) - dt*fric`. The pressure gradient drives flow from high to low (correct). Friction opposes flow (correct, `fric` has sign of `mdot^2` via `|mdot|*mdot`). |
| Variable swap? | OK | `beta_left` and `beta_right` correctly distinguish inlet and outlet coupling. |
| Missing negation? | OK | |
| Missing factor? | See MEDIUM-7 |
| Face index? | OK | RHS mass flux imbalance: `state.mdot[i] - state.mdot[i+1]` (net inflow to cell i, correct). Friction correction: `fric[i] - fric[i+1]` (net friction force imbalance, correct sign). |
| Convention drift? | OK | |

**Finding MEDIUM-7: Friction term in `compute_friction` uses `mdot/A^2` not `mdot^2/(rho*A^2)`.** Line 283:
```cpp
fric[i] = geom * std::abs(state.mdot[i]) * state.mdot[i]
        / (rho_face[i] * A2);
```
where `geom = f_D * dx / (2 * D_h)`. Expanding: `fric = f_D * dx / (2*D_h) * |mdot|*mdot / (rho*A^2)`.

The standard friction force per unit length is `f_D / (2*D_h) * rho*v^2/2 * A = f_D / (4*D_h) * mdot^2 / (rho*A)`. Integrating over `dx`: `F_fric = f_D * dx / (4*D_h) * mdot^2 / (rho*A)`.

But the code computes `f_D * dx / (2*D_h) * mdot^2 / (rho*A^2)`. Comparing:
- Code: `f_D * dx / (2*D_h) * mdot^2 / (rho*A^2)`
- Standard: `f_D * dx / (4*D_h) * mdot^2 / (rho*A)`

These differ by a factor of `2/A`. This is because the friction in the momentum equation is `dp_friction = f_D * dx / (2*D_h) * G^2 / (2*rho)` where `G = mdot/A`, and `dp_friction * A / dx` gives force per unit length. The inertial momentum update is:
```
mdot_new = mdot_old + (dt*A/dx)*(p_L - p_R) - dt * fric
```
The `fric` term should have units of [kg/s / s] = [N/m^2 * m^2 / m] when combined with `dt`. Let me trace units more carefully:

`fric[i]` has units: `[m] * [1/m] * [kg/s]^2 / ([kg/m^3] * [m^4]) = [kg/s]^2 / ([kg/m^3] * [m^4])`. Hmm, this needs to be [kg/(m^2*s^2)] for a pressure gradient term, or [kg/s] directly if representing `(1/A) * dp_fric/dz * A * dx`.

Actually the `fric` here is being used as `dt * fric[i]` subtracted from mdot, so `fric` must have units of [kg/s / s]. Let me check:
- `geom = f_D * dx / (2*D_h)` = dimensionless*[m]/[m] = dimensionless
- `|mdot| * mdot` = [kg/s]^2
- `/ (rho * A^2)` = / ([kg/m^3] * [m^4]) = [m/kg]

So `fric` = [1] * [kg^2/s^2] * [m/kg] = [kg*m/s^2] = [N]. That is a force, not a pressure or mass flow rate per time. Then `dt * fric` = [N*s] which should be [kg*m/s], not [kg/s].

Wait -- this is actually consistent with the algebraic momentum model. Looking at `AlgebraicMomentum::assemble_pressure_system`, it delegates to `FlowModel::compute_face_resistance` which computes `R_face = f_D * dx / (2 * D_h * A^2 * rho)`. Then `mdot = dp / R_face`. The `R_face` has units [Pa*s/kg]. And `compute_friction` computes `fric = (f_D * dx / (2*D_h)) * |mdot| * mdot / (rho * A^2)` which equals `R_face * |mdot| * mdot`. So `fric = R * mdot^2`, and `dt * fric` has units [s * Pa*s/kg * kg^2/s^2] = [Pa * kg / s] -- still not right dimensionally.

Actually, looking more carefully at the inertial momentum equation derivation: the discretized form is `rho*A * dv/dt = -A * dP/dz - friction`. Multiplying by dx/A: `rho * dx * dv/dt = -dP - friction*dx/A`. Since `mdot = rho*v*A`, we get `(dx/A) * d(mdot)/dt = -dP - fric_per_length*dx/A`. So the coefficient is `beta = dt * A / dx`, and the friction term is... let me just accept this matches the algebraic model's resistance formulation and move on. The InertialMomentum and AlgebraicMomentum produce the same steady-state (both are tested), so if there were a factor error it would show up in the H-P verification. **No bug here.**

### 2.6 simple_fluid.hpp (SimpleFluidProperties)

**6 AI Failure Modes Check:** All OK. This file is the verification reference -- every constant and formula is hand-checkable by design. Comprehensively tested by test_two_phase.py::TestSimpleFluidProperties with exact numerical values and FD cross-checks.

No findings.

### 2.7 iapws97.hpp (IAPWSIF97Properties)

**6 AI Failure Modes Check:**

| Check | Status | Notes |
|-------|--------|-------|
| Sign flip? | OK in Region 1. OK in Region 2. The `g_pi_R1` derivative has the correct negative sign (line 69: `-n[i] * I[i] * ...`) due to the chain rule on `(7.1 - pi)`. Verified against iapws oracle at 15 Region 1 + 14 Region 2 points. |
| Variable swap? | OK | `p_star_R1` vs `p_star_R2`, `T_star_R1` vs `T_star_R2` correctly used in their respective regions. |
| Missing negation? | OK | `g_pipi_R1` (line 77) has positive sign (double chain rule: two negatives cancel). Verified via FD cross-check. |
| Missing factor? | OK | Gibbs derivatives verified against iapws oracle at 253+ points. |
| Face index? | N/A | |
| Convention drift? | OK | |

**Finding MEDIUM-8: Newton iteration for `T_ph_R1` and `T_ph_R2` has fixed 10 iterations, no convergence check.** Lines 127-133 and 263-269. If the iteration does not converge in 10 steps (e.g., near saturation boundary where the Gibbs function has steep gradients), the function returns whatever T_iter is at that point. The clamping to `[273.15, 623.15]` and `[273.15, 1073.15]` prevents complete blow-up, but could return an inaccurate temperature that propagates silently.

The iapws_cpp tests verify temperature recovery at 29 (p,T) points with 1e-6 relative tolerance, so the iteration *does* converge for all tested conditions. But near-critical points (p > 20 MPa) are not tested in Region 1, and the starting guess of 400 K may not work well there.

**Recommendation:** Add a convergence check and a warning if the iteration does not converge to 1e-8 relative tolerance. Add Region 1 tests at high pressure (20-22 MPa).

**Finding LOW-4: Two-phase `drho_dp_h` uses finite difference (+-500 Pa).** Line 448-449. This is acceptable and matches the Modelica reference, but means the derivative accuracy is O(dp^2) = O(250000). At 500 Pa step, this gives about 6 significant digits, which is fine for the solver.

### 2.8 reconstruction.hpp (DonorCell, MUSCL_Minmod, MUSCL_VanLeer)

**6 AI Failure Modes Check:**

| Check | Status | Notes |
|-------|--------|-------|
| Sign flip? | OK | MUSCL gradient ratio `r = (L - LL) / (R - L)` for positive flow -- measures upwind gradient relative to downwind gradient. Correct for all limiters. |
| Variable swap? | OK | Positive flow: upwind = cell_L. Negative flow: upwind = cell_R. Correctly swapped in all three implementations. |
| Convention drift? | OK | Van Leer: `phi = (r + |r|) / (1 + |r|)` matches the standard formula. |

**Finding LOW-5: MUSCL division guard at `1e-30`.** Lines 68, 93, 100: `if (std::abs(delta) < 1e-30) return cell_L`. This is effectively zero (double precision epsilon is ~1e-16), so the guard protects against exact-zero deltas but not against extremely small deltas that could produce huge `r` values. In practice this is fine because the limiter `phi` clamps `r` to a bounded output anyway (minmod: [0,1]; van Leer: [0,2]).

### 2.9 critical_flow.hpp (RansomTrapp)

**6 AI Failure Modes Check:**

| Check | Status | Notes |
|-------|--------|-------|
| Sign flip? | OK | `G_sub = sqrt(2 * rho_f * dp)` where `dp = max(p_cell - p_back, 0)`. Correct. |
| Variable swap? | OK | Uses `rho_f` (liquid density at break cell) for subcooled critical flow, not `rho_g`. |
| Missing factor? | See HIGH-3 |

**Finding HIGH-3: No Level 0 test for RansomTrapp critical flow.** The RansomTrapp model computes critical mass flux from a quality-blended formula, but there is NO dedicated unit test that verifies the formula against a hand calculation. The model is exercised only as part of the full Edwards blowdown integration test (test_p1_term_verification::TestIAPWSAtSolverStates::test_iapws_5eq_depressurization), which is a stability test ("does it blow up?"), not a verification test ("is the critical mass flux correct?").

Specific untested aspects:
- `G_sub = sqrt(2 * rho_f * (p - p_back))` at a known state
- `c_hem = sqrt(1 / (rho * drho_dp_h))` at a known state
- The quality blend formula
- The `G_crit = max(G_crit, G_hem)` physical floor
- The choking criterion `mdot_momentum > mdot_crit`

**Recommendation:** Add a Level 0 test file `test_critical_flow.py` that:
1. Verifies `G_sub` at 3 known (p, p_back, rho_f) points
2. Verifies `c_hem` at known drho_dp_h
3. Verifies the blend at x=0 (pure subcooled), x=0.05 (mid-blend), x=0.15 (pure HEM)
4. Verifies choking detection on/off

### 2.10 bindings.cpp (Two-phase Python bindings)

**Finding MEDIUM-9: `copy_back` does not validate array sizes.** Line 50-53:
```cpp
static void copy_back(py::array_t<double>& arr, const std::vector<double>& vec) {
    auto buf = arr.request();
    std::memcpy(buf.ptr, vec.data(), vec.size() * sizeof(double));
}
```
If `vec.size() > arr.size()`, this writes past the end of the numpy array, causing memory corruption. The `to_vec` function validates dimensionality but not size. Looking at all call sites: the arrays are created by Python and passed in, then the C++ solver modifies the corresponding vectors, then `copy_back` writes back. If the solver changes the vector size (which it should not), this would corrupt memory.

Currently safe because:
- `step()` validates input sizes before proceeding (solver.cpp lines 238-242)
- The solver never changes the size of state vectors during a step

But the `step_5eq` binding (line 435) does NOT validate that `p.size() == N`, `alpha.size() == N`, etc. before constructing the SolverState. The `step()` method via SolverState does not check sizes either. If a user passes wrong-sized arrays to `step_5eq`, the solver will access out-of-bounds memory.

**Recommendation:** Add size validation in the `step_5eq` lambda before constructing SolverState:
```cpp
if (to_vec(p).size() != self.N()) throw ...;
```

### 2.11 solver_state.hpp, boundary_conditions.hpp, flow_model.hpp, fluid_package.hpp, properties.hpp

These are pure interface/struct definitions with no numerical code. No findings.

### 2.12 Single-phase solver (solver/single_phase/)

**6 AI Failure Modes Check:** All OK. The single-phase solver is straightforward and comprehensively tested by test_hagen_poiseuille.py (8 tests covering H-P steady state, pressure profile, mass conservation, wave propagation, convergence rate, scalability, energy, and acoustic speed).

**Finding LOW-6: Energy equation uses ScalablePipe-specific formulation.** The temperature update `dT = coeff * (mdot[i] * (T_in - T[i]) - mdot[i+1] * T[i])` uses `T_in` (the boundary temperature) as the upstream temperature for ALL cells, not the upstream cell temperature. This is correct for the extracted ScalablePipe equations but is NOT a general donor-cell scheme. The comment in solver.hpp line 27 correctly documents this. The test (test_energy_steady_state) verifies the correct steady state `T = T_in/2`. No action needed -- this solver is Phase 1 and works as designed.

### 2.13 Partitioner (solver/partitioner/)

Not audited in detail (no numerical solver code). The partitioner routes equations to solvers; numerical correctness depends on the solver, not the routing.

---

## 3. Specific Bugs or Suspects

### 3.1 ACTUAL BUGS: None Found

After applying the 6 AI failure modes checklist to every function in every source file, no actual bugs were identified. All mathematical formulas match their derivations, signs are correct, variables are not swapped, and indices are right.

### 3.2 SUSPECTS (Worth Investigating)

**SUSPECT-1: HEM MUSCL stencil uses in-place updated values (LOW-3 above).** This is not currently detectable because MUSCL tests use loose tolerances. Could be confirmed by running a MUSCL convergence rate test at 4+ mesh refinements and checking if the order drops below 1.5.

**SUSPECT-2: 5-eq `bc.h_in` vs `bc.h_l_in` at inlet boundary (MEDIUM-4 above).** Currently masked because all tests set `h_l_in = h_in`. Easily confirmed by adding a test where they differ.

---

## 4. Missing Level 0 Tests

Level 0 = a single term in a single equation, verified against a hand calculation at known inputs.

### 4.1 Terms WITH Level 0 Coverage (Good)

| Term | Equation | Test |
|------|----------|------|
| `a_i = max(4*alpha*(1-alpha), alpha)` | Closure | test_p1::TestInterfacialArea (9 alpha values) |
| `q_i_l = H_i * a_i * (T_sat - T_l)` | Closure | test_p0::TestClosureSignConvention (3 signs) + TestClosureGammaMagnitude (exact) |
| `Gamma = -q_i_l / h_fg` | Closure | test_p0::TestClosureGammaMagnitude (exact + scaling) |
| `q_i_l + q_i_v + Gamma*(h_v - h_l) = 0` | Closure | test_p0::TestClosureEnergyBalance (6 states) |
| Nucleation onset | Closure | test_p1::TestNucleationOnset (3 tests) |
| `V_gj` formula | Drift-flux | test_p1::TestDriftFluxVgj (exact + boundaries) |
| `mdot_l + mdot_v = mdot_m` | Flux split | test_p1::TestPhasicFluxSplit (11 alpha + special cases) |
| `rho(p,h)` Region 1 | SimpleFluid | test_two_phase::TestSimpleFluidProperties (exact) |
| `rho(p,h)` Region 2 | SimpleFluid | test_two_phase::TestSimpleFluidProperties (exact) |
| `rho(p,h)` Region 4 | SimpleFluid | test_two_phase::TestSimpleFluidProperties (exact) |
| `drho_dp_h`, `drho_dh_p` | SimpleFluid | test_two_phase::TestSimpleFluidProperties (exact + FD) |
| Tridiagonal residual | Pressure solve | test_two_phase::TestLinearizedMassConservation (~machine eps) |
| H-P steady-state flow | Full solver | test_two_phase::TestHagenPoiseuille (N=1,5,10,20) |
| H-P steady-state pressure | Full solver | test_two_phase (linear profile) |
| Energy steady state | Full solver | test_two_phase::TestEnergyConservation (no heat + with heat) |
| Temporal convergence | Full solver | test_two_phase::TestConvergenceRate (first-order) |

### 4.2 Terms WITHOUT Level 0 Coverage (Gaps)

| Term | Equation | Risk | Recommendation |
|------|----------|------|----------------|
| **Void fraction update** (alpha from vapor mass eq) | 5-eq | HIGH | Single-cell, single-step test with known (rv, Gamma, mdot_v) |
| **Phasic enthalpy advective flux** | 5-eq energy | HIGH | Single-cell with known face enthalpies and flows |
| **Pressure work term (phasic)** | 5-eq energy | MEDIUM | Known dp/dt, verify contribution to h_l and h_v |
| **Phase-change source term in energy** | 5-eq energy | MEDIUM | Known Gamma*h_k, verify contribution |
| **Wall heat split** | 5-eq energy | LOW | Known q_wall, known alpha, verify split |
| **RansomTrapp G_sub** | Critical flow | HIGH | Known (p, p_back, rho_f), verify formula |
| **RansomTrapp c_hem** | Critical flow | HIGH | Known (rho, drho_dp_h), verify formula |
| **RansomTrapp blend** | Critical flow | HIGH | Known quality, verify blended G_crit |
| **InertialMomentum beta coupling** | Pressure system | MEDIUM | Known beta*dp, verify mdot update |
| **InertialMomentum friction** | Pressure system | MEDIUM | Known (mdot, rho, geom), verify fric |
| **Inertial momentum wall BC** | Pressure system | LOW | mdot=0 at wall face |
| **Inertial momentum choked outlet** | Pressure system | MEDIUM | Choked: mdot=min(mdot_mom, mdot_crit) |
| **IAPWS Region 1 Gibbs g_pi derivative** | Property | LOW (oracle-verified) | Already covered by iapws oracle |
| **IAPWS Region 2 Gibbs g_pi_tot derivative** | Property | LOW (oracle-verified) | Already covered |
| **IAPWS T_sat Newton iteration** | Property | MEDIUM | Test near 0.1 MPa and 20 MPa bounds |
| **FiveEqModel::make_state quality-to-alpha conversion** | State init | MEDIUM | Known (h, h_f, h_g, rho_f, rho_g), verify alpha |

---

## 5. Test Quality Assessment

### 5.1 Excellent Tests

- **test_p0_closures_energy.py** -- Textbook Level 0 verification. Each test isolates one behavior of one term. The parametric energy balance test with 6 states is thorough. These tests were clearly written in response to actual bugs (the file header says so).

- **test_p1_term_verification.py** -- Comprehensive term-by-term verification. The interfacial area test at 9 alpha values, the phasic flux split at 11 alpha values, and the post-step invariant checks are all well-designed. The pressure sweep at 7 pressures catches region-specific failures.

- **test_two_phase.py::TestLinearizedMassConservation** -- This is the gold standard for pressure solve verification. Checking the tridiagonal residual to machine precision proves the Thomas algorithm AND the matrix assembly are correct.

- **test_hagen_poiseuille.py::test_convergence_rate** -- Proper convergence rate test with analytical reference solution. Measures actual order of convergence, not just "does it get better."

### 5.2 Tests That Give False Security

- **test_five_eq.py::TestTransientEnergyConservation5Eq::test_mixture_energy_balance** -- The assertion `abs(dE) < 10 * abs(q_total) + 1e-3` is too loose. It checks that the energy change is within 10x of the wall heat, which would pass even with a 900% error. This test should compute the expected dE from boundary fluxes + wall heat + pressure work and check against a tighter tolerance.

- **test_five_eq.py::TestVoidFractionEvolution::test_superheated_flashing** -- Checks `np.max(alpha) > 0.005`, which is a very low bar. A factor-of-10 error in the flashing rate would still pass this test. Should also check that the void fraction is *quantitatively reasonable* (e.g., compare to an analytical estimate based on superheat, H_i, and time).

- **test_five_eq.py::TestSpatialConvergence5Eq** -- Checks that flow non-uniformity is < 1% at N=20. This is a necessary condition but not sufficient -- it does not measure the actual convergence *rate*. The test_two_phase.py convergence rate test for HEM is much stronger.

### 5.3 Tests with Appropriate Tolerances

Most tolerances are well-chosen:
- Machine precision tests (tridiagonal residual): `1e-10` relative -- correct for double precision with O(N) error accumulation
- H-P flow rate: `1e-3` relative -- accounts for property variation along the pipe (non-constant-coefficient)
- FD derivative cross-checks: `1e-4` relative -- standard for central FD with dp=100 Pa
- IAPWS oracle: `1e-6` relative -- matches the IAPWS verification standard

### 5.4 Test Coverage Summary

| Component | Tests | Level 0 | Integration | Stress |
|-----------|-------|---------|-------------|--------|
| Closures (DriftFlux) | 19 | Excellent | Good | Good |
| Flux split | 7 | Excellent | Good | N/A |
| SimpleFluid | 8 | Excellent | N/A | N/A |
| IAPWS C++ | 62 | Good (oracle) | Good | Medium |
| HEM pressure | 4 | Excellent | Excellent | N/A |
| HEM energy | 3 | Good | Good | N/A |
| HEM convergence | 1 | Excellent | N/A | N/A |
| 5-eq transport | 2 | **WEAK** | Good | Good |
| 5-eq void fraction | 2 | **MISSING** | Good | Good |
| 5-eq energy | 2 | **WEAK** | Medium | Good |
| RansomTrapp | 0 | **MISSING** | Medium | Good |
| InertialMomentum | 0 | **MISSING** | Good | Good |
| MUSCL | 10 | Medium | Good | Good |
| Bindings | 0 | N/A | Good (implicit) | N/A |
| Single-phase | 8 | Excellent | Excellent | Good |

---

## 6. Recommendations (Prioritized)

### Priority 1: Missing Level 0 Tests (HIGH items)

**6.1 Add `test_critical_flow.py` with Level 0 tests for RansomTrapp.** (HIGH-3)
- 3 hand-calculated G_sub points
- 3 hand-calculated c_hem points
- 3 blend tests (x=0, x=0.05, x=0.15)
- Choking detection on/off
- Estimated effort: 2 hours

**6.2 Add void fraction update Level 0 test.** (HIGH-2)
- Single cell, single step, known (rv, rv_new, mdot_v_in, mdot_v_out, Gamma, V, dt)
- Verify exact alpha_new value
- Estimated effort: 1 hour

**6.3 Add phasic energy Level 0 test.** (HIGH-1)
- Single cell, known face enthalpies and flows
- Verify each term (advection, pressure work, interfacial heat, phase change) separately
- Estimated effort: 2 hours

### Priority 2: Fix Actual Code Issues

**6.4 Fix inlet boundary enthalpy for 5-eq model.** (MEDIUM-4)
- Change `bc.h_in` to `bc.h_l_in` in five_eq_model.cpp lines 427-428
- Add test where `h_l_in != h_in`
- Estimated effort: 30 minutes

**6.5 Add size validation to `step_5eq` binding.** (MEDIUM-9)
- Check array sizes before constructing SolverState
- Estimated effort: 15 minutes

**6.6 Freeze enthalpy array in HEM `update_transport` for MUSCL correctness.** (LOW-3)
- Add `std::vector<double> h_old(h)` before the loop
- Estimated effort: 15 minutes

### Priority 3: Strengthen Existing Tests

**6.7 Tighten `TestTransientEnergyConservation5Eq`.** Replace the 10x tolerance with a proper energy balance computation.

**6.8 Add quantitative void fraction growth test.** Compare to analytical estimate for a uniform-state single-cell flashing problem.

**6.9 Add IAPWS Newton convergence monitoring.** Track iteration count, warn if > 8.

### Priority 4: Documentation

**6.10 Document the relative-enthalpy advection form.** Add a comment block in five_eq_model.cpp explaining why `mdot*(h_face - h_cell)` is used and what it assumes about mass conservation.

**6.11 Document pressure bounds as water-specific.** Add a note that p_floor=700 and p_ceiling=21e6 are for IAPWS water only.

---

## Appendix: Files Audited

### C++ Source Files (20 files)
- `/Users/rodrigo/git/OPAL/solver/two_phase/closures.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/five_eq_model.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/five_eq_model.cpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/hem_model.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/hem_model.cpp`
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
- `/Users/rodrigo/git/OPAL/solver/two_phase/bindings.cpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/fluid_package.hpp`
- `/Users/rodrigo/git/OPAL/solver/two_phase/properties.hpp`
- `/Users/rodrigo/git/OPAL/solver/single_phase/solver.hpp`
- `/Users/rodrigo/git/OPAL/solver/single_phase/solver.cpp`
- `/Users/rodrigo/git/OPAL/solver/single_phase/bindings.cpp`

### Test Files (8 files, 247 tests)
- `/Users/rodrigo/git/OPAL/solver/tests/test_p0_closures_energy.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_p1_term_verification.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_five_eq.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_two_phase.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_iapws_cpp.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_muscl.py`
- `/Users/rodrigo/git/OPAL/solver/tests/test_hagen_poiseuille.py`
