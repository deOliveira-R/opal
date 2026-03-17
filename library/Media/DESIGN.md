# Media Package Design Decisions

This file records the *why* behind design choices made in `library/Media/`.
The code shows *what*; this file shows *why*.

---

## Region selection (Regions 1, 2, 4 only; Regions 3 and 5 deferred)

**Context:** Implemented during OPAL Phase 1.5 (Option A pull-forward, 2026-03-16)
before Phase 2 two-phase solver. The immediate driver is Phase 2's need for
`∂ρ/∂p|h` and `∂ρ/∂h|p` as explicit algebraic expressions.

**Options considered:**
1. Implement all five IAPWS-IF97 regions up-front.
2. Implement Regions 1, 2, 4 only (minimum viable set for LWR primary circuit).
3. Use MSL StandardWater (already exists) — rejected by FM3.

**Decision:** Option 2.
- Region 1 covers subcooled water at all LWR primary-circuit conditions
  (T < 623 K at typical operating pressures of 7–16 MPa).
- Region 2 covers superheated steam in the secondary circuit (turbine, steam
  generator outlet side).
- Region 4 covers two-phase flow in the primary circuit during accident
  scenarios (LOCA, etc.) — the core use case for Phase 2.
- Region 3 (near-critical, T > 623 K at high p) is not reached in normal
  or most accident LWR conditions; implementing it requires Helmholtz-energy
  formulation (different mathematical structure) that adds significant
  complexity disproportionate to Phase 2 scope.
- Region 5 (T > 1073 K) is irrelevant to water-cooled reactors.

**What fails at the region boundaries:**
- If p/T wanders into Region 3 (e.g. p > 16.5 MPa AND T > 623 K), the
  current code will silently use Region 1 or 2 extrapolation, giving wrong
  answers. A guard assertion `assert(T < 620 or p < 16e6, "Region 3 not
  implemented")` should be added to Water.mo before production use.
- Region 5 failure mode is identical.

**Risks / revisit triggers:**
- If OPAL is extended to supercritical reactors (SCWRs), Region 3 is required.
- If accident analysis approaches the critical point (p > 20 MPa, T > 620 K)
  continuously, add Region 3 before using results for licensing.

---

## `sum()` over coefficient arrays vs. explicit scalar expressions

**Context:** IAPWS-IF97 polynomial sums have 34 (Region 1) or 9+43 (Region 2)
terms. OpenModelica must be able to symbolically differentiate these sums when
doing BLT analysis and equation flattening.

**Options considered:**
1. `sum(n[i] * (7.1 - pi)^I[i] * (tau - 1.222)^J[i] for i in 1:34)` — array form.
2. Explicit scalar expression: `n[1]*(7.1-pi)^0*(tau-1.222)^(-2) + n[2]*... + ...`
   (34 terms written out).
3. For-loop over index.

**Decision:** Option 1 (array form with `sum()` iterator).
- OpenModelica's frontend reduces `sum(f(i) for i in 1:N)` to a single
  algebraic expression during flattening (scalar unrolling) — confirmed
  by testing on a 5-term example during feasibility.
- Option 2 would be 34–43 lines of nearly identical code per function —
  unreadable and error-prone (copy-paste errors in exponents are hard to spot).
- Option 3 (for-loop) requires an accumulator variable and is less amenable
  to symbolic differentiation.

**What was tested:**
- TODO: The extraction transparency check in `tests/verify_if97.py` (test 3)
  explicitly verifies that no OPAQUE markers appear in the extracted XML.
  This must pass before Phase 2 integration.

**Risks / revisit triggers:**
- If OpenModelica fails to inline the `sum()` expression (shows `OPAQUE` in
  extracted XML or fails to differentiate symbolically), switch to Option 2
  for the affected functions. Region-specific fallback is acceptable.

---

## ∂ρ/∂p|h and ∂ρ/∂h|p derivation path

**Context:** The Phase 2 semi-implicit pressure scheme needs these two
derivatives as explicit functions of (p, h) evaluated at the *new* time level.

**Full chain-rule derivation:**

Starting from the IAPWS-IF97 Gibbs representation for specific volume:

```
v(p, T) = (R T / p) · π · g_π(π, τ)
```

where `π = p/p*`, `τ = T*/T`.

**Step 1: partials in (p, T) basis**

```
∂v/∂p|_T = (R T / p) · [∂(π g_π)/∂π · ∂π/∂p]
         = (R T / p) · [g_π + π g_ππ] / p*
         = (R T / (p · p*)) · (g_π + π g_ππ)

Wait — more carefully:
v = (R T / p) · π · g_π  =  (R T / p*) · g_π   [since π/p = 1/p*]

∂v/∂p|_T = (R T / p*) · g_ππ · (∂π/∂p) = (R T / p*²) · g_ππ
         = (R T / (p · p*)) · π · g_ππ · (1/π) · (p*/p)
```

Equivalently, using the reduced form `v = R T π g_π / p`:

```
∂v/∂p|_T = R T [g_π + π g_ππ (1/p*)] / p  −  R T π g_π / p²
         = (R T / p²) [π g_π + π² g_ππ − π g_π]
         = (R T π² / p²) g_ππ   ÷ p*  [dimensional]
         = (R T / (p · p*)) · π² · g_ππ
```

```
∂v/∂T|_p = (R / p) π g_π  +  (R T / p) π g_πτ · (dτ/dT)
         = (R π / p) [g_π − τ g_πτ]        [since dτ/dT = −T*/T² = −τ/T]
```

```
∂ρ/∂p|_T = −ρ² · ∂v/∂p|_T = −ρ² · (R T π² / (p · p*)) · g_ππ
∂ρ/∂T|_p = −ρ² · ∂v/∂T|_p = −ρ² · (R π / p) · (g_π − τ g_πτ)
```

**Step 2: h partial derivatives**

```
h(p, T) = R T τ g_τ(π, τ)

c_p = ∂h/∂T|_p = R τ g_τ + R T (dτ/dT) g_τ + R T τ g_ττ (dτ/dT)
    = R τ g_τ − R τ g_τ + R T τ g_ττ (−τ/T)
    = −R τ² g_ττ

∂h/∂p|_T = R T τ · g_τπ · (∂π/∂p)
          = R T τ g_πτ / p*
          = (R T π τ / p) g_πτ   [since π/p* = 1/p]
```

**Step 3: change of basis (p,T) → (p,h)**

At constant p:
```
dh = c_p dT   →   ∂T/∂h|_p = 1/c_p
∂ρ/∂h|_p = (∂ρ/∂T|_p) / c_p
```

At constant h:
```
0 = ∂h/∂p|_T dp + c_p dT   →   ∂T/∂p|_h = −(∂h/∂p|_T) / c_p
∂ρ/∂p|_h = ∂ρ/∂p|_T + ∂ρ/∂T|_p · ∂T/∂p|_h
          = ∂ρ/∂p|_T − ∂ρ/∂T|_p · (∂h/∂p|_T) / c_p
```

**Risks / revisit triggers:**
- If Phase 2 shows pressure oscillations or divergence, check sign and magnitude
  of these derivatives against central finite differences (test 2 in verify_if97.py
  does this at 5 state points).
- The two-phase derivative `∂ρ/∂p|h` in Region 4 currently uses a ±500 Pa
  finite difference — adequate for Phase 2 but should be replaced with the
  analytical Clausius-Clapeyron expression before production:
  ```
  ∂ρ/∂p|h,2phase = analytical expression using dp_sat/dT, hfg, vfg
  ```

---

## Saturation T_sat(p) inversion strategy

**Context:** p_sat(T) is given by IAPWS-IF97 Eq. (30) as a direct formula.
The inverse T_sat(p) is needed for region detection and boundary property
evaluation.

**Options considered:**
1. Direct closed-form inverse via IAPWS-IF97 Eq. (31) (Wagner-form approximation).
2. Newton iteration on p_sat(T) = p, starting from Option 1.
3. Look-up table (rejected: incompatible with symbolic differentiation).

**Decision:** Option 2 — direct approximation (Eq. 31) as starting guess,
polished with Newton iteration (10 iterations, converges to machine precision
in ~3).
- Option 1 alone gives ~0.01 K accuracy — adequate for region detection but
  not for boundary property evaluation (h_f, h_g) where ~1 Pa error in p_sat
  maps to ~0.001 kJ/kg error in h_f.
- Newton uses a numerical ±0.005 K central difference for dp/dT because the
  analytical derivative of Eq. (30) is tedious and the numerical one is stable.

**Risks / revisit triggers:**
- Very close to the critical point (T > 645 K), dp/dT becomes very large and
  Newton converges very fast — no risk there.
- Below the triple point (T < 273.15 K), p_sat is not defined — add a
  lower-bound guard if sub-freezing scenarios are needed.

---

## Region boundary blending (smooth() / noEvent())

**Context:** OPAL design rule #1: all components must be event-free for
real-time mode. Region changes at h_f(p) and h_g(p) are discontinuous in
the first derivative of ρ(p, h).

**Options considered:**
1. Hard if/else (current implementation in Water.mo) — generates events.
2. `smooth(1, if h < hf then ... else ...)` with `noEvent()` on the condition.
3. Continuous blending: `w = 0.5*(1 + tanh((h-hf)/delta_h))`, then
   `rho = (1-w)*rho_R1 + w*rho_R4`.

**Decision:** Option 1 (hard if/else) in this initial implementation.
This is acceptable for Phase 2 analysis-mode use where event detection is
active. The real-time-mode refactor (Phase 6) will switch to Option 3.
Rationale: Option 3 requires evaluating both Region 1 and Region 4 at every
step even when far from saturation — ~2× compute cost. For Phase 2 correctness
verification, this overhead is not justified.

`delta_h_blend = 1 kJ/kg` is defined in Water.mo as a named constant for
when Option 3 is implemented, documenting the intended transition width.
1 kJ/kg was chosen as:
- ~0.05% of the latent heat (h_fg ≈ 2000 kJ/kg) — thermodynamically negligible.
- Wide enough to ensure the blending function has a bounded slope.

**Risks / revisit triggers:**
- If Phase 6 real-time benchmarks show event-handling overhead exceeds budget,
  implement Option 3 blending for the primary circuit components.
- If any simulation crosses saturation rapidly (e.g. rapid depressurisation),
  verify that the event-detection converges correctly in analysis mode.

---

## Deferred items

### Region 3 (near-critical, T > 623 K, p > 16.5 MPa)

**Trigger for need:** Supercritical reactor analysis (SCWR), or LWR accident
analysis that reaches near-critical conditions.

**What fails today:** Water.mo region_ph() will assign Region 1 or 2 to a
near-critical state, giving wrong densities (can be off by 10–50%).

**Upgrade path:** Implement Helmholtz-energy formulation `f(ρ, T)` from
IAPWS-IF97 §8. This requires an additional Newton solve for T from ρ and
the Helmholtz-based expression for ρ(p, T). Estimated ~200 lines of Modelica.

### Region 5 (high-temperature steam, T > 1073 K)

**Trigger for need:** Gas-cooled reactor analysis, or very high-temperature
steam turbine stages.

**What fails today:** Region2.mo polynomial coefficients are only validated
to T = 1073 K. Extrapolation above this will give increasingly wrong results.

**Upgrade path:** Add Region 5 Gibbs function (IAPWS-IF97 §12, Table 37,
6 ideal + 6 residual terms) alongside Region 2.

### Transport properties (viscosity μ, thermal conductivity λ)

**Trigger for need:** Heat transfer correlations (Dittus-Boelter, Chen, etc.)
in Phase 3 heat exchanger and steam generator models.

**What fails today:** No μ or λ functions exist yet.

**Upgrade path:** Implement IAPWS 2008 viscosity release and IAPWS 2011
thermal conductivity release (separate from IF97). Both are polynomial
correlations in ρ and T — similar structure to IF97, ~100 lines each.

### Analytical two-phase ∂ρ/∂p|h

**Trigger for need:** Phase 3 production accuracy requirements, or if Phase 2
shows numerical noise from the finite-difference approximation.

**What fails today:** The ±500 Pa central difference in Water.drho_dp_h for
Region 4 evaluates rho_ph_2phase twice per call. The error is ~1e-8 kg/(m³·Pa)
relative to the analytical value — negligible for typical two-phase
compressibilities of ~1e-4 to 1e-3 kg/(m³·Pa).

**Upgrade path:** Implement Clausius-Clapeyron expression:
```
∂ρ/∂p|h,2phase = (1/ρ² · h_fg) · [dp_sat/dT · (h - h_f) · v_g/h_fg
                                    + x · d(v_g)/dp + (1-x) · d(v_f)/dp]
```
(exact form with saturation-line Jacobian).
