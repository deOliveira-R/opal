# Edwards Blowdown Validation: Lessons Learned

## What This Document Is

A systematic record of what each validation improvement revealed about the physics,
numerics, and architecture needed for two-phase thermal-hydraulic simulation. Written
during the Edwards blowdown validation campaign (Phase 2.5 → Phase 3). This knowledge
feeds into the OPAL Claude Skill for future validation work.

## MAPE Progression

| Step | MAPE | What changed | What we learned |
|------|------|-------------|-----------------|
| HEM + IAPWS (baseline) | 81.0% | 3-equation HEM, SimpleFluid → IAPWS | IAPWS properties are essential; HEM can't capture non-equilibrium |
| HEM + temperature profile IC | 72.8% | Axial T(x) from experiment | Initial condition quality matters ~8 pts; subcooling distribution affects flashing onset |
| 5-eq Python (no flash) | 79.8% | 5 equations but T_l = T_sat | Non-equilibrium MODEL without non-equilibrium PROPERTIES is useless |
| 5-eq Python (metastable fix) | 30.0% | T_l = T_sat + dh/cp_f when h_l > h_f | Metastable liquid extension is the single most important physics feature |
| 5-eq Bridge (initial) | ~36% | All physics from Modelica pipeline | Pipeline works end-to-end; OM variable elimination needs handling |
| 5-eq Bridge (implicit friction) | 28.4% | Semi-implicit friction for tridiagonal stability | drho_dp at h_mix + implicit friction = correct semi-implicit scheme |
| 5-eq Bridge, RT + Ramp (Modelica ramp) | 31.8% | RampedBreak BC moved fully into Modelica | Break ramp is now extracted from Modelica, not hardcoded in Python |
| 5-eq Bridge, HF + Ramp (Modelica ramp) | 28.3% | Henry-Fauske replaces Ransom-Trapp | Non-equilibrium critical flow better for sharp-edged glass disk break |

## Physics Features and Why They Matter

### 1. Metastable liquid state extension
**Impact: ~50 MAPE points (79.8% → 30%)**

When pressure drops below saturation, the equilibrium EOS returns T_l = T_sat. But the
liquid is superheated — it hasn't flashed yet. The 5-equation model needs T_l > T_sat
to drive interfacial heat transfer (q_i_l) which drives mass transfer (Gamma) which
drives void growth.

Formula: `T_l = T_sat + (h_l - h_f) / cp_f(p)` when `h_l > h_f(p)`.
Reference: RELAP5/MOD3 Vol I §3.2.

**Key insight:** This is NOT a property evaluation issue — it's a MODELING choice. The
equilibrium EOS is "correct" for equilibrium; the metastable extension is needed
specifically because the 5-eq model tracks non-equilibrium states.

### 2. Pressure-dependent cp_f(p) instead of constant 4200
**Impact: Small for Edwards (~1-2 pts) but important for correctness**

At 7 MPa, cp_f ≈ 5400 J/(kg·K). At 1 MPa, cp_f ≈ 4400. The constant 4200 was too
low for high pressure, over-estimating T_l superheat and over-driving flashing.

**Key insight:** "Good enough" constants accumulate errors. When multiple approximations
each add a few percent, the combined effect can be significant.

### 3. Drift-flux phasic split (V_gj + C_0)
**Impact: Small for Edwards (horizontal pipe, C_0=1.0) but essential for vertical flows**

The drift-flux algebraic slip relation: `v_v = C_0 * j + V_gj` determines how fast
vapor moves relative to liquid. In Edwards (horizontal, high-velocity blowdown),
the drift effect is small. But for vertical flows (boiling channels, steam generators),
the drift velocity drives phase separation.

**Key insight:** Features that don't help the current validation case may be essential
for the next one. Build them anyway when the physics demands it.

### 4. Semi-implicit friction (implicit friction resistance)
**Impact: Enables stability without lazy numerical floors**

When cells transition from two-phase to single-phase vapor, drho_dp drops by 3-5
orders of magnitude. The pressure tridiagonal becomes ill-conditioned. The solution:
semi-implicitize the Darcy friction term, yielding `beta_eff = beta / (1 + sigma)`.
This smoothly blends inertial and algebraic momentum at each face based on local
flow conditions.

**Key insight:** Numerical instability often has a principled solution if you look for
it. The "easy" fix (drho_dp floor) was rejected; the correct fix (implicit friction)
was derived from the semi-implicit momentum discretization. Always look for the
mathematically derived fix before resorting to hacks.

### 5. Break opening ramp
**Impact: -6 pts on GS-1, -5 pts on GS-2**

The Edwards experiment has a ~1-2ms glass disk break time. Instantaneous opening
causes an unphysical initial pressure spike at the break station. A linear C_d ramp
over 3ms (effective opening time including expansion geometry) significantly improves
the near-break stations.

**Key insight:** Boundary condition dynamics are first-order effects for stations near
the boundary. The optimal effective opening time (3ms) exceeds the physical break
time (1-2ms), suggesting that 1D pipe models need "effective" BC parameters that
compensate for geometry effects not captured by the 1D model.

### 6. Mixture drho_dp for semi-implicit scheme
**Impact: Required for stability (phasic drho_dp gave 150% MAPE)**

The 5-eq model has independent phasic enthalpies, suggesting phasic-weighted
compressibility: `(1-alpha)*drho_dp(p,h_l) + alpha*drho_dp(p,h_v)`. This is
theoretically correct but gives tiny values for single-phase regions, making the
semi-implicit scheme overdamped.

The mixture drho_dp at h_mix includes the saturation curve shift ("thermal
compressibility") — density change from phase change at constant h_mix. This
thermal compressibility is what allows the semi-implicit scheme to use practical
timesteps (dt >> dx/c). All production codes (RELAP5, TRACE) use mixture
compressibility for this reason.

**Key insight:** Theoretical correctness and numerical practicality can conflict.
The semi-implicit scheme's implicit treatment of the pressure equation requires
"effective compressibility" that includes phenomena (phase change) that are
treated explicitly elsewhere. This is a known and accepted practice.

## Numerical Techniques and Why They're Needed

### Conservative void fraction update
The Modelica model writes `rho_v * V * der(alpha)` (non-conservative). The solver
does the conservative update: `alpha_rho_v_new = alpha_old * rho_v_old + dt * flux`.
This is operator splitting: Modelica provides the equation structure, the solver
chooses the discrete update. During rapid depressurization, the conservative form
is essential — rho_v changes by 60x, and the non-conservative form loses mass.

### Nucleation floor
When Gamma > 0 (flashing), enforce alpha >= 1e-3. Without this, advective washout
removes the nucleation seed before interfacial HT can grow it. This is a numerical
seeding mechanism, not a physics closure.

### h_l_old / h_v_old saves
All energy advection must use old-timestep enthalpy values. Without this, sequential
cell updates introduce a directional bias (upstream cells see old values, downstream
cells see partially-updated values).

## Architecture Insights

### Physics in Modelica, numerics in solver
The cardinal rule was validated: ALL physics (closures, properties, friction multiplier,
critical flow, drift-flux split) comes from Modelica via the bridge. The solver provides
ONLY: Thomas algorithm, semi-implicit pressure solve, momentum update with implicit
friction, explicit transport with conservative void and nucleation floor.

### OM variable elimination
OpenModelica aggressively eliminates boundary variables during compilation (inlining
them into the equations that use them). The bridge handles this with sentinel indices
(-1) and nearest-neighbor gap filling. This is a pipeline robustness issue, not a
physics issue.

### Parameter initialization from XML
Boolean parameters from OM come as 'true'/'false' strings, not floats. The bridge
must handle this explicitly. Without it, critical flow was disabled (use_critical_flow
read as None → default False).

### 7. Henry-Fauske critical flow
**Impact: -12 pts on GS-1 (64.2% → 52.3%), -1.6 pts overall**

Henry-Fauske accounts for non-equilibrium (delayed flashing) at the break plane.
With N_param=0 (frozen flow, sharp orifice), the model gives Bernoulli discharge
at the throat pressure p_c = 2/3*p_0, which is HIGHER than Ransom-Trapp's
HEM-blended value at low quality.

For Edwards (glass disk break, L/D ≈ 0), the liquid leaves the break plane as
metastable liquid without flashing. Henry-Fauske captures this physically;
Ransom-Trapp's equilibrium blend artificially reduces the discharge rate.

Combined with break ramp: GS-1 = 43.5%, overall = 28.2%.

**Key insight:** The critical flow model matters more for near-break stations than
interior stations. The choice of critical flow model (equilibrium vs non-equilibrium)
is a physical modeling question that depends on the break geometry (L/D ratio).
The C_d parameter has a different physical meaning in each model: 0.87 for
Ransom-Trapp (semi-empirical), 0.61 for Henry-Fauske (sharp-edged orifice theory).

### 8. Mesh convergence study (anti-convergence)
**Impact: N=24 is the practical mesh; finer meshes diverge**

Mesh convergence study with N=12, 24, 48, 96 (CFL-scaled dt):

| N | dx (m) | dt (µs) | MAPE | GS-1 | GS-7 | Wall time |
|---|--------|---------|------|------|------|-----------|
| 12 | 0.341 | 100 | 23.1% | 46.5% | 16.8% | 3.6 s |
| 24 | 0.171 | 50 | 28.3% | 43.7% | 24.6% | 13.8 s |
| 48 | 0.085 | 25 | 39.9% | 43.2% | 36.2% | 55.6 s |
| 96 | 0.043 | 12.5 | 49.9% | 50.7% | 44.5% | 220.9 s |

MAPE *increases* with refinement — classic anti-convergence. N=96 is clearly unstable:
pressure collapses to vacuum by 250 ms, outlet mass flow reaches -112 kg/s (unphysical
inflow). N=48 shows similar late-time instability.

**Root cause:** The semi-implicit scheme treats only pressure implicitly. Void fraction
and phasic enthalpies use explicit donor-cell transport, which has a CFL limit on the
material Courant number (u·dt/dx). Scaling dt ∝ dx preserves the acoustic CFL but not
the transport CFL — as N increases, the higher-velocity cells (near-break) exceed the
transport CFL. Numerical diffusion from donor-cell at N=12/24 stabilizes the solution;
at N=48/96 it is insufficient.

**Key insights:**
- N=12 giving the lowest MAPE (23.1%) is error cancellation: coarse-grid numerical
  diffusion happens to smooth the solution toward experimental data. This is not
  convergence — it is a fortuitous balance of errors.
- N=24 (28.3%) is the defensible production mesh: fine enough for reasonable spatial
  resolution (~2.3 D/cell) but coarse enough that the explicit transport is stable.
- Achieving mesh convergence would require implicit treatment of the transport terms
  (void fraction, enthalpy advection), not just the pressure equation. This is a known
  limitation of semi-implicit staggered-mesh schemes for two-phase flow.
- The anti-convergence should be reported honestly in any publication: it identifies the
  solver's spatial resolution limit and motivates future implicit transport work.

## Remaining Gaps (as of 2026-03-22)

1. **GS-1 late-time error (~43.7%)**: Depressurization still too fast after 200ms.
   Physics-based H_i (d_b=3e-4, Nu=2) gives weaker flashing than tuned H_i=1e7.
   May need turbulence-enhanced Nu or pressure-dependent d_b.
2. **No flow regime map**: Single bubbly-flow drift-flux correlation everywhere.
3. **Mesh anti-convergence**: Explicit transport CFL limits practical resolution to
   N~24. Implicit transport needed for mesh-converged solutions (see §8 above).
4. **Integer parameter limitation**: OM bridge cannot handle integerParameter
   references; critical_flow_model uses Real instead of Integer (with OM warning).

## Architecture Notes (2026-03-22)

- **Break ramp fully in Modelica**: The break opening ramp is now implemented in
  `Boundary/RampedBreak.mo` and extracted via the bridge. There is no Python-side
  ramp override. This is the canonical approach.
- **Parameter type collision bug (fixed)**: OM uses separate index spaces for Real,
  Integer, and Boolean parameters, but the info.json reports a single `index` field.
  Integer parameters (e.g., `pipe.N=24`, index=0) were overwriting Real parameters
  (e.g., `break_bc.C_d_final=0.87`, also index=0) in the bridge's `set_params`
  call. Fixed by adding `var_type` to VarInfo and filtering on `type=="Real"` in
  `set_params_from_spec`. This bug was latent in all prior results but only manifested
  when the RampedBreak model introduced a critical Real parameter at index 0.
- **Henry-Fauske better than Ransom-Trapp for this benchmark**: HF (frozen flow, N=0)
  gives 28.3% vs RT's 31.8%. For sharp-edged breaks (L/D ≈ 0), the liquid at the
  break plane exits as metastable liquid without equilibrium flashing. HF captures
  this physically; RT's equilibrium blend reduces discharge rate artificially.
  Choice of critical flow model depends on break geometry (L/D ratio).
