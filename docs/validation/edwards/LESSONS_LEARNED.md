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
| 5-eq Bridge (break ramp) | 26.9% | 3ms break opening ramp | Break opening dynamics are first-order for near-break stations |

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

## Remaining Gaps (as of 2026-03-20)

1. **GS-1 late-time error (~43%)**: Depressurization still too fast after 200ms.
   Physics-based H_i (d_b=3e-4, Nu=2) gives weaker flashing than tuned H_i=1e7.
   May need turbulence-enhanced Nu or pressure-dependent d_b.
2. **No flow regime map**: Single bubbly-flow drift-flux correlation everywhere.
3. **N=24 spatial resolution**: 2.3 diameters per cell. N=48 study needed.
4. **Integer parameter limitation**: OM bridge cannot handle integerParameter
   references; critical_flow_model uses Real instead of Integer (with OM warning).
