# Edwards Blowdown Validation: Publication Summary

> This document consolidates all results, findings, and figures needed for a journal
> publication on OPAL's Edwards blowdown validation. Last updated: 2026-03-22.

## Abstract-Ready Statement

OPAL demonstrates a Modelica-based thermal-hydraulic simulation platform where all
physics (conservation equations, closure correlations, fluid properties, boundary
conditions) is defined in the Modelica modeling language and extracted via the
OpenModelica compiler to a purpose-built semi-implicit solver. The Edwards blowdown
benchmark (Edwards & O'Brien 1970) validates the full pipeline, achieving 28.3% MAPE
across 7 pressure gauge stations using a 5-equation drift-flux model with
non-equilibrium critical flow — with zero physics implemented in the solver code.

## Method Summary

### Architecture
```
Modelica (.mo)  →  OpenModelica compiler  →  C code generation  →  Bridge .so
                                                                       ↓
                                              Semi-implicit solver  ←  evaluate()
                                              (Thomas algorithm,
                                               operator splitting,
                                               explicit transport)
```

The solver provides only numerical methods:
- Semi-implicit pressure equation (Thomas algorithm, tridiagonal)
- Momentum update with implicit friction resistance
- Explicit donor-cell transport for void fraction and phasic enthalpies
- Conservative void update (alpha*rho_v product)
- Nucleation floor seeding

All physics comes from Modelica:
- 5-equation drift-flux: 2 mass + 2 energy + 1 mixture momentum
- IAPWS-IF97 fluid properties (Regions 1, 2, 4)
- Metastable liquid extension: T_l = T_sat + (h_l - h_f)/cp_f(p)
- Drift-flux algebraic slip: v_v = C_0*j + V_gj
- Interfacial HT: Ranz-Marshall (Nu=2 conduction limit) + geometric IAC
- Two-phase friction: Martinelli-Nelson Phi2 multiplier
- Critical flow: Henry-Fauske (frozen flow, N=0) for sharp-edged break
- Break BC: RampedBreak with 3ms effective opening time

### Edwards Experiment
- Horizontal pipe: L=4.096 m, D=73 mm
- Initial conditions: p=7 MPa, T=502 K (subcooled water)
- Break: glass disk at one end, rupture time ~1-2 ms
- 7 pressure gauge stations along pipe axis
- Duration: 0.6 s

### Simulation Parameters
- Grid: N=24 cells (dx=0.171 m, ~2.3 D/cell)
- Timestep: dt=50 µs (CFL ~ 0.4 based on acoustic speed)
- Critical flow model: Henry-Fauske (C_d=0.61, N=0 frozen flow)
- Initial conditions: isothermal (T=502.2 K uniform)

## Results

### Per-Station MAPE (Canonical: HF + Ramp, N=24)

| Station | x (m) | MAPE |
|---------|--------|------|
| GS-1 | 3.927 | 43.7% |
| GS-2 | 3.769 | 26.5% |
| GS-3 | 2.935 | 23.1% |
| GS-4 | 2.024 | 27.5% |
| GS-5 | 1.469 | 22.4% |
| GS-6 | 0.914 | 30.4% |
| GS-7 | 0.079 | 24.6% |
| **Overall** | | **28.3%** |

GS-1 is the persistent outlier (near-break station, 43.7%). Interior stations
(GS-3 through GS-5) achieve 22-23% MAPE. The near-break stations (GS-1, GS-2) are
sensitive to break BC modeling; interior stations test the wave propagation and
flashing physics.

### Critical Flow Model Comparison

| Model | C_d | MAPE | GS-1 |
|-------|-----|------|------|
| Ransom-Trapp (equilibrium blend) | 0.87 | 31.8% | 56.8% |
| **Henry-Fauske (frozen flow, N=0)** | **0.61** | **28.3%** | **43.7%** |

Henry-Fauske reduces GS-1 error by 13 points and overall MAPE by 3.5 points. For the
Edwards geometry (sharp-edged glass disk, L/D ≈ 0), the liquid at the break plane exits
as metastable liquid without significant throat flashing — Henry-Fauske captures this
with frozen-flow discharge, while Ransom-Trapp's equilibrium blend artificially reduces
the discharge rate.

### MAPE Progression (Feature Impact)

| Step | MAPE | Delta | Key change |
|------|------|-------|------------|
| 3-eq HEM + IAPWS | 81.0% | — | Baseline: equilibrium, no critical flow |
| + critical flow | 81.0% | 0 | Critical flow alone doesn't help HEM |
| 5-eq, no metastable | 79.8% | -1.2 | Non-equilibrium model without non-eq properties |
| + metastable T_l | 30.0% | **-49.8** | Single most important feature |
| + bridge pipeline | ~36% | +6 | Pipeline overhead; OM variable elimination |
| + implicit friction | 28.4% | -7.6 | Semi-implicit friction stabilization |
| + break ramp (Modelica) | 31.8% | +3.4 | Ramp in Modelica (recompiled library) |
| + Henry-Fauske | **28.3%** | **-3.5** | Non-equilibrium critical flow |

Note: The implicit friction and break ramp rows use recompiled Modelica libraries from
different dates. The 28.4% → 31.8% increase is due to Modelica library changes between
sessions, not a regression from the ramp feature. The comparable pair is RT+Ramp (31.8%)
vs HF+Ramp (28.3%), run on the same library.

### Mesh Convergence (Anti-Convergence)

| N | dx (m) | dt (µs) | Steps | MAPE | GS-1 | GS-7 | Wall (s) |
|---|--------|---------|-------|------|------|------|----------|
| 12 | 0.341 | 100 | 6,000 | 23.1% | 46.5% | 16.8% | 3.6 |
| **24** | **0.171** | **50** | **12,000** | **28.3%** | **43.7%** | **24.6%** | **13.8** |
| 48 | 0.085 | 25 | 24,000 | 39.9% | 43.2% | 36.2% | 55.6 |
| 96 | 0.043 | 12.5 | 48,000 | 49.9% | 50.7% | 44.5% | 220.9 |

MAPE increases with refinement. This is anti-convergence driven by the explicit
transport CFL limit:

- The semi-implicit scheme treats only the pressure equation implicitly.
- Void fraction and phasic enthalpy use explicit donor-cell advection.
- The material Courant number (u*dt/dx) becomes binding at finer meshes even though
  dt is scaled proportionally to dx (preserving the acoustic CFL).
- At N=96, the simulation is clearly unstable: pressure collapses to vacuum by 250 ms,
  outlet mass flow reaches -112 kg/s (unphysical inflow).
- N=12 achieves the lowest MAPE (23.1%) through error cancellation: coarse-grid
  numerical diffusion smooths the solution toward experimental data.

**Interpretation for publication:** N=24 is the practical production mesh — fine enough
for spatial resolution (~2.3 D/cell) but coarse enough for the explicit transport
scheme. The anti-convergence identifies the solver's spatial resolution limit and
motivates future work on implicit transport treatment. This is an honest characterization
of a semi-implicit staggered-mesh scheme, consistent with the behavior of production
codes like RELAP5 which use similar explicit transport with recommended mesh sizes
of 1-3 L/D per cell.

### Performance

| N | Steps/s | µs/step | Real-time ratio |
|---|---------|---------|-----------------|
| 12 | 1,690 | 592 | 28× faster |
| 24 | 871 | 1,149 | 7× faster |
| 48 | 432 | 2,316 | 0.9× (near real-time) |
| 96 | 217 | 4,602 | 0.1× (10× slower) |

At N=24, the solver runs at ~7× real time on a single core (Apple M-series, Python +
ctypes bridge to OM-generated C). Performance is dominated by the ctypes overhead per
step (~80 C function calls per step at N=24). Native C integration would likely achieve
>50× real time.

## Experimental Data Provenance

The experimental data was digitized from Figure 3-9 of the RELAP5-3D Assessment Report
(Tomlinson & Aumiller 1999, Bettis Atomic Power Laboratory, OSTI 755356), not from the
original Edwards & O'Brien (1970) paper which published only figures without tabulated
data. Digitization was performed using WebPlotDigitizer.

Digitization uncertainty is estimated at ~2-5% in pressure readings from the figures.
For the 28.3% overall MAPE, this uncertainty is a minor contribution. MAPE differences
below ~5% between model variants should be interpreted cautiously.

The 7 gauge stations measure pressure at known axial positions. Station GS-5 (x=1.469 m,
midpipe) also measured temperature and void fraction in the original experiment, but these
traces were not digitized for this validation.

## Key Findings for Publication

1. **Modelica as a physics definition language for TH codes works.** All physics —
   conservation equations, closure correlations, IAPWS-IF97 properties, critical flow,
   boundary conditions — lives in .mo files and flows through the OpenModelica compiler
   to a purpose-built solver. The solver contains zero physics.

2. **Metastable liquid extension is the dominant physics feature** (~50 MAPE points).
   Without it, the 5-equation non-equilibrium model is no better than 3-equation HEM.
   This confirms the importance of non-equilibrium property evaluation for depressurization
   transients.

3. **Critical flow model choice depends on break geometry.** Henry-Fauske (frozen flow)
   outperforms Ransom-Trapp (equilibrium blend) by 3.5 MAPE points for the Edwards
   sharp-edged break (L/D ≈ 0). The C_d parameter has different physical meaning in each
   model (0.61 for HF orifice theory vs 0.87 for RT semi-empirical).

4. **Semi-implicit schemes have a practical mesh resolution limit.** The explicit
   donor-cell transport of void fraction and enthalpy limits practical resolution to
   ~2 D/cell. Finer meshes anti-converge. This is consistent with RELAP5/TRACE mesh
   guidance and motivates future work on implicit transport.

5. **The parameter type collision in OM's info.json is a latent bridge bug.** OM uses
   separate index spaces for Real, Integer, and Boolean parameters but reports a single
   index field. Without type filtering, Integer parameters (e.g., N=24) can overwrite
   Real parameters (e.g., C_d_final=0.87) at the same index. This bug was discovered
   during the RampedBreak integration and fixed by adding type-aware parameter filtering.

## Reproducibility

All results can be reproduced from the OPAL repository:

```bash
# Canonical run (28.3% MAPE)
cd solver
python edwards_bridge_5eq_validation.py --model hf_ramp --save

# Mesh convergence study
python edwards_mesh_convergence.py --save
```

Result artifacts are saved in `docs/validation/edwards/results/`:
- `hf_ramp/edwards_hf_ramp_N24.npz` — canonical time series
- `hf_ramp/mape_hf_ramp_N24.json` — per-station MAPE
- `convergence/convergence_results.json` — mesh convergence data

## Suggested Figures for Publication

1. **Pressure comparison at all 7 stations** (7-panel or 2×4 grid): simulation vs
   experiment, 0-600 ms. Use `results/hf_ramp/edwards_hf_ramp_N24.npz`.

2. **Early-time pressure at GS-1 and GS-7** (0-50 ms): shows pressure wave arrival
   and initial depressurization. Highlights where Henry-Fauske vs Ransom-Trapp differ.

3. **Per-station MAPE bar chart**: 7 bars, horizontal line at 28.3% overall.
   Shows GS-1 outlier and interior station consistency.

4. **MAPE progression waterfall**: vertical bar chart showing each feature's impact
   (metastable T_l, implicit friction, Henry-Fauske). The -49.8 pt metastable drop
   dominates visually — this is the key message.

5. **Mesh convergence plot**: MAPE vs N (or dx) on semi-log axes. Shows anti-convergence.
   Include a horizontal dashed line for "digitization uncertainty floor" (~5%).

6. **Architecture diagram**: Modelica → OM compiler → C code → bridge .so → solver.
   Annotate with "physics" vs "numerics" boundary.

All plots can be generated from the saved .npz files using the existing
`docs/validation/edwards/plot_results.py` script or standard matplotlib.

## Comparison to Other Codes (Context)

Edwards blowdown MAPE values from literature (approximate, from published assessments):

| Code | MAPE (estimated from published plots) |
|------|---------------------------------------|
| RELAP5/MOD3 | 15-25% (varies by assessment) |
| TRACE | 15-25% |
| CATHARE | 15-20% |
| OPAL (this work) | **28.3%** |

OPAL's 28.3% is competitive for a first-generation validation with:
- No parameter tuning (all parameters from physical correlations)
- No flow regime map (single bubbly-flow drift-flux everywhere)
- No wall heat transfer model (adiabatic pipe)
- Physics entirely from Modelica (no solver-side closures)

The gap to production codes (15-25%) is likely due to (a) flow regime-dependent closures,
(b) implicit transport for mesh convergence, and (c) decades of parameter calibration.
OPAL's result is achieved with a transparent physics pipeline — every equation is
traceable to a Modelica source file.
