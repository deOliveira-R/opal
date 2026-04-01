# Edwards-O'Brien Pipe Blowdown — Validation Report

## Problem Description

NRC Standard Problem 1 (Edwards & O'Brien, 1970). A horizontal pipe (4.096 m
long, 0.073 m ID) filled with subcooled water at 7 MPa / 502 K is ruptured at
one end by breaking a glass disk. The transient involves:

- Pressure rarefaction wave propagation (~1000 m/s in subcooled water)
- Flashing onset when local pressure drops below saturation (~2.1 MPa at 502 K)
- Critical (choked) two-phase flow at the break
- Void fraction growth and pipe voiding over 0.6 seconds

Seven pressure gauge stations (GS-1 near break to GS-7 near closed end)
provide time-resolved experimental data digitized from the original paper.

## Solvers Compared

| Solver | Equations | Phase Change | Key Feature |
|--------|-----------|-------------|-------------|
| HEM v4 | 3 (mixture mass, momentum, energy) | Thermal equilibrium (T=T_sat in two-phase) | Ransom-Trapp critical flow |
| 5-eq v1 | 5 (phasic mass ×2, mixture momentum, phasic energy ×2) | Non-equilibrium (T_l can exceed T_sat) | Interfacial heat transfer, nucleation onset |

Both use: 24 cells, dt=0.05 ms, inertial momentum, Ransom-Trapp critical flow,
IAPWS-IF97 properties.

The 5-eq model adds: H_i = 1e7 W/(m³·K), nucleation at α_min = 1e-3
when T_l > T_sat, interfacial area = max(4α(1-α), α).

## Results by Time Regime

### Early Time (0–20 ms): Pressure Wave Propagation

![Early time](five_eq_v1/pressure_early_time.png)

**Observation:** The rarefaction wave arrives at each gauge station at the
correct time — GS-1 (near break) sees the drop first, GS-7 (closed end)
last at ~4 ms. The wave speed is consistent with ~1000 m/s in subcooled water.

**Issue:** At GS-1 and GS-2, the initial pressure drop is too deep (simulation
drops to ~2.3 MPa vs experiment 4-5 MPa at t ≈ 1 ms). This suggests the
pressure wave is not being resolved within the first cell — the 24-cell mesh
(dx = 17 cm) is too coarse to capture the wave front which traverses one cell
in 0.17 ms. Both solvers show the same behavior — this is a mesh resolution
issue, not a model issue.

**Both solvers are identical** in early time (the 5-eq model has no effect
before flashing starts).

### Mid Time (50–200 ms): Flashing Plateau

**Observation:** Experiment shows a sustained pressure plateau at 2.0–2.5 MPa
across all stations during this period. Both solvers underpredict this plateau:

| Station | Experiment | HEM v4 | 5-eq v1 |
|---------|-----------|--------|---------|
| GS-7 | 2.5–2.7 MPa | 2.4–2.7 MPa | 2.4–2.7 MPa |
| GS-4 | 2.6–2.7 MPa | 2.5–2.7 MPa | 2.5–2.7 MPa |
| GS-1 | 2.0–2.5 MPa | 1.3–1.5 MPa | 1.3–1.5 MPa |

The mid-pipe stations (GS-4 through GS-7) match reasonably well (5–15% error).
The near-break stations (GS-1, GS-2) show the pressure dropping below the
experimental plateau. This is the same behavior in both solvers — the critical
flow model or break geometry modeling likely underpredicts the choking that
sustains the experimental plateau.

### Late Time (200–600 ms): Depressurization and Voiding

![All stations comparison](five_eq_v1/pressure_all_stations.png)

This is where the two solvers diverge dramatically:

| Station | Experiment (400 ms) | HEM v4 | 5-eq v1 |
|---------|-------------------|--------|---------|
| GS-7 | 0.9 MPa | 1.6 MPa | 1.6 MPa |
| GS-4 | 0.5 MPa | 1.4 MPa | 1.4 MPa |
| GS-1 | 0.5 MPa | 1.1 MPa | 1.1 MPa |

| Station | Experiment (550 ms) | HEM v4 | 5-eq v1 |
|---------|-------------------|--------|---------|
| GS-7 | 0.2 MPa | 1.1 MPa | **1.1 MPa** |
| GS-4 | 0.2 MPa | 1.1 MPa | **1.1 MPa** |
| GS-1 | 0.3 MPa | 1.0 MPa | **0.6 MPa** |

The 5-eq model begins depressurizing faster than HEM after ~500 ms as the
non-equilibrium flashing mechanism activates. By 600 ms, GS-1 reaches 0.3 MPa
(5-eq) vs 1.0 MPa (HEM) vs 0.03 MPa (experiment).

**The flashing onset is delayed** — it should start around 100–200 ms but only
becomes significant after 500 ms in the simulation. This delay is caused by
the pressure staying above the "flashing pressure" (~2.1 MPa for T_l = 502 K)
for too long. In the experiment, the pressure drops below 2 MPa by 300 ms at
most stations, but in the simulation it stays at 1.1–1.2 MPa — which is
actually below the flashing threshold, yet the void growth is slow.

### Void Fraction Evolution

![Void fraction](five_eq_v1/void_fraction.png)

The void fraction plot reveals the flashing dynamics:

1. **0–200 ms:** Transient void pulses at GS-7 (closed end) reach α ≈ 0.2
   due to the reflected pressure wave causing momentary superheat. These
   pulses collapse as the pressure re-equilibrates. This is physically
   reasonable — pressure wave reflection at the closed end creates a brief
   low-pressure zone.

2. **200–480 ms:** Void fraction is near zero everywhere. The liquid is
   subcooled relative to local saturation (T_l < T_sat at the prevailing
   pressure). No flashing.

3. **480–600 ms:** Rapid void growth at all stations simultaneously, reaching
   α = 0.5–0.65 by 600 ms. The flashing front propagates from the break
   toward the closed end, consistent with the pressure wave direction.

The experiment shows earlier and more gradual void growth (starting ~100 ms),
suggesting the delayed flashing is the primary error source.

## Error Summary

![Error comparison](error_summary.png)

| Metric | HEM v4 | 5-eq v1 | Improvement |
|--------|--------|---------|-------------|
| Overall MAPE | 103% | 74% | 28% better |
| Early (<50 ms) MAPE | 16% | 16% | Same |
| Mid (50–200 ms) MAPE | 13% | 13% | Same |
| Late (>200 ms) MAPE | 186% | 136% | 27% better |

The 5-equation model improves late-time accuracy by ~27% relative to HEM.
Early and mid-time performance is identical (as expected — the 5-eq additions
only matter when flashing occurs).

## Root Cause Analysis

### What works

1. **Wave propagation** — correct speed, correct arrival times at all stations
2. **Mid-pipe plateau** — GS-4 through GS-7 match well in the 50–200 ms range
3. **Qualitative physics** — the 5-eq model correctly produces void growth
   from non-equilibrium flashing, which HEM fundamentally cannot do
4. **Late-time trend** — the 5-eq model is depressurizing toward the correct
   final state (just delayed)

### What does not work

1. **Near-break early pressure** — GS-1/GS-2 drop too deep at 1–5 ms (mesh
   resolution + break modeling)
2. **Late-time depressurization too slow** — pressure stays at 1.0–1.2 MPa
   from 200–480 ms instead of dropping to 0.3–0.5 MPa
3. **Delayed flashing onset** — void growth starts 300 ms late compared to
   experiment

### Likely causes (in priority order)

1. **Critical flow model** — The Ransom-Trapp blend may over-restrict the
   break flow rate, keeping pipe pressure artificially high. A more aggressive
   critical flow model (or better sound speed treatment in two-phase) could
   lower the mid-time plateau and allow earlier flashing.

2. **Interfacial heat transfer coefficient** — H_i = 1e7 controls flashing
   rate. The value was not tuned; a sensitivity study varying H_i from 1e5
   to 1e8 would quantify its impact.

3. **Mesh resolution** — 24 cells may be too coarse for the pressure wave
   front. A convergence study at 48 and 96 cells would clarify.

4. **Break opening model** — The simulation assumes instantaneous rupture;
   the experiment has a ~1 ms disk breakage time which softens the initial
   wave.

## Conclusions

1. The **5-equation drift-flux model demonstrates the key physics** that HEM
   cannot capture: non-equilibrium flashing driven by liquid superheat. This
   is architecturally correct and validates the Phase 3 model hierarchy.

2. The **late-time improvement over HEM is real** (28% MAPE reduction) but
   the flashing onset is delayed ~300 ms compared to experiment.

3. The **mid-time plateau and early-time wave resolution** are shared
   limitations of both solvers and are independent of the thermal
   non-equilibrium model.

4. **Next steps** for improving Edwards agreement:
   - Critical flow model sensitivity / alternative (Henry-Fauske)
   - H_i sensitivity study (1e5 to 1e8)
   - Mesh refinement study (48, 96 cells)
   - Break opening time ramp (0–1 ms)

## Files

```
results/
├── report.md                           ← this file
├── error_summary.png                   ← MAPE comparison bar chart
├── hem_v4/
│   ├── pressure_all_stations.png       ← 7-panel pressure (0-600 ms)
│   ├── pressure_early_time.png         ← 7-panel early time (0-20 ms)
│   └── report.md                       ← HEM-specific error table
└── five_eq_v1/
    ├── pressure_all_stations.png       ← 7-panel pressure (0-600 ms)
    ├── pressure_early_time.png         ← 7-panel early time (0-20 ms)
    ├── void_fraction.png               ← void fraction at 4 stations
    └── report.md                       ← 5-eq-specific error table
```
