# Edwards-O'Brien Pipe Blowdown Validation

## Experiment

A 4.096 m horizontal pipe filled with subcooled water at 7 MPa / 502 K is
ruptured at one end (glass disk break). The transient involves pressure wave
propagation, flashing onset, and critical two-phase discharge over 0.6 seconds.

NRC Standard Problem 1. Reference: Edwards & O'Brien, *J. British Nuclear
Energy Society*, vol. 9, pp. 125-135, 1970.

## Data

- `data/edwards_blowdown_data.py` — full problem specification (geometry, ICs, gauge stations)
- `data/fig3.csv` — digitized pressure at GS-1 (near break) vs time
- `data/EdwardsTest_backEnd.xml` — extracted equation system from Modelica Pipe1D

## Incremental Validation Results

Five iterations, each adding one physics feature:

| Version | Feature | Early (2-10 ms) | Mid (100-200 ms) | Late (400-600 ms) |
|---------|---------|-----------------|-------------------|-------------------|
| v1 | Algebraic momentum | Empties instantly | 0.1 MPa | 0.1 MPa |
| v2 | + Inertial momentum | 1.5-2.1 MPa | 0.8-1.0 MPa | 0.3 MPa |
| v3 | + HEM critical flow | 2.1 MPa | 1.6-1.8 MPa (stuck) | 1.0 MPa (stuck) |
| v4 | + Ransom-Trapp blend | 1.5-2.1 MPa | 1.2-1.3 MPa | 1.1 MPa |
| v5 | + Non-eq flashing | HEM limitation: T=T_sat, no superheat | — | — |
| **Experiment** | | **2.4-2.5 MPa** | **1.9-2.1 MPa** | **0.3-0.5 MPa** |

## Key Findings

1. **Inertial momentum** was the single largest improvement — enables finite
   wave speed and the characteristic flashing plateau.

2. **Critical flow** improves early-time match but the HEM sound speed
   collapses to ~1 m/s in low-quality two-phase, requiring a floor or blend.

3. **Late-time depressurization** is fundamentally limited by the HEM
   single-mixture model. The enthalpy stays nearly constant because advective
   flux and pressure work almost cancel. The pipe doesn't void fast enough.

4. **Non-equilibrium flashing** cannot be modeled with HEM because T = T_sat
   in the two-phase region by definition. There's no superheat to drive
   flashing. A two-fluid model with separate liquid superheat tracking is
   needed for full Edwards match.

## What This Means for OPAL

The HEM semi-implicit solver is appropriate for:
- Heated channels and boiling (verified, 26 tests)
- Quasi-steady two-phase flow
- Slow transients where thermal equilibrium is a good approximation

For rapid depressurization (blowdowns, LOCAs), the solver needs:
- Two-fluid model (separate liquid/vapor conservation) — Phase 3+
- Or drift-flux with mechanical non-equilibrium

This is consistent with RELAP5's architecture (two-fluid) and is the
expected model boundary for an HEM code.
