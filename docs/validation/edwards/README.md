# Edwards Blowdown Validation

## Introduction

The Edwards blowdown is a standard two-phase thermal-hydraulic benchmark (NRC Standard
Problem 1). A 4.096 m horizontal pipe filled with subcooled water at 7 MPa / 502 K is
ruptured at one end by a glass disk break. The transient involves pressure wave propagation,
flashing onset, and critical two-phase discharge over 0.6 seconds.

Reference: Edwards & O'Brien, *J. British Nuclear Energy Society*, vol. 9, pp. 125-135, 1970.

## Experimental Data

- Source: Digitized from Tomlinson & Aumiller (1999) RELAP5-3D assessment report.
- Digitization uncertainty: ~2-5% in pressure readings.
- 7 gauge stations (GS-1 through GS-7) along the pipe axis.
- `data/fig3-gs1.csv` through `data/fig9-gs7.csv` — digitized pressure traces.

## MAPE Progression

| Model | Physics Source | MAPE |
|-------|---------------|------|
| HEM + IAPWS | Pipe1D.mo + Water.mo | 100.7% |
| HEM + IAPWS + critical flow | + CriticalFlow.mo | 81.0% |
| 5-eq Case 1 (Python, no flash) | C++ fluid + Python closures | 79.8% |
| 5-eq Case 1 (Python, w/ flash fix) | C++ fluid + Python closures | 30.0% |
| 5-eq Bridge, RT + Ramp | ALL from Modelica | 31.8% |
| **5-eq Bridge, HF + Ramp** | **ALL from Modelica** | **28.3%** |

## Canonical Run

- **Model**: `hf_ramp` (Henry-Fauske critical flow + RampedBreak BC from Modelica)
- **Grid**: N=24 cells, dx=0.171 m
- **Timestep**: dt=50 µs
- **End time**: t_end=0.6 s
- **Critical flow comparison**: Ransom-Trapp → 31.8% MAPE, Henry-Fauske → 28.3% MAPE

All physics (closures, friction, critical flow, break BC) extracted from Modelica via the
OpenModelica bridge pipeline. Zero physics in the solver.

## Per-Station MAPE (Canonical: HF + Ramp)

| Station | MAPE |
|---------|------|
| GS-1 | 43.7% |
| GS-2 | 26.5% |
| GS-3 | 23.1% |
| GS-4 | 27.5% |
| GS-5 | 22.4% |
| GS-6 | 30.4% |
| GS-7 | 24.6% |
| **Overall** | **28.3%** |

## Key Physics Features (Canonical Model)

- **Metastable T_l**: `T_l = T_sat + (h_l - h_f) / cp_f(p)` when `h_l > h_f` — single
  most impactful feature (~50 MAPE points)
- **Drift-flux phasic split**: V_gj + C_0 algebraic slip for void fraction distribution
- **Physics-based interfacial HT**: Ranz-Marshall correlation + geometric IAC
  (`a_i = 6*alpha*(1-alpha)/d_b`)
- **Martinelli-Nelson Phi2**: two-phase friction multiplier from Modelica extraction
- **Henry-Fauske critical flow**: frozen-flow non-equilibrium model (N=0, sharp-edged
  orifice), better than Ransom-Trapp for glass disk break (L/D ≈ 0)
- **Implicit friction resistance**: `beta_eff = beta/(1+sigma)` — principled semi-implicit
  tridiagonal stabilization, no floors or hacks
- **RampedBreak BC**: ~3 ms effective break opening ramp, extracted from Modelica
  `Boundary/RampedBreak.mo`

## Validation Drivers

| Command | Description |
|---------|-------------|
| `solver/edwards_bridge_5eq_validation.py --model hf_ramp --save` | **Canonical run (28.3% MAPE)** |
| `solver/edwards_bridge_5eq_validation.py --model rt_ramp --save` | Ransom-Trapp + Ramp (31.8% MAPE) |
| `solver/edwards_5eq_modelica_validation.py` | Case 1: params from XML (30% MAPE) |
| `solver/edwards_modelica_validation.py` | HEM + IAPWS + critical flow (81% MAPE) |

## Result Artifacts

Results from the canonical run are stored in `docs/validation/edwards/results/hf_ramp/`.

## Data Files

- `data/edwards_blowdown_data.py` — full problem specification (geometry, ICs, gauge stations)
- `data/EdwardsTest_IAPWS_backEnd.xml` — extracted Pipe1D with Water (N=24)
- `data/EdwardsTest_IAPWS_CritFlow_backEnd.xml` — extracted Pipe1D with Water + critical flow
- `data/EdwardsTest_DriftFlux_backEnd.xml` — extracted Pipe1D_DriftFlux with all closures

## Mesh Convergence Study

| N | dx (m) | dt (µs) | MAPE | GS-1 | Wall time |
|---|--------|---------|------|------|-----------|
| 12 | 0.341 | 100 | 23.1% | 46.5% | 3.6 s |
| **24** | **0.171** | **50** | **28.3%** | **43.7%** | **13.8 s** |
| 48 | 0.085 | 25 | 39.9% | 43.2% | 55.6 s |
| 96 | 0.043 | 12.5 | 49.9% | 50.7% | 220.9 s |

Anti-convergence: MAPE increases with refinement. The semi-implicit scheme treats only
pressure implicitly; explicit donor-cell transport of void fraction and enthalpy has a
material Courant number limit that becomes binding at finer meshes. N=24 is the practical
production mesh — fine enough for reasonable resolution, stable enough for the explicit
transport scheme.

N=12's lower MAPE (23.1%) is error cancellation from coarse-grid numerical diffusion,
not true convergence. See LESSONS_LEARNED.md §8 for full analysis.

## Known Limitations

- **GS-1 outlier (43.7%)**: Near-break station depressurizes too fast after 200 ms.
  Physics-based H_i (d_b=3e-4, Nu=2) gives weaker flashing than tuned H_i=1e7. May
  need turbulence-enhanced Nu or pressure-dependent d_b.
- **No flow regime map**: Single bubbly-flow drift-flux correlation applied everywhere.
- **Mesh anti-convergence**: Explicit transport CFL limits practical resolution to N~24.
  Mesh convergence would require implicit transport (not just implicit pressure).

See `docs/validation/edwards/LESSONS_LEARNED.md` for full analysis of each physics feature.
