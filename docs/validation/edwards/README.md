# Edwards Blowdown Validation

## Introduction

The Edwards blowdown is a standard two-phase thermal-hydraulic benchmark (NRC Standard
Problem 1). A 4.096 m horizontal pipe filled with subcooled water at 7 MPa / 502 K is
ruptured at one end by a glass disk break. The transient involves pressure wave propagation,
flashing onset, and critical two-phase discharge over 0.6 seconds.

Reference: Edwards & O'Brien, *J. British Nuclear Energy Society*, vol. 9, pp. 125-135, 1970.

## Current Best Result

**5-equation drift-flux: 22.7% MAPE** (V24 solver + AsymCond C_tau_alpha=10, tau_mix=4.5e-4)

RELAP5-3D reference: ~20% MAPE, VoidMAE ~0.111

## MAPE Progression

| Model | Physics Source | MAPE |
|-------|---------------|------|
| HEM + IAPWS | Pipe1D.mo + Water.mo | 81.0% |
| 5-eq Bridge, HF + Ramp | ALL from Modelica | 28.3% |
| + Jones/Lahey + break form loss | + V24 solver | 24.1% |
| + C_tau_alpha=10 + tau_mix=4.5e-4 | + AsymCond model | **22.7%** |

## Production Configuration (5-equation)

```
Solver:     bridge_5eq_solver_v24_break_form_loss.py
Physics:    Pipe1D_DriftFlux_AsymCond.mo + Water.mo (IAPWS-IF97)
Model:      EdwardsTest_DriftFlux_HF_Ramp_Flash_AC_t0_002_c10
Parameters: tau_mix=4.5e-4, use_isentropic_a11=True, break_form_loss=True
Flashing:   Jones/Lahey relaxation, tau_flash=0.025, C_tau_alpha=10.0
Result:     22.7% MAPE, VoidMAE=0.221
```

## Per-Station MAPE (Production)

| Station | x (m) | MAPE |
|---------|-------|------|
| GS-1 | 3.927 | 37.1% |
| GS-2 | 3.769 | 29.6% |
| GS-3 | 2.935 | 18.3% |
| GS-4 | 2.024 | 11.9% |
| GS-5 | 1.469 | 22.9% |
| GS-6 | 0.914 | 16.0% |
| GS-7 | 0.079 | 23.0% |
| **Overall** | | **22.7%** |

## Validation Drivers

| Command | Description |
|---------|-------------|
| `solver/edwards_bridge_5eq_validation.py --model hf_ramp_flash_ac10 --save` | **Production (22.7% MAPE)** |
| `solver/edwards_bridge_5eq_validation.py --model hf_ramp --save` | Baseline HF+Ramp (28.3% MAPE) |
| `solver/compare_void_variants.py --variants v24,v33 --model ...` | Solver variant comparison |

## Directory Structure

```
docs/validation/edwards/
  results/
    5eq/                    # All 5-equation results
      v24_ac10_production/  # Latest production (22.7% MAPE)
      hf_ramp/              # Baseline HF+Ramp (28.3% MAPE)
      ...                   # Historical variants
    6eq/                    # 6-equation results (in development)
  publication/              # Publication figures and data export
  data/                     # Experimental data + RELAP5 reference
```

## Key Documents

- `edwards_optimization_report.md` — Full 38+ variant optimization campaign
- `LESSONS_LEARNED.md` — Physics insights from each improvement
- `publication_summary.md` — Publication-ready analysis
- `solver_variant_trajectory.md` — Detailed variant-by-variant results

## Next: 6-Equation Two-Fluid Model

The 5-equation drift-flux model has a **structural Pareto frontier** between pressure
accuracy and void accuracy (proven across 38+ solver variants). The 6-equation two-fluid
model eliminates this frontier by giving vapor its own momentum equation. See
`docs/6eq_implementation_plan.md`.
