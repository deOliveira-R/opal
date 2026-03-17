# OPAL Validation Benchmarks

Experimental benchmarks for validating the two-phase solver against measured data.
Each subdirectory contains a description of the experiment, acceptance criteria,
and digitised data files.

## Benchmarks

| Directory | Experiment | What it tests | Phase |
|-----------|-----------|---------------|-------|
| `edwards/` | Edwards & O'Brien (1970) pipe blowdown | Pressure waves, flashing, two-phase depressurisation | 2 |
| `bennett/` | Bennett heated tube (ORNL) | Forced convection boiling, void generation, CHF | 2 |
| `bartolomei/` | Bartolomei subcooled boiling | Subcooled void fraction profile | 2 |

## Future (Phase 3+)

| Experiment | What it tests | Phase |
|-----------|---------------|-------|
| Marviken critical flow | Choked two-phase discharge | 3 |
| Canon / Super Canon (CEA) | Vertical pipe blowdown, higher pressure | 3 |
| LOFT / ROSA / PKL | Integral LOCA transients | 3+ |

## Validation vs Verification

These are **validation** cases (comparison to experimental data), not verification
(comparison to analytical solutions). Solver verification is done with SimpleFluid
in `solver/tests/test_two_phase.py`. See the QA agent for the full V&V hierarchy.
