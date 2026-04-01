# 6-Equation Edwards Blowdown Test Models

Modelica test models for the 6-equation two-fluid Edwards blowdown validation.

Uses `Pipe1D_TwoFluid` (library/Pipes/Pipe1D_TwoFluid.mo) with separate
phasic momentum equations and interfacial drag.

## Current models

- `EdwardsTest_TwoFluid_HF_Ramp.mo` — Base 6-eq model (Henry-Fauske + RampedBreak)

## Compiled bridges

Built artifacts go to `feasibility/results/edwards_6eq/`.

Build with:
```python
from partitioner.codegen.translate_model import translate_and_build
translate_and_build("EdwardsTest_TwoFluid_HF_Ramp")
```
