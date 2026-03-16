# Feasibility Phase — Equation Extraction Tests

This is the decision gate for the entire OPAL project.

## Tests

### Test 1: Basic equation extraction
Write a simple TH loop (heated pipe, pump, heat sink, single-phase). Extract using all APIs:
- `instantiateModel()` → flat Modelica (human-readable, pre-sorting)
- `dumpXMLDAE(model, translationLevel="flat")` → XML of flat DAE
- `dumpXMLDAE(model, translationLevel="optimiser")` → XML after sorting/matching
- `dumpXMLDAE(model, translationLevel="backEnd")` → XML after index reduction, BLT, tearing
- `dumpXMLDAE(model, translationLevel="stateSpace")` → state-space form
- `translateModelXML()` → CasADi-compatible XML export

### Test 2: Information completeness
For each level: Can we reconstruct ODE/DAE? Jacobian sparsity? Incidence matrix? Events preserved? Component traceability? `stream` semantics resolved? Media calls inlined or opaque?

### Test 3: Modelica.Fluid compatibility
Test with `Modelica.Fluid` components. Specifically test `Modelica.Media.Water.StandardWater` (IAPWS-IF97) — most likely failure point.

### Test 4: Scale test
Simple PWR primary + SG secondary. Check extraction time and equation count scaling.

### Test 5: 3D array component extraction
Simplified 3D vessel: single-phase, 3×3×5 mesh, Approach B (monolithic). Check: array indices preserved? For-loop structure recoverable? BLT keeps vessel together? Extraction time?

## Decision Criteria

- All pass → proceed to solver backend design
- Tests 1-2 pass, Test 3 fails → write own simplified media library or fix OpenModelica
- Tests 1-2 partial (missing sparsity/traceability) → extend OpenModelica XML export
- Test 1 fails → pivot to Option 3 (FMI co-simulation)
- Test 5 fails → vessel internals in C++ via `external "C"`, rest of architecture unaffected

## Extraction API Reference

```python
from OMPython import OMCSessionZMQ
omc = OMCSessionZMQ()
omc.sendExpression('loadModel(Modelica)')
omc.sendExpression('loadFile("MyModel.mo")')
flat = omc.sendExpression('instantiateModel(MyModel)')
omc.sendExpression('dumpXMLDAE(MyModel, translationLevel="backEnd")')
omc.sendExpression('translateModelXML(MyModel)')
```
