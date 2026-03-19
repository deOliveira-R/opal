# OPAL Solver Backend

## Architecture

**Physics lives in Modelica. The solver provides ONLY numerical methods.**

The extraction pipeline takes Modelica models → OpenModelica extraction → equation
classification → semi-implicit operator splitting. The solver never reimplements
physics (closures, friction, critical flow) — those come from the Modelica .mo files.

Two OM outputs work together:
- `dumpXMLDAE(backEnd)` → equation **structure** (partitioner uses this)
- `translateModel` → compiled **C code** (solver runtime evaluation — Case 2, in progress)

## Extraction Pipeline

```
Modelica (.mo) → OpenModelica → XML → xml_reader → pipe1d_mapper
  → equation_classifier → semi-implicit solver → results
```

### Key files

| File | Purpose |
|------|---------|
| `partitioner/xml_reader.py` | Parse OM XML → EquationSystem |
| `partitioner/pipe1d_mapper.py` | EquationSystem → Pipe1DGridSpec |
| `partitioner/equation_classifier.py` | Classify equations by role (mass/momentum/energy/property) |
| `partitioner/model_spec.py` | ExtractedModelSpec — complete model spec from XML |
| `partitioner/extracted_solver.py` | 3-eq HEM semi-implicit solver |
| `partitioner/extracted_5eq_solver.py` | 5-eq drift-flux solver (Case 0: hardcoded closures) |
| `partitioner/parameterized_5eq_solver.py` | 5-eq solver (Case 1: all params from extraction) |

### Validation drivers

| File | What |
|------|------|
| `phase2_extracted_driver.py` | HEM Edwards via extraction pipeline |
| `edwards_modelica_validation.py` | HEM + IAPWS + critical flow (81% MAPE) |
| `edwards_5eq_modelica_validation.py` | 5-eq + IAPWS + critical flow (79.8% MAPE) |
| `edwards_case_comparison.py` | Case 0 vs Case 1 parameter comparison |

## Why Generic DAE Solvers Fail for Two-Phase TH

- Acoustic vs. transport timescale stiffness (speed of sound ~1000 m/s vs. flow ~5 m/s)
- Phase appearance/disappearance: void fraction at 0 or 1 → singular closures
- Flow regime transitions: discontinuous closure changes → convergence failures
- Critical flow: choked flow decouples mass flow from downstream pressure

## Semi-Implicit Method

1. Evaluate properties at old state (from Modelica via C++ FluidPackage or OM codegen)
2. Assemble pressure tridiagonal (from mass + momentum coupling)
3. Thomas algorithm → new pressure
4. Update momentum (inertial, from new pressure)
5. Update transport explicitly (enthalpy, void fraction, phasic energy)

## Subdirectories

- `partitioner/` — Extraction pipeline (xml_reader, classifier, solvers)
- `tests/` — 549 tests (330 C++ reference + 219 Modelica-side)
- `two_phase/` — Compiled .so (property evaluation + C++ reference tests)
- `single_phase/` — Compiled .so (Phase 1 reference tests)

## C++ Prototype (Archived)

The C++ solver source (.hpp, .cpp) has been moved to `archive/cpp_prototype/`.
The compiled .so files remain here for test compatibility and property evaluation.
**DO NOT create new C++ files. Physics changes go in `library/*.mo`.**

## Three-Case Comparison Framework

- **Case 0**: `extracted_5eq_solver.py` — hardcoded closure parameters (the wrong way)
- **Case 1**: `parameterized_5eq_solver.py` — all parameters from Modelica extraction
- **Case 2**: (in progress) — `translateModel` C codegen for equation evaluation
