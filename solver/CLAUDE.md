# OPAL Solver Backend

## Architecture

**Physics lives in Modelica. The solver provides ONLY numerical methods.**

The extraction pipeline takes Modelica models → OpenModelica extraction → equation
classification → semi-implicit operator splitting. The solver never reimplements
physics (closures, friction, critical flow) — those come from the Modelica .mo files.

Two OM outputs work together:
- `dumpXMLDAE(backEnd)` → equation **structure** (partitioner uses this)
- `translateModel` → compiled **C code** (solver runtime evaluation — Case 2, **COMPLETE**)

## Extraction Pipeline

```
Case 1: Modelica (.mo) → OM dumpXMLDAE → XML → xml_reader → pipe1d_mapper
          → equation_classifier → Parameterized5EqSolver → results

Case 2: Modelica (.mo) → OM translateModel → C code → bridge_codegen → .so
          → OMEquationBridge → BridgeDriftFluxSolver → results (PRODUCTION PATH)
```

### Key files

| File | Purpose |
|------|---------|
| `partitioner/xml_reader.py` | Parse OM XML → EquationSystem |
| `partitioner/pipe1d_mapper.py` | EquationSystem → Pipe1DGridSpec |
| `partitioner/equation_classifier.py` | Classify equations by role |
| `partitioner/model_spec.py` | ExtractedModelSpec — complete model spec from XML |
| `partitioner/extracted_solver.py` | 3-eq HEM semi-implicit solver |
| `partitioner/parameterized_5eq_solver.py` | 5-eq solver (Case 1: params from extraction) |
| `partitioner/bridge_solver.py` | HEM bridge solver (Case 2) |
| `partitioner/bridge_5eq_solver.py` | **5-eq bridge solver (Case 2, PRODUCTION)** |
| `partitioner/codegen/translate_model.py` | End-to-end: Modelica → OM → bridge .so |
| `partitioner/codegen/bridge_codegen.py` | Generate C bridge from OM equations |
| `partitioner/codegen/equation_bridge.py` | Python wrapper: set_state → evaluate → get |
| `partitioner/codegen/info_parser.py` | Parse _info.json for variable/param metadata |

### Validation drivers

| File | What |
|------|------|
| `edwards_bridge_5eq_validation.py --model hf_ramp` | **5-eq bridge, HF+Ramp (28.3% MAPE, canonical)** |
| `edwards_bridge_5eq_validation.py --model ramp` | 5-eq bridge, RT+Ramp (31.8% MAPE) |
| `edwards_5eq_modelica_validation.py` | 5-eq Case 1 Edwards (30% MAPE) |
| `edwards_modelica_validation.py` | HEM + IAPWS + critical flow (81% MAPE) |

## Semi-Implicit Method

1. Set simulation time (for time-varying BCs like RampedBreak)
2. Evaluate ALL physics via bridge (properties, closures, drift-flux split, friction)
3. Compute implicit friction resistance: `sigma`, `beta_eff = beta / (1 + sigma)`
4. Assemble pressure tridiagonal (with `beta_eff` coupling)
5. Thomas algorithm → new pressure
6. Update momentum (inertial, with implicit friction)
7. Update void fraction (conservative: alpha*rho_v product, divide by new rho_v)
8. Update phasic enthalpies (donor-cell or MUSCL, using _old values)
9. Apply nucleation floor (Gamma > 0 → alpha >= 1e-3)

## Key Numerical Techniques

- **Implicit friction resistance**: `beta_eff = beta/(1+sigma)` smoothly regularizes
  the pressure tridiagonal without floors or hacks. Derived from semi-implicit
  friction treatment. See `bridge_5eq_solver.py`.
- **Conservative void update**: alpha*rho_v product update, divide by new rho_v.
  Handles rapid depressurization where rho_v changes by 60x.
- **MUSCL reconstruction**: Optional minmod TVD limiter for second-order enthalpy
  advection. Selectable via `reconstruction='muscl'`.

## Subdirectories

- `partitioner/` — Extraction pipeline (xml_reader, classifier, solvers, codegen)
- `tests/` — 830 tests (C++ reference + Modelica + bridge + QA gaps + MMS + HF)
- `two_phase/` — Compiled .so (property evaluation + C++ reference tests)
- `single_phase/` — Compiled .so (Phase 1 reference tests)

## Three-Case Comparison Framework

- **Case 0**: `extracted_5eq_solver.py` — hardcoded closure parameters (superseded)
- **Case 1**: `parameterized_5eq_solver.py` — all parameters from Modelica extraction
- **Case 2**: `bridge_5eq_solver.py` — **ALL physics from OM-generated C (PRODUCTION)**

## After Modelica Changes: Pipeline Verification

The pipeline between Modelica and solver is a failure surface. After any `.mo` change:

1. **Recompile bridge:** `translate_and_build('ModelName')` — must succeed
2. **Check variable presence:** `bridge.has('new_var')` for any new variables
3. **Check boundary faces:** Conservation identities at face 0 and face N
4. **Check activation:** Run with feature ON and OFF — results must differ
5. **Run `/checklist`** before writing solver code that uses new bridge variables

See `library/CLAUDE.md` "OpenModelica Pitfalls" for known silent failure modes.

## C++ Prototype (Archived)

The C++ solver source (.hpp, .cpp) has been moved to `archive/cpp_prototype/`.
The compiled .so files remain here for test compatibility and property evaluation.
**DO NOT create new C++ files. Physics changes go in `library/*.mo`.**
