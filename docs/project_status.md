# OPAL Project Status

> Keep this file up to date as milestones are reached. CLAUDE.md points here for reference.

## Current Phase

**Phase 3 preparation.** Phases 0–2.5 complete. Edwards blowdown validated at
**28.3% MAPE** through the full Modelica→OM→bridge pipeline (True Case 2: zero
physics in solver). Now preparing for multi-component systems (Phase 3).

## Build Path

| Phase | Status |
|-------|--------|
| Phase 0: Feasibility (extraction tests) | **COMPLETE** |
| Phase 1: Single-phase solver coupling | **COMPLETE** |
| Phase 2: Two-phase solver plugin + IAPWS-IF97 | **COMPLETE** |
| Phase 2.5: MUSCL, Modelica parity, extraction pipeline | **COMPLETE** |
| **Phase 3: Multi-domain plant demo + real-time benchmark** | **NEXT** |
| Phase 4: Component library + point kinetics | Planned |
| Phase 4.5: 3D vessel component | Planned |
| Phase 5: 3D spatial kinetics (few-group diffusion) | Planned |
| Phase 6: Real-time mode (fixed-step, bounded iterations) | Planned |

Full phase descriptions: `docs/architecture.md`

## Edwards Blowdown Validation

| Model | Physics source | MAPE |
|-------|---------------|------|
| Modelica HEM + IAPWS | Pipe1D.mo + Water.mo | 100.7% |
| Modelica HEM + IAPWS + critical flow | + CriticalFlow.mo | 81.0% |
| 5-eq Python (metastable fix) | C++ fluid + Python closures | 30.0% |
| 5-eq Bridge, RT + Ramp | ALL from Modelica | 31.8% |
| **5-eq Bridge, HF + Ramp** | **ALL from Modelica** | **28.3%** |

Key features in the 28.3% result: metastable T_l via cp_f(p), drift-flux phasic
split (V_gj + C_0), physics-based interfacial HT (Ranz-Marshall + geometric IAC),
Martinelli-Nelson Phi2, Henry-Fauske critical flow, implicit friction resistance,
RampedBreak BC (break opening ramp from Modelica).

Mesh convergence study (N=12,24,48,96) shows anti-convergence: explicit transport CFL
limits practical resolution to N~24. See `docs/validation/edwards/LESSONS_LEARNED.md`.

## Current Capabilities (QA Audit 2026-03-20)

**Working:** 1D pipe (HEM + 5-eq drift-flux), SimpleFluid + IAPWS-IF97 (R1/R2/R4),
wall/pressure/break BCs, critical flow, two-phase friction, MUSCL, full extraction
pipeline.

**Missing for Phase 3:** Multi-component coupling, pump/valve/HX components, heat
structure coupling, transport properties (mu, k), Region 3 IAPWS, time-varying BC
extraction, subsystem partitioner.

## Repository Layout

```
opal/
├── library/           # OPAL Modelica component library — THE PHYSICS SOURCE
│   ├── Pipes/         #   Pipe1D (HEM), Pipe1D_DriftFlux (5-eq)
│   ├── Media/         #   PartialMedium, SimpleFluid, Water (IAPWS-IF97)
│   ├── Boundary/      #   ClosedEnd, PressureSource, BreakSource, RampedBreak
│   ├── Connectors/    #   FluidPort (stream connector)
│   └── Numerics/      #   Limiters, CriticalFlow, TwoPhaseFriction
├── solver/            # Extraction pipeline + numerical solvers
│   ├── partitioner/   #   xml_reader, pipe1d_mapper, equation_classifier,
│   │                  #   model_spec, extracted_solver, parameterized_5eq_solver
│   ├── tests/         #   830+ tests (C++ reference + Modelica + bridge + QA)
│   ├── two_phase/     #   Compiled .so for C++ property evaluation + tests
│   └── single_phase/  #   Compiled .so for Phase 1 tests
├── feasibility/       # Phase 0: extraction tests — COMPLETE
├── diagnostics/       # AI failure diagnosis skill (has its own CLAUDE.md)
├── docs/              # Detailed architecture, physics, design docs
├── archive/           # C++ prototype solver — DO NOT EDIT (reference only)
│   └── cpp_prototype/ #   Source for two_phase + single_phase C++ solvers
├── external/          # OpenModelica, reference libraries, Python venv
└── .claude/
    ├── agents/        #   QA, solver-architect, physics-reviewer
    └── commands/      #   /verify-iapws, /verify-solver, etc.
```

## Test Counts

| Area | Count | Notes |
|------|-------|-------|
| Solver tests | 830+ | C++ reference + Modelica + bridge + QA |
| Feasibility tests | 5 | Phase 0, all passing |
| IAPWS verification | 253+ | 9 groups, against iapws oracle |
