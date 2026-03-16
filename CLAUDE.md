# OPAL — Open Platform for Analytical Thermalhydraulics

## What This Is

Thermal-hydraulic simulation platform: Modelica front end, OpenModelica compiler, custom solver backend for two-phase flow. Independent, open.

Architecture: "Option 4" — extract equations from OpenModelica, route to purpose-built solvers. See `@docs/architecture.md` for full design.

## Current Phase

**Feasibility testing.** Verifying that equation extraction from OpenModelica works for TH models. This is the decision gate — nothing else proceeds until extraction is proven viable. See `feasibility/CLAUDE.md` for test plan and decision criteria.

## Repository Layout

```
opal/
├── feasibility/       # Phase 0: extraction tests (has its own CLAUDE.md)
├── solver/            # Custom solver backend (has its own CLAUDE.md)
├── library/           # OPAL Modelica component library (has its own CLAUDE.md)
├── diagnostics/       # AI failure diagnosis skill (has its own CLAUDE.md)
├── tests/             # Verification, validation, benchmarking, real-time
├── docs/              # Detailed architecture, physics, design docs
│   ├── architecture.md
│   ├── vessel.md
│   ├── kinetics.md
│   ├── realtime.md
│   ├── diagnostics.md
│   ├── extraction_failure_modes.md
│   ├── openmodelica_internals.md
│   ├── xs_format.md
│   └── physics/
└── .claude/
    └── rules/         # Path-scoped coding rules
```

## Build Path (Summary)

Phase 1: Single-phase solver coupling (proves extraction→solver pipeline)
Phase 2: Two-phase solver plugin + oracle benchmarking
Phase 3: Multi-domain plant demo + real-time benchmark
Phase 4: Component library + point kinetics + own media package
Phase 4.5: 3D vessel component
Phase 5: 3D spatial kinetics (few-group diffusion)
Phase 6: Real-time mode (fixed-step, bounded iterations)

Full phase descriptions: `@docs/architecture.md`

## Oracle

We have an oracle code for benchmarking. Will sort that when necessary.

## Working With Claude

### Where Claude is strong
- Modelica models, XML parsing, OpenModelica source reading (MetaModelica/C)
- Solver backend (C++ numerical methods), equation routing/partitioning
- Neutron diffusion solver, cross-section infrastructure, IAPWS-IF97 properties

### Where Claude needs human verification
- Physical completeness of extracted equations
- OpenModelica behavior vs. its (possibly stale) documentation
- Solver stability for novel partitioning (must run, not just reason)
- `stream` connector semantics resolution
- Quasi-static kinetics coupling stability
- Cross-section data quality
- Real-time performance claims (measure, don't estimate)

### Self-correction approach
- Conservation checks at every level
- Limiting-case tests (must reduce to known solutions)
- Benchmark against oracle on identical problems
- On failure: determine if bug is in extraction, routing, or solver

## Key Differences From established system codes

OPAL: independent/open, Modelica+C++, partitioned solver, multi-domain, real-time capable, everyone else in the world. Complementary projects, not competing.
