# OPAL — Open Platform for Analytical Thermalhydraulics

## What This Is

Thermal-hydraulic simulation platform: Modelica front end, OpenModelica compiler, custom solver backend for two-phase flow. Independent, open.

Architecture: "Option 4" — extract equations from OpenModelica, route to purpose-built solvers. See `@docs/architecture.md` for full design.

## Current Phase

**Phase 2 — Two-phase solver.** Feasibility proven (Phase 0), single-phase solver built and verified (Phase 1). Now implementing property-dependent two-phase semi-implicit solver with IAPWS-IF97. See `docs/architecture.md` for full roadmap.

## Repository Layout

```
opal/
├── feasibility/       # Phase 0: extraction tests — COMPLETE (has its own CLAUDE.md)
├── solver/            # Custom solver backend (has its own CLAUDE.md)
│   ├── single_phase/  #   Phase 1 C++ solver — COMPLETE
│   ├── partitioner/   #   Equation routing (XML → grid) — COMPLETE
│   └── tests/         #   8 solver tests + 36 partitioner tests
├── library/           # OPAL Modelica component library (has its own CLAUDE.md)
│   └── Media/         #   IAPWS-IF97 (Water.mo) + SimpleFluid.mo + tests
├── diagnostics/       # AI failure diagnosis skill (has its own CLAUDE.md)
├── docs/              # Detailed architecture, physics, design docs
│   ├── architecture.md
│   ├── vessel.md
│   ├── kinetics.md
│   ├── realtime.md
│   ├── diagnostics.md
│   ├── extraction_failure_modes.md
│   ├── openmodelica_internals.md
│   └── xs_format.md
├── external/          # OpenModelica (submodule), Python venv, requirements.txt
└── .claude/
    ├── agents/        #   QA agent, solver-architect agent
    └── commands/      #   /verify-iapws, /verify-solver, etc.
```

## Build Path (Summary)

Phase 1: Single-phase solver coupling (proves extraction→solver pipeline)
Phase 2: Two-phase solver plugin + oracle benchmarking
Phase 2.5: Second-order spatial accuracy (MUSCL + slope limiters)
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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **OPAL** (347557 symbols, 493865 relationships, 300 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/OPAL/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/OPAL/context` | Codebase overview, check index freshness |
| `gitnexus://repo/OPAL/clusters` | All functional areas |
| `gitnexus://repo/OPAL/processes` | All execution flows |
| `gitnexus://repo/OPAL/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## CLI

- Re-index: `npx gitnexus analyze`
- Check freshness: `npx gitnexus status`
- Generate docs: `npx gitnexus wiki`

<!-- gitnexus:end -->
