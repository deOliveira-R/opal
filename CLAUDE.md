# OPAL — Open Platform for Analytical Thermalhydraulics

## What This Is

Thermal-hydraulic simulation platform: Modelica front end, OpenModelica compiler, custom solver backend for two-phase flow. Independent, open.

Architecture: "Option 4" — extract equations from OpenModelica, route to purpose-built solvers. See `@docs/architecture.md` for full design.

## Current Phase

**Phase 3 preparation.** Feasibility (Phase 0), single-phase solver (Phase 1), two-phase solver (Phase 2), and Modelica parity (Phase 2.5) complete. Edwards blowdown validated at 79.8% MAPE through the Modelica extraction pipeline. Now preparing for multi-component systems (Phase 3).

## Cardinal Rule

**ALL physics lives in Modelica.** The solver provides ONLY numerical methods (operator splitting, tridiagonal solve, Thomas algorithm). If a physics change is needed, edit the `.mo` files — never the solver. See `docs/architecture.md`.

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
│   ├── tests/         #   549 tests (330 C++ reference + 219 Modelica-side)
│   ├── two_phase/     #   Compiled .so for C++ property evaluation + tests
│   └── single_phase/  #   Compiled .so for Phase 1 tests
├── feasibility/       # Phase 0: extraction tests — COMPLETE
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
├── archive/           # C++ prototype solver — DO NOT EDIT (reference only)
│   └── cpp_prototype/ #   Source for two_phase + single_phase C++ solvers
├── external/          # OpenModelica, reference libraries, Python venv
└── .claude/
    ├── agents/        #   QA agent, solver-architect agent
    └── commands/      #   /verify-iapws, /verify-solver, etc.
```

## DO NOT EDIT

- `archive/cpp_prototype/` — C++ solver source (frozen, reference only)
- `solver/two_phase/*.so` — compiled C++ (for property evaluation + tests, do not recompile)
- Any C++ file (.hpp, .cpp) — physics belongs in Modelica

## Build Path (Summary)

- ~~Phase 0: Feasibility (extraction tests)~~ — **COMPLETE**
- ~~Phase 1: Single-phase solver coupling~~ — **COMPLETE**
- ~~Phase 2: Two-phase solver plugin + IAPWS-IF97~~ — **COMPLETE**
- ~~Phase 2.5: MUSCL, Modelica parity, extraction pipeline~~ — **COMPLETE**
- **Phase 3: Multi-domain plant demo + real-time benchmark** — NEXT
- Phase 4: Component library + point kinetics
- Phase 4.5: 3D vessel component
- Phase 5: 3D spatial kinetics (few-group diffusion)
- Phase 6: Real-time mode (fixed-step, bounded iterations)

Full phase descriptions: `@docs/architecture.md`

## Edwards Blowdown Validation (current best)

| Model | Physics source | MAPE |
|-------|---------------|------|
| Modelica HEM + IAPWS | Pipe1D.mo + Water.mo | 100.7% |
| Modelica HEM + IAPWS + critical flow | + CriticalFlow.mo | 81.0% |
| Modelica 5-eq drift-flux | Pipe1D_DriftFlux.mo (all closures) | 79.8% |

## Working With Claude

### Where Claude should focus
- Modelica models (.mo files) — ALL physics changes go here
- Extraction pipeline (solver/partitioner/) — structural analysis + numerical methods
- Tests — QA rigor for AI-generated code (L0 term verification)

### Where Claude must NOT implement physics
- `archive/cpp_prototype/` — frozen, reference only
- `solver/two_phase/*.so` — compiled binary, do not modify
- Any new C++ file — physics belongs in Modelica

### Where Claude needs human verification
- Physical completeness of extracted equations
- OpenModelica behavior vs. its documentation
- Solver stability for novel partitioning (must run, not just reason)
- Real-time performance claims (measure, don't estimate)

### Self-correction approach
- Conservation checks at every level
- Limiting-case tests (must reduce to known solutions)
- Benchmark against experimental data (Edwards blowdown)
- On failure: determine if bug is in Modelica model, extraction, or solver numerics

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **OPAL** (1118 symbols, 2849 relationships, 28 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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

## Keeping the Index Fresh

After committing code changes, the GitNexus index becomes stale. Re-run analyze to update it:

```bash
npx gitnexus analyze
```

If the index previously included embeddings, preserve them by adding `--embeddings`:

```bash
npx gitnexus analyze --embeddings
```

To check whether embeddings exist, inspect `.gitnexus/meta.json` — the `stats.embeddings` field shows the count (0 means no embeddings). **Running analyze without `--embeddings` will delete any previously generated embeddings.**

> Claude Code users: A PostToolUse hook handles this automatically after `git commit` and `git merge`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
