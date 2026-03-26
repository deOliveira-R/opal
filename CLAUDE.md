# OPAL — Open Platform for Analytical Thermalhydraulics

Thermal-hydraulic simulation platform: Modelica front end, OpenModelica compiler,
custom solver backend for two-phase flow. Architecture: "Option 4" — extract
equations from OpenModelica, route to purpose-built solvers.

Current state and build path: `docs/project_status.md`
Full architecture: `docs/architecture.md`
Developer guide: `docs/developer_guide.md`

## Cardinal Rule

**ALL physics lives in Modelica.** The solver provides ONLY numerical methods
(operator splitting, tridiagonal solve, Thomas algorithm). If a physics change
is needed, edit the `.mo` files — never the solver.

## Do Not Edit

- `archive/cpp_prototype/` — C++ solver source (frozen, reference only)
- `solver/two_phase/*.so` — compiled binaries, do not recompile
- Any C++ file (.hpp, .cpp) — physics belongs in Modelica

## How to Work

### 1. Plan First
- Enter plan mode for any non-trivial task (3+ steps or architecture change)
- Define both execution and verification steps
- If something breaks mid-plan — STOP and re-plan
- Write detailed specs to remove ambiguity

### 2. Use Subagents
- Split complex problems: research, execution, analysis
- One task per agent for clarity
- Parallelize thinking, not just execution

### 3. Verify Before Done
- Never mark done without proof
- Run tests, check logs, simulate real usage
- Compare expected vs actual behavior
- Ask: "Would a senior engineer approve this?"

### 4. Demand Elegance
- Ask: "Is there a simpler / cleaner way?"
- Avoid hacky or temporary fixes — solve root cause
- Optimize for long-term maintainability
- Skip overengineering for small fixes

### 5. Fix Bugs Autonomously
- Trace logs, errors, failing tests
- Find root cause, not symptoms
- Fix CI failures proactively

### 6. QA Rigor for AI-Generated Code
- Level 0: every equation term needs sign AND magnitude test
- Level 1: integration tests with full solver
- Level 2: validation against experimental data
- See `solver/tests/QA_AI_CODE_METHODOLOGY.md`

### 7. Iterate, Don't Guess
- Add features one at a time, compare to data after each step
- Don't celebrate MAPE without verifying the physics mechanisms are active
- No floors, hacks, or workarounds — every fix must be principled

## Where Claude Should Focus
- Modelica models (`.mo` files) — all physics changes
- Extraction pipeline (`solver/partitioner/`) — structural analysis + numerical methods
- Tests — QA rigor for AI-generated code (L0 term verification)

## Where Claude Needs Human Verification
- Physical completeness of extracted equations
- OpenModelica behavior vs its documentation
- Solver stability for novel partitioning (must run, not just reason)
- Real-time performance claims (measure, don't estimate)

## Self-Correction
- Conservation checks at every level
- Limiting-case tests (must reduce to known solutions)
- Benchmark against experimental data (Edwards blowdown)
- On failure: determine if bug is in Modelica model, extraction, or solver numerics

## Core Principles
- Simplicity First — minimal, clean solutions
- Systems > Prompts
- Verification > Generation
- Iteration > Perfection
- No Lazy Fixes — solve root cause
