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

### 2. Use Subagents — and Escalate Early
- Split complex problems: research, execution, analysis
- One task per agent for clarity
- Parallelize thinking, not just execution
- **If your approach fails twice, STOP.** Spawn solver-architect or
  physics-reviewer with what you tried and why it failed. Don't burn
  context window iterating solo — expert agents get fresh context.
- Use round-trips between specialist agents to narrow down problems

### 3. Verify Before Done — All Tests Must Pass
- Never mark done without proof
- Run tests, check logs, simulate real usage
- Compare expected vs actual behavior
- **Run full test suite before committing or ending a session.
  Zero tolerance for failing tests at session boundary — the next
  session inherits the mess with no context for why it was left broken.**
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
- Run `/checklist` before writing physics or solver code
- After implementation, spawn QA agent for full review
- Fix test gaps BEFORE developing new features
- See `solver/tests/QA_AI_CODE_METHODOLOGY.md` for L0/L1/L2 methodology

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
