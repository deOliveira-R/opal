# AI-Assisted Failure Diagnosis

## Purpose

Two-phase TH simulations fail frequently. Recovery requires expert knowledge most users don't have. This skill accumulates that expertise and exposes it via MCP.

## Architecture

AI operates BETWEEN runs, not inside the solver loop. LLM latency (500-2000ms) vs. solver step (1-10ms) makes in-loop AI impossible. Non-deterministic solver decisions also kill V&V.

## Failure Context Schema

When a simulation fails, OPAL generates a structured snapshot:

| Field | Content |
|---|---|
| What failed | Which solver, which equation set, convergence vs. stability |
| Where | Which cells/components, what region of the plant |
| When | Simulation time, transient phase, time since initiating event |
| State at failure | P, T, α, flow rates in failing region + neighbors |
| Trajectory | Last N timesteps of key variables (oscillating? diverging? bounded?) |
| Solver internals | Newton residuals, Jacobian condition, timestep history |
| Model config | Active closures, parameter values, discretization |

Define this schema during Phase 1, extend in Phase 2.

## Failure Classification

**Category A — Model/setup error:** bad ICs, wrong connections, nonsensical parameters, coarse discretization, missing BCs. Fix is model-side. Signature: early failure, failure at benign conditions.

**Category B — Numerical hard spot:** correct model, solver can't handle the physics (loop seal clearing, water hammer, rapid depressurization). Fix is solver-side. Signature: failure at specific transient event after period of success.

**Always check A before B.** A Category B fix on a Category A problem = converged wrong answer.

## Skill Development (Four Artifacts Per Failure)

1. `scripts/` — Test case reproducing the exact failure
2. Solution verified against the test
3. `references/` — Error signature → root cause → fix documentation
4. `assets/` — Executable programmatic fix

## MCP Interface

- `diagnose_failure(context)` → diagnosis, root cause, executable fix
- `list_known_failures(filter)` → browse corpus by symptom/component/transient
- `retry_with_fix(checkpoint, fix)` → apply fix, rerun from checkpoint

## Checkpoint & Replay

The solver saves full state at configurable intervals. AI suggests fix → OPAL rewinds to last good checkpoint → retries. Not from t=0.

## Subdirectories

- `schema/` — Failure context schema definitions
- `corpus/category_a/` — Model/setup error entries
- `corpus/category_b/` — Numerical hard spot entries
- `scripts/` — Reproducible failure test cases
- `assets/` — Programmatic fix templates
- `references/` — Error→cause→fix documentation
- `mcp_server/` — MCP server implementation
