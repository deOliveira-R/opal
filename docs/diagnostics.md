# AI-Assisted Failure Diagnosis — Design Document

## Problem

Two-phase TH simulations fail frequently. The mapping from failure symptom to fix is not a simple lookup — it requires contextual reasoning over plant state, transient phase, component configuration, and physics interactions. Example: "void fraction oscillating 0.3-0.7 in hot leg during SBLOCA at 50s when loop seal clearing, pressure dropping 0.2 MPa/s" → "flow regime map fighting interfacial drag at bubbly-slug transition; widen regime transition smoothing in components 12-18, and check condensation Nusselt in pressurizer."

This is expert knowledge. Most OPAL users won't have it. The diagnostic skill captures and delivers it.

## Architecture Constraints

**No AI in the solver loop.** Semi-implicit solver: 1-10ms/step. LLM inference: 500-2000ms. In-loop AI destroys performance, kills real-time mode, makes results non-deterministic (fatal for V&V).

**AI operates between runs.** Simulation fails → OPAL dumps structured context → AI diagnoses → user or automation applies fix → OPAL retries from checkpoint.

## Failure Context Schema

Defined in `diagnostics/schema/`. Initialized during Phase 1 (single-phase, simple fields), extended in Phase 2 (two-phase fields: void fraction, flow regime, interfacial terms).

Core fields:

- **failure_type**: `convergence` | `stability` | `physical_bound` | `timestep_collapse`
- **solver_id**: which solver partition failed
- **failing_equations**: list of equation indices with largest residuals
- **failing_components**: Modelica component names (from variable name prefixes)
- **failing_cells**: cell indices (for spatial components)
- **sim_time**: simulation time at failure
- **transient_phase**: user-tagged or auto-detected (steady-state, blowdown, reflood, etc.)
- **state_snapshot**: P, T, α, ṁ, ρ for failing region + neighbors
- **trajectory**: last N timesteps of key variables in failing region
- **solver_history**: Newton residual norms, Jacobian condition estimates, dt history
- **model_config**: active closures, key parameter values, mesh sizes

## Failure Classification

### Category A: Model/Setup Error

The model or its inputs are wrong. The solver is fine.

**Signatures:**
- Failure at t=0 or first few timesteps (bad initial conditions)
- Failure at conditions that should be trivial (single-phase, low power, no transient)
- Unphysical state values at failure (negative pressure, α > 1, T > 10000K)
- Failure localized to a single component with unusual parameter values

**Common causes:**
- Inconsistent initial conditions (P and T don't match saturation state)
- Missing or wrong boundary conditions
- Wrong sign on heat flux or flow direction
- Pipe discretization too coarse for the geometry
- Pump curve outside operating range
- Valve coefficient off by orders of magnitude

**Fixes are always model-side:** change parameters, fix connections, add/refine components.

### Category B: Numerical Hard Spot

The model is correct. The physics is genuinely hard for the solver.

**Signatures:**
- Failure at a specific transient event after period of successful simulation
- Failure coincides with known difficult phenomena (loop seal clearing, phase transition, flow reversal, critical flow onset)
- Solver history shows gradual timestep reduction before failure
- Jacobian condition number spiking

**Common causes and fixes:**
- Phase appearance/disappearance → widen regularization band for α near 0 or 1
- Flow regime transition → increase regime transition smoothing width
- Water hammer / pressure wave → reduce max timestep, increase mesh density locally
- Counter-current flow limitation → adjust CCFL correlation parameters
- Critical flow at break → check choking model activation, verify break area
- Loop seal clearing → reduce dt during clearing period, check condensation model

**Fixes are solver/parameter-side:** timestep limits, regularization widths, scheme options.

## Checkpoint and Replay

The solver saves full state at configurable intervals (default: every 100 steps or every 1s simulated, whichever is more frequent). On failure:

1. OPAL identifies the last good checkpoint before the failure
2. AI diagnoses the failure and proposes a fix
3. Fix is applied (parameter change, solver setting, model adjustment)
4. Simulation restarts from checkpoint, not from t=0

Checkpoints must include: all state variables, all solver internal state (Newton iteration state, timestep controller state), all event indicators, RNG state if any.

## MCP Server

Thin wrapper. Three tools:

### `diagnose_failure`
- Input: failure context (structured JSON matching schema)
- Output: classification (A or B), root cause hypothesis, confidence, suggested fix as executable parameter changes
- Always returns Category A hypotheses first if plausible

### `list_known_failures`
- Input: optional filters (component type, transient type, failure type, keyword)
- Output: matching entries from corpus with summaries

### `retry_with_fix`
- Input: checkpoint ID, fix specification (parameter changes as key-value pairs)
- Output: triggers OPAL to reload checkpoint, apply changes, resume simulation
- Returns new simulation status (success, new failure, completed)

## Knowledge Base Growth

The corpus starts empty and grows during OPAL development. Every developer fighting a solver failure is generating training data. The discipline is recording it in the four-artifact format instead of just fixing and moving on.

Expected corpus size at each phase:
- Phase 1 (single-phase): ~10-20 entries, mostly Category A
- Phase 2 (two-phase): ~50-100 entries, mix of A and B
- Phase 3 (multi-domain): ~30-50 more, mostly coupling-related
- Phase 4+ (users): unbounded, dominated by Category A (new users making setup mistakes)

The corpus is version-controlled alongside the code. Entries reference specific OPAL versions since failure signatures may change as the solver improves.
