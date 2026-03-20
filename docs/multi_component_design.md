# Multi-Component Coupling Architecture — Phase 3 Design

## Problem Statement

OPAL currently simulates a single pipe connected to two boundary components.
Phase 3 requires multi-component systems: pipe networks, pipe + pump, pipe + heat
exchanger, primary loop + BOP. This document designs the coupling architecture.

## Current Limitations

The single-pipe assumption is embedded in:
1. **Extraction pipeline** — `pipe1d_mapper.py` assumes one pipe with prefix "pipe."
2. **Bridge variable groups** — `_build_var_group` uses a single prefix
3. **Solver state management** — single arrays for p, alpha, h_l, h_v, mdot
4. **Equation classification** — regex patterns assume one pipe's variable naming
5. **Boundary handling** — inlet_closed / p_out hardcoded for wall + pressure BC

## Design Principles

1. **Component-level extraction**: each Modelica component maps to its own
   variable group in the bridge, with its own prefix
2. **Connection-level coupling**: Modelica `connect()` generates coupling
   equations that OM resolves into shared variables at connection points
3. **Subsystem partitioning**: the solver partitioner identifies which components
   form a coupled subsystem (e.g., primary loop) vs independent subsystems
4. **One bridge per system**: a single bridge .so evaluates ALL equations,
   but the solver's state management tracks per-component arrays

## Architecture

### Level 1: OM Extraction (unchanged)

OM's `translateModel` already handles multi-component systems correctly.
The generated C code evaluates ALL equations for ALL components. The bridge
codegen wraps these into the same flat-array interface. No change needed here.

### Level 2: Bridge Variable Mapping (needs extension)

Currently: `_build_var_group('p', N)` looks for `pipe.p[1]..pipe.p[N]`.

Extension: multi-prefix support:
```python
bridge = OMEquationBridge(so_path, info)
# Auto-detect all pipe-like components
prefixes = bridge.detect_prefixes()  # ['pipe1', 'pipe2', 'pump1']
# Build variable groups per component
for prefix in prefixes:
    bridge.build_component_vars(prefix)
```

### Level 3: Solver Orchestration (new)

The solver needs to manage state for multiple components. Options:

**Option A: Monolithic solver**
- Single pressure tridiagonal spanning all connected components
- Pros: implicit coupling between components
- Cons: complex assembly, not modular

**Option B: Component-level solvers with coupling iteration**
- Each pipe has its own `BridgeDriftFluxSolver`
- Coupling via shared boundary pressures/enthalpies at connection points
- Pros: modular, reuses existing solver
- Cons: explicit coupling may need iteration

**Option C: Hybrid (recommended)**
- Single pressure solve across the connected system (implicit acoustics)
- Per-component transport updates (explicit void/energy)
- Bridge evaluates ALL equations in one call
- Solver assembles the global pressure tridiagonal from per-component contributions

### Level 4: Connection Point Handling

At a connection between pipe1.port_b and pipe2.port_a:
- Shared variables: pressure (p), mass flow (mdot)
- OM resolves: `pipe1.mdot[N+1] = -pipe2.mdot[1]` and `pipe1.port_b.p = pipe2.port_a.p`
- The solver needs to know which cells are coupled

**Detection**: The bridge info.json contains the equation structure. Connection
equations can be identified by their pattern: they relate variables from
different components without derivatives.

## Implementation Roadmap

### Phase 3a: Two connected pipes
- Simplest multi-component case: pipe1 → pipe2 (series connection)
- Test: pressure wave propagation across the connection
- Validates: multi-prefix extraction, global pressure solve, flow continuity

### Phase 3b: Pipe + simple boundary (pump, valve)
- Pump: provides pressure rise as function of flow rate
- Valve: provides flow resistance as function of opening fraction
- These are algebraic (no states), so they affect the pressure tridiagonal
  as modified coupling coefficients

### Phase 3c: Pipe + heat exchanger
- Two pipes with wall heat coupling
- Primary side: pipe1 with wall heat = f(T_wall, T_fluid)
- Secondary side: pipe2 with wall heat = f(T_wall, T_fluid)
- T_wall from a heat structure model (1D radial conduction)

### Phase 3d: Closed loop
- pipe1 → pipe2 → pump → pipe1 (circular topology)
- The pressure tridiagonal becomes cyclic (not strictly tridiagonal)
- Solution: either break the loop at one point (explicit coupling)
  or use a cyclic Thomas algorithm

## Key Questions for Phase 3

1. **Does OM's translateModel handle multi-pipe systems?** Test with a simple
   two-pipe series connection and verify the bridge codegen produces valid C.

2. **How does OM name variables for multi-component systems?** Does it use
   `pipe1.p[1]`, `pipe2.p[1]`, etc.? The info_parser needs to handle this.

3. **What does the pressure tridiagonal look like for connected pipes?**
   At the connection: pipe1's outlet cell is coupled to pipe2's inlet cell
   through the shared pressure variable.

4. **Can the bridge evaluate selectively?** For subsystem partitioning, we
   may want to evaluate only primary-side equations during a nuclear
   iteration, then only BOP-side during a BOP iteration.

## Next Steps

1. Create a test model: `TwoPipeSeriesTest.mo` — two Pipe1D instances connected
2. Extract it through the pipeline: `translate_and_extract('TwoPipeSeriesTest')`
3. Inspect the bridge variables: are both pipes' variables accessible?
4. Design the global pressure tridiagonal assembly
5. Implement and test: verify pressure wave crosses the connection correctly
