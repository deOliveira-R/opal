# OPAL Solver Backend

## Architecture

Partitioned solver: subsystems needing specialized numerics (two-phase primary) get a semi-implicit staggered-mesh solver. Well-behaved subsystems (single-phase BOP, controls, electrical) get standard DAE solver (IDA/DASSL).

The partitioner identifies subsystems from extracted equation metadata (variable name prefixes → component origin) and routes to the appropriate solver.

## Why Generic DAE Solvers Fail for Two-Phase TH

- Acoustic vs. transport timescale stiffness (speed of sound ~1000 m/s vs. flow ~5 m/s)
- Phase appearance/disappearance: void fraction at 0 or 1 → singular interfacial correlations → near-singular Jacobian → timestep collapse
- Flow regime transitions: discontinuous closure changes (bubbly → slug → annular) → convergence failures
- Critical flow: choked flow decouples mass flow from downstream pressure

## Semi-Implicit Staggered Mesh

- Staggered mesh: scalars at cell centers, velocities at cell faces
- Semi-implicit: linearize pressure → implicit pressure solve → back-substitute velocities → explicit thermodynamic update
- Donor-cell (upwind) advection
- Regularized flow regime transitions

## Solver Modes

- **Analysis mode:** adaptive timestep, full event handling, maximum accuracy. For licensing.
- **Real-time mode:** fixed timestep, bounded Newton iterations (3-5, accept regardless), regularized everything. For training simulators.

Mode switch is solver configuration, not model change. Same Modelica source for both.

## 1D → 3D Extension (Phase 4.5)

Same semi-implicit scheme, larger sparse matrix (7-diagonal for 3D vs. tridiagonal for 1D). At 2500 cells: SuperLU direct or BiCGSTAB+ILU iterative, a few milliseconds.

## Performance Budget

At Δt=0.05s (20 steps/sec simulated), budget is 50ms/step. Training fidelity: ~2-3ms (15-20× margin). Licensing fidelity: ~10-20ms (2-5× margin).

## Subdirectories

- `single_phase/` — Phase 1 solver
- `two_phase/` — Phase 2 solver (1D)
- `vessel_3d/` — Phase 4.5 solver (3D semi-implicit)
- `partitioner/` — Equation routing logic
- `realtime/` — Phase 6 fixed-step real-time mode
