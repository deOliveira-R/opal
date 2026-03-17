# OPAL Architecture

## Option 4 Design

1. **Modelica front end** — Models in standard Modelica. Component libraries for pipes, pumps, valves, HXs, pressurizers, cores, turbines, condensers, controls. Acausal connections.

2. **OpenModelica compiler** — Parsing, flattening (inheritance + connections), index reduction (Pantelides), BLT decomposition (sorting), tearing (iteration variables). We don't reimplement this.

3. **Equation extraction** — Flat sorted system via OM APIs (`dumpXMLDAE`, `translateModelXML`, `instantiateModel`). Gives equations, variable list, incidence matrix, sparsity.

4. **Custom solver backend** — Specialized subsystems (two-phase primary) → semi-implicit staggered-mesh solver. Well-behaved subsystems (BOP, controls) → IDA/DASSL. Partitioner handles coupling.

### What this buys over standard Modelica
- Purpose-built two-phase numerics where generic solvers fail
- Multi-domain coupling (TH + controls + electrical) in one model
- Modelica ecosystem for non-nuclear components
- Open, no export controls

### What this buys over established system codes
- Standard language instead of proprietary input format
- BOP from existing Modelica libraries
- Extensible by anyone who knows Modelica
- Multi-physics coupling is native

## OpenModelica Relationship

We treat OM as a dependency we're willing to improve:
1. Navigate and understand (use Claude to read source)
2. Document our findings
3. Contribute general improvements upstream (XML export, bugs, docs)
4. Keep OPAL-specific extensions in our fork/layer

## Build Path (Detailed)

### Phase 1 — Single-phase solver coupling
- Simple single-phase pipe loop in Modelica
- Extract equations from OpenModelica
- Route to basic semi-implicit solver (single-phase)
- Verify: conservation, Hagen-Poiseuille, Joukowski waterhammer

### Phase 2 — Two-phase solver plugin
- Semi-implicit two-phase solver as pluggable backend
- Property-dependent EOS: ρ(p,h), ∂ρ/∂p|h, ∂ρ/∂h|p from IAPWS-IF97
- Equation routing: which subsystem → which solver
- First-order spatial (donor-cell) and temporal (implicit Euler / forward Euler)
- Verify with SimpleFluid (synthetic linear fluid, isolates solver from properties)
- Validate against Edwards blowdown, benchmark against oracle

### Phase 2.5 — Second-order spatial accuracy
- MUSCL reconstruction with slope limiter (minmod default, van Leer for void fronts)
- Face enthalpy: h_face = h_upwind + 0.5 · φ(r) · (h_down − h_up), all algebraic
- Pressure solve unchanged (tridiagonal) — MUSCL only affects advective fluxes
- Optional: predictor-corrector temporal for energy (~20-30% cost, second-order time)
- Stencil widens from 2-point to 4-point for energy, partitioner updated accordingly
- Pure Modelica expressions (min/max/if) — no OPAQUE risk, extraction-transparent
- Motivation: first-order donor-cell smears thermal/void fronts over ~√N cells;
  second-order enables coarse-mesh accuracy (20 cells ≈ 50 first-order cells),
  directly benefiting real-time performance target
- Industry precedent: RELAP5/CATHARE-2 started first-order, TRACE/CATHARE-3/ATHLET
  all upgraded to second-order MUSCL for front tracking (boron, temperature, void)

### Phase 3 — Multi-domain demonstration
- Complete plant: reactor + SG + turbine + condenser + feedwater + controls
- Primary → custom solver, BOP → Modelica solver, controls → Modelica solver
- Run plant transient (turbine trip, loss of feedwater)
- **Real-time benchmark:** target within 5× of real time

### Phase 4 — Component library + point kinetics
- Open Modelica component library for nuclear TH
- Point kinetics (6-group, ANS 5.1 decay heat, Xe/Sm)
- Own media package (pure-Modelica IAPWS-IF97 + analytical derivatives)
- All components event-free for real-time

### Phase 4.5 — 3D vessel component
- Approach B monolithic vessel, single-phase first then two-phase
- Extend semi-implicit: tridiagonal → sparse 7-diagonal
- Nozzle coupling: vessel boundary cells ↔ 1D pipe network
- Verify against RELAP5/TRACE vessel benchmarks (if possible)

### Phase 5 — 3D spatial kinetics
- Few-group neutron diffusion solver (C++)
- Eigenvalue solver for initialization (power iteration + Wielandt shift)
- Quasi-static transient: amplitude (Modelica DAE) + shape (C++ periodic solve)
- Cross-section data: HDF5 format, converters from Serpent/OpenMC/DRAGON5
- Verify against PARCS/DYN3D published benchmarks

### Phase 6 — Real-time mode
- Fixed-step solver for BOP/controls (bounded Newton iterations, no convergence check)
- Verify consistency between analysis mode and real-time mode for slow transients
- Performance optimization, bottleneck identification
- Demonstrate real-time on full-plant training-simulator scenario
