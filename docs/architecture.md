# OPAL Architecture

## Option 4 Design

1. **Modelica front end** — Models in standard Modelica. Component libraries for pipes, pumps, valves, HXs, pressurizers, cores, turbines, condensers, controls. Acausal connections. **ALL physics lives here** — conservation equations, closures, property evaluation, boundary conditions.

2. **OpenModelica compiler** — Parsing, flattening (inheritance + connections), index reduction (Pantelides), BLT decomposition (sorting), tearing (iteration variables). We don't reimplement this.

3. **Extraction + code generation** — Two complementary OM outputs:
   - `dumpXMLDAE(backEnd)` → equation **structure**: variable lists, adjacency matrix, BLT order, equation text (with function calls as symbolic references). Used by the **partitioner** to identify which equations are mass/momentum/energy, which variables are states, and how they couple.
   - `translateModel` → compiled **C code**: all Modelica functions fully expanded into C, each equation as a callable C function, property constants embedded. Used at **solver runtime** for equation evaluation.

4. **Custom solver backend** — The solver is a **numerical engine**, not a physics implementation. It receives the equation structure from the partitioner and calls the OM-generated C code for evaluation. Specialized subsystems (two-phase primary) → semi-implicit staggered-mesh. Well-behaved subsystems (BOP, controls) → IDA/DASSL. Partitioner handles coupling.

### Cardinal Rule

**The solver does NOT reimplement physics.** Conservation equations, fluid properties, closure correlations, friction models, reconstruction schemes — all of these are defined in Modelica and come through the extraction/codegen pipeline. The C++ solver provides numerical methods (tridiagonal pressure solve, operator splitting, Thomas algorithm) that Modelica's standard DAE solvers cannot handle for two-phase flow. If a physics change is needed, it is made in the Modelica model — never by editing the solver C++.

The C++ solver implementations built during Phase 2–3 (HEMModel, FiveEqModel, closures, friction, MUSCL, etc.) served as prototypes to verify the numerical methods. They will be superseded by the extraction-driven path as the pipeline matures.

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

## OpenModelica Extraction — Key Findings

Verified during feasibility (Phase 0) and Phase 3b:

1. **`dumpXMLDAE(backEnd)`** preserves function calls as symbolic references even for pure Modelica functions. `annotation(Inline=true)` does NOT affect XML output. The XML is for structural analysis, not numerical evaluation.

2. **`translateModel`** compiles ALL Modelica functions to C. Per-equation C functions are generated (e.g., `Model_eqFunction_52()`). Function bodies are fully expanded. Property constants are embedded. This is the runtime evaluation path.

3. **MSL media (IF97_Utilities)** uses `external "C"` → opaque in both XML and C. OPAL's pure-Modelica media (SimpleFluid.mo, Water.mo) avoid this — their functions compile cleanly to OM-generated C.

4. **`inStream()`** is expanded by the compiler at `flat` level — no residual calls in backEnd XML or C code.

5. **The partitioner uses `dumpXMLDAE` for structure; the solver uses `translateModel` C code for evaluation.** These are complementary, not alternatives.

## OpenModelica Relationship

We treat OM as a dependency we're willing to improve:
1. Navigate and understand (use Claude to read source)
2. Document our findings
3. Contribute general improvements upstream (XML export, bugs, docs)
4. Keep OPAL-specific extensions in our fork/layer

## Build Path (Detailed)

### Phase 1 — Single-phase solver coupling — COMPLETE
- Single-phase pipe loop in Modelica, extracted via OpenModelica
- C++ semi-implicit solver verified: Hagen-Poiseuille to machine precision
- 8 solver tests, end-to-end pipeline proven

### Phase 2 — Two-phase solver plugin — COMPLETE
- C++ prototype: HEM + 5-eq drift-flux, IAPWS-IF97, semi-implicit staggered mesh
- Edwards blowdown validated (C++ prototype: 149% MAPE)
- 330 C++ tests, QA methodology established (L0 term verification)
- C++ prototype archived in `archive/cpp_prototype/`

### Phase 2.5 — Modelica parity + extraction pipeline — COMPLETE
- ALL physics migrated to Modelica (.mo files)
- Pipe1D (HEM) + Pipe1D_DriftFlux (5-eq) with replaceable Medium
- MUSCL limiters, critical flow, two-phase friction in Numerics/
- Extraction pipeline: xml_reader → equation_classifier → semi-implicit solver
- Edwards via Modelica extraction: 79.8% MAPE (best result)
- 219 Modelica-side tests, QA-audited (4 rounds, final PASS)
- Case comparison framework: Case 0 (hardcoded) vs Case 1 (extracted params)

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
