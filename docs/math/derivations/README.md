# OPAL Derivations

This directory contains the **source of truth** for every equation in OPAL.

Each file is a self-contained Python script that:
1. States the physics (conservation law, constitutive relation)
2. Derives the discrete/linearized form using SymPy
3. Verifies the derivation symbolically and numerically
4. Generates Modelica and/or C code

## Rules

- **Never hand-edit generated code.** If a Modelica or C equation needs to change,
  change the derivation script and regenerate.
- **Every script must end with verification.** No derivation is complete without
  numerical spot-checks passing.
- **Cite references.** Every derivation script includes the reference (textbook,
  paper, code manual) for the physics being derived.
- **Scripts must be runnable standalone.** `python derivations/foo.py` must work
  and print PASS/FAIL.

## Derivation index

| Script | Equations | Status |
|--------|-----------|--------|
| `pressure_linearization.py` | Semi-implicit pressure matrix (accumulation term) | Complete |
| `face_resistance.py` | Darcy-Weisbach face resistance R = f_D·dx/(2·D_h·A²·ρ) | Complete |
| `algebraic_momentum.py` | Algebraic momentum mdot = dp/R + H-P steady-state proof | Complete |
| `enthalpy_update.py` | Donor-cell enthalpy + pressure work + conservation checks | Complete |
| `gibbs_derivatives.py` | (∂ρ/∂p)_h and (∂ρ/∂h)_p chain rule from Gibbs functions | Complete |
| `simple_fluid_derivatives.py` | SimpleFluid Region 4 analytical derivatives via SymPy diff | Complete |
| `semi_implicit_momentum.py` | Inertial momentum linearization (Phase 3) | TODO |
| `donor_cell_advection_1d.py` | 1D donor-cell advection (formal) | TODO |
| `diffusion_operator_stencil.py` | Few-group diffusion FD stencil (3D Cartesian) | TODO |
| `point_kinetics_discretization.py` | 6-group PK time integration | TODO |
