# 3D Vessel Component

The vessel (RELAP5 VESSEL / TRACE VESSEL equivalent) is a 3D thermal-hydraulic volume for the reactor pressure vessel. Required for spatial kinetics coupling and separates a system code from a piping network model.

## Design: Monolithic Equation-Based (Approach B)

Single Modelica component with internal array equations. NOT array-of-connected-cells (Approach A) because:
- Approach A generates thousands of connection equations that overwhelm the flattener
- Approach A's `stream` connectors are for 1D pipe networks, not inter-cell fluxes on structured meshes
- Approach A's extraction output loses mesh structure — partitioner can't reconstruct topology
- Approach B flattens cleanly, preserves array structure in variable names, presents as one BLT block

## Modelica Structure

```modelica
model Vessel
  parameter Integer Nr = 5;
  parameter Integer Ntheta = 6;
  parameter Integer Nz = 20;

  // Scalar fields at cell centers
  Real P[Nr, Ntheta, Nz](each start = P_init);
  Real alpha[Nr, Ntheta, Nz](each start = 0.0);
  Real T_l[Nr, Ntheta, Nz];
  Real T_v[Nr, Ntheta, Nz];
  Real rho_l[Nr, Ntheta, Nz];
  Real rho_v[Nr, Ntheta, Nz];

  // Velocities at cell faces (staggered mesh)
  Real v_r[Nr+1, Ntheta, Nz];
  Real v_theta[Nr, Ntheta, Nz];
  Real v_z[Nr, Ntheta, Nz+1];

  // Heat source from kinetics
  input Real q_vol[Nr, Ntheta, Nz];

  // External connections at nozzle locations
  Modelica.Fluid.Interfaces.FluidPort_a inlet;
  Modelica.Fluid.Interfaces.FluidPort_b outlet;

equation
  for i in 1:Nr, j in 1:Ntheta, k in 1:Nz loop
    // Mass, energy, momentum conservation
    // Inter-cell coupling via array indexing, not connectors
  end for;
end Vessel;
```

External piping connects at nozzle locations via FluidPort connectors. Vessel translates between 3D internal mesh and 1D external connections at specific cells.

## Solver Integration

- Sparse matrix (Nr·Nθ·Nz) × (Nr·Nθ·Nz), ≤7 non-zeros/row
- 2500 cells: sparse direct (SuperLU) or iterative (BiCGSTAB+ILU), a few ms
- Nozzle BCs come from connected pipe junctions at partitioning interface

## Fallback if Extraction Fails (Test 5)

Vessel internals move to C++ via `external "C"`. Modelica exposes only external interface (nozzle FluidPorts, kinetics coupling arrays). This is how every other system code handles the vessel. Rest of plant (1D piping, BOP, controls) stays in Modelica regardless.
