// Vessel3D.mo — Feasibility Test 5: 3D array component extraction
//
// Approach B monolithic vessel: single Modelica model with 3D array equations.
// Mirrors the production vessel design from docs/vessel.md.
//
// Mesh: Nr × Ntheta × Nz cylindrical grid, staggered velocities.
// Physics: single-phase, incompressible with small compressibility (pressure ODE).
//          Axial flow only (radial/azimuthal suppressed for this test).
//
// Purpose (Test 5):
//   FM1: Are for-loops unrolled in instantiateModel output?
//   FM2: Does BLT scatter or preserve column structure?
//   FM4: How does extraction time scale with mesh size?
//   FM5: Are array indices preserved in variable names (p[i,j,k])?

model Vessel3D
  "Monolithic 3D cylindrical vessel — Approach B feasibility test"

  parameter Integer Nr     = 3   "Radial cells";
  parameter Integer Ntheta = 3   "Azimuthal cells";
  parameter Integer Nz     = 5   "Axial cells";

  // ---- Cell-centre states ----
  Real p[Nr, Ntheta, Nz](each start = 15.5e6, each fixed = true)
    "Cell pressure [Pa]";
  Real T[Nr, Ntheta, Nz](each start = 563.0, each fixed = true)
    "Cell temperature [K]";

  // ---- Face velocities (staggered mesh) ----
  Real u_z[Nr, Ntheta, Nz + 1]  "Axial face velocity [m/s]";
  Real u_r[Nr + 1, Ntheta, Nz]  "Radial face velocity [m/s]  (= 0 in this test)";
  Real u_t[Nr, Ntheta, Nz]      "Azimuthal velocity [m/s]    (= 0 in this test)";

  // ---- Parameters ----
  parameter Real dz    = 0.30   "Axial cell height [m]";
  parameter Real dr    = 0.10   "Radial cell width [m]";
  parameter Real rho0  = 720.0  "Reference density [kg/m3]";
  parameter Real C_comp = 1e-9  "Compressibility [1/Pa]";
  parameter Real mu    = 1e-4   "Dynamic viscosity [Pa·s]";
  parameter Real Cp    = 5000.0 "Specific heat [J/(kg·K)]";
  parameter Real K_fric = 12.0 * mu / (dr * dr)
    "Friction coefficient [Pa·s/m2] (Darcy-Brinkman approximation)";

  // ---- Boundary conditions ----
  parameter Real p_in  = 15.50e6  "Inlet pressure (bottom face) [Pa]";
  parameter Real p_out = 15.40e6  "Outlet pressure (top face) [Pa]";
  parameter Real T_in  = 563.0    "Inlet temperature [K]";

equation
  // -----------------------------------------------------------------------
  // Radial and azimuthal velocities: suppressed in this test
  // (produces many alias variables — tests alias elimination scale)
  for i in 1:Nr + 1 loop
    for j in 1:Ntheta loop
      for k in 1:Nz loop
        u_r[i, j, k] = 0.0;
      end for;
    end for;
  end for;

  for i in 1:Nr loop
    for j in 1:Ntheta loop
      for k in 1:Nz loop
        u_t[i, j, k] = 0.0;
      end for;
    end for;
  end for;

  // -----------------------------------------------------------------------
  // Axial momentum: Darcy pressure drop across each axial face
  for i in 1:Nr loop
    for j in 1:Ntheta loop
      // Bottom boundary face (k=1): inlet pressure → cell 1
      (p_in - p[i, j, 1]) / dz = K_fric * u_z[i, j, 1];
      // Interior faces (k = 2..Nz)
      for k in 2:Nz loop
        (p[i, j, k - 1] - p[i, j, k]) / dz = K_fric * u_z[i, j, k];
      end for;
      // Top boundary face (k = Nz+1): cell Nz → outlet pressure
      (p[i, j, Nz] - p_out) / dz = K_fric * u_z[i, j, Nz + 1];
    end for;
  end for;

  // -----------------------------------------------------------------------
  // Mass continuity (compressibility gives pressure ODE per cell)
  for i in 1:Nr loop
    for j in 1:Ntheta loop
      for k in 1:Nz loop
        rho0 * C_comp * der(p[i, j, k]) =
          (u_z[i, j, k] - u_z[i, j, k + 1]) / dz;
      end for;
    end for;
  end for;

  // -----------------------------------------------------------------------
  // Energy (temperature ODE: first-order upwind axial advection)
  for i in 1:Nr loop
    for j in 1:Ntheta loop
      for k in 1:Nz loop
        rho0 * Cp * der(T[i, j, k]) =
          rho0 * Cp * u_z[i, j, k]   * (T_in   - T[i, j, k]) / dz
        + rho0 * Cp * u_z[i, j, k+1] * (T[i, j, k] - T_in)   / dz;
      end for;
    end for;
  end for;

end Vessel3D;
