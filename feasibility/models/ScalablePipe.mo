// ScalablePipe.mo — Feasibility Test 4: equation count scaling
//
// A parameterised 1D compressible pipe with N cells.
// Each cell has two ODE states (pressure, temperature) and one algebraic (flow).
// Equation count scales linearly with N.
//
// Used by test4_scale.py with N = 1, 5, 10, 20 to measure:
//   - Equation count vs N
//   - Extraction time (instantiate, flat, backEnd) vs N
//   - BLT block count vs N (should scale linearly)
//
// The model is self-contained (no external library).
// Temperature is tracked to produce a non-trivial 2-state-per-cell system.

model ScalablePipe
  "N-cell 1D compressible pipe: 2 ODEs + 1 algebraic per cell"

  parameter Integer N    = 5       "Number of cells";
  parameter Real    R    = 1e4     "Cell friction resistance [Pa/(kg/s)]";
  parameter Real    C    = 1e-9    "Cell compressibility [kg/Pa]";
  parameter Real    rho  = 720.0   "Water density [kg/m3]";
  parameter Real    Cp   = 5000.0  "Specific heat [J/(kg·K)]";
  parameter Real    V    = 0.01    "Cell volume [m3]";
  parameter Real    p_in  = 15.5e6 "Inlet pressure [Pa]";
  parameter Real    p_out = 15.4e6 "Outlet pressure [Pa]";
  parameter Real    T_in  = 563.0  "Inlet temperature [K]";

  // Cell-centre states
  Real p[N](each start = 15.5e6, each fixed = true)  "Cell pressure [Pa]";
  Real T[N](each start = 563.0,  each fixed = true)  "Cell temperature [K]";

  // Cell-face flows (N+1 faces: 0..N)
  //   mdot[1]   = inlet flow (from p_in boundary into cell 1)
  //   mdot[i+1] = flow from cell i into cell i+1   (i = 1..N-1)
  //   mdot[N+1] = outlet flow (from cell N into p_out boundary)
  Real mdot[N+1]  "Face mass flow rates [kg/s]";

equation
  // ---- Inlet boundary face (Darcy from p_in to cell 1) ----
  p_in - p[1] = R * mdot[1];

  // ---- Interior faces ----
  for i in 1:N-1 loop
    p[i] - p[i+1] = R * mdot[i+1];
  end for;

  // ---- Outlet boundary face ----
  p[N] - p_out = R * mdot[N+1];

  // ---- Cell conservation equations ----
  for i in 1:N loop
    // Mass (pressure ODE via compressibility)
    C * der(p[i]) = mdot[i] - mdot[i+1];

    // Energy (temperature ODE via first-order upwind advection)
    rho * V * Cp * der(T[i]) =
      mdot[i]   * Cp * (T_in - T[i])    // inflow enthalpy relative to cell
    - mdot[i+1] * Cp * T[i];            // outflow enthalpy (simplified)
  end for;

end ScalablePipe;
