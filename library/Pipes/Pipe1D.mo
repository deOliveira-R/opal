within library.Pipes;
model Pipe1D
  "1D pipe with inertial momentum, staggered mesh, two-phase capable"

  // ═══════════════════════════════════════════════════════════════════
  // Parameters
  // ═══════════════════════════════════════════════════════════════════
  parameter Integer N = 5 "Number of cells";
  parameter Real L = 1.0 "Pipe length [m]";
  parameter Real D = 0.1 "Inner diameter [m]";
  parameter Real f_D = 0.02 "Darcy friction factor [-]";

  // Derived geometry
  parameter Real dx = L / N "Cell length [m]";
  parameter Real A_flow = Modelica.Constants.pi / 4 * D^2 "Flow area [m^2]";
  parameter Real D_h = D "Hydraulic diameter [m]";
  parameter Real V_cell = dx * A_flow "Cell volume [m^3]";

  // Initial conditions
  parameter Real p_init = 10e6 "Initial pressure [Pa]";
  parameter Real h_init = 800e3 "Initial enthalpy [J/kg]";

  // Wall heat (per cell)
  parameter Real q_wall[N] = zeros(N) "Wall heat source per cell [W]";

  // ═══════════════════════════════════════════════════════════════════
  // Connectors
  // ═══════════════════════════════════════════════════════════════════
  library.Connectors.FluidPort port_a "Inlet (face 0)";
  library.Connectors.FluidPort port_b "Outlet (face N)";

  // ═══════════════════════════════════════════════════════════════════
  // State variables — staggered mesh
  //   Cell centres: p[1..N], h[1..N]
  //   Cell faces:   mdot[1..N+1]  (mdot[1]=inlet face, mdot[N+1]=outlet face)
  // ═══════════════════════════════════════════════════════════════════
  Real p[N](each start = p_init, each fixed = true) "Cell pressure [Pa]";
  Real h[N](each start = h_init, each fixed = true) "Cell enthalpy [J/kg]";
  Real mdot[N + 1](each start = 0, each fixed = true) "Face mass flow [kg/s]";

  // ═══════════════════════════════════════════════════════════════════
  // Property evaluation (per cell) — using SimpleFluid for now
  // All pure Modelica, no external C, extraction-transparent.
  // ═══════════════════════════════════════════════════════════════════
  Real rho[N] "Cell density [kg/m^3]";
  Real drho_dp[N] "drho/dp at constant h [kg/(m^3*Pa)]";
  Real drho_dh[N] "drho/dh at constant p [kg/(m^3*J/kg)]";
  Real T_cell[N] "Cell temperature [K] (diagnostic)";

  // Face densities (arithmetic average of adjacent cells)
  Real rho_face[N + 1] "Face density for momentum equation [kg/m^3]";

  // ═══════════════════════════════════════════════════════════════════
  // Donor-cell face enthalpies
  // ═══════════════════════════════════════════════════════════════════
  Real h_face[N + 1] "Face enthalpy for energy advection [J/kg]";

equation
  // ─────────────────────────────────────────────────────────────────
  // Connector-to-internal coupling
  //
  // port_a.p and port_b.p are NOT set here — they are determined by
  // the connected boundary components (PressureSource sets p, ClosedEnd
  // leaves p free). The momentum equations use port_a.p and port_b.p
  // as boundary pressures.
  //
  // port_a.m_flow is positive INTO the component.
  // mdot[1] is positive left-to-right (into cell 1).
  // mdot[N+1] is positive left-to-right (out of cell N).
  // port_b.m_flow is positive INTO the component (negative for outflow).
  // ─────────────────────────────────────────────────────────────────
  mdot[1] = port_a.m_flow;
  mdot[N + 1] = -port_b.m_flow;
  port_a.h_outflow = h[1];
  port_b.h_outflow = h[N];

  // ─────────────────────────────────────────────────────────────────
  // Properties (SimpleFluid — will be replaceable later)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho[i] = library.Media.SimpleFluid.rho_ph(p[i], h[i]);
    drho_dp[i] = library.Media.SimpleFluid.drho_dp_h(p[i], h[i]);
    drho_dh[i] = library.Media.SimpleFluid.drho_dh_p(p[i], h[i]);
    T_cell[i] = library.Media.SimpleFluid.T_ph(p[i], h[i]);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Face densities
  // ─────────────────────────────────────────────────────────────────
  rho_face[1] = rho[1];  // inlet face: use first cell density
  for i in 2:N loop
    rho_face[i] = 0.5 * (rho[i - 1] + rho[i]);
  end for;
  rho_face[N + 1] = rho[N];  // outlet face: use last cell density

  // ─────────────────────────────────────────────────────────────────
  // Donor-cell face enthalpies
  // ─────────────────────────────────────────────────────────────────
  // Face 1 (inlet): if flow enters (mdot>0), use upstream (port_a) enthalpy
  h_face[1] = if mdot[1] >= 0 then inStream(port_a.h_outflow) else h[1];

  for i in 2:N loop
    h_face[i] = if mdot[i] >= 0 then h[i - 1] else h[i];
  end for;

  // Face N+1 (outlet): if flow leaves (mdot>0), use last cell enthalpy
  h_face[N + 1] = if mdot[N + 1] >= 0 then h[N] else inStream(port_b.h_outflow);

  // ─────────────────────────────────────────────────────────────────
  // MASS CONSERVATION (per cell)
  //   V * d(rho)/dt = mdot_in - mdot_out
  //   Using chain rule: drho/dt = drho_dp * dp/dt + drho_dh * dh/dt
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    V_cell * (drho_dp[i] * der(p[i]) + drho_dh[i] * der(h[i]))
      = mdot[i] - mdot[i + 1];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // MOMENTUM (per face) — INERTIAL, the key difference from ScalablePipe
  //
  //   (rho_face * dx / A) * d(mdot)/dt
  //     = A * (p_left - p_right)
  //     - f_D * dx / (2 * D_h) * |mdot| * mdot / (rho_face * A^2)
  //
  //   The LHS is inertia: (rho * L_cell * A) / A * dv/dt = rho * dx * dv/dt
  //   expressed in terms of mdot = rho * A * v.
  //
  //   Interior faces (i = 2..N): p_left = p[i-1], p_right = p[i]
  //   Boundary faces: p_left/p_right come from the connector pressures.
  // ─────────────────────────────────────────────────────────────────

  // Face 1 (inlet boundary)
  (rho_face[1] * dx / A_flow) * der(mdot[1])
    = A_flow * (port_a.p - p[1])
    - f_D * dx / (2 * D_h) * abs(mdot[1]) * mdot[1] / (rho_face[1] * A_flow^2);

  // Interior faces
  for i in 2:N loop
    (rho_face[i] * dx / A_flow) * der(mdot[i])
      = A_flow * (p[i - 1] - p[i])
      - f_D * dx / (2 * D_h) * abs(mdot[i]) * mdot[i] / (rho_face[i] * A_flow^2);
  end for;

  // Face N+1 (outlet boundary)
  (rho_face[N + 1] * dx / A_flow) * der(mdot[N + 1])
    = A_flow * (p[N] - port_b.p)
    - f_D * dx / (2 * D_h) * abs(mdot[N + 1]) * mdot[N + 1] / (rho_face[N + 1] * A_flow^2);

  // ─────────────────────────────────────────────────────────────────
  // ENERGY (per cell) — enthalpy form with pressure work
  //
  //   rho * V * dh/dt = mdot_in*(h_in - h) - mdot_out*(h_out - h)
  //                     + V * dp/dt + q_wall
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho[i] * V_cell * der(h[i])
      = mdot[i] * (h_face[i] - h[i])
      - mdot[i + 1] * (h_face[i + 1] - h[i])
      + V_cell * der(p[i])
      + q_wall[i];
  end for;

  annotation(Documentation(info="<html>
<p><b>1D pipe with inertial momentum equation.</b></p>
<p>Staggered mesh: pressure and enthalpy at cell centres, mass flow at faces.
Three conservation equations per cell:</p>
<ul>
<li>Mass: linearised EOS with drho/dp and drho/dh</li>
<li>Momentum: inertial (d(mdot)/dt) with Darcy friction — enables acoustic
wave propagation and water hammer</li>
<li>Energy: enthalpy form with donor-cell advection and pressure work</li>
</ul>
<p>Uses SimpleFluid properties. For IAPWS-IF97, change the property calls
(replaceable media package planned for Phase 3).</p>
</html>"));
end Pipe1D;
