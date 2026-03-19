within library.Pipes;
model Pipe1D
  "1D pipe with inertial momentum, staggered mesh, replaceable medium"

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
  // Replaceable medium — swap SimpleFluid ↔ Water at system level
  // ═══════════════════════════════════════════════════════════════════
  replaceable package Medium = library.Media.SimpleFluid
    constrainedby library.Media.PartialMedium
    annotation(choicesAllMatching=true);

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
  // Property evaluation (per cell) — via replaceable Medium
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
  // ─────────────────────────────────────────────────────────────────
  mdot[1] = port_a.m_flow;
  mdot[N + 1] = -port_b.m_flow;
  port_a.h_outflow = h[1];
  port_b.h_outflow = h[N];

  // ─────────────────────────────────────────────────────────────────
  // Properties (via replaceable Medium)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho[i] = Medium.rho_ph(p[i], h[i]);
    drho_dp[i] = Medium.drho_dp_h(p[i], h[i]);
    drho_dh[i] = Medium.drho_dh_p(p[i], h[i]);
    T_cell[i] = Medium.T_ph(p[i], h[i]);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Face densities
  // ─────────────────────────────────────────────────────────────────
  rho_face[1] = rho[1];
  for i in 2:N loop
    rho_face[i] = 0.5 * (rho[i - 1] + rho[i]);
  end for;
  rho_face[N + 1] = rho[N];

  // ─────────────────────────────────────────────────────────────────
  // Donor-cell face enthalpies
  // ─────────────────────────────────────────────────────────────────
  h_face[1] = if mdot[1] >= 0 then inStream(port_a.h_outflow) else h[1];

  for i in 2:N loop
    h_face[i] = if mdot[i] >= 0 then h[i - 1] else h[i];
  end for;

  h_face[N + 1] = if mdot[N + 1] >= 0 then h[N] else inStream(port_b.h_outflow);

  // ─────────────────────────────────────────────────────────────────
  // MASS CONSERVATION (per cell)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    V_cell * (drho_dp[i] * der(p[i]) + drho_dh[i] * der(h[i]))
      = mdot[i] - mdot[i + 1];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // MOMENTUM (per face) — inertial with Darcy friction
  // ─────────────────────────────────────────────────────────────────
  (rho_face[1] * dx / A_flow) * der(mdot[1])
    = A_flow * (port_a.p - p[1])
    - f_D * dx / (2 * D_h) * abs(mdot[1]) * mdot[1] / (rho_face[1] * A_flow^2);

  for i in 2:N loop
    (rho_face[i] * dx / A_flow) * der(mdot[i])
      = A_flow * (p[i - 1] - p[i])
      - f_D * dx / (2 * D_h) * abs(mdot[i]) * mdot[i] / (rho_face[i] * A_flow^2);
  end for;

  (rho_face[N + 1] * dx / A_flow) * der(mdot[N + 1])
    = A_flow * (p[N] - port_b.p)
    - f_D * dx / (2 * D_h) * abs(mdot[N + 1]) * mdot[N + 1] / (rho_face[N + 1] * A_flow^2);

  // ─────────────────────────────────────────────────────────────────
  // ENERGY (per cell) — enthalpy form with pressure work
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho[i] * V_cell * der(h[i])
      = mdot[i] * (h_face[i] - h[i])
      - mdot[i + 1] * (h_face[i + 1] - h[i])
      + V_cell * der(p[i])
      + q_wall[i];
  end for;

  annotation(Documentation(info="<html>
<p><b>1D pipe with inertial momentum equation and replaceable medium.</b></p>
<p>Staggered mesh: pressure and enthalpy at cell centres, mass flow at faces.</p>
<p>Swap medium: <code>Pipe1D pipe(redeclare package Medium = library.Media.Water)</code></p>
<p>This is the HEM (Homogeneous Equilibrium Model) variant. For multi-equation
models (drift-flux, two-fluid), see Pipe1D_DriftFlux and Pipe1D_TwoFluid.</p>
</html>"));
end Pipe1D;
