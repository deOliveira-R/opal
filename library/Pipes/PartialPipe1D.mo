within library.Pipes;
partial model PartialPipe1D
  "Base class for 1D staggered-mesh pipe models — shared geometry, momentum, face densities"

  // ═══════════════════════════════════════════════════════════════════
  // Parameters — geometry
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

  // Gravity
  parameter Real g_axial = 0.0
    "Gravity projection on pipe axis [m/s^2]. Positive opposes positive flow (upward pipe).";

  // Initial conditions (pressure only — enthalpy ICs are model-specific)
  parameter Real p_init = 10e6 "Initial pressure [Pa]";

  // Wall heat (per cell)
  parameter Real q_wall[N] = zeros(N) "Wall heat source per cell [W]";

  // Generic source terms
  parameter Real S_mass[N] = zeros(N) "Mass source per cell [kg/s]";
  parameter Real S_momentum[N + 1] = zeros(N + 1) "Momentum source per face [N]";

  // Critical flow at outlet
  parameter Boolean use_critical_flow = false "Enable critical flow limiter at outlet";
  parameter Real critical_flow_model = 1
    "Critical flow model: 1 = Ransom-Trapp, 2 = Henry-Fauske";
  parameter Real C_d = 1.0 "Break discharge coefficient [-] (default, overridable via C_d_eff)";
  parameter Real x_trans = 0.10 "Ransom-Trapp: quality transition for blend [-]";
  parameter Real x_ne = 0.05 "Henry-Fauske: non-equilibrium transition quality [-]";
  parameter Real N_param = 0.0
    "Henry-Fauske: non-equilibrium parameter (0=frozen/sharp orifice, 1=full HF) [-]";
  parameter Real c_floor = 10.0 "Minimum sound speed for critical flow [m/s] (numerical floor only)";
  parameter Real use_acoustic_cf_limit = 0
    "1=enable acoustic choking limit on critical flow (requires c_ph in Medium)";

  // Time-varying discharge coefficient (for break opening ramp).
  // Must be set at system level: pipe.C_d_eff = C_d (constant) or
  // pipe.C_d_eff = ramp.C_d (time-varying from RampedBreak).
  Real C_d_eff "Effective discharge coefficient (time-varying) [-]";

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
  // Shared state variables — staggered mesh
  //   Cell centres: p[1..N]   (enthalpy states are model-specific)
  //   Cell faces:   mdot[1..N+1]
  // ═══════════════════════════════════════════════════════════════════
  Real p[N](each start = p_init, each fixed = true) "Cell pressure [Pa]";
  Real mdot[N + 1](each start = 0, each fixed = true) "Face mass flow [kg/s]";

  // ═══════════════════════════════════════════════════════════════════
  // Abstract variables — concrete models provide equations
  // ═══════════════════════════════════════════════════════════════════

  // Cell density (HEM: rho[i], DriftFlux: rho_m[i])
  Real rho_cell[N] "Cell density for face averaging [kg/m^3]";

  // Pressure-linearisation derivatives (equations in concrete)
  Real drho_dp[N] "drho/dp at constant h [kg/(m^3*Pa)]";
  Real drho_dh[N] "drho/dh at constant p [kg/(m^3*J/kg)]";

  // Two-phase friction multiplier (HEM: 1.0, DriftFlux: Martinelli-Nelson)
  Real Phi2[N + 1] "Two-phase friction multiplier [-]";

  // Outlet-cell references for critical flow (equations in concrete)
  Real h_mix_outlet "Mixture enthalpy at outlet cell [J/kg]";
  Real rho_outlet "Density at outlet cell [kg/m^3]";

  // ═══════════════════════════════════════════════════════════════════
  // Computed in base — face densities and critical flow
  // ═══════════════════════════════════════════════════════════════════
  Real rho_face[N + 1] "Face density for momentum equation [kg/m^3]";
  Real mdot_crit "Critical mass flow rate at outlet [kg/s]";

equation
  // ─────────────────────────────────────────────────────────────────
  // Connector mass flow coupling
  //   (enthalpy outflow is model-specific — concrete models set h_outflow)
  // ─────────────────────────────────────────────────────────────────
  mdot[1] = port_a.m_flow;
  mdot[N + 1] = -port_b.m_flow;

  // ─────────────────────────────────────────────────────────────────
  // Face densities (arithmetic average, from abstract rho_cell)
  // ─────────────────────────────────────────────────────────────────
  rho_face[1] = rho_cell[1];
  for i in 2:N loop
    rho_face[i] = 0.5 * (rho_cell[i - 1] + rho_cell[i]);
  end for;
  rho_face[N + 1] = rho_cell[N];

  // ─────────────────────────────────────────────────────────────────
  // Critical flow at outlet (selectable model)
  // ─────────────────────────────────────────────────────────────────
  mdot_crit = if use_critical_flow then
    (if critical_flow_model == 2 then
      library.Numerics.CriticalFlow.henry_fauske(
        p[N], h_mix_outlet, rho_outlet, drho_dp[N],
        Medium.h_f(p[N]), Medium.h_g(p[N]), Medium.rho_f(p[N]), Medium.rho_g(p[N]),
        Medium.rho_f(max(port_b.p, (2.0 / 3.0) * p[N])),
        Medium.rho_g(max(port_b.p, (2.0 / 3.0) * p[N])),
        port_b.p, A_flow, C_d_eff, x_ne, N_param, c_floor,
        if use_acoustic_cf_limit == 1 then Medium.c_ph(p[N], h_mix_outlet) else 0.0)
    else
      library.Numerics.CriticalFlow.ransom_trapp(
        p[N], h_mix_outlet, rho_outlet, drho_dp[N],
        Medium.h_f(p[N]), Medium.h_g(p[N]), Medium.rho_f(p[N]),
        port_b.p, A_flow, C_d_eff, x_trans, c_floor))
    else 1e10;

  // ─────────────────────────────────────────────────────────────────
  // MOMENTUM (per face) — inertial + Darcy friction*Phi2 + gravity
  // ─────────────────────────────────────────────────────────────────

  // Inlet face
  (rho_face[1] * dx / A_flow) * der(mdot[1])
    = A_flow * (port_a.p - p[1])
    - Phi2[1] * f_D * dx / (2 * D_h) * abs(mdot[1]) * mdot[1] / (rho_face[1] * A_flow^2)
    - rho_face[1] * g_axial * A_flow * dx
    + S_momentum[1];

  // Interior faces
  for i in 2:N loop
    (rho_face[i] * dx / A_flow) * der(mdot[i])
      = A_flow * (p[i - 1] - p[i])
      - Phi2[i] * f_D * dx / (2 * D_h) * abs(mdot[i]) * mdot[i] / (rho_face[i] * A_flow^2)
      - rho_face[i] * g_axial * A_flow * dx
      + S_momentum[i];
  end for;

  // Outlet face — with optional critical flow limiter
  if use_critical_flow then
    (rho_face[N + 1] * dx / A_flow) * der(mdot[N + 1])
      = A_flow * (p[N] - port_b.p)
      - Phi2[N + 1] * f_D * dx / (2 * D_h) * abs(mdot[N + 1]) * mdot[N + 1] / (rho_face[N + 1] * A_flow^2)
      - rho_face[N + 1] * g_axial * A_flow * dx
      + S_momentum[N + 1]
      - (if mdot[N + 1] > mdot_crit then
           (mdot[N + 1] - mdot_crit) / (dx / A_flow)
         else 0.0);
  else
    (rho_face[N + 1] * dx / A_flow) * der(mdot[N + 1])
      = A_flow * (p[N] - port_b.p)
      - Phi2[N + 1] * f_D * dx / (2 * D_h) * abs(mdot[N + 1]) * mdot[N + 1] / (rho_face[N + 1] * A_flow^2)
      - rho_face[N + 1] * g_axial * A_flow * dx
      + S_momentum[N + 1];
  end if;

  annotation(Documentation(info="<html>
<p><b>Base class for 1D staggered-mesh pipe models.</b></p>
<p>Provides shared infrastructure: geometry parameters, connectors, face density
averaging, momentum equations (with pluggable friction multiplier Phi2), and
critical flow computation.</p>
<p>Concrete models (Pipe1D, Pipe1D_DriftFlux) extend this and provide:</p>
<ul>
<li>Enthalpy state variables and energy equations</li>
<li>Property evaluation and <code>rho_cell[i]</code> alias</li>
<li>Friction multiplier <code>Phi2[i]</code> (1.0 for HEM, Martinelli-Nelson for drift-flux)</li>
<li>Outlet references <code>h_mix_outlet</code>, <code>rho_outlet</code> for critical flow</li>
<li>Mass conservation equation</li>
</ul>
</html>"));
end PartialPipe1D;
