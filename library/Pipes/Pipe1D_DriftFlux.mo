within library.Pipes;
model Pipe1D_DriftFlux
  "1D pipe with 5-equation drift-flux model — two-phase with phasic energy"

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
  parameter Real h_l_init = 800e3 "Initial liquid enthalpy [J/kg]";
  parameter Real h_v_init = 2800e3 "Initial vapour enthalpy [J/kg]";
  parameter Real alpha_init = 1e-6 "Initial void fraction [-]";

  // Wall heat (per cell)
  parameter Real q_wall[N] = zeros(N) "Wall heat source per cell [W]";

  // Closure parameters
  parameter Real H_i = 1e5 "Interfacial heat transfer coefficient [W/(m^3*K)]";
  parameter Real C_0 = 1.13 "Drift-flux distribution parameter [-]";
  parameter Real alpha_nucleation = 1e-3 "Nucleation onset void fraction [-]";

  // ═══════════════════════════════════════════════════════════════════
  // Replaceable medium
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
  // State variables — staggered mesh, 5 equations per cell
  //   Cell centres: p[1..N], alpha[1..N], h_l[1..N], h_v[1..N]
  //   Cell faces:   mdot[1..N+1]
  // ═══════════════════════════════════════════════════════════════════
  Real p[N](each start = p_init, each fixed = true) "Cell pressure [Pa]";
  Real alpha[N](each start = alpha_init, each fixed = true) "Void fraction [-]";
  Real h_l[N](each start = h_l_init, each fixed = true) "Liquid enthalpy [J/kg]";
  Real h_v[N](each start = h_v_init, each fixed = true) "Vapour enthalpy [J/kg]";
  Real mdot[N + 1](each start = 0, each fixed = true) "Face mass flow [kg/s]";

  // ═══════════════════════════════════════════════════════════════════
  // Phasic properties (per cell)
  // ═══════════════════════════════════════════════════════════════════
  Real rho_l[N] "Liquid density [kg/m^3]";
  Real rho_v[N] "Vapour density [kg/m^3]";
  Real rho_m[N] "Mixture density [kg/m^3]";
  Real T_l[N] "Liquid temperature [K]";
  Real T_sat_cell[N] "Saturation temperature [K]";
  Real h_sat_l[N] "Sat. liquid enthalpy [J/kg]";
  Real h_sat_v[N] "Sat. vapour enthalpy [J/kg]";
  Real drho_dp[N] "Mixture drho/dp at constant h [kg/(m^3*Pa)]";
  Real drho_dh[N] "Mixture drho/dh at constant p [kg/(m^3*J/kg)]";

  // Face densities
  Real rho_face[N + 1] "Face density [kg/m^3]";

  // ═══════════════════════════════════════════════════════════════════
  // Interfacial closures (per cell)
  // ═══════════════════════════════════════════════════════════════════
  Real Gamma[N] "Interfacial mass transfer [kg/(m^3*s)], >0 = evaporation";
  Real q_i_l[N] "Interfacial heat to liquid [W/m^3]";
  Real q_i_v[N] "Interfacial heat to vapour [W/m^3]";
  Real a_i[N] "Interfacial area concentration [-]";
  Real alpha_eff[N] "Effective void fraction (with nucleation) [-]";

  // Drift-flux (per cell)
  Real V_gj[N] "Drift velocity [m/s]";

  // ═══════════════════════════════════════════════════════════════════
  // Donor-cell face enthalpies (mixture, for connector coupling)
  // ═══════════════════════════════════════════════════════════════════
  Real h_mix[N] "Mixture enthalpy [J/kg]";

equation
  // ─────────────────────────────────────────────────────────────────
  // Connector coupling
  // ─────────────────────────────────────────────────────────────────
  mdot[1] = port_a.m_flow;
  mdot[N + 1] = -port_b.m_flow;
  for i in 1:N loop
    h_mix[i] = (1 - alpha[i]) * h_l[i] + alpha[i] * h_v[i];
  end for;
  port_a.h_outflow = h_mix[1];
  port_b.h_outflow = h_mix[N];

  // ─────────────────────────────────────────────────────────────────
  // Phasic property evaluation (via replaceable Medium)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho_l[i] = Medium.rho_ph(p[i], h_l[i]);
    rho_v[i] = Medium.rho_ph(p[i], h_v[i]);
    rho_m[i] = (1 - alpha[i]) * rho_l[i] + alpha[i] * rho_v[i];
    T_l[i] = Medium.T_ph(p[i], h_l[i]);
    T_sat_cell[i] = Medium.T_sat(p[i]);
    h_sat_l[i] = Medium.h_f(p[i]);
    h_sat_v[i] = Medium.h_g(p[i]);

    // Mixture derivatives for pressure linearisation
    drho_dp[i] = Medium.drho_dp_h(p[i], h_mix[i]);
    drho_dh[i] = Medium.drho_dh_p(p[i], h_mix[i]);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Face densities (arithmetic average)
  // ─────────────────────────────────────────────────────────────────
  rho_face[1] = rho_m[1];
  for i in 2:N loop
    rho_face[i] = 0.5 * (rho_m[i - 1] + rho_m[i]);
  end for;
  rho_face[N + 1] = rho_m[N];

  // ─────────────────────────────────────────────────────────────────
  // Interfacial closures
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    // Nucleation onset: when T_l > T_sat, enforce minimum void
    alpha_eff[i] = if T_l[i] > T_sat_cell[i] and alpha[i] < alpha_nucleation
                   then alpha_nucleation else alpha[i];

    // Interfacial area: max(4*alpha*(1-alpha), alpha)
    a_i[i] = max(4 * alpha_eff[i] * (1 - alpha_eff[i]), alpha_eff[i]);

    // Interfacial heat transfer (linear relaxation)
    q_i_l[i] = H_i * a_i[i] * (T_sat_cell[i] - T_l[i]);

    // Mass transfer from interfacial heat
    Gamma[i] = -q_i_l[i] / max(h_sat_v[i] - h_sat_l[i], 1.0);

    // Vapour heat: interface energy balance
    q_i_v[i] = -Gamma[i] * (h_v[i] - h_l[i]) - q_i_l[i];

    // Zuber-Findlay drift velocity
    V_gj[i] = 1.41 * (Medium.rho_f(p[i]) * 9.81 *
               max(rho_l[i] - rho_v[i], 0.01) /
               max(rho_l[i]^2, 1.0))^0.25 *
               4 * alpha[i] * (1 - alpha[i]);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // MASS CONSERVATION (per cell) — pressure linearisation
  //   V * (drho_dp * der(p) + drho_dh * der(h_mix)) = mdot_in - mdot_out
  //   where h_mix = (1-α)*h_l + α*h_v
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    V_cell * (drho_dp[i] * der(p[i])
            + drho_dh[i] * ((1 - alpha[i]) * der(h_l[i])
                          + alpha[i] * der(h_v[i])))
      = mdot[i] - mdot[i + 1];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // MOMENTUM (per face) — inertial with Darcy friction
  //   Same as HEM: mixture momentum, single velocity field
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
  // VOID FRACTION (per cell) — vapour mass transport
  //   d(α*ρ_v)/dt = (mdot_v_in - mdot_v_out) / V + Γ
  //   Simplified: use mixture mdot with donor-cell void fraction
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho_v[i] * V_cell * der(alpha[i])
      = (if mdot[i] >= 0 then
           mdot[i] * (if i > 1 then alpha[i - 1] else alpha[i])
         else
           mdot[i] * alpha[i])
      - (if mdot[i + 1] >= 0 then
           mdot[i + 1] * alpha[i]
         else
           mdot[i + 1] * (if i < N then alpha[i + 1] else alpha[i]))
      + V_cell * Gamma[i];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // LIQUID ENERGY (per cell) — phasic enthalpy with interfacial HT
  //   (1-α)*ρ_l*V * dh_l/dt = advection + (1-α)*V*dp/dt + q_wall_l
  //                          + q_i_l - Γ*h_l*V
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    (1 - alpha[i]) * rho_l[i] * V_cell * der(h_l[i])
      = (if mdot[i] >= 0 then
           mdot[i] * (1 - (if i > 1 then alpha[i - 1] else alpha[i]))
         else
           mdot[i] * (1 - alpha[i]))
        * ((if mdot[i] >= 0 then (if i > 1 then h_l[i - 1] else h_l[i]) else h_l[i]) - h_l[i])
      - (if mdot[i + 1] >= 0 then
           mdot[i + 1] * (1 - alpha[i])
         else
           mdot[i + 1] * (1 - (if i < N then alpha[i + 1] else alpha[i])))
        * ((if mdot[i + 1] >= 0 then h_l[i] else (if i < N then h_l[i + 1] else h_l[i])) - h_l[i])
      + (1 - alpha[i]) * V_cell * der(p[i])
      + q_wall[i] * (1 - alpha[i])
      + q_i_l[i] * V_cell
      - Gamma[i] * h_l[i] * V_cell;
  end for;

  // ─────────────────────────────────────────────────────────────────
  // VAPOUR ENERGY (per cell) — phasic enthalpy with interfacial HT
  //   α*ρ_v*V * dh_v/dt = advection + α*V*dp/dt + q_wall_v
  //                      + q_i_v + Γ*h_v*V
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    alpha[i] * rho_v[i] * V_cell * der(h_v[i])
      = (if mdot[i] >= 0 then
           mdot[i] * (if i > 1 then alpha[i - 1] else alpha[i])
         else
           mdot[i] * alpha[i])
        * ((if mdot[i] >= 0 then (if i > 1 then h_v[i - 1] else h_v[i]) else h_v[i]) - h_v[i])
      - (if mdot[i + 1] >= 0 then
           mdot[i + 1] * alpha[i]
         else
           mdot[i + 1] * (if i < N then alpha[i + 1] else alpha[i]))
        * ((if mdot[i + 1] >= 0 then h_v[i] else (if i < N then h_v[i + 1] else h_v[i])) - h_v[i])
      + alpha[i] * V_cell * der(p[i])
      + q_wall[i] * alpha[i]
      + q_i_v[i] * V_cell
      + Gamma[i] * h_v[i] * V_cell;
  end for;

  annotation(Documentation(info="<html>
<p><b>5-equation drift-flux model for two-phase pipe flow.</b></p>
<p>State variables per cell: pressure p, void fraction α, liquid enthalpy h_l,
vapour enthalpy h_v. Mass flow at faces (staggered mesh).</p>
<p>Equations:</p>
<ol>
<li>Mass conservation (mixture, pressure-linearised)</li>
<li>Momentum (inertial, mixture, Darcy friction)</li>
<li>Void fraction transport (vapour mass with interfacial transfer Γ)</li>
<li>Liquid energy (phasic, with interfacial HT and phase-change coupling)</li>
<li>Vapour energy (phasic, with interfacial HT and phase-change coupling)</li>
</ol>
<p>Closures: linear relaxation interfacial HT, Zuber-Findlay drift velocity.</p>
<p>Swap medium: <code>redeclare package Medium = library.Media.Water</code></p>
</html>"));
end Pipe1D_DriftFlux;
