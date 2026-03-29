within library.Pipes;
model Pipe1D_TwoFluid
  "1D pipe with 6-equation two-fluid model — separate phasic momentum equations.
   Derived in: derivations/two_fluid_momentum.py, derivations/interfacial_drag.py"

  // ═══════════════════════════════════════════════════════════════════
  // NOTE: Does NOT extend PartialPipe1D. That base class defines
  // mixture momentum (single mdot[N+1]). The two-fluid model replaces
  // that with two phasic momentum equations (mdot_l[N+1], mdot_v[N+1]).
  // Geometry, connectors, property evaluation, and critical flow are
  // copied from PartialPipe1D and Pipe1D_DriftFlux.
  // ═══════════════════════════════════════════════════════════════════

  // ═══════════════════════════════════════════════════════════════════
  // Parameters — geometry (same as PartialPipe1D)
  // ═══════════════════════════════════════════════════════════════════
  parameter Integer N = 5 "Number of cells";
  parameter Real L = 1.0 "Pipe length [m]";
  parameter Real D = 0.1 "Inner diameter [m]";
  parameter Real f_D = 0.02 "Darcy friction factor [-]";
  parameter Real dx = L / N "Cell length [m]";
  parameter Real A_flow = Modelica.Constants.pi / 4 * D^2 "Flow area [m^2]";
  parameter Real D_h = D "Hydraulic diameter [m]";
  parameter Real V_cell = dx * A_flow "Cell volume [m^3]";
  parameter Real g_axial = 0.0
    "Gravity projection on pipe axis [m/s^2]. Positive opposes positive flow.";

  // Initial conditions
  parameter Real p_init = 10e6 "Initial pressure [Pa]";
  parameter Real h_l_init = 800e3 "Initial liquid enthalpy [J/kg]";
  parameter Real h_v_init = 2800e3 "Initial vapour enthalpy [J/kg]";
  parameter Real alpha_init = 1e-6 "Initial void fraction [-]";

  // Wall heat and source terms
  parameter Real q_wall[N] = zeros(N) "Wall heat source per cell [W]";
  parameter Real S_mass[N] = zeros(N) "Mass source per cell [kg/s]";
  parameter Real S_momentum_l[N + 1] = zeros(N + 1) "Liquid momentum source per face [N]";
  parameter Real S_momentum_v[N + 1] = zeros(N + 1) "Vapour momentum source per face [N]";
  parameter Real S_energy_l[N] = zeros(N) "Liquid energy source per cell [W]";
  parameter Real S_energy_v[N] = zeros(N) "Vapour energy source per cell [W]";
  parameter Real S_void[N] = zeros(N) "Vapour mass source per cell [kg/s]";

  // Critical flow at outlet (same as PartialPipe1D)
  parameter Boolean use_critical_flow = false "Enable critical flow limiter at outlet";
  parameter Real critical_flow_model = 1
    "Critical flow model: 1 = Ransom-Trapp, 2 = Henry-Fauske";
  parameter Real C_d = 1.0 "Break discharge coefficient [-]";
  parameter Real x_trans = 0.10 "Ransom-Trapp: quality transition for blend [-]";
  parameter Real x_ne = 0.05 "Henry-Fauske: non-equilibrium transition quality [-]";
  parameter Real N_param = 0.0 "Henry-Fauske: non-equilibrium parameter [-]";
  parameter Real c_floor = 10.0 "Minimum sound speed for critical flow [m/s]";
  Real C_d_eff "Effective discharge coefficient (time-varying) [-]";

  // ═══════════════════════════════════════════════════════════════════
  // Two-fluid specific parameters
  // ═══════════════════════════════════════════════════════════════════
  parameter Real d_b = 1e-3 "Reference bubble diameter [m]";
  parameter Real Nu_i = 2.0 "Interfacial Nusselt number [-]";
  parameter Real alpha_nucleation = 1e-3 "Nucleation onset void fraction [-]";
  parameter Real mu_l_const = 2.8e-4
    "Liquid dynamic viscosity [Pa.s] (constant; future: from Medium.mu_f)";
  parameter Real drag_model = 1
    "Interfacial drag model: 1 = Ishii bubbly (Schiller-Naumann), 2 = regime map (bubbly+slug+annular)";

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
  // 6-equation state variables — staggered mesh
  //   Cell centres: p[1..N], alpha[1..N], h_l[1..N], h_v[1..N]
  //   Cell faces:   mdot_l[1..N+1], mdot_v[1..N+1]
  // ═══════════════════════════════════════════════════════════════════
  Real p[N](each start = p_init, each fixed = true) "Cell pressure [Pa]";
  Real alpha[N](each start = alpha_init, each fixed = true) "Void fraction [-]";
  Real h_l[N](each start = h_l_init, each fixed = true) "Liquid enthalpy [J/kg]";
  Real h_v[N](each start = h_v_init, each fixed = true) "Vapour enthalpy [J/kg]";
  Real mdot_l[N + 1](each start = 0, each fixed = true) "Liquid mass flow at face [kg/s]";
  Real mdot_v[N + 1](each start = 0, each fixed = true) "Vapour mass flow at face [kg/s]";

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
  Real drho_dp[N] "drho/dp at constant h [kg/(m^3*Pa)]";
  Real drho_dh[N] "drho/dh at constant p [kg/(m^3*J/kg)]";
  Real h_mix[N] "Mixture enthalpy [J/kg]";

  // ═══════════════════════════════════════════════════════════════════
  // Interfacial closures (per cell) — same as drift-flux model
  // ═══════════════════════════════════════════════════════════════════
  Real Gamma[N] "Interfacial mass transfer [kg/(m^3*s)], >0 = evaporation";
  Real q_i_l[N] "Interfacial heat to liquid [W/m^3]";
  Real q_i_v[N] "Interfacial heat to vapour [W/m^3]";
  Real a_i[N] "Interfacial area concentration [1/m]";
  Real h_i[N] "Interfacial film heat transfer coefficient [W/(m^2*K)]";
  Real alpha_eff[N] "Effective void fraction (with nucleation) [-]";

  // ═══════════════════════════════════════════════════════════════════
  // Face-averaged properties (for phasic momentum)
  // ═══════════════════════════════════════════════════════════════════
  Real alpha_face[N + 1] "Face-averaged void fraction [-]";
  Real rho_l_face[N + 1] "Face-averaged liquid density [kg/m^3]";
  Real rho_v_face[N + 1] "Face-averaged vapour density [kg/m^3]";
  Real rho_face[N + 1] "Face-averaged mixture density [kg/m^3]";

  // ═══════════════════════════════════════════════════════════════════
  // Phasic velocities and interfacial drag (per face)
  // ═══════════════════════════════════════════════════════════════════
  Real v_l[N + 1] "Liquid velocity at face [m/s]";
  Real v_v[N + 1] "Vapour velocity at face [m/s]";
  Real F_drag[N + 1] "Interfacial drag force per unit volume [N/m^3]";

  // Per-phase wall friction (per face)
  Real fric_l[N + 1] "Liquid wall friction force [Pa]";
  Real fric_v[N + 1] "Vapour wall friction force [Pa]";

  // Critical flow
  Real mdot_crit "Critical mass flow rate at outlet [kg/s]";

equation
  // ─────────────────────────────────────────────────────────────────
  // Connector mass flow coupling (total = liquid + vapour)
  // ─────────────────────────────────────────────────────────────────
  mdot_l[1] + mdot_v[1] = port_a.m_flow;
  mdot_l[N + 1] + mdot_v[N + 1] = -port_b.m_flow;

  // ─────────────────────────────────────────────────────────────────
  // Mixture enthalpy and connector outflow
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    h_mix[i] = (1 - alpha[i]) * h_l[i] + alpha[i] * h_v[i];
  end for;
  port_a.h_outflow = h_mix[1];
  port_b.h_outflow = h_mix[N];

  // ─────────────────────────────────────────────────────────────────
  // Phasic property evaluation (same as Pipe1D_DriftFlux)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho_l[i] = if h_l[i] <= Medium.h_f(p[i]) then
                 Medium.rho_ph(p[i], h_l[i])
               else
                 Medium.rho_f(p[i]);
    rho_v[i] = if h_v[i] >= Medium.h_g(p[i]) then
                 Medium.rho_ph(p[i], h_v[i])
               else
                 Medium.rho_g(p[i]);
    rho_m[i] = (1 - alpha[i]) * rho_l[i] + alpha[i] * rho_v[i];
    T_sat_cell[i] = Medium.T_sat(p[i]);
    h_sat_l[i] = Medium.h_f(p[i]);
    h_sat_v[i] = Medium.h_g(p[i]);

    // Metastable liquid temperature extension
    T_l[i] = if h_l[i] <= h_sat_l[i] then
               Medium.T_ph(p[i], h_l[i])
             else
               T_sat_cell[i] + (h_l[i] - h_sat_l[i]) / Medium.cp_f(p[i]);

    // Mixture derivatives for pressure linearisation
    drho_dp[i] = Medium.drho_dp_h(p[i], h_mix[i]);
    drho_dh[i] = Medium.drho_dh_p(p[i], h_mix[i]);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Interfacial closures (same as Pipe1D_DriftFlux)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    alpha_eff[i] = if T_l[i] > T_sat_cell[i] and alpha[i] < alpha_nucleation
                   then alpha_nucleation else alpha[i];
    a_i[i] = 6 * alpha_eff[i] * (1 - alpha_eff[i]) / d_b;
    h_i[i] = Nu_i * Medium.k_f(p[i]) / d_b;
    q_i_l[i] = h_i[i] * a_i[i] * (T_sat_cell[i] - T_l[i]);
    Gamma[i] = -q_i_l[i] / max(h_sat_v[i] - h_sat_l[i], 1.0);
    q_i_v[i] = -Gamma[i] * (h_v[i] - h_l[i]) - q_i_l[i];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Face-averaged properties (boundary: nearest cell; interior: arithmetic mean)
  // ─────────────────────────────────────────────────────────────────
  alpha_face[1] = alpha[1];
  rho_l_face[1] = rho_l[1];
  rho_v_face[1] = rho_v[1];
  for i in 2:N loop
    alpha_face[i] = 0.5 * (alpha[i - 1] + alpha[i]);
    rho_l_face[i] = 0.5 * (rho_l[i - 1] + rho_l[i]);
    rho_v_face[i] = 0.5 * (rho_v[i - 1] + rho_v[i]);
  end for;
  alpha_face[N + 1] = alpha[N];
  rho_l_face[N + 1] = rho_l[N];
  rho_v_face[N + 1] = rho_v[N];

  for i in 1:N + 1 loop
    rho_face[i] = (1 - alpha_face[i]) * rho_l_face[i] + alpha_face[i] * rho_v_face[i];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Phasic velocities at faces (from phasic mass flows)
  // Phase-absence guards: max(alpha, 1e-6), max(1-alpha, 1e-6)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N + 1 loop
    v_l[i] = mdot_l[i] / (max(1 - alpha_face[i], 1e-6) * rho_l_face[i] * A_flow);
    v_v[i] = mdot_v[i] / (max(alpha_face[i], 1e-6) * rho_v_face[i] * A_flow);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Interfacial drag at faces (selectable: 1=Ishii bubbly, 2=regime map)
  // Derived in: derivations/interfacial_drag.py, derivations/drag_regime_map.py
  // F_drag > 0 when v_v > v_l (pushes liquid +x, vapor -x)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N + 1 loop
    F_drag[i] = if drag_model == 2 then
      library.Numerics.InterfacialDrag.regime_map_drag(
        alpha_face[i], rho_l_face[i], rho_v_face[i],
        v_l[i], v_v[i], d_b, mu_l_const, D)
    else
      library.Numerics.InterfacialDrag.ishii_drag(
        alpha_face[i], rho_l_face[i], v_l[i], v_v[i], d_b, mu_l_const);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Per-phase wall friction at faces (Darcy, no Martinelli-Nelson)
  // Each phase has its own friction using phase-specific mass flow & density
  // fric_k = f_D*dx/(2*D_h) * |mdot_k| * mdot_k / (alpha_k * rho_k * A^2)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N + 1 loop
    fric_l[i] = f_D * dx / (2 * D_h)
                * abs(mdot_l[i]) * mdot_l[i]
                / (max(1 - alpha_face[i], 1e-6) * rho_l_face[i] * A_flow^2);
    fric_v[i] = f_D * dx / (2 * D_h)
                * abs(mdot_v[i]) * mdot_v[i]
                / (max(alpha_face[i], 1e-6) * rho_v_face[i] * A_flow^2);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Critical flow at outlet (same as PartialPipe1D)
  // Applied to TOTAL mass flow (mdot_l + mdot_v)
  // ─────────────────────────────────────────────────────────────────
  mdot_crit = if use_critical_flow then
    (if critical_flow_model == 2 then
      library.Numerics.CriticalFlow.henry_fauske(
        p[N], h_mix[N], rho_m[N], drho_dp[N],
        Medium.h_f(p[N]), Medium.h_g(p[N]), Medium.rho_f(p[N]), Medium.rho_g(p[N]),
        port_b.p, A_flow, C_d_eff, x_ne, N_param, c_floor)
    else
      library.Numerics.CriticalFlow.ransom_trapp(
        p[N], h_mix[N], rho_m[N], drho_dp[N],
        Medium.h_f(p[N]), Medium.h_g(p[N]), Medium.rho_f(p[N]),
        port_b.p, A_flow, C_d_eff, x_trans, c_floor))
    else 1e10;

  // ─────────────────────────────────────────────────────────────────
  // LIQUID MOMENTUM (per face)
  // Derived in: derivations/two_fluid_momentum.py
  //   (1-α_f)·ρ_l_f·(dx/A) · der(mdot_l) =
  //     (1-α_f)·A·dp - fric_l - (1-α_f)·ρ_l_f·g·A·dx + F_drag·V_face + S
  // Sign: +F_drag accelerates liquid toward vapor velocity
  // ─────────────────────────────────────────────────────────────────

  // Inlet face (connects to port_a)
  max(1 - alpha_face[1], 1e-6) * rho_l_face[1] * dx / A_flow * der(mdot_l[1])
    = max(1 - alpha_face[1], 1e-6) * A_flow * (port_a.p - p[1])
    - fric_l[1]
    - max(1 - alpha_face[1], 1e-6) * rho_l_face[1] * g_axial * A_flow * dx
    + F_drag[1] * V_cell
    + S_momentum_l[1];

  // Interior faces
  for i in 2:N loop
    max(1 - alpha_face[i], 1e-6) * rho_l_face[i] * dx / A_flow * der(mdot_l[i])
      = max(1 - alpha_face[i], 1e-6) * A_flow * (p[i - 1] - p[i])
      - fric_l[i]
      - max(1 - alpha_face[i], 1e-6) * rho_l_face[i] * g_axial * A_flow * dx
      + F_drag[i] * V_cell
      + S_momentum_l[i];
  end for;

  // Outlet face — with optional critical flow limiter on total flow
  if use_critical_flow then
    max(1 - alpha_face[N + 1], 1e-6) * rho_l_face[N + 1] * dx / A_flow * der(mdot_l[N + 1])
      = max(1 - alpha_face[N + 1], 1e-6) * A_flow * (p[N] - port_b.p)
      - fric_l[N + 1]
      - max(1 - alpha_face[N + 1], 1e-6) * rho_l_face[N + 1] * g_axial * A_flow * dx
      + F_drag[N + 1] * V_cell
      + S_momentum_l[N + 1]
      - (if (mdot_l[N + 1] + mdot_v[N + 1]) > mdot_crit then
           max(1 - alpha_face[N + 1], 1e-6)
           * ((mdot_l[N + 1] + mdot_v[N + 1]) - mdot_crit)
           / (dx / A_flow)
         else 0.0);
  else
    max(1 - alpha_face[N + 1], 1e-6) * rho_l_face[N + 1] * dx / A_flow * der(mdot_l[N + 1])
      = max(1 - alpha_face[N + 1], 1e-6) * A_flow * (p[N] - port_b.p)
      - fric_l[N + 1]
      - max(1 - alpha_face[N + 1], 1e-6) * rho_l_face[N + 1] * g_axial * A_flow * dx
      + F_drag[N + 1] * V_cell
      + S_momentum_l[N + 1];
  end if;

  // ─────────────────────────────────────────────────────────────────
  // VAPOUR MOMENTUM (per face)
  // Derived in: derivations/two_fluid_momentum.py
  //   α_f·ρ_v_f·(dx/A) · der(mdot_v) =
  //     α_f·A·dp - fric_v - α_f·ρ_v_f·g·A·dx - F_drag·V_face + S
  // Sign: -F_drag decelerates vapor toward liquid velocity (Newton's 3rd)
  // ─────────────────────────────────────────────────────────────────

  // Inlet face
  max(alpha_face[1], 1e-6) * rho_v_face[1] * dx / A_flow * der(mdot_v[1])
    = max(alpha_face[1], 1e-6) * A_flow * (port_a.p - p[1])
    - fric_v[1]
    - max(alpha_face[1], 1e-6) * rho_v_face[1] * g_axial * A_flow * dx
    - F_drag[1] * V_cell
    + S_momentum_v[1];

  // Interior faces
  for i in 2:N loop
    max(alpha_face[i], 1e-6) * rho_v_face[i] * dx / A_flow * der(mdot_v[i])
      = max(alpha_face[i], 1e-6) * A_flow * (p[i - 1] - p[i])
      - fric_v[i]
      - max(alpha_face[i], 1e-6) * rho_v_face[i] * g_axial * A_flow * dx
      - F_drag[i] * V_cell
      + S_momentum_v[i];
  end for;

  // Outlet face
  if use_critical_flow then
    max(alpha_face[N + 1], 1e-6) * rho_v_face[N + 1] * dx / A_flow * der(mdot_v[N + 1])
      = max(alpha_face[N + 1], 1e-6) * A_flow * (p[N] - port_b.p)
      - fric_v[N + 1]
      - max(alpha_face[N + 1], 1e-6) * rho_v_face[N + 1] * g_axial * A_flow * dx
      - F_drag[N + 1] * V_cell
      + S_momentum_v[N + 1]
      - (if (mdot_l[N + 1] + mdot_v[N + 1]) > mdot_crit then
           max(alpha_face[N + 1], 1e-6)
           * ((mdot_l[N + 1] + mdot_v[N + 1]) - mdot_crit)
           / (dx / A_flow)
         else 0.0);
  else
    max(alpha_face[N + 1], 1e-6) * rho_v_face[N + 1] * dx / A_flow * der(mdot_v[N + 1])
      = max(alpha_face[N + 1], 1e-6) * A_flow * (p[N] - port_b.p)
      - fric_v[N + 1]
      - max(alpha_face[N + 1], 1e-6) * rho_v_face[N + 1] * g_axial * A_flow * dx
      - F_drag[N + 1] * V_cell
      + S_momentum_v[N + 1];
  end if;

  // ─────────────────────────────────────────────────────────────────
  // MASS CONSERVATION (per cell) — pressure linearisation
  //   V * (drho_dp * der(p) + drho_dh * der(h_mix)) = mdot_in - mdot_out
  //   where total mass flow at face = mdot_l + mdot_v
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    V_cell * (drho_dp[i] * der(p[i])
            + drho_dh[i] * ((1 - alpha[i]) * der(h_l[i])
                          + alpha[i] * der(h_v[i])))
      = (mdot_l[i] + mdot_v[i]) - (mdot_l[i + 1] + mdot_v[i + 1]) + S_mass[i];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // VOID FRACTION (per cell) — vapour mass transport
  //   rho_v * V * der(alpha) = mdot_v_in - mdot_v_out + V*Gamma
  //   Conservative update handled by bridge solver.
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho_v[i] * V_cell * der(alpha[i])
      = mdot_v[i] - mdot_v[i + 1]
      + V_cell * Gamma[i]
      + S_void[i];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // LIQUID ENERGY (per cell) — phasic enthalpy
  //   Uses mdot_l directly (not from algebraic split).
  //   Same equation structure as Pipe1D_DriftFlux.
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    max(1 - alpha[i], 1e-6) * rho_l[i] * V_cell * der(h_l[i])
      = mdot_l[i]
        * ((if mdot_l[i] >= 0 then (if i > 1 then h_l[i - 1] else h_l[i]) else h_l[i]) - h_l[i])
      - mdot_l[i + 1]
        * ((if mdot_l[i + 1] >= 0 then h_l[i] else (if i < N then h_l[i + 1] else h_l[i])) - h_l[i])
      + (1 - alpha[i]) * V_cell * der(p[i])
      + q_wall[i] * (1 - alpha[i])
      + q_i_l[i] * V_cell
      - Gamma[i] * h_l[i] * V_cell
      + S_energy_l[i];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // VAPOUR ENERGY (per cell) — phasic enthalpy
  //   Uses mdot_v directly (not from algebraic split).
  //   Same equation structure as Pipe1D_DriftFlux.
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    max(alpha[i], 1e-6) * rho_v[i] * V_cell * der(h_v[i])
      = mdot_v[i]
        * ((if mdot_v[i] >= 0 then (if i > 1 then h_v[i - 1] else h_v[i]) else h_v[i]) - h_v[i])
      - mdot_v[i + 1]
        * ((if mdot_v[i + 1] >= 0 then h_v[i] else (if i < N then h_v[i + 1] else h_v[i])) - h_v[i])
      + alpha[i] * V_cell * der(p[i])
      + q_wall[i] * alpha[i]
      + q_i_v[i] * V_cell
      + Gamma[i] * h_v[i] * V_cell
      + S_energy_v[i];
  end for;

  annotation(Documentation(info="<html>
<p><b>6-equation two-fluid model for two-phase pipe flow.</b></p>
<p>Solves separate phasic momentum equations instead of the mixture momentum +
algebraic drift-flux slip used by Pipe1D_DriftFlux. This allows the inter-phase
velocity difference to evolve dynamically via interfacial drag coupling.</p>
<p><b>State variables per cell:</b> pressure p, void fraction α, liquid enthalpy h_l,
vapour enthalpy h_v. Per face: liquid mass flow mdot_l, vapour mass flow mdot_v.</p>
<p><b>Equations:</b></p>
<ol>
<li>Mass conservation (mixture, pressure-linearised)</li>
<li>Void fraction transport (vapour mass with interfacial transfer Γ)</li>
<li>Liquid momentum (inertial + pressure + Darcy friction + gravity + drag)</li>
<li>Vapour momentum (inertial + pressure + Darcy friction + gravity − drag)</li>
<li>Liquid energy (phasic, with interfacial HT and phase-change coupling)</li>
<li>Vapour energy (phasic, with interfacial HT and phase-change coupling)</li>
</ol>
<p><b>New closures vs drift-flux:</b> Ishii bubbly interfacial drag (Schiller-Naumann C_D),
per-phase Darcy wall friction (no Martinelli-Nelson Φ² needed).</p>
<p><b>Retained from drift-flux:</b> interfacial heat transfer (Nu=2), metastable T_l,
nucleation onset, Henry-Fauske/Ransom-Trapp critical flow, IAPWS-IF97 properties.</p>
<p>Derived in: derivations/two_fluid_momentum.py, derivations/interfacial_drag.py</p>
</html>"));
end Pipe1D_TwoFluid;
