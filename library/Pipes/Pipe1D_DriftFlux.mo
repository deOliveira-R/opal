within library.Pipes;
model Pipe1D_DriftFlux
  "1D pipe with 5-equation drift-flux model — extends shared base class"
  extends PartialPipe1D;

  // ═══════════════════════════════════════════════════════════════════
  // Drift-flux specific parameters
  // ═══════════════════════════════════════════════════════════════════
  parameter Real h_l_init = 800e3 "Initial liquid enthalpy [J/kg]";
  parameter Real h_v_init = 2800e3 "Initial vapour enthalpy [J/kg]";
  parameter Real alpha_init = 1e-6 "Initial void fraction [-]";

  // Closure parameters
  parameter Real H_i = 1e5 "Interfacial heat transfer coefficient [W/(m^3*K)]";
  parameter Real C_0 = 1.13 "Drift-flux distribution parameter [-]";
  parameter Real alpha_nucleation = 1e-3 "Nucleation onset void fraction [-]";

  // Generic source terms (phasic)
  parameter Real S_energy_l[N] = zeros(N) "Liquid energy source per cell [W]";
  parameter Real S_energy_v[N] = zeros(N) "Vapour energy source per cell [W]";
  parameter Real S_void[N] = zeros(N) "Vapour mass source per cell [kg/s]";

  // Two-phase friction multiplier
  parameter Boolean use_two_phase_friction = false "Enable Martinelli-Nelson friction multiplier";
  parameter Real Phi2_max = 20.0 "Maximum two-phase friction multiplier [-]";

  // ═══════════════════════════════════════════════════════════════════
  // 5-equation state variables — staggered mesh
  //   Cell centres: p[1..N] (inherited), alpha[1..N], h_l[1..N], h_v[1..N]
  //   Cell faces:   mdot[1..N+1] (inherited)
  // ═══════════════════════════════════════════════════════════════════
  Real alpha[N](each start = alpha_init, each fixed = true) "Void fraction [-]";
  Real h_l[N](each start = h_l_init, each fixed = true) "Liquid enthalpy [J/kg]";
  Real h_v[N](each start = h_v_init, each fixed = true) "Vapour enthalpy [J/kg]";

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
  // Mixture enthalpy (for connector coupling and critical flow)
  // ═══════════════════════════════════════════════════════════════════
  Real h_mix[N] "Mixture enthalpy [J/kg]";

equation
  // ─────────────────────────────────────────────────────────────────
  // Abstract variable bindings for base class
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho_cell[i] = rho_m[i];
  end for;

  h_mix_outlet = h_mix[N];
  rho_outlet = rho_m[N];

  // ─────────────────────────────────────────────────────────────────
  // Two-phase friction multiplier (per face)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N + 1 loop
    if use_two_phase_friction then
      Phi2[i] = min(library.Numerics.TwoPhaseFriction.martinelli_nelson(
        if i <= N then alpha[i] else alpha[N],
        if i <= N then rho_l[i] else rho_l[N],
        if i <= N then rho_v[i] else rho_v[N]),
        Phi2_max);
    else
      Phi2[i] = 1.0;
    end if;
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Connector enthalpy outflow
  // ─────────────────────────────────────────────────────────────────
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
    // V_gj = 1.41 * [σ * g * Δρ / ρ_l²]^0.25 * 4α(1-α)
    // Ref: Ishii & Hibiki, "Thermo-Fluid Dynamics of Two-Phase Flow", Eq. 11.21
    V_gj[i] = 1.41 * (Medium.sigma(p[i]) * 9.81 *
               max(rho_l[i] - rho_v[i], 0.01) /
               max(rho_l[i]^2, 1.0))^0.25 *
               4 * alpha[i] * (1 - alpha[i]);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // MASS CONSERVATION (per cell) — pressure linearisation
  //   V * (drho_dp * der(p) + drho_dh * der(h_mix)) = mdot_in - mdot_out + S_mass
  //   where h_mix = (1-α)*h_l + α*h_v
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    V_cell * (drho_dp[i] * der(p[i])
            + drho_dh[i] * ((1 - alpha[i]) * der(h_l[i])
                          + alpha[i] * der(h_v[i])))
      = mdot[i] - mdot[i + 1] + S_mass[i];
  end for;

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
      + V_cell * Gamma[i]
      + S_void[i];
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
      - Gamma[i] * h_l[i] * V_cell
      + S_energy_l[i];
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
      + Gamma[i] * h_v[i] * V_cell
      + S_energy_v[i];
  end for;

  annotation(Documentation(info="<html>
<p><b>5-equation drift-flux model for two-phase pipe flow.</b></p>
<p>Extends PartialPipe1D for shared geometry, momentum, and face density.
Adds phasic state variables, interfacial closures, and two phasic energy equations.</p>
<p>State variables per cell: pressure p, void fraction α, liquid enthalpy h_l,
vapour enthalpy h_v. Mass flow at faces (staggered mesh).</p>
<p>Equations:</p>
<ol>
<li>Mass conservation (mixture, pressure-linearised)</li>
<li>Momentum (inertial, mixture, Darcy friction — inherited from base)</li>
<li>Void fraction transport (vapour mass with interfacial transfer Γ)</li>
<li>Liquid energy (phasic, with interfacial HT and phase-change coupling)</li>
<li>Vapour energy (phasic, with interfacial HT and phase-change coupling)</li>
</ol>
<p>Closures: linear relaxation interfacial HT, Zuber-Findlay drift velocity.</p>
<p>Swap medium: <code>redeclare package Medium = library.Media.Water</code></p>
</html>"));
end Pipe1D_DriftFlux;
