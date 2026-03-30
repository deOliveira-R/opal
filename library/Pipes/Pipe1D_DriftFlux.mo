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
  parameter Real d_b = 1e-3 "Reference bubble diameter for interfacial area [m]";
  parameter Real d_b_min = 1e-5 "Minimum bubble diameter (numerical floor) [m]";
  parameter Real Nu_i = 2.0
    "Interfacial Nusselt number [-]. 2.0 = conduction limit (no-slip model).
     Future: Nu = 2 + 0.6*Re^0.5*Pr^0.33 (Ranz-Marshall) when slip velocity added.";
  parameter Real C_0 = 1.13 "Drift-flux distribution parameter [-]";
  parameter Real alpha_nucleation = 1e-3 "Nucleation onset void fraction [-]";
  // Flashing model selection — direct weights avoid OM CSE bug where
  // parameter arithmetic (min/max/clamp) gets pre-evaluated to $cseN with
  // value=None in the XML, causing the bridge to set them to 0.
  // Set exactly ONE of these to 1.0 (or leave both at 0 for baseline).
  parameter Real use_inception = 0
    "1.0 to enable d_b_eff inception model. WARNING: causes runaway evaporation
     in rapid depressurization (53.4% vs 28.3% baseline on Edwards). The 100x H_i
     enhancement has no inertial rate limiter — a_i grows with alpha creating positive
     feedback. Parked for research. Ref: Shin & Jones (1993), Blinkov et al. (1993).";
  parameter Real use_relaxation = 0
    "1.0 to enable Jones/Lahey relaxation model. RECOMMENDED for rapid depressurization.
     H_eff = (1-alpha)*rho_l*cp_f/tau_flash. Self-limiting: H_eff decreases monotonically
     with alpha (less liquid = less flashing potential). Unlike geometric H_i which peaks
     at alpha=0.5. Ref: Jones (1982), Lahey & Moody (1993), RELAP5/MOD3 Vol I.";
  parameter Real d_b_flash = 3e-5
    "Nucleation bubble diameter [m] (use_inception=1 only). Default 30um.";
  parameter Real alpha_flash = 0.05
    "Void fraction above which d_b_eff returns to bulk d_b [-] (use_inception=1 only)";
  parameter Real tau_flash = 0.005
    "Flashing relaxation time [s] (use_relaxation=1). Controls how fast superheated
     liquid generates vapour relative to geometric model. Smaller = faster flashing.
     H_eff_relax = alpha*(1-alpha)*rho_l*cp_f/tau, giving enhancement ratio over
     geometric H_geo of: rho_l*cp_f*d_b^2 / (6*Nu*k_f*tau). At tau=0.005, ratio~10x.
     Ref: Jones (1982), Lahey (1978). Typical range: 0.001-0.05s.";

  // Generic source terms (phasic)
  parameter Real S_energy_l[N] = zeros(N) "Liquid energy source per cell [W]";
  parameter Real S_energy_v[N] = zeros(N) "Vapour energy source per cell [W]";
  parameter Real S_void[N] = zeros(N) "Vapour mass source per cell [kg/s]";

  // Two-phase friction multiplier
  parameter Boolean use_two_phase_friction = true "Enable Martinelli-Nelson friction multiplier";
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
  Real a_i[N] "Interfacial area concentration [1/m]";
  Real h_i[N] "Interfacial film heat transfer coefficient [W/(m^2*K)]";
  Real alpha_eff[N] "Effective void fraction (with nucleation) [-]";
  Real d_b_eff[N] "Effective bubble diameter (reduced during flashing inception) [m]";

  // Phasic mechanical compressibility (for block-coupled pressure-void solve)
  Real drho_l_dp[N] "Liquid compressibility at h_l [kg/(m^3*Pa)]";
  Real drho_v_dp[N] "Vapour compressibility at h_v [kg/(m^3*Pa)]";

  // Drift-flux (per cell)
  Real V_gj[N] "Drift velocity [m/s]";

  // ═══════════════════════════════════════════════════════════════════
  // Drift-flux phasic mass flows (per face, from algebraic slip)
  // ═══════════════════════════════════════════════════════════════════
  Real alpha_face[N + 1] "Face-averaged void fraction [-]";
  Real rho_l_face[N + 1] "Face-averaged liquid density [kg/m^3]";
  Real rho_v_face[N + 1] "Face-averaged vapour density [kg/m^3]";
  Real V_gj_face[N + 1] "Face-averaged drift velocity [m/s]";
  Real j_face[N + 1] "Volumetric flux at face [m/s]";
  Real mdot_v[N + 1] "Vapour mass flow at face [kg/s]";
  Real mdot_l[N + 1] "Liquid mass flow at face [kg/s]";

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
    // Phasic densities: use saturated values when enthalpy crosses saturation
    // (metastable extension — rho_l should be liquid density, not two-phase mixture)
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

    // Liquid temperature: metastable extension beyond saturation.
    // When h_l > h_f (superheated liquid after depressurization), the equilibrium
    // T_ph returns T_sat. For the 5-eq model we need the actual liquid temperature
    // to drive interfacial heat transfer and flashing.
    // T_l = T_sat + (h_l - h_f) / cp_f(p), using pressure-dependent cp_f.
    // Ref: RELAP5/MOD3 Vol I §3.2 — metastable liquid state extension.
    T_l[i] = if h_l[i] <= h_sat_l[i] then
               Medium.T_ph(p[i], h_l[i])
             else
               T_sat_cell[i] + (h_l[i] - h_sat_l[i]) / Medium.cp_f(p[i]);

    // Mixture derivatives for pressure linearisation.
    // Evaluated at h_mix (equilibrium mixture enthalpy). This is the CORRECT
    // effective compressibility for the semi-implicit scheme where enthalpies are
    // frozen during the pressure solve: drho_dp|_{h=h_mix} anticipates the density
    // change that void growth will cause, preventing pressure overshoot.
    //
    // The thermal compressibility (saturation curve shift) is physical, not
    // numerical damping — experimental data at GS-5 confirms a 1000x slowdown
    // in depressurization rate at the saturation crossing.
    //
    // Phasic drho_dp (mechanical compressibility only, ~5e-7) is correct only
    // WITH fully implicit void-pressure coupling (RELAP5-style block matrix).
    // Without it, the rarefaction wave overshoots to atmospheric (75.7% MAPE).
    // See docs/6eq_implementation_plan.md Session 3 for analysis.
    // Ref: RELAP5/MOD3 Vol I §3.1.2 (pressure linearization).
    drho_dp[i] = Medium.drho_dp_h(p[i], h_mix[i]);
    drho_dh[i] = Medium.drho_dh_p(p[i], h_mix[i]);

    // Phasic mechanical compressibility for block-coupled pressure-void solve.
    // Evaluated in single-phase regions only (Region 1 for liquid, Region 2 for
    // vapour). The 100 J/kg margin prevents evaluation at h = h_f/h_g exactly,
    // which falls into Region 4 due to strict inequality in region_ph.
    // These are used by the Schur complement solver; drho_dp (h_mix) above is
    // kept for critical flow sound speed and as fallback.
    drho_l_dp[i] = Medium.drho_dp_h(p[i], noEvent(min(h_l[i], h_sat_l[i] - 100.0)));
    drho_v_dp[i] = Medium.drho_dp_h(p[i], noEvent(max(h_v[i], h_sat_v[i] + 100.0)));
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Interfacial closures
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    // Nucleation onset: when T_l > T_sat, enforce minimum void
    alpha_eff[i] = if T_l[i] > T_sat_cell[i] and alpha[i] < alpha_nucleation
                   then alpha_nucleation else alpha[i];

    // Effective bubble diameter: ramp from d_b_flash to d_b with void fraction.
    // use_inception=0: d_b_eff = d_b (baseline).
    // use_inception=1: d_b_eff blends from d_b_flash to d_b over [alpha_nucleation, alpha_flash].
    //   WARNING: causes runaway in rapid depressurization (see use_inception parameter note).
    d_b_eff[i] = max(d_b - use_inception * (d_b - d_b_flash)
                     * (1 - noEvent(min(max((alpha_eff[i] - alpha_nucleation)
                                           / (alpha_flash - alpha_nucleation), 0.0), 1.0))),
                     d_b_min);

    // Interfacial area concentration [1/m] — bubbly flow geometry
    // a_i = 6*alpha*(1-alpha)/d_b_eff for monodisperse spherical bubbles.
    // Ref: Ishii & Hibiki, "Thermo-Fluid Dynamics of Two-Phase Flow", Ch. 9
    a_i[i] = 6 * alpha_eff[i] * (1 - alpha_eff[i]) / d_b_eff[i];

    // Interfacial film heat transfer coefficient [W/(m^2*K)]
    // Nu = 2 for conduction around sphere (Ranz-Marshall at Re_b = 0, no-slip).
    // Ref: Ranz & Marshall (1952), conduction limit.
    h_i[i] = Nu_i * Medium.k_f(p[i]) / d_b_eff[i];

    // Volumetric interfacial heat transfer [W/m^3]
    // Split into condensation (T_l < T_sat) and evaporation (T_l > T_sat) to apply
    // the Jones/Lahey relaxation ONLY to the evaporation direction.
    //
    // Condensation: q_cond = h_i*a_i * max(T_sat-T_l, 0)  [always geometric]
    // Evaporation:  q_evap = -H_eff_evap * max(T_l-T_sat, 0)
    //   H_eff_evap = h_i*a_i + use_relaxation * (H_relax - h_i*a_i)
    //   H_relax = α*(1-α)*ρ_l*cp_f/τ  (area-weighted relaxation)
    //   The α*(1-α) factor ensures H_relax scales with interfacial area, same as
    //   geometric H_geo ~ α*(1-α)/d_b². The enhancement ratio is then independent
    //   of α: ratio = ρ_l*cp_f*d_b² / (6*Nu*k_f*τ). Self-limiting at both α→0
    //   (no bubbles) and α→1 (no liquid).
    // Ref: Jones (1982), Lahey & Moody (1993), RELAP5/MOD3 Vol I §3.2.
    q_i_l[i] = h_i[i] * a_i[i] * noEvent(max(T_sat_cell[i] - T_l[i], 0.0))
              - (h_i[i] * a_i[i]
                 + use_relaxation
                   * (alpha_eff[i] * (1 - alpha_eff[i]) * rho_l[i]
                      * Medium.cp_f(p[i]) / tau_flash
                      - h_i[i] * a_i[i]))
                * noEvent(max(T_l[i] - T_sat_cell[i], 0.0));

    // Mass transfer from interfacial heat
    Gamma[i] = -q_i_l[i] / max(h_sat_v[i] - h_sat_l[i], 1.0);

    // Vapour heat: interface energy balance
    q_i_v[i] = -Gamma[i] * (h_v[i] - h_l[i]) - q_i_l[i];

    // Zuber-Findlay drift velocity (uses bulk d_b, not d_b_eff —
    // drift is set by established bubble population, not nucleation-scale bubbles)
    // V_gj = 1.41 * [σ * g * Δρ / ρ_l²]^0.25 * 4α(1-α)
    // Ref: Ishii & Hibiki, "Thermo-Fluid Dynamics of Two-Phase Flow", Eq. 11.21
    V_gj[i] = 1.41 * (Medium.sigma(p[i]) * 9.81 *
               max(rho_l[i] - rho_v[i], 0.01) /
               max(rho_l[i]^2, 1.0))^0.25 *
               4 * alpha[i] * (1 - alpha[i]);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Drift-flux phasic mass flow split (per face)
  //   From mixture mdot, compute phasic mdot_v, mdot_l using the
  //   drift-flux algebraic slip relation:
  //     j = [G_m - α·V_gj·(ρ_v - ρ_l)] / [ρ_l + α·C_0·(ρ_v - ρ_l)]
  //     v_v = C_0·j + V_gj
  //     mdot_v = α·ρ_v·v_v·A,  mdot_l = mdot - mdot_v
  //   Ref: Ishii & Hibiki, "Thermo-Fluid Dynamics of Two-Phase Flow"
  //   Verified against: archive/cpp_prototype/two_phase/five_eq_model.cpp:266-305
  // ─────────────────────────────────────────────────────────────────

  // Step 1: Face-averaged properties (boundary: nearest cell; interior: arithmetic mean)
  alpha_face[1] = alpha[1];
  rho_l_face[1] = rho_l[1];
  rho_v_face[1] = rho_v[1];
  V_gj_face[1] = V_gj[1];
  for i in 2:N loop
    alpha_face[i] = 0.5 * (alpha[i - 1] + alpha[i]);
    rho_l_face[i] = 0.5 * (rho_l[i - 1] + rho_l[i]);
    rho_v_face[i] = 0.5 * (rho_v[i - 1] + rho_v[i]);
    V_gj_face[i] = 0.5 * (V_gj[i - 1] + V_gj[i]);
  end for;
  alpha_face[N + 1] = alpha[N];
  rho_l_face[N + 1] = rho_l[N];
  rho_v_face[N + 1] = rho_v[N];
  V_gj_face[N + 1] = V_gj[N];

  // Step 2: Drift-flux split at each face
  for i in 1:N + 1 loop
    // Volumetric flux from drift-flux relation
    j_face[i] = (mdot[i] / A_flow - alpha_face[i] * V_gj_face[i] * (rho_v_face[i] - rho_l_face[i]))
                / max(rho_l_face[i] + alpha_face[i] * C_0 * (rho_v_face[i] - rho_l_face[i]), 1.0);

    // Vapour velocity: v_v = C_0*j + V_gj, then vapour mass flow
    mdot_v[i] = alpha_face[i] * rho_v_face[i] * (C_0 * j_face[i] + V_gj_face[i]) * A_flow;

    // Liquid mass flow from mixture mass conservation: mdot_l = mdot - mdot_v
    mdot_l[i] = mdot[i] - mdot_v[i];
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
  // VOID FRACTION (per cell) — vapour mass transport with drift-flux
  //   d(α*ρ_v)/dt = (mdot_v_in - mdot_v_out) / V + Γ
  //   NOTE: The Modelica form uses rho_v * der(alpha) which drops the
  //   α*der(rho_v) coupling term. The bridge solver handles the conservative
  //   update (alpha*rho_v product) in its explicit time-stepping, matching
  //   the C++ prototype (five_eq_model.cpp:394-399). This split avoids
  //   OM generating massive symbolic derivatives through the EOS chain rule.
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho_v[i] * V_cell * der(alpha[i])
      = mdot_v[i] - mdot_v[i + 1]
      + V_cell * Gamma[i]
      + S_void[i];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // LIQUID ENERGY (per cell) — phasic enthalpy with drift-flux advection
  //   (1-α)*ρ_l*V * dh_l/dt = mdot_l_in*(h_in-h_cell) - mdot_l_out*(h_out-h_cell)
  //                          + (1-α)*V*dp/dt + q_wall_l + q_i_l - Γ*h_l*V
  //   Donor-cell enthalpy selection based on mdot_l direction.
  //   Phase-absence guard: max(1-α, 1e-6) prevents singular ODE when α → 1.
  //   Ref: C++ prototype five_eq_model.cpp:439 checks m_l > m_phase_min.
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
  // VAPOUR ENERGY (per cell) — phasic enthalpy with drift-flux advection
  //   α*ρ_v*V * dh_v/dt = mdot_v_in*(h_in-h_cell) - mdot_v_out*(h_out-h_cell)
  //                      + α*V*dp/dt + q_wall_v + q_i_v + Γ*h_v*V
  //   Donor-cell enthalpy selection based on mdot_v direction.
  //   Phase-absence guard: max(α, 1e-6) prevents singular ODE when α → 0.
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
<p>Closures: linear relaxation interfacial HT, Zuber-Findlay drift velocity,
drift-flux phasic flux split (C_0, V_gj), Martinelli-Nelson two-phase friction.</p>
<p>Swap medium: <code>redeclare package Medium = library.Media.Water</code></p>
</html>"));
end Pipe1D_DriftFlux;
