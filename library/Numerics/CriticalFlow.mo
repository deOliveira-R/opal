within library.Numerics;
package CriticalFlow "Critical (choked) flow models for break boundaries"

  function ransom_trapp
    "Ransom-Trapp critical mass flux [kg/s] at a break face"
    input Real p_cell "Cell pressure adjacent to break [Pa]";
    input Real h_mix "Mixture enthalpy at break cell [J/kg]";
    input Real rho "Mixture density at break cell [kg/m^3]";
    input Real drho_dp_h "drho/dp at constant h [kg/(m^3*Pa)]";
    input Real h_f "Saturated liquid enthalpy [J/kg]";
    input Real h_g "Saturated vapour enthalpy [J/kg]";
    input Real rho_f "Saturated liquid density [kg/m^3]";
    input Real p_back "Back-pressure [Pa]";
    input Real A_flow "Flow area [m^2]";
    input Real C_d "Discharge coefficient [-]";
    input Real x_trans "Quality transition for blend [-] (default 0.10)";
    input Real c_floor "Minimum sound speed [m/s] (default 10 — numerical floor only)";
    output Real mdot_crit "Critical mass flow rate [kg/s]";
  protected
    Real h_fg = max(h_g - h_f, 1e3);
    Real x_local;
    Real dp_sub, G_sub, c_hem, G_hem, G_crit;
    Real blend;
  algorithm
    // Local quality
    if h_mix <= h_f then
      x_local := 0.0;
    elseif h_mix >= h_g then
      x_local := 1.0;
    else
      x_local := (h_mix - h_f) / h_fg;
    end if;

    // Subcooled critical mass flux (Bernoulli discharge)
    dp_sub := max(p_cell - p_back, 0.0);
    G_sub := sqrt(2.0 * rho_f * dp_sub);

    // HEM critical mass flux: G = rho * c, where c = 1/sqrt(rho * drho_dp_h)
    if drho_dp_h > 0 then
      c_hem := max(sqrt(1.0 / (rho * drho_dp_h)), c_floor);
    else
      c_hem := c_floor;
    end if;
    G_hem := rho * c_hem;

    // Blend subcooled → HEM based on quality
    if x_local < x_trans then
      blend := x_local / x_trans;
      G_crit := G_sub * (1.0 - blend) + G_hem * blend;
    else
      G_crit := G_hem;
    end if;

    // Ensure critical flux doesn't fall below HEM value
    G_crit := max(G_crit, G_hem);

    mdot_crit := C_d * A_flow * G_crit;
    annotation(Inline=true);
  end ransom_trapp;

  function henry_fauske
    "Henry-Fauske critical mass flux [kg/s] — non-equilibrium model.
     For subcooled and low-quality blowdowns where flashing is delayed.
     Ref: Henry & Fauske (1971), J. Heat Transfer 93(2):179-187.
     Also: RELAP5/MOD3 Code Manual Vol 1, Section 3.5.2."
    input Real p_cell "Cell pressure adjacent to break [Pa]";
    input Real h_mix "Mixture enthalpy at break cell [J/kg]";
    input Real rho "Mixture density at break cell [kg/m^3]";
    input Real drho_dp_h "drho/dp at constant h [kg/(m^3*Pa)]";
    input Real h_f "Saturated liquid enthalpy [J/kg]";
    input Real h_g "Saturated vapour enthalpy [J/kg]";
    input Real rho_f "Saturated liquid density [kg/m^3]";
    input Real rho_g "Saturated vapour density [kg/m^3]";
    input Real rho_f_c "Saturated liquid density at throat pressure [kg/m^3]";
    input Real rho_g_c "Saturated vapour density at throat pressure [kg/m^3]";
    input Real p_back "Back-pressure [Pa]";
    input Real A_flow "Flow area [m^2]";
    input Real C_d "Discharge coefficient [-]";
    input Real x_ne "Non-equilibrium transition quality [-] (0.05 for sharp orifice)";
    input Real N_param "Non-equilibrium parameter [-] (0=frozen, 1=full HF)";
    input Real c_floor "Minimum sound speed [m/s] (numerical floor only)";
    input Real c_cell "Sound speed at cell conditions [m/s] (0 = disable acoustic limit)";
    output Real mdot_crit "Critical mass flow rate [kg/s]";
  protected
    Real h_fg = max(h_g - h_f, 1e3);
    Real x_e "Equilibrium quality [-]";
    Real dp, G_sub "Subcooled Bernoulli";
    Real c_hem, G_HEM "HEM critical flux";
    Real N_eff "Effective non-equilibrium parameter [-]";
    Real p_c "Critical (throat) pressure [Pa]";
    Real v_f "Sat. liquid specific volume [m^3/kg]";
    Real v_fg "Specific volume difference v_g - v_f [m^3/kg]";
    Real dp_c "Pressure drop to throat [Pa]";
    Real denom_corr "Henry-Fauske denominator correction [-]";
    Real G_HF "Henry-Fauske non-equilibrium mass flux [kg/(m^2*s)]";
    Real G_crit "Selected critical mass flux [kg/(m^2*s)]";
    Real blend "Blending factor [-]";
  algorithm
    // ── Equilibrium quality ──
    if h_mix <= h_f then
      x_e := 0.0;
    elseif h_mix >= h_g then
      x_e := 1.0;
    else
      x_e := (h_mix - h_f) / h_fg;
    end if;

    // ── Subcooled Bernoulli mass flux ──
    dp := max(p_cell - p_back, 0.0);
    G_sub := sqrt(2.0 * rho_f * dp);

    // ── HEM critical mass flux (high-quality branch) ──
    if drho_dp_h > 0 then
      c_hem := max(sqrt(1.0 / (rho * drho_dp_h)), c_floor);
    else
      c_hem := c_floor;
    end if;
    G_HEM := rho * c_hem;

    // ── Henry-Fauske non-equilibrium mass flux (low-quality branch) ──
    // N_eff ramps from 0 (frozen, no flashing at throat) to N_param
    // as quality increases from 0 to x_ne.
    // N_param = 0 for sharp orifice (Edwards glass disk): frozen flow.
    // N_param = 1 for long nozzle (L/D > 12): full HF correction.
    N_eff := N_param * min(x_e / max(x_ne, 1e-6), 1.0);

    // Critical pressure: isentropic incompressible limit = 2/3 * p_0,
    // but not below back-pressure.
    p_c := max(p_back, (2.0 / 3.0) * p_cell);

    // Specific volume terms evaluated at throat pressure p_c.
    // rho_f_c and rho_g_c are saturation densities at p_c, passed by the caller.
    v_f := 1.0 / rho_f_c;
    v_fg := 1.0 / max(rho_g_c, 0.01) - v_f;

    // Pressure drop to throat
    dp_c := max(p_cell - p_c, 0.0);

    // HF correction denominator: accounts for two-phase choking at throat.
    // When N_eff = 0: denom = 1 → G_HF = Bernoulli at p_c (frozen flow).
    // When N_eff > 0: denom > 1 → G_HF < Bernoulli (flashing reduces flux).
    denom_corr := 1.0 + 2.0 * N_eff * x_e * v_fg * dp_c
                  / max(p_c * v_f, 1.0);
    denom_corr := max(denom_corr, 0.01);
    G_HF := sqrt(2.0 * rho_f * dp_c / denom_corr);

    // ── Regime selection and blending ──
    if x_e <= 0.0 then
      // Subcooled: pure Bernoulli
      G_crit := G_sub;
    elseif x_e < x_ne then
      // Low quality: blend HF → HEM as quality increases toward x_ne
      blend := x_e / x_ne;
      G_crit := G_HF * (1.0 - blend) + G_HEM * blend;
    else
      // High quality: full HEM (same as Ransom-Trapp)
      G_crit := G_HEM;
    end if;

    // Ensure critical flux is at least HEM value
    G_crit := max(G_crit, G_HEM);

    // Acoustic choking limit: critical flux cannot exceed rho*c_cell.
    // When c_cell = 0, this limit is disabled (backward compatibility).
    // Ref: RELAP5/MOD3 Vol I §3.5.1 — subcooled choking is min(Bernoulli, acoustic).
    if c_cell > c_floor then
      G_crit := min(G_crit, rho * c_cell);
    end if;

    mdot_crit := C_d * A_flow * G_crit;
    annotation(Inline=true);
  end henry_fauske;

  annotation(Documentation(info="<html>
<p>Critical (choked) flow models for break boundaries.</p>
<p><b>ransom_trapp</b>: Quality-blended subcooled Bernoulli + HEM sound speed model.
Based on RELAP5/MOD3 Code Manual Vol 1, §3.5.1.</p>
<p><b>henry_fauske</b>: Non-equilibrium model for subcooled/low-quality blowdowns.
Accounts for delayed flashing at the break plane (frozen flow for sharp orifices).
Based on Henry &amp; Fauske (1971), J. Heat Transfer 93(2):179-187.
See also RELAP5/MOD3 Code Manual Vol 1, §3.5.2.</p>
</html>"));
end CriticalFlow;
