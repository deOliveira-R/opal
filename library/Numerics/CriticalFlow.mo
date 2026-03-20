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

  annotation(Documentation(info="<html>
<p>Critical (choked) flow models for break boundaries.</p>
<p><b>ransom_trapp</b>: Quality-blended subcooled Bernoulli + HEM sound speed model.
Based on RELAP5/MOD3 Code Manual Vol 1, §3.5.</p>
<p>Subcooled: G_sub = sqrt(2 * rho_f * dp). HEM: G_hem = rho * c_hem.
Blend: smooth transition at quality x_trans (default 10%).</p>
</html>"));
end CriticalFlow;
