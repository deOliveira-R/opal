within library.Media;
partial package PartialMedium
  "Abstract medium interface — all OPAL media packages implement this API"

  function rho_ph "Density [kg/m^3] from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real rho_val;
  end rho_ph;

  function drho_dp_h "Partial derivative drho/dp at constant h [kg/(m^3*Pa)]"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real val;
  end drho_dp_h;

  function drho_dh_p "Partial derivative drho/dh at constant p [kg/(m^3 * J/kg)]"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real val;
  end drho_dh_p;

  function T_ph "Temperature [K] from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real T_val;
  end T_ph;

  // ═══════════════════════════════════════════════════════════════════
  // Phasic (saturation) properties — needed by multi-equation models
  // ═══════════════════════════════════════════════════════════════════

  function T_sat "Saturation temperature [K] from pressure"
    input Real p "Pressure [Pa]";
    output Real T_val;
  end T_sat;

  function h_f "Saturated liquid enthalpy [J/kg] from pressure"
    input Real p "Pressure [Pa]";
    output Real h_val;
  end h_f;

  function h_g "Saturated vapour enthalpy [J/kg] from pressure"
    input Real p "Pressure [Pa]";
    output Real h_val;
  end h_g;

  function h_fg "Latent heat of vaporisation [J/kg] from pressure"
    input Real p "Pressure [Pa]";
    output Real h_val;
  end h_fg;

  function rho_f "Saturated liquid density [kg/m^3] from pressure"
    input Real p "Pressure [Pa]";
    output Real rho_val;
  end rho_f;

  function rho_g "Saturated vapour density [kg/m^3] from pressure"
    input Real p "Pressure [Pa]";
    output Real rho_val;
  end rho_g;

  function sigma "Surface tension [N/m] from pressure"
    input Real p "Pressure [Pa]";
    output Real sigma_val;
  end sigma;

  annotation(Documentation(info="<html>
<p>Abstract interface for OPAL thermodynamic media packages.</p>
<p>All media (SimpleFluid, Water) implement these functions:</p>
<ul>
<li>Mixture: <code>rho_ph</code>, <code>T_ph</code>, <code>drho_dp_h</code>, <code>drho_dh_p</code></li>
<li>Phasic: <code>T_sat</code>, <code>h_f</code>, <code>h_g</code>, <code>h_fg</code>, <code>rho_f</code>, <code>rho_g</code></li>
</ul>
<p>Components use <code>replaceable package Medium constrainedby PartialMedium</code>
to allow fluid swapping without changing equations.</p>
</html>"));
end PartialMedium;
