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

  annotation(Documentation(info="<html>
<p>Abstract interface for OPAL thermodynamic media packages.</p>
<p>All media (SimpleFluid, Water) implement these four functions.
Components use <code>replaceable package Medium constrainedby PartialMedium</code>
to allow fluid swapping without changing equations.</p>
</html>"));
end PartialMedium;
