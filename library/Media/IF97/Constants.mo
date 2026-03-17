within OPAL.library.Media.IF97;
package Constants "IAPWS-IF97 universal and region-specific constants"
  // --- Universal ---
  constant Real R = 461.526
    "Specific gas constant for water [J/(kg·K)]  (IAPWS-IF97 Table 1)";
  constant Real Tc = 647.096
    "Critical temperature [K]";
  constant Real pc = 22.064e6
    "Critical pressure [Pa]";
  constant Real rhoc = 322.0
    "Critical density [kg/m³]";

  // --- Region 1 reducing properties ---
  constant Real p_star_R1 = 16.53e6
    "Reducing pressure for Region 1 [Pa]  (IAPWS-IF97 §6)";
  constant Real T_star_R1 = 1386.0
    "Reducing temperature for Region 1 [K]";

  // --- Region 2 reducing properties ---
  constant Real p_star_R2 = 1.0e6
    "Reducing pressure for Region 2 [Pa]  (IAPWS-IF97 §7)";
  constant Real T_star_R2 = 540.0
    "Reducing temperature for Region 2 [K]";

  // --- Saturation (Region 4) ---
  constant Real T_star_sat = 1.0
    "Saturation equation uses T/K directly (no separate reducing T)";
  constant Real p_star_sat = 1.0e6
    "Saturation pressure is expressed in MPa then converted; reducing factor [Pa]";

  annotation(Documentation(info="<html>
<p>Physical constants from IAPWS-IF97 Table 1 and region-specific reducing quantities.</p>
<p>R = 461.526 J/(kg·K) is the IAPWS value, slightly different from CODATA R/M
because IAPWS uses M = 18.015268 g/mol.</p>
</html>"));
end Constants;
