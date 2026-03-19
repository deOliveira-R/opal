within library.Media.IF97;
package Saturation "IAPWS-IF97 Region 4 — saturation curve"

  // ---------------------------------------------------------------------------
  // IAPWS-IF97 §8, Eq. (30): saturation pressure as function of temperature
  //
  //   p_sat(T):  ϑ = T + n9/(T−n10)
  //              A = ϑ² + n1 ϑ + n2
  //              B = n3 ϑ² + n4 ϑ + n5
  //              C = n6 ϑ² + n7 ϑ + n8
  //              p* = p_star_sat (1 MPa)
  //              p_sat = p* (2C / (−B + √(B²−4AC)))²
  //
  // Valid: T ∈ [273.15, 647.096] K
  // ---------------------------------------------------------------------------
  constant Real n_sat[10] = {
     1.1670521452767e3,
    -7.2421316703206e5,
    -1.7073846940092e1,
     1.2020824702470e4,
    -3.2325550322333e6,
     1.4915108613530e1,
    -4.8232657361591e3,
     4.0511340542057e5,
    -2.3855557567849e-1,
     6.5017534844798e2};

  function p_sat "Saturation pressure [Pa]  (IAPWS-IF97 Eq. 30)"
    input Real T "Temperature [K]";
    output Real p_val;
  protected
    Real theta = T + n_sat[9] / (T - n_sat[10]);
    Real A = theta^2 + n_sat[1] * theta + n_sat[2];
    Real B = n_sat[3] * theta^2 + n_sat[4] * theta + n_sat[5];
    Real C = n_sat[6] * theta^2 + n_sat[7] * theta + n_sat[8];
  algorithm
    // p in MPa, convert to Pa
    p_val := 1.0e6 * (2.0 * C / (-B + sqrt(B^2 - 4.0 * A * C)))^4;
  end p_sat;

  // ---------------------------------------------------------------------------
  // Inverse: T_sat(p) via Newton iteration on p_sat(T) = p
  // Starting guess: Wagner approximation (IAPWS-IF97 Eq. 31)
  //   T_sat(p) = β/2 + ... where β = (p/p*)^0.25
  // ---------------------------------------------------------------------------
  function T_sat "Saturation temperature [K]  (Newton on p_sat)"
    input Real p "Pressure [Pa]";
    output Real T_val;
  protected
    // Wagner-form direct approximation as starting guess
    Real beta  = (p / 1.0e6)^0.25;
    Real E = beta^2 + n_sat[3] * beta + n_sat[6];
    Real F = n_sat[1] * beta^2 + n_sat[4] * beta + n_sat[7];
    Real G = n_sat[2] * beta^2 + n_sat[5] * beta + n_sat[8];
    Real D = 2.0 * G / (-F - sqrt(F^2 - 4.0 * E * G));
    Real T_guess = (n_sat[10] + D - sqrt((n_sat[10] + D)^2 - 4.0 * (n_sat[9] + n_sat[10] * D))) / 2.0;
    Real T_iter;
    Real f, dfdT, theta, A, B, C, dA, dB, dC, dtheta;
    Real dp_dT;
    Integer iter;
  algorithm
    T_iter := T_guess;
    for iter in 1:10 loop
      f := p_sat(T_iter) - p;
      // Numerical derivative dp_sat/dT (central difference, ~0.01 K step)
      dp_dT := (p_sat(T_iter + 0.005) - p_sat(T_iter - 0.005)) / 0.01;
      T_iter := T_iter - f / dp_dT;
    end for;
    T_val := T_iter;
  end T_sat;

  // ---------------------------------------------------------------------------
  // Saturation-line boundary properties
  //   rho_f = liquid density at saturation     (Region 1 at (p, T_sat))
  //   rho_g = vapour density at saturation     (Region 2 at (p, T_sat))
  //   h_f   = liquid enthalpy at saturation    (Region 1 at (p, T_sat))
  //   h_g   = vapour enthalpy at saturation    (Region 2 at (p, T_sat))
  // ---------------------------------------------------------------------------
  function rho_f "Saturated liquid density [kg/m³]"
    input Real p "Pressure [Pa]";
    output Real rho_val;
  protected
    Real T_s = T_sat(p);
  algorithm
    rho_val := Region1.rho_pT(p, T_s);
  end rho_f;

  function rho_g "Saturated vapour density [kg/m³]"
    input Real p "Pressure [Pa]";
    output Real rho_val;
  protected
    Real T_s = T_sat(p);
  algorithm
    rho_val := Region2.rho_pT(p, T_s);
  end rho_g;

  function h_f "Saturated liquid enthalpy [J/kg]"
    input Real p "Pressure [Pa]";
    output Real h_val;
  protected
    Real T_s = T_sat(p);
  algorithm
    h_val := Region1.h_pT(p, T_s);
  end h_f;

  function h_g "Saturated vapour enthalpy [J/kg]"
    input Real p "Pressure [Pa]";
    output Real h_val;
  protected
    Real T_s = T_sat(p);
  algorithm
    h_val := Region2.h_pT(p, T_s);
  end h_g;

  function h_fg "Latent heat of vaporisation [J/kg]"
    input Real p "Pressure [Pa]";
    output Real h_val;
  algorithm
    h_val := h_g(p) - h_f(p);
  end h_fg;

  // ---------------------------------------------------------------------------
  // Two-phase quality and density  (used by Water.mo for region 4 states)
  //   x = (h - h_f) / (h_g - h_f)
  //   rho = 1 / (x/rho_g + (1-x)/rho_f)    (mixture specific volume)
  // ---------------------------------------------------------------------------
  function quality_ph "Steam quality x ∈ [0,1] from (p,h) in two-phase region"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real x_val;
  algorithm
    x_val := (h - h_f(p)) / h_fg(p);
  end quality_ph;

  function rho_ph_2phase "Two-phase mixture density [kg/m³]"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real rho_val;
  protected
    Real x   = quality_ph(p, h);
    Real rf  = rho_f(p);
    Real rg  = rho_g(p);
  algorithm
    rho_val := 1.0 / (x / rg + (1.0 - x) / rf);
  end rho_ph_2phase;

  annotation(Documentation(info="<html>
<p><b>IAPWS-IF97 Region 4 — saturation curve</b></p>
<p>Valid range: T ∈ [273.15, 647.096] K  (triple point to critical point).</p>
<p>p_sat(T): IAPWS-IF97 Eq. (30) — exact, direct formula.</p>
<p>T_sat(p): inverse via Newton iteration on p_sat; starting guess from IAPWS-IF97 Eq. (31).</p>
<p>Boundary properties h_f, h_g, rho_f, rho_g obtained by evaluating Region 1 and Region 2
at (p, T_sat(p)).</p>
</html>"));
end Saturation;
