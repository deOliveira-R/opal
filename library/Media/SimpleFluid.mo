within library.Media;
package SimpleFluid "Synthetic test fluid — linear properties for rigorous two-phase solver verification"
  extends library.Media.PartialMedium;

  // =========================================================================
  // Constants
  // =========================================================================
  constant Real p_ref = 10.0e6 "Reference pressure [Pa]";

  // Saturation curve: property = X_0 + X_1 * p_hat,  p_hat = (p - p_ref)/p_ref
  constant Real T_sat_0   = 400.0   "Saturation temperature at p_ref [K]";
  constant Real T_sat_1   = 20.0    "dT_sat/dp_hat [K]";

  constant Real h_f_0     = 800.0e3 "Saturated liquid enthalpy at p_ref [J/kg]";
  constant Real h_f_1     = 100.0e3 "dh_f/dp_hat [J/kg]";

  constant Real h_g_0     = 2800.0e3 "Saturated vapour enthalpy at p_ref [J/kg]";
  constant Real h_g_1     = 50.0e3   "dh_g/dp_hat [J/kg]";

  constant Real rho_f_0   = 750.0   "Saturated liquid density at p_ref [kg/m^3]";
  constant Real rho_f_1   = 20.0    "drho_f/dp_hat [kg/m^3]";

  constant Real rho_g_0   = 40.0    "Saturated vapour density at p_ref [kg/m^3]";
  constant Real rho_g_1   = 5.0     "drho_g/dp_hat [kg/m^3]";

  // Single-phase density slope from saturation boundary
  constant Real A_L = 6.25e-5 "Liquid density slope [kg/(m^3 * J/kg)]";
  constant Real A_G = 2.0e-5  "Vapour density slope [kg/(m^3 * J/kg)]";

  // Effective specific heats for temperature
  constant Real cp_L = 4000.0 "Liquid specific heat [J/(kg*K)]";
  constant Real cp_G = 2000.0 "Vapour specific heat [J/(kg*K)]";

  // =========================================================================
  // Saturation functions
  // =========================================================================
  function T_sat "Saturation temperature [K]"
    input Real p "Pressure [Pa]";
    output Real T_val;
  algorithm
    T_val := T_sat_0 + T_sat_1 * (p - p_ref) / p_ref;
    annotation(Inline=true);
  end T_sat;

  function h_f "Saturated liquid enthalpy [J/kg]"
    input Real p "Pressure [Pa]";
    output Real h_val;
  algorithm
    h_val := h_f_0 + h_f_1 * (p - p_ref) / p_ref;
    annotation(Inline=true);
  end h_f;

  function h_g "Saturated vapour enthalpy [J/kg]"
    input Real p "Pressure [Pa]";
    output Real h_val;
  algorithm
    h_val := h_g_0 + h_g_1 * (p - p_ref) / p_ref;
    annotation(Inline=true);
  end h_g;

  function h_fg "Latent heat [J/kg]"
    input Real p "Pressure [Pa]";
    output Real h_val;
  algorithm
    h_val := h_g(p) - h_f(p);
    annotation(Inline=true);
  end h_fg;

  function rho_f "Saturated liquid density [kg/m^3]"
    input Real p "Pressure [Pa]";
    output Real rho_val;
  algorithm
    rho_val := rho_f_0 + rho_f_1 * (p - p_ref) / p_ref;
    annotation(Inline=true);
  end rho_f;

  function rho_g "Saturated vapour density [kg/m^3]"
    input Real p "Pressure [Pa]";
    output Real rho_val;
  algorithm
    rho_val := rho_g_0 + rho_g_1 * (p - p_ref) / p_ref;
    annotation(Inline=true);
  end rho_g;

  function sigma "Surface tension [N/m] — constant approximation"
    input Real p "Pressure [Pa]";
    output Real sigma_val;
  algorithm
    // Typical water surface tension at ~10 MPa (~0.02 N/m)
    // Linear decrease toward critical point
    sigma_val := 0.06 - 0.04 * (p - p_ref) / p_ref;
    annotation(Inline=true);
  end sigma;

  // =========================================================================
  // Region detection
  // =========================================================================
  function region_ph "Region flag: 1=liquid, 2=steam, 4=two-phase"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Integer region;
  algorithm
    if h < h_f(p) then
      region := 1;
    elseif h > h_g(p) then
      region := 2;
    else
      region := 4;
    end if;
    annotation(Inline=true);
  end region_ph;

  // =========================================================================
  // Two-phase
  // =========================================================================
  function quality_ph "Steam quality x from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real x_val;
  algorithm
    x_val := (h - h_f(p)) / h_fg(p);
    annotation(Inline=true);
  end quality_ph;

  function rho_ph_2phase "Two-phase mixture density [kg/m^3]"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real rho_val;
  protected
    Real x   = quality_ph(p, h);
    Real rf  = rho_f(p);
    Real rg  = rho_g(p);
  algorithm
    rho_val := 1.0 / (x / rg + (1.0 - x) / rf);
    annotation(Inline=true);
  end rho_ph_2phase;

  // =========================================================================
  // Primary API
  // =========================================================================
  function rho_ph "Density [kg/m^3] from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real rho_val;
  protected
    Integer reg = region_ph(p, h);
  algorithm
    if reg == 1 then
      rho_val := rho_f(p) + A_L * (h_f(p) - h);
    elseif reg == 2 then
      rho_val := rho_g(p) - A_G * (h - h_g(p));
    else
      rho_val := rho_ph_2phase(p, h);
    end if;
    annotation(Inline=true);
  end rho_ph;

  function T_ph "Temperature [K] from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real T_val;
  protected
    Integer reg = region_ph(p, h);
  algorithm
    if reg == 1 then
      T_val := T_sat(p) - (h_f(p) - h) / cp_L;
    elseif reg == 2 then
      T_val := T_sat(p) + (h - h_g(p)) / cp_G;
    else
      T_val := T_sat(p);
    end if;
    annotation(Inline=true);
  end T_ph;

  // =========================================================================
  // Analytical derivatives
  //
  // Region 1:
  //   rho = rho_f(p) + A_L * (h_f(p) - h)
  //   drho/dp|h = drho_f/dp + A_L * dh_f/dp
  //             = rho_f_1/p_ref + A_L * h_f_1/p_ref     (constant)
  //   drho/dh|p = -A_L                                   (constant)
  //
  // Region 2:
  //   rho = rho_g(p) - A_G * (h - h_g(p))
  //   drho/dp|h = drho_g/dp + A_G * dh_g/dp
  //             = rho_g_1/p_ref + A_G * h_g_1/p_ref      (constant)
  //   drho/dh|p = -A_G                                    (constant)
  //
  // Region 4:
  //   v = x/rho_g + (1-x)/rho_f,  x = (h - h_f) / h_fg
  //   drho/dh|p = -rho^2 * (1/rho_g - 1/rho_f) / h_fg
  //   drho/dp|h = -rho^2 * dv/dp|h  (fully analytical, see below)
  // =========================================================================
  function drho_dp_h "Partial derivative drho/dp at constant h [kg/(m^3*Pa)]"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real val;
  protected
    Integer reg = region_ph(p, h);
    // Two-phase intermediates
    Real rf, rg, hfv, hgv, hfgv, x, vf, vg, v, rho2;
    Real drf_dp, drg_dp, dhf_dp, dhg_dp, dhfg_dp;
    Real dvf_dp, dvg_dp, dx_dp, dv_dp;
  algorithm
    if reg == 1 then
      val := (rho_f_1 + A_L * h_f_1) / p_ref;
    elseif reg == 2 then
      val := (rho_g_1 + A_G * h_g_1) / p_ref;
    else
      // Fully analytical two-phase derivative
      rf    := rho_f(p);
      rg    := rho_g(p);
      hfv   := h_f(p);
      hgv   := h_g(p);
      hfgv  := hgv - hfv;
      x     := (h - hfv) / hfgv;
      vf    := 1.0 / rf;
      vg    := 1.0 / rg;
      v     := x * vg + (1.0 - x) * vf;
      rho2  := 1.0 / (v * v);

      // All saturation dp derivatives are constants
      drf_dp  := rho_f_1 / p_ref;
      drg_dp  := rho_g_1 / p_ref;
      dhf_dp  := h_f_1 / p_ref;
      dhg_dp  := h_g_1 / p_ref;
      dhfg_dp := dhg_dp - dhf_dp;

      dvf_dp := -drf_dp / (rf * rf);
      dvg_dp := -drg_dp / (rg * rg);

      // dx/dp|h = (-dh_f/dp - x * dh_fg/dp) / h_fg
      dx_dp := (-dhf_dp - x * dhfg_dp) / hfgv;

      // dv/dp|h = dx/dp*(vg-vf) + x*dvg/dp + (1-x)*dvf/dp
      dv_dp := dx_dp * (vg - vf) + x * dvg_dp + (1.0 - x) * dvf_dp;

      val := -rho2 * dv_dp;
    end if;
    annotation(Inline=true);
  end drho_dp_h;

  function drho_dh_p "Partial derivative drho/dh at constant p [kg/(m^3 * J/kg)]"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real val;
  protected
    Integer reg = region_ph(p, h);
    Real rf, rg, hfgv, rho_mix;
  algorithm
    if reg == 1 then
      val := -A_L;
    elseif reg == 2 then
      val := -A_G;
    else
      rf      := rho_f(p);
      rg      := rho_g(p);
      hfgv    := h_fg(p);
      rho_mix := rho_ph_2phase(p, h);
      val     := -rho_mix * rho_mix * (1.0 / rg - 1.0 / rf) / hfgv;
    end if;
    annotation(Inline=true);
  end drho_dh_p;

  // =========================================================================
  // Convenience: (p, T) interface
  // =========================================================================
  function h_pT "Enthalpy [J/kg] from (p, T)"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real h_val;
  protected
    Real T_s = T_sat(p);
  algorithm
    if T < T_s then
      h_val := h_f(p) - cp_L * (T_s - T);
    else
      h_val := h_g(p) + cp_G * (T - T_s);
    end if;
    annotation(Inline=true);
  end h_pT;

  function rho_pT "Density [kg/m^3] from (p, T)"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real rho_val;
  algorithm
    rho_val := rho_ph(p, h_pT(p, T));
    annotation(Inline=true);
  end rho_pT;

  annotation(Documentation(info="<html>
<p><b>Synthetic test fluid for two-phase solver verification.</b></p>
<p>All saturation properties are linear in pressure. Single-phase densities are
bilinear in (p, h). Every single-phase derivative is a constant.</p>
<p>Same API as <code>Water</code> — drop-in replacement for solver testing.</p>
<p>Use this fluid to isolate solver bugs from property-evaluation bugs.
IAPWS-IF97 has 34+43-term polynomials; any test failure could be the solver
or the properties. SimpleFluid makes every property hand-verifiable.</p>
</html>"));
end SimpleFluid;
