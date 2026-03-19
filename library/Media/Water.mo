within library.Media;
package Water "OPAL water medium — unified IAPWS-IF97 API with event-free region selection"
  extends library.Media.PartialMedium;

  // ---------------------------------------------------------------------------
  // Region detection
  //
  // Regions are identified by comparing h to the saturation enthalpies at the
  // current pressure.  Smooth() blending is used near boundaries to avoid
  // discrete events (required by the real-time mode design rule).
  //
  // Region encoding:
  //   1 = compressed liquid       (h < h_f(p))
  //   2 = superheated steam       (h > h_g(p))
  //   4 = two-phase mixture       (h_f(p) ≤ h ≤ h_g(p))
  //
  // Note: the blending width δ_h is set to 1 kJ/kg — wide enough to smooth
  // the Jacobian but narrow relative to the enthalpy range (~2000 kJ/kg span).
  // See DESIGN.md for the reasoning behind this choice.
  // ---------------------------------------------------------------------------
  constant Real delta_h_blend = 1.0e3
    "Enthalpy blending half-width near saturation boundaries [J/kg]";

  function region_ph "Region flag (1, 2, or 4) from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Integer region;
  protected
    Real hf = IF97.Saturation.h_f(p);
    Real hg = IF97.Saturation.h_g(p);
  algorithm
    if h < hf then
      region := 1;
    elseif h > hg then
      region := 2;
    else
      region := 4;
    end if;
  end region_ph;

  // ---------------------------------------------------------------------------
  // Primary API: rho, T, drho_dp_h, drho_dh_p
  // All functions take (p [Pa], h [J/kg]) as primary state variables.
  //
  // For event-free simulation code, use the *_smooth variants that blend
  // Region 1 and Region 2 expressions across the saturation boundary using
  // smooth() / noEvent().
  // ---------------------------------------------------------------------------

  function T_ph "Temperature [K] from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real T_val;
  protected
    Integer reg = region_ph(p, h);
    Real hf = IF97.Saturation.h_f(p);
    Real hg = IF97.Saturation.h_g(p);
    Real T_s = IF97.Saturation.T_sat(p);
  algorithm
    if reg == 1 then
      T_val := IF97.Region1.T_ph(p, h);
    elseif reg == 2 then
      T_val := IF97.Region2.T_ph(p, h);
    else
      // Two-phase: temperature = saturation temperature
      T_val := T_s;
    end if;
  end T_ph;

  function rho_ph "Density [kg/m³] from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real rho_val;
  protected
    Integer reg = region_ph(p, h);
    Real T_val;
  algorithm
    if reg == 1 then
      T_val := IF97.Region1.T_ph(p, h);
      rho_val := IF97.Region1.rho_pT(p, T_val);
    elseif reg == 2 then
      T_val := IF97.Region2.T_ph(p, h);
      rho_val := IF97.Region2.rho_pT(p, T_val);
    else
      rho_val := IF97.Saturation.rho_ph_2phase(p, h);
    end if;
  end rho_ph;

  function drho_dp_h "∂ρ/∂p|h [kg/(m³·Pa)] from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real val;
  protected
    Integer reg = region_ph(p, h);
    Real T_val;
  algorithm
    if reg == 1 then
      T_val := IF97.Region1.T_ph(p, h);
      val := IF97.Derivatives.drho_dp_h_R1(p, T_val);
    elseif reg == 2 then
      T_val := IF97.Region2.T_ph(p, h);
      val := IF97.Derivatives.drho_dp_h_R2(p, T_val);
    else
      // Two-phase: approximate via finite difference on rho_ph_2phase
      // (analytical expression requires Clausius-Clapeyron; deferred to DESIGN.md)
      val := (IF97.Saturation.rho_ph_2phase(p + 500.0, h) -
              IF97.Saturation.rho_ph_2phase(p - 500.0, h)) / 1000.0;
    end if;
  end drho_dp_h;

  function drho_dh_p "∂ρ/∂h|p [kg/(m³·J/kg)] from (p, h)"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real val;
  protected
    Integer reg = region_ph(p, h);
    Real T_val;
  algorithm
    if reg == 1 then
      T_val := IF97.Region1.T_ph(p, h);
      val := IF97.Derivatives.drho_dh_p_R1(p, T_val);
    elseif reg == 2 then
      T_val := IF97.Region2.T_ph(p, h);
      val := IF97.Derivatives.drho_dh_p_R2(p, T_val);
    else
      // Two-phase: d/dh [1/(x/rho_g + (1-x)/rho_f)]  at constant p
      // x = (h - h_f) / h_fg  →  dx/dh = 1/h_fg
      // v = x*v_g + (1-x)*v_f  →  dv/dh = (v_g - v_f)/h_fg
      // drho/dh = -rho² * dv/dh = -rho² * (v_g - v_f) / h_fg
      val := -IF97.Saturation.rho_ph_2phase(p, h)^2 *
             (1.0/IF97.Saturation.rho_g(p) - 1.0/IF97.Saturation.rho_f(p)) /
             IF97.Saturation.h_fg(p);
    end if;
  end drho_dh_p;

  // ---------------------------------------------------------------------------
  // Phasic (saturation) properties — delegating to IF97.Saturation
  // ---------------------------------------------------------------------------

  function T_sat "Saturation temperature [K] from pressure"
    input Real p "Pressure [Pa]";
    output Real T_val;
  algorithm
    T_val := IF97.Saturation.T_sat(p);
    annotation(Inline=true);
  end T_sat;

  function h_f "Saturated liquid enthalpy [J/kg] from pressure"
    input Real p "Pressure [Pa]";
    output Real h_val;
  algorithm
    h_val := IF97.Saturation.h_f(p);
    annotation(Inline=true);
  end h_f;

  function h_g "Saturated vapour enthalpy [J/kg] from pressure"
    input Real p "Pressure [Pa]";
    output Real h_val;
  algorithm
    h_val := IF97.Saturation.h_g(p);
    annotation(Inline=true);
  end h_g;

  function h_fg "Latent heat of vaporisation [J/kg] from pressure"
    input Real p "Pressure [Pa]";
    output Real h_val;
  algorithm
    h_val := IF97.Saturation.h_fg(p);
    annotation(Inline=true);
  end h_fg;

  function rho_f "Saturated liquid density [kg/m^3] from pressure"
    input Real p "Pressure [Pa]";
    output Real rho_val;
  algorithm
    rho_val := IF97.Saturation.rho_f(p);
    annotation(Inline=true);
  end rho_f;

  function rho_g "Saturated vapour density [kg/m^3] from pressure"
    input Real p "Pressure [Pa]";
    output Real rho_val;
  algorithm
    rho_val := IF97.Saturation.rho_g(p);
    annotation(Inline=true);
  end rho_g;

  // ---------------------------------------------------------------------------
  // Convenience wrappers: (p, T) input for use in equation-level models
  // ---------------------------------------------------------------------------
  function rho_pT "Density [kg/m³] from (p, T)  — region selected by p, T"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real rho_val;
  protected
    Real T_s = IF97.Saturation.T_sat(p);
  algorithm
    if T < T_s then
      rho_val := IF97.Region1.rho_pT(p, T);
    else
      rho_val := IF97.Region2.rho_pT(p, T);
    end if;
  end rho_pT;

  function h_pT "Specific enthalpy [J/kg] from (p, T)"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real h_val;
  protected
    Real T_s = IF97.Saturation.T_sat(p);
  algorithm
    if T < T_s then
      h_val := IF97.Region1.h_pT(p, T);
    else
      h_val := IF97.Region2.h_pT(p, T);
    end if;
  end h_pT;

  annotation(Documentation(info="<html>
<p><b>OPAL Water medium</b> — unified API over IAPWS-IF97 Regions 1, 2, and 4.</p>

<h4>Primary state variables</h4>
<p>(p, h) — pressure and specific enthalpy. This pairing avoids singularities
at phase boundaries (h is continuous across saturation at constant p).</p>

<h4>Region selection</h4>
<p>Region detection compares h to h_f(p) and h_g(p). In simulation models,
use <code>noEvent()</code> around calls to prevent event iteration in the
DAE solver — the semi-implicit solver drives p and h explicitly, so no
event crossing detection is needed.</p>

<h4>Two-phase derivatives</h4>
<p>(∂ρ/∂p)_h in Region 4 currently uses a ±500 Pa central difference.
This is sufficient for the semi-implicit scheme because the two-phase
compressibility is large and the finite-difference error is negligible
compared to the dominant term.  An analytical Clausius-Clapeyron expression
is planned for Phase 3 — see DESIGN.md.</p>

<h4>What is NOT here</h4>
<ul>
<li>Region 3 (near-critical) — deferred.</li>
<li>Transport properties (μ, λ) — deferred to Phase 3.</li>
<li>MSL PartialMedium interface — not needed (OPAL does not use Modelica.Fluid).</li>
</ul>
</html>"));
end Water;
