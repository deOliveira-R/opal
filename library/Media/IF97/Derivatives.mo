within library.Media.IF97;
package Derivatives
  "Analytical thermodynamic derivatives needed by the semi-implicit pressure solver"

  // ---------------------------------------------------------------------------
  // DERIVATION OF  (∂ρ/∂p)_h  AND  (∂ρ/∂h)_p
  //
  // The Phase 2 semi-implicit pressure scheme linearises the mass ODE
  //
  //   d(ρ V)/dt = ṁ_in − ṁ_out
  //
  // expanding dρ/dt using the chain rule on ρ(p, h):
  //
  //   dρ/dt = (∂ρ/∂p)_h · dp/dt + (∂ρ/∂h)_p · dh/dt
  //
  // Both partials must be explicit algebraic functions of (p, h) — no Newton
  // iteration, no opaque calls — so OpenModelica can inline them into the
  // extracted equation system.
  //
  // STEP 1: Express (∂ρ/∂p)_T and (∂ρ/∂T)_p in terms of Gibbs derivatives.
  //
  //   v(p,T) = (R T / p) π g_π       [m³/kg],  π = p/p*
  //   ρ = 1/v
  //
  //   (∂v/∂p)_T = (R T / p²)(π g_π + π² g_ππ - π g_π)
  //             = (R T π² / p²) g_ππ
  //   (∂v/∂T)_p = (R / p) π g_π + (R T / p) π g_πτ · (dτ/dT)
  //             = (R / p) π g_π  −  (R τ / p) π g_πτ
  //             = (R π / p)(g_π − τ g_πτ)
  //
  //   (∂ρ/∂p)_T = −ρ² (∂v/∂p)_T = −(R T π² / p²) g_ππ / v²
  //   (∂ρ/∂T)_p = −ρ² (∂v/∂T)_p = −(R π / p)(g_π − τ g_πτ) / v²
  //
  // STEP 2: Express h partial derivatives.
  //
  //   h(p,T) = R T τ g_τ
  //   (∂h/∂T)_p = cp = −R τ² g_ττ
  //   (∂h/∂p)_T = R T (g_π − τ g_πτ) · (1/p*)   [from h = R T τ g_τ, chain rule]
  //             = v − T (∂v/∂T)_p                 (Maxwell relation, as a check)
  //             = (R T / p) π g_π − T · (R π / p)(g_π − τ g_πτ)
  //             = (R T π / p) τ g_πτ  ... simplifies to:
  //             = (R T / p*)(g_π − τ g_πτ)  [note p* appears because π = p/p*]
  //
  //   More carefully:
  //   h = R T τ g_τ(π, τ)
  //   ∂h/∂p|_T = R T τ · ∂g_τ/∂p|_T
  //            = R T τ · g_τπ · ∂π/∂p
  //            = R T τ g_πτ / p*
  //
  //   And from v:
  //   (∂v/∂T)_p = (R/p) π g_π + (R T/p) π · g_πτ · (dτ/dT)
  //             = (R π/p)(g_π − τ g_πτ)
  //
  //   So:  h_p = R T τ g_πτ / p*  =  (R T / p) π τ g_πτ   (since π/p* = 1/p)
  //
  // STEP 3: Change of variables from (p, T) to (p, h).
  //
  //   At constant p:   dh = cp dT   →   (∂T/∂h)_p = 1/cp
  //
  //   (∂ρ/∂h)_p = (∂ρ/∂T)_p · (∂T/∂h)_p = (∂ρ/∂T)_p / cp
  //
  //   At constant h:   0 = (∂h/∂p)_T dp + cp dT
  //                    →  (∂T/∂p)_h = −(∂h/∂p)_T / cp
  //
  //   (∂ρ/∂p)_h = (∂ρ/∂p)_T + (∂ρ/∂T)_p · (∂T/∂p)_h
  //             = (∂ρ/∂p)_T  −  (∂ρ/∂T)_p · (∂h/∂p)_T / cp
  //
  // STEP 4: Final working expressions (used in functions below).
  //
  //   Let:   v      = (R T π / p) g_π  =  R T g_π / p*
  //          rho    = 1/v
  //          cp     = −R τ² g_ττ
  //          h_p    = (R T π / p) τ g_πτ
  //          dv_dp  = (R T π² / p²) g_ππ  =  R T g_ππ / p*²
  //          dv_dT  = (R π / p)(g_π − τ g_πτ)
  //
  //          drho_dT_p = −rho² · dv_dT
  //          drho_dp_T = −rho² · dv_dp
  //
  //          drho_dh_p = drho_dT_p / cp
  //          drho_dp_h = drho_dp_T − drho_dT_p · h_p / cp
  //
  // These expressions are purely algebraic in (p, T) — no iteration.
  // For Water.mo, T is obtained from the (p, h) → T backward equation before
  // calling these functions.
  // ---------------------------------------------------------------------------

  function drho_dp_h_R1
    "∂ρ/∂p|h for Region 1 (compressed liquid) [kg/(m³·Pa)]"
    input Real p   "Pressure [Pa]";
    input Real T   "Temperature [K]  (from Region1.T_ph)";
    output Real val;
  protected
    Real pi_v    = Region1.pi_R1(p);
    Real tau_v   = Region1.tau_R1(T);
    Real gpi     = Region1.g_pi(pi_v, tau_v);
    Real gpipi   = Region1.g_pipi(pi_v, tau_v);
    Real gpitau  = Region1.g_pitau(pi_v, tau_v);
    Real gtautau = Region1.g_tautau(pi_v, tau_v);
    Real v_val   = (Constants.R * T / p) * pi_v * gpi;
    Real rho_val = 1.0 / v_val;
    Real cp_val  = -Constants.R * tau_v^2 * gtautau;
    // dv/dp|T = (R T π² / p²) g_ππ = R T g_ππ / p*²
    Real dv_dp = (Constants.R * T / (p * p)) * pi_v^2 * gpipi;
    // dv/dT|p = (R pi / p) * (g_π − τ g_πτ)
    Real dv_dT = (Constants.R * pi_v / p) * (gpi - tau_v * gpitau);
    Real drho_dT_p = -rho_val^2 * dv_dT;
    Real drho_dp_T = -rho_val^2 * dv_dp;
    // h_p = ∂h/∂p|T = (R T pi / p) * tau * g_πτ
    Real h_p_val = (Constants.R * T * pi_v / p) * tau_v * gpitau;
  algorithm
    val := drho_dp_T - drho_dT_p * h_p_val / cp_val;
  end drho_dp_h_R1;

  function drho_dh_p_R1
    "∂ρ/∂h|p for Region 1 (compressed liquid) [kg·s²/m⁵  =  kg/(m³·J/kg)]"
    input Real p   "Pressure [Pa]";
    input Real T   "Temperature [K]  (from Region1.T_ph)";
    output Real val;
  protected
    Real pi_v    = Region1.pi_R1(p);
    Real tau_v   = Region1.tau_R1(T);
    Real gpi     = Region1.g_pi(pi_v, tau_v);
    Real gpitau  = Region1.g_pitau(pi_v, tau_v);
    Real gtautau = Region1.g_tautau(pi_v, tau_v);
    Real v_val   = (Constants.R * T / p) * pi_v * gpi;
    Real rho_val = 1.0 / v_val;
    Real cp_val  = -Constants.R * tau_v^2 * gtautau;
    Real dv_dT   = (Constants.R * pi_v / p) * (gpi - tau_v * gpitau);
    Real drho_dT_p = -rho_val^2 * dv_dT;
  algorithm
    val := drho_dT_p / cp_val;
  end drho_dh_p_R1;

  function drho_dp_h_R2
    "∂ρ/∂p|h for Region 2 (superheated steam) [kg/(m³·Pa)]"
    input Real p   "Pressure [Pa]";
    input Real T   "Temperature [K]  (from Region2.T_ph)";
    output Real val;
  protected
    Real pi_v    = Region2.pi_R2(p);
    Real tau_v   = Region2.tau_R2(T);
    Real gpi     = Region2.g_pi_tot(pi_v, tau_v);
    Real gpipi   = Region2.g_pipi_tot(pi_v, tau_v);
    Real gpitau  = Region2.g_pitau_tot(pi_v, tau_v);
    Real gtautau = Region2.g_tautau_tot(pi_v, tau_v);
    Real v_val   = (Constants.R * T / p) * pi_v * gpi;
    Real rho_val = 1.0 / v_val;
    Real cp_val  = -Constants.R * tau_v^2 * gtautau;
    // dv/dp|T = (R T π² / p²) g_ππ = R T g_ππ / p*²
    Real dv_dp   = (Constants.R * T / (p * p)) * pi_v^2 * gpipi;
    Real dv_dT   = (Constants.R * pi_v / p) * (gpi - tau_v * gpitau);
    Real drho_dT_p = -rho_val^2 * dv_dT;
    Real drho_dp_T = -rho_val^2 * dv_dp;
    Real h_p_val = (Constants.R * T * pi_v / p) * tau_v * gpitau;
  algorithm
    val := drho_dp_T - drho_dT_p * h_p_val / cp_val;
  end drho_dp_h_R2;

  function drho_dh_p_R2
    "∂ρ/∂h|p for Region 2 (superheated steam) [kg/(m³·J/kg)]"
    input Real p   "Pressure [Pa]";
    input Real T   "Temperature [K]  (from Region2.T_ph)";
    output Real val;
  protected
    Real pi_v    = Region2.pi_R2(p);
    Real tau_v   = Region2.tau_R2(T);
    Real gpi     = Region2.g_pi_tot(pi_v, tau_v);
    Real gpitau  = Region2.g_pitau_tot(pi_v, tau_v);
    Real gtautau = Region2.g_tautau_tot(pi_v, tau_v);
    Real v_val   = (Constants.R * T / p) * pi_v * gpi;
    Real rho_val = 1.0 / v_val;
    Real cp_val  = -Constants.R * tau_v^2 * gtautau;
    Real dv_dT   = (Constants.R * pi_v / p) * (gpi - tau_v * gpitau);
    Real drho_dT_p = -rho_val^2 * dv_dT;
  algorithm
    val := drho_dT_p / cp_val;
  end drho_dh_p_R2;

  annotation(Documentation(info="<html>
<p><b>Analytical thermodynamic derivatives for the Phase 2 semi-implicit solver</b></p>
<h4>Purpose</h4>
<p>The semi-implicit pressure equation requires (∂ρ/∂p)_h and (∂ρ/∂h)_p to linearise
dρ/dt = (∂ρ/∂p)_h · dp/dt + (∂ρ/∂h)_p · dh/dt.</p>
<h4>Derivation</h4>
<p>Starting from the Gibbs-function representation ρ(p,T) = p / (R T π g_π), the
change-of-basis from (p,T) to (p,h) is performed using:</p>
<ul>
<li>(∂ρ/∂h)_p = (∂ρ/∂T)_p / c_p</li>
<li>(∂ρ/∂p)_h = (∂ρ/∂p)_T − (∂ρ/∂T)_p · (∂h/∂p)_T / c_p</li>
</ul>
<p>All intermediate quantities are explicit polynomial expressions in (π, τ) —
no Newton iteration.  The caller supplies T obtained from the region-specific
T_ph backward equation (also polynomial + Newton polish).</p>
<h4>Full derivation</h4>
<p>See the source code comments in this file for the step-by-step derivation.</p>
</html>"));
end Derivatives;
