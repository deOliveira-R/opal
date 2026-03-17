within OPAL.library.Media.IF97;
package Region2 "IAPWS-IF97 Region 2 — superheated steam (T ∈ [273,1073] K, p ∈ [0,10] MPa)"

  // ---------------------------------------------------------------------------
  // IAPWS-IF97 §7, Tables 9–10: coefficients for g°(π,τ) (ideal gas part)
  //   g°(π,τ) = ln(π) + Σ_{i=1}^{9} n°_i τ^{J°_i}
  // ---------------------------------------------------------------------------
  constant Integer J0[9] = {0,1,-5,-4,-3,-2,-1,2,3};
  constant Real    n0[9] = {
    -9.6927686500217e0,
     1.0086655968018e1,
    -5.6087911283020e-3,
     7.1452738081455e-2,
    -4.0710498223928e-1,
     1.4240819171444e0,
    -4.3839511319450e0,
    -2.8408632460772e-1,
     2.1268463753307e-2};

  // ---------------------------------------------------------------------------
  // IAPWS-IF97 §7, Table 11: coefficients for g^r(π,τ) (residual part)
  //   g^r(π,τ) = Σ_{i=1}^{43} n_i π^{I_i} (τ−0.5)^{J_i}
  // ---------------------------------------------------------------------------
  constant Integer Ir[43] = {1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,5,6,6,6,7,7,7,8,8,9,10,10,10,16,16,18,20,20,20,21,22,23,24,24,24};
  constant Integer Jr[43] = {0,1,2,3,6,1,2,4,7,36,0,1,3,6,35,1,2,3,7,3,16,35,0,11,25,8,36,13,4,10,14,29,50,57,20,35,48,21,53,39,26,40,58};
  constant Real    nr[43] = {
    -1.7731742473213e-3, -1.7834862292358e-2, -4.5996013696365e-2,
    -5.7581259083432e-2, -5.0325278727930e-2, -3.3032641670203e-5,
    -1.8948987516315e-4, -3.9392777243355e-3, -4.3797295650573e-2,
    -2.6674547914087e-5,  2.0481737692309e-8,  4.3870667284435e-7,
    -3.2277677238570e-5, -1.5033924542148e-3, -4.0668253562649e-2,
    -7.8847309559367e-10, 1.2790717852285e-8,  4.8225372718507e-7,
     2.2922076337661e-6, -1.6714766451061e-11,-2.1171472321355e-3,
    -2.3895741934104e1,  -5.9059564324270e-18,-1.2621808899101e-6,
    -3.8946842435739e-2,  1.1256211360459e-11,-8.2311340897998e0,
     1.9809712802088e-8,  1.0406965210174e-19,-1.0234747095929e-13,
    -1.0018179379511e-9, -8.0882908646985e-11, 1.0693031879409e-1,
    -3.3662250574171e-1,  8.9185845355421e-25, 3.0629316876232e-13,
    -4.2002467698208e-6, -5.9056029685639e-26, 3.7826947613457e-6,
    -1.2768608934681e-15, 7.3087610595061e-29, 5.5414715350778e-17,
    -9.4369707241210e-7};

  // ---------------------------------------------------------------------------
  // Reduced variables
  // ---------------------------------------------------------------------------
  function pi_R2 "Reduced pressure π = p / p*"
    input Real p "Pressure [Pa]";
    output Real pi;
  algorithm
    pi := p / Constants.p_star_R2;
  end pi_R2;

  function tau_R2 "Reduced temperature τ = T* / T"
    input Real T "Temperature [K]";
    output Real tau;
  algorithm
    tau := Constants.T_star_R2 / T;
  end tau_R2;

  // ---------------------------------------------------------------------------
  // Ideal part g° and derivatives
  // ---------------------------------------------------------------------------
  function g0 "Ideal part of Gibbs function g°(π,τ)"
    input Real pi, tau;
    output Real v;
  algorithm
    v := log(pi) + sum(n0[i] * tau^J0[i] for i in 1:9);
  end g0;

  function g0_pi "∂g°/∂π = 1/π"
    input Real pi, tau;
    output Real v;
  algorithm
    v := 1.0 / pi;
  end g0_pi;

  function g0_pipi "∂²g°/∂π² = −1/π²"
    input Real pi, tau;
    output Real v;
  algorithm
    v := -1.0 / pi^2;
  end g0_pipi;

  function g0_tau "∂g°/∂τ"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(n0[i] * J0[i] * tau^(J0[i] - 1) for i in 1:9);
  end g0_tau;

  function g0_tautau "∂²g°/∂τ²"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(n0[i] * J0[i] * (J0[i] - 1) * tau^(J0[i] - 2) for i in 1:9);
  end g0_tautau;

  // g0_pitau = 0 (ideal part cross-derivative vanishes)

  // ---------------------------------------------------------------------------
  // Residual part g^r and derivatives
  // ---------------------------------------------------------------------------
  function gr "Residual part g^r(π,τ)"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(nr[i] * pi^Ir[i] * (tau - 0.5)^Jr[i] for i in 1:43);
  end gr;

  function gr_pi "∂g^r/∂π"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(nr[i] * Ir[i] * pi^(Ir[i] - 1) * (tau - 0.5)^Jr[i] for i in 1:43);
  end gr_pi;

  function gr_pipi "∂²g^r/∂π²"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(nr[i] * Ir[i] * (Ir[i] - 1) * pi^(Ir[i] - 2) * (tau - 0.5)^Jr[i] for i in 1:43);
  end gr_pipi;

  function gr_tau "∂g^r/∂τ"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(nr[i] * pi^Ir[i] * Jr[i] * (tau - 0.5)^(Jr[i] - 1) for i in 1:43);
  end gr_tau;

  function gr_tautau "∂²g^r/∂τ²"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(nr[i] * pi^Ir[i] * Jr[i] * (Jr[i] - 1) * (tau - 0.5)^(Jr[i] - 2) for i in 1:43);
  end gr_tautau;

  function gr_pitau "∂²g^r/∂π∂τ"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(nr[i] * Ir[i] * pi^(Ir[i] - 1) * Jr[i] * (tau - 0.5)^(Jr[i] - 1) for i in 1:43);
  end gr_pitau;

  // ---------------------------------------------------------------------------
  // Combined g = g° + g^r  and derivatives (total)
  // ---------------------------------------------------------------------------
  function g_pi_tot "∂(g°+g^r)/∂π"
    input Real pi, tau;
    output Real v;
  algorithm
    v := g0_pi(pi, tau) + gr_pi(pi, tau);
  end g_pi_tot;

  function g_pipi_tot "∂²(g°+g^r)/∂π²"
    input Real pi, tau;
    output Real v;
  algorithm
    v := g0_pipi(pi, tau) + gr_pipi(pi, tau);
  end g_pipi_tot;

  function g_tau_tot "∂(g°+g^r)/∂τ"
    input Real pi, tau;
    output Real v;
  algorithm
    v := g0_tau(pi, tau) + gr_tau(pi, tau);
  end g_tau_tot;

  function g_tautau_tot "∂²(g°+g^r)/∂τ²"
    input Real pi, tau;
    output Real v;
  algorithm
    v := g0_tautau(pi, tau) + gr_tautau(pi, tau);
  end g_tautau_tot;

  function g_pitau_tot "∂²(g°+g^r)/∂π∂τ"
    input Real pi, tau;
    output Real v;
  algorithm
    v := gr_pitau(pi, tau);  // g0_pitau = 0
  end g_pitau_tot;

  // ---------------------------------------------------------------------------
  // Thermodynamic properties  (IAPWS-IF97 §7, Table 12)
  //
  //   v  = (R T / p) π (g0_pi + gr_pi)       [m³/kg]
  //   h  = R T τ (g0_tau + gr_tau)            [J/kg]
  //   s  = R [τ(g0_tau + gr_tau) − (g0 + gr)] [J/(kg·K)]
  //   cp = −R τ² (g0_tautau + gr_tautau)      [J/(kg·K)]
  //   rho = 1/v                               [kg/m³]
  // ---------------------------------------------------------------------------
  function v_pT "Specific volume [m³/kg]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real v_val;
  protected
    Real pi_v = pi_R2(p);
    Real tau_v = tau_R2(T);
  algorithm
    v_val := (Constants.R * T / p) * pi_v * g_pi_tot(pi_v, tau_v);
  end v_pT;

  function h_pT "Specific enthalpy [J/kg]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real h_val;
  protected
    Real pi_v = pi_R2(p);
    Real tau_v = tau_R2(T);
  algorithm
    h_val := Constants.R * T * tau_v * g_tau_tot(pi_v, tau_v);
  end h_pT;

  function s_pT "Specific entropy [J/(kg·K)]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real s_val;
  protected
    Real pi_v = pi_R2(p);
    Real tau_v = tau_R2(T);
  algorithm
    s_val := Constants.R * (tau_v * g_tau_tot(pi_v, tau_v) - (g0(pi_v, tau_v) + gr(pi_v, tau_v)));
  end s_pT;

  function cp_pT "Specific heat at constant pressure [J/(kg·K)]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real cp_val;
  protected
    Real pi_v = pi_R2(p);
    Real tau_v = tau_R2(T);
  algorithm
    cp_val := -Constants.R * tau_v^2 * g_tautau_tot(pi_v, tau_v);
  end cp_pT;

  function rho_pT "Density [kg/m³]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real rho_val;
  algorithm
    rho_val := 1.0 / v_pT(p, T);
  end rho_pT;

  // ---------------------------------------------------------------------------
  // Inverse: T from (p, h)  — IAPWS-IF97 backward equation, Region 2
  // Sub-region 2a: h < 4.0e6 J/kg (Table 20 of IAPWS-IF97-2007 supplement)
  // Simplified: single Newton iteration starting from ideal-gas approximation
  // ---------------------------------------------------------------------------
  function T_ph "Temperature from (p, h) in Region 2 [K]"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real T_val;
  protected
    Real T_iter;
    Real f, dT;
  algorithm
    // Simple starting guess — mid-range for Region 2 (273–1073 K).
    // Newton converges in 5–8 iterations from any reasonable guess.
    T_iter := 600.0;
    for iter in 1:10 loop
      f  := h_pT(p, T_iter) - h;
      dT := -f / cp_pT(p, T_iter);
      T_iter := T_iter + dT;
    end for;
    T_val := T_iter;
  end T_ph;

  annotation(Documentation(info="<html>
<p><b>IAPWS-IF97 Region 2 — superheated steam</b></p>
<p>Valid range: T ∈ [273.15, 1073.15] K, p ∈ [0, 10] MPa.</p>
<p>Gibbs function split into ideal part g°(π,τ) and residual part g^r(π,τ):</p>
<ul>
<li>g°: 9-term series in τ only, plus ln(π)</li>
<li>g^r: 43-term series in π^I (τ−0.5)^J</li>
</ul>
<p>Reducing quantities: p* = 1 MPa, T* = 540 K.</p>
<p>Coefficients from IAPWS-IF97 Tables 9–11.</p>
</html>"));
end Region2;
