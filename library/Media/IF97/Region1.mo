within OPAL.library.Media.IF97;
package Region1 "IAPWS-IF97 Region 1 — compressed liquid (T ∈ [273,623] K, p ∈ [0.6,100] MPa)"

  // ---------------------------------------------------------------------------
  // IAPWS-IF97 Table 2 coefficients for g(π,τ) = Σ n_i (7.1−π)^I_i (τ−1.222)^J_i
  // Reference: IAPWS-IF97 §6, Table 2
  // ---------------------------------------------------------------------------
  constant Integer I[34] = {0,0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,4,4,4,5,8,8,21,23,29,30,31,32};
  constant Integer J[34] = {-2,-1,0,1,2,3,4,5,-9,-7,-1,0,1,3,-3,0,1,3,17,-4,0,6,-5,-2,10,-8,-11,-6,-29,-31,-38,-39,-40,-41};
  constant Real    n[34] = {
     1.4632971213167e-1,  -8.4548187169114e-1, -3.7563603672040e0,
     3.3855169168385e0,   -9.5791963387872e-1,  1.5772038513228e-1,
    -1.6616417199501e-2,   8.1214629983568e-4,  2.8319080123804e-4,
    -6.0706301565874e-4,  -1.8990068218419e-2, -3.2529748770505e-2,
    -2.1841717175414e-2,  -5.2838357969930e-5, -4.7184321073267e-4,
    -3.0001780793026e-4,   4.7661393906987e-5, -4.4141845330846e-6,
    -7.2694996297594e-16, -3.1679644845054e-5, -2.8270797985312e-6,
    -8.5205128120103e-10, -2.2425281908000e-6, -6.5171222895601e-7,
    -1.4341729937924e-13, -4.0516996860117e-7, -1.2734301741641e-9,
    -1.7424871230634e-10, -6.8762131295531e-19, 1.4478307828521e-20,
     2.6335781662795e-23, -1.1947622640071e-23, 1.8228094581404e-24,
    -9.3537087292458e-26};

  // ---------------------------------------------------------------------------
  // Reduced variables
  // ---------------------------------------------------------------------------
  function pi_R1 "Reduced pressure π = p / p*"
    input Real p "Pressure [Pa]";
    output Real pi;
  algorithm
    pi := p / Constants.p_star_R1;
  end pi_R1;

  function tau_R1 "Reduced temperature τ = T* / T"
    input Real T "Temperature [K]";
    output Real tau;
  algorithm
    tau := Constants.T_star_R1 / T;
  end tau_R1;

  // ---------------------------------------------------------------------------
  // Gibbs function and partial derivatives  (dimensionless, per IAPWS-IF97 §6)
  //
  //   g(π,τ)  = Σ n_i (7.1−π)^I_i (τ−1.222)^J_i
  //   g_pi    = ∂g/∂π
  //   g_pipi  = ∂²g/∂π²
  //   g_tau   = ∂g/∂τ
  //   g_tautau = ∂²g/∂τ²
  //   g_pitau  = ∂²g/∂π∂τ
  //
  // All sums run i = 1..34.
  // ---------------------------------------------------------------------------
  function g "Dimensionless Gibbs function g(π,τ)"
    input Real pi, tau;
    output Real gval;
  algorithm
    gval := sum(n[i] * (7.1 - pi)^I[i] * (tau - 1.222)^J[i] for i in 1:34);
  end g;

  function g_pi "∂g/∂π"
    input Real pi, tau;
    output Real v;
  algorithm
    // d/dπ [(7.1−π)^I] = −I (7.1−π)^(I−1)
    v := sum(-n[i] * I[i] * (7.1 - pi)^(I[i] - 1) * (tau - 1.222)^J[i] for i in 1:34);
  end g_pi;

  function g_pipi "∂²g/∂π²"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(n[i] * I[i] * (I[i] - 1) * (7.1 - pi)^(I[i] - 2) * (tau - 1.222)^J[i] for i in 1:34);
  end g_pipi;

  function g_tau "∂g/∂τ"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(n[i] * (7.1 - pi)^I[i] * J[i] * (tau - 1.222)^(J[i] - 1) for i in 1:34);
  end g_tau;

  function g_tautau "∂²g/∂τ²"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(n[i] * (7.1 - pi)^I[i] * J[i] * (J[i] - 1) * (tau - 1.222)^(J[i] - 2) for i in 1:34);
  end g_tautau;

  function g_pitau "∂²g/∂π∂τ"
    input Real pi, tau;
    output Real v;
  algorithm
    v := sum(-n[i] * I[i] * (7.1 - pi)^(I[i] - 1) * J[i] * (tau - 1.222)^(J[i] - 1) for i in 1:34);
  end g_pitau;

  // ---------------------------------------------------------------------------
  // Thermodynamic properties  (IAPWS-IF97 §6, Table 3)
  //
  //   v  = (R T / p) π g_pi                 [m³/kg]
  //   h  = R T τ g_tau                       [J/kg]
  //   s  = R (τ g_tau − g)                   [J/(kg·K)]
  //   cp = −R τ² g_tautau                    [J/(kg·K)]
  //   rho = 1/v                              [kg/m³]
  //
  // All properties are functions of (p [Pa], T [K]).
  // ---------------------------------------------------------------------------
  function v_pT "Specific volume [m³/kg]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real v_val;
  protected
    Real pi_v = pi_R1(p);
    Real tau_v = tau_R1(T);
  algorithm
    v_val := (Constants.R * T / p) * pi_v * g_pi(pi_v, tau_v);
  end v_pT;

  function h_pT "Specific enthalpy [J/kg]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real h_val;
  protected
    Real pi_v = pi_R1(p);
    Real tau_v = tau_R1(T);
  algorithm
    h_val := Constants.R * T * tau_v * g_tau(pi_v, tau_v);
  end h_pT;

  function s_pT "Specific entropy [J/(kg·K)]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real s_val;
  protected
    Real pi_v = pi_R1(p);
    Real tau_v = tau_R1(T);
  algorithm
    s_val := Constants.R * (tau_v * g_tau(pi_v, tau_v) - g(pi_v, tau_v));
  end s_pT;

  function cp_pT "Specific heat at constant pressure [J/(kg·K)]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real cp_val;
  protected
    Real pi_v = pi_R1(p);
    Real tau_v = tau_R1(T);
  algorithm
    cp_val := -Constants.R * tau_v^2 * g_tautau(pi_v, tau_v);
  end cp_pT;

  function rho_pT "Density [kg/m³]"
    input Real p "Pressure [Pa]";
    input Real T "Temperature [K]";
    output Real rho_val;
  algorithm
    rho_val := 1.0 / v_pT(p, T);
  end rho_pT;

  // ---------------------------------------------------------------------------
  // Inverse: T from (p, h)  — needed for region detection in Water.mo
  // Newton iteration: h_pT(p, T) = h_target
  // Starting guess from IAPWS-IF97 backward equation (Region 1, Table 20)
  // ---------------------------------------------------------------------------
  function T_ph "Temperature from (p, h) in Region 1 [K]"
    input Real p "Pressure [Pa]";
    input Real h "Specific enthalpy [J/kg]";
    output Real T_val;
  protected
    // Backward equation coefficients — IAPWS-IF97 Table 20
    // T(π,η) = Σ n_i π^I_i (η+1)^J_i,  π=p/1e6,  η=h/2500e3
    constant Integer I_bw[20] = {0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,3,3,4,5,6};
    constant Integer J_bw[20] = {0,1,2,6,22,32,0,1,2,3,4,10,32,10,32,10,32,32,32,32};
    constant Real    n_bw[20] = {
       -2.38724899245210e2,  4.04211886379497e2,  1.13497468817180e3,
       -5.84576160480760e0,  1.94833814891320e-2, -1.31810637641710e-2,
       -1.97526707585300e-3, -3.12654168748730e-1,  6.75947772051340e0,
        7.38692042996260e-1, -2.23400981718580e-2, -1.75854800633840e-2,
        1.40635383285710e-4,  1.26516012387810e-3, -9.76139529165740e-5,
        2.14668226424400e-3, -6.29079757662260e-5, -1.00791388328680e-3,
        2.02839590285650e-4, -1.07478030487020e-5};
    Real pi_bw = p / 1.0e6;
    Real eta_bw = h / 2500.0e3;
    Real T_guess;
    Real f, df, dT;
    Integer iter;
    Real T_iter;
  algorithm
    // Backward equation gives a good starting guess (within ~0.1 K)
    T_guess := sum(n_bw[i] * pi_bw^I_bw[i] * (eta_bw + 1.0)^J_bw[i] for i in 1:20);

    // Newton refinement (1–2 iterations sufficient)
    T_iter := T_guess;
    for iter in 1:5 loop
      f  := h_pT(p, T_iter) - h;
      df := -Constants.R * tau_R1(T_iter)^2 / T_iter * g_tautau(pi_R1(p), tau_R1(T_iter))
            + Constants.R * tau_R1(T_iter) * g_tau(pi_R1(p), tau_R1(T_iter)) / T_iter;
      // dh/dT|p = cp_pT(p, T_iter)
      df := cp_pT(p, T_iter);
      dT := -f / df;
      T_iter := T_iter + dT;
    end for;
    T_val := T_iter;
  end T_ph;

  annotation(Documentation(info="<html>
<p><b>IAPWS-IF97 Region 1 — compressed liquid</b></p>
<p>Valid range: T ∈ [273.15, 623.15] K, p ∈ [0.6, 100] MPa (also extends to lower T for subcooled liquid).</p>
<p>Gibbs function: g(π,τ) = Σ_{i=1}^{34} n_i (7.1−π)^{I_i} (τ−1.222)^{J_i}</p>
<p>where π = p/p*, τ = T*/T, p* = 16.53 MPa, T* = 1386 K.</p>
<p>Coefficients from IAPWS-IF97 Table 2.</p>
<p>Backward equation T(p,h) from IAPWS-IF97 Table 20, refined by Newton iteration.</p>
</html>"));
end Region1;
