within library.Numerics;
package InterfacialDrag "Interfacial drag closures for two-fluid models"

  function ishii_drag
    "Ishii bubbly-flow interfacial drag force per unit volume [N/m^3].
     Uses Schiller-Naumann drag coefficient.
     Positive F_drag when v_v > v_l (pushes liquid +x, vapor -x).
     Ref: Ishii & Hibiki, Ch. 9, Eq. 9.85; Schiller & Naumann (1933).
     Derived in: derivations/interfacial_drag.py"
    input Real alpha "Void fraction [-]";
    input Real rho_l "Liquid density [kg/m^3]";
    input Real v_l "Liquid velocity [m/s]";
    input Real v_v "Vapour velocity [m/s]";
    input Real d_b "Bubble diameter [m]";
    input Real mu_l "Liquid dynamic viscosity [Pa.s]";
    output Real F_drag "Interfacial drag force per unit volume [N/m^3]";
  protected
    Real eps_v = 1e-10 "Minimum relative velocity for Re computation";
    Real eps_a = 1e-6 "Phase-absence guard";
    Real v_rel = v_v - v_l "Relative velocity [m/s]";
    Real v_rel_abs = max(abs(v_rel), eps_v) "Regularized |v_rel|";
    Real alpha_eff = max(alpha, eps_a) "Guarded void fraction";
    Real Re_b "Bubble Reynolds number [-]";
    Real C_D "Drag coefficient [-]";
  algorithm
    // Bubble Reynolds number
    Re_b := rho_l * v_rel_abs * d_b / mu_l;

    // Schiller-Naumann drag coefficient, capped at Newton regime (0.44)
    // C_D = min((24/Re) * (1 + 0.1 * Re^0.75), 0.44)
    C_D := min((24.0 / Re_b) * (1.0 + 0.1 * Re_b ^ 0.75), 0.44);

    // Ishii bubbly drag: F = (3/4) * (C_D/d_b) * alpha * rho_l * |v_rel| * v_rel
    // Simplified: F = 18 * mu_l * alpha / d_b^2 * (1 + 0.1*Re^0.75) * v_rel
    F_drag := 0.75 * C_D / d_b * alpha_eff * rho_l * v_rel_abs * v_rel;

    annotation(Inline=true);
  end ishii_drag;

  function regime_map_drag
    "Flow regime-dependent interfacial drag: bubbly -> slug -> annular.
     Smooth linear blending between regimes (event-free).
     Same sign convention as ishii_drag: positive when v_v > v_l.
     Ref: Ishii & Hibiki Ch. 9 (bubbly), Ishii & Mishima 1984 (slug),
          Wallis 1969 p. 320 (annular).
     Derived in: derivations/drag_regime_map.py"
    input Real alpha "Void fraction [-]";
    input Real rho_l "Liquid density [kg/m^3]";
    input Real rho_v "Vapour density [kg/m^3]";
    input Real v_l "Liquid velocity [m/s]";
    input Real v_v "Vapour velocity [m/s]";
    input Real d_b "Bubble diameter [m]";
    input Real mu_l "Liquid dynamic viscosity [Pa.s]";
    input Real D "Pipe hydraulic diameter [m]";
    output Real F_drag "Interfacial drag force per unit volume [N/m^3]";
  protected
    Real eps_v = 1e-10 "Minimum relative velocity for Re computation";
    Real eps_a = 1e-6 "Phase-absence guard";
    Real v_rel = v_v - v_l "Relative velocity [m/s]";
    Real v_rel_abs = max(abs(v_rel), eps_v) "Regularized |v_rel|";
    Real alpha_eff = max(alpha, eps_a) "Guarded void fraction";
    Real alpha_l_eff = max(1 - alpha, eps_a) "Guarded liquid fraction";

    // Transition band boundaries (widened per physics review)
    Real alpha_bs_lo = 0.20 "Bubbly-to-slug lower";
    Real alpha_bs_hi = 0.40 "Bubbly-to-slug upper";
    Real alpha_sa_lo = 0.50 "Slug-to-annular lower";
    Real alpha_sa_hi = 0.80 "Slug-to-annular upper";

    // Blending fractions (clamped to [0,1], event-free)
    Real blend_bs "Bubbly-to-slug blend [-]";
    Real blend_sa "Slug-to-annular blend [-]";

    // Bubbly regime (Ishii-Zuber + Schiller-Naumann)
    Real Re_b "Bubble Reynolds number [-]";
    Real C_D_bubbly "Schiller-Naumann drag coefficient [-]";
    Real F_bubbly "Bubbly drag [N/m^3]";

    // Slug/cap regime (Ishii-Mishima distorted bubble)
    Real C_D_cap "Cap bubble drag coefficient [-]";
    Real d_cap "Cap bubble diameter [m]";
    Real F_slug "Slug drag [N/m^3]";

    // Annular regime (Wallis interfacial friction)
    Real f_i "Interfacial friction factor [-]";
    Real a_i_ann "Annular interfacial area concentration [1/m]";
    Real F_annular "Annular drag [N/m^3]";
  algorithm
    // Blending fractions
    blend_bs := noEvent(min(max((alpha - alpha_bs_lo) / (alpha_bs_hi - alpha_bs_lo), 0.0), 1.0));
    blend_sa := noEvent(min(max((alpha - alpha_sa_lo) / (alpha_sa_hi - alpha_sa_lo), 0.0), 1.0));

    // Bubbly: (3/4)*(C_D/d_b)*alpha*rho_l*|v_rel|*v_rel
    Re_b := rho_l * v_rel_abs * d_b / mu_l;
    C_D_bubbly := min((24.0 / Re_b) * (1.0 + 0.1 * Re_b ^ 0.75), 0.44);
    F_bubbly := 0.75 * C_D_bubbly / d_b * alpha_eff * rho_l * v_rel_abs * v_rel;

    // Slug/cap: C_D = (8/3)*(1-alpha)^2, d_cap = 4*d_b
    C_D_cap := (8.0 / 3.0) * alpha_l_eff ^ 2;
    d_cap := 4.0 * d_b;
    F_slug := 0.75 * C_D_cap / d_cap * alpha_eff * rho_l * v_rel_abs * v_rel;

    // Annular: (1/2)*f_i*rho_v*|v_rel|*v_rel*a_i
    f_i := 0.005 * (1.0 + 75.0 * max(1.0 - alpha, 0.0));
    a_i_ann := 4.0 * sqrt(alpha_eff) / D;
    F_annular := 0.5 * f_i * rho_v * v_rel_abs * v_rel * a_i_ann;

    // Blend: bubbly -> slug -> annular
    F_drag := (1.0 - blend_bs) * F_bubbly
            + blend_bs * ((1.0 - blend_sa) * F_slug + blend_sa * F_annular);

    annotation(Inline=true);
  end regime_map_drag;

  annotation(Documentation(info="<html>
<p>Interfacial drag closures for two-fluid (6-equation) models.</p>
<p><b>ishii_drag</b>: Ishii bubbly-flow interfacial drag force using
Schiller-Naumann drag coefficient. Valid for bubbly flow (alpha &lt; 0.3).</p>
<p><b>regime_map_drag</b>: Flow regime-dependent drag with smooth blending:
bubbly (Ishii-Zuber) &rarr; slug/cap (Ishii-Mishima) &rarr; annular (Wallis).
Transitions at alpha = [0.25-0.35] and [0.60-0.70].</p>
<p>Sign convention: positive when v_v &gt; v_l (drag accelerates liquid toward
vapor velocity). In momentum equations: liquid gets +F_drag, vapor gets -F_drag.</p>
<p>References: Ishii &amp; Hibiki (2006) Ch. 9, Ishii &amp; Mishima (1984),
Wallis (1969).</p>
</html>"));
end InterfacialDrag;
