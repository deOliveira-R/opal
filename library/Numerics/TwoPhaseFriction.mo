within library.Numerics;
package TwoPhaseFriction "Two-phase friction multiplier correlations"

  function martinelli_nelson
    "Martinelli-Nelson two-phase friction multiplier Phi^2"
    input Real alpha "Void fraction [-]";
    input Real rho_l "Liquid density [kg/m^3]";
    input Real rho_v "Vapour density [kg/m^3]";
    output Real Phi2 "Two-phase multiplier [-] (multiply single-phase dp)";
  protected
    Real rho_ratio = max(rho_l / max(rho_v, 0.1), 1.0);
  algorithm
    // Simplified Martinelli-Nelson: Phi^2 = 1 + C*x + x^2*(rho_l/rho_v - 1)
    // where x is approximated from alpha via HEM: x = alpha*rho_v / ((1-alpha)*rho_l + alpha*rho_v)
    // For the Chisholm-Laird form: Phi^2_lo = (1-alpha)^2 + 2*(1-alpha)*alpha*sqrt(rho_ratio) + alpha^2*rho_ratio
    // This reduces to 1 for single-phase liquid and rho_l/rho_v for single-phase vapour.
    Phi2 := (1 - alpha)^2 + 2 * (1 - alpha) * alpha * sqrt(rho_ratio) + alpha^2 * rho_ratio;
    annotation(Inline=true);
  end martinelli_nelson;

  annotation(Documentation(info="<html>
<p>Two-phase friction multiplier correlations.</p>
<p><b>martinelli_nelson</b>: Chisholm-Laird form of the Martinelli-Nelson
two-phase friction multiplier. Φ² = (1-α)² + 2(1-α)α√(ρ_l/ρ_v) + α²(ρ_l/ρ_v).</p>
<p>Multiply the single-phase Darcy friction pressure drop by Φ² to get
the two-phase pressure drop.</p>
</html>"));
end TwoPhaseFriction;
