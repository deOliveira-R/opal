within library;
package Media "OPAL thermodynamic media — pure Modelica, no external C"
  annotation(Documentation(info="<html>
<p>Two fluid packages for the OPAL solver:</p>
<ul>
<li><b>Water</b> — IAPWS-IF97 steam tables (production). 34+43-term Gibbs polynomials.</li>
<li><b>SimpleFluid</b> — Synthetic linear fluid (verification). Hand-verifiable, constant derivatives.</li>
</ul>
<p>Both share the same API: <code>rho_ph</code>, <code>T_ph</code>, <code>drho_dp_h</code>, <code>drho_dh_p</code>.
SimpleFluid isolates solver bugs from property bugs. See DESIGN.md.</p>
<p>No external C calls — all functions flatten to algebraic equations in extracted XML.</p>
</html>"));
end Media;
