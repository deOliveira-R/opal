within library.Numerics;
package Limiters "TVD slope limiters for MUSCL reconstruction"

  function minmod "Most diffusive TVD limiter: phi(r) = max(0, min(r, 1))"
    input Real r "Gradient ratio";
    output Real phi;
  algorithm
    phi := max(0, min(r, 1));
    annotation(Inline=true);
  end minmod;

  function vanLeer "Smooth TVD limiter: phi(r) = (r + |r|) / (1 + |r|)"
    input Real r "Gradient ratio";
    output Real phi;
  algorithm
    phi := (r + abs(r)) / (1 + abs(r));
    annotation(Inline=true);
  end vanLeer;

  function superbee "Least diffusive TVD limiter: phi(r) = max(0, min(2r,1), min(r,2))"
    input Real r "Gradient ratio";
    output Real phi;
  algorithm
    phi := max(0, max(min(2 * r, 1), min(r, 2)));
    annotation(Inline=true);
  end superbee;

  function mc "Monotonized central: phi(r) = max(0, min((1+r)/2, 2, 2r))"
    input Real r "Gradient ratio";
    output Real phi;
  algorithm
    phi := max(0, min(min((1 + r) / 2, 2), 2 * r));
    annotation(Inline=true);
  end mc;

  function muscl_face
    "MUSCL face value: h_upwind + 0.5 * phi(r) * delta, with minmod limiter"
    input Real h_LL "Two cells upstream";
    input Real h_L "One cell upstream (upwind)";
    input Real h_R "One cell downstream";
    input Real mdot "Mass flow at face (positive = L→R)";
    output Real h_face;
  protected
    Real delta, r, phi;
  algorithm
    if mdot >= 0 then
      delta := h_R - h_L;
      if abs(delta) < 1e-30 then
        h_face := h_L;
      else
        r := (h_L - h_LL) / delta;
        phi := max(0, min(r, 1));  // minmod
        h_face := h_L + 0.5 * phi * delta;
      end if;
    else
      delta := h_L - h_R;
      if abs(delta) < 1e-30 then
        h_face := h_R;
      else
        r := (h_R - h_L) / delta;  // Note: h_R is upstream for negative flow
        // For negative flow, "upstream" is the R side, so gradient ratio
        // uses (R - RR) / (L - R). Since we only have LL/L/R here,
        // we fall back to first-order for the negative-flow branch
        // (proper MUSCL for negative flow needs RR, not LL).
        h_face := h_R;
      end if;
    end if;
    annotation(Inline=true);
  end muscl_face;

  annotation(Documentation(info="<html>
<p>TVD slope limiters for second-order MUSCL reconstruction.</p>
<p>All limiters take gradient ratio r and return flux limiter phi in [0, 2].</p>
<ul>
<li><b>minmod</b>: most diffusive, φ = max(0, min(r, 1))</li>
<li><b>vanLeer</b>: smooth, φ = (r+|r|)/(1+|r|)</li>
<li><b>superbee</b>: least diffusive, sharpest fronts</li>
<li><b>mc</b>: monotonized central, balanced</li>
</ul>
</html>"));
end Limiters;
