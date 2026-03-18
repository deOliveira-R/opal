within library.Boundary;
model PressureSource "Fixed pressure and enthalpy boundary"
  parameter Real p_set "Boundary pressure [Pa]";
  parameter Real h_set "Boundary enthalpy [J/kg]";

  library.Connectors.FluidPort port;

equation
  port.p = p_set;
  port.h_outflow = h_set;

  annotation(Documentation(info="<html>
<p>Imposes a fixed pressure and enthalpy at the boundary.
Flow direction is determined by the connected component.</p>
</html>"));
end PressureSource;
