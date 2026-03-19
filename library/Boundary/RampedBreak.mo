within library.Boundary;
model RampedBreak
  "Time-ramped break: C_d ramps from 0 to C_d_final over t_open seconds"

  parameter Real p_back = 101325.0 "Back-pressure behind break [Pa]";
  parameter Real C_d_final = 1.0 "Final discharge coefficient [-]";
  parameter Real t_open = 0.001 "Opening time [s]";
  parameter Real h_set = 0.0 "Enthalpy of incoming fluid if flow reverses [J/kg]";

  Real C_d "Current discharge coefficient [-]";

  library.Connectors.FluidPort port;

equation
  C_d = C_d_final * min(time / t_open, 1.0);

  port.p = p_back;
  port.h_outflow = h_set;

  annotation(Documentation(info="<html>
<p><b>Time-ramped break boundary.</b></p>
<p>Discharge coefficient ramps linearly from 0 to <code>C_d_final</code> over
<code>t_open</code> seconds. After <code>t_open</code>, C_d stays at C_d_final.</p>
<p>The C_d value is available for the solver's critical flow model. The boundary
pressure is always <code>p_back</code> — choking is handled by the solver.</p>
</html>"));
end RampedBreak;
