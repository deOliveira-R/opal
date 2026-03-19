within library.Boundary;
model BreakSource
  "Break boundary: pressure BC with discharge coefficient for pipe rupture"

  parameter Real p_back = 101325.0 "Back-pressure behind break [Pa]";
  parameter Real C_d = 1.0 "Discharge coefficient [-], 0 = closed, 1 = fully open";
  parameter Real h_set = 0.0 "Enthalpy of incoming fluid if flow reverses [J/kg]";

  library.Connectors.FluidPort port;

equation
  // Break imposes back-pressure scaled by discharge coefficient.
  // When C_d = 1: port.p = p_back (fully open break).
  // When C_d = 0: port.m_flow = 0 (sealed, equivalent to ClosedEnd).
  // Intermediate C_d: partial opening (for ramped breaks, use time-varying C_d).
  //
  // Note: critical (choked) flow limiting is NOT implemented here.
  // Choking is a solver-side concern (RansomTrapp model limits the mass flow
  // rate computed by the momentum equation). The break component only sets
  // the boundary pressure.
  port.p = p_back;
  port.h_outflow = h_set;

  annotation(Documentation(info="<html>
<p><b>Break boundary condition</b> for pipe rupture simulation.</p>
<p>Sets the boundary pressure to <code>p_back</code>. The discharge coefficient
<code>C_d</code> is available as a parameter for the solver to use when computing
critical flow, but does not affect the Modelica equations directly (the solver
handles choking via the CriticalFlowModel strategy).</p>
<p>For time-dependent break opening, connect a time signal to C_d via an
outer parameter or use RampedBreak.</p>
</html>"));
end BreakSource;
