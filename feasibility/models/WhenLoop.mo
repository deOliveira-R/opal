// WhenLoop.mo — Feasibility Test 2: event / when-equation model
//
// Extends SimpleLoop with a discrete pressure-triggered valve.
// Purpose: verify that when-equations and zero-crossings survive
// into the dumpXMLDAE output at all translation levels.
//
// Expected extraction results:
//   - valveOpen: variability="discrete" in orderedVariables
//   - n_zero_crossings >= 1 (threshold crossing)
//   - <whenEquation> tag present (or equivalent) in equations block
//   - mdotThrough: present as continuous algebraic

// Re-declare the connectors and sub-models here so the file is self-contained.

connector FluidPort
  Real p          "Pressure [Pa]";
  flow Real mdot  "Mass flow rate [kg/s]";
end FluidPort;

model Pump
  FluidPort inlet, outlet;
  parameter Real dP = 1e5;
equation
  outlet.p - inlet.p = dP;
end Pump;

model HeatedPipe
  FluidPort inlet, outlet;
  parameter Real R = 5e4;
  parameter Real C = 1e-9;
  Real p(start = 2e5, fixed = true);
  Real mdot;
equation
  C * der(p) = inlet.mdot + outlet.mdot;
  inlet.p - outlet.p = R * mdot;
  inlet.mdot  =  mdot;
  outlet.mdot = -mdot;
  p = 0.5 * (inlet.p + outlet.p);
end HeatedPipe;

model HeatSink
  FluidPort inlet, outlet;
  parameter Real R = 3e4;
  Real mdot;
equation
  inlet.p  - outlet.p = R * mdot;
  inlet.mdot  =  mdot;
  outlet.mdot = -mdot;
end HeatSink;

// ---------------------------------------------------------------------------
model WhenLoop
  "SimpleLoop with a discrete valve triggered by a pressure threshold"

  Pump       pump;
  HeatedPipe pipe;
  HeatSink   sink;

  // Discrete state: valve position
  discrete Boolean valveOpen(start = true)
    "True when valve is open (pressure below upper threshold)";

  // Continuous output: effective flow through valve
  Real mdotThrough  "Effective flow [kg/s], zero when valve closed";

  parameter Real p_open  = 1.5e5  "Re-open pressure threshold [Pa]";
  parameter Real p_close = 2.5e5  "Close pressure threshold [Pa]";

equation
  connect(pump.outlet, pipe.inlet);
  connect(pipe.outlet, sink.inlet);
  connect(sink.outlet, pump.inlet);

  // Discrete valve logic: two zero-crossings (pipe.p - p_close) and (p_open - pipe.p)
  when pipe.p > p_close then
    valveOpen = false;
  elsewhen pipe.p < p_open then
    valveOpen = true;
  end when;

  // Continuous algebraic equation gated on valve state
  mdotThrough = if valveOpen then pipe.mdot else 0.0;

end WhenLoop;
