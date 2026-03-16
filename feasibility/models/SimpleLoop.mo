// SimpleLoop.mo — Feasibility Test 1 model
//
// A minimal isothermal single-phase hydraulic loop:
//   Pump → HeatedPipe → HeatSink → (back to Pump)
//
// Physics:
//   - Pressure-driven flow (acausal Modelica connectors)
//   - One compressible volume (pipe) gives an ODE state for pressure
//   - Darcy-Weisbach friction in pipe and heat sink
//   - No external library dependencies (self-contained)
//
// Expected DAE structure:
//   - 1 differential state: pipe.p  (pressure ODE from compressibility)
//   - ~5 algebraic variables: port pressures and flows around the loop
//   - Index 1: no Pantelides step needed (compressibility breaks the algebraic loop)
//   - BLT: mostly scalar blocks, 1 ODE block
//   - No events / zero crossings

connector FluidPort
  Real p          "Pressure [Pa]";
  flow Real mdot  "Mass flow rate, positive into component [kg/s]";
end FluidPort;

// ---------------------------------------------------------------------------
model Pump
  "Ideal pressure-rise pump (drives flow around the loop)"
  FluidPort inlet, outlet;
  parameter Real dP = 1e5  "Pump pressure rise [Pa]";
equation
  outlet.p - inlet.p = dP;
end Pump;

// ---------------------------------------------------------------------------
model HeatedPipe
  "Slightly compressible pipe segment — provides the ODE state"
  FluidPort inlet, outlet;
  parameter Real R  = 5e4   "Darcy friction resistance [Pa/(kg/s)]";
  parameter Real C  = 1e-9  "Hydraulic compressibility [kg/Pa]";

  Real p(start = 2e5, fixed = true)  "Average pipe pressure [Pa]";
  Real mdot                           "Through-flow [kg/s]";
equation
  // Mass continuity: compressibility stores/releases mass
  C * der(p) = inlet.mdot + outlet.mdot;
  // Momentum: Darcy-Weisbach pressure drop
  inlet.p - outlet.p = R * mdot;
  // Directed-flow convention (sign imposed by component, not by connect)
  inlet.mdot  =  mdot;
  outlet.mdot = -mdot;
  // Average pressure ties the state to the connector pressures
  p = 0.5 * (inlet.p + outlet.p);
end HeatedPipe;

// ---------------------------------------------------------------------------
model HeatSink
  "Passive hydraulic resistance (acts as the heat-removal element)"
  FluidPort inlet, outlet;
  parameter Real R = 3e4  "Friction resistance [Pa/(kg/s)]";
  Real mdot               "Through-flow [kg/s]";
equation
  inlet.p  - outlet.p = R * mdot;
  inlet.mdot  =  mdot;
  outlet.mdot = -mdot;
end HeatSink;

// ---------------------------------------------------------------------------
model SimpleLoop
  "Closed single-phase loop: pump + heated pipe + heat sink"
  Pump       pump;
  HeatedPipe pipe;
  HeatSink   sink;
equation
  connect(pump.outlet, pipe.inlet);
  connect(pipe.outlet, sink.inlet);
  connect(sink.outlet, pump.inlet);
end SimpleLoop;
