// StreamLoop.mo — Feasibility Test 2: stream connector semantics
//
// A minimal three-component chain using stream connectors.
// Mirrors the structure of Modelica.Fluid.FluidPort without requiring MSL.
//
// Purpose: determine how OM expands inStream() calls in the extraction output.
// Key questions:
//   1. At "flat" level: is inStream() still a named call or already expanded?
//   2. At "backEnd" level: is h_outflow in aliasVariables (eliminated)?
//   3. How many equations does the mixing rule expansion add?

connector StreamPort
  Real       p          "Pressure [Pa]";
  flow Real  m_flow     "Mass flow rate, positive into component [kg/s]";
  stream Real h_outflow "Specific enthalpy leaving the component [J/kg]";
end StreamPort;

// ---------------------------------------------------------------------------
model StreamSource
  "Fixed boundary: imposes pressure and outflow enthalpy"
  StreamPort outlet;
  parameter Real p0 = 2e5    "Pressure [Pa]";
  parameter Real h0 = 1.2e5  "Outflow specific enthalpy [J/kg]";
equation
  outlet.p         = p0;
  outlet.h_outflow = h0;
end StreamSource;

// ---------------------------------------------------------------------------
model StreamPipe
  "Purely algebraic hydraulic resistance with enthalpy pass-through"
  StreamPort inlet, outlet;
  parameter Real R = 1e4  "Friction resistance [Pa/(kg/s)]";
  Real m_flow              "Through-flow [kg/s]";
  Real h_in                "Enthalpy entering pipe [J/kg]";
equation
  // Hydraulic (purely algebraic — no compressibility for this stream test)
  inlet.p - outlet.p = R * m_flow;
  inlet.m_flow  =  m_flow;
  outlet.m_flow = -m_flow;
  // Enthalpy: upstream value enters, same exits (no heat transfer)
  h_in = inStream(inlet.h_outflow);
  outlet.h_outflow = h_in;
  // Reverse-flow enthalpy (required by stream connector semantics)
  inlet.h_outflow = inStream(outlet.h_outflow);
end StreamPipe;

// ---------------------------------------------------------------------------
model StreamSink
  "Fixed-pressure sink: absorbs flow and measures arriving enthalpy"
  StreamPort inlet;
  Real h_received  "Enthalpy arriving at sink [J/kg]";
equation
  inlet.p       = 1e5;
  h_received    = inStream(inlet.h_outflow);
  inlet.h_outflow = h_received;  // dummy (no reverse flow expected)
end StreamSink;

// ---------------------------------------------------------------------------
model StreamLoop
  "Source → pipe → sink with stream connectors (no loop)"
  StreamSource src;
  StreamPipe   pipe;
  StreamSink   snk;
equation
  connect(src.outlet, pipe.inlet);
  connect(pipe.outlet, snk.inlet);
end StreamLoop;
