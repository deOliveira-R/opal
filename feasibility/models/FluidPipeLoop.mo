// FluidPipeLoop.mo — Feasibility Test 3: Modelica.Fluid + IAPWS-IF97
//
// Tests whether OM's extraction pipeline handles the full MSL Fluid stack.
// Three sub-models of increasing complexity:
//
//   FluidPipeLoopStatic   — StaticPipe, SteadyState, StandardWaterOnePhase
//   FluidPipeLoopDynamic  — DynamicPipe, FixedInitial, StandardWaterOnePhase
//   FluidPipeLoopTwophase — StaticPipe, SteadyState, StandardWater (ph)
//
// Key question for FM3: are IAPWS-IF97 calls opaque (external "C") or
// inlined as algebraic equations?  Report the equation count explosion (if any).
//
// Requires: loadModel(Modelica) before loadFile(this file)

// ---------------------------------------------------------------------------
model FluidPipeLoopStatic
  "Simplest Modelica.Fluid test: static pipe, steady-state energy, liquid water"
  extends Modelica.Icons.Example;

  replaceable package Medium = Modelica.Media.Water.StandardWaterOnePhase
    constrainedby Modelica.Media.Interfaces.PartialMedium;

  inner Modelica.Fluid.System system(
    energyDynamics = Modelica.Fluid.Types.Dynamics.SteadyState,
    massDynamics   = Modelica.Fluid.Types.Dynamics.SteadyState)
    "Global fluid system object (required by Modelica.Fluid)";

  Modelica.Fluid.Sources.FixedBoundary inlet(
    nPorts = 1,
    p      = 15.5e6,
    T      = 290 + 273.15,
    redeclare package Medium = Medium)
    "PWR primary cold-leg inlet";

  Modelica.Fluid.Pipes.StaticPipe pipe(
    length   = 4.0,
    diameter = 0.025,
    redeclare package Medium = Medium)
    "Short pipe segment";

  Modelica.Fluid.Sources.FixedBoundary outlet(
    nPorts = 1,
    p      = 15.4e6,
    redeclare package Medium = Medium)
    "Fixed-pressure outlet";

equation
  connect(inlet.ports[1],  pipe.port_a);
  connect(pipe.port_b,    outlet.ports[1]);

end FluidPipeLoopStatic;

// ---------------------------------------------------------------------------
model FluidPipeLoopDynamic
  "Dynamic pipe with energy/mass storage — adds ODE states"
  extends Modelica.Icons.Example;

  replaceable package Medium = Modelica.Media.Water.StandardWaterOnePhase
    constrainedby Modelica.Media.Interfaces.PartialMedium;

  inner Modelica.Fluid.System system(
    energyDynamics = Modelica.Fluid.Types.Dynamics.FixedInitial,
    massDynamics   = Modelica.Fluid.Types.Dynamics.FixedInitial);

  Modelica.Fluid.Sources.FixedBoundary inlet(
    nPorts = 1,
    p      = 15.5e6,
    T      = 290 + 273.15,
    redeclare package Medium = Medium);

  Modelica.Fluid.Pipes.DynamicPipe pipe(
    length          = 4.0,
    diameter        = 0.025,
    nNodes          = 3,
    redeclare package Medium = Medium);

  Modelica.Fluid.Sources.FixedBoundary outlet(
    nPorts = 1,
    p      = 15.4e6,
    redeclare package Medium = Medium);

equation
  connect(inlet.ports[1],  pipe.port_a);
  connect(pipe.port_b,    outlet.ports[1]);

end FluidPipeLoopDynamic;

// ---------------------------------------------------------------------------
model FluidPipeLoopTwophase
  "Two-phase medium (StandardWater, p-h) — highest-risk test"
  extends Modelica.Icons.Example;

  replaceable package Medium = Modelica.Media.Water.StandardWater
    constrainedby Modelica.Media.Interfaces.PartialMedium;

  inner Modelica.Fluid.System system(
    energyDynamics = Modelica.Fluid.Types.Dynamics.SteadyState,
    massDynamics   = Modelica.Fluid.Types.Dynamics.SteadyState);

  Modelica.Fluid.Sources.FixedBoundary inlet(
    nPorts = 1,
    p      = 15.5e6,
    T      = 290 + 273.15,
    redeclare package Medium = Medium);

  Modelica.Fluid.Pipes.StaticPipe pipe(
    length   = 4.0,
    diameter = 0.025,
    redeclare package Medium = Medium);

  Modelica.Fluid.Sources.FixedBoundary outlet(
    nPorts = 1,
    p      = 15.4e6,
    redeclare package Medium = Medium);

equation
  connect(inlet.ports[1],  pipe.port_a);
  connect(pipe.port_b,    outlet.ports[1]);

end FluidPipeLoopTwophase;
