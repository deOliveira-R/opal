within library.Boundary;
model ClosedEnd "Wall boundary — zero mass flow"
  library.Connectors.FluidPort port;

equation
  port.m_flow = 0;
  port.h_outflow = 0;  // unused (no outflow), but must be defined for stream balance

  annotation(Documentation(info="<html>
<p>Imposes zero mass flow (wall/closed end). Pressure is free —
determined by the connected component.</p>
</html>"));
end ClosedEnd;
