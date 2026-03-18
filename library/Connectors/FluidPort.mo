within library.Connectors;
connector FluidPort "Thermalhydraulic fluid port"
  Real p "Pressure [Pa]";
  flow Real m_flow "Mass flow rate [kg/s], positive into component";
  stream Real h_outflow "Specific enthalpy of outgoing fluid [J/kg]";

  annotation(Documentation(info="<html>
<p>Standard OPAL fluid connector with stream enthalpy.</p>
<p>The <code>stream</code> attribute on <code>h_outflow</code> lets OpenModelica
automatically resolve mixing and flow direction via <code>inStream()</code>.</p>
</html>"));
end FluidPort;
