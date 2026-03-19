within library.Pipes;
model Pipe1D
  "1D pipe with HEM (3-equation) — extends shared base class"
  extends PartialPipe1D;

  // ═══════════════════════════════════════════════════════════════════
  // HEM-specific parameters
  // ═══════════════════════════════════════════════════════════════════
  parameter Real h_init = 800e3 "Initial enthalpy [J/kg]";

  // Generic energy source term
  parameter Real S_energy[N] = zeros(N) "Energy source per cell [W]";

  // ═══════════════════════════════════════════════════════════════════
  // HEM state variables — single mixture enthalpy per cell
  // ═══════════════════════════════════════════════════════════════════
  Real h[N](each start = h_init, each fixed = true) "Cell enthalpy [J/kg]";

  // ═══════════════════════════════════════════════════════════════════
  // Property evaluation (per cell)
  // ═══════════════════════════════════════════════════════════════════
  Real rho[N] "Cell density [kg/m^3]";
  Real T_cell[N] "Cell temperature [K] (diagnostic)";

  // ═══════════════════════════════════════════════════════════════════
  // Donor-cell face enthalpies
  // ═══════════════════════════════════════════════════════════════════
  Real h_face[N + 1] "Face enthalpy for energy advection [J/kg]";

equation
  // ─────────────────────────────────────────────────────────────────
  // Abstract variable bindings for base class
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho_cell[i] = rho[i];
  end for;

  for i in 1:N + 1 loop
    Phi2[i] = 1.0;  // no two-phase friction multiplier for HEM
  end for;

  h_mix_outlet = h[N];
  rho_outlet = rho[N];

  // ─────────────────────────────────────────────────────────────────
  // Connector enthalpy outflow
  // ─────────────────────────────────────────────────────────────────
  port_a.h_outflow = h[1];
  port_b.h_outflow = h[N];

  // ─────────────────────────────────────────────────────────────────
  // Properties (via replaceable Medium)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho[i] = Medium.rho_ph(p[i], h[i]);
    drho_dp[i] = Medium.drho_dp_h(p[i], h[i]);
    drho_dh[i] = Medium.drho_dh_p(p[i], h[i]);
    T_cell[i] = Medium.T_ph(p[i], h[i]);
  end for;

  // ─────────────────────────────────────────────────────────────────
  // Donor-cell face enthalpies
  // ─────────────────────────────────────────────────────────────────
  h_face[1] = if mdot[1] >= 0 then inStream(port_a.h_outflow) else h[1];

  for i in 2:N loop
    h_face[i] = if mdot[i] >= 0 then h[i - 1] else h[i];
  end for;

  h_face[N + 1] = if mdot[N + 1] >= 0 then h[N] else inStream(port_b.h_outflow);

  // ─────────────────────────────────────────────────────────────────
  // MASS CONSERVATION (per cell)
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    V_cell * (drho_dp[i] * der(p[i]) + drho_dh[i] * der(h[i]))
      = mdot[i] - mdot[i + 1] + S_mass[i];
  end for;

  // ─────────────────────────────────────────────────────────────────
  // ENERGY (per cell) — enthalpy form with pressure work
  // ─────────────────────────────────────────────────────────────────
  for i in 1:N loop
    rho[i] * V_cell * der(h[i])
      = mdot[i] * (h_face[i] - h[i])
      - mdot[i + 1] * (h_face[i + 1] - h[i])
      + V_cell * der(p[i])
      + q_wall[i]
      + S_energy[i];
  end for;

  annotation(Documentation(info="<html>
<p><b>1D pipe with HEM (Homogeneous Equilibrium Model) — 3 equations per cell.</b></p>
<p>Extends PartialPipe1D for shared geometry, momentum, and face density.
Adds mixture enthalpy state, single energy equation, and donor-cell reconstruction.</p>
<p>Staggered mesh: pressure and enthalpy at cell centres, mass flow at faces.</p>
<p>Swap medium: <code>Pipe1D pipe(redeclare package Medium = library.Media.Water)</code></p>
<p>For multi-equation models (drift-flux, two-fluid), see Pipe1D_DriftFlux.</p>
</html>"));
end Pipe1D;
