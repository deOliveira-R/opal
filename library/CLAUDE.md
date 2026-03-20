# OPAL Modelica Component Library

## THIS IS THE PHYSICS SOURCE

All physics (conservation equations, closures, property evaluation, boundary conditions)
lives in these .mo files. The solver provides only numerical methods.
See `docs/architecture.md` Cardinal Rule.

## Design Principles

1. **Event-free for real-time.** Use `smooth()`, `noEvent()`, regularized transitions.
2. **No Modelica.Fluid / Modelica.Media at runtime.** Own media (PartialMedium interface).
3. **Every component carries three layers:** verification tests, validation cases, physics documentation.
4. **Same model, two modes.** Analysis mode (adaptive) and real-time mode (fixed-step).
5. **Replaceable media.** All pipes use `replaceable package Medium constrainedby PartialMedium`.

## Subdirectories

### Implemented
- `Pipes/` — 1D pipe components
  - `PartialPipe1D.mo` — Base class: geometry, connectors, face densities, momentum (with Phi2), critical flow
  - `Pipe1D.mo` — 3-equation HEM (extends PartialPipe1D + mixture energy)
  - `Pipe1D_DriftFlux.mo` — 5-equation drift-flux (extends PartialPipe1D + phasic energy, closures, drift-flux split)
  - `PartialPipe1D` features: selectable critical flow (Ransom-Trapp/Henry-Fauske), time-varying C_d_eff, gravity
  - Concrete models provide: `rho_cell`, `Phi2`, `h_mix_outlet`, `rho_outlet`, properties, mass conservation, energy
- `Media/` — Thermodynamic property packages
  - `PartialMedium.mo` — Abstract interface (14 functions: rho_ph, drho_dp_h, drho_dh_p, T_ph, T_sat, h_f, h_g, h_fg, rho_f, rho_g, cp_f, k_f, mu_f, sigma)
  - `SimpleFluid.mo` — Linear verification fluid (hand-verifiable)
  - `Water.mo` — IAPWS-IF97 Regions 1, 2, 4 (pure Modelica)
  - `IF97/` — Gibbs functions, saturation, derivatives (Constants, Region1, Region2, Saturation, Derivatives)
  - `tests/` — 300+ verification points against iapws oracle
- `Boundary/` — Boundary condition components
  - `ClosedEnd.mo` — Wall BC (zero mass flow)
  - `PressureSource.mo` — Fixed pressure + enthalpy BC
  - `BreakSource.mo` — Break BC with discharge coefficient
  - `RampedBreak.mo` — Time-ramped break opening
- `Connectors/` — Interface definitions
  - `FluidPort.mo` — Stream connector (p, m_flow, h_outflow)
- `Numerics/` — Numerical methods in Modelica
  - `Limiters.mo` — TVD slope limiters (minmod, vanLeer, superbee, mc) + muscl_face
  - `CriticalFlow.mo` — Ransom-Trapp + Henry-Fauske critical flow models (selectable)
  - `TwoPhaseFriction.mo` — Martinelli-Nelson two-phase friction multiplier

### Planned (Phase 3+)
- `Pumps/` — Centrifugal pump models
- `HeatExchangers/` — Heat exchangers, steam generators
- `Vessels/` — 3D vessel (Approach B). See `@docs/vessel.md`
- `Kinetics/` — Reactor kinetics. See `@docs/kinetics.md`

## Media Package

Two fluid models with identical API (both extend `PartialMedium`):

- **Water.mo** — Production: IAPWS-IF97 Regions 1, 2, 4. Pure Modelica (no external C).
- **SimpleFluid.mo** — Verification: linear saturation, constant single-phase derivatives.

Swap at system level: `Pipe1D pipe(redeclare package Medium = library.Media.Water)`

## Reference Libraries (external/)

Cloned for architectural reference (not dependencies):
- TRANSFORM (ORNL) — nuclear TH, Apache 2.0
- ThermoPower (Politecnico di Milano) — power plants
- ThermoSysPro (EDF) — thermal power plants
- ClaRa (XRG/TLK) — Clausius-Rankine cycles
- MSL — Modelica Standard Library
- Buildings (LBNL) — HVAC/fluid
- OpenIPSL — power systems
