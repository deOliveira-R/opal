# OPAL Modelica Component Library

## Design Principles

1. **Event-free for real-time.** All components use `smooth()`, `noEvent()`, regularized transitions. No `if/else` events for flow reversal, phase change, regime transitions.
2. **No Modelica.Fluid / Modelica.Media at runtime.** These are incompatible with real-time guarantees (dynamic state selection, event-based phase detection). OPAL has its own equivalents.
3. **Every component carries three layers:** verification tests, validation cases, physics documentation.
4. **Same model, two modes.** Components must work in both analysis mode (adaptive, full events) and real-time mode (fixed-step, bounded iterations).

## Subdirectories

### Implemented
- `Media/` — IAPWS-IF97 (Water.mo) + SimpleFluid (verification fluid) + analytical derivatives
  - `IF97/` — Gibbs functions, saturation, derivatives (Regions 1, 2, 4)
  - `tests/` — 12 IAPWS tests (300+ points), 6 SimpleFluid tests

### Planned (Phase 3+)
- `Pipes/` — 1D pipe components
- `Pumps/` — Pump models
- `HeatExchangers/` — Heat exchangers, steam generators
- `Vessels/` — 3D vessel (Approach B). See `@docs/vessel.md`
- `Kinetics/` — Reactor kinetics. See `@docs/kinetics.md`

## Kinetics Interface

Both kinetics levels implement `PartialKineticsModel`. Swapping point kinetics for spatial diffusion is a one-line `redeclare`. TH side does not care which is active.

## Media Package

Two fluid models with identical API (`rho_ph`, `T_ph`, `drho_dp_h`, `drho_dh_p`):

- **Water.mo** — Production: IAPWS-IF97 Regions 1, 2, 4. Pure Modelica (no external C). 12 verification tests, 300+ points against iapws oracle + IAPWS Tables 5/15/35. Coefficients verified via OMPython execution of the actual .mo files.
- **SimpleFluid.mo** — Verification: linear saturation, constant single-phase derivatives. For isolating solver bugs from property bugs. 6 tests, machine-precision accuracy.
