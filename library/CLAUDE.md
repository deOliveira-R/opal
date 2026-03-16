# OPAL Modelica Component Library

## Design Principles

1. **Event-free for real-time.** All components use `smooth()`, `noEvent()`, regularized transitions. No `if/else` events for flow reversal, phase change, regime transitions.
2. **No Modelica.Fluid / Modelica.Media at runtime.** These are incompatible with real-time guarantees (dynamic state selection, event-based phase detection). OPAL has its own equivalents.
3. **Every component carries three layers:** verification tests, validation cases, physics documentation.
4. **Same model, two modes.** Components must work in both analysis mode (adaptive, full events) and real-time mode (fixed-step, bounded iterations).

## Subdirectories

- `Pipes/` — 1D pipe components
- `Pumps/` — Pump models
- `HeatExchangers/` — Heat exchangers, steam generators
- `Vessels/` — 3D vessel (Approach B: monolithic equation-based). See `@docs/vessel.md`
- `Kinetics/` — Reactor kinetics. See `@docs/kinetics.md`
  - `Interfaces/PartialKineticsModel.mo` — Common interface for both kinetics levels
  - `PointKinetics/` — Level 1: 6-group PK, ANS 5.1 decay heat, Xe/Sm
  - `SpatialDiffusion/` — Level 2: few-group 3D diffusion (C++ solver, Modelica wrapper)
  - `Data/` — Reactivity coefficients, delayed neutron data, power shapes, rod worth
- `Media/` — Own IAPWS-IF97 in pure Modelica with analytical thermodynamic derivatives

## Kinetics Interface

Both kinetics levels implement `PartialKineticsModel`. Swapping point kinetics for spatial diffusion is a one-line `redeclare`. TH side does not care which is active.

## Media Package

Pure-Modelica IAPWS-IF97 implementation. Must flatten to algebraic equations (no external C calls). Must provide analytical derivatives (∂ρ/∂P|_h, ∂ρ/∂h|_P) needed by semi-implicit pressure linearization.
