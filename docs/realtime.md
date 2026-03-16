# Real-Time Performance Target

Target: 1s wall-clock per 1s simulated (training fidelity), 2-5× slower (licensing fidelity). Second market beyond safety analysis: training simulators.

## Performance Budget (Δt = 0.05s, 20 steps/sec)

| Component | Training | Licensing |
|---|---|---|
| TH semi-implicit | 400 cells, <1ms | 2500 cells, 5-10ms |
| Property evaluations | ~0.5ms | 2-5ms |
| Point kinetics | <0.01ms | <0.01ms |
| XS interpolation (amortized) | ~0.1ms | ~0.5ms |
| Shape update (amortized) | ~0.5ms | ~1-3ms |
| BOP + Controls (DAE) | <1ms | <1ms |
| **Total/step** | **~2-3ms** | **~10-20ms** |
| **Budget/step** | **50ms** | **50ms** |
| **Margin** | **~15-20×** | **~2-5×** |

## Requirements

1. **Compiled code, not interpreted.** Extraction produces C++ source → compiled. Interpreted expression trees or Python callbacks kill real-time. Non-negotiable.
2. **Own media package, compiled.** No `Modelica.Media.Water.StandardWater`. Pure-Modelica IAPWS-IF97 → C. Analytical derivatives alongside values.
3. **Event-free components.** `smooth()`, `noEvent()`, regularized transitions. Events cause solver halt/locate/restart → unbounded timing.
4. **Fixed-step real-time mode.** BOP/controls: fixed-step implicit Euler, bounded Newton (3-5 iterations, accept regardless). Guarantees bounded wall-clock per step.
5. **No Modelica.Fluid / Modelica.Media at runtime.** Patterns incompatible with real-time (dynamic state selection, event-based phase detection).

## Stretch Goal: Same Model, Two Modes

One Modelica model runs in analysis mode (adaptive, full events, max accuracy) and real-time mode (fixed step, bounded iterations, regularized). Mode = solver config flag, not model change. Genuinely unique capability.

## Measurement

Add real-time benchmark to Phase 3. After plant transient runs correctly, measure per-step wall-clock. Target: within 5× at Phase 3 fidelity. If 100× off → architectural overhead problem.
