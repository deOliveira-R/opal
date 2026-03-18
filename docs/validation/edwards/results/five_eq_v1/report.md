# Edwards-O'Brien Blowdown — 5-Equation Drift-Flux (non-equilibrium flashing)

## Problem

NRC Standard Problem 1. Horizontal pipe (4.096 m, 0.073 m ID) filled
with subcooled water at 7 MPa / 502 K, ruptured at one end. Duration 0.6 s.

## Solver: 5-Equation Drift-Flux (non-equilibrium flashing)

5-equation drift-flux model with separate liquid/vapor energy,
nucleation onset, and interfacial heat transfer.
Phase 3b, iteration v1.

- Mesh: 24 cells, dx = 0.1707 m
- Interfacial HTC: H_i = 1e+07 W/(m³·K)

## Pressure Comparison — Mean Absolute Percent Error

| Station | x [m] | Overall | Early (<50ms) | Mid (50-200ms) | Late (>200ms) |
|---------|-------|---------|---------------|----------------|---------------|
| GS-1 | 3.927 | 68.8% | 41.5% | 32.5% | 106.0% |
| GS-2 | 3.769 | 102.3% | 6.4% | 21.2% | 216.3% |
| GS-3 | 2.935 | 69.8% | 6.3% | 6.2% | 144.8% |
| GS-4 | 2.024 | 55.1% | 19.1% | 12.2% | 103.4% |
| GS-5 | 1.469 | 87.4% | 18.7% | 4.8% | 156.2% |
| GS-6 | 0.914 | 72.7% | 9.2% | 15.3% | 117.6% |
| GS-7 | 0.079 | 58.3% | 21.5% | 4.2% | 123.1% |
| **Overall** | | **73.5%** | | | |

## Figures

- `pressure_all_stations.png` — Pressure at all 7 gauge stations (0-600 ms)
- `pressure_early_time.png` — Early time detail (0-20 ms, wave propagation)
- `void_fraction.png` — Void fraction evolution at 4 selected stations
