# Edwards-O'Brien Blowdown — HEM (3-equation, inertial momentum, Ransom-Trapp)

## Problem

NRC Standard Problem 1. Horizontal pipe (4.096 m, 0.073 m ID) filled
with subcooled water at 7 MPa / 502 K, ruptured at one end. Duration 0.6 s.

## Solver: HEM (3-equation, inertial momentum, Ransom-Trapp)

3-equation homogeneous equilibrium model with inertial momentum
and Ransom-Trapp critical flow blend. Phase 2.5a, iteration v4.

- Mesh: 24 cells, dx = 0.1707 m

## Pressure Comparison — Mean Absolute Percent Error

| Station | x [m] | Overall | Early (<50ms) | Mid (50-200ms) | Late (>200ms) |
|---------|-------|---------|---------------|----------------|---------------|
| GS-1 | 3.927 | 98.0% | 41.4% | 41.4% | 168.7% |
| GS-2 | 3.769 | 116.1% | 6.3% | 22.4% | 247.1% |
| GS-3 | 2.935 | 97.0% | 6.4% | 4.7% | 205.1% |
| GS-4 | 2.024 | 79.1% | 19.1% | 5.6% | 160.5% |
| GS-5 | 1.469 | 128.3% | 18.4% | 2.0% | 236.1% |
| GS-6 | 0.914 | 106.8% | 9.2% | 11.0% | 178.0% |
| GS-7 | 0.079 | 82.1% | 21.6% | 0.5% | 186.8% |
| **Overall** | | **101.1%** | | | |

## Figures

- `pressure_all_stations.png` — Pressure at all 7 gauge stations (0-600 ms)
- `pressure_early_time.png` — Early time detail (0-20 ms, wave propagation)
