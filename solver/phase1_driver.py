"""
phase1_driver.py — Phase 1 end-to-end pipeline demonstration.

Pipeline:
  1. Parse OpenModelica backEnd XML → EquationSystem
  2. Recognise staggered-mesh pipe → PipeGridSpec
  3. Run semi-implicit C++ solver
  4. Report conservation and Hagen-Poiseuille steady-state accuracy

Usage:
  python phase1_driver.py [XML_PATH] [--N N] [--dt DT] [--steps STEPS]

If XML_PATH is omitted, defaults to the feasibility ScalablePipe N=5 result.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root and add solver/ to path
# ---------------------------------------------------------------------------
SOLVER_DIR  = Path(__file__).parent.resolve()
OPAL_ROOT   = SOLVER_DIR.parent
FEASIBILITY = OPAL_ROOT / "feasibility"

sys.path.insert(0, str(OPAL_ROOT))
sys.path.insert(0, str(SOLVER_DIR))

# ---------------------------------------------------------------------------
# Partitioner imports
# ---------------------------------------------------------------------------
from solver.partitioner.xml_reader  import load_equation_system
from solver.partitioner.grid_mapper import map_pipe_grid

# ---------------------------------------------------------------------------
# C++ solver import (must be built first: see solver/single_phase/)
# ---------------------------------------------------------------------------
_SP_DIR = SOLVER_DIR / "single_phase"
sys.path.insert(0, str(_SP_DIR))
try:
    import opal_single_phase as sp
except ImportError as exc:
    sys.exit(
        f"ERROR: opal_single_phase extension not found.\n"
        f"  Build it first:\n"
        f"    cd {_SP_DIR}\n"
        f"    mkdir -p build && cd build\n"
        f"    cmake .. && cmake --build . && cmake --install .\n"
        f"  ({exc})"
    )

import numpy as np


# ---------------------------------------------------------------------------
# Analytical reference
# ---------------------------------------------------------------------------

def hagen_poiseuille_steady(N: int, R: float, p_in: float, p_out: float) -> float:
    """
    Steady-state mass flow rate for N cells in series with equal resistance R.

    Total resistance = (N+1)*R  (N cell resistances + 2 half-boundary faces
    but ScalablePipe uses full R for every face including boundaries, so
    total = (N+1)*R).

    mdot_ss = (p_in - p_out) / ((N+1)*R)
    """
    return (p_in - p_out) / ((N + 1) * R)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(xml_path: Path, dt: float, n_steps: int, stride: int) -> None:
    print("=" * 65)
    print("OPAL Phase 1 — Single-phase solver coupling")
    print("=" * 65)

    # ---- Step 1: Parse XML -----------------------------------------------
    print(f"\n[1] Parsing {xml_path.name} ...")
    t0 = time.perf_counter()
    es = load_equation_system(xml_path)
    print(f"    {es.summary()}  ({time.perf_counter()-t0:.3f}s)")

    # ---- Step 2: Map to pipe grid ----------------------------------------
    print("\n[2] Mapping to staggered pipe grid ...")
    grid = map_pipe_grid(es)
    print(f"    {grid.summary()}")

    # ---- Step 3: Construct solver ----------------------------------------
    print("\n[3] Constructing C++ solver ...")
    solver = sp.SinglePhaseSolver(
        N=grid.N, R=grid.R, C=grid.C,
        rho=grid.rho, Cp=grid.Cp, V=grid.V,
    )
    bc = sp.BoundaryConditions(
        p_in=grid.p_in, p_out=grid.p_out, T_in=grid.T_in,
    )
    print(f"    {solver}")

    # ---- Initial conditions ----------------------------------------------
    p0    = np.array(grid.p0,       dtype=np.float64)
    T0    = np.array(grid.T0,       dtype=np.float64)
    # Initial flows: compute from initial pressures using momentum equations
    mdot0 = np.zeros(grid.N + 1,   dtype=np.float64)
    mdot0[0] = (grid.p_in - p0[0]) / grid.R
    for i in range(1, grid.N):
        mdot0[i] = (p0[i - 1] - p0[i]) / grid.R
    mdot0[grid.N] = (p0[grid.N - 1] - grid.p_out) / grid.R

    # ---- Step 4: Run solver ----------------------------------------------
    print(f"\n[4] Running {n_steps} steps  (dt={dt:.2e} s, stride={stride}) ...")
    t0 = time.perf_counter()
    history = solver.solve(p0, T0, mdot0, bc, dt, n_steps, stride)
    elapsed = time.perf_counter() - t0
    n_snap   = history.shape[0]
    print(f"    {n_snap} snapshots in {elapsed:.3f}s  "
          f"({elapsed/n_steps*1e6:.1f} µs/step)")

    # history shape: (n_snap, 2N+1)  — columns: [p[0..N-1], T[0..N-1], mdot[0..N]]
    N = grid.N
    p_hist    = history[:, :N]
    T_hist    = history[:, N:2*N]
    mdot_hist = history[:, 2*N:]

    # ---- Step 5: Verification -------------------------------------------
    print("\n[5] Verification")
    _check_hagen_poiseuille(p_hist, mdot_hist, grid, n_steps, dt)
    _check_mass_conservation(p_hist, mdot_hist, grid, dt, stride)
    _check_steady_state_pressure(p_hist, mdot_hist, grid)

    print("\nDone.")


def _check_hagen_poiseuille(p_hist, mdot_hist, grid, n_steps, dt):
    """Check that steady-state flow matches Hagen-Poiseuille."""
    mdot_ss_analytical = hagen_poiseuille_steady(
        grid.N, grid.R, grid.p_in, grid.p_out
    )
    # Last snapshot flows (all faces should equalise)
    mdot_final = mdot_hist[-1, :]
    mdot_mean  = mdot_final.mean()
    mdot_range = mdot_final.max() - mdot_final.min()

    rel_err = abs(mdot_mean - mdot_ss_analytical) / abs(mdot_ss_analytical)
    uniformity_err = mdot_range / abs(mdot_ss_analytical)

    sim_time = n_steps * dt
    print(f"  Hagen-Poiseuille (t={sim_time:.3g}s):")
    print(f"    Analytical mdot_ss = {mdot_ss_analytical:.6g} kg/s")
    print(f"    Simulated  mdot̄   = {mdot_mean:.6g} kg/s  (mean of {len(mdot_final)} faces)")
    print(f"    Flow uniformity err= {uniformity_err:.2e}  (ideal=0)")
    print(f"    Rel error vs H-P   = {rel_err:.2e}", end="  ")
    if rel_err < 1e-3:
        print("PASS")
    elif rel_err < 1e-2:
        print("MARGINAL (run longer or reduce dt)")
    else:
        print("FAIL (check solver or extend simulation time)")


def _check_mass_conservation(p_hist, mdot_hist, grid, dt, stride):
    """
    Global mass conservation: d/dt(sum C*p[i]) = mdot_in - mdot_out.

    Check over consecutive snapshots.
    """
    if len(p_hist) < 2:
        print("  Mass conservation: skipped (need >= 2 snapshots)")
        return

    dt_snap = dt * stride
    # dp/dt per snapshot pair
    dp = np.diff(p_hist, axis=0)  # (n_snap-1, N)
    mass_rate_stored = (grid.C * dp / dt_snap).sum(axis=1)  # sum over cells

    # Conservation: C*(p[s+1]-p[s])/dt_snap = mdot[s+1][inlet] - mdot[s+1][outlet]
    mdot_in  = mdot_hist[1:, 0]    # inlet face at later snapshot
    mdot_out = mdot_hist[1:, -1]   # outlet face at later snapshot
    net_inflow = mdot_in - mdot_out

    # Residual
    residual = mass_rate_stored - net_inflow
    rel_res  = np.abs(residual) / (np.abs(net_inflow).mean() + 1e-30)

    print(f"  Mass conservation: max |residual| / mean_inflow = {rel_res.max():.2e}", end="  ")
    if rel_res.max() < 1e-10:
        print("PASS (machine precision)")
    elif rel_res.max() < 1e-6:
        print("PASS")
    else:
        print("FAIL")


def _check_steady_state_pressure(p_hist, mdot_hist, grid):
    """Check that final pressure profile is linear (uniform resistance)."""
    p_final  = p_hist[-1, :]
    N        = grid.N
    # Expected linear profile: p[i] = p_in - (i+1)*R*mdot_ss  (cell centres at i+0.5 faces)
    mdot_ss  = hagen_poiseuille_steady(N, grid.R, grid.p_in, grid.p_out)
    p_expect = np.array([
        grid.p_in - (i + 1) * grid.R * mdot_ss for i in range(N)
    ])
    rel_err  = np.abs(p_final - p_expect) / grid.p_in
    print(f"  Pressure profile linearity: max rel err = {rel_err.max():.2e}", end="  ")
    if rel_err.max() < 1e-3:
        print("PASS")
    else:
        print("MARGINAL (may need more steps to reach steady state)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    default_xml = FEASIBILITY / "results" / "scale_N5_backEnd.xml"

    p = argparse.ArgumentParser(description="OPAL Phase 1 single-phase driver")
    p.add_argument("xml", nargs="?", default=str(default_xml),
                   help="dumpXMLDAE backEnd XML path (default: ScalablePipe N=5)")
    p.add_argument("--dt",    type=float, default=1e-3,
                   help="Timestep [s] (default: 1e-3)")
    p.add_argument("--steps", type=int,   default=5000,
                   help="Number of timesteps (default: 5000)")
    p.add_argument("--stride", type=int,  default=50,
                   help="Snapshot stride (default: 50)")
    args = p.parse_args()

    xml_path = Path(args.xml)
    if not xml_path.exists():
        sys.exit(f"ERROR: XML not found: {xml_path}")

    run(xml_path, dt=args.dt, n_steps=args.steps, stride=args.stride)


if __name__ == "__main__":
    main()
