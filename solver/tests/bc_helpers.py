"""
BC migration helpers: convert common test BC patterns to BoundaryFace objects.

Usage in tests:
    from bc_helpers import make_bc_faces, step_bf

    bc_in, bc_out = make_bc_faces(p_in=10e6, p_out=9.5e6, h_l=700e3, h_v=2800e3)
    step_bf(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, t, dt)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_phase"))
import opal_two_phase as tp


def make_pressure_faces(p_in, p_out, h_l, h_v=0.0, alpha=0.0):
    """Create PressureFace objects for inlet and outlet."""
    return (tp.PressureFace(p_in, h_l, h_v, alpha),
            tp.PressureFace(p_out, h_l, h_v, alpha))


def make_wall_pressure_faces(p_out, h_l, h_v=0.0):
    """Create WallFace inlet + PressureFace outlet."""
    return (tp.WallFace(h_l, h_v),
            tp.PressureFace(p_out, h_l, h_v, 0.0))


def make_wall_wall_faces(h_l=0.0, h_v=0.0):
    """Create WallFace on both sides."""
    return (tp.WallFace(h_l, h_v),
            tp.WallFace(h_l, h_v))


def make_wall_break_faces(p_back, C_d, h_l, h_v=0.0):
    """Create WallFace inlet + BreakFace outlet."""
    return (tp.WallFace(h_l, h_v),
            tp.BreakFace(p_back, C_d, h_l, h_v))


def bc_from_legacy(bc):
    """Convert a legacy BoundaryConditions struct to BoundaryFace pair."""
    h_l = bc.h_l_in if bc.h_l_in != 0 else bc.h_in
    h_v = bc.h_v_in

    # Inlet
    if bc.bc_type_in == tp.BCType.WALL:
        bc_in = tp.WallFace(h_l, h_v)
    else:
        bc_in = tp.PressureFace(bc.p_in, h_l, h_v, bc.alpha_in)

    # Outlet
    if bc.bc_type_out == tp.BCType.WALL:
        bc_out = tp.WallFace(h_l, h_v)
    elif bc.bc_type_out == tp.BCType.BREAK:
        bc_out = tp.BreakFace(bc.p_out, bc.break_area_fraction, h_l, h_v)
    else:
        bc_out = tp.PressureFace(bc.p_out, h_l, h_v, 0.0)

    return bc_in, bc_out


# ---------------------------------------------------------------------------
# Drop-in replacement for solver.step_5eq that routes through step_bf
# ---------------------------------------------------------------------------

_t_accum = {}  # per-solver time accumulator

def step_5eq_migrated(solver, p, alpha, h_l, h_v, mdot, bc, dt,
                       q_wall=None, sources=None):
    """Drop-in replacement for solver.step_5eq that uses step_bf internally.
    Tracks simulation time per solver instance."""
    solver_id = id(solver)
    t = _t_accum.get(solver_id, 0.0)
    bc_in, bc_out = bc_from_legacy(bc)
    solver.step_bf(p, alpha, h_l, h_v, mdot, bc_in, bc_out,
                   t, dt, q_wall, sources)
    _t_accum[solver_id] = t + dt


def reset_time(solver=None):
    """Reset accumulated time for a solver (or all solvers)."""
    if solver is None:
        _t_accum.clear()
    else:
        _t_accum.pop(id(solver), None)


def step_hem_migrated(solver, p, h, mdot, bc_legacy, dt, q_wall=None):
    """Drop-in for solver.step(p, h, mdot, TwoPhaseBCs, dt, q_wall)
    that routes through step_hem_bf."""
    solver_id = id(solver)
    t = _t_accum.get(solver_id, 0.0)

    # TwoPhaseBCs → BoundaryFace
    bc_in = tp.PressureFace(bc_legacy.p_in, bc_legacy.h_in, 0.0, 0.0)
    bc_out = tp.PressureFace(bc_legacy.p_out, bc_legacy.h_in, 0.0, 0.0)

    solver.step_hem_bf(p, h, mdot, bc_in, bc_out, t, dt, q_wall)
    _t_accum[solver_id] = t + dt


def solve_migrated(solver, p, h, mdot, bc_legacy, dt, n_steps, stride=1,
                   q_wall=None):
    """Drop-in for solver.solve() that routes through step_hem_bf."""
    import numpy as np
    p = p.copy(); h = h.copy(); mdot = mdot.copy()

    bc_in = tp.PressureFace(bc_legacy.p_in, bc_legacy.h_in, 0.0, 0.0)
    bc_out = tp.PressureFace(bc_legacy.p_out, bc_legacy.h_in, 0.0, 0.0)

    state_size = len(p) + len(h) + len(mdot)  # HEM: p + h + mdot
    snapshots = []
    t = 0.0

    for s in range(n_steps):
        solver.step_hem_bf(p, h, mdot, bc_in, bc_out, t, dt, q_wall)
        t += dt
        if (s + 1) % stride == 0 or s == n_steps - 1:
            snap = np.concatenate([p, h, mdot])
            snapshots.append(snap)

    return np.array(snapshots)
