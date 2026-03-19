"""
BC helpers: construct BoundaryFace objects for tests and call step_bf/step_hem_bf.

No legacy types (BoundaryConditions, TwoPhaseBCs, BCType) — tests construct
BoundaryFace objects directly and pass them to the solver.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_phase"))
import opal_two_phase as tp


# ---------------------------------------------------------------------------
# BoundaryFace constructors for common test patterns
# ---------------------------------------------------------------------------

def pressure_bcs(p_in, p_out, h_l, h_v=0.0, alpha=0.0):
    """Two PressureFace objects (most common test pattern)."""
    return (tp.PressureFace(p_in, h_l, h_v, alpha),
            tp.PressureFace(p_out, h_l, h_v, alpha))


def wall_pressure_bcs(p_out, h_l, h_v=0.0):
    """WallFace inlet + PressureFace outlet."""
    return (tp.WallFace(h_l, h_v),
            tp.PressureFace(p_out, h_l, h_v, 0.0))


def wall_wall_bcs(h_l=0.0, h_v=0.0):
    """WallFace on both sides."""
    return (tp.WallFace(h_l, h_v), tp.WallFace(h_l, h_v))


def wall_break_bcs(p_back, C_d, h_l, h_v=0.0):
    """WallFace inlet + BreakFace outlet."""
    return (tp.WallFace(h_l, h_v),
            tp.BreakFace(p_back, C_d, h_l, h_v))


# ---------------------------------------------------------------------------
# Step wrappers that take BoundaryFace objects + track time
# ---------------------------------------------------------------------------

_t_accum = {}  # per-solver time accumulator


def step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out,
             dt, q_wall=None, sources=None):
    """5-eq step via BoundaryFace. Tracks time per solver instance."""
    sid = id(solver)
    t = _t_accum.get(sid, 0.0)
    solver.step_bf(p, alpha, h_l, h_v, mdot, bc_in, bc_out,
                   t, dt, q_wall, sources)
    _t_accum[sid] = t + dt


def step_hem(solver, p, h, mdot, bc_in, bc_out, dt, q_wall=None):
    """HEM step via BoundaryFace. Tracks time per solver instance."""
    sid = id(solver)
    t = _t_accum.get(sid, 0.0)
    solver.step_hem_bf(p, h, mdot, bc_in, bc_out, t, dt, q_wall)
    _t_accum[sid] = t + dt


def solve_hem(solver, p, h, mdot, bc_in, bc_out, dt, n_steps,
              stride=1, q_wall=None):
    """HEM multi-step via step_hem_bf loop. Returns snapshot array."""
    import numpy as np
    p = p.copy(); h = h.copy(); mdot = mdot.copy()
    snapshots = []
    t = 0.0
    for s in range(n_steps):
        solver.step_hem_bf(p, h, mdot, bc_in, bc_out, t, dt, q_wall)
        t += dt
        if (s + 1) % stride == 0 or s == n_steps - 1:
            snapshots.append(np.concatenate([p, h, mdot]))
    return np.array(snapshots)


def reset_time(solver=None):
    """Reset accumulated time for a solver (or all solvers)."""
    if solver is None:
        _t_accum.clear()
    else:
        _t_accum.pop(id(solver), None)


    # Legacy wrappers removed — all callers now use BoundaryFace directly.
