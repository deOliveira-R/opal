"""
BC Migration verification: legacy step_5eq() vs new step_bf() must produce
identical results for every BC type.

This test runs the SAME problem through both code paths and asserts the
outputs are bitwise identical. If any difference appears, the migration
has a bug. Once all tests pass, the legacy path can be safely deprecated.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_phase"))
import opal_two_phase as tp


def make_solver(N=5, f_D=0.02):
    fluid = tp.SimpleFluidProperties()
    pp = fluid.evaluate_phasic(10e6)
    closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0, alpha_nucleation=1e-3)
    model = tp.FiveEqModel(fluid, closures)
    solver = tp.TwoPhaseSolver(N, 1.0, 0.01, 0.1, f_D, fluid,
                                tp.DonorCell(), model, tp.InertialMomentum())
    return solver, fluid, pp


def assert_identical(p1, a1, h1, hv1, m1, p2, a2, h2, hv2, m2, ctx=""):
    """Assert two state sets are bitwise identical."""
    np.testing.assert_array_equal(p1, p2, err_msg=f"{ctx} p differs")
    np.testing.assert_array_equal(a1, a2, err_msg=f"{ctx} alpha differs")
    np.testing.assert_array_equal(h1, h2, err_msg=f"{ctx} h_l differs")
    np.testing.assert_array_equal(hv1, hv2, err_msg=f"{ctx} h_v differs")
    np.testing.assert_array_equal(m1, m2, err_msg=f"{ctx} mdot differs")


class TestMigrationPressureBC:
    """Pressure BCs: legacy struct vs PressureFace must be identical."""

    def test_pressure_bc_identical(self):
        solver, fluid, pp = make_solver()
        N = 5; dt = 1e-4

        # Legacy path
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = 700e3; bc.h_l_in = 700e3; bc.h_v_in = pp.h_sat_v

        p1 = np.full(N, 10e6); a1 = np.full(N, 1e-8)
        h1 = np.full(N, 700e3); hv1 = np.full(N, pp.h_sat_v)
        m1 = np.zeros(N + 1)

        # New path
        bc_in = tp.PressureFace(10e6, 700e3, pp.h_sat_v, 0.0)
        bc_out = tp.PressureFace(9.5e6, 700e3, pp.h_sat_v, 0.0)

        p2 = p1.copy(); a2 = a1.copy()
        h2 = h1.copy(); hv2 = hv1.copy()
        m2 = m1.copy()

        for step in range(100):
            solver.step_5eq(p1, a1, h1, hv1, m1, bc, dt)
            solver.step_bf(p2, a2, h2, hv2, m2, bc_in, bc_out, step * dt, dt)
            assert_identical(p1, a1, h1, hv1, m1,
                           p2, a2, h2, hv2, m2,
                           f"step {step}")


class TestMigrationWallBC:
    """Wall BC at inlet: legacy struct vs WallFace must be identical."""

    def test_wall_inlet_identical(self):
        solver, fluid, pp = make_solver()
        N = 5; dt = 1e-4

        # Legacy
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.WALL
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_out = 5e6
        bc.h_in = 700e3; bc.h_l_in = 700e3; bc.h_v_in = pp.h_sat_v

        p1 = np.full(N, 10e6); a1 = np.full(N, 1e-8)
        h1 = np.full(N, 700e3); hv1 = np.full(N, pp.h_sat_v)
        m1 = np.zeros(N + 1)

        # New
        bc_in = tp.WallFace(700e3, pp.h_sat_v)
        bc_out = tp.PressureFace(5e6, 700e3, pp.h_sat_v, 0.0)

        p2 = p1.copy(); a2 = a1.copy()
        h2 = h1.copy(); hv2 = hv1.copy()
        m2 = m1.copy()

        for step in range(100):
            solver.step_5eq(p1, a1, h1, hv1, m1, bc, dt)
            solver.step_bf(p2, a2, h2, hv2, m2, bc_in, bc_out, step * dt, dt)
            assert_identical(p1, a1, h1, hv1, m1,
                           p2, a2, h2, hv2, m2,
                           f"step {step}")


class TestMigrationBreakBC:
    """Break BC: legacy struct vs BreakFace must be identical."""

    def test_break_outlet_identical(self):
        fluid = tp.IAPWSIF97Properties()
        pp = fluid.evaluate_phasic(7e6)
        closures = tp.DriftFluxClosures(H_i=1e7, C_0=1.0, alpha_nucleation=1e-3)
        model = tp.FiveEqModel(fluid, closures)
        N = 10; dx = 0.4; A = 0.0042; D_h = 0.073; f_D = 0.02
        critical_flow = tp.RansomTrapp(fluid, x_trans=0.10, c_floor=1200.0)
        solver = tp.TwoPhaseSolver(N, dx, A, D_h, f_D, fluid,
                                    tp.DonorCell(), model,
                                    tp.InertialMomentum(), critical_flow)

        dt = 5e-5; h_init = pp.h_sat_l - 200e3

        # Legacy
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.WALL
        bc.bc_type_out = tp.BCType.BREAK
        bc.p_out = 101325.0
        bc.break_area_fraction = 0.87
        bc.h_in = h_init; bc.h_l_in = h_init; bc.h_v_in = pp.h_sat_v

        p1 = np.full(N, 7e6); a1 = np.full(N, 1e-6)
        h1 = np.full(N, h_init); hv1 = np.full(N, pp.h_sat_v)
        m1 = np.zeros(N + 1)

        # New
        bc_in = tp.WallFace(h_init, pp.h_sat_v)
        bc_out = tp.BreakFace(101325.0, 0.87, h_init, pp.h_sat_v)

        p2 = p1.copy(); a2 = a1.copy()
        h2 = h1.copy(); hv2 = hv1.copy()
        m2 = m1.copy()

        for step in range(200):
            solver.step_5eq(p1, a1, h1, hv1, m1, bc, dt)
            solver.step_bf(p2, a2, h2, hv2, m2, bc_in, bc_out, step * dt, dt)
            assert_identical(p1, a1, h1, hv1, m1,
                           p2, a2, h2, hv2, m2,
                           f"step {step}")


class TestMigrationWithSources:
    """Verify SourceTerms work identically through both paths."""

    def test_sources_identical(self):
        solver, fluid, pp = make_solver()
        N = 5; dt = 1e-4

        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = 700e3; bc.h_l_in = 700e3; bc.h_v_in = pp.h_sat_v

        bc_in = tp.PressureFace(10e6, 700e3, pp.h_sat_v, 0.0)
        bc_out = tp.PressureFace(9.5e6, 700e3, pp.h_sat_v, 0.0)

        src = tp.SourceTerms()
        src.energy_l = [1e5] * N

        p1 = np.full(N, 10e6); a1 = np.full(N, 1e-8)
        h1 = np.full(N, 700e3); hv1 = np.full(N, pp.h_sat_v)
        m1 = np.zeros(N + 1)

        p2 = p1.copy(); a2 = a1.copy()
        h2 = h1.copy(); hv2 = hv1.copy()
        m2 = m1.copy()

        for step in range(50):
            solver.step_5eq(p1, a1, h1, hv1, m1, bc, dt, None, src)
            solver.step_bf(p2, a2, h2, hv2, m2, bc_in, bc_out, step * dt, dt,
                          None, src)
            assert_identical(p1, a1, h1, hv1, m1,
                           p2, a2, h2, hv2, m2,
                           f"step {step}")


class TestMigrationTwoPhase:
    """Two-phase flow with significant void — most demanding test."""

    def test_two_phase_flow_identical(self):
        solver, fluid, pp = make_solver()
        N = 5; dt = 1e-4

        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_l + 50e3
        bc.h_l_in = pp.h_sat_l + 50e3
        bc.h_v_in = pp.h_sat_v
        bc.alpha_in = 0.1

        bc_in = tp.PressureFace(10e6, pp.h_sat_l + 50e3, pp.h_sat_v, 0.1)
        bc_out = tp.PressureFace(9.5e6, pp.h_sat_l + 50e3, pp.h_sat_v, 0.1)

        p1 = np.full(N, 10e6); a1 = np.full(N, 0.1)
        h1 = np.full(N, pp.h_sat_l + 50e3); hv1 = np.full(N, pp.h_sat_v)
        m1 = np.zeros(N + 1)

        p2 = p1.copy(); a2 = a1.copy()
        h2 = h1.copy(); hv2 = hv1.copy()
        m2 = m1.copy()

        for step in range(100):
            solver.step_5eq(p1, a1, h1, hv1, m1, bc, dt)
            solver.step_bf(p2, a2, h2, hv2, m2, bc_in, bc_out, step * dt, dt)
            assert_identical(p1, a1, h1, hv1, m1,
                           p2, a2, h2, hv2, m2,
                           f"step {step}")
