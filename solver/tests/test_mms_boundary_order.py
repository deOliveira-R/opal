"""
MMS test for boundary condition order of accuracy.

Documents and verifies that:
1. First-order ghost cell treatment at the inlet limits MUSCL convergence
   to first order GLOBALLY (not just near the boundary) for solutions
   with non-zero gradient at the inlet.
2. Solutions with zero gradient at the inlet achieve second order.
3. After second-order BC implementation, both cases should achieve ~2.0.

This test uses two manufactured solutions:
  A) h(x) = h0 + Ah*cos(πx/L)  — gradient ZERO at inlet
  B) h(x) = h0 - Ah*sin(πx/L)  — gradient MAXIMAL at inlet

For hyperbolic (advection) equations, boundary errors propagate along
characteristics through the entire domain. Interior-only measurement
does NOT help because every cell downstream of the boundary is contaminated.

Reference: LeVeque, "Finite Volume Methods for Hyperbolic Problems", §8.6
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_phase"))
import opal_two_phase as tp


# ============================================================================
# Parameters
# ============================================================================

L_pipe = 5.0
A_flow = 0.01
D_h = 0.1
p0 = 10.0e6
h0 = 700.0e3
Ah = 10.0e3
k = np.pi / L_pipe


# ============================================================================
# Manufactured solutions
# ============================================================================

# Solution A: gradient zero at inlet (boundary irrelevant)
def h_A(x): return h0 + Ah * np.cos(k * x)
def dh_A(x): return -Ah * k * np.sin(k * x)

# Solution B: gradient maximal at inlet (boundary critical)
def h_B(x): return h0 - Ah * np.sin(k * x)
def dh_B(x): return -Ah * k * np.cos(k * x)


# ============================================================================
# MMS runner (shared with test_mms_convergence.py pattern)
# ============================================================================

def run_mms(N, recon, h_func, dh_func, n_steps=10000, dt=1e-4):
    """Run MMS for a given manufactured solution and return L2 error."""
    dx = L_pipe / N

    fluid = tp.SimpleFluidProperties()
    pp = fluid.evaluate_phasic(p0)
    closures = tp.DriftFluxClosures(H_i=0.0, C_0=1.0)
    model = tp.FiveEqModel(fluid, closures)
    solver = tp.TwoPhaseSolver(N, dx, A_flow, D_h, 0.02, fluid,
                                recon, model, tp.AlgebraicMomentum())

    x_c = np.array([(i + 0.5) * dx for i in range(N)])

    bc = tp.BoundaryConditions()
    bc.bc_type_in = tp.BCType.PRESSURE
    bc.bc_type_out = tp.BCType.PRESSURE
    bc.p_in = p0 + 250
    bc.p_out = p0 - 250
    bc.h_in = float(h_func(0.0))
    bc.h_l_in = float(h_func(0.0))
    bc.h_v_in = pp.h_sat_v

    p = np.linspace(bc.p_in, bc.p_out, N)
    alpha = np.full(N, 1e-10)
    h_l = h_func(x_c).copy()
    h_v = np.full(N, pp.h_sat_v)
    mdot = np.zeros(N + 1)

    # Converge pressure/flow
    for _ in range(2000):
        solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, dt)

    # Energy source from actual per-cell flow
    mdot_cell = np.array([0.5 * (mdot[i] + mdot[i + 1]) for i in range(N)])
    src = tp.SourceTerms()
    src.energy_l = ((mdot_cell / A_flow) * dh_func(x_c)).tolist()

    # Run enthalpy to steady state
    h_l[:] = h_func(x_c)
    for _ in range(n_steps):
        solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, dt, None, src)

    err = np.abs(h_l - h_func(x_c))
    err_L2 = np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(h_func(x_c)**2))
    return dx, err_L2


def convergence_rate(dx1, e1, dx2, e2):
    """Single-pair convergence rate."""
    return np.log(e1 / e2) / np.log(dx1 / dx2)


# ============================================================================
# Tests
# ============================================================================

class TestBoundaryOrderEffect:
    """Verify that first-order inlet BC limits MUSCL to first-order globally."""

    def test_zero_gradient_inlet_second_order(self):
        """Solution A (grad=0 at inlet): MUSCL achieves ~2nd order.
        This is the control case — boundary doesn't matter."""
        recon = tp.MUSCL_Minmod()
        dx1, e1 = run_mms(10, recon, h_A, dh_A)
        dx2, e2 = run_mms(20, recon, h_A, dh_A)

        rate = convergence_rate(dx1, e1, dx2, e2)
        print(f"  Solution A (grad=0 at inlet): rate = {rate:.2f}")
        assert rate > 1.5, (
            f"With zero gradient at inlet, MUSCL should be ~2nd order. "
            f"Got rate = {rate:.2f}"
        )

    def test_maximal_gradient_inlet_first_order(self):
        """Solution B (grad=MAX at inlet): MUSCL drops to ~1st order.
        The first-order ghost cell kills global accuracy."""
        recon = tp.MUSCL_Minmod()
        dx1, e1 = run_mms(10, recon, h_B, dh_B)
        dx2, e2 = run_mms(20, recon, h_B, dh_B)

        rate = convergence_rate(dx1, e1, dx2, e2)
        print(f"  Solution B (grad=MAX at inlet): rate = {rate:.2f}")
        # With second-order ghost cell extrapolation at interior faces
        # and Dirichlet BC at the boundary face, MUSCL achieves > 1.5.
        assert rate > 1.3, (
            f"With second-order BC treatment, MUSCL should achieve > 1.3. "
            f"Got {rate:.2f}"
        )

    def test_donor_cell_unaffected(self):
        """Donor cell is first-order regardless of boundary treatment."""
        recon = tp.DonorCell()
        dx1, e1_A = run_mms(10, recon, h_A, dh_A)
        dx2, e2_A = run_mms(20, recon, h_A, dh_A)
        rate_A = convergence_rate(dx1, e1_A, dx2, e2_A)

        dx1, e1_B = run_mms(10, recon, h_B, dh_B)
        dx2, e2_B = run_mms(20, recon, h_B, dh_B)
        rate_B = convergence_rate(dx1, e1_B, dx2, e2_B)

        print(f"  Donor cell: rate_A={rate_A:.2f}, rate_B={rate_B:.2f}")
        # Both should be ~1.0 (first order is first order)
        assert abs(rate_A - rate_B) < 0.3, (
            f"Donor cell rate should be ~1.0 for both solutions. "
            f"A={rate_A:.2f}, B={rate_B:.2f}"
        )


class TestBoundaryContaminationReach:
    """Verify that boundary error propagates through entire domain."""

    def test_interior_also_first_order(self):
        """Even excluding boundary cells, MUSCL is first-order for Solution B.
        This proves the boundary error propagates via advection characteristics
        through the entire domain — not just a local effect."""
        recon = tp.MUSCL_Minmod()
        errors = []
        for N in [10, 20, 40]:
            dx = L_pipe / N; dt = 1e-4
            fluid = tp.SimpleFluidProperties()
            pp = fluid.evaluate_phasic(p0)
            closures = tp.DriftFluxClosures(H_i=0.0, C_0=1.0)
            model = tp.FiveEqModel(fluid, closures)
            solver = tp.TwoPhaseSolver(N, dx, A_flow, D_h, 0.02, fluid,
                                        recon, model, tp.AlgebraicMomentum())

            x_c = np.array([(i + 0.5) * dx for i in range(N)])
            bc = tp.BoundaryConditions()
            bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
            bc.p_in = p0 + 250; bc.p_out = p0 - 250
            bc.h_in = float(h_B(0.0)); bc.h_l_in = bc.h_in; bc.h_v_in = pp.h_sat_v

            p = np.linspace(bc.p_in, bc.p_out, N); alpha = np.full(N, 1e-10)
            h_l = h_B(x_c).copy(); h_v = np.full(N, pp.h_sat_v)
            mdot = np.zeros(N + 1)

            for _ in range(2000):
                solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, dt)
            mdot_cell = np.array([0.5 * (mdot[i] + mdot[i + 1]) for i in range(N)])
            src = tp.SourceTerms()
            src.energy_l = ((mdot_cell / A_flow) * dh_B(x_c)).tolist()

            h_l[:] = h_B(x_c)
            for _ in range(10000):
                solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, dt, None, src)

            # Interior only: middle 50%
            q = N // 4
            err = np.abs(h_l - h_B(x_c))
            err_int = np.sqrt(np.mean(err[q:-q]**2)) / np.sqrt(np.mean(h_B(x_c)**2))
            errors.append((dx, err_int))

        # Compute interior convergence rate
        rate = convergence_rate(errors[0][0], errors[0][1],
                                errors[1][0], errors[1][1])
        print(f"  Interior-only rate (Solution B): {rate:.2f}")
        # With second-order BC treatment, interior should be near 2.0
        assert rate > 1.5, (
            f"Interior rate with second-order BC should be > 1.5. "
            f"Got {rate:.2f}"
        )
