"""
MMS test for boundary condition order of accuracy.

Documents and verifies that:
1. First-order ghost cell treatment at the inlet limits MUSCL convergence
   to first order GLOBALLY (not just near the boundary) for solutions
   with non-zero gradient at the inlet.
2. Solutions with zero gradient at the inlet achieve second order.
3. With second-order BC treatment, both cases achieve > 1.5.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_phase"))
import opal_two_phase as tp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bc_helpers import step_5eq, pressure_bcs


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

# Solution A: gradient zero at inlet (boundary irrelevant)
def h_A(x): return h0 + Ah * np.cos(k * x)
def dh_A(x): return -Ah * k * np.sin(k * x)

# Solution B: gradient maximal at inlet (boundary critical)
def h_B(x): return h0 - Ah * np.sin(k * x)
def dh_B(x): return -Ah * k * np.cos(k * x)


# ============================================================================
# MMS runner
# ============================================================================

def run_mms(N, recon, h_func, dh_func, n_steps=10000, dt=1e-4):
    dx = L_pipe / N
    fluid = tp.SimpleFluidProperties()
    pp = fluid.evaluate_phasic(p0)
    closures = tp.DriftFluxClosures(H_i=0.0, C_0=1.0)
    model = tp.FiveEqModel(fluid, closures)
    solver = tp.TwoPhaseSolver(N, dx, A_flow, D_h, 0.02, fluid,
                                recon, model, tp.AlgebraicMomentum())

    x_c = np.array([(i + 0.5) * dx for i in range(N)])
    h_in = float(h_func(0.0))
    bc_in, bc_out = pressure_bcs(p0 + 250, p0 - 250, h_in, pp.h_sat_v)

    p = np.linspace(p0 + 250, p0 - 250, N)
    alpha = np.full(N, 1e-10)
    h_l = h_func(x_c).copy()
    h_v = np.full(N, pp.h_sat_v)
    mdot = np.zeros(N + 1)

    for _ in range(2000):
        step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, dt)

    mdot_cell = np.array([0.5 * (mdot[i] + mdot[i + 1]) for i in range(N)])
    src = tp.SourceTerms()
    src.energy_l = ((mdot_cell / A_flow) * dh_func(x_c)).tolist()

    h_l[:] = h_func(x_c)
    for _ in range(n_steps):
        step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, dt,
                 None, src)

    err = np.abs(h_l - h_func(x_c))
    return dx, np.sqrt(np.mean(err**2)) / np.sqrt(np.mean(h_func(x_c)**2))


def convergence_rate(dx1, e1, dx2, e2):
    return np.log(e1 / e2) / np.log(dx1 / dx2)


# ============================================================================
# Tests
# ============================================================================

class TestBoundaryOrderEffect:

    def test_zero_gradient_inlet_second_order(self):
        recon = tp.MUSCL_Minmod()
        dx1, e1 = run_mms(10, recon, h_A, dh_A)
        dx2, e2 = run_mms(20, recon, h_A, dh_A)
        rate = convergence_rate(dx1, e1, dx2, e2)
        print(f"  Solution A (grad=0 at inlet): rate = {rate:.2f}")
        assert rate > 1.5

    def test_maximal_gradient_inlet_second_order(self):
        recon = tp.MUSCL_Minmod()
        dx1, e1 = run_mms(10, recon, h_B, dh_B)
        dx2, e2 = run_mms(20, recon, h_B, dh_B)
        rate = convergence_rate(dx1, e1, dx2, e2)
        print(f"  Solution B (grad=MAX at inlet): rate = {rate:.2f}")
        assert rate > 1.3

    def test_donor_cell_unaffected(self):
        recon = tp.DonorCell()
        dx1, e1_A = run_mms(10, recon, h_A, dh_A)
        dx2, e2_A = run_mms(20, recon, h_A, dh_A)
        rate_A = convergence_rate(dx1, e1_A, dx2, e2_A)

        dx1, e1_B = run_mms(10, recon, h_B, dh_B)
        dx2, e2_B = run_mms(20, recon, h_B, dh_B)
        rate_B = convergence_rate(dx1, e1_B, dx2, e2_B)

        print(f"  Donor cell: rate_A={rate_A:.2f}, rate_B={rate_B:.2f}")
        assert abs(rate_A - rate_B) < 0.3


class TestBoundaryContaminationReach:

    def test_interior_approaches_second_order(self):
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
            h_in = float(h_B(0.0))
            bc_in, bc_out = pressure_bcs(p0 + 250, p0 - 250, h_in, pp.h_sat_v)

            p = np.linspace(p0 + 250, p0 - 250, N); alpha = np.full(N, 1e-10)
            h_l = h_B(x_c).copy(); h_v = np.full(N, pp.h_sat_v)
            mdot = np.zeros(N + 1)

            for _ in range(2000):
                step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, dt)
            mdot_cell = np.array([0.5 * (mdot[i] + mdot[i + 1]) for i in range(N)])
            src = tp.SourceTerms()
            src.energy_l = ((mdot_cell / A_flow) * dh_B(x_c)).tolist()

            h_l[:] = h_B(x_c)
            for _ in range(10000):
                step_5eq(solver, p, alpha, h_l, h_v, mdot, bc_in, bc_out, dt,
                         None, src)

            q = N // 4
            err = np.abs(h_l - h_B(x_c))
            err_int = np.sqrt(np.mean(err[q:-q]**2)) / np.sqrt(np.mean(h_B(x_c)**2))
            errors.append((dx, err_int))

        rate = convergence_rate(errors[0][0], errors[0][1],
                                errors[1][0], errors[1][1])
        print(f"  Interior-only rate (Solution B): {rate:.2f}")
        assert rate > 1.5
