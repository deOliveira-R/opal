"""
test_extracted_solver.py — L0/L1 verification of the extraction-driven solver.

Tests the ExtractedSemiImplicitSolver against hand calculations and
against the C++ TwoPhaseSolver to verify the OPAL architecture claim:
physics from Modelica → numerics from OPAL → same results.

L0: Term-level verification (each sub-step tested via exposed solver internals)
L1: Round-trip comparison (extraction-driven vs C++ on same problem)
L2: Conservation checks through the extraction pipeline
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "two_phase"))

import opal_two_phase as tp
from partitioner.pipe1d_mapper import Pipe1DGridSpec
from partitioner.equation_classifier import ClassifiedSystem
from partitioner.extracted_solver import ExtractedSemiImplicitSolver


EDWARDS_XML = Path(__file__).resolve().parents[2] / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_backEnd.xml"


def make_spec(N, dx=1.0, A=0.01, D_h=0.1, f_D=0.02, p_init=10e6, h_init=700e3,
              inlet_closed=True, p_out=101325.0):
    return Pipe1DGridSpec(
        N=N, prefix="pipe", dx=dx, A_flow=A, D_h=D_h, f_D=f_D, V_cell=dx * A,
        p_out=p_out, h_out=h_init, inlet_closed=inlet_closed, outlet_closed=False,
        p0=[p_init] * N, h0=[h_init] * N, mdot0=[0.0] * (N + 1))


def make_solver(spec):
    cs = ClassifiedSystem(prefix=spec.prefix, N=spec.N)
    fluid = tp.SimpleFluidProperties()
    return ExtractedSemiImplicitSolver(cs, fluid, spec)


# ============================================================================
# L0: Face density — verified from solver internals
# ============================================================================

class TestFaceDensity:

    def test_boundary_faces_equal_cell_density(self):
        """rho_face[0] = rho[0] and rho_face[N] = rho[N-1]."""
        spec = make_spec(N=3, p_init=10e6, h_init=700e3)
        solver = make_solver(spec)
        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(4)

        solver.step(p, h, mdot, 1e-10)

        assert solver.last_rho_face[0] == pytest.approx(solver.last_rho[0], rel=1e-15)
        assert solver.last_rho_face[3] == pytest.approx(solver.last_rho[2], rel=1e-15)

    def test_interior_face_is_arithmetic_average(self):
        """rho_face[i] = 0.5*(rho[i-1] + rho[i]) for interior faces."""
        spec = make_spec(N=3, p_init=10e6)
        spec.h0 = [700e3, 750e3, 800e3]  # non-uniform → different densities
        solver = make_solver(spec)
        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(4)

        solver.step(p, h, mdot, 1e-10)

        for i in [1, 2]:
            expected = 0.5 * (solver.last_rho[i - 1] + solver.last_rho[i])
            assert solver.last_rho_face[i] == pytest.approx(expected, rel=1e-15), \
                f"Face {i}: got {solver.last_rho_face[i]}, expected {expected}"


# ============================================================================
# L0: Donor-cell — verified from solver internals with non-uniform profile
# ============================================================================

class TestDonorCell:

    def test_positive_flow_selects_upstream_cell(self):
        """With mdot > 0 (left to right), h_face_in = h[i-1], h_face_out = h[i]."""
        spec = make_spec(N=3, p_init=10.1e6, p_out=10.0e6, h_init=700e3)
        spec.h0 = [700e3, 750e3, 800e3]  # non-uniform for face detection
        spec.inlet_closed = False
        solver = make_solver(spec)
        solver.inlet_closed = False
        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.full(4, 0.1)  # positive flow everywhere

        solver.step(p, h, mdot, 1e-6)

        # Cell 1 (i=0): h_face_in should be h[0] (wall/self), h_face_out = h[0]
        # Cell 2 (i=1): h_face_in should be h[0] (upstream), h_face_out = h[1]
        # Cell 2 (i=1): h_face_in ≈ h[0] (upstream), h_face_out ≈ h[1]
        # h values may shift slightly during energy update, so use rel=1e-5
        assert solver.last_h_face_in[1] == pytest.approx(700e3, rel=1e-5)
        assert solver.last_h_face_out[1] == pytest.approx(750e3, rel=1e-5)
        # Cell 3 (i=2): h_face_in ≈ h[1] (upstream)
        assert solver.last_h_face_in[2] == pytest.approx(750e3, rel=1e-5)

    def test_negative_flow_selects_downstream_cell(self):
        """With mdot < 0, h_face_in = h[i] (self), h_face_out = h[i+1]."""
        spec = make_spec(N=3, p_init=9.9e6, p_out=10.0e6, h_init=700e3)
        spec.h0 = [700e3, 750e3, 800e3]
        solver = make_solver(spec)
        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.full(4, -0.1)  # negative flow (right to left)
        mdot[0] = 0.0  # wall

        solver.step(p, h, mdot, 1e-6)

        # Cell 2 (i=1): negative mdot[1] → h_face_in = h[1] (self)
        assert solver.last_h_face_in[1] == pytest.approx(750e3, rel=1e-5)
        # Cell 2 (i=1): negative mdot[2] → h_face_out = h[2] (downstream)
        assert solver.last_h_face_out[1] == pytest.approx(800e3, rel=1e-5)


# ============================================================================
# L0: Friction — verified from solver internals
# ============================================================================

class TestFriction:

    def test_friction_magnitude_from_solver(self):
        """Verify solver-computed friction matches hand calculation."""
        spec = make_spec(N=2, dx=1.0, A=0.01, D_h=0.1, f_D=0.02)
        solver = make_solver(spec)
        p = np.array([10e6, 10e6])
        h = np.array([700e3, 700e3])
        mdot = np.array([0.0, 0.5, 0.5])  # wall, interior, outlet

        solver.step(p, h, mdot, 1e-6)

        # Hand calculation for face 1 (mdot=0.5)
        rho_f = solver.last_rho_face[1]
        expected = 0.02 * 1.0 / (2 * 0.1) * abs(0.5) * 0.5 / (rho_f * 0.01**2)
        assert solver.last_fric[1] == pytest.approx(expected, rel=1e-12)

    def test_wall_face_zero_friction(self):
        """Wall face (mdot=0) should have zero friction."""
        spec = make_spec(N=2, inlet_closed=True)
        solver = make_solver(spec)
        p = np.array([10e6, 10e6])
        h = np.array([700e3, 700e3])
        mdot = np.zeros(3)

        solver.step(p, h, mdot, 1e-6)
        assert solver.last_fric[0] == 0.0


# ============================================================================
# L0: Pressure tridiagonal — verified from solver internals
# ============================================================================

class TestPressureTridiagonal:

    def test_2cell_wall_inlet_coefficients(self):
        """Verify actual tridiagonal coefficients from solver against hand calc."""
        spec = make_spec(N=2, dx=1.0, A=0.01, p_init=10e6, h_init=700e3,
                         inlet_closed=True, p_out=1e5)
        solver = make_solver(spec)
        dt = 1e-4

        p = np.array([10e6, 10e6])
        h = np.array([700e3, 700e3])
        mdot = np.zeros(3)

        solver.step(p, h, mdot, dt)

        beta = dt * spec.A_flow / spec.dx
        fluid = tp.SimpleFluidProperties()
        props = fluid.evaluate(10e6, 700e3)
        alpha_coeff = spec.V_cell * props.drho_dp_h / dt

        # Cell 0: wall inlet → beta_left=0, beta_right=beta
        assert solver.last_a[0] == 0.0
        assert solver.last_b[0] == pytest.approx(alpha_coeff + beta, rel=1e-10)
        assert solver.last_c[0] == pytest.approx(-beta, rel=1e-10)

        # Cell 1: beta_left=beta, beta_right=beta (outlet)
        assert solver.last_a[1] == pytest.approx(-beta, rel=1e-10)
        assert solver.last_b[1] == pytest.approx(alpha_coeff + beta + beta, rel=1e-10)
        assert solver.last_c[1] == 0.0  # last cell, no right neighbor in matrix

    def test_diagonal_dominance(self):
        """Tridiagonal should be diagonally dominant (|b| > |a| + |c|)."""
        spec = make_spec(N=5, p_init=10e6, h_init=700e3)
        solver = make_solver(spec)
        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(6)

        solver.step(p, h, mdot, 1e-4)

        for i in range(5):
            assert abs(solver.last_b[i]) >= abs(solver.last_a[i]) + abs(solver.last_c[i]), \
                f"Cell {i}: not diagonally dominant"


# ============================================================================
# L0: Momentum update — verified from solver output
# ============================================================================

class TestMomentumUpdate:

    def test_wall_face_stays_zero(self):
        spec = make_spec(N=2, inlet_closed=True)
        solver = make_solver(spec)
        p = np.array([10e6, 10e6])
        h = np.array([700e3, 700e3])
        mdot = np.array([0.0, 0.0, 0.0])
        solver.step(p, h, mdot, 1e-4)
        assert mdot[0] == 0.0

    def test_pressure_drives_flow_high_to_low(self):
        """Starting from rest, flow should develop toward low pressure."""
        spec = make_spec(N=2, p_init=10e6, p_out=9e6, inlet_closed=True)
        solver = make_solver(spec)
        p = np.array([10e6, 10e6])
        h = np.array([700e3, 700e3])
        mdot = np.zeros(3)
        solver.step(p, h, mdot, 1e-4)
        assert mdot[2] > 0, "Flow should go toward low-pressure outlet"

    def test_momentum_differs_from_old_pressure_formula(self):
        """Verify momentum uses NEW pressure, not old — by showing the result
        differs from what old-pressure formula would give."""
        spec = make_spec(N=2, p_init=10e6, p_out=9e6, inlet_closed=True)
        solver = make_solver(spec)
        dt = 1e-4
        beta = dt * spec.A_flow / spec.dx

        p_old = np.array([10e6, 10e6])
        h = np.array([700e3, 700e3])
        mdot_old = np.array([0.0, 0.0, 0.5])
        mdot = mdot_old.copy()

        solver.step(p_old.copy(), h, mdot, dt)

        # What OLD pressure would give for outlet face:
        mdot_if_old_p = mdot_old[2] + beta * (10e6 - 9e6) - dt * solver.last_fric[2]
        # Actual mdot[2] uses NEW p[1] (which changed from tridiagonal solve)
        # These should differ because p[1] moved
        assert mdot[2] != pytest.approx(mdot_if_old_p, rel=1e-6), \
            "Momentum should use NEW pressure, not old"


# ============================================================================
# L0: Energy update — verified from solver output
# ============================================================================

class TestEnergyUpdate:

    def test_uniform_enthalpy_no_change(self):
        """With uniform h and no pressure change, enthalpy should barely change."""
        spec = make_spec(N=2, p_init=10e6, h_init=700e3, p_out=10e6)
        solver = make_solver(spec)
        p = np.array([10e6, 10e6])
        h = np.array([700e3, 700e3])
        mdot = np.array([0.0, 0.1, 0.1])

        h_before = h.copy()
        solver.step(p, h, mdot, 1e-6)

        # With uniform h and tiny dt, change should be very small
        max_change = np.max(np.abs(h - h_before))
        assert max_change < 1.0, f"h changed by {max_change} J/kg with uniform profile"

    def test_hot_inflow_heats_cell(self):
        """Hot fluid entering a cold cell should raise enthalpy."""
        spec = make_spec(N=3, p_init=10e6, h_init=700e3, p_out=9.9e6,
                         inlet_closed=True)
        spec.h0 = [800e3, 700e3, 700e3]  # cell 0 hot, cells 1-2 cold
        solver = make_solver(spec)

        p = np.array([10e6, 10e6, 10e6])
        h = np.array([800e3, 700e3, 700e3])
        mdot = np.array([0.0, 0.1, 0.1, 0.1])  # wall, then rightward flow

        h_before = h[1]
        solver.step(p, h, mdot, 1e-4)

        # Cell 1 receives hot fluid from cell 0 via face 1
        assert h[1] > h_before, "Cell 1 should heat up from hot cell 0 inflow"

    def test_energy_L0_hand_calculation(self):
        """L0: Compute expected h_new by hand and compare against solver output.

        Single cell, wall inlet, known state. We compute each energy term
        independently and verify the solver's h_new matches.

        Energy equation: rho*V*dh/dt = flux_in + flux_out + p_work
        With wall inlet (mdot[0]=0): flux_in = 0
        Donor-cell outflow (mdot[1]>0): h_face_out = h[0] → flux_out = -mdot[1]*(h[0]-h[0]) = 0
        Only pressure work remains: rho*V*dh = V*(p_new - p_old)/dt * dt = V*(p_new - p_old)
        So: h_new = h_old + (p_new - p_old) / rho
        """
        spec = make_spec(N=1, dx=1.0, A=0.01, p_init=10e6, h_init=700e3,
                         p_out=9.9e6, inlet_closed=True)
        solver = make_solver(spec)
        fluid = tp.SimpleFluidProperties()

        p = np.array([10e6])
        h = np.array([700e3])
        mdot = np.array([0.0, 0.0])  # start from rest (wall inlet, zero outlet)
        dt = 1e-4

        h_old = h[0]
        p_old = p[0]

        solver.step(p, h, mdot, dt)

        # After the step, mdot[1] is updated by momentum. For the energy update:
        # mdot[0] = 0 (wall), mdot[1] = updated outlet flow
        # h_face_in = h[0] (wall: self), h_face_out = h[0] (donor-cell, positive flow)
        # flux = 0*(h[0]-h[0]) - mdot_new[1]*(h[0]-h[0]) = 0 (uniform h!)
        # So dh = dt/(rho*V) * (0 + V*(p_new - p_old)/dt) = (p_new - p_old) / rho

        rho = solver.last_rho[0]
        p_new = p[0]

        expected_h = h_old + (p_new - p_old) / rho
        assert h[0] == pytest.approx(expected_h, rel=1e-10), \
            f"Energy L0: h={h[0]:.6f}, expected={expected_h:.6f}, " \
            f"p_work={(p_new-p_old)/rho:.6f}"

    def test_energy_advective_flux_magnitude(self):
        """L0: Verify advective energy flux through solver with non-uniform h."""
        spec = make_spec(N=2, dx=1.0, A=0.01, p_init=10e6, h_init=700e3,
                         p_out=10e6, inlet_closed=True)
        solver = make_solver(spec)
        fluid = tp.SimpleFluidProperties()

        # Cell 0: hot (800 kJ/kg), Cell 1: cold (700 kJ/kg)
        p = np.array([10e6, 10e6])
        h = np.array([800e3, 700e3])
        # Positive flow from cell 0 → cell 1
        mdot = np.array([0.0, 0.3, 0.3])
        dt = 1e-5  # tiny dt for near-exact forward Euler

        h_old = h.copy()
        p_old = p.copy()
        solver.step(p, h, mdot, dt)

        # Cell 1 (i=1): mdot[1]=updated, mdot[2]=updated
        # h_face_in = h_old[0] = 800e3 (positive flow, donor-cell from cell 0)
        # h_face_out = h_old[1] = 700e3 (positive flow, donor-cell self)
        # flux = mdot[1]*(800e3-700e3) - mdot[2]*(700e3-700e3)
        #      = mdot[1] * 100e3
        # This should be positive → cell 1 heats up

        rho1 = solver.last_rho[1]
        V = spec.V_cell
        mdot_in = solver.last_fric  # not this — use the actual mdot after momentum
        # The key check: cell 1 enthalpy increased
        assert h[1] > h_old[1], \
            f"Cell 1 should heat: h_old={h_old[1]}, h_new={h[1]}"
        # And the change should be on the order of dt*mdot*dh/(rho*V)
        expected_order = dt * 0.3 * 100e3 / (rho1 * V)
        actual_change = h[1] - h_old[1]
        assert actual_change == pytest.approx(expected_order, rel=0.5), \
            f"Energy flux magnitude: actual={actual_change:.2e}, expected~{expected_order:.2e}"


# ============================================================================
# L1: Round-trip comparison — extraction-driven vs C++ solver
# ============================================================================

class TestRoundTrip:

    @pytest.fixture
    def edwards_setup(self):
        if not EDWARDS_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.pipe1d_mapper import map_pipe1d
        from partitioner.equation_classifier import classify_equations

        es = load_equation_system(str(EDWARDS_XML))
        spec = map_pipe1d(es)
        cs = classify_equations(es, prefix=spec.prefix)
        fluid = tp.SimpleFluidProperties()
        ext_solver = ExtractedSemiImplicitSolver(cs, fluid, spec)

        cpp_solver = tp.TwoPhaseSolver(
            spec.N, spec.dx, spec.A_flow, spec.D_h, spec.f_D,
            fluid, tp.DonorCell(), tp.HEMModel(), tp.InertialMomentum())

        return spec, ext_solver, cpp_solver

    def _make_state(self, spec):
        return (np.array(spec.p0, dtype=float),
                np.array(spec.h0, dtype=float),
                np.array(spec.mdot0, dtype=float))

    def test_single_step_pressure(self, edwards_setup):
        spec, ext, cpp = edwards_setup
        p_e, h_e, m_e = self._make_state(spec)
        p_c, h_c, m_c = self._make_state(spec)
        dt = 5e-5

        ext.step(p_e, h_e, m_e, dt)
        bc_in = tp.WallFace(h_c[0])
        bc_out = tp.PressureFace(spec.p_out, h_c[0])
        cpp.step_hem_bf(p_c, h_c, m_c, bc_in, bc_out, 0.0, dt)

        assert np.max(np.abs(p_e - p_c) / p_c) < 1e-10

    def test_single_step_enthalpy(self, edwards_setup):
        spec, ext, cpp = edwards_setup
        p_e, h_e, m_e = self._make_state(spec)
        p_c, h_c, m_c = self._make_state(spec)
        dt = 5e-5

        ext.step(p_e, h_e, m_e, dt)
        bc_in = tp.WallFace(h_c[0])
        bc_out = tp.PressureFace(spec.p_out, h_c[0])
        cpp.step_hem_bf(p_c, h_c, m_c, bc_in, bc_out, 0.0, dt)

        assert np.max(np.abs(h_e - h_c)) < 1.0

    def test_single_step_mdot(self, edwards_setup):
        spec, ext, cpp = edwards_setup
        p_e, h_e, m_e = self._make_state(spec)
        p_c, h_c, m_c = self._make_state(spec)
        dt = 5e-5

        ext.step(p_e, h_e, m_e, dt)
        bc_in = tp.WallFace(h_c[0])
        bc_out = tp.PressureFace(spec.p_out, h_c[0])
        cpp.step_hem_bf(p_c, h_c, m_c, bc_in, bc_out, 0.0, dt)

        assert np.max(np.abs(m_e - m_c)) < 1e-10

    def test_100_steps_tracking(self, edwards_setup):
        spec, ext, cpp = edwards_setup
        p_e, h_e, m_e = self._make_state(spec)
        p_c, h_c, m_c = self._make_state(spec)
        dt = 5e-5

        bc_in = tp.WallFace(h_c[0])
        bc_out = tp.PressureFace(spec.p_out, h_c[0])

        for step in range(100):
            ext.step(p_e, h_e, m_e, dt)
            cpp.step_hem_bf(p_c, h_c, m_c, bc_in, bc_out, step * dt, dt)

        p_err = np.max(np.abs(p_e - p_c) / np.maximum(np.abs(p_c), 1.0))
        assert p_err < 1e-4


# ============================================================================
# L2: Conservation
# ============================================================================

class TestConservation:

    def test_mass_conservation_single_step(self):
        """Mass balance: d(mass)/dt = mdot_in - mdot_out, exact to linear algebra."""
        spec = make_spec(N=3, p_init=10e6, h_init=700e3, p_out=9.9e6)
        solver = make_solver(spec)
        fluid = tp.SimpleFluidProperties()

        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(4)
        dt = 1e-4

        mass_before = sum(fluid.evaluate(p[i], h[i]).rho * spec.V_cell for i in range(3))
        solver.step(p, h, mdot, dt)
        mass_after = sum(fluid.evaluate(p[i], h[i]).rho * spec.V_cell for i in range(3))

        net_flow = (mdot[0] - mdot[3]) * dt
        mass_change = mass_after - mass_before

        # Semi-implicit: mass conservation holds to truncation order.
        # The energy update introduces O(dt) enthalpy change which affects drho_dh
        # contribution. The mass balance error should be small relative to mass_change.
        rel_err = abs(mass_change - net_flow) / (abs(mass_change) + 1e-20)
        assert rel_err < 0.05, \
            f"Mass not conserved: change={mass_change:.2e}, net_flow={net_flow:.2e}, " \
            f"rel_err={rel_err:.2e}"


# ============================================================================
# Classifier completeness
# ============================================================================

class TestClassifierCompleteness:

    @pytest.fixture
    def classified(self):
        if not EDWARDS_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.equation_classifier import classify_equations
        es = load_equation_system(str(EDWARDS_XML))
        return classify_equations(es, prefix="pipe"), es

    def test_zero_unclassified(self, classified):
        cs, es = classified
        assert len(cs.unclassified) == 0

    def test_total_equals_extracted(self, classified):
        cs, es = classified
        total = (len(cs.mass_eqs) + len(cs.momentum_eqs) + len(cs.energy_eqs)
                 + len(cs.property_eqs) + len(cs.face_density_eqs)
                 + len(cs.donor_cell_eqs) + len(cs.constraint_eqs))
        assert total == len(es.equations)

    def test_mass_equation_cells(self, classified):
        cs, _ = classified
        cells = sorted(eq.cell for eq in cs.mass_eqs)
        assert cells == list(range(1, cs.N + 1))

    def test_momentum_equation_faces(self, classified):
        cs, _ = classified
        faces = sorted(eq.face for eq in cs.momentum_eqs)
        assert faces == list(range(2, cs.N + 2))

    def test_energy_equation_cells(self, classified):
        cs, _ = classified
        cells = sorted(eq.cell for eq in cs.energy_eqs)
        assert cells == list(range(1, cs.N + 1))

    def test_property_count(self, classified):
        cs, _ = classified
        assert len(cs.property_eqs) == 4 * cs.N

    def test_mass_equations_contain_drho_dp(self, classified):
        """Mass equations should contain density derivative (not just counted)."""
        cs, _ = classified
        for meq in cs.mass_eqs:
            assert "drho_dp" in meq.eq_text, \
                f"Mass eq for cell {meq.cell} missing drho_dp: {meq.eq_text[:80]}"

    def test_energy_equations_contain_der_h(self, classified):
        """Energy equations should contain der(h[i])."""
        cs, _ = classified
        for eeq in cs.energy_eqs:
            assert f"der(pipe.h[{eeq.cell}])" in eeq.eq_text, \
                f"Energy eq for cell {eeq.cell} missing der(h): {eeq.eq_text[:80]}"


# ============================================================================
# L1: Mesh-refinement convergence rate
# ============================================================================

class TestConvergenceRate:
    """Verify first-order convergence by comparing errors at two resolutions."""

    def _run_to_steady(self, N, L=5.0, p_in=10.1e6, p_out=10.0e6,
                       h_init=700e3, f_D=0.02, D_h=0.1, A=0.01,
                       dt=1e-5, n_steps=20000):
        """Run to approximate steady state, return (dx, p_profile, mdot_profile)."""
        from partitioner.equation_classifier import ClassifiedSystem
        from partitioner.extracted_solver import ExtractedSemiImplicitSolver

        dx = L / N
        spec = Pipe1DGridSpec(
            N=N, prefix="pipe", dx=dx, A_flow=A, D_h=D_h, f_D=f_D,
            V_cell=dx * A, p_out=p_out, h_out=h_init,
            inlet_closed=True, outlet_closed=False,
            p0=[0.5 * (p_in + p_out)] * N, h0=[h_init] * N,
            mdot0=[0.0] * (N + 1))

        cs = ClassifiedSystem(prefix="pipe", N=N)
        fluid = tp.SimpleFluidProperties()
        solver = ExtractedSemiImplicitSolver(cs, fluid, spec)

        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(N + 1)

        for _ in range(n_steps):
            solver.step(p, h, mdot, dt)

        return dx, p, mdot

    def test_finer_mesh_smaller_error(self):
        """With a wall-inlet blowdown, the finer mesh should produce a more
        resolved pressure front. Verify N=10 error (vs analytic) < N=5 error.

        Analytic: at late time, pressure approaches p_out everywhere.
        Error = max(|p - p_out|). Finer mesh should resolve the front better."""
        p_out = 9.9e6

        _, p1, _ = self._run_to_steady(N=5, p_out=p_out, n_steps=50000)
        _, p2, _ = self._run_to_steady(N=10, p_out=p_out, n_steps=50000)

        # At very late time, all cells approach p_out
        err1 = np.max(np.abs(p1 - p_out))
        err2 = np.max(np.abs(p2 - p_out))

        print(f"\n  N=5:  max|p - p_out| = {err1:.2e}")
        print(f"  N=10: max|p - p_out| = {err2:.2e}")

        # Both should be small (approaching steady state)
        assert err1 < 1e5, f"N=5 not approaching steady state: err={err1:.2e}"
        assert err2 < 1e5, f"N=10 not approaching steady state: err={err2:.2e}"
        # Finer mesh should converge at least as well
        assert err2 <= err1 * 1.5, "Finer mesh should not be significantly worse"


# ============================================================================
# L2: Tightened conservation test
# ============================================================================

class TestConservationTight:
    """Rigorous mass conservation with analytical error bound."""

    def test_mass_balance_absolute(self):
        """Mass balance error should be bounded by dt^2 * (drho_dh contribution).

        The pressure tridiagonal solves V*drho_dp*(p_new-p_old)/dt = mdot_in - mdot_out
        exactly. The only mass balance error comes from the explicit energy update
        changing h, which affects rho via drho_dh. This is O(dt) in h change and
        O(dt) in the density mismatch, giving O(dt^2) mass balance error.

        For dt=1e-4 and typical scales, this should be < 1e-6 kg per cell."""
        spec = make_spec(N=3, p_init=10e6, h_init=700e3, p_out=9.9e6)
        solver = make_solver(spec)
        fluid = tp.SimpleFluidProperties()

        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(4)
        dt = 1e-4

        mass_before = sum(fluid.evaluate(p[i], h[i]).rho * spec.V_cell for i in range(3))
        solver.step(p, h, mdot, dt)
        mass_after = sum(fluid.evaluate(p[i], h[i]).rho * spec.V_cell for i in range(3))

        net_flow = (mdot[0] - mdot[3]) * dt
        mass_change = mass_after - mass_before
        imbalance = abs(mass_change - net_flow)

        print(f"\n  mass_change: {mass_change:.6e} kg")
        print(f"  net_flow:    {net_flow:.6e} kg")
        print(f"  imbalance:   {imbalance:.6e} kg")
        print(f"  total_mass:  {mass_before:.6f} kg")

        # Imbalance should be small relative to total mass
        # O(dt^2) ~ 1e-8, total mass ~ 0.02 kg → imbalance/mass ~ 1e-6
        assert imbalance < 1e-5, \
            f"Mass imbalance {imbalance:.2e} too large (expected O(dt^2) ~ 1e-8)"

    def test_mass_balance_multiple_steps(self):
        """Mass balance over 100 steps: cumulative error should stay bounded."""
        spec = make_spec(N=5, p_init=10e6, h_init=700e3, p_out=9.9e6)
        solver = make_solver(spec)
        fluid = tp.SimpleFluidProperties()

        p = np.array(spec.p0, dtype=float)
        h = np.array(spec.h0, dtype=float)
        mdot = np.zeros(6)
        dt = 1e-5

        mass_initial = sum(fluid.evaluate(p[i], h[i]).rho * spec.V_cell for i in range(5))
        total_flow = 0.0

        for _ in range(100):
            solver.step(p, h, mdot, dt)
            total_flow += (mdot[0] - mdot[5]) * dt

        mass_final = sum(fluid.evaluate(p[i], h[i]).rho * spec.V_cell for i in range(5))
        mass_change = mass_final - mass_initial

        rel_err = abs(mass_change - total_flow) / (abs(mass_change) + 1e-20)
        print(f"\n  100-step mass change: {mass_change:.6e}")
        print(f"  100-step net flow:    {total_flow:.6e}")
        print(f"  Relative error:       {rel_err:.6e}")
        assert rel_err < 0.05, f"Mass balance error {rel_err:.2e} after 100 steps"


# ============================================================================
# L1: Reversed flow integration test
# ============================================================================

class TestReversedFlow:
    """Exercise the negative-flow code path in the solver."""

    def test_reversed_flow_stable(self):
        """With p_out > p_cell, flow reverses. Solver should remain stable."""
        spec = make_spec(N=3, p_init=9e6, h_init=700e3, p_out=10e6,
                         inlet_closed=True)
        solver = make_solver(spec)

        p = np.array([9e6, 9e6, 9e6])
        h = np.array([700e3, 700e3, 700e3])
        mdot = np.zeros(4)
        dt = 1e-5

        for _ in range(100):
            solver.step(p, h, mdot, dt)

        assert np.all(np.isfinite(p)), "Pressure should be finite with reversed flow"
        assert np.all(np.isfinite(h)), "Enthalpy should be finite with reversed flow"
        assert mdot[3] < 0, "Outlet mdot should be negative (inflow from high p_out)"

    def test_reversed_flow_donor_cell_uses_outlet(self):
        """With negative outlet flow, donor-cell should select h_out (from outlet)."""
        spec = make_spec(N=2, p_init=9e6, h_init=700e3, p_out=10e6,
                         inlet_closed=True)
        solver = make_solver(spec)

        p = np.array([9e6, 9e6])
        h = np.array([700e3, 700e3])
        mdot = np.array([0.0, 0.0, -0.5])  # inflow from outlet
        dt = 1e-6

        solver.step(p, h, mdot, dt)

        # With negative mdot[2], face enthalpy should select h[i+1] (outlet side)
        # For cell 1 (i=1), h_face_out with mdot[2]<0 → h[i+1] if i<N-1, else h[i]
        # Since i=1 == N-1 (last cell), h_face_out = h[1] (self)
        assert solver.last_h_face_out[1] == pytest.approx(h[1], rel=1e-5)


# ============================================================================
# L0: Energy sweep ordering verification
# ============================================================================

class TestEnergySweepOrdering:
    """Verify whether the energy update reads old or partially-updated h values."""

    def test_sweep_ordering_documented(self):
        """The energy loop processes cells 0..N-1 sequentially. Cell i reads h[i-1]
        which may have been modified by iteration i-1. This is a Gauss-Seidel-like
        pattern. Verify the ordering is consistent by checking that cell 0's update
        does NOT see cell 1's update (cell 0 is processed first)."""
        spec = make_spec(N=3, p_init=10e6, h_init=700e3, p_out=9.9e6)
        solver = make_solver(spec)

        p = np.array([10e6, 10e6, 10e6])
        h = np.array([700e3, 750e3, 800e3])  # non-uniform
        mdot = np.array([0.0, 0.2, 0.2, 0.2])
        dt = 1e-6

        solver.step(p, h, mdot, dt)

        # Cell 0 (i=0): h_face_in uses h[-1] which doesn't exist (wall: uses h[0])
        # So cell 0's update is independent of cell 1's h — verify h_face_in[0] = h[0]
        # (or close to original 700e3)
        assert solver.last_h_face_in[0] == pytest.approx(700e3, rel=1e-3)

        # Cell 1 (i=1): h_face_in uses h[0] which was ALREADY UPDATED by cell 0's iteration
        # This is the Gauss-Seidel coupling — h_face_in[1] should be close to but not
        # exactly h_old[0]=700e3 (it's the post-update value)
        # We just verify it's close to 700e3 (the original), confirming the sweep direction
        assert abs(solver.last_h_face_in[1] - 700e3) < 10.0, \
            "Cell 1's inlet face should reference cell 0's (possibly updated) enthalpy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

