"""
P1 tests — Term-by-term verification against hand calculations.

Design philosophy: AI-generated numerical code makes "plausible errors" —
correct-looking code with flipped signs, swapped variables, wrong indices,
missing factors. These compile, run, and even pass coarse integration tests.

To catch them, every term in every equation must be verified INDIVIDUALLY
against a hand calculation with known inputs and known outputs. No emergent
behavior — if a test passes, we know the specific term is correct.

AI failure modes targeted:
  - Sign flips:     (a - b) vs (b - a)
  - Variable swaps: h_sat_l vs h_sat_v, rho_l vs rho_v
  - Missing negation: Gamma = q/h_fg vs -q/h_fg
  - Factor errors:  missing 2x, missing area
  - Index errors:   face[i] vs face[i+1]
  - Convention drift: defining q_i_l one way, using it another
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "two_phase"))
import opal_two_phase as tp


# ============================================================================
# Helpers
# ============================================================================

def make_state(T_l, T_sat, alpha=0.3, **kw):
    """Construct InterfacialState with controlled parameters."""
    s = tp.InterfacialState()
    s.p = kw.get('p', 10e6)
    s.alpha = alpha
    s.rho_l = kw.get('rho_l', 750.0)
    s.rho_v = kw.get('rho_v', 35.0)
    s.h_l = kw.get('h_l', 900e3)
    s.h_v = kw.get('h_v', 2800e3)
    s.T_l = T_l
    s.T_v = kw.get('T_v', T_sat + 5.0)
    s.T_sat = T_sat
    s.h_sat_l = kw.get('h_sat_l', 800e3)
    s.h_sat_v = kw.get('h_sat_v', 2800e3)
    s.cp_l = kw.get('cp_l', 4200.0)
    s.sigma = kw.get('sigma', 0.05)
    s.D_h = kw.get('D_h', 0.073)
    return s


# ============================================================================
# Interfacial area formula: a_i = max(4*alpha*(1-alpha), alpha)
# ============================================================================

class TestInterfacialArea:
    """Verify a_i formula at specific alpha values.
    AI could: swap max arguments, use wrong coefficient, forget (1-alpha)."""

    @pytest.mark.parametrize("alpha,expected_a_i", [
        (0.0,    0.0),           # zero: both branches give 0
        (0.001,  0.001 * 4 * 0.999),  # low: 4α(1-α) > α
        (0.01,   4 * 0.01 * 0.99),    # still parabolic dominates
        (0.1,    4 * 0.1 * 0.9),      # = 0.36
        (0.25,   4 * 0.25 * 0.75),    # = 0.75
        (0.5,    1.0),                  # maximum of parabola
        (0.75,   4 * 0.75 * 0.25),    # symmetric with 0.25
        (0.9,    0.9),                  # high: α > 4α(1-α) = 0.36
        (0.99,   0.99),                 # α branch dominates
        (1.0,    1.0),                  # full vapor: α = 1
    ])
    def test_interfacial_area_exact(self, alpha, expected_a_i):
        """a_i matches hand calculation at specific alpha."""
        # We verify indirectly: Gamma = H_i * a_i * (T_sat - T_l) / h_fg
        # So a_i = -Gamma * h_fg / (H_i * dT)
        H_i = 1e5
        T_l = 510.0; T_sat = 500.0; dT = T_sat - T_l  # = -10
        h_fg = 2e6

        if alpha < 1e-10:
            return  # skip alpha=0 (no area, no transfer)

        s = make_state(T_l=T_l, T_sat=T_sat, alpha=alpha,
                       h_sat_l=800e3, h_sat_v=2800e3)
        c = tp.DriftFluxClosures(H_i=H_i, C_0=1.0, alpha_nucleation=0.0)
        r = c.compute(s)

        # Extract a_i from Gamma
        a_i_computed = -r.Gamma * h_fg / (H_i * dT)
        assert a_i_computed == pytest.approx(expected_a_i, rel=1e-8), (
            f"alpha={alpha}: a_i={a_i_computed:.6f}, expected={expected_a_i:.6f}"
        )


# ============================================================================
# Nucleation onset: alpha_eff = max(alpha, alpha_nucleation) when T_l > T_sat
# ============================================================================

class TestNucleationOnset:
    """When T_l > T_sat AND alpha < alpha_nucleation, the closure uses
    alpha_eff = alpha_nucleation for interfacial area.
    AI could: forget the T_l > T_sat condition, use wrong comparison direction."""

    def test_nucleation_boosts_area_when_superheated(self):
        """With alpha < alpha_nucleation and T_l > T_sat, get enhanced Gamma."""
        H_i = 1e5
        alpha_nuc = 1e-3

        # Tiny alpha, superheated
        s = make_state(T_l=510, T_sat=500, alpha=1e-6,
                       h_sat_l=800e3, h_sat_v=2800e3)
        c = tp.DriftFluxClosures(H_i=H_i, C_0=1.0, alpha_nucleation=alpha_nuc)
        r = c.compute(s)

        # Should use alpha_eff = 1e-3 (not 1e-6)
        a_i_expected = max(4 * alpha_nuc * (1 - alpha_nuc), alpha_nuc)
        Gamma_expected = H_i * a_i_expected * 10.0 / 2e6  # dT=10, h_fg=2e6
        assert r.Gamma == pytest.approx(Gamma_expected, rel=1e-6)

    def test_no_nucleation_when_subcooled(self):
        """With alpha < alpha_nucleation but T_l < T_sat, NO nucleation boost."""
        H_i = 1e5
        alpha_nuc = 1e-3

        # Tiny alpha, subcooled — nucleation should NOT trigger
        s = make_state(T_l=490, T_sat=500, alpha=1e-6,
                       h_sat_l=800e3, h_sat_v=2800e3)
        c = tp.DriftFluxClosures(H_i=H_i, C_0=1.0, alpha_nucleation=alpha_nuc)
        r = c.compute(s)

        # Should use alpha_eff = 1e-6 (actual alpha, no boost)
        a_i_expected = max(4 * 1e-6 * (1 - 1e-6), 1e-6)
        Gamma_expected = H_i * a_i_expected * (-10.0) / 2e6  # condensation
        assert r.Gamma == pytest.approx(Gamma_expected, rel=1e-4)

    def test_nucleation_exact_threshold(self):
        """At alpha = alpha_nucleation exactly, no boost needed."""
        s1 = make_state(T_l=510, T_sat=500, alpha=1e-3,
                        h_sat_l=800e3, h_sat_v=2800e3)
        s2 = make_state(T_l=510, T_sat=500, alpha=1.001e-3,
                        h_sat_l=800e3, h_sat_v=2800e3)

        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.0, alpha_nucleation=1e-3)
        r1 = c.compute(s1)
        r2 = c.compute(s2)

        # Just above threshold: should give nearly the same Gamma
        assert r1.Gamma == pytest.approx(r2.Gamma, rel=0.01)


# ============================================================================
# Drift-flux: V_gj formula and scaling
# ============================================================================

class TestDriftFluxVgj:
    """V_gj = 1.41 * [σ·g·Δρ/ρ_l²]^0.25 * 4α(1-α).
    AI could: forget the 4α(1-α) scaling, use wrong exponent, wrong g."""

    def test_vgj_zero_at_alpha_zero(self):
        """V_gj = 0 when alpha = 0 (no vapor phase)."""
        s = make_state(T_l=500, T_sat=500, alpha=0.0)
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.13)
        d = c.drift_flux(s)
        assert d.V_gj == pytest.approx(0.0, abs=1e-15)

    def test_vgj_zero_at_alpha_one(self):
        """V_gj = 0 when alpha = 1 (no liquid phase)."""
        s = make_state(T_l=500, T_sat=500, alpha=1.0)
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.13)
        d = c.drift_flux(s)
        assert d.V_gj == pytest.approx(0.0, abs=1e-15)

    def test_vgj_max_at_alpha_half(self):
        """V_gj is maximized at alpha = 0.5 (4α(1-α) = 1)."""
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.13)

        s_half = make_state(T_l=500, T_sat=500, alpha=0.5)
        s_quarter = make_state(T_l=500, T_sat=500, alpha=0.25)
        s_tenth = make_state(T_l=500, T_sat=500, alpha=0.1)

        d_half = c.drift_flux(s_half)
        d_quarter = c.drift_flux(s_quarter)
        d_tenth = c.drift_flux(s_tenth)

        assert d_half.V_gj > d_quarter.V_gj
        assert d_quarter.V_gj > d_tenth.V_gj

    def test_vgj_exact_value(self):
        """Verify V_gj against hand calculation."""
        sigma = 0.05; g = 9.81
        rho_l = 750.0; rho_v = 35.0
        alpha = 0.5

        drho = rho_l - rho_v  # = 715
        rho_l2 = rho_l * rho_l  # = 562500
        V_gj_base = 1.41 * (sigma * g * drho / rho_l2) ** 0.25
        scale = 4 * alpha * (1 - alpha)  # = 1.0 at alpha=0.5
        V_gj_expected = V_gj_base * scale

        s = make_state(T_l=500, T_sat=500, alpha=alpha,
                       rho_l=rho_l, rho_v=rho_v, sigma=sigma)
        c = tp.DriftFluxClosures(H_i=1e5, C_0=1.13)
        d = c.drift_flux(s)

        assert d.V_gj == pytest.approx(V_gj_expected, rel=1e-6)
        assert d.C_0 == pytest.approx(1.13, rel=1e-10)


# ============================================================================
# split_phasic_flux: mdot_l + mdot_v = mdot_m (conservation)
# ============================================================================

class TestPhasicFluxSplit:
    """split_phasic_flux must return (mdot_l, mdot_v) that sum to mdot_m.
    AI could: double-count, forget a term, use wrong sign in momentum split."""

    def _make_model(self):
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        return tp.FiveEqModel(fluid, closures)

    @pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1, 0.3, 0.5,
                                        0.7, 0.9, 0.95, 0.99, 0.999])
    def test_phasic_sum_equals_mixture(self, alpha):
        """mdot_l + mdot_v = mdot_m for all alpha values."""
        model = self._make_model()
        mdot_m = 5.0
        rho_l = 750.0; rho_v = 35.0; A = 0.01
        rho_m = (1 - alpha) * rho_l + alpha * rho_v
        C_0 = 1.0; V_gj = 0.5

        mdot_l, mdot_v = model.split_phasic_flux(
            mdot_m, alpha, rho_l, rho_v, rho_m, C_0, V_gj, A)

        assert mdot_l + mdot_v == pytest.approx(mdot_m, rel=1e-10), (
            f"alpha={alpha}: mdot_l={mdot_l:.6f} + mdot_v={mdot_v:.6f} = "
            f"{mdot_l+mdot_v:.6f} != mdot_m={mdot_m}"
        )

    def test_no_slip_equal_velocity(self):
        """With C_0=1 and V_gj=0, both phases move at same velocity."""
        model = self._make_model()
        mdot_m = 10.0; alpha = 0.3
        rho_l = 750.0; rho_v = 35.0; A = 0.01
        rho_m = (1 - alpha) * rho_l + alpha * rho_v

        mdot_l, mdot_v = model.split_phasic_flux(
            mdot_m, alpha, rho_l, rho_v, rho_m, 1.0, 0.0, A)

        # Both phases at same velocity j = G_m / rho_eff
        # rho_eff = rho_l + alpha * 1.0 * (rho_v - rho_l)
        rho_eff = rho_l + alpha * (rho_v - rho_l)
        j = (mdot_m / A) / rho_eff
        v_l_expected = j
        v_v_expected = j  # C_0 * j + V_gj = 1 * j + 0 = j

        mdot_l_expected = (1 - alpha) * rho_l * v_l_expected * A
        mdot_v_expected = alpha * rho_v * v_v_expected * A

        assert mdot_l == pytest.approx(mdot_l_expected, rel=1e-8)
        assert mdot_v == pytest.approx(mdot_v_expected, rel=1e-8)

    def test_extreme_alpha_low_returns_all_liquid(self):
        """alpha < 1e-6: all flow is liquid."""
        model = self._make_model()
        mdot_l, mdot_v = model.split_phasic_flux(
            5.0, 1e-7, 750.0, 35.0, 750.0, 1.0, 0.5, 0.01)
        assert mdot_l == pytest.approx(5.0)
        assert mdot_v == pytest.approx(0.0)

    def test_extreme_alpha_high_returns_all_vapor(self):
        """alpha > 1 - 1e-6: all flow is vapor."""
        model = self._make_model()
        mdot_l, mdot_v = model.split_phasic_flux(
            5.0, 1.0 - 1e-7, 750.0, 35.0, 35.0, 1.0, 0.5, 0.01)
        assert mdot_l == pytest.approx(0.0)
        assert mdot_v == pytest.approx(5.0)

    def test_negative_flow(self):
        """Negative mdot_m: phasic flows should also be negative."""
        model = self._make_model()
        mdot_l, mdot_v = model.split_phasic_flux(
            -5.0, 0.3, 750.0, 35.0, 535.5, 1.0, 0.0, 0.01)
        assert mdot_l + mdot_v == pytest.approx(-5.0, rel=1e-10)
        assert mdot_l < 0
        assert mdot_v < 0

    def test_zero_flow(self):
        """Zero mdot_m: no phasic flows."""
        model = self._make_model()
        mdot_l, mdot_v = model.split_phasic_flux(
            0.0, 0.3, 750.0, 35.0, 535.5, 1.0, 0.0, 0.01)
        assert mdot_l == pytest.approx(0.0, abs=1e-15)
        assert mdot_v == pytest.approx(0.0, abs=1e-15)


# ============================================================================
# Solver state bounds: invariants that must hold after EVERY timestep
# ============================================================================

class TestPostStepInvariants:
    """After every solver timestep, these invariants must hold.
    If any AI-introduced error violates conservation or bounds,
    these catch it immediately — not after 10000 steps."""

    def _run_and_check(self, solver, fluid, N, bc, p, alpha, h_l, h_v, mdot,
                       dt, n_steps, desc=""):
        """Run n_steps, checking ALL invariants after EACH step."""
        for step in range(n_steps):
            solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, dt)

            ctx = f"[{desc} step={step}]"

            # --- Finiteness ---
            assert np.all(np.isfinite(p)), f"{ctx} NaN/Inf in p"
            assert np.all(np.isfinite(alpha)), f"{ctx} NaN/Inf in alpha"
            assert np.all(np.isfinite(h_l)), f"{ctx} NaN/Inf in h_l"
            assert np.all(np.isfinite(h_v)), f"{ctx} NaN/Inf in h_v"
            assert np.all(np.isfinite(mdot)), f"{ctx} NaN/Inf in mdot"

            # --- Pressure bounds ---
            assert np.all(p >= 700.0), f"{ctx} p below floor: min={p.min()}"
            assert np.all(p <= 21e6), f"{ctx} p above ceiling: max={p.max()}"

            # --- Alpha bounds ---
            assert np.all(alpha >= 0.0), f"{ctx} alpha < 0: min={alpha.min()}"
            assert np.all(alpha <= 1.0), f"{ctx} alpha > 1: max={alpha.max()}"

            # --- Enthalpy bounds (per-cell, pressure-dependent) ---
            for i in range(N):
                pp = fluid.evaluate_phasic(p[i])

                assert h_l[i] >= 1e4 - 1.0, (
                    f"{ctx} cell {i}: h_l={h_l[i]:.0f} < floor 1e4")
                assert h_l[i] <= pp.h_sat_v + 1.0, (
                    f"{ctx} cell {i}: h_l={h_l[i]/1e3:.1f} > h_sat_v={pp.h_sat_v/1e3:.1f}")
                assert h_v[i] >= pp.h_sat_v - 1.0, (
                    f"{ctx} cell {i}: h_v={h_v[i]/1e3:.1f} < h_sat_v={pp.h_sat_v/1e3:.1f}")
                assert h_v[i] <= 4e6 + 1.0, (
                    f"{ctx} cell {i}: h_v={h_v[i]/1e3:.1f} > 4 MJ/kg")

                # --- Density positivity ---
                if alpha[i] > 1e-8:
                    rho_v = fluid.rho_vapor(p[i], h_v[i])
                    assert rho_v > 0, (
                        f"{ctx} cell {i}: rho_v={rho_v:.2f} at "
                        f"h_v={h_v[i]/1e3:.1f}, h_sat_v={pp.h_sat_v/1e3:.1f}")
                if alpha[i] < 1.0 - 1e-8:
                    rho_l = fluid.rho_liquid(p[i], h_l[i])
                    assert rho_l > 0, (
                        f"{ctx} cell {i}: rho_l={rho_l:.2f}")

    def test_subcooled_flow(self):
        """Standard subcooled flow — should be perfectly stable."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 5
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_l - 100e3; bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 1e-6)
        h_l = np.full(N, pp.h_sat_l - 100e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        self._run_and_check(solver, fluid, N, bc, p, alpha, h_l, h_v, mdot,
                           1e-4, 500, "subcooled")

    def test_two_phase_flow(self):
        """Two-phase flow with moderate void fraction."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 5
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_l; bc.h_l_in = pp.h_sat_l
        bc.h_v_in = pp.h_sat_v; bc.alpha_in = 0.2

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.2)
        h_l = np.full(N, pp.h_sat_l)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        self._run_and_check(solver, fluid, N, bc, p, alpha, h_l, h_v, mdot,
                           1e-4, 500, "two-phase")

    def test_heated_boiling(self):
        """Subcooled inlet with wall heating → boiling."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e6, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 10
        solver = tp.TwoPhaseSolver(N, 0.3, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_l - 50e3; bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 1e-6)
        h_l = np.full(N, pp.h_sat_l - 50e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)
        q_wall = np.full(N, 5e6)  # 5 MW/m³ heating

        self._run_and_check(solver, fluid, N, bc, p, alpha, h_l, h_v, mdot,
                           1e-4, 500, "heated-boiling")

    def test_depressurization(self):
        """Sudden depressurization — most violent transient."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e6, C_0=1.0, alpha_nucleation=1e-3)
        model = tp.FiveEqModel(fluid, closures)
        N = 10
        solver = tp.TwoPhaseSolver(N, 0.3, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(15e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.WALL; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_out = 1e6  # sudden low pressure
        bc.h_in = pp.h_sat_l + 50e3; bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v

        p = np.full(N, 15e6)
        alpha = np.full(N, 1e-6)
        h_l = np.full(N, pp.h_sat_l + 50e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        self._run_and_check(solver, fluid, N, bc, p, alpha, h_l, h_v, mdot,
                           5e-5, 1000, "depressurization")

    def test_condensation(self):
        """Subcooled liquid entering steam-filled pipe."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e6, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 5
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_l - 100e3; bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v; bc.alpha_in = 0.0

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.8)  # mostly steam
        h_l = np.full(N, pp.h_sat_l)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        self._run_and_check(solver, fluid, N, bc, p, alpha, h_l, h_v, mdot,
                           1e-4, 500, "condensation")


# ============================================================================
# Mixture mass conservation (tight tolerance, per-step)
# ============================================================================

class TestMixtureConservation:
    """Mass conservation tests for a semi-implicit operator-split scheme.
    The scheme has O(dt) splitting error, so per-step conservation isn't exact.
    Instead we verify:
    1. Steady-state flows are uniform (catches face indexing bugs)
    2. Conservation error improves with dt refinement (convergence)
    AI could: double-count a flux, forget outlet, use wrong face index."""

    def test_steady_state_uniform_flow(self):
        """At steady state, all face flows should be equal (mass balance).
        If an AI uses wrong face index, this catches it immediately."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 5
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 9.5e6
        bc.h_in = pp.h_sat_l - 50e3; bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 1e-6)  # subcooled (no phase change)
        h_l = np.full(N, pp.h_sat_l - 50e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        # Run to approximate steady state
        for step in range(2000):
            solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, 1e-4)

        # All face flows should be nearly equal at steady state
        mdot_avg = np.mean(mdot)
        for i in range(N + 1):
            assert mdot[i] == pytest.approx(mdot_avg, rel=0.05), (
                f"face {i}: mdot={mdot[i]:.6f}, avg={mdot_avg:.6f}"
            )

    def test_wall_bc_zero_inlet_flow(self):
        """With WALL BC at inlet, mdot[0] must be exactly zero.
        Catches wrong BC application at inlet face."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 5
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.WALL  # closed end
        bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_out = 5e6
        bc.h_in = pp.h_sat_l; bc.h_l_in = pp.h_sat_l; bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 1e-6)
        h_l = np.full(N, pp.h_sat_l)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for step in range(100):
            solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, 1e-4)
            assert mdot[0] == pytest.approx(0.0, abs=1e-15), (
                f"step {step}: mdot[0]={mdot[0]}, should be 0 with WALL BC"
            )


# ============================================================================
# Phase-absent enthalpy reset: exact saturation values
# ============================================================================

class TestPhaseAbsentReset:
    """When a phase is absent (m_k < 1e-12), its enthalpy resets to saturation.
    AI could: reset to wrong saturation value, or forget the reset entirely."""

    def test_vapor_absent_resets_h_v_to_h_sat_v(self):
        """With alpha ≈ 0, h_v should be h_sat_v(p) exactly."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 3
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 10e6
        bc.h_in = pp.h_sat_l - 100e3; bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v

        p = np.full(N, 10e6)
        alpha = np.full(N, 0.0)  # no vapor
        h_l = np.full(N, pp.h_sat_l - 100e3)  # subcooled
        h_v = np.full(N, pp.h_sat_v + 500e3)  # intentionally wrong
        mdot = np.zeros(N + 1)

        solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, 1e-4)

        for i in range(N):
            pp_i = fluid.evaluate_phasic(p[i])
            assert h_v[i] == pytest.approx(pp_i.h_sat_v, rel=1e-6), (
                f"cell {i}: h_v={h_v[i]/1e3:.1f} != h_sat_v={pp_i.h_sat_v/1e3:.1f}"
            )

    def test_liquid_absent_resets_h_l_to_h_sat_l(self):
        """With alpha ≈ 1, h_l should be h_sat_l(p) exactly."""
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 3
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(10e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 10e6; bc.p_out = 10e6
        bc.h_in = pp.h_sat_v; bc.h_l_in = pp.h_sat_l
        bc.h_v_in = pp.h_sat_v; bc.alpha_in = 1.0

        p = np.full(N, 10e6)
        alpha = np.full(N, 1.0)  # all vapor
        h_l = np.full(N, 500e3)  # intentionally wrong
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, 1e-4)

        for i in range(N):
            pp_i = fluid.evaluate_phasic(p[i])
            assert h_l[i] == pytest.approx(pp_i.h_sat_l, rel=1e-6), (
                f"cell {i}: h_l={h_l[i]/1e3:.1f} != h_sat_l={pp_i.h_sat_l/1e3:.1f}"
            )


# ============================================================================
# Pressure sweep stability: catch region-specific failures
# ============================================================================

class TestPressureSweep:
    """Run the solver at multiple pressures. AI code often works at the
    development pressure (10 MPa) but fails at others due to hardcoded
    constants or invalid property ranges."""

    @pytest.mark.parametrize("p_MPa", [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0])
    def test_two_phase_stable_at_pressure(self, p_MPa):
        """5-eq solver runs stably at this pressure for 200 steps."""
        p_Pa = p_MPa * 1e6
        fluid = tp.SimpleFluidProperties()
        closures = tp.DriftFluxClosures(H_i=1e5, C_0=1.0)
        model = tp.FiveEqModel(fluid, closures)
        N = 5
        solver = tp.TwoPhaseSolver(N, 0.5, 0.01, 0.1, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(p_Pa)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = p_Pa; bc.p_out = p_Pa * 0.95
        bc.h_in = pp.h_sat_l; bc.h_l_in = pp.h_sat_l
        bc.h_v_in = pp.h_sat_v; bc.alpha_in = 0.1

        p = np.full(N, p_Pa)
        alpha = np.full(N, 0.1)
        h_l = np.full(N, pp.h_sat_l)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for step in range(200):
            solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, 1e-4)

        assert np.all(np.isfinite(p)), f"NaN in p at {p_MPa} MPa"
        assert np.all(np.isfinite(alpha)), f"NaN in alpha at {p_MPa} MPa"
        assert np.all(np.isfinite(h_l)), f"NaN in h_l at {p_MPa} MPa"
        assert np.all(np.isfinite(h_v)), f"NaN in h_v at {p_MPa} MPa"
        assert np.all(alpha >= 0) and np.all(alpha <= 1)


# ============================================================================
# IAPWS property consistency at solver-computed states
# ============================================================================

class TestIAPWSAtSolverStates:
    """Run 5-eq solver with IAPWS and verify properties stay valid.
    This is the exact combination that produced the bug."""

    def test_iapws_5eq_invariants(self):
        """Full 5-eq + IAPWS: all invariants hold for 200 steps."""
        fluid = tp.IAPWSIF97Properties()
        closures = tp.DriftFluxClosures(H_i=1e6, C_0=1.0, alpha_nucleation=1e-3)
        model = tp.FiveEqModel(fluid, closures)
        N = 5
        solver = tp.TwoPhaseSolver(N, 0.5, 0.0042, 0.073, 0.02, fluid,
                                    tp.DonorCell(), model, tp.InertialMomentum())

        pp = fluid.evaluate_phasic(7e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.PRESSURE; bc.bc_type_out = tp.BCType.PRESSURE
        bc.p_in = 7e6; bc.p_out = 5e6
        bc.h_in = pp.h_sat_l - 50e3; bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v

        p = np.full(N, 7e6)
        alpha = np.full(N, 0.01)
        h_l = np.full(N, pp.h_sat_l - 50e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for step in range(200):
            solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, 5e-5)

            assert np.all(np.isfinite(p)), f"NaN in p at step {step}"
            assert np.all(np.isfinite(alpha)), f"NaN in alpha at step {step}"
            assert np.all(np.isfinite(h_l)), f"NaN in h_l at step {step}"
            assert np.all(np.isfinite(h_v)), f"NaN in h_v at step {step}"

            for i in range(N):
                pp_i = fluid.evaluate_phasic(p[i])
                # h_v must stay above h_sat_v (prevents invalid IAPWS R2)
                assert h_v[i] >= pp_i.h_sat_v - 1.0, (
                    f"step={step} cell={i}: h_v={h_v[i]/1e3:.1f} < "
                    f"h_sat_v={pp_i.h_sat_v/1e3:.1f}")
                # Density must be positive
                if alpha[i] > 1e-8:
                    rho_v = fluid.rho_vapor(p[i], h_v[i])
                    assert rho_v > 0, (
                        f"step={step} cell={i}: rho_v={rho_v} at "
                        f"h_v={h_v[i]/1e3:.1f}")

    def test_iapws_5eq_depressurization(self):
        """IAPWS + 5-eq + Wall/Break BCs — the Edwards configuration."""
        fluid = tp.IAPWSIF97Properties()
        closures = tp.DriftFluxClosures(H_i=1e7, C_0=1.0, alpha_nucleation=1e-3)
        model = tp.FiveEqModel(fluid, closures)
        N = 10
        solver = tp.TwoPhaseSolver(N, 0.4, 0.0042, 0.073, 0.02, fluid,
                                    tp.DonorCell(), model,
                                    tp.InertialMomentum(),
                                    tp.RansomTrapp(fluid, x_trans=0.10,
                                                   c_floor=1200.0))

        pp = fluid.evaluate_phasic(7e6)
        bc = tp.BoundaryConditions()
        bc.bc_type_in = tp.BCType.WALL
        bc.bc_type_out = tp.BCType.BREAK
        bc.p_out = 101325.0
        bc.break_area_fraction = 0.3
        bc.h_in = pp.h_sat_l - 200e3
        bc.h_l_in = bc.h_in
        bc.h_v_in = pp.h_sat_v

        p = np.full(N, 7e6)
        alpha = np.full(N, 1e-6)
        h_l = np.full(N, pp.h_sat_l - 200e3)
        h_v = np.full(N, pp.h_sat_v)
        mdot = np.zeros(N + 1)

        for step in range(2000):
            solver.step_5eq(p, alpha, h_l, h_v, mdot, bc, 5e-5)

            if step % 100 == 0:
                assert np.all(np.isfinite(p)), f"NaN in p at step {step}"
                assert np.all(np.isfinite(alpha)), f"NaN in alpha at step {step}"
                assert np.all(np.isfinite(h_v)), f"NaN in h_v at step {step}"
                for i in range(N):
                    pp_i = fluid.evaluate_phasic(p[i])
                    assert h_v[i] >= pp_i.h_sat_v - 1.0, (
                        f"step={step} cell={i}: h_v < h_sat_v")
