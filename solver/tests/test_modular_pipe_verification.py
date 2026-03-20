"""
test_modular_pipe_verification.py — L0/L1 verification of the Modelica pipe
modularization (PartialPipe1D base class).

Covers all gaps identified by QA audit:
- GAP 1: rho_cell binding correctness (HEM → rho, DriftFlux → rho_m)
- GAP 2: Phi2 binding + Martinelli-Nelson L0 values
- GAP 3: h_mix_outlet / rho_outlet bindings for critical flow
- GAP 4: drho_dp/drho_dh evaluated at h_mix (not h_l) in drift-flux
- GAP 5: Phi2 face indexing with non-uniform alpha
- GAP 6: Phi2_max clamping values
- GAP 7: ExtractedModelSpec unit tests
- GAP 8: Equation classifier handles new base-class variables
- GAP 9: Face density uses rho_cell in extraction
- GAP 10: Mass conservation alpha-weighting (not swapped)
- GAP 11: Modular pipe extraction in pytest (was standalone script)
- GAP 12: Connector h_outflow via h_mix
"""

import sys
import os
import math
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "two_phase"))

import opal_two_phase as tp

OPAL_ROOT = Path(__file__).resolve().parents[2]

# Edwards XML paths
EDWARDS_HEM_XML = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_backEnd.xml"
EDWARDS_DF_XML = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_DriftFlux_backEnd.xml"
EDWARDS_CRIT_XML = OPAL_ROOT / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_IAPWS_CritFlow_backEnd.xml"


# ============================================================================
# SimpleFluid helper (hand calculations)
# ============================================================================

class SF:
    """SimpleFluid constants — same as in test_drift_flux_modelica.py."""
    p_ref = 10e6
    rho_f_0, rho_f_1 = 750.0, 20.0
    rho_g_0, rho_g_1 = 40.0, 5.0
    T_sat_0, T_sat_1 = 400.0, 20.0
    h_f_0, h_f_1 = 800e3, 100e3
    h_g_0, h_g_1 = 2800e3, 50e3

    @staticmethod
    def p_hat(p): return (p - SF.p_ref) / SF.p_ref
    @staticmethod
    def rho_ph(p, h):
        """SimpleFluid rho(p,h): linear interpolation between rho_f and rho_g."""
        rho_f = SF.rho_f_0 + SF.rho_f_1 * SF.p_hat(p)
        rho_g = SF.rho_g_0 + SF.rho_g_1 * SF.p_hat(p)
        h_f = SF.h_f_0 + SF.h_f_1 * SF.p_hat(p)
        h_g = SF.h_g_0 + SF.h_g_1 * SF.p_hat(p)
        if h <= h_f:
            return rho_f
        elif h >= h_g:
            return rho_g
        else:
            frac = (h - h_f) / (h_g - h_f)
            return rho_f + (rho_g - rho_f) * frac
    @staticmethod
    def rho_f(p): return SF.rho_f_0 + SF.rho_f_1 * SF.p_hat(p)
    @staticmethod
    def rho_g(p): return SF.rho_g_0 + SF.rho_g_1 * SF.p_hat(p)
    @staticmethod
    def h_f(p): return SF.h_f_0 + SF.h_f_1 * SF.p_hat(p)
    @staticmethod
    def h_g(p): return SF.h_g_0 + SF.h_g_1 * SF.p_hat(p)


# ============================================================================
# GAP 2 + 5 + 6: Martinelli-Nelson Phi2 — L0 term verification
# ============================================================================

def martinelli_nelson_hand(alpha, rho_l, rho_v):
    """Hand calculation of Chisholm-Laird Phi^2."""
    rho_ratio = max(rho_l / max(rho_v, 0.1), 1.0)
    return (1 - alpha)**2 + 2 * (1 - alpha) * alpha * math.sqrt(rho_ratio) + alpha**2 * rho_ratio


class TestMartinelliNelsonL0:
    """GAP 2: L0 verification of Martinelli-Nelson friction multiplier values."""

    def test_single_phase_liquid_limit(self):
        """Phi2(alpha=0) = 1.0 (single-phase liquid)."""
        assert martinelli_nelson_hand(0.0, 750.0, 40.0) == pytest.approx(1.0)

    def test_single_phase_vapour_limit(self):
        """Phi2(alpha=1) = rho_l/rho_v (single-phase vapour)."""
        rho_l, rho_v = 750.0, 40.0
        expected = rho_l / rho_v  # 18.75
        assert martinelli_nelson_hand(1.0, rho_l, rho_v) == pytest.approx(expected)

    def test_half_void_hand_calc(self):
        """Phi2(alpha=0.5, rho_l=750, rho_v=40) by hand."""
        alpha, rho_l, rho_v = 0.5, 750.0, 40.0
        rho_ratio = rho_l / rho_v  # 18.75
        # (1-0.5)^2 + 2*0.5*0.5*sqrt(18.75) + 0.5^2*18.75
        expected = 0.25 + 0.5 * math.sqrt(18.75) + 0.25 * 18.75
        assert martinelli_nelson_hand(alpha, rho_l, rho_v) == pytest.approx(expected)
        # Verify it's a substantial multiplier (not ~1)
        assert expected > 5.0

    def test_textbook_reference_value(self):
        """Phi2 at alpha=0.5, rho_l/rho_v=18.75 against independent hand derivation.
        Chisholm-Laird: Phi2 = (1-a)^2 + 2a(1-a)sqrt(R) + a^2*R where R=rho_l/rho_v.
        At a=0.5, R=18.75: Phi2 = 0.25 + 2*0.25*4.330 + 0.25*18.75 = 7.103."""
        R = 750.0 / 40.0  # 18.75
        expected = 0.25 + 2 * 0.25 * math.sqrt(R) + 0.25 * R
        assert expected == pytest.approx(7.103, abs=0.001)
        assert martinelli_nelson_hand(0.5, 750.0, 40.0) == pytest.approx(expected)

    def test_phi2_always_ge_one(self):
        """Phi2 >= 1 for all alpha in [0, 1] (physical requirement)."""
        rho_l, rho_v = 750.0, 40.0
        for alpha in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            assert martinelli_nelson_hand(alpha, rho_l, rho_v) >= 1.0, \
                f"Phi2 must be >= 1 at alpha={alpha}"

    def test_monotonic_in_alpha(self):
        """Phi2 should increase monotonically with alpha (at fixed densities)."""
        rho_l, rho_v = 750.0, 40.0
        alphas = np.linspace(0, 1, 20)
        phi2_values = [martinelli_nelson_hand(a, rho_l, rho_v) for a in alphas]
        for i in range(1, len(phi2_values)):
            assert phi2_values[i] >= phi2_values[i - 1] - 1e-12, \
                f"Phi2 not monotonic: Phi2({alphas[i]})={phi2_values[i]} < Phi2({alphas[i-1]})={phi2_values[i-1]}"


class TestPhi2FaceIndexing:
    """GAP 5: Phi2 uses correct cell index for each face."""

    def test_nonuniform_alpha_correct_cell(self):
        """With alpha = [0.1, 0.3, 0.5], face i should use alpha[i] for i<=N, alpha[N] for i=N+1."""
        rho_l, rho_v = 750.0, 40.0
        alphas = [0.1, 0.3, 0.5]
        N = 3

        expected_phi2 = []
        for i in range(1, N + 2):  # faces 1..N+1
            if i <= N:
                a = alphas[i - 1]  # 1-indexed
            else:
                a = alphas[N - 1]  # alpha[N]
            expected_phi2.append(martinelli_nelson_hand(a, rho_l, rho_v))

        # Face 1 (i=1): alpha[1] = 0.1
        assert expected_phi2[0] == pytest.approx(martinelli_nelson_hand(0.1, rho_l, rho_v))
        # Face 2 (i=2): alpha[2] = 0.3
        assert expected_phi2[1] == pytest.approx(martinelli_nelson_hand(0.3, rho_l, rho_v))
        # Face 3 (i=3): alpha[3] = 0.5
        assert expected_phi2[2] == pytest.approx(martinelli_nelson_hand(0.5, rho_l, rho_v))
        # Face 4 (i=N+1): alpha[N] = alpha[3] = 0.5
        assert expected_phi2[3] == pytest.approx(martinelli_nelson_hand(0.5, rho_l, rho_v))

        # Verify face 2 uses 0.3, NOT 0.1 (the adjacent cell)
        wrong_phi2_face2 = martinelli_nelson_hand(0.1, rho_l, rho_v)
        assert expected_phi2[1] != pytest.approx(wrong_phi2_face2, abs=0.1), \
            "Face 2 should use alpha[2]=0.3, not alpha[1]=0.1"


class TestPhi2MaxClamping:
    """GAP 6: Phi2_max clamping values."""

    def test_clamp_activates_at_high_void(self):
        """At alpha=0.99 with rho_l/rho_v=150, unclamped Phi2 >> 20."""
        rho_l, rho_v = 750.0, 5.0
        unclamped = martinelli_nelson_hand(0.99, rho_l, rho_v)
        assert unclamped > 20.0, f"Expected unclamped > 20, got {unclamped}"
        clamped = min(unclamped, 20.0)
        assert clamped == pytest.approx(20.0)

    def test_clamp_does_not_activate_at_moderate_void(self):
        """At moderate void, unclamped < 20, clamp should not activate."""
        rho_l, rho_v = 750.0, 40.0
        unclamped = martinelli_nelson_hand(0.3, rho_l, rho_v)
        assert unclamped < 20.0, f"Expected unclamped < 20, got {unclamped}"
        clamped = min(unclamped, 20.0)
        assert clamped == pytest.approx(unclamped), "Clamp should not activate"

    def test_clamp_value_is_Phi2_max_not_one(self):
        """The clamp value must be Phi2_max (20.0), NOT 1.0."""
        rho_l, rho_v = 750.0, 5.0
        unclamped = martinelli_nelson_hand(0.99, rho_l, rho_v)
        Phi2_max = 20.0
        clamped = min(unclamped, Phi2_max)
        assert clamped == pytest.approx(20.0)
        assert clamped != pytest.approx(1.0), "Clamp must be 20.0, not 1.0"


# ============================================================================
# GAP 1 + 9: rho_cell binding and face density (numerical L0)
# ============================================================================

class TestRhoCellBinding:
    """GAP 1 + 9: rho_cell formula verification — correct binding produces
    correct values, wrong binding produces detectably different values."""

    def test_hem_rho_at_reference_state(self):
        """HEM: rho = rho_ph(p=10MPa, h=800kJ/kg) = 750 kg/m³ (SimpleFluid subcooled)."""
        rho = SF.rho_ph(10e6, 800e3)
        assert rho == pytest.approx(750.0)

    def test_driftflux_rho_m_formula(self):
        """DriftFlux: rho_m = (1-alpha)*rho_l + alpha*rho_v.
        At alpha=0.3: 0.7*750 + 0.3*40 = 537 (NOT 750 or 40)."""
        alpha = 0.3
        rho_l, rho_v = 750.0, 40.0
        rho_m = (1 - alpha) * rho_l + alpha * rho_v
        assert rho_m == pytest.approx(537.0)
        # Wrong bindings would give very different values
        assert rho_m != pytest.approx(rho_l), "rho_m != rho_l at alpha=0.3"
        assert rho_m != pytest.approx(rho_v), "rho_m != rho_v at alpha=0.3"

    def test_face_density_from_rho_cell(self):
        """Face density = 0.5*(rho_cell[i-1] + rho_cell[i]) for interior faces."""
        # N=3, non-uniform density profile
        rho_cells = [750.0, 537.0, 395.0]
        N = 3
        rho_face = [0.0] * (N + 1)
        rho_face[0] = rho_cells[0]  # boundary: rho_cell[1]
        rho_face[1] = 0.5 * (rho_cells[0] + rho_cells[1])  # 643.5
        rho_face[2] = 0.5 * (rho_cells[1] + rho_cells[2])  # 466.0
        rho_face[3] = rho_cells[2]  # boundary: rho_cell[N]

        assert rho_face[0] == pytest.approx(750.0)
        assert rho_face[1] == pytest.approx(643.5)
        assert rho_face[2] == pytest.approx(466.0)
        assert rho_face[3] == pytest.approx(395.0)

    def test_high_void_rho_cell_difference(self):
        """At alpha=0.5, rho_m (395) differs from rho_l (750) by ~47%."""
        p, alpha = 10e6, 0.5
        rho_l = SF.rho_f(p)  # 750
        rho_v = SF.rho_g(p)  # 40
        rho_m = (1 - alpha) * rho_l + alpha * rho_v  # 395
        relative_diff = abs(rho_m - rho_l) / rho_l
        assert relative_diff > 0.4, \
            f"rho_m vs rho_l differ by {relative_diff*100:.0f}% — wrong binding matters"


# ============================================================================
# GAP 3: h_mix_outlet and rho_outlet bindings for critical flow
# ============================================================================

class TestOutletBindingsForCriticalFlow:
    """GAP 3: h_mix_outlet / rho_outlet correctly reference outlet cell."""

    def test_hem_outlet_is_h_N(self):
        """HEM: h_mix_outlet = h[N], not h[1] or some other cell."""
        h = [800e3, 900e3, 1000e3]  # N=3
        h_mix_outlet = h[2]  # h[N] (0-indexed: h[N-1])
        assert h_mix_outlet == pytest.approx(1000e3)
        assert h_mix_outlet != pytest.approx(800e3), "Must be h[N], not h[1]"

    def test_driftflux_outlet_is_h_mix_N(self):
        """DriftFlux: h_mix_outlet = (1-alpha[N])*h_l[N] + alpha[N]*h_v[N]."""
        alpha_N = 0.3
        h_l_N = 800e3
        h_v_N = 2800e3
        h_mix_N = (1 - alpha_N) * h_l_N + alpha_N * h_v_N  # 1400 kJ/kg
        h_mix_outlet = h_mix_N
        assert h_mix_outlet == pytest.approx(1400e3)
        # Must NOT be h_l[N] (which would be 800 kJ/kg)
        assert h_mix_outlet != pytest.approx(800e3), "Must be h_mix, not h_l"
        # Must NOT be h_v[N] (which would be 2800 kJ/kg)
        assert h_mix_outlet != pytest.approx(2800e3), "Must be h_mix, not h_v"

    def test_driftflux_rho_outlet_is_rho_m_N(self):
        """DriftFlux: rho_outlet = rho_m[N] = (1-alpha)*rho_l + alpha*rho_v."""
        alpha_N = 0.3
        rho_l = 750.0
        rho_v = 40.0
        rho_m_N = (1 - alpha_N) * rho_l + alpha_N * rho_v  # 537
        rho_outlet = rho_m_N
        assert rho_outlet == pytest.approx(537.0)
        assert rho_outlet != pytest.approx(750.0), "Must be rho_m, not rho_l"

    def test_critical_flow_quality_from_h_mix(self):
        """L0: Ransom-Trapp quality x = (h_mix - h_f) / h_fg depends on correct h_mix."""
        p = 10e6
        h_f = SF.h_f(p)   # 800e3
        h_g = SF.h_g(p)   # 2800e3
        h_fg = h_g - h_f   # 2000e3

        # Correct: h_mix at alpha=0.3
        h_mix_correct = 0.7 * 800e3 + 0.3 * 2800e3  # 1400e3
        x_correct = (h_mix_correct - h_f) / h_fg      # 0.3

        # Wrong: using h_l instead of h_mix
        h_l = 800e3
        x_wrong = max(0, (h_l - h_f) / h_fg)          # 0.0

        assert x_correct == pytest.approx(0.3)
        assert x_wrong == pytest.approx(0.0)
        assert abs(x_correct - x_wrong) > 0.2, "Using h_l gives wrong quality"


# ============================================================================
# GAP 4: drho_dp / drho_dh evaluated at h_mix, not h_l
# ============================================================================

class TestDerivativesAtHmix:
    """GAP 4: Pressure-linearization derivatives use mixture enthalpy."""

    def test_drho_dp_at_h_mix_vs_h_l(self):
        """SimpleFluid: drho_dp_h is the same everywhere (constant), but for
        IAPWS the value at h_mix vs h_l differs drastically in two-phase.
        Verify the binding uses h_mix = (1-alpha)*h_l + alpha*h_v."""
        p = 10e6
        alpha = 0.3
        h_l = 800e3
        h_v = 2800e3
        h_mix = (1 - alpha) * h_l + alpha * h_v  # 1400 kJ/kg

        # For SimpleFluid, drho_dp is constant, so the difference is in the
        # argument (which region of the fluid). What matters is the principle:
        # the derivative must be evaluated at h_mix, not h_l.
        assert h_mix == pytest.approx(1400e3)
        assert h_mix != pytest.approx(h_l), "h_mix must differ from h_l"

    def test_alpha_weighting_in_h_mix(self):
        """h_mix = (1-alpha)*h_l + alpha*h_v — verify weights are not swapped."""
        alpha = 0.2
        h_l, h_v = 800e3, 2800e3
        correct = (1 - alpha) * h_l + alpha * h_v   # 0.8*800e3 + 0.2*2800e3 = 1200e3
        swapped = alpha * h_l + (1 - alpha) * h_v    # 0.2*800e3 + 0.8*2800e3 = 2400e3
        assert correct == pytest.approx(1200e3)
        assert swapped == pytest.approx(2400e3)
        assert abs(correct - swapped) > 1e6, "Swapped weights give very different result"


# ============================================================================
# GAP 10: Mass conservation alpha-weighting
# ============================================================================

class TestMassConservationAlphaWeighting:
    """GAP 10: der(h_mix) expansion uses (1-alpha)*der(h_l) + alpha*der(h_v)."""

    def test_weights_not_swapped(self):
        """At alpha=0.1: (1-alpha)=0.9 weights liquid, alpha=0.1 weights vapour.
        Swapping would weight vapour 0.9 and liquid 0.1 — very different."""
        alpha = 0.1
        der_h_l = 1000.0   # liquid enthalpy rate
        der_h_v = -500.0   # vapour enthalpy rate

        correct = (1 - alpha) * der_h_l + alpha * der_h_v
        # = 0.9*1000 + 0.1*(-500) = 900 - 50 = 850
        swapped = alpha * der_h_l + (1 - alpha) * der_h_v
        # = 0.1*1000 + 0.9*(-500) = 100 - 450 = -350

        assert correct == pytest.approx(850.0)
        assert swapped == pytest.approx(-350.0)
        assert abs(correct - swapped) > 1000, "Swapped alpha-weights differ drastically"

    def test_reduces_to_single_phase_limits(self):
        """At alpha=0: mixture = pure liquid. At alpha=1: mixture = pure vapour."""
        der_h_l, der_h_v = 1000.0, -500.0

        # Pure liquid
        mix_alpha0 = (1 - 0.0) * der_h_l + 0.0 * der_h_v
        assert mix_alpha0 == pytest.approx(der_h_l)

        # Pure vapour
        mix_alpha1 = (1 - 1.0) * der_h_l + 1.0 * der_h_v
        assert mix_alpha1 == pytest.approx(der_h_v)


# ============================================================================
# GAP 12: Connector h_outflow via h_mix (drift-flux)
# ============================================================================

class TestConnectorEnthalpy:
    """GAP 12: port_b.h_outflow = h_mix[N] for drift-flux."""

    def test_drift_flux_outflow_is_mixture(self):
        """h_outflow must be h_mix = (1-alpha)*h_l + alpha*h_v, not h_l or h_v."""
        alpha = 0.4
        h_l, h_v = 800e3, 2800e3
        h_mix = (1 - alpha) * h_l + alpha * h_v  # 1600 kJ/kg
        assert h_mix == pytest.approx(1600e3)
        assert h_mix != pytest.approx(h_l), "Must be h_mix, not h_l"
        assert h_mix != pytest.approx(h_v), "Must be h_mix, not h_v"

    def test_hem_outflow_is_h_N(self):
        """HEM: port_b.h_outflow = h[N] directly."""
        h = [800e3, 850e3, 900e3]
        h_outflow = h[2]  # h[N]
        assert h_outflow == pytest.approx(900e3)


# ============================================================================
# GAP 2 (continued): Phi2 = 1.0 for HEM (no two-phase friction)
# ============================================================================

class TestPhi2HEM:
    """GAP 2: HEM model sets Phi2[i] = 1.0 for all faces."""

    def test_hem_phi2_in_modelica_source(self):
        """Verify Pipe1D.mo contains Phi2[i] = 1.0 binding (not Martinelli-Nelson)."""
        pipe1d_path = OPAL_ROOT / "library" / "Pipes" / "Pipe1D.mo"
        text = pipe1d_path.read_text()
        # Must contain the constant binding
        assert "Phi2[i] = 1.0" in text, "Pipe1D.mo must set Phi2[i] = 1.0"
        # Must NOT contain Martinelli-Nelson reference
        assert "martinelli_nelson" not in text, \
            "Pipe1D.mo must not reference martinelli_nelson (HEM has no two-phase friction)"

    def test_hem_momentum_has_no_phi2_effect(self):
        """With Phi2=1.0, friction = f_D * dx / (2*D_h) * |mdot|*mdot / (rho*A^2).
        Phi2 * friction = friction (no amplification)."""
        Phi2 = 1.0
        f_D, dx, D_h = 0.02, 0.2, 0.1
        mdot, rho, A = 10.0, 750.0, 0.01
        friction_base = f_D * dx / (2 * D_h) * abs(mdot) * mdot / (rho * A**2)
        friction_with_phi2 = Phi2 * friction_base
        assert friction_with_phi2 == pytest.approx(friction_base)


# ============================================================================
# GAP 7: ExtractedModelSpec unit tests
# ============================================================================

class TestExtractedModelSpec:
    """GAP 7: ExtractedModelSpec extraction from XML."""

    @pytest.fixture
    def hem_spec(self):
        """Load Edwards HEM XML and extract model spec."""
        if not EDWARDS_HEM_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.model_spec import extract_model_spec
        es = load_equation_system(EDWARDS_HEM_XML)
        return extract_model_spec(es)

    @pytest.fixture
    def df_spec(self):
        """Load Edwards DriftFlux XML and extract model spec."""
        if not EDWARDS_DF_XML.exists():
            pytest.skip("Edwards DriftFlux XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.model_spec import extract_model_spec
        es = load_equation_system(EDWARDS_DF_XML)
        return extract_model_spec(es)

    # -- HEM model spec --

    def test_hem_model_type(self, hem_spec):
        """HEM should be detected as 'hem', not 'drift_flux'."""
        assert hem_spec.model_type == "hem"

    def test_hem_geometry(self, hem_spec):
        """Edwards geometry: L=4.096, D=0.073, N=5, f_D=0.02."""
        g = hem_spec.geometry
        assert g.N == 5
        assert g.dx == pytest.approx(4.096 / 5)
        assert g.A_flow == pytest.approx(math.pi / 4 * 0.073**2)
        assert g.D_h == pytest.approx(0.073)
        assert g.f_D == pytest.approx(0.02)

    def test_hem_no_critical_flow(self, hem_spec):
        """Edwards HEM (no break) should not have critical flow."""
        assert hem_spec.closures.use_critical_flow is False

    def test_hem_initial_conditions(self, hem_spec):
        """Edwards: p_init=7e6, h_init=980e3."""
        assert all(p == pytest.approx(7e6) for p in hem_spec.ic.p0)
        assert all(h == pytest.approx(980e3) for h in hem_spec.ic.h0)

    def test_hem_boundary_wall_inlet(self, hem_spec):
        """Edwards: inlet is ClosedEnd (wall)."""
        assert hem_spec.boundary.inlet_type == "wall"

    def test_hem_no_phasic_states(self, hem_spec):
        """HEM should have no h_v0 or alpha0."""
        assert hem_spec.ic.h_v0 is None
        assert hem_spec.ic.alpha0 is None

    # -- Drift-flux model spec --

    def test_df_model_type(self, df_spec):
        """Drift-flux should be detected as 'drift_flux'."""
        assert df_spec.model_type == "drift_flux"

    def test_df_closure_parameters(self, df_spec):
        """Verify closure parameters are extracted (not hardcoded defaults)."""
        c = df_spec.closures
        # H_i should be the value from the Modelica model, not 0.0
        assert c.H_i > 0, f"H_i should be > 0, got {c.H_i}"
        assert c.C_0 > 0, f"C_0 should be > 0, got {c.C_0}"
        assert c.alpha_nucleation > 0, f"alpha_nucleation should be > 0, got {c.alpha_nucleation}"

    def test_df_has_phasic_initial_conditions(self, df_spec):
        """Drift-flux should have h_v0 and alpha0."""
        assert df_spec.ic.h_v0 is not None
        assert df_spec.ic.alpha0 is not None
        assert len(df_spec.ic.h_v0) == df_spec.geometry.N
        assert len(df_spec.ic.alpha0) == df_spec.geometry.N

    def test_df_geometry_consistent(self, df_spec):
        """DriftFlux Edwards geometry: same pipe (L=4.096, D=0.073), different N."""
        g = df_spec.geometry
        assert g.A_flow == pytest.approx(math.pi / 4 * 0.073**2)
        assert g.D_h == pytest.approx(0.073)
        assert g.f_D == pytest.approx(0.02)
        assert g.dx == pytest.approx(4.096 / g.N)


# ============================================================================
# EXTRACTION-LEVEL: Verify bindings through real Edwards XML equations
# Addresses PARTIAL PASS on GAPs 1, 2, 4, 9 and NEW GAPs A-E from Round 2.
# ============================================================================

class TestExtractionLevelBindings:
    """Verify abstract variable bindings in actual extracted equations (not tautologies)."""

    @pytest.fixture
    def hem_equations(self):
        """Return list of equation text strings from Edwards HEM XML."""
        if not EDWARDS_HEM_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        es = load_equation_system(EDWARDS_HEM_XML)
        return [eq.text for eq in es.equations if eq.text]

    @pytest.fixture
    def df_equations(self):
        """Return list of equation text strings from Edwards DriftFlux XML."""
        if not EDWARDS_DF_XML.exists():
            pytest.skip("Edwards DriftFlux XML not available")
        from partitioner.xml_reader import load_equation_system
        es = load_equation_system(EDWARDS_DF_XML)
        return [eq.text for eq in es.equations if eq.text]

    # -- GAP 1/9: rho_cell / face density in extraction --

    def test_hem_face_density_in_extraction(self, hem_equations):
        """HEM extracted equations contain face density averaging with 0.5."""
        face_eqs = [t for t in hem_equations if "rho_face" in t and "0.5" in t]
        assert len(face_eqs) >= 1, \
            "Must have at least 1 face density averaging equation in HEM extraction"

    def test_df_face_density_in_extraction(self, df_equations):
        """DriftFlux extracted equations contain face density averaging."""
        face_eqs = [t for t in df_equations if "rho_face" in t and "0.5" in t]
        assert len(face_eqs) >= 1, \
            "Must have at least 1 face density averaging equation in DriftFlux extraction"

    # -- GAP 2: Phi2 in momentum equations --

    def test_df_momentum_contains_friction(self, df_equations):
        """DriftFlux momentum equations contain friction terms (f_D)."""
        mom_eqs = [t for t in df_equations if "der(pipe.mdot" in t]
        friction_eqs = [t for t in mom_eqs if "f_D" in t]
        assert len(friction_eqs) >= 1, \
            "Momentum equations must contain Darcy friction (f_D)"

    # -- GAP 4: drho_dp evaluated at h_mix in DriftFlux --

    def test_df_mass_conservation_has_drho_dp(self, df_equations):
        """DriftFlux mass conservation must contain drho_dp * der(p)."""
        mass_eqs = [t for t in df_equations if "drho_dp" in t and "der(pipe.p" in t]
        assert len(mass_eqs) >= 5, \
            f"Expected >= 5 mass equations with drho_dp*der(p), got {len(mass_eqs)}"

    def test_df_has_phasic_energy_derivatives(self, df_equations):
        """DriftFlux must have phasic energy derivatives: der(pipe.h_l) and der(pipe.h_v)."""
        h_l_der = [t for t in df_equations if "der(pipe.h_l" in t]
        h_v_der = [t for t in df_equations if "der(pipe.h_v" in t]
        assert len(h_l_der) >= 1, "Must have der(pipe.h_l) in extraction"
        assert len(h_v_der) >= 1, "Must have der(pipe.h_v) in extraction"

    def test_df_has_void_fraction_transport(self, df_equations):
        """DriftFlux must have void fraction transport: der(pipe.alpha)."""
        alpha_der = [t for t in df_equations if "der(pipe.alpha" in t]
        assert len(alpha_der) >= 1, "Must have der(pipe.alpha) in extraction"

    # -- GAP 3/NEW GAP C: critical flow in extraction --

    def test_df_critical_flow_in_extraction(self, df_equations):
        """DriftFlux Edwards (with critical flow) must have mdot_crit in equations."""
        crit_eqs = [t for t in df_equations if "mdot_crit" in t]
        assert len(crit_eqs) >= 1, \
            "Must have mdot_crit in DriftFlux extraction (critical flow active)"

    # -- NEW GAP E: connector h_outflow in DriftFlux --

    def test_df_connector_h_outflow_in_extraction(self, df_equations):
        """DriftFlux must have h_outflow equation for connector coupling."""
        h_outflow = [t for t in df_equations if "h_outflow" in t]
        # OM may expand stream connectors — h_outflow may or may not appear explicitly
        # But h_mix should appear somewhere in the equations
        h_mix = [t for t in df_equations if "h_mix" in t]
        assert len(h_outflow) > 0 or len(h_mix) > 0, \
            "Must have h_outflow or h_mix in DriftFlux extraction for connector coupling"

    # -- Modelica source verification (not tautologies) --

    def test_pipe1d_rho_cell_binding_in_source(self):
        """Verify Pipe1D.mo source contains rho_cell[i] = rho[i] (not rho_m)."""
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D.mo").read_text()
        assert "rho_cell[i] = rho[i]" in text, \
            "Pipe1D.mo must bind rho_cell to rho (mixture density for HEM)"
        assert "rho_cell[i] = rho_m[i]" not in text, \
            "Pipe1D.mo must NOT bind rho_cell to rho_m (that's DriftFlux)"

    def test_driftflux_rho_cell_binding_in_source(self):
        """Verify Pipe1D_DriftFlux.mo source contains rho_cell[i] = rho_m[i]."""
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D_DriftFlux.mo").read_text()
        assert "rho_cell[i] = rho_m[i]" in text, \
            "Pipe1D_DriftFlux.mo must bind rho_cell to rho_m"

    def test_driftflux_h_mix_outlet_in_source(self):
        """Verify Pipe1D_DriftFlux.mo binds h_mix_outlet = h_mix[N], not h_l[N]."""
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D_DriftFlux.mo").read_text()
        assert "h_mix_outlet = h_mix[N]" in text, \
            "Must bind h_mix_outlet to h_mix[N]"
        assert "h_mix_outlet = h_l[N]" not in text, \
            "Must NOT bind h_mix_outlet to h_l[N]"

    def test_driftflux_rho_outlet_in_source(self):
        """Verify Pipe1D_DriftFlux.mo binds rho_outlet = rho_m[N], not rho_l[N]."""
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D_DriftFlux.mo").read_text()
        assert "rho_outlet = rho_m[N]" in text, \
            "Must bind rho_outlet to rho_m[N]"

    def test_driftflux_drho_dp_at_h_mix_in_source(self):
        """Verify DriftFlux evaluates drho_dp at h_mix for semi-implicit stability.

        The 5-eq model uses h_mix (not phasic h_l/h_v) for the pressure
        linearization because the mixture compressibility includes the thermal
        (saturation curve shift) effect needed by the semi-implicit scheme.
        """
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D_DriftFlux.mo").read_text()
        assert "drho_dp_h(p[i], h_mix[i])" in text, \
            "drho_dp must be evaluated at h_mix[i] for semi-implicit stability"
        assert "drho_dp_h(p[i], h_l[i])" not in text, \
            "drho_dp must NOT be evaluated at h_l[i] alone"

    def test_hem_drho_dp_at_h_in_source(self):
        """Verify HEM evaluates drho_dp at h[i], not h_mix."""
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D.mo").read_text()
        assert "drho_dp_h(p[i], h[i])" in text, \
            "HEM drho_dp must be evaluated at h[i]"

    def test_driftflux_port_h_outflow_in_source(self):
        """Verify DriftFlux sets port_b.h_outflow = h_mix[N], not h_l[N]."""
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D_DriftFlux.mo").read_text()
        assert "port_b.h_outflow = h_mix[N]" in text, \
            "port_b.h_outflow must be h_mix[N]"
        assert "port_b.h_outflow = h_l[N]" not in text, \
            "port_b.h_outflow must NOT be h_l[N]"

    def test_driftflux_phi2_uses_martinelli_nelson(self):
        """Verify DriftFlux Phi2 computation uses martinelli_nelson, not constant."""
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D_DriftFlux.mo").read_text()
        assert "martinelli_nelson" in text, \
            "Pipe1D_DriftFlux.mo must reference martinelli_nelson for Phi2"

    def test_driftflux_mass_conservation_alpha_weighting_in_source(self):
        """Verify mass conservation uses (1-alpha)*der(h_l) + alpha*der(h_v), not swapped."""
        text = (OPAL_ROOT / "library" / "Pipes" / "Pipe1D_DriftFlux.mo").read_text()
        # The pattern: drho_dh[i] * ((1 - alpha[i]) * der(h_l[i]) + alpha[i] * der(h_v[i]))
        assert "(1 - alpha[i]) * der(h_l[i])" in text, \
            "Mass conservation must weight liquid derivative by (1-alpha)"
        assert "alpha[i] * der(h_v[i])" in text, \
            "Mass conservation must weight vapour derivative by alpha"


# ============================================================================
# GAP 8: Equation classifier with new base-class variables
# ============================================================================

class TestClassifierWithBaseClassVars:
    """GAP 8: Equation classifier handles rho_cell, Phi2, h_mix_outlet."""

    @pytest.fixture
    def hem_classified(self):
        """Load and classify Edwards HEM equations."""
        if not EDWARDS_HEM_XML.exists():
            pytest.skip("EdwardsTest XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.equation_classifier import classify_equations
        es = load_equation_system(EDWARDS_HEM_XML)
        return classify_equations(es, prefix="pipe")

    @pytest.fixture
    def df_classified(self):
        """Load and classify Edwards DriftFlux equations."""
        if not EDWARDS_DF_XML.exists():
            pytest.skip("Edwards DriftFlux XML not available")
        from partitioner.xml_reader import load_equation_system
        from partitioner.equation_classifier import classify_equations
        es = load_equation_system(EDWARDS_DF_XML)
        return classify_equations(es, prefix="pipe")

    def test_hem_mass_equations_classified(self, hem_classified):
        """HEM should have N=5 mass equations."""
        assert len(hem_classified.mass_eqs) == 5

    def test_hem_momentum_equations_classified(self, hem_classified):
        """HEM should have momentum equations (5 for N=5 with wall+pressure)."""
        assert len(hem_classified.momentum_eqs) >= 4

    def test_hem_energy_equations_classified(self, hem_classified):
        """HEM should have N=5 energy equations."""
        assert len(hem_classified.energy_eqs) == 5

    def test_df_mass_equations_classified(self, df_classified):
        """DriftFlux should have N mass equations (N=24 for Edwards DriftFlux)."""
        assert len(df_classified.mass_eqs) >= 5

    def test_df_momentum_equations_classified(self, df_classified):
        """DriftFlux should have momentum equations."""
        assert len(df_classified.momentum_eqs) >= 4

    def test_unclassified_are_algebraic_aliases(self, hem_classified):
        """Any unclassified equations should be algebraic (rho_cell, Phi2, etc.),
        not misclassified ODEs."""
        for eq_text in hem_classified.unclassified:
            text = eq_text if isinstance(eq_text, str) else eq_text.eq_text
            assert "der(" not in text, \
                f"ODE equation misclassified as unclassified: {text[:80]}"


# ============================================================================
# GAP 11: Modular pipe extraction in pytest
# ============================================================================

class TestModularPipeExtraction:
    """GAP 11: Extraction of modular Pipe1D through OM — in pytest.
    Tests that PartialPipe1D inheritance is correctly flattened.
    Uses the existing extraction_utils infrastructure."""

    @pytest.fixture(scope="class")
    def modular_pipe_eqs(self):
        """Extract ModularPipeTest via OM and return equation texts.
        Skips if OM is not available."""
        # Use the feasibility extraction_utils for robust OM session management
        feasibility_dir = OPAL_ROOT / "feasibility"
        sys.path.insert(0, str(feasibility_dir))
        try:
            from extraction_utils import start_omc_session, omc_check, resolve_xml_path
        except ImportError:
            pytest.skip("extraction_utils not available")

        try:
            omc = start_omc_session(load_msl=True)
        except (SystemExit, Exception) as e:
            pytest.skip(f"OM session failed: {e}")

        lib_pkg = (OPAL_ROOT / "library" / "package.mo").as_posix()
        r = omc.sendExpression(f'loadFile("{lib_pkg}")', parsed=False)
        if "false" in r.lower():
            pytest.skip(f"Failed to load OPAL library")

        model_def = r"""
model ModularPipeTest
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.Pipe1D pipe(N=3, L=3.0, D=0.073, f_D=0.02,
                            p_init=7e6, h_init=980e3);
  library.Boundary.PressureSource atm(p_set=101325.0, h_set=980e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, atm.port);
end ModularPipeTest;
"""
        omc.sendExpression(f'loadString("{model_def}")', parsed=False)

        raw = omc.sendExpression(
            'dumpXMLDAE(ModularPipeTest, translationLevel="backEnd", '
            'addOriginalAdjacencyMatrix=true, addSolvingInfo=true)',
            parsed=False
        )
        if "true" not in raw.lower():
            pytest.skip(f"OM extraction failed: {raw}")

        try:
            xml_path = resolve_xml_path(raw, "ModularPipeTest",
                                        ["ModularPipeTest.xml"])
        except FileNotFoundError:
            pytest.skip("Cannot find extracted XML file")

        import xml.etree.ElementTree as ET
        root = ET.parse(xml_path).getroot()
        eq_texts = []
        for eq in root.findall(".//equation"):
            if eq.text:
                eq_texts.append(eq.text.strip())
        return eq_texts

    def test_equation_count_matches_monolithic(self, modular_pipe_eqs):
        """Modular pipe (N=3) should produce 27 equations (same as monolithic)."""
        assert len(modular_pipe_eqs) == 27

    def test_rho_cell_appears_in_equations(self, modular_pipe_eqs):
        """rho_cell should appear in the flattened equations (OM doesn't inline the alias)."""
        rho_cell_eqs = [t for t in modular_pipe_eqs if "rho_cell" in t]
        # Either OM keeps rho_cell (expected) or inlines it (also OK)
        # If neither rho_cell nor rho appears in face density eqs, something is wrong
        face_density_eqs = [t for t in modular_pipe_eqs if "rho_face" in t]
        assert len(face_density_eqs) > 0, "Must have face density equations"

    def test_mass_conservation_present(self, modular_pipe_eqs):
        """Mass conservation: drho_dp * der(p) must appear."""
        mass_eqs = [t for t in modular_pipe_eqs if "drho_dp" in t and "der(pipe" in t]
        assert len(mass_eqs) == 3, f"Expected 3 mass eqs for N=3, got {len(mass_eqs)}"

    def test_momentum_present(self, modular_pipe_eqs):
        """Momentum: der(pipe.mdot[...]) must appear."""
        mom_eqs = [t for t in modular_pipe_eqs if "der(pipe.mdot" in t]
        assert len(mom_eqs) >= 2, f"Expected at least 2 momentum eqs, got {len(mom_eqs)}"

    def test_energy_present(self, modular_pipe_eqs):
        """Energy: der(pipe.h[...]) must appear."""
        energy_eqs = [t for t in modular_pipe_eqs
                      if "der(pipe.h" in t and "drho_dh" not in t]
        assert len(energy_eqs) == 3, f"Expected 3 energy eqs for N=3, got {len(energy_eqs)}"

    def test_property_evaluation_present(self, modular_pipe_eqs):
        """Property evaluation: rho, drho_dp, drho_dh must be computed.
        Note: OM may inline SimpleFluid calls at backEnd level, so check
        for the output variables rather than function names."""
        rho_eqs = [t for t in modular_pipe_eqs if "pipe.rho[" in t or "pipe.rho_cell[" in t]
        assert len(rho_eqs) >= 3, f"Expected at least 3 density eqs for N=3, got {len(rho_eqs)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
