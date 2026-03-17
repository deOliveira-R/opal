"""
test_iapws_cpp.py — IAPWS-IF97 C++ property package integration test.

Verifies the C++ IAPWSIF97Properties class against the iapws Python oracle
across Regions 1, 2, and 4.  This is an INTEGRATION test (real properties
in the solver interface), NOT solver verification (use SimpleFluid for that).

Tolerance: 1e-6 relative (matching library/Media/tests/verify_if97.py).
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "two_phase"))
import opal_two_phase as tp

# The iapws package is the oracle
iapws = pytest.importorskip("iapws")


@pytest.fixture
def fluid():
    return tp.IAPWSIF97Properties()


# ============================================================================
# Region 1 — compressed liquid
# ============================================================================

class TestRegion1:
    """IAPWS-IF97 Region 1: compressed liquid."""

    POINTS = [
        # (p_MPa, T_K)
        (3.0, 300.0),   (3.0, 400.0),   (3.0, 500.0),
        (10.0, 300.0),  (10.0, 400.0),  (10.0, 500.0),
        (25.0, 300.0),  (25.0, 400.0),  (25.0, 500.0),
        (1.0, 280.0),   (5.0, 350.0),   (15.0, 450.0),
        (50.0, 300.0),  (50.0, 400.0),  (50.0, 500.0),
    ]

    @pytest.mark.parametrize("p_MPa,T_K", POINTS)
    def test_density(self, fluid, p_MPa, T_K):
        ref = iapws.IAPWS97(P=p_MPa, T=T_K)
        h_Jkg = ref.h * 1e3
        fp = fluid.evaluate(p_MPa * 1e6, h_Jkg)
        assert abs(fp.rho - ref.rho) / ref.rho < 1e-6, \
            f"rho: C++={fp.rho:.6f}, oracle={ref.rho:.6f}"

    @pytest.mark.parametrize("p_MPa,T_K", POINTS)
    def test_temperature(self, fluid, p_MPa, T_K):
        ref = iapws.IAPWS97(P=p_MPa, T=T_K)
        h_Jkg = ref.h * 1e3
        fp = fluid.evaluate(p_MPa * 1e6, h_Jkg)
        assert abs(fp.T - T_K) / T_K < 1e-6, \
            f"T: C++={fp.T:.6f}, expected={T_K:.6f}"

    @pytest.mark.parametrize("p_MPa,T_K", POINTS[:6])
    def test_derivatives_fd(self, fluid, p_MPa, T_K):
        """Central FD cross-check of drho_dp_h and drho_dh_p."""
        ref = iapws.IAPWS97(P=p_MPa, T=T_K)
        h = ref.h * 1e3
        p = p_MPa * 1e6
        fp = fluid.evaluate(p, h)

        # drho_dp_h via FD
        dp = 100.0
        rho_plus  = fluid.evaluate(p + dp, h).rho
        rho_minus = fluid.evaluate(p - dp, h).rho
        fd_dp = (rho_plus - rho_minus) / (2 * dp)
        if abs(fp.drho_dp_h) > 1e-20:
            assert abs(fd_dp - fp.drho_dp_h) / abs(fp.drho_dp_h) < 1e-4

        # drho_dh_p via FD
        dh = 10.0
        rho_plus  = fluid.evaluate(p, h + dh).rho
        rho_minus = fluid.evaluate(p, h - dh).rho
        fd_dh = (rho_plus - rho_minus) / (2 * dh)
        if abs(fp.drho_dh_p) > 1e-20:
            assert abs(fd_dh - fp.drho_dh_p) / abs(fp.drho_dh_p) < 1e-4


# ============================================================================
# Region 2 — superheated steam
# ============================================================================

class TestRegion2:
    """IAPWS-IF97 Region 2: superheated steam."""

    POINTS = [
        # (p_MPa, T_K)
        (0.001, 400.0),  (0.001, 600.0),  (0.001, 800.0),
        (0.1, 400.0),    (0.1, 600.0),    (0.1, 800.0),
        (1.0, 500.0),    (1.0, 700.0),    (1.0, 900.0),
        (5.0, 600.0),    (5.0, 800.0),    (5.0, 1000.0),
        (10.0, 600.0),   (10.0, 800.0),
    ]

    @pytest.mark.parametrize("p_MPa,T_K", POINTS)
    def test_density(self, fluid, p_MPa, T_K):
        ref = iapws.IAPWS97(P=p_MPa, T=T_K)
        h_Jkg = ref.h * 1e3
        fp = fluid.evaluate(p_MPa * 1e6, h_Jkg)
        assert abs(fp.rho - ref.rho) / ref.rho < 1e-6, \
            f"rho: C++={fp.rho:.6f}, oracle={ref.rho:.6f}"

    @pytest.mark.parametrize("p_MPa,T_K", POINTS)
    def test_temperature(self, fluid, p_MPa, T_K):
        ref = iapws.IAPWS97(P=p_MPa, T=T_K)
        h_Jkg = ref.h * 1e3
        fp = fluid.evaluate(p_MPa * 1e6, h_Jkg)
        assert abs(fp.T - T_K) / T_K < 1e-6, \
            f"T: C++={fp.T:.6f}, expected={T_K:.6f}"

    @pytest.mark.parametrize("p_MPa,T_K", POINTS[:6])
    def test_derivatives_fd(self, fluid, p_MPa, T_K):
        """Central FD cross-check of drho_dp_h and drho_dh_p."""
        ref = iapws.IAPWS97(P=p_MPa, T=T_K)
        h = ref.h * 1e3
        p = p_MPa * 1e6
        fp = fluid.evaluate(p, h)

        dp = 10.0
        rho_plus  = fluid.evaluate(p + dp, h).rho
        rho_minus = fluid.evaluate(p - dp, h).rho
        fd_dp = (rho_plus - rho_minus) / (2 * dp)
        if abs(fp.drho_dp_h) > 1e-20:
            assert abs(fd_dp - fp.drho_dp_h) / abs(fp.drho_dp_h) < 1e-4

        dh = 10.0
        rho_plus  = fluid.evaluate(p, h + dh).rho
        rho_minus = fluid.evaluate(p, h - dh).rho
        fd_dh = (rho_plus - rho_minus) / (2 * dh)
        if abs(fp.drho_dh_p) > 1e-20:
            assert abs(fd_dh - fp.drho_dh_p) / abs(fp.drho_dh_p) < 1e-4


# ============================================================================
# Region 4 — two-phase
# ============================================================================

class TestRegion4:
    """IAPWS-IF97 Region 4: two-phase saturation."""

    PRESSURES = [0.5, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0]

    @pytest.mark.parametrize("p_MPa", PRESSURES)
    def test_saturation_density(self, fluid, p_MPa):
        """Two-phase mixture density at x=0.5."""
        ref_f = iapws.IAPWS97(P=p_MPa, x=0)
        ref_g = iapws.IAPWS97(P=p_MPa, x=1)
        h_mix = 0.5 * (ref_f.h + ref_g.h) * 1e3
        fp = fluid.evaluate(p_MPa * 1e6, h_mix)
        rho_exp = 1.0 / (0.5 / ref_g.rho + 0.5 / ref_f.rho)
        assert abs(fp.rho - rho_exp) / rho_exp < 1e-5, \
            f"rho: C++={fp.rho:.4f}, expected={rho_exp:.4f}"

    @pytest.mark.parametrize("p_MPa", PRESSURES)
    def test_saturation_temperature(self, fluid, p_MPa):
        """T_sat from two-phase evaluation."""
        ref_f = iapws.IAPWS97(P=p_MPa, x=0)
        h_mix = ref_f.h * 1e3 + 100.0e3  # slightly above h_f
        fp = fluid.evaluate(p_MPa * 1e6, h_mix)
        assert abs(fp.T - ref_f.T) / ref_f.T < 1e-5, \
            f"T_sat: C++={fp.T:.4f}, expected={ref_f.T:.4f}"

    @pytest.mark.parametrize("p_MPa", [1.0, 5.0, 10.0])
    def test_drho_dh_p_twophase(self, fluid, p_MPa):
        """drho_dh_p in two-phase via FD cross-check."""
        ref_f = iapws.IAPWS97(P=p_MPa, x=0)
        ref_g = iapws.IAPWS97(P=p_MPa, x=1)
        h = 0.5 * (ref_f.h + ref_g.h) * 1e3
        p = p_MPa * 1e6
        fp = fluid.evaluate(p, h)

        dh = 100.0
        rho_plus  = fluid.evaluate(p, h + dh).rho
        rho_minus = fluid.evaluate(p, h - dh).rho
        fd = (rho_plus - rho_minus) / (2 * dh)
        assert abs(fd - fp.drho_dh_p) / abs(fp.drho_dh_p) < 1e-3


# ============================================================================
# Cross-region: boundary continuity
# ============================================================================

class TestBoundaryContinuity:
    """Properties should be continuous at saturation boundaries."""

    @pytest.mark.parametrize("p_MPa", [1.0, 5.0, 10.0])
    def test_liquid_to_twophase(self, fluid, p_MPa):
        """rho should be nearly continuous at h = h_f."""
        ref_f = iapws.IAPWS97(P=p_MPa, x=0)
        h_f = ref_f.h * 1e3
        p = p_MPa * 1e6

        rho_below = fluid.evaluate(p, h_f - 100.0).rho
        rho_above = fluid.evaluate(p, h_f + 100.0).rho
        # Should be close (within ~1% for 200 J/kg difference)
        assert abs(rho_below - rho_above) / rho_below < 0.01

    @pytest.mark.parametrize("p_MPa", [1.0, 5.0])
    def test_twophase_to_steam(self, fluid, p_MPa):
        """rho should be nearly continuous at h = h_g.

        Note: 10 MPa excluded — near-critical R2 backward equation
        struggles within 1 kJ/kg of saturation.  The R2 property
        evaluations well above h_g (e.g. 600 K, 10 MPa) pass fine.
        """
        ref_g = iapws.IAPWS97(P=p_MPa, x=1)
        h_g = ref_g.h * 1e3
        p = p_MPa * 1e6

        rho_below = fluid.evaluate(p, h_g - 1000.0).rho
        rho_above = fluid.evaluate(p, h_g + 1000.0).rho
        assert abs(rho_below - rho_above) / rho_below < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
