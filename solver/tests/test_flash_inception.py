"""
test_flash_inception.py — L0 term verification for the flashing inception model.

Tests the Modelica interfacial HT closure formulas (Pipe1D_DriftFlux.mo) and the
semi-implicit void-pressure coupling (bridge_5eq_solver.py).

Coverage gaps addressed (from QA review 2026-03-29):
  Gap 1: Modelica q_i_l formula verification (not C++ closure)
  Gap 2: Baseline equivalence (use_relaxation=0 vs unsplit formula)
  Gap 3: H_relax magnitude verification
  Gap 4: Depressurization flashing with use_relaxation=1
  Gap 5: Nucleation onset + relaxation interaction
  Gap 6: d_b_eff with use_inception=1
  Gap 7: Semi-implicit void-pressure coupling sign and magnitude

SimpleFluid constants (from library/Media/SimpleFluid.mo):
  p_ref = 10e6, T_sat_0 = 400 K, T_sat_1 = 20 K
  h_f_0 = 800e3, h_f_1 = 100e3, h_g_0 = 2800e3, h_g_1 = 50e3
  rho_f_0 = 750, rho_f_1 = 20, rho_g_0 = 40, rho_g_1 = 5
  cp_L = 4000, cp_G = 2000, k_f = 0.56 W/(m*K)
"""

import numpy as np
import pytest

# ============================================================================
# SimpleFluid analytical helpers (replicate SimpleFluid.mo exactly)
# ============================================================================

P_REF = 10e6
T_SAT_0, T_SAT_1 = 400.0, 20.0
H_F_0, H_F_1 = 800e3, 100e3
H_G_0, H_G_1 = 2800e3, 50e3
RHO_F_0, RHO_F_1 = 750.0, 20.0
RHO_G_0, RHO_G_1 = 40.0, 5.0
CP_L = 4000.0
K_F = 0.56  # W/(m*K), constant in SimpleFluid


def _p_hat(p):
    return (p - P_REF) / P_REF

def sf_T_sat(p):
    return T_SAT_0 + T_SAT_1 * _p_hat(p)

def sf_h_f(p):
    return H_F_0 + H_F_1 * _p_hat(p)

def sf_h_g(p):
    return H_G_0 + H_G_1 * _p_hat(p)

def sf_h_fg(p):
    return sf_h_g(p) - sf_h_f(p)

def sf_rho_f(p):
    return RHO_F_0 + RHO_F_1 * _p_hat(p)

def sf_rho_g(p):
    return RHO_G_0 + RHO_G_1 * _p_hat(p)

def sf_T_l(p, h_l):
    """Liquid temperature with metastable extension."""
    h_f = sf_h_f(p)
    if h_l <= h_f:
        return sf_T_sat(p) - (h_f - h_l) / CP_L
    else:
        return sf_T_sat(p) + (h_l - h_f) / CP_L


# ============================================================================
# Modelica formula replicas (from Pipe1D_DriftFlux.mo interfacial closure)
# ============================================================================

def compute_d_b_eff(alpha_eff, use_inception, d_b, d_b_flash, d_b_min,
                    alpha_nucleation, alpha_flash):
    """Replica of d_b_eff computation from Pipe1D_DriftFlux.mo."""
    ramp = min(max((alpha_eff - alpha_nucleation) / (alpha_flash - alpha_nucleation), 0.0), 1.0)
    return max(d_b - use_inception * (d_b - d_b_flash) * (1 - ramp), d_b_min)


def compute_tau_eff(tau_flash, T_l, T_sat, tau_flash_n=0, tau_flash_DT_ref=1.0):
    """Replica of tau_eff computation from Pipe1D_DriftFlux.mo.

    tau_eff = tau_flash / max((T_l - T_sat) / DT_ref, 1.0) ^ n
    Backward compatible: n=0 → tau_eff = tau_flash.
    """
    return tau_flash / max((T_l - T_sat) / tau_flash_DT_ref, 1.0) ** tau_flash_n


def compute_q_i_l(p, alpha_eff, h_l, use_relaxation, tau_flash,
                  Nu_i, d_b_eff, tau_flash_n=0, tau_flash_DT_ref=1.0):
    """Replica of q_i_l computation from Pipe1D_DriftFlux.mo (split formula)."""
    T_sat = sf_T_sat(p)
    T_l = sf_T_l(p, h_l)
    rho_l = sf_rho_f(p)

    h_i = Nu_i * K_F / d_b_eff
    a_i = 6 * alpha_eff * (1 - alpha_eff) / d_b_eff

    DT_sub = max(T_sat - T_l, 0.0)  # subcooling (condensation direction)
    DT_sup = max(T_l - T_sat, 0.0)  # superheat (evaporation direction)

    tau_eff = compute_tau_eff(tau_flash, T_l, T_sat, tau_flash_n, tau_flash_DT_ref)

    H_geo = h_i * a_i
    H_relax = alpha_eff * (1 - alpha_eff) * rho_l * CP_L / tau_eff

    H_eff_evap = H_geo + use_relaxation * (H_relax - H_geo)

    q_i_l = H_geo * DT_sub - H_eff_evap * DT_sup
    return q_i_l, h_i, a_i, H_geo, H_relax


def compute_Gamma(q_i_l, p):
    """Gamma = -q_i_l / h_fg."""
    h_fg = max(sf_h_fg(p), 1.0)
    return -q_i_l / h_fg


def compute_q_i_v(Gamma, h_v, h_l, q_i_l):
    """Interface energy balance."""
    return -Gamma * (h_v - h_l) - q_i_l


# ============================================================================
# Semi-implicit void-pressure coupling replica (from bridge_5eq_solver.py)
# ============================================================================

def compute_void_diag(p, alpha, h_l, h_v, Gamma, T_l, T_sat,
                      rho_l, rho_v, h_sat_l, h_sat_v, V_cell):
    """Replica of the semi-implicit void coupling diagonal term."""
    if Gamma <= 0 or T_l <= T_sat:
        return 0.0
    rv = max(rho_v, 0.01)
    superheat = max(T_l - T_sat, 0.1)
    h_fg = max(h_sat_v - h_sat_l, 1.0)
    dTsat_dp = T_sat * (1.0 / rv - 1.0 / max(rho_l, 1.0)) / h_fg
    dGamma_dp = -Gamma * dTsat_dp / superheat
    void_diag = -V_cell * (rho_l - rho_v) / rv * dGamma_dp
    return max(void_diag, 0.0)


# ============================================================================
# Test parameters
# ============================================================================

# Default Modelica parameters
D_B = 3e-4          # bulk bubble diameter [m]
D_B_FLASH = 3e-5    # nucleation bubble diameter [m]
D_B_MIN = 1e-5      # minimum bubble diameter [m]
NU_I = 2.0          # interfacial Nusselt number
ALPHA_NUCLEATION = 1e-3
ALPHA_FLASH = 0.05
TAU_FLASH = 0.025   # relaxation time [s]

# Test states
P_NOM = 10e6        # nominal pressure (SimpleFluid reference)
P_LOW = 7e6         # lower pressure (used for depressurization flashing test)

# NOTE: D_B, TAU_FLASH match EdwardsTest_DriftFlux_HF_Ramp_Flash.mo parameters,
# not the Pipe1D_DriftFlux.mo defaults (d_b=1e-3, tau_flash=0.005).


# ============================================================================
# Gap 6: d_b_eff tests (use_inception)
# ============================================================================

class TestDbEff:
    """Verify d_b_eff computation for use_inception=0 and use_inception=1."""

    def test_baseline_passthrough(self):
        """use_inception=0: d_b_eff = d_b for all alpha."""
        for alpha in [1e-6, 1e-3, 0.01, 0.05, 0.1, 0.5, 0.95]:
            d = compute_d_b_eff(alpha, use_inception=0, d_b=D_B, d_b_flash=D_B_FLASH,
                               d_b_min=D_B_MIN, alpha_nucleation=ALPHA_NUCLEATION,
                               alpha_flash=ALPHA_FLASH)
            assert d == D_B, f"alpha={alpha}: d_b_eff={d} != d_b={D_B}"

    def test_inception_at_nucleation(self):
        """use_inception=1, alpha <= alpha_nucleation: d_b_eff = d_b_flash."""
        for alpha in [1e-6, 1e-4, ALPHA_NUCLEATION]:
            d = compute_d_b_eff(alpha, use_inception=1, d_b=D_B, d_b_flash=D_B_FLASH,
                               d_b_min=D_B_MIN, alpha_nucleation=ALPHA_NUCLEATION,
                               alpha_flash=ALPHA_FLASH)
            assert d == pytest.approx(D_B_FLASH, abs=1e-10), \
                f"alpha={alpha}: d_b_eff={d} != d_b_flash={D_B_FLASH}"

    def test_inception_at_bulk(self):
        """use_inception=1, alpha >= alpha_flash: d_b_eff = d_b."""
        for alpha in [ALPHA_FLASH, 0.1, 0.5, 0.95]:
            d = compute_d_b_eff(alpha, use_inception=1, d_b=D_B, d_b_flash=D_B_FLASH,
                               d_b_min=D_B_MIN, alpha_nucleation=ALPHA_NUCLEATION,
                               alpha_flash=ALPHA_FLASH)
            assert d == pytest.approx(D_B, abs=1e-10), \
                f"alpha={alpha}: d_b_eff={d} != d_b={D_B}"

    def test_inception_monotonicity(self):
        """d_b_eff must be non-decreasing with alpha for use_inception=1."""
        alphas = np.linspace(0, 1, 200)
        d_prev = 0
        for alpha in alphas:
            d = compute_d_b_eff(alpha, use_inception=1, d_b=D_B, d_b_flash=D_B_FLASH,
                               d_b_min=D_B_MIN, alpha_nucleation=ALPHA_NUCLEATION,
                               alpha_flash=ALPHA_FLASH)
            assert d >= d_prev - 1e-15, \
                f"Non-monotonic: d_b_eff({alpha:.4f})={d} < d_b_eff_prev={d_prev}"
            d_prev = d

    def test_inception_mid_ramp_magnitude(self):
        """d_b_eff at ramp midpoint matches hard-coded value (catches factor errors)."""
        alpha_mid = (ALPHA_NUCLEATION + ALPHA_FLASH) / 2  # 0.0255
        d = compute_d_b_eff(alpha_mid, use_inception=1, d_b=D_B, d_b_flash=D_B_FLASH,
                           d_b_min=D_B_MIN, alpha_nucleation=ALPHA_NUCLEATION,
                           alpha_flash=ALPHA_FLASH)
        # ramp = (0.0255 - 0.001) / (0.05 - 0.001) = 0.5
        # d_b_eff = 3e-4 - 1*(3e-4 - 3e-5)*(1-0.5) = 3e-4 - 1.35e-4 = 1.65e-4
        assert d == pytest.approx(1.65e-4, rel=1e-6), f"Mid-ramp d_b_eff={d}, expected 1.65e-4"

    def test_inception_continuity(self):
        """d_b_eff is continuous at transition boundaries."""
        eps = 1e-8
        for boundary in [ALPHA_NUCLEATION, ALPHA_FLASH]:
            d_lo = compute_d_b_eff(boundary - eps, use_inception=1, d_b=D_B,
                                   d_b_flash=D_B_FLASH, d_b_min=D_B_MIN,
                                   alpha_nucleation=ALPHA_NUCLEATION,
                                   alpha_flash=ALPHA_FLASH)
            d_hi = compute_d_b_eff(boundary + eps, use_inception=1, d_b=D_B,
                                   d_b_flash=D_B_FLASH, d_b_min=D_B_MIN,
                                   alpha_nucleation=ALPHA_NUCLEATION,
                                   alpha_flash=ALPHA_FLASH)
            assert abs(d_hi - d_lo) < 1e-6, \
                f"Discontinuity at alpha={boundary}: {d_lo} vs {d_hi}"


# ============================================================================
# Gap 2: Baseline equivalence (use_relaxation=0)
# ============================================================================

class TestBaselineEquivalence:
    """Verify split formula reproduces old unsplit formula when use_relaxation=0."""

    def _old_formula(self, p, alpha_eff, h_l, d_b_eff):
        """Original unsplit: q_i_l = h_i * a_i * (T_sat - T_l)."""
        T_sat = sf_T_sat(p)
        T_l = sf_T_l(p, h_l)
        h_i = NU_I * K_F / d_b_eff
        a_i = 6 * alpha_eff * (1 - alpha_eff) / d_b_eff
        return h_i * a_i * (T_sat - T_l)

    def test_subcooled_equivalence(self):
        """Subcooled state (T_l < T_sat): split = unsplit."""
        p = P_NOM
        alpha = 0.1
        h_l = sf_h_f(p) - 50e3  # 50 kJ below saturation
        q_new, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=0,
                                   tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        q_old = self._old_formula(p, alpha, h_l, D_B)
        assert q_new == pytest.approx(q_old, rel=1e-12), \
            f"Subcooled mismatch: new={q_new}, old={q_old}"

    def test_superheated_equivalence(self):
        """Superheated state (T_l > T_sat): split = unsplit."""
        p = P_NOM
        alpha = 0.1
        h_l = sf_h_f(p) + 20e3  # 20 kJ above saturation → superheated
        q_new, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=0,
                                   tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        q_old = self._old_formula(p, alpha, h_l, D_B)
        assert q_new == pytest.approx(q_old, rel=1e-12), \
            f"Superheated mismatch: new={q_new}, old={q_old}"

    def test_equilibrium_zero(self):
        """At T_l = T_sat: q_i_l = 0 for both formulas."""
        p = P_NOM
        alpha = 0.1
        h_l = sf_h_f(p)  # exactly at saturation
        q_new, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=0,
                                   tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        assert q_new == pytest.approx(0.0, abs=1e-6), f"Not zero at equilibrium: {q_new}"

    def test_sweep_equivalence(self):
        """Sweep h_l across sub/superheated: split always matches unsplit."""
        p = P_NOM
        alpha = 0.15
        h_f = sf_h_f(p)
        for dh in np.linspace(-100e3, 100e3, 50):
            h_l = h_f + dh
            q_new, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=0,
                                       tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
            q_old = self._old_formula(p, alpha, h_l, D_B)
            assert q_new == pytest.approx(q_old, rel=1e-12, abs=1e-6), \
                f"dh={dh/1e3:.0f}kJ: new={q_new:.2f}, old={q_old:.2f}"


# ============================================================================
# Gap 1 + 3: q_i_l sign and H_relax magnitude (use_relaxation=1)
# ============================================================================

class TestRelaxationModel:
    """Verify Jones/Lahey relaxation closure signs and magnitudes."""

    def test_condensation_uses_geometric(self):
        """Subcooled (T_l < T_sat): use_relaxation=1 uses geometric H_eff (same as baseline)."""
        p = P_NOM
        alpha = 0.1
        h_l = sf_h_f(p) - 50e3  # subcooled
        q_relax, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=1,
                                     tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        q_base, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=0,
                                    tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        assert q_relax == pytest.approx(q_base, rel=1e-12), \
            "Condensation should use geometric H_eff regardless of use_relaxation"
        assert q_relax > 0, f"q_i_l should be positive during condensation: {q_relax}"

    def test_evaporation_sign(self):
        """Superheated (T_l > T_sat): q_i_l < 0 (heat from liquid), Gamma > 0."""
        p = P_NOM
        alpha = 0.1
        h_l = sf_h_f(p) + 20e3  # superheated
        q, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=1,
                               tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        Gamma = compute_Gamma(q, p)
        assert q < 0, f"q_i_l should be negative during evaporation: {q}"
        assert Gamma > 0, f"Gamma should be positive during evaporation: {Gamma}"

    def test_H_relax_magnitude(self):
        """Verify H_relax = alpha*(1-alpha)*rho_l*cp_f/tau at reference state."""
        p = P_NOM
        alpha = 0.1
        rho_l = sf_rho_f(p)
        H_relax_expected = alpha * (1 - alpha) * rho_l * CP_L / TAU_FLASH
        # Verify via compute_q_i_l
        _, _, _, _, H_relax_actual = compute_q_i_l(
            p, alpha, sf_h_f(p) + 10e3, use_relaxation=1,
            tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        assert H_relax_actual == pytest.approx(H_relax_expected, rel=1e-10), \
            f"H_relax: got {H_relax_actual}, expected {H_relax_expected}"

    def test_H_relax_hardcoded_literal(self):
        """H_relax against hard-coded number (catches rho_l/rho_v swap, FM2).
        At p=10e6, alpha=0.1: rho_l=750, cp_f=4000, tau=0.025
        H_relax = 0.1 * 0.9 * 750 * 4000 / 0.025 = 10,800,000 W/(m³·K)
        If rho_v(=40) used instead of rho_l(=750): H_relax = 576,000 (19x smaller → fails)."""
        p = P_NOM
        _, _, _, _, H_relax = compute_q_i_l(
            p, 0.1, sf_h_f(p) + 10e3, use_relaxation=1,
            tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        assert H_relax == pytest.approx(1.08e7, rel=1e-3), \
            f"H_relax={H_relax:.0f}, expected 10,800,000 (rho_l/rho_v swap?)"

    def test_enhancement_ratio(self):
        """Verify H_relax/H_geo ratio = rho_l*cp_f*d_b^2 / (6*Nu*k_f*tau)."""
        p = P_NOM
        alpha = 0.1
        rho_l = sf_rho_f(p)
        ratio_expected = rho_l * CP_L * D_B**2 / (6 * NU_I * K_F * TAU_FLASH)
        _, _, _, H_geo, H_relax = compute_q_i_l(
            p, alpha, sf_h_f(p) + 10e3, use_relaxation=1,
            tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        ratio_actual = H_relax / H_geo
        assert ratio_actual == pytest.approx(ratio_expected, rel=1e-10), \
            f"Enhancement ratio: got {ratio_actual:.2f}, expected {ratio_expected:.2f}"

    def test_enhancement_ratio_alpha_independent(self):
        """H_relax/H_geo ratio is independent of alpha (same alpha*(1-alpha) factor)."""
        p = P_NOM
        ratios = []
        for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
            _, _, _, H_geo, H_relax = compute_q_i_l(
                p, alpha, sf_h_f(p) + 10e3, use_relaxation=1,
                tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
            ratios.append(H_relax / H_geo)
        # All ratios should be identical
        for r in ratios:
            assert r == pytest.approx(ratios[0], rel=1e-10), \
                f"Ratio varies with alpha: {ratios}"

    def test_Gamma_magnitude_evaporation(self):
        """Hand-calculate Gamma at a known superheated state."""
        p = P_NOM
        alpha = 0.1
        h_l = sf_h_f(p) + 20e3  # 20 kJ/kg superheat → ΔT = 20e3/4000 = 5 K
        T_sat = sf_T_sat(p)
        T_l = sf_T_l(p, h_l)
        DT_sup = T_l - T_sat
        assert DT_sup == pytest.approx(5.0, abs=0.01), f"ΔT_sup={DT_sup}, expected 5K"

        rho_l = sf_rho_f(p)
        H_relax = alpha * (1 - alpha) * rho_l * CP_L / TAU_FLASH
        q_expected = -H_relax * DT_sup
        Gamma_expected = -q_expected / sf_h_fg(p)

        q_actual, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=1,
                                      tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        Gamma_actual = compute_Gamma(q_actual, p)

        assert q_actual == pytest.approx(q_expected, rel=1e-10), \
            f"q_i_l: got {q_actual:.1f}, expected {q_expected:.1f}"
        assert Gamma_actual == pytest.approx(Gamma_expected, rel=1e-10), \
            f"Gamma: got {Gamma_actual:.4f}, expected {Gamma_expected:.4f}"

    def test_self_limiting_alpha_zero(self):
        """H_relax at alpha→0 is negligible relative to peak (alpha=0.5)."""
        p = P_NOM
        alpha_small = 1e-6
        _, _, _, _, H_small = compute_q_i_l(
            p, alpha_small, sf_h_f(p) + 10e3, use_relaxation=1,
            tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        _, _, _, _, H_peak = compute_q_i_l(
            p, 0.5, sf_h_f(p) + 10e3, use_relaxation=1,
            tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        ratio = H_small / H_peak
        assert ratio < 1e-4, \
            f"H_relax(alpha={alpha_small}) should be <<H_relax(0.5): ratio={ratio:.2e}"

    def test_self_limiting_alpha_one(self):
        """H_relax → 0 as alpha → 1 (no liquid, no flashing)."""
        p = P_NOM
        alpha = 0.999
        _, _, _, _, H_relax = compute_q_i_l(
            p, alpha, sf_h_f(p) + 10e3, use_relaxation=1,
            tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        rho_l = sf_rho_f(p)
        H_relax_full = 0.5 * 0.5 * rho_l * CP_L / TAU_FLASH  # peak at alpha=0.5
        assert H_relax < 0.01 * H_relax_full, \
            f"H_relax should be <<peak at alpha→1: {H_relax:.1f} vs peak {H_relax_full:.1f}"


# ============================================================================
# Gap 5: Nucleation onset + relaxation interaction
# ============================================================================

class TestNucleationOnset:
    """Verify alpha_eff nucleation floor interacts correctly with relaxation."""

    def test_nucleation_floor_activates(self):
        """When T_l > T_sat and alpha < alpha_nucleation, alpha_eff = alpha_nucleation."""
        p = P_NOM
        alpha = 1e-6  # tiny void
        h_l = sf_h_f(p) + 10e3  # superheated
        T_l = sf_T_l(p, h_l)
        T_sat = sf_T_sat(p)
        assert T_l > T_sat, "Must be superheated for this test"

        # With nucleation floor: alpha_eff = alpha_nucleation
        alpha_eff = ALPHA_NUCLEATION
        q, _, _, _, H_relax = compute_q_i_l(
            p, alpha_eff, h_l, use_relaxation=1,
            tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        Gamma = compute_Gamma(q, p)

        # Gamma should be positive (evaporation)
        assert Gamma > 0, f"Gamma should be positive at nucleation onset: {Gamma}"
        # H_relax at alpha_nucleation should be small but non-zero
        assert H_relax > 0, f"H_relax should be positive: {H_relax}"
        rho_l = sf_rho_f(p)
        H_expected = ALPHA_NUCLEATION * (1 - ALPHA_NUCLEATION) * rho_l * CP_L / TAU_FLASH
        assert H_relax == pytest.approx(H_expected, rel=1e-10)


# ============================================================================
# Interface energy balance
# ============================================================================

class TestEnergyBalance:
    """Verify q_i_l + q_i_v + Gamma*(h_v - h_l) = 0 (interface conservation)."""

    @pytest.mark.parametrize("dh_l", [-50e3, -10e3, 0, 10e3, 50e3])
    @pytest.mark.parametrize("use_relax", [0, 1])
    def test_interface_energy_balance(self, dh_l, use_relax):
        """Interface energy balance holds for all states and models."""
        p = P_NOM
        alpha = 0.15
        h_l = sf_h_f(p) + dh_l
        h_v = sf_h_g(p)

        q_i_l, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=use_relax,
                                    tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        Gamma = compute_Gamma(q_i_l, p)
        q_i_v = compute_q_i_v(Gamma, h_v, h_l, q_i_l)

        balance = q_i_l + q_i_v + Gamma * (h_v - h_l)
        assert balance == pytest.approx(0.0, abs=1e-6), \
            f"Energy balance violated: {balance} (dh_l={dh_l/1e3}kJ, relax={use_relax})"


# ============================================================================
# Gap 7: Semi-implicit void-pressure coupling
# ============================================================================

class TestVoidPressureCoupling:
    """Verify the semi-implicit diagonal term for void-pressure stabilization."""

    def test_inactive_during_condensation(self):
        """void_diag = 0 when T_l < T_sat (no evaporation)."""
        p = P_NOM
        T_sat = sf_T_sat(p)
        T_l = T_sat - 10.0  # subcooled
        diag = compute_void_diag(
            p=p, alpha=0.1, h_l=sf_h_f(p) - 40e3, h_v=sf_h_g(p),
            Gamma=-5.0, T_l=T_l, T_sat=T_sat,
            rho_l=sf_rho_f(p), rho_v=sf_rho_g(p),
            h_sat_l=sf_h_f(p), h_sat_v=sf_h_g(p), V_cell=7e-4)
        assert diag == 0.0, f"Should be zero during condensation: {diag}"

    def test_inactive_when_Gamma_negative(self):
        """void_diag = 0 when Gamma <= 0."""
        p = P_NOM
        diag = compute_void_diag(
            p=p, alpha=0.1, h_l=sf_h_f(p), h_v=sf_h_g(p),
            Gamma=-1.0, T_l=sf_T_sat(p) + 5, T_sat=sf_T_sat(p),
            rho_l=sf_rho_f(p), rho_v=sf_rho_g(p),
            h_sat_l=sf_h_f(p), h_sat_v=sf_h_g(p), V_cell=7e-4)
        assert diag == 0.0

    def test_positive_during_evaporation(self):
        """void_diag > 0 when Gamma > 0 and T_l > T_sat (stabilizing)."""
        p = P_NOM
        T_sat = sf_T_sat(p)
        diag = compute_void_diag(
            p=p, alpha=0.1, h_l=sf_h_f(p) + 20e3, h_v=sf_h_g(p),
            Gamma=50.0, T_l=T_sat + 5.0, T_sat=T_sat,
            rho_l=sf_rho_f(p), rho_v=sf_rho_g(p),
            h_sat_l=sf_h_f(p), h_sat_v=sf_h_g(p), V_cell=7e-4)
        assert diag > 0, f"Should be positive (stabilizing) during evaporation: {diag}"

    def test_magnitude_scales_with_Gamma(self):
        """void_diag scales linearly with Gamma."""
        p = P_NOM
        T_sat = sf_T_sat(p)
        kwargs = dict(p=p, alpha=0.1, h_l=sf_h_f(p) + 20e3, h_v=sf_h_g(p),
                      T_l=T_sat + 5.0, T_sat=T_sat,
                      rho_l=sf_rho_f(p), rho_v=sf_rho_g(p),
                      h_sat_l=sf_h_f(p), h_sat_v=sf_h_g(p), V_cell=7e-4)
        d1 = compute_void_diag(Gamma=50.0, **kwargs)
        d2 = compute_void_diag(Gamma=100.0, **kwargs)
        assert d2 == pytest.approx(2.0 * d1, rel=1e-10), \
            f"Should scale linearly: d(100)={d2}, 2*d(50)={2*d1}"

    def test_negligible_at_baseline_Gamma(self):
        """At baseline Gamma (~4), void_diag << alpha_coeff."""
        p = P_NOM
        T_sat = sf_T_sat(p)
        V_cell = 7.15e-4  # Edwards cell volume
        dt = 5e-5
        drho_dp = 6e-4  # subcooled compressibility
        alpha_coeff = V_cell * drho_dp / dt  # ~ 0.0086

        diag = compute_void_diag(
            p=p, alpha=0.1, h_l=sf_h_f(p) + 20e3, h_v=sf_h_g(p),
            Gamma=4.0, T_l=T_sat + 5.0, T_sat=T_sat,
            rho_l=sf_rho_f(p), rho_v=sf_rho_g(p),
            h_sat_l=sf_h_f(p), h_sat_v=sf_h_g(p), V_cell=V_cell)
        ratio = diag / alpha_coeff
        assert ratio < 0.1, \
            f"void_diag should be <10% of alpha_coeff at baseline: ratio={ratio:.4f}"

    def test_scales_with_Gamma_squared(self):
        """At enhanced Gamma, void_diag grows linearly (since dGamma/dp ~ Gamma)."""
        p = P_NOM
        T_sat = sf_T_sat(p)
        V_cell = 7.15e-4
        kwargs = dict(p=p, alpha=0.1, h_l=sf_h_f(p) + 20e3, h_v=sf_h_g(p),
                      T_l=T_sat + 5.0, T_sat=T_sat,
                      rho_l=sf_rho_f(p), rho_v=sf_rho_g(p),
                      h_sat_l=sf_h_f(p), h_sat_v=sf_h_g(p), V_cell=V_cell)
        d_base = compute_void_diag(Gamma=4.0, **kwargs)
        d_enhanced = compute_void_diag(Gamma=400.0, **kwargs)
        # Should scale by 100x (400/4)
        assert d_enhanced == pytest.approx(100 * d_base, rel=1e-10), \
            f"Should scale linearly: d(400)={d_enhanced}, 100*d(4)={100*d_base}"
        # Verify the enhanced value is positive and non-trivial
        assert d_enhanced > 0, f"void_diag should be positive: {d_enhanced}"

    def test_hand_calculation(self):
        """Verify void_diag against full hand calculation."""
        p = P_NOM
        alpha = 0.1
        rho_l = sf_rho_f(p)  # 750
        rho_v = sf_rho_g(p)  # 40
        T_sat = sf_T_sat(p)  # 400 K
        h_fg = sf_h_fg(p)    # 2e6
        superheat = 5.0      # K
        Gamma = 50.0
        V_cell = 7e-4

        # dT_sat/dp = T_sat * (1/rho_v - 1/rho_l) / h_fg
        dTsat_dp = T_sat * (1.0 / rho_v - 1.0 / rho_l) / h_fg
        # = 400 * (0.025 - 0.00133) / 2e6 = 400 * 0.02367 / 2e6 = 4.73e-6 K/Pa
        assert dTsat_dp == pytest.approx(4.733e-6, rel=0.01)

        # dGamma/dp = -Gamma * dTsat_dp / superheat
        dGamma_dp = -Gamma * dTsat_dp / superheat
        # = -50 * 4.73e-6 / 5 = -4.73e-5

        # void_diag = -V * (rho_l - rho_v) / rho_v * dGamma_dp
        # = -7e-4 * 710 / 40 * (-4.73e-5) = -7e-4 * 17.75 * (-4.73e-5) = 5.88e-7
        void_diag_expected = -V_cell * (rho_l - rho_v) / rho_v * dGamma_dp
        assert void_diag_expected > 0, "Should be positive"

        void_diag_actual = compute_void_diag(
            p=p, alpha=alpha, h_l=sf_h_f(p) + 20e3, h_v=sf_h_g(p),
            Gamma=Gamma, T_l=T_sat + superheat, T_sat=T_sat,
            rho_l=rho_l, rho_v=rho_v,
            h_sat_l=sf_h_f(p), h_sat_v=sf_h_g(p), V_cell=V_cell)
        assert void_diag_actual == pytest.approx(void_diag_expected, rel=1e-10)
        # Hard-coded literal (FM2 swap detector)
        assert void_diag_actual == pytest.approx(5.88e-7, rel=0.02), \
            f"void_diag={void_diag_actual:.2e}, expected ~5.88e-7"


# ============================================================================
# Gap 4: Depressurization flashing test (the core Edwards physics scenario)
# ============================================================================

class TestDepressurizationFlashing:
    """Verify the relaxation model under depressurization — liquid initially at
    equilibrium at P_NOM, then evaluate at P_LOW where h_l > h_f(P_LOW)."""

    def test_superheat_from_depressurization(self):
        """Liquid at equilibrium at 10 MPa, depressurized to 7 MPa → superheated."""
        h_l_init = sf_h_f(P_NOM)  # 800 kJ/kg at 10 MPa
        h_f_low = sf_h_f(P_LOW)   # h_f(7 MPa) = 800e3 + 100e3 * (-0.3) = 770 kJ/kg
        T_sat_low = sf_T_sat(P_LOW)  # 400 + 20 * (-0.3) = 394 K
        T_l = sf_T_l(P_LOW, h_l_init)  # 394 + (800e3 - 770e3)/4000 = 394 + 7.5 = 401.5 K
        superheat = T_l - T_sat_low
        assert superheat == pytest.approx(7.5, abs=0.01), \
            f"Superheat from depressurization: {superheat:.2f} K, expected 7.5 K"

    def test_flashing_Gamma_positive(self):
        """Depressurized liquid produces positive Gamma (evaporation)."""
        h_l = sf_h_f(P_NOM)  # equilibrium at 10 MPa
        alpha = ALPHA_NUCLEATION  # nucleation floor
        q, *_ = compute_q_i_l(P_LOW, alpha, h_l, use_relaxation=1,
                               tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        Gamma = compute_Gamma(q, P_LOW)
        assert q < 0, f"q_i_l should be negative (heat from liquid): {q}"
        assert Gamma > 0, f"Gamma should be positive (evaporation): {Gamma}"

    def test_flashing_Gamma_hardcoded(self):
        """Hand-calculated Gamma for depressurization flashing (hard-coded literal).
        p=7e6, h_l=800e3 (from 10 MPa equil), alpha=1e-3 (nucleation floor)
        T_sat = 394 K, T_l = 401.5 K, ΔT_sup = 7.5 K
        H_relax = 0.001 * 0.999 * 770 * 4000 / 0.025 = 123,046 W/(m³·K)
        q_i_l = -123,046 * 7.5 = -922,846 W/m³
        h_fg(7 MPa) = h_g - h_f = (2800e3 + 50e3*(-0.3)) - (800e3 + 100e3*(-0.3))
                     = 2785e3 - 770e3 = 2,015,000 J/kg
        Gamma = 922,846 / 2,015,000 = 0.458 kg/(m³·s)"""
        h_l = sf_h_f(P_NOM)  # 800e3
        alpha = ALPHA_NUCLEATION
        rho_l_low = sf_rho_f(P_LOW)  # 750 + 20*(-0.3) = 744
        h_fg_low = sf_h_fg(P_LOW)    # 2015000

        H_relax_expected = alpha * (1 - alpha) * rho_l_low * CP_L / TAU_FLASH
        assert H_relax_expected == pytest.approx(119136.0, rel=0.01), \
            f"H_relax={H_relax_expected:.0f}"

        q_expected = -H_relax_expected * 7.5
        Gamma_expected = -q_expected / h_fg_low

        q_actual, *_ = compute_q_i_l(P_LOW, alpha, h_l, use_relaxation=1,
                                      tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        Gamma_actual = compute_Gamma(q_actual, P_LOW)

        assert Gamma_actual == pytest.approx(Gamma_expected, rel=1e-3), \
            f"Gamma={Gamma_actual:.4f}, expected {Gamma_expected:.4f}"
        # Sanity: Gamma should be modest (0.1-1 kg/m³/s at nucleation onset)
        assert 0.01 < Gamma_actual < 10, \
            f"Gamma out of physical range: {Gamma_actual}"

    def test_condensation_preserved_during_depressurization(self):
        """At P_LOW with subcooled liquid: condensation still uses geometric H_eff."""
        h_l = sf_h_f(P_LOW) - 50e3  # subcooled at P_LOW
        alpha = 0.1
        q_relax, *_ = compute_q_i_l(P_LOW, alpha, h_l, use_relaxation=1,
                                     tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        q_base, *_ = compute_q_i_l(P_LOW, alpha, h_l, use_relaxation=0,
                                    tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        assert q_relax == pytest.approx(q_base, rel=1e-12), \
            "Condensation at P_LOW should use geometric regardless of use_relaxation"


# ============================================================================
# Small-superheat realizability
# ============================================================================

class TestRealizability:
    """Verify physically reasonable behavior at extreme or edge-case states."""

    def test_tiny_superheat_small_Gamma(self):
        """At peak alpha with tiny superheat (0.001 K), Gamma should be small."""
        p = P_NOM
        alpha = 0.5  # peak H_relax
        h_l = sf_h_f(p) + CP_L * 0.001  # 0.001 K superheat → dh = 4 J/kg
        q, *_ = compute_q_i_l(p, alpha, h_l, use_relaxation=1,
                               tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B)
        Gamma = compute_Gamma(q, p)
        # H_relax = 0.25 * 750 * 4000 / 0.025 = 30,000,000
        # q_i_l = -30e6 * 0.001 = -30,000 → Gamma = 30000 / 2e6 = 0.015
        assert Gamma > 0, "Should be evaporating"
        assert Gamma < 1.0, \
            f"Gamma={Gamma:.4f} should be small at 0.001K superheat"
        assert Gamma == pytest.approx(0.015, rel=0.01), \
            f"Gamma={Gamma:.4f}, expected ~0.015"


# ============================================================================
# Superheat-dependent tau_flash (tau_eff)
# ============================================================================

class TestSuperHeatTau:
    """L0 tests for superheat-dependent tau_eff = tau_flash / max(DT/DT_ref, 1)^n."""

    def test_backward_compat_n_zero(self):
        """tau_flash_n=0: tau_eff = tau_flash for all superheats."""
        for DT in [-10.0, 0.0, 5.0, 50.0, 200.0]:
            T_sat = 400.0
            T_l = T_sat + DT
            tau = compute_tau_eff(TAU_FLASH, T_l, T_sat, tau_flash_n=0, tau_flash_DT_ref=1.0)
            assert tau == pytest.approx(TAU_FLASH, rel=1e-15), \
                f"n=0, DT={DT}: tau_eff={tau}, expected {TAU_FLASH}"

    def test_subcooled_returns_baseline(self):
        """Subcooled (T_l < T_sat): tau_eff = tau_flash regardless of n."""
        T_sat = 400.0
        for n in [0, 0.5, 1.0, 2.0]:
            tau = compute_tau_eff(TAU_FLASH, T_l=380.0, T_sat=T_sat,
                                 tau_flash_n=n, tau_flash_DT_ref=1.0)
            assert tau == pytest.approx(TAU_FLASH, rel=1e-15), \
                f"n={n}, subcooled: tau_eff={tau}, expected {TAU_FLASH}"

    def test_small_superheat_below_DT_ref(self):
        """Superheat < DT_ref: tau_eff = tau_flash (max clamps to 1)."""
        tau = compute_tau_eff(TAU_FLASH, T_l=400.5, T_sat=400.0,
                             tau_flash_n=1, tau_flash_DT_ref=1.0)
        assert tau == pytest.approx(TAU_FLASH, rel=1e-15), \
            f"DT=0.5 < DT_ref=1.0: tau_eff should be tau_flash"

    def test_exact_at_DT_ref(self):
        """Superheat = DT_ref: tau_eff = tau_flash (boundary, no enhancement)."""
        tau = compute_tau_eff(TAU_FLASH, T_l=401.0, T_sat=400.0,
                             tau_flash_n=1, tau_flash_DT_ref=1.0)
        assert tau == pytest.approx(TAU_FLASH, rel=1e-15)

    def test_linear_n1_20K(self):
        """n=1, DT=20K, DT_ref=1K: tau_eff = 0.025/20 = 0.00125."""
        tau = compute_tau_eff(TAU_FLASH, T_l=420.0, T_sat=400.0,
                             tau_flash_n=1, tau_flash_DT_ref=1.0)
        assert tau == pytest.approx(0.025 / 20.0, rel=1e-12)

    def test_linear_n1_100K(self):
        """n=1, DT=100K, DT_ref=1K: tau_eff = 0.025/100 = 0.00025."""
        tau = compute_tau_eff(TAU_FLASH, T_l=500.0, T_sat=400.0,
                             tau_flash_n=1, tau_flash_DT_ref=1.0)
        assert tau == pytest.approx(0.025 / 100.0, rel=1e-12)

    def test_quadratic_n2(self):
        """n=2, DT=10K: tau_eff = 0.025/100 = 0.00025."""
        tau = compute_tau_eff(TAU_FLASH, T_l=410.0, T_sat=400.0,
                             tau_flash_n=2, tau_flash_DT_ref=1.0)
        assert tau == pytest.approx(0.025 / 100.0, rel=1e-12)

    def test_DT_ref_scaling(self):
        """DT_ref=5K, n=1, DT=20K: tau_eff = 0.025/4 = 0.00625."""
        tau = compute_tau_eff(TAU_FLASH, T_l=420.0, T_sat=400.0,
                             tau_flash_n=1, tau_flash_DT_ref=5.0)
        assert tau == pytest.approx(0.025 / 4.0, rel=1e-12)

    def test_tau_eff_always_positive(self):
        """tau_eff > 0 for all physical states (prevents division by zero)."""
        for DT in [0.0, 0.01, 1.0, 50.0, 200.0]:
            tau = compute_tau_eff(TAU_FLASH, T_l=400.0 + DT, T_sat=400.0,
                                 tau_flash_n=1, tau_flash_DT_ref=1.0)
            assert tau > 0, f"tau_eff must be positive: {tau} at DT={DT}"

    def test_tau_eff_never_exceeds_baseline(self):
        """tau_eff <= tau_flash always (enhancement, never suppression)."""
        for DT in [-20.0, 0.0, 5.0, 50.0, 200.0]:
            tau = compute_tau_eff(TAU_FLASH, T_l=400.0 + DT, T_sat=400.0,
                                 tau_flash_n=1, tau_flash_DT_ref=1.0)
            assert tau <= TAU_FLASH + 1e-15, \
                f"tau_eff should not exceed tau_flash: {tau} > {TAU_FLASH} at DT={DT}"

    def test_q_i_l_enhanced_by_superheat(self):
        """q_i_l with superheat-dependent tau is stronger than baseline at high superheat.
        At p=7 MPa with 7.5K superheat and n=1: tau_eff = 0.025/7.5 = 0.00333.
        H_relax scales as 1/tau_eff, so ~7.5x stronger than baseline."""
        h_l = sf_h_f(P_NOM)  # equilibrium at 10 MPa → superheated at 7 MPa
        alpha = ALPHA_NUCLEATION

        q_base, *_ = compute_q_i_l(P_LOW, alpha, h_l, use_relaxation=1,
                                    tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B,
                                    tau_flash_n=0)
        q_enhanced, *_ = compute_q_i_l(P_LOW, alpha, h_l, use_relaxation=1,
                                        tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B,
                                        tau_flash_n=1, tau_flash_DT_ref=1.0)
        # Both should be negative (evaporation)
        assert q_base < 0 and q_enhanced < 0
        # Enhanced should be ~7.5x stronger (7.5K superheat, n=1, DT_ref=1K)
        ratio = q_enhanced / q_base
        assert ratio == pytest.approx(7.5, rel=0.01), \
            f"Enhancement ratio: {ratio:.2f}, expected ~7.5"

    def test_condensation_unaffected_by_superheat_tau(self):
        """Condensation (T_l < T_sat) is unaffected by tau_flash_n."""
        h_l = sf_h_f(P_NOM) - 50e3  # subcooled
        alpha = 0.1
        q_base, *_ = compute_q_i_l(P_NOM, alpha, h_l, use_relaxation=1,
                                    tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B,
                                    tau_flash_n=0)
        q_enhanced, *_ = compute_q_i_l(P_NOM, alpha, h_l, use_relaxation=1,
                                        tau_flash=TAU_FLASH, Nu_i=NU_I, d_b_eff=D_B,
                                        tau_flash_n=1, tau_flash_DT_ref=1.0)
        assert q_base == pytest.approx(q_enhanced, rel=1e-12), \
            "Condensation should be identical regardless of tau_flash_n"
