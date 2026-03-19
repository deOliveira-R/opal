#pragma once
/**
 * simple_fluid.hpp — SimpleFluid property package (C++).
 *
 * Hand-coded to exactly match library/Media/SimpleFluid.mo.
 * Linear saturation, constant single-phase derivatives, analytical two-phase.
 * Use this for solver verification — every value is hand-checkable.
 */

#include "fluid_package.hpp"
#include <cmath>

namespace opal {

class SimpleFluidProperties : public FluidPackage {
public:
    // Linear model works over a wide pressure range.
    double p_min() const override { return 1.0e4; }
    double p_max() const override { return 50.0e6; }

    // ---- Constants (match SimpleFluid.mo lines 8-32) ----
    static constexpr double p_ref   = 10.0e6;

    static constexpr double T_sat_0 = 400.0;
    static constexpr double T_sat_1 = 20.0;

    static constexpr double h_f_0   = 800.0e3;
    static constexpr double h_f_1   = 100.0e3;

    static constexpr double h_g_0   = 2800.0e3;
    static constexpr double h_g_1   = 50.0e3;

    static constexpr double rho_f_0 = 750.0;
    static constexpr double rho_f_1 = 20.0;

    static constexpr double rho_g_0 = 40.0;
    static constexpr double rho_g_1 = 5.0;

    static constexpr double A_L  = 6.25e-5;
    static constexpr double A_G  = 2.0e-5;

    static constexpr double cp_L = 4000.0;
    static constexpr double cp_G = 2000.0;

    // ---- Saturation functions ----

    static double T_sat(double p) {
        return T_sat_0 + T_sat_1 * (p - p_ref) / p_ref;
    }
    static double h_f(double p) {
        return h_f_0 + h_f_1 * (p - p_ref) / p_ref;
    }
    static double h_g(double p) {
        return h_g_0 + h_g_1 * (p - p_ref) / p_ref;
    }
    static double h_fg(double p) {
        return h_g(p) - h_f(p);
    }
    static double rho_f(double p) {
        return rho_f_0 + rho_f_1 * (p - p_ref) / p_ref;
    }
    static double rho_g(double p) {
        return rho_g_0 + rho_g_1 * (p - p_ref) / p_ref;
    }

    // ---- Region detection ----

    static int region_ph(double p, double h) {
        if (h < h_f(p)) return 1;
        if (h > h_g(p)) return 2;
        return 4;
    }

    // ---- Batch evaluation (one region check, four outputs) ----

    FluidProps evaluate(double p, double h) const override {
        FluidProps fp{};
        int reg = region_ph(p, h);

        if (reg == 1) {
            // Subcooled liquid
            double hf = h_f(p);
            double rf = rho_f(p);
            fp.rho       = rf + A_L * (hf - h);
            fp.drho_dp_h = (rho_f_1 + A_L * h_f_1) / p_ref;
            fp.drho_dh_p = -A_L;
            fp.T         = T_sat(p) - (hf - h) / cp_L;
        }
        else if (reg == 2) {
            // Superheated steam
            double hg = h_g(p);
            double rg = rho_g(p);
            fp.rho       = rg - A_G * (h - hg);
            fp.drho_dp_h = (rho_g_1 + A_G * h_g_1) / p_ref;
            fp.drho_dh_p = -A_G;
            fp.T         = T_sat(p) + (h - hg) / cp_G;
        }
        else {
            // Two-phase (Region 4)
            double rf   = rho_f(p);
            double rg   = rho_g(p);
            double hfv  = h_f(p);
            double hgv  = h_g(p);
            double hfgv = hgv - hfv;
            double x    = (h - hfv) / hfgv;

            double vf = 1.0 / rf;
            double vg = 1.0 / rg;
            double v  = x * vg + (1.0 - x) * vf;
            double rho_mix = 1.0 / v;
            double rho2    = rho_mix * rho_mix;

            fp.rho = rho_mix;

            // drho/dh|p = -rho^2 * (1/rho_g - 1/rho_f) / h_fg
            fp.drho_dh_p = -rho2 * (vg - vf) / hfgv;

            // drho/dp|h — fully analytical (matches SimpleFluid.mo lines 190-218)
            double drf_dp  = rho_f_1 / p_ref;
            double drg_dp  = rho_g_1 / p_ref;
            double dhf_dp  = h_f_1 / p_ref;
            double dhg_dp  = h_g_1 / p_ref;
            double dhfg_dp = dhg_dp - dhf_dp;

            double dvf_dp = -drf_dp / (rf * rf);
            double dvg_dp = -drg_dp / (rg * rg);

            double dx_dp = (-dhf_dp - x * dhfg_dp) / hfgv;
            double dv_dp = dx_dp * (vg - vf) + x * dvg_dp + (1.0 - x) * dvf_dp;

            fp.drho_dp_h = -rho2 * dv_dp;

            fp.T = T_sat(p);
        }

        return fp;
    }

    // ---- PhasicProperties interface ----

    PhasicProps evaluate_phasic(double p) const override {
        PhasicProps pp{};
        pp.rho_l       = rho_f(p);
        pp.rho_v       = rho_g(p);
        pp.h_sat_l     = h_f(p);
        pp.h_sat_v     = h_g(p);
        pp.T_sat       = T_sat(p);
        pp.drho_l_dp   = rho_f_1 / p_ref;
        pp.drho_v_dp   = rho_g_1 / p_ref;
        pp.dh_sat_l_dp = h_f_1 / p_ref;
        pp.dh_sat_v_dp = h_g_1 / p_ref;
        pp.cp_l        = cp_L;
        pp.cp_v        = cp_G;
        pp.sigma       = 0.05;  // constant surface tension [N/m]
        return pp;
    }

    double rho_liquid(double p, double h_l) const override {
        double hf = h_f(p);
        return rho_f(p) + A_L * (hf - h_l);
    }

    double rho_vapor(double p, double h_v) const override {
        double hg = h_g(p);
        return rho_g(p) - A_G * (h_v - hg);
    }

    double T_liquid(double p, double h_l) const override {
        return T_sat(p) - (h_f(p) - h_l) / cp_L;
    }

    double T_vapor(double p, double h_v) const override {
        return T_sat(p) + (h_v - h_g(p)) / cp_G;
    }
};

} // namespace opal
