#pragma once
/**
 * critical_flow.hpp — Critical (choked) flow models for break boundaries.
 *
 * When a pipe ruptures, the flow at the break can become choked (limited
 * by the local speed of sound). The critical flow model computes the
 * maximum mass flux and determines if the outlet is choked.
 *
 * Implementations:
 *   NoCriticalFlow — never choked (default)
 *   RansomTrapp    — quality-blended subcooled/HEM critical flow
 *
 * Reference: RELAP5/MOD3 Code Manual, Volume 1, Section 3.5
 */

#include "momentum.hpp"  // for CriticalFlowResult
#include "fluid_package.hpp"
#include <cmath>
#include <algorithm>

namespace opal {

/**
 * Abstract critical flow model.
 */
class CriticalFlowModel {
public:
    virtual ~CriticalFlowModel() = default;
    virtual const char* name() const = 0;

    /**
     * Evaluate critical flow at the break face.
     *
     * @param p_cell      Break cell pressure [Pa]
     * @param h_mix       Break cell mixture enthalpy [J/kg]
     * @param rho         Break cell mixture density [kg/m³]
     * @param drho_dp_h   ∂ρ/∂p at constant h [kg/(m³·Pa)]
     * @param p_back      Back-pressure downstream of break [Pa]
     * @param A_flow      Flow area [m²]
     * @param C_d         Discharge coefficient [-]
     * @param mdot_momentum  Momentum-predicted flow rate [kg/s]
     * @return            Critical flow result (G_crit, mdot_crit, is_choked)
     */
    virtual CriticalFlowResult evaluate(
        double p_cell, double h_mix, double rho, double drho_dp_h,
        double p_back, double A_flow, double C_d,
        double mdot_momentum) const = 0;
};

/**
 * No critical flow — never choked. Default for non-break boundaries.
 */
class NoCriticalFlow : public CriticalFlowModel {
public:
    const char* name() const override { return "none"; }

    CriticalFlowResult evaluate(
        double, double, double, double,
        double, double, double, double) const override
    {
        return {0.0, false};
    }
};

/**
 * Ransom-Trapp critical flow model.
 *
 * Blends subcooled (Bernoulli) and two-phase (HEM) critical mass flux
 * based on local quality, with smooth transition at x = x_trans.
 *
 *   Subcooled: G_sub = sqrt(2 * rho_f * (p - p_back))
 *   Two-phase: G_hem = rho * c_hem  where c_hem = sqrt(1/(rho * drho_dp_h))
 *   Blend:     G = G_sub*(1-x/x_trans) + G_hem*(x/x_trans)  for x < x_trans
 *              G = G_hem                                       for x >= x_trans
 *
 * Requires saturation properties (h_f, h_g, rho_f) at the break cell pressure.
 * These are obtained from the PhasicProperties interface if available,
 * or from SimpleFluid/IAPWS static methods.
 *
 * Reference: RELAP5/MOD3 Code Manual, Vol 1, §3.5
 */
class RansomTrapp : public CriticalFlowModel {
public:
    /**
     * @param phasic   Phasic property evaluator for saturation lookup (caller owns)
     * @param x_trans  Quality transition for blend [-], default 0.10
     * @param c_floor  Minimum sound speed [m/s], default 1200 (prevents
     *                 collapse of c_hem in low-quality two-phase)
     */
    explicit RansomTrapp(const FluidPackage& phasic,
                         double x_trans = 0.10, double c_floor = 1200.0)
        : phasic_(phasic), x_trans_(x_trans), c_floor_(c_floor) {}

    const char* name() const override { return "Ransom-Trapp"; }

    CriticalFlowResult evaluate(
        double p_cell, double h_mix, double rho, double drho_dp_h,
        double p_back, double A_flow, double C_d,
        double mdot_momentum) const override
    {
        CriticalFlowResult result{};

        // Get saturation properties at break cell pressure (from C++ phasic)
        double p_safe = std::clamp(p_cell, phasic_.p_min(), phasic_.p_max());
        auto pp = phasic_.evaluate_phasic(p_safe);
        double h_f   = pp.h_sat_l;
        double h_g   = pp.h_sat_v;
        double rho_f = pp.rho_l;

        // Local quality
        double h_fg = h_g - h_f;
        if (h_fg < 1e3) h_fg = 1e3;
        double x_local;
        if (h_mix <= h_f) {
            x_local = 0.0;
        } else if (h_mix >= h_g) {
            x_local = 1.0;
        } else {
            x_local = (h_mix - h_f) / h_fg;
        }

        // Subcooled critical mass flux (Bernoulli)
        double dp = std::max(p_cell - p_back, 0.0);
        double G_sub = std::sqrt(2.0 * rho_f * dp);

        // HEM critical mass flux
        double c_hem;
        if (drho_dp_h > 0.0) {
            c_hem = std::sqrt(1.0 / (rho * drho_dp_h));
        } else {
            c_hem = c_floor_;
        }
        double G_hem = rho * c_hem;

        // Quality-blended critical flux
        double G_crit;
        if (x_local < x_trans_) {
            double blend = x_local / x_trans_;
            G_crit = G_sub * (1.0 - blend) + G_hem * blend;
        } else {
            G_crit = G_hem;
        }
        G_crit = std::max(G_crit, G_hem);  // physical floor

        result.mdot_crit = C_d * A_flow * G_crit;

        // Check if momentum prediction exceeds critical
        result.is_choked = (mdot_momentum > result.mdot_crit)
                        && (result.mdot_crit > 0.0);

        return result;
    }

    double x_trans() const { return x_trans_; }

private:
    const FluidPackage& phasic_;
    double x_trans_;
    double c_floor_;
};

} // namespace opal
