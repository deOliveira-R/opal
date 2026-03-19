#pragma once
/**
 * closures.hpp — Interfacial closure relations for multi-equation models.
 *
 * Abstract interface for interfacial mass transfer (Γ), heat transfer (q_i),
 * drag (F_i), and drift-flux parameters (C_0, V_gj).
 *
 * Architecture:
 *   InterfacialClosures         — top-level interface (compute + drift_flux)
 *     NoClosures                — for HEM (no interfacial terms)
 *     DriftFluxClosures         — composer for 4-eq/5-eq, takes:
 *       HeatTransferModel       — interfacial heat transfer sub-model
 *         LinearRelaxation      — q = H_i · a_i · (T_sat - T_l)
 *       DriftVelocityModel      — drift velocity sub-model
 *         ZuberFindlay          — churn-turbulent bubbly flow
 */

#include <cmath>
#include <algorithm>

namespace opal {

struct InterfacialState {
    double p;        ///< Pressure [Pa]
    double alpha;    ///< Void fraction [-]
    double rho_l;    ///< Liquid density [kg/m³]
    double rho_v;    ///< Vapor density [kg/m³]
    double h_l;      ///< Liquid enthalpy [J/kg]
    double h_v;      ///< Vapor enthalpy [J/kg]
    double T_l;      ///< Liquid temperature [K]
    double T_v;      ///< Vapor temperature [K]
    double T_sat;    ///< Saturation temperature [K]
    double h_sat_l;  ///< Sat. liquid enthalpy [J/kg]
    double h_sat_v;  ///< Sat. vapor enthalpy [J/kg]
    double cp_l;     ///< Liquid heat capacity [J/(kg·K)]
    double sigma;    ///< Surface tension [N/m]
    double D_h;      ///< Hydraulic diameter [m]
    double g_mag = 9.81;  ///< |g⃗| for buoyancy correlations [m/s²]
};

struct ClosureResult {
    double Gamma;    ///< Interfacial mass transfer [kg/(m³·s)], >0 = evap
    double q_i_l;    ///< Interfacial heat to liquid [W/m³]
    double q_i_v;    ///< Interfacial heat to vapor [W/m³]
};

struct DriftFluxResult {
    double C_0;      ///< Distribution parameter [-]
    double V_gj;     ///< Drift velocity [m/s]
};

class InterfacialClosures {
public:
    virtual ~InterfacialClosures() = default;

    /// Compute interfacial transfer rates (Gamma, q_i_l, q_i_v).
    virtual ClosureResult compute(const InterfacialState& s) const = 0;

    /// Compute drift-flux parameters.
    virtual DriftFluxResult drift_flux(const InterfacialState& s) const = 0;
};

/**
 * No closures (HEM). Returns zero for everything.
 */
class NoClosures : public InterfacialClosures {
public:
    ClosureResult compute(const InterfacialState&) const override {
        return {0.0, 0.0, 0.0};
    }
    DriftFluxResult drift_flux(const InterfacialState&) const override {
        return {1.0, 0.0};  // C_0=1, V_gj=0 → no slip
    }
};

// ===========================================================================
// Sub-model interfaces — pluggable components of InterfacialClosures.
// ===========================================================================

/**
 * Interfacial heat transfer model.
 *
 * Computes interfacial mass transfer (Γ) and heat fluxes (q_i_l, q_i_v)
 * from the local two-phase state.  Different implementations represent
 * different physics: linear relaxation, boiling curves, Henry-Fauske, etc.
 */
struct HeatTransferModel {
    virtual ~HeatTransferModel() = default;
    virtual ClosureResult evaluate(const InterfacialState& s) const = 0;
};

/**
 * Drift velocity model.
 *
 * Computes the distribution parameter C_0 and drift velocity V_gj
 * that relate phasic velocities to the mixture velocity via drift-flux.
 * Different implementations: Zuber-Findlay, Wallis, Chexal-Lellouche, etc.
 */
struct DriftVelocityModel {
    virtual ~DriftVelocityModel() = default;
    virtual DriftFluxResult evaluate(const InterfacialState& s) const = 0;
};

// ===========================================================================
// Concrete sub-model implementations
// ===========================================================================

/**
 * Linear relaxation heat transfer with nucleation onset.
 *
 *   q_i_l = H_i · a_i · (T_sat - T_l)
 *   Γ = -q_i_l / h_fg
 *   q_i_v from interface energy balance
 *
 * Interfacial area: a_i = max(4α(1-α), α)
 * Nucleation: when T_l > T_sat, enforce α ≥ alpha_nucleation.
 */
class LinearRelaxation : public HeatTransferModel {
public:
    /// @param H_i   Interfacial heat transfer coefficient [W/(m³·K)]
    ///              Typical: 1e5 (slow transients) to 1e7 (rapid blowdowns)
    /// @param alpha_nucleation  Minimum void fraction when T_l > T_sat [-]
    explicit LinearRelaxation(double H_i = 1e5, double alpha_nucleation = 1e-3)
        : H_i_(H_i), alpha_nucleation_(alpha_nucleation) {}

    ClosureResult evaluate(const InterfacialState& s) const override {
        ClosureResult r{};

        double h_fg = s.h_sat_v - s.h_sat_l;
        if (h_fg < 1.0) h_fg = 1.0;  // avoid division by zero near critical point

        double alpha_eff = s.alpha;

        // ── Nucleation onset ──
        // When liquid is superheated (T_l > T_sat) but no vapor exists
        // (α ≈ 0), flashing cannot start because the interfacial area
        // is zero. This is the "nucleation" problem — physically, vapor
        // nucleates on wall imperfections and dissolved gas sites.
        //
        // Model: when T_l > T_sat, enforce a minimum void fraction so
        // the interfacial heat transfer has area to work with. This is
        // equivalent to RELAP5's flashing onset model.
        if (s.T_l > s.T_sat && alpha_eff < alpha_nucleation_) {
            alpha_eff = alpha_nucleation_;
        }

        // ── Interfacial area ──
        // Standard model: a_i = 4α(1-α), parabolic, zero at boundaries.
        // Enhancement at low void: max(4α(1-α), α) to maintain area
        // during nucleation/early flashing. Without this, the growth
        // rate is too slow at α ~ 0.001 to compete with advective loss.
        // Physically: at low α, bubbles are small and surface-area/volume
        // ratio is high (scales as α^{2/3} for spheres, not α).
        double a_i = std::max(4.0 * alpha_eff * (1.0 - alpha_eff), alpha_eff);

        // ── Interfacial heat transfer ──
        // q_i_l = heat flux INTO liquid from interface [W/m³]
        // When T_l > T_sat: q_i_l < 0 (heat leaves liquid → drives evaporation)
        // When T_l < T_sat: q_i_l > 0 (heat enters liquid → drives condensation)
        // Γ = -q_i_l / h_fg: positive = evaporation, negative = condensation
        r.q_i_l = H_i_ * a_i * (s.T_sat - s.T_l);
        r.Gamma = -r.q_i_l / h_fg;

        // Interface energy balance: q_i_l + q_i_v + Γ·(h_v - h_l) = 0
        // Enforce balance using actual phasic enthalpies (not saturation)
        // so that mixture energy is conserved exactly:
        r.q_i_v = -r.Gamma * (s.h_v - s.h_l) - r.q_i_l;

        return r;
    }

    double H_i() const { return H_i_; }
    double alpha_nucleation() const { return alpha_nucleation_; }

private:
    double H_i_;
    double alpha_nucleation_;
};

/**
 * Zuber-Findlay drift velocity for churn-turbulent bubbly flow.
 *
 *   C_0 = distribution parameter (1.13 for round tube bubbly)
 *   V_gj = 1.41 · [σ·g·Δρ / ρ_l²]^0.25
 *
 * Ref: Ishii & Hibiki, "Thermo-Fluid Dynamics of Two-Phase Flow", Eq. 11.21
 *
 * OPAL regularization: V_gj scaled by 4α(1-α) to smoothly recover HEM
 * behavior at single-phase limits.
 */
class ZuberFindlay : public DriftVelocityModel {
public:
    /// @param C_0  Distribution parameter [-], default 1.13
    explicit ZuberFindlay(double C_0 = 1.13) : C_0_(C_0) {}

    DriftFluxResult evaluate(const InterfacialState& s) const override {
        DriftFluxResult r{};
        r.C_0 = C_0_;

        double drho = s.rho_l - s.rho_v;
        if (drho < 0.01) drho = 0.01;  // clamp near critical
        double rho_l2 = s.rho_l * s.rho_l;
        if (rho_l2 < 1.0) rho_l2 = 1.0;

        double g = s.g_mag;
        r.V_gj = 1.41 * std::pow(s.sigma * g * drho / rho_l2, 0.25);

        // OPAL regularization: scale V_gj by 4α(1-α) to smoothly
        // transition to zero at single-phase limits. This is NOT from
        // the original Zuber-Findlay correlation (which defines V_gj
        // for all α > 0). The scaling prevents numerical issues when
        // α ≈ 0 or α ≈ 1 and ensures the 5-eq model exactly recovers
        // HEM behavior in single-phase limits.
        double scale = 4.0 * s.alpha * (1.0 - s.alpha);
        r.V_gj *= scale;

        return r;
    }

    double C_0() const { return C_0_; }

private:
    double C_0_;
};

// ===========================================================================
// Composed closures
// ===========================================================================

/**
 * Drift-flux closures — composes a HeatTransferModel and DriftVelocityModel.
 *
 * This is the standard closure set for 4-eq and 5-eq drift-flux models.
 * Sub-models are injected at construction and can be swapped independently.
 */
class DriftFluxClosures : public InterfacialClosures {
public:
    DriftFluxClosures(const HeatTransferModel& ht, const DriftVelocityModel& drift)
        : ht_(ht), drift_(drift) {}

    ClosureResult compute(const InterfacialState& s) const override {
        return ht_.evaluate(s);
    }

    DriftFluxResult drift_flux(const InterfacialState& s) const override {
        return drift_.evaluate(s);
    }

private:
    const HeatTransferModel& ht_;
    const DriftVelocityModel& drift_;
};

} // namespace opal
