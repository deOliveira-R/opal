#pragma once
/**
 * phasic_properties.hpp — Phasic (per-phase) property interface.
 *
 * The existing FluidProperties interface returns mixture properties at (p,h).
 * For multi-equation models (4-eq, 5-eq, 6-eq), we need separate liquid and
 * vapor properties at a given pressure, plus saturation properties.
 *
 * PhasicProperties is an optional extension: models that need it dynamic_cast
 * the FluidProperties reference. HEM does not need it.
 */

namespace opal {

struct PhasicProps {
    double rho_l;          ///< Saturated liquid density [kg/m³]
    double rho_v;          ///< Saturated vapor density [kg/m³]
    double h_sat_l;        ///< Saturated liquid enthalpy [J/kg]
    double h_sat_v;        ///< Saturated vapor enthalpy [J/kg]
    double T_sat;          ///< Saturation temperature [K]
    double drho_l_dp;      ///< ∂ρ_l/∂P at constant h [kg/(m³·Pa)]
    double drho_v_dp;      ///< ∂ρ_v/∂P at constant h [kg/(m³·Pa)]
    double dh_sat_l_dp;    ///< ∂h_sat_l/∂P [J/(kg·Pa)]
    double dh_sat_v_dp;    ///< ∂h_sat_v/∂P [J/(kg·Pa)]
    double cp_l;           ///< Liquid specific heat [J/(kg·K)]
    double cp_v;           ///< Vapor specific heat [J/(kg·K)]
    double sigma;          ///< Surface tension [N/m]
};

class PhasicProperties {
public:
    virtual ~PhasicProperties() = default;

    /// Evaluate phasic properties at saturation for a given pressure.
    virtual PhasicProps evaluate_phasic(double p) const = 0;

    /// Evaluate liquid density at (p, h_l) — may differ from saturation if subcooled.
    virtual double rho_liquid(double p, double h_l) const = 0;

    /// Evaluate vapor density at (p, h_v) — may differ from saturation if superheated.
    virtual double rho_vapor(double p, double h_v) const = 0;

    /// Liquid temperature from (p, h_l).
    virtual double T_liquid(double p, double h_l) const = 0;

    /// Vapor temperature from (p, h_v).
    virtual double T_vapor(double p, double h_v) const = 0;
};

} // namespace opal
