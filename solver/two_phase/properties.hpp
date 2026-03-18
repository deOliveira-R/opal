#pragma once
/**
 * properties.hpp — Fluid property interface for the two-phase solver.
 *
 * The solver calls evaluate(p, h) once per cell per timestep.  The batch
 * method avoids redundant region detection (one region check gives all four
 * outputs).  Concrete implementations: SimpleFluidProperties,
 * IAPWSIF97Properties (iapws97.hpp).
 */

namespace opal {

struct FluidProps {
    double rho;         ///< Density [kg/m^3]
    double drho_dp_h;   ///< d(rho)/dp at constant h [kg/(m^3*Pa)]
    double drho_dh_p;   ///< d(rho)/dh at constant p [kg/(m^3*J/kg)]
    double T;           ///< Temperature [K] (diagnostic)
};

class FluidProperties {
public:
    virtual ~FluidProperties() = default;

    /// Evaluate all properties at (p, h) in one call.
    virtual FluidProps evaluate(double p, double h) const = 0;
};

} // namespace opal
