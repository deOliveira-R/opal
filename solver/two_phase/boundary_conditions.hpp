#pragma once
/**
 * boundary_conditions.hpp — Generalized boundary conditions for multi-equation models.
 *
 * Extends the original TwoPhaseBCs (p_in, p_out, h_in) with fields needed
 * by higher-order models (void fraction, phasic enthalpies, phasic velocities).
 *
 * For backward compatibility, TwoPhaseBCs is preserved and can be implicitly
 * converted to BoundaryConditions.
 */

namespace opal {

struct BoundaryConditions {
    // Pressure BCs (always needed)
    double p_in  = 0.0;   ///< Inlet pressure [Pa]
    double p_out = 0.0;   ///< Outlet pressure [Pa]

    // Enthalpy BCs
    double h_in  = 0.0;   ///< Mixture inlet enthalpy [J/kg] (HEM/4-eq)
    double h_l_in = 0.0;  ///< Liquid inlet enthalpy [J/kg] (5-eq/6-eq)
    double h_v_in = 0.0;  ///< Vapor inlet enthalpy [J/kg] (5-eq/6-eq)

    // Void fraction BC
    double alpha_in = 0.0; ///< Inlet void fraction [-] (4-eq+)

    // Velocity BCs (6-eq only)
    double v_l_in = 0.0;  ///< Inlet liquid velocity [m/s]
    double v_v_in = 0.0;  ///< Inlet vapor velocity [m/s]
};

} // namespace opal
