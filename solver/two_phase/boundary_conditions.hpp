#pragma once
/**
 * boundary_conditions.hpp — Generalized boundary conditions for multi-equation models.
 *
 * Extends the original TwoPhaseBCs (p_in, p_out, h_in) with fields needed
 * by higher-order models (void fraction, phasic enthalpies, phasic velocities)
 * and by boundary condition types (wall, break with critical flow).
 *
 * For backward compatibility, TwoPhaseBCs is preserved and can be implicitly
 * converted to BoundaryConditions.
 */

namespace opal {

/// Boundary condition type for each face.
enum class BCType {
    PRESSURE,  ///< Specified pressure (default, existing behavior)
    WALL,      ///< Closed wall: mdot = 0, no pressure coupling
    BREAK,     ///< Break/rupture: pressure BC with critical flow limiter
};

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

    // Face BC types (default PRESSURE for backward compatibility)
    BCType bc_type_in  = BCType::PRESSURE;
    BCType bc_type_out = BCType::PRESSURE;

    // Break parameters (only used when bc_type == BREAK)
    double break_area_fraction = 1.0;  ///< Discharge coefficient C_d [-]
};

} // namespace opal
