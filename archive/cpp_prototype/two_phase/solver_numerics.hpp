#pragma once
/**
 * solver_numerics.hpp — Tunable numerical thresholds for the two-phase solver.
 *
 * All thresholds that regularize the solver numerics live here with their
 * default values.  Changing a threshold means editing this file (or setting
 * the value from Python / a future input file) — never the solver .cpp files.
 *
 * Default-constructed SolverNumerics reproduces the original hardcoded behavior.
 */

namespace opal {

struct SolverNumerics {
    // ── Phasic flux split (FiveEqModel::split_phasic_flux) ──
    double alpha_min = 1e-6;              ///< Single-phase shortcut threshold [-]

    // ── Vapor density floor (FiveEqModel::update_transport) ──
    double rv_floor_frac = 0.01;          ///< Floor as fraction of rho_v_sat [-]
    double rv_floor_abs  = 0.01;          ///< Absolute floor [kg/m³]

    // ── Void fraction (FiveEqModel::update_transport) ──
    double alpha_nucleation_floor = 1e-3; ///< Min void when Gamma > 0 [-]

    // ── Phase presence (FiveEqModel::update_transport) ──
    double m_phase_min = 1e-12;           ///< Phase mass threshold [kg]

    // ── Enthalpy bounds (FiveEqModel::update_transport) ──
    double h_l_min = 1e4;                ///< Liquid enthalpy floor [J/kg]
    double h_v_max = 4.0e6;              ///< Vapor enthalpy ceiling [J/kg]

    // ── Face density (CFL check in solver.cpp) ──
    double rho_face_min = 0.01;           ///< Skip CFL if rho_face below this [kg/m³]
};

} // namespace opal
