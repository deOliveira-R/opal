#pragma once
/**
 * solver_state.hpp — Generic state container for multi-equation flow models.
 *
 * Each flow model uses a different subset of these fields:
 *   HEM (3-eq):  p, h_l (as mixture h), mdot
 *   4-eq:        p, alpha, h_l (as mixture h), mdot
 *   5-eq:        p, alpha, h_l, h_v, mdot
 *   6-eq:        p, alpha, h_l, h_v, v_l, v_v
 *
 * Unused fields remain empty (size 0). The FlowModel is responsible for
 * interpreting which fields are active.
 */

#include <vector>

namespace opal {

struct SolverState {
    std::vector<double> p;       ///< Pressure [N cells]
    std::vector<double> alpha;   ///< Void fraction [N cells] (4-eq+)
    std::vector<double> h_l;     ///< Liquid enthalpy [N cells] (or mixture h for HEM/4-eq)
    std::vector<double> h_v;     ///< Vapor enthalpy [N cells] (5-eq+)
    std::vector<double> mdot;    ///< Mixture mass flow [N+1 faces] (HEM through 5-eq)
    std::vector<double> v_l;     ///< Liquid velocity [N+1 faces] (6-eq only)
    std::vector<double> v_v;     ///< Vapor velocity [N+1 faces] (6-eq only)
};

/**
 * Generic source terms for all conservation equations.
 *
 * Every equation in the solver is fundamentally advection with sources.
 * This struct allows injecting arbitrary volumetric sources into each
 * equation, enabling:
 *   - Method of Manufactured Solutions (MMS) verification
 *   - Gravity body forces in momentum
 *   - Neutron heating from point kinetics
 *   - Any future volumetric source/sink
 *
 * All fields are optional (empty = no source). The solver checks .empty()
 * before accessing. Units are volumetric so they are mesh-independent
 * for h-refinement convergence studies.
 *
 * q_wall is preserved separately for backward compatibility — it handles
 * wall heat transfer (physics). SourceTerms.energy_l/v handle additional
 * forcing (MMS, nuclear heating, etc.). They add.
 */
struct SourceTerms {
    // Per-cell volumetric sources [size N each, or empty]
    std::vector<double> mass;       ///< Mass source [kg/(m³·s)] in continuity eq
    std::vector<double> energy_l;   ///< Liquid energy source [W/m³]
    std::vector<double> energy_v;   ///< Vapor energy source [W/m³]
    std::vector<double> void_frac;  ///< Vapor mass source [kg/(m³·s)] (like Gamma)

    // Per-face source [size N+1, or empty]
    std::vector<double> momentum;   ///< Body force [N/m³] in momentum eq

    bool empty() const {
        return mass.empty() && energy_l.empty() && energy_v.empty()
            && void_frac.empty() && momentum.empty();
    }
};

} // namespace opal
