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

} // namespace opal
