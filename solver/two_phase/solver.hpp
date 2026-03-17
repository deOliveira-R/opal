#pragma once
/**
 * solver.hpp — Semi-implicit staggered-mesh two-phase solver.
 *
 * Grid layout (0-indexed):
 *   Cell centres : p[0..N-1], h[0..N-1]   (pressure, enthalpy)
 *   Cell faces   : mdot[0..N]              (mass flow rate)
 *
 * One timestep (semi-implicit, operator-split):
 *   1. Evaluate properties at old state → rho, drho_dp_h, drho_dh_p per cell
 *   2. Compute face resistances (density-dependent friction)
 *   3. Implicit pressure: tridiagonal system with variable coefficients
 *   4. Algebraic flows from new pressures
 *   5. Explicit enthalpy update (donor-cell upwind, forward Euler)
 *
 * Key difference from single-phase (Phase 1):
 *   - State is (p, h, mdot) not (p, T, mdot)
 *   - Properties vary in space and time: rho(p,h), not constant rho
 *   - Tridiagonal coefficients vary per cell (from local drho_dp_h)
 *   - Face resistance varies per face (from local density)
 *   - Energy equation includes pressure-work term V*dp/dt
 *
 * Thread safety: NOT thread-safe per instance.  Mutable scratch arrays
 * (props_, R_face_, Thomas coefficients) are shared across calls to step().
 * Each thread must use its own TwoPhaseSolver instance.
 *
 * CFL constraint: the explicit enthalpy update requires dt < rho*V/|mdot|.
 * The solver prints a one-time warning to stderr if this is violated.
 */

#include "properties.hpp"

#include <vector>
#include <stdexcept>

namespace opal {

struct TwoPhaseBCs {
    double p_in;    ///< Inlet pressure [Pa]
    double p_out;   ///< Outlet pressure [Pa]
    double h_in;    ///< Inlet enthalpy [J/kg]
};

class TwoPhaseSolver {
public:
    /**
     * @param N      Number of cells
     * @param dx     Cell length [m]
     * @param A_flow Flow area [m^2]
     * @param D_h    Hydraulic diameter [m]
     * @param f_D    Darcy friction factor [-] (constant for now)
     * @param fluid  Property evaluator (caller owns lifetime)
     */
    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidProperties& fluid);

    /**
     * Advance one timestep.
     *
     * @param p      Cell pressures [N], updated in-place
     * @param h      Cell enthalpies [N], updated in-place
     * @param mdot   Face mass flows [N+1], updated in-place
     * @param bc     Boundary conditions
     * @param dt     Timestep [s]
     * @param q_wall Wall heat per cell [W], length N (nullptr → 0)
     */
    void step(std::vector<double>& p,
              std::vector<double>& h,
              std::vector<double>& mdot,
              const TwoPhaseBCs& bc,
              double dt,
              const std::vector<double>* q_wall = nullptr) const;

    /**
     * Run n_steps timesteps, collecting snapshots every stride steps.
     *
     * Returns flat vector, per snapshot: [p[N], h[N], mdot[N+1]] = 3N+1 doubles.
     */
    std::vector<double> solve(std::vector<double> p,
                              std::vector<double> h,
                              std::vector<double> mdot,
                              const TwoPhaseBCs& bc,
                              double dt, int n_steps,
                              int stride = 1,
                              const std::vector<double>* q_wall = nullptr) const;

    // Accessors
    int    N()      const { return n_; }
    double dx()     const { return dx_; }
    double A_flow() const { return A_; }
    double D_h()    const { return D_h_; }
    double f_D()    const { return f_D_; }
    double V()      const { return V_; }

private:
    int    n_;
    double dx_, A_, D_h_, f_D_;
    double V_;   // = dx * A (cell volume)
    const FluidProperties& fluid_;

    // Per-cell scratch (mutable: logical const, computational scratch)
    mutable std::vector<FluidProps> props_;      // property cache
    mutable std::vector<double>     R_face_;     // face resistance [N+1]
    mutable bool cfl_warned_ = false;            // one-shot CFL warning flag

    // Thomas algorithm scratch
    mutable std::vector<double> a_, b_, c_, d_;  // tridiagonal coefficients
    mutable std::vector<double> c_prime_, d_prime_;

    void evaluate_properties(const std::vector<double>& p,
                             const std::vector<double>& h) const;
    void compute_face_resistance(const TwoPhaseBCs& bc) const;
    void solve_pressure(std::vector<double>& p,
                        const TwoPhaseBCs& bc,
                        double dt) const;
    void update_flows(const std::vector<double>& p,
                      std::vector<double>& mdot,
                      const TwoPhaseBCs& bc) const;
    void update_enthalpy(std::vector<double>& h,
                         const std::vector<double>& p,
                         const std::vector<double>& p_old,
                         const std::vector<double>& mdot,
                         const TwoPhaseBCs& bc,
                         double dt,
                         const std::vector<double>* q_wall) const;
};

} // namespace opal
