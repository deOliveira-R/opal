#pragma once
/**
 * solver.hpp — Semi-implicit staggered-mesh single-phase solver.
 *
 * Grid layout (0-indexed throughout):
 *   Cell centres : p[0..N-1], T[0..N-1]
 *   Cell faces   : mdot[0..N]
 *                  mdot[0]   = inlet face  (between p_in BC and cell 0)
 *                  mdot[i]   = face between cell i-1 and cell i   (i=1..N-1)
 *                  mdot[N]   = outlet face (between cell N-1 and p_out BC)
 *
 * One timestep (semi-implicit):
 *   1. Implicit pressure: tridiagonal system (Thomas algorithm), O(N)
 *   2. Explicit flows:    mdot[i] from new pressures
 *   3. Explicit energy:   forward-Euler T update using new mdot
 *
 * Physics (from ScalablePipe extracted equations):
 *   Momentum (algebraic):
 *     mdot[0]   = (p_in  - p[0])   / R
 *     mdot[i]   = (p[i-1] - p[i])  / R   i=1..N-1
 *     mdot[N]   = (p[N-1] - p_out) / R
 *
 *   Mass ODE:
 *     C dp[i]/dt = mdot[i] - mdot[i+1]
 *
 *   Energy ODE (simplified first-order upwind):
 *     rho*V dT[i]/dt = mdot[i]*(T_in - T[i]) - mdot[i+1]*T[i]
 */

#include <vector>
#include <stdexcept>
#include <cstddef>

namespace opal {

struct BoundaryConditions {
    double p_in;   ///< Inlet pressure  [Pa]
    double p_out;  ///< Outlet pressure [Pa]
    double T_in;   ///< Inlet temperature [K]
};

/**
 * SinglePhaseSolver — pre-factored staggered-mesh pipe.
 *
 * Construct once for a given grid; call step() repeatedly.
 * Thread-safe per-instance (no shared mutable state between instances).
 */
class SinglePhaseSolver {
public:
    /**
     * @param N    Number of cells
     * @param R    Cell friction resistance [Pa/(kg/s)]
     * @param C    Cell hydraulic compressibility [kg/Pa]
     * @param rho  Fluid density [kg/m³]
     * @param Cp   Specific heat [J/(kg·K)]
     * @param V    Cell volume [m³]
     */
    SinglePhaseSolver(int N, double R, double C,
                      double rho, double Cp, double V);

    /**
     * Advance one timestep.
     *
     * @param p    Cell pressures, length N (updated in-place)
     * @param T    Cell temperatures, length N (updated in-place)
     * @param mdot Face flows, length N+1 (updated in-place)
     * @param bc   Boundary conditions (const)
     * @param dt   Timestep [s]
     */
    void step(std::vector<double>& p,
              std::vector<double>& T,
              std::vector<double>& mdot,
              const BoundaryConditions& bc,
              double dt) const;

    /**
     * Run n_steps timesteps, collecting state snapshots every stride steps.
     *
     * Returns a flat vector of length (n_snapshots * (2N+1)):
     *   [ p[0..N-1], T[0..N-1], mdot[0..N] ]  per snapshot.
     *
     * n_snapshots = ceil(n_steps / stride).
     */
    std::vector<double> solve(std::vector<double> p,
                              std::vector<double> T,
                              std::vector<double> mdot,
                              const BoundaryConditions& bc,
                              double dt, int n_steps,
                              int stride = 1) const;

    int    N()   const { return n_; }
    double R()   const { return R_; }
    double C()   const { return C_; }
    double rho() const { return rho_; }
    double Cp()  const { return Cp_; }
    double V()   const { return V_; }

private:
    int    n_;
    double R_, C_, rho_, Cp_, V_;

    // Thomas algorithm scratch (pre-allocated, const across steps)
    mutable std::vector<double> c_prime_;   // modified upper diagonal
    mutable std::vector<double> d_prime_;   // modified RHS

    void solve_pressure(std::vector<double>& p,
                        const BoundaryConditions& bc,
                        double dt) const;
    void update_flows  (const std::vector<double>& p,
                        std::vector<double>& mdot,
                        const BoundaryConditions& bc) const;
    void update_temp   (std::vector<double>& T,
                        const std::vector<double>& mdot,
                        const BoundaryConditions& bc,
                        double dt) const;
};

} // namespace opal
