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
 * Phase 3 architecture: the solver delegates equation-set-specific work to
 * a FlowModel (strategy pattern). The default is HEMModel for backward
 * compatibility. All original API signatures are preserved.
 *
 * Thread safety: NOT thread-safe per instance.
 */

#include "properties.hpp"
#include "reconstruction.hpp"
#include "flow_model.hpp"
#include "hem_model.hpp"

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
     * Legacy constructors (backward compatible — use HEMModel internally).
     */
    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidProperties& fluid);

    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidProperties& fluid,
                   const FaceReconstruction& recon);

    /**
     * New constructor with explicit FlowModel selection.
     */
    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidProperties& fluid,
                   const FaceReconstruction& recon,
                   const FlowModel& model);

    /**
     * Legacy step — operates on separate p, h, mdot arrays.
     * Delegates to FlowModel internally.
     */
    void step(std::vector<double>& p,
              std::vector<double>& h,
              std::vector<double>& mdot,
              const TwoPhaseBCs& bc,
              double dt,
              const std::vector<double>* q_wall = nullptr) const;

    /**
     * New step — operates on SolverState directly.
     */
    void step(SolverState& state,
              const BoundaryConditions& bc,
              double dt,
              const std::vector<double>* q_wall = nullptr) const;

    /**
     * Legacy solve — collects snapshots as flat array.
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
    const FlowModel& model() const { return *model_; }

private:
    int    n_;
    double dx_, A_, D_h_, f_D_;
    double V_;   // = dx * A (cell volume)
    const FluidProperties& fluid_;
    const FaceReconstruction* recon_;
    const FlowModel* model_;

    static const DonorCell default_donor_cell_;
    static const HEMModel default_hem_model_;

    MeshParams mesh_params() const;

    // Scratch arrays (mutable for logical-const step/solve)
    mutable std::vector<FluidProps> props_;
    mutable std::vector<double>     R_face_;
    mutable TridiagCoeffs           tri_;
    mutable std::vector<double>     c_prime_, d_prime_;  // Thomas scratch
    mutable bool cfl_warned_ = false;

    void solve_tridiagonal(std::vector<double>& p) const;
};

} // namespace opal
