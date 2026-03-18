#pragma once
/**
 * solver.hpp — Semi-implicit staggered-mesh two-phase solver.
 *
 * Grid layout (0-indexed):
 *   Cell centres : p[0..N-1], h[0..N-1]   (pressure, enthalpy)
 *   Cell faces   : mdot[0..N]              (mass flow rate)
 *
 * One timestep (semi-implicit, operator-split):
 *   1. Evaluate properties at old state
 *   2. Compute face densities
 *   3. Assemble pressure system (algebraic or inertial momentum)
 *   4. Solve tridiagonal pressure system
 *   5. Update face flows (algebraic or inertial momentum)
 *   6. Explicit transport update (enthalpy, void fraction, etc.)
 *
 * Strategies:
 *   FlowModel     — equation set (HEM, 5-eq, etc.)
 *   MomentumModel — momentum treatment (algebraic or inertial)
 *   CriticalFlowModel — choked flow at break (optional)
 *
 * Thread safety: NOT thread-safe per instance.
 */

#include "fluid_package.hpp"
#include "reconstruction.hpp"
#include "flow_model.hpp"
#include "hem_model.hpp"
#include "momentum.hpp"
#include "critical_flow.hpp"

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
     * Legacy constructors (backward compatible — algebraic momentum, no critical flow).
     */
    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidPackage& fluid);

    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidPackage& fluid,
                   const FaceReconstruction& recon);

    /**
     * Constructor with FlowModel selection (algebraic momentum).
     */
    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidPackage& fluid,
                   const FaceReconstruction& recon,
                   const FlowModel& model);

    /**
     * Full constructor with all strategies.
     */
    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidPackage& fluid,
                   const FaceReconstruction& recon,
                   const FlowModel& model,
                   const MomentumModel& momentum,
                   const CriticalFlowModel* critical_flow = nullptr);

    /**
     * Legacy step — operates on separate p, h, mdot arrays.
     */
    void step(std::vector<double>& p,
              std::vector<double>& h,
              std::vector<double>& mdot,
              const TwoPhaseBCs& bc,
              double dt,
              const std::vector<double>* q_wall = nullptr) const;

    /**
     * Step with legacy BoundaryConditions struct.
     */
    void step(SolverState& state,
              const BoundaryConditions& bc,
              double dt,
              const std::vector<double>* q_wall = nullptr,
              const SourceTerms* sources = nullptr) const;

    /**
     * Step with BoundaryFace strategy objects — time-aware.
     *
     * Each boundary face is an independent strategy that produces
     * mathematical contributions (pressure coupling, transport ghost
     * cells, critical flow limiting). The solver evaluates them at
     * time t and applies the results.
     *
     * @param bc_in    Inlet boundary face (left, face 0)
     * @param bc_out   Outlet boundary face (right, face N)
     * @param t        Current simulation time [s]
     */
    void step(SolverState& state,
              const BoundaryFace& bc_in,
              const BoundaryFace& bc_out,
              double t, double dt,
              const std::vector<double>* q_wall = nullptr,
              const SourceTerms* sources = nullptr) const;

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
    double V_;
    const FluidPackage& fluid_;
    const FaceReconstruction* recon_;
    const FlowModel* model_;
    const MomentumModel* momentum_;
    const CriticalFlowModel* critical_flow_;

    static const DonorCell default_donor_cell_;
    static const HEMModel default_hem_model_;
    static const AlgebraicMomentum default_algebraic_momentum_;

    MeshParams mesh_params() const;

    /// Core implementation — no legacy types. Both public step() overloads
    /// decompose their BCs and delegate here.
    void step_internal(SolverState& state,
                       const FacePressureBC& pbc_in,
                       const FacePressureBC& pbc_out,
                       const FaceTransportBC& tbc_in,
                       bool has_critical_flow, double C_d,
                       double dt,
                       const std::vector<double>* q_wall,
                       const SourceTerms* sources) const;

    // Scratch arrays
    mutable std::vector<FluidProps> props_;
    mutable std::vector<double>     rho_face_;
    mutable std::vector<double>     R_face_;
    mutable TridiagCoeffs           tri_;
    mutable std::vector<double>     c_prime_, d_prime_;
    mutable bool cfl_warned_ = false;

    void compute_face_densities(const FacePressureBC& pbc_in,
                                const FaceTransportBC& tbc_in) const;
    void solve_tridiagonal(std::vector<double>& p) const;
};

} // namespace opal
