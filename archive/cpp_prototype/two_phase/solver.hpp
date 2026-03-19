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

class TwoPhaseSolver {
public:
    /// Convenience constructors (default strategies for missing args).
    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidPackage& fluid);

    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidPackage& fluid,
                   const FaceReconstruction& recon);

    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidPackage& fluid,
                   const FaceReconstruction& recon,
                   const FlowModel& model);

    /// Full constructor with all strategies.
    TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                   double f_D, const FluidPackage& fluid,
                   const FaceReconstruction& recon,
                   const FlowModel& model,
                   const MomentumModel& momentum,
                   const CriticalFlowModel* critical_flow = nullptr);

    /**
     * Step with BoundaryFace strategy objects — time-aware.
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

    // Accessors
    int    N()      const { return n_; }
    double dx()     const { return dx_; }
    double A_flow() const { return A_; }
    double D_h()    const { return D_h_; }
    double f_D()    const { return f_D_; }
    double V()      const { return V_; }
    const FlowModel& model() const { return *model_; }
    const SolverNumerics& numerics() const { return numerics_; }

    /// Set gravity for this pipe segment.
    /// @param g_axial  Gravity projection on pipe axis [m/s²].
    ///                 Positive = opposes positive flow direction (upward flow).
    /// @param g_mag    |g⃗| for buoyancy correlations [m/s²].
    void set_gravity(double g_axial, double g_mag) {
        g_axial_ = g_axial;
        g_mag_ = g_mag;
    }

private:
    int    n_;
    double dx_, A_, D_h_, f_D_;
    double V_;
    double g_axial_ = 9.81;
    double g_mag_   = 9.81;
    SolverNumerics numerics_;
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
