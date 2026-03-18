#pragma once
/**
 * flow_model.hpp — Abstract interface for pluggable flow equation sets.
 *
 * The FlowModel encapsulates the equation set (which conservation equations
 * are solved, how many state variables, how the pressure matrix is assembled).
 * The TwoPhaseSolver delegates to a FlowModel for all physics-dependent work
 * while handling time stepping, mesh, and I/O itself.
 *
 * Implementations:
 *   HEMModel     — 3-equation homogeneous equilibrium (Phase 2, extracted)
 *   FiveEqModel  — 5-equation drift-flux (Phase 3b)
 *   FourEqModel  — 4-equation drift-flux (Phase 3c)
 *   SixEqModel   — 6-equation two-fluid (Phase 3d)
 */

#include "solver_state.hpp"
#include "boundary_conditions.hpp"
#include "fluid_package.hpp"
#include "reconstruction.hpp"

#include <vector>

namespace opal {

/// Mesh + geometry parameters passed to FlowModel methods.
struct MeshParams {
    int    N;       ///< Number of cells
    double dx;      ///< Cell length [m]
    double A_flow;  ///< Flow area [m^2]
    double D_h;     ///< Hydraulic diameter [m]
    double f_D;     ///< Darcy friction factor [-]
    double V;       ///< Cell volume = dx * A_flow [m^3]
};

/// Tridiagonal system coefficients (output from assemble_pressure_system).
struct TridiagCoeffs {
    std::vector<double> a;  ///< Sub-diagonal [N]
    std::vector<double> b;  ///< Diagonal [N]
    std::vector<double> c;  ///< Super-diagonal [N]
    std::vector<double> d;  ///< RHS [N]

    void resize(int N) {
        a.resize(N); b.resize(N); c.resize(N); d.resize(N);
    }
};

class FlowModel {
public:
    virtual ~FlowModel() = default;

    /// Human-readable model name (e.g. "HEM", "5-equation drift-flux").
    virtual const char* name() const = 0;

    /// Number of state variables per cell (3 for HEM, 5 for drift-flux, etc.)
    virtual int vars_per_cell() const = 0;

    /// Total state vector size for N cells (includes face variables).
    virtual int state_size(int N) const = 0;

    /**
     * Initialize a SolverState for this model from pressure and enthalpy arrays.
     * Used for backward compatibility (HEM) and default initialization.
     */
    virtual SolverState make_state(
        const std::vector<double>& p,
        const std::vector<double>& h,
        const std::vector<double>& mdot) const = 0;

    /**
     * Evaluate fluid properties at old state.
     * Fills props[0..N-1] from the state's thermodynamic variables.
     */
    virtual void evaluate_properties(
        const SolverState& state,
        const FluidPackage& fluid,
        std::vector<FluidProps>& props) const = 0;

    /**
     * Compute face resistances from properties and mesh.
     * R_face[i] = f_D * dx / (2 * D_h * A^2 * rho_face[i])
     */
    virtual void compute_face_resistance(
        const SolverState& state,
        const BoundaryConditions& bc,
        const FluidPackage& fluid,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        std::vector<double>& R_face) const = 0;

    /**
     * Assemble the semi-implicit pressure system (tridiagonal).
     * Output: coefficients a, b, c, d for Thomas algorithm.
     */
    virtual void assemble_pressure_system(
        const SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        const std::vector<double>& R_face,
        double dt,
        TridiagCoeffs& tri) const = 0;

    /**
     * Back-substitute: given new pressures, update face velocities/flows.
     */
    virtual void update_velocities(
        SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<double>& R_face) const = 0;

    /**
     * Explicit transport update (enthalpy, void fraction, etc.).
     * Called after pressure solve and velocity update.
     *
     * @param sources  Optional generic source terms (MMS, gravity, heating).
     *                 nullptr = no additional sources. Additive with q_wall.
     */
    virtual void update_transport(
        SolverState& state,
        const std::vector<double>& p_old,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        const FaceReconstruction& recon,
        double dt,
        const std::vector<double>* q_wall,
        const SourceTerms* sources = nullptr) const = 0;

    /**
     * Pack state into flat vector for snapshot storage.
     * Layout is model-dependent but must be consistent with unpack_state.
     */
    virtual void pack_state(
        const SolverState& state,
        std::vector<double>& out) const = 0;

    /**
     * Unpack flat vector into SolverState.
     */
    virtual SolverState unpack_state(
        const double* data, int N) const = 0;
};

} // namespace opal
