#pragma once
/**
 * hem_model.hpp — 3-equation Homogeneous Equilibrium Model.
 *
 * Extracted from the original TwoPhaseSolver (Phase 2). Implements FlowModel
 * for the classic HEM semi-implicit scheme:
 *
 *   State: P (pressure), h (mixture enthalpy), mdot (mass flow rate)
 *   3 variables per cell: pressure + enthalpy at centers, flow at faces
 *
 *   - Pressure: implicit tridiagonal (linearized mass conservation)
 *   - Momentum: algebraic (mdot = dp/R)
 *   - Energy: explicit forward Euler with donor-cell/MUSCL reconstruction
 *
 * This is a direct extraction of the proven Phase 2 solver code.
 * All 26 existing tests must pass with this implementation.
 */

#include "flow_model.hpp"

namespace opal {

class HEMModel : public FlowModel {
public:
    const char* name() const override { return "HEM"; }
    int vars_per_cell() const override { return 3; }
    int state_size(int N) const override { return 3 * N + 1; }

    SolverState make_state(
        const std::vector<double>& p,
        const std::vector<double>& h,
        const std::vector<double>& mdot) const override;

    void evaluate_properties(
        const SolverState& state,
        const FluidPackage& fluid,
        std::vector<FluidProps>& props) const override;

    void compute_face_resistance(
        const SolverState& state,
        const BoundaryConditions& bc,
        const FluidPackage& fluid,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        std::vector<double>& R_face) const override;

    void assemble_pressure_system(
        const SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        const std::vector<double>& R_face,
        double dt,
        TridiagCoeffs& tri) const override;

    void update_velocities(
        SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<double>& R_face) const override;

    void update_transport(
        SolverState& state,
        const std::vector<double>& p_old,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        const FaceReconstruction& recon,
        double dt,
        const std::vector<double>* q_wall,
        const SourceTerms* sources = nullptr) const override;

    void pack_state(
        const SolverState& state,
        std::vector<double>& out) const override;

    SolverState unpack_state(
        const double* data, int N) const override;
};

} // namespace opal
