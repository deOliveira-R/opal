#pragma once
/**
 * five_eq_model.hpp — 5-equation drift-flux FlowModel.
 *
 * State: (P, α, h_l, h_v) at cell centers, mdot_m at faces.
 * 5 variables per cell.
 *
 * Equations:
 *   1. Liquid mass:  ∂/∂t[(1-α)ρ_l] + ∂/∂z[(1-α)ρ_l·v_l] = -Γ
 *   2. Vapor mass:   ∂/∂t[α·ρ_v]    + ∂/∂z[α·ρ_v·v_v]    = +Γ
 *   3. Mixture momentum: algebraic + drift-flux slip
 *   4. Liquid energy: separate h_l with interfacial heat transfer
 *   5. Vapor energy:  separate h_v with interfacial heat transfer
 *
 * Pressure is solved implicitly (tridiagonal, from summed mass equations).
 * Void fraction, phasic enthalpies are updated explicitly.
 * Drift-flux gives phasic velocities from mixture flow.
 *
 * Derived in: derivations/five_eq_pressure_linearization.py
 *             derivations/five_eq_phasic_energy.py
 */

#include "flow_model.hpp"
#include "phasic_properties.hpp"
#include "closures.hpp"

namespace opal {

class FiveEqModel : public FlowModel {
public:
    /**
     * @param phasic   Phasic property evaluator (caller owns)
     * @param closures Interfacial closures (caller owns)
     */
    FiveEqModel(const PhasicProperties& phasic,
                const InterfacialClosures& closures);

    const char* name() const override { return "5-equation drift-flux"; }
    int vars_per_cell() const override { return 5; }
    int state_size(int N) const override { return 4 * N + (N + 1); }

    SolverState make_state(
        const std::vector<double>& p,
        const std::vector<double>& h,
        const std::vector<double>& mdot) const override;

    /// Initialize with full 5-eq state.
    SolverState make_state_5eq(
        const std::vector<double>& p,
        const std::vector<double>& alpha,
        const std::vector<double>& h_l,
        const std::vector<double>& h_v,
        const std::vector<double>& mdot) const;

    void evaluate_properties(
        const SolverState& state,
        const FluidProperties& fluid,
        std::vector<FluidProps>& props) const override;

    void compute_face_resistance(
        const SolverState& state,
        const BoundaryConditions& bc,
        const FluidProperties& fluid,
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
        const std::vector<double>* q_wall) const override;

    void pack_state(
        const SolverState& state,
        std::vector<double>& out) const override;

    SolverState unpack_state(
        const double* data, int N) const override;

private:
    const PhasicProperties& phasic_;
    const InterfacialClosures& closures_;

    // Cached phasic properties (mutable for logical-const)
    mutable std::vector<PhasicProps> phasic_props_;
    mutable std::vector<InterfacialState> iface_states_;
    mutable std::vector<ClosureResult> closure_results_;
    mutable std::vector<DriftFluxResult> drift_results_;

    void compute_phasic_state(
        const SolverState& state,
        const MeshParams& mesh) const;

    /// Split mixture mass flux into phasic fluxes via drift-flux.
    /// Returns (G_l * A, G_v * A) = phasic mass flow rates at a face.
    std::pair<double, double> split_phasic_flux(
        double mdot_m, double alpha_face,
        double rho_l, double rho_v, double rho_m,
        double C_0, double V_gj, double A_flow) const;
};

} // namespace opal
