#pragma once
/**
 * momentum.hpp — Momentum equation models for the semi-implicit solver.
 *
 * The solver's pressure equation couples to momentum through the face
 * mass flow rates. Two momentum treatments are available:
 *
 *   AlgebraicMomentum — steady-state: mdot = dp / R_face
 *     Good for: steady-state, slow transients, verification
 *     This is the default (preserves all existing test behavior).
 *
 *   InertialMomentum — time-advanced: mdot_new = mdot_old + beta*dp - dt*fric
 *     Good for: fast transients, wave propagation, blowdowns
 *     Required for Edwards blowdown validation.
 *
 * The MomentumModel is a solver-level strategy (NOT per FlowModel), because
 * the momentum equation is the same for all equation sets (HEM, 5-eq, 6-eq).
 */

#include "solver_state.hpp"
#include "boundary_conditions.hpp"
#include "flow_model.hpp"

#include <vector>
#include <cmath>

namespace opal {

// Forward declaration for critical flow result
struct CriticalFlowResult {
    double mdot_crit = 0.0;   ///< Critical mass flow rate [kg/s]
    bool   is_choked = false;  ///< True if outlet is choked
};

/**
 * Abstract momentum model.
 */
class MomentumModel {
public:
    virtual ~MomentumModel() = default;
    virtual const char* name() const = 0;

    /**
     * Build the full tridiagonal pressure system and update velocities.
     *
     * For algebraic: delegates to FlowModel's assemble_pressure_system
     * and update_velocities.
     * For inertial: builds its own pressure matrix from beta = dt*A/dx
     * coupling, old-time flows, and friction.
     *
     * @param state        Current solver state (p, mdot, etc.)
     * @param bc           Boundary conditions
     * @param mesh         Mesh parameters
     * @param props        Per-cell fluid properties
     * @param rho_face     Face densities [N+1] for friction computation
     * @param dt           Timestep
     * @param model        FlowModel (used by algebraic for delegation)
     * @param fluid        FluidProperties (used by algebraic for face R)
     * @param tri          Output tridiagonal coefficients
     * @param cf           Critical flow result (nullptr if no critical flow)
     */
    virtual void assemble_pressure_system(
        const SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        const std::vector<double>& rho_face,
        double dt,
        const FlowModel& model,
        const FluidPackage& fluid,
        TridiagCoeffs& tri,
        const CriticalFlowResult* cf,
        const SourceTerms* sources = nullptr) const = 0;

    /**
     * Update face mass flow rates after pressure solve.
     */
    virtual void update_velocities(
        SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<double>& rho_face,
        double dt,
        const FlowModel& model,
        const CriticalFlowResult* cf,
        const SourceTerms* sources = nullptr) const = 0;
};

/**
 * Algebraic (steady-state) momentum: mdot = dp / R_face.
 * This is the existing behavior — delegates to FlowModel.
 */
class AlgebraicMomentum : public MomentumModel {
public:
    const char* name() const override { return "algebraic"; }

    void assemble_pressure_system(
        const SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        const std::vector<double>& /*rho_face*/,
        double dt,
        const FlowModel& model,
        const FluidPackage& fluid,
        TridiagCoeffs& tri,
        const CriticalFlowResult* /*cf*/,
        const SourceTerms* /*sources*/) const override
    {
        // Delegate entirely to FlowModel (which computes its own R_face)
        // Note: algebraic momentum ignores source terms (no time derivative)
        int N = mesh.N;
        R_face_.resize(N + 1);
        model.compute_face_resistance(state, bc, fluid, mesh, props, R_face_);
        model.assemble_pressure_system(state, bc, mesh, props, R_face_, dt, tri);
    }

    void update_velocities(
        SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<double>& /*rho_face*/,
        double /*dt*/,
        const FlowModel& model,
        const CriticalFlowResult* /*cf*/,
        const SourceTerms* /*sources*/) const override
    {
        // Use the SAME R_face computed in assemble_pressure_system.
        // The solver guarantees these are called in sequence.
        model.update_velocities(state, bc, mesh, R_face_);
    }

private:
    // R_face computed once in assemble, reused in update_velocities.
    // The solver calls these in strict sequence within a single timestep.
    mutable std::vector<double> R_face_;
};

/**
 * Inertial momentum: mdot_new = mdot_old + beta*(p_left - p_right) - dt*fric
 *
 * The pressure matrix uses beta = dt*A/dx for face coupling instead of 1/R.
 * This enables finite wave speed and is required for transient problems
 * (blowdowns, water hammer, pressure wave propagation).
 *
 * Derived from: ρA ∂v/∂t = -A ∂P/∂z - f_wall
 * Discretized:  mdot^{n+1} = mdot^n + (dt·A/dx)·(P_left - P_right) - dt·fric
 *
 * Reference: RELAP5/MOD3 Code Manual, Volume 1, Section 3.1
 */
class InertialMomentum : public MomentumModel {
public:
    const char* name() const override { return "inertial"; }

    void assemble_pressure_system(
        const SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<FluidProps>& props,
        const std::vector<double>& rho_face,
        double dt,
        const FlowModel& /*model*/,
        const FluidPackage& /*fluid*/,
        TridiagCoeffs& tri,
        const CriticalFlowResult* cf,
        const SourceTerms* sources) const override
    {
        int N = mesh.N;
        tri.resize(N);

        double beta = dt * mesh.A_flow / mesh.dx;

        // Compute old-time friction per face
        std::vector<double> fric(N + 1, 0.0);
        compute_friction(state, mesh, rho_face, fric);

        // Determine if outlet is choked
        bool outlet_choked = cf && cf->is_choked;

        for (int i = 0; i < N; ++i) {
            double alpha_coeff = mesh.V * props[i].drho_dp_h / dt;

            // Wall BC at inlet: no left coupling
            double beta_left = (bc.bc_type_in == BCType::WALL && i == 0)
                             ? 0.0 : (i == 0 ? 0.0 : beta);

            // Choked/wall at outlet: no right coupling
            double beta_right;
            if (i == N - 1) {
                if (bc.bc_type_out == BCType::WALL || outlet_choked) {
                    beta_right = 0.0;
                } else {
                    beta_right = beta;
                }
            } else {
                beta_right = beta;
            }

            tri.a[i] = (i > 0) ? -beta_left : 0.0;
            tri.c[i] = (i < N - 1) ? -beta_right : 0.0;
            tri.b[i] = alpha_coeff + beta_left + beta_right;
            tri.d[i] = alpha_coeff * state.p[i];

            // RHS: old-time mass flux imbalance + friction correction
            tri.d[i] += (state.mdot[i] - state.mdot[i + 1])
                       - dt * (fric[i] - fric[i + 1]);

            // Generic source terms (mass source, momentum body force)
            if (sources && !sources->mass.empty())
                tri.d[i] += sources->mass[i] * mesh.V * dt;
            if (sources && !sources->momentum.empty())
                tri.d[i] += dt * (sources->momentum[i] - sources->momentum[i + 1])
                           * mesh.A_flow;

            // Boundary pressure terms
            if (i == 0 && bc.bc_type_in == BCType::PRESSURE) {
                tri.d[i] += beta_left * bc.p_in;
            }
            if (i == N - 1) {
                if (outlet_choked) {
                    // Choked: outlet flow is fixed at mdot_critical
                    tri.d[i] += (state.mdot[N] - cf->mdot_crit);
                } else if (bc.bc_type_out == BCType::PRESSURE
                        || bc.bc_type_out == BCType::BREAK) {
                    tri.d[i] += beta_right * bc.p_out;
                }
                // WALL: no boundary term (beta_right = 0)
            }
        }
    }

    void update_velocities(
        SolverState& state,
        const BoundaryConditions& bc,
        const MeshParams& mesh,
        const std::vector<double>& rho_face,
        double dt,
        const FlowModel& /*model*/,
        const CriticalFlowResult* cf,
        const SourceTerms* sources) const override
    {
        int N = mesh.N;
        double beta = dt * mesh.A_flow / mesh.dx;

        // Friction (same as pressure assembly)
        std::vector<double> fric(N + 1, 0.0);
        compute_friction(state, mesh, rho_face, fric);

        // Save old mdot for inertial update
        std::vector<double> mdot_old = state.mdot;

        // Momentum source (body force per unit volume at faces)
        auto S_mom = [&](int i) -> double {
            return (sources && !sources->momentum.empty())
                   ? dt * sources->momentum[i] * mesh.A_flow : 0.0;
        };

        // Inlet face
        if (bc.bc_type_in == BCType::WALL) {
            state.mdot[0] = 0.0;
        } else {
            state.mdot[0] = mdot_old[0]
                + beta * (bc.p_in - state.p[0]) - dt * fric[0] + S_mom(0);
        }

        // Interior faces
        for (int i = 1; i < N; ++i) {
            state.mdot[i] = mdot_old[i]
                + beta * (state.p[i - 1] - state.p[i]) - dt * fric[i] + S_mom(i);
        }

        // Outlet face
        if (bc.bc_type_out == BCType::WALL) {
            state.mdot[N] = 0.0;
        } else {
            double mdot_momentum = mdot_old[N]
                + beta * (state.p[N - 1] - bc.p_out) - dt * fric[N];

            // Critical flow limiter
            if (cf && cf->is_choked && mdot_momentum > 0) {
                state.mdot[N] = std::min(mdot_momentum, cf->mdot_crit);
            } else {
                state.mdot[N] = mdot_momentum;
            }
        }
    }

private:
    /// Compute friction force per face from old-time flows.
    /// fric[i] = f_D * dx / (2 * D_h) * |mdot|*mdot / (rho_face * A^2)
    void compute_friction(
        const SolverState& state,
        const MeshParams& mesh,
        const std::vector<double>& rho_face,
        std::vector<double>& fric) const
    {
        int N = mesh.N;
        double geom = mesh.f_D * mesh.dx / (2.0 * mesh.D_h);
        double A2 = mesh.A_flow * mesh.A_flow;

        for (int i = 0; i <= N; ++i) {
            if (rho_face[i] > 0.01) {
                fric[i] = geom * std::abs(state.mdot[i]) * state.mdot[i]
                        / (rho_face[i] * A2);
            }
        }
    }
};

} // namespace opal
