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

// ===========================================================================
// Friction model — pluggable wall friction for inertial momentum.
// ===========================================================================

/**
 * Abstract friction model.
 *
 * Computes per-face friction force from old-time flows. Different
 * implementations: single-phase Darcy-Weisbach, two-phase multipliers
 * (Martinelli-Nelson, Friedel, Lockhart-Martinelli), etc.
 */
struct FrictionModel {
    virtual ~FrictionModel() = default;

    /// Compute friction force at each face.
    /// fric[i] has units of [N/m²] (momentum flux).
    virtual void compute(
        const SolverState& state,
        const MeshParams& mesh,
        const std::vector<double>& rho_face,
        std::vector<double>& fric) const = 0;
};

/**
 * Darcy-Weisbach single-phase friction.
 *
 *   fric[i] = (f_D · dx) / (2 · D_h) · |mdot| · mdot / (ρ_face · A²)
 *
 * Uses the constant Darcy factor f_D from MeshParams.
 */
class DarcyFriction : public FrictionModel {
public:
    /// @param rho_face_min  Skip friction if face density below this [kg/m³]
    explicit DarcyFriction(double rho_face_min = 0.01)
        : rho_face_min_(rho_face_min) {}

    void compute(
        const SolverState& state,
        const MeshParams& mesh,
        const std::vector<double>& rho_face,
        std::vector<double>& fric) const override
    {
        int N = mesh.N;
        double geom = mesh.f_D * mesh.dx / (2.0 * mesh.D_h);
        double A2 = mesh.A_flow * mesh.A_flow;

        for (int i = 0; i <= N; ++i) {
            if (rho_face[i] > rho_face_min_) {
                fric[i] = geom * std::abs(state.mdot[i]) * state.mdot[i]
                        / (rho_face[i] * A2);
            }
        }
    }

    double rho_face_min() const { return rho_face_min_; }

private:
    double rho_face_min_;
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
        const FacePressureBC& pbc_in,
        const FacePressureBC& pbc_out,
        const FaceTransportBC& tbc_in,
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
        const FacePressureBC& pbc_in,
        const FacePressureBC& pbc_out,
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
        const FacePressureBC& pbc_in,
        const FacePressureBC& pbc_out,
        const FaceTransportBC& tbc_in,
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
        int N = mesh.N;
        R_face_.resize(N + 1);
        model.compute_face_resistance(state, pbc_in.p_boundary, tbc_in,
                                       fluid, mesh, props, R_face_);
        model.assemble_pressure_system(state, pbc_in.p_boundary, pbc_out.p_boundary,
                                        mesh, props, R_face_, dt, tri);
    }

    void update_velocities(
        SolverState& state,
        const FacePressureBC& pbc_in,
        const FacePressureBC& pbc_out,
        const MeshParams& mesh,
        const std::vector<double>& /*rho_face*/,
        double /*dt*/,
        const FlowModel& model,
        const CriticalFlowResult* /*cf*/,
        const SourceTerms* /*sources*/) const override
    {
        model.update_velocities(state, pbc_in.p_boundary, pbc_out.p_boundary,
                                mesh, R_face_);
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
    /// Default: uses Darcy-Weisbach friction.
    InertialMomentum() : friction_(default_friction()) {}

    /// Explicit friction model injection.
    explicit InertialMomentum(const FrictionModel& friction) : friction_(friction) {}

    const char* name() const override { return "inertial"; }

    void assemble_pressure_system(
        const SolverState& state,
        const FacePressureBC& pbc_in,
        const FacePressureBC& pbc_out,
        const FaceTransportBC& /*tbc_in*/,
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
        bool inlet_wall  = (pbc_in.type  == FacePressureBC::ZERO_FLUX);
        bool outlet_wall = (pbc_out.type == FacePressureBC::ZERO_FLUX);
        bool outlet_choked = cf && cf->is_choked;

        std::vector<double> fric(N + 1, 0.0);
        friction_.compute(state, mesh, rho_face, fric);

        for (int i = 0; i < N; ++i) {
            double alpha_coeff = mesh.V * props[i].drho_dp_h / dt;

            double beta_left = (inlet_wall && i == 0)
                             ? 0.0 : (i == 0 ? 0.0 : beta);

            double beta_right;
            if (i == N - 1) {
                beta_right = (outlet_wall || outlet_choked) ? 0.0 : beta;
            } else {
                beta_right = beta;
            }

            tri.a[i] = (i > 0) ? -beta_left : 0.0;
            tri.c[i] = (i < N - 1) ? -beta_right : 0.0;
            tri.b[i] = alpha_coeff + beta_left + beta_right;
            tri.d[i] = alpha_coeff * state.p[i];

            tri.d[i] += (state.mdot[i] - state.mdot[i + 1])
                       - dt * (fric[i] - fric[i + 1]);

            if (sources && !sources->mass.empty())
                tri.d[i] += sources->mass[i] * mesh.V;
            if (sources && !sources->momentum.empty())
                tri.d[i] += dt * (sources->momentum[i] - sources->momentum[i + 1])
                           * mesh.A_flow;

            // Boundary pressure coupling
            if (i == 0 && !inlet_wall) {
                tri.d[i] += beta_left * pbc_in.p_boundary;
            }
            if (i == N - 1) {
                if (outlet_choked) {
                    tri.d[i] += (state.mdot[N] - cf->mdot_crit);
                } else if (!outlet_wall) {
                    tri.d[i] += beta_right * pbc_out.p_boundary;
                }
            }
        }
    }

    void update_velocities(
        SolverState& state,
        const FacePressureBC& pbc_in,
        const FacePressureBC& pbc_out,
        const MeshParams& mesh,
        const std::vector<double>& rho_face,
        double dt,
        const FlowModel& /*model*/,
        const CriticalFlowResult* cf,
        const SourceTerms* sources) const override
    {
        int N = mesh.N;
        double beta = dt * mesh.A_flow / mesh.dx;
        bool inlet_wall  = (pbc_in.type  == FacePressureBC::ZERO_FLUX);
        bool outlet_wall = (pbc_out.type == FacePressureBC::ZERO_FLUX);

        std::vector<double> fric(N + 1, 0.0);
        friction_.compute(state, mesh, rho_face, fric);

        std::vector<double> mdot_old = state.mdot;

        auto S_mom = [&](int i) -> double {
            return (sources && !sources->momentum.empty())
                   ? dt * sources->momentum[i] * mesh.A_flow : 0.0;
        };

        // Inlet face
        if (inlet_wall) {
            state.mdot[0] = 0.0;
        } else {
            state.mdot[0] = mdot_old[0]
                + beta * (pbc_in.p_boundary - state.p[0]) - dt * fric[0] + S_mom(0);
        }

        // Interior faces
        for (int i = 1; i < N; ++i) {
            state.mdot[i] = mdot_old[i]
                + beta * (state.p[i - 1] - state.p[i]) - dt * fric[i] + S_mom(i);
        }

        // Outlet face
        if (outlet_wall) {
            state.mdot[N] = 0.0;
        } else {
            double mdot_momentum = mdot_old[N]
                + beta * (state.p[N - 1] - pbc_out.p_boundary) - dt * fric[N];

            if (cf && cf->is_choked && mdot_momentum > 0) {
                state.mdot[N] = std::min(mdot_momentum, cf->mdot_crit);
            } else {
                state.mdot[N] = mdot_momentum;
            }
        }
    }

private:
    const FrictionModel& friction_;

    static const DarcyFriction& default_friction() {
        static const DarcyFriction instance;
        return instance;
    }
};

} // namespace opal
