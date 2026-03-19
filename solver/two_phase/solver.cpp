/**
 * solver.cpp — Semi-implicit staggered-mesh two-phase solver implementation.
 *
 * The solver orchestrates the timestep:
 *   1. Properties at old state
 *   2. Face densities
 *   3. Pressure system assembly (via MomentumModel — algebraic or inertial)
 *   4. Tridiagonal solve + pressure floor
 *   5. Velocity update (via MomentumModel)
 *   6. Transport update (via FlowModel — enthalpy, void fraction)
 *
 * Critical flow (optional): evaluated before pressure assembly when the
 * outlet BC is BREAK. If choked, the outlet face decouples from downstream
 * pressure and the flow rate is limited.
 */

#include "solver.hpp"
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace opal {

// ---------------------------------------------------------------------------
// Static defaults
// ---------------------------------------------------------------------------

const DonorCell TwoPhaseSolver::default_donor_cell_{};
const HEMModel  TwoPhaseSolver::default_hem_model_{};
const AlgebraicMomentum TwoPhaseSolver::default_algebraic_momentum_{};

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

TwoPhaseSolver::TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                               double f_D, const FluidPackage& fluid)
    : TwoPhaseSolver(N, dx, A_flow, D_h, f_D, fluid,
                     default_donor_cell_, default_hem_model_,
                     default_algebraic_momentum_, nullptr)
{
}

TwoPhaseSolver::TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                               double f_D, const FluidPackage& fluid,
                               const FaceReconstruction& recon)
    : TwoPhaseSolver(N, dx, A_flow, D_h, f_D, fluid, recon,
                     default_hem_model_, default_algebraic_momentum_, nullptr)
{
}

TwoPhaseSolver::TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                               double f_D, const FluidPackage& fluid,
                               const FaceReconstruction& recon,
                               const FlowModel& model)
    : TwoPhaseSolver(N, dx, A_flow, D_h, f_D, fluid, recon, model,
                     default_algebraic_momentum_, nullptr)
{
}

TwoPhaseSolver::TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                               double f_D, const FluidPackage& fluid,
                               const FaceReconstruction& recon,
                               const FlowModel& model,
                               const MomentumModel& momentum,
                               const CriticalFlowModel* critical_flow)
    : n_(N), dx_(dx), A_(A_flow), D_h_(D_h), f_D_(f_D),
      V_(dx * A_flow), fluid_(fluid), recon_(&recon), model_(&model),
      momentum_(&momentum), critical_flow_(critical_flow),
      props_(N), rho_face_(N + 1), R_face_(N + 1),
      c_prime_(N), d_prime_(N)
{
    if (N < 1)       throw std::invalid_argument("N must be >= 1");
    if (dx <= 0)     throw std::invalid_argument("dx must be > 0");
    if (A_flow <= 0) throw std::invalid_argument("A_flow must be > 0");
    if (D_h <= 0)    throw std::invalid_argument("D_h must be > 0");
    if (f_D < 0)     throw std::invalid_argument("f_D must be >= 0");

    tri_.resize(N);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

MeshParams TwoPhaseSolver::mesh_params() const {
    return {n_, dx_, A_, D_h_, f_D_, V_};
}

void TwoPhaseSolver::compute_face_densities(
    const FacePressureBC& pbc_in,
    const FaceTransportBC& tbc_in) const
{
    // Face 0 (inlet)
    if (pbc_in.type == FacePressureBC::ZERO_FLUX) {
        rho_face_[0] = props_[0].rho;
    } else {
        double h_in = (tbc_in.h_mix != 0.0) ? tbc_in.h_mix : tbc_in.h_l;
        double rho_in = fluid_.evaluate(pbc_in.p_boundary, h_in).rho;
        rho_face_[0] = 0.5 * (rho_in + props_[0].rho);
    }

    // Interior faces
    for (int i = 1; i < n_; ++i) {
        rho_face_[i] = 0.5 * (props_[i - 1].rho + props_[i].rho);
    }

    // Face N (outlet)
    rho_face_[n_] = props_[n_ - 1].rho;
}

// ---------------------------------------------------------------------------
// Thomas algorithm
// ---------------------------------------------------------------------------

void TwoPhaseSolver::solve_tridiagonal(std::vector<double>& p) const {
    c_prime_[0] = tri_.c[0] / tri_.b[0];
    d_prime_[0] = tri_.d[0] / tri_.b[0];

    for (int i = 1; i < n_; ++i) {
        double denom = tri_.b[i] - tri_.a[i] * c_prime_[i - 1];
        c_prime_[i] = tri_.c[i] / denom;
        d_prime_[i] = (tri_.d[i] - tri_.a[i] * d_prime_[i - 1]) / denom;
    }

    p[n_ - 1] = d_prime_[n_ - 1];
    for (int i = n_ - 2; i >= 0; --i) {
        p[i] = d_prime_[i] - c_prime_[i] * p[i + 1];
    }
}

// ---------------------------------------------------------------------------
// BoundaryFace step — the only public step method
// ---------------------------------------------------------------------------

void TwoPhaseSolver::step(SolverState& state,
                          const BoundaryFace& bc_in,
                          const BoundaryFace& bc_out,
                          double t, double dt,
                          const std::vector<double>* q_wall,
                          const SourceTerms* sources) const
{
    auto pbc_in  = bc_in.pressure(t);
    auto pbc_out = bc_out.pressure(t);
    auto tbc_in  = bc_in.transport(t);

    // Handle break with C_d=0 (fully closed) as wall
    double C_d = bc_out.discharge_coefficient(t);
    bool has_cf = bc_out.has_critical_flow() && critical_flow_ && C_d > 0.0;

    if (bc_out.is_wall() || (bc_out.has_critical_flow() && C_d <= 0.0)) {
        pbc_out = {FacePressureBC::ZERO_FLUX, 0.0};
    }

    step_internal(state, pbc_in, pbc_out, tbc_in, has_cf, C_d,
                  dt, q_wall, sources);
}

// ---------------------------------------------------------------------------
// Core implementation — no legacy types
// ---------------------------------------------------------------------------

void TwoPhaseSolver::step_internal(
    SolverState& state,
    const FacePressureBC& pbc_in,
    const FacePressureBC& pbc_out,
    const FaceTransportBC& tbc_in,
    bool has_critical_flow, double C_d,
    double dt,
    const std::vector<double>* q_wall,
    const SourceTerms* sources) const
{
    if (dt <= 0) throw std::invalid_argument("dt must be > 0");

    auto mesh = mesh_params();
    std::vector<double> p_old(state.p);

    // 1. Evaluate properties at old state
    model_->evaluate_properties(state, fluid_, props_);

    // 2. Compute face densities
    compute_face_densities(pbc_in, tbc_in);

    // 3. Critical flow check
    CriticalFlowResult cf_result{};
    if (has_critical_flow && critical_flow_) {
        int last = n_ - 1;
        double h_mix = state.h_l.empty() ? 0.0 :
            (state.alpha.empty()
                ? state.h_l[last]
                : (1.0 - state.alpha[last]) * state.h_l[last]
                  + state.alpha[last] * (state.h_v.empty()
                      ? state.h_l[last] : state.h_v[last]));

        double beta = dt * mesh.A_flow / mesh.dx;
        double fric_out = 0.0;
        if (rho_face_[n_] > 0.01) {
            fric_out = mesh.f_D * mesh.dx / (2.0 * mesh.D_h)
                     * std::abs(state.mdot[n_]) * state.mdot[n_]
                     / (rho_face_[n_] * mesh.A_flow * mesh.A_flow);
        }
        double mdot_mom_est = state.mdot[n_]
            + beta * (state.p[last] - pbc_out.p_boundary) - dt * fric_out;

        cf_result = critical_flow_->evaluate(
            state.p[last], h_mix,
            props_[last].rho, props_[last].drho_dp_h,
            pbc_out.p_boundary, mesh.A_flow, C_d,
            mdot_mom_est);
    }

    // 4. Assemble pressure system
    // If choked, force outlet to ZERO_FLUX in the pressure matrix
    FacePressureBC pbc_out_eff = pbc_out;
    if (cf_result.is_choked) {
        pbc_out_eff = {FacePressureBC::ZERO_FLUX, pbc_out.p_boundary};
    }

    momentum_->assemble_pressure_system(
        state, pbc_in, pbc_out_eff, tbc_in, mesh, props_, rho_face_, dt,
        *model_, fluid_, tri_,
        cf_result.is_choked ? &cf_result : nullptr,
        sources);

    // 5. Solve tridiagonal + pressure bounds
    solve_tridiagonal(state.p);
    const double p_floor   = fluid_.p_min();
    const double p_ceiling = fluid_.p_max();
    for (int i = 0; i < n_; ++i) {
        state.p[i] = std::clamp(state.p[i], p_floor, p_ceiling);
    }

    // 6. Update velocities
    momentum_->update_velocities(
        state, pbc_in, pbc_out, mesh, rho_face_, dt, *model_,
        cf_result.is_choked ? &cf_result : nullptr,
        sources);

    // 7. CFL check
    if (!cfl_warned_ && !state.mdot.empty()) {
        double dt_cfl = 1e30;
        for (int i = 0; i < n_; ++i) {
            double mdot_max = std::max(std::abs(state.mdot[i]),
                                       std::abs(state.mdot[i + 1]));
            if (mdot_max > 0.0) {
                double dt_local = props_[i].rho * V_ / mdot_max;
                dt_cfl = std::min(dt_cfl, dt_local);
            }
        }
        if (dt > dt_cfl) {
            std::fprintf(stderr,
                "OPAL WARNING: dt=%.3e exceeds enthalpy CFL limit %.3e "
                "(ratio %.1f). Explicit transport update may be unstable.\n",
                dt, dt_cfl, dt / dt_cfl);
            cfl_warned_ = true;
        }
    }

    // 8. Transport update
    model_->update_transport(state, p_old, tbc_in, mesh, props_, *recon_, dt, q_wall, sources);
}

} // namespace opal
