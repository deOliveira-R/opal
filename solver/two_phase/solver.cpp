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
    const SolverState& /*state*/,
    const BoundaryConditions& bc) const
{
    // Face 0 (inlet)
    if (bc.bc_type_in == BCType::WALL) {
        rho_face_[0] = props_[0].rho;
    } else {
        double rho_in = fluid_.evaluate(bc.p_in, bc.h_in).rho;
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
// New step — operates on SolverState
// ---------------------------------------------------------------------------

void TwoPhaseSolver::step(SolverState& state,
                          const BoundaryConditions& bc,
                          double dt,
                          const std::vector<double>* q_wall) const
{
    if (dt <= 0) throw std::invalid_argument("dt must be > 0");

    auto mesh = mesh_params();

    // Save old pressures for pressure-work term
    std::vector<double> p_old(state.p);

    // 1. Evaluate properties at old state
    model_->evaluate_properties(state, fluid_, props_);

    // 2. Compute face densities
    compute_face_densities(state, bc);

    // 3. Critical flow check (if break BC at outlet)
    CriticalFlowResult cf_result{};
    if (bc.bc_type_out == BCType::BREAK && critical_flow_) {
        int last = n_ - 1;
        double h_mix = state.h_l.empty() ? 0.0 :
            (state.alpha.empty()
                ? state.h_l[last]
                : (1.0 - state.alpha[last]) * state.h_l[last]
                  + state.alpha[last] * (state.h_v.empty()
                      ? state.h_l[last] : state.h_v[last]));

        // Pre-compute momentum estimate at outlet (what mdot would be
        // without critical flow limit) for the choke check.
        double beta = dt * mesh.A_flow / mesh.dx;
        double fric_out = 0.0;
        if (rho_face_[n_] > 0.01) {
            fric_out = mesh.f_D * mesh.dx / (2.0 * mesh.D_h)
                     * std::abs(state.mdot[n_]) * state.mdot[n_]
                     / (rho_face_[n_] * mesh.A_flow * mesh.A_flow);
        }
        double mdot_mom_est = state.mdot[n_]
            + beta * (state.p[last] - bc.p_out) - dt * fric_out;

        cf_result = critical_flow_->evaluate(
            state.p[last], h_mix,
            props_[last].rho, props_[last].drho_dp_h,
            bc.p_out, mesh.A_flow, bc.break_area_fraction,
            mdot_mom_est);
    }

    // 4. Assemble pressure system (momentum model handles algebraic vs inertial)
    momentum_->assemble_pressure_system(
        state, bc, mesh, props_, rho_face_, dt,
        *model_, fluid_, tri_,
        cf_result.is_choked ? &cf_result : nullptr);

    // 5. Solve tridiagonal + pressure bounds
    solve_tridiagonal(state.p);
    // Water-specific pressure bounds (IAPWS-IF97 validity range).
    // p_floor > triple point (611 Pa), p_ceiling < critical point (22.064 MPa).
    // Must be updated if OPAL ever supports other fluids (CO2, sodium, etc.).
    constexpr double p_floor   = 700.0;
    constexpr double p_ceiling = 21.0e6;
    for (int i = 0; i < n_; ++i) {
        state.p[i] = std::clamp(state.p[i], p_floor, p_ceiling);
    }

    // 6. Update velocities (momentum model handles algebraic vs inertial)
    momentum_->update_velocities(
        state, bc, mesh, rho_face_, dt, *model_,
        cf_result.is_choked ? &cf_result : nullptr);

    // CFL check
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

    // 7. Transport update (FlowModel handles equation-set-specific transport)
    model_->update_transport(state, p_old, bc, mesh, props_, *recon_, dt, q_wall);
}

// ---------------------------------------------------------------------------
// Legacy step — wraps new step
// ---------------------------------------------------------------------------

void TwoPhaseSolver::step(std::vector<double>& p,
                          std::vector<double>& h,
                          std::vector<double>& mdot,
                          const TwoPhaseBCs& bc,
                          double dt,
                          const std::vector<double>* q_wall) const
{
    if (static_cast<int>(p.size())    != n_)     throw std::invalid_argument("p size mismatch");
    if (static_cast<int>(h.size())    != n_)     throw std::invalid_argument("h size mismatch");
    if (static_cast<int>(mdot.size()) != n_ + 1) throw std::invalid_argument("mdot size mismatch");
    if (q_wall && static_cast<int>(q_wall->size()) != n_)
        throw std::invalid_argument("q_wall size mismatch");

    SolverState state = model_->make_state(p, h, mdot);
    BoundaryConditions new_bc;
    new_bc.p_in  = bc.p_in;
    new_bc.p_out = bc.p_out;
    new_bc.h_in  = bc.h_in;

    step(state, new_bc, dt, q_wall);

    p    = state.p;
    h    = state.h_l;
    mdot = state.mdot;
}

// ---------------------------------------------------------------------------
// Legacy solve
// ---------------------------------------------------------------------------

std::vector<double> TwoPhaseSolver::solve(std::vector<double> p,
                                           std::vector<double> h,
                                           std::vector<double> mdot,
                                           const TwoPhaseBCs& bc,
                                           double dt, int n_steps,
                                           int stride,
                                           const std::vector<double>* q_wall) const
{
    if (stride < 1) throw std::invalid_argument("stride must be >= 1");

    int state_sz = model_->state_size(n_);
    int n_snap = (n_steps + stride - 1) / stride;
    std::vector<double> out;
    out.reserve(static_cast<size_t>(n_snap) * static_cast<size_t>(state_sz));

    SolverState state = model_->make_state(p, h, mdot);
    BoundaryConditions new_bc;
    new_bc.p_in  = bc.p_in;
    new_bc.p_out = bc.p_out;
    new_bc.h_in  = bc.h_in;

    std::vector<double> snap_buf;

    for (int s = 0; s < n_steps; ++s) {
        step(state, new_bc, dt, q_wall);
        if ((s + 1) % stride == 0 || s == n_steps - 1) {
            model_->pack_state(state, snap_buf);
            out.insert(out.end(), snap_buf.begin(), snap_buf.end());
        }
    }
    return out;
}

} // namespace opal
