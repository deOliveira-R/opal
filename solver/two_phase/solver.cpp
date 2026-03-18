/**
 * solver.cpp — Semi-implicit staggered-mesh two-phase solver implementation.
 *
 * Phase 3 refactor: the solver is now a thin orchestrator that delegates
 * equation-set work to a FlowModel. The time-stepping loop, Thomas algorithm,
 * and CFL check remain here. The physics (pressure matrix, flow update,
 * enthalpy update) is in the FlowModel.
 *
 * Legacy API (step with p,h,mdot vectors) is preserved as a wrapper.
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

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

TwoPhaseSolver::TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                               double f_D, const FluidProperties& fluid)
    : TwoPhaseSolver(N, dx, A_flow, D_h, f_D, fluid,
                     default_donor_cell_, default_hem_model_)
{
}

TwoPhaseSolver::TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                               double f_D, const FluidProperties& fluid,
                               const FaceReconstruction& recon)
    : TwoPhaseSolver(N, dx, A_flow, D_h, f_D, fluid, recon, default_hem_model_)
{
}

TwoPhaseSolver::TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                               double f_D, const FluidProperties& fluid,
                               const FaceReconstruction& recon,
                               const FlowModel& model)
    : n_(N), dx_(dx), A_(A_flow), D_h_(D_h), f_D_(f_D),
      V_(dx * A_flow), fluid_(fluid), recon_(&recon), model_(&model),
      props_(N), R_face_(N + 1),
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
// Helper
// ---------------------------------------------------------------------------

MeshParams TwoPhaseSolver::mesh_params() const {
    return {n_, dx_, A_, D_h_, f_D_, V_};
}

// ---------------------------------------------------------------------------
// Thomas algorithm (shared infrastructure — model-independent)
// ---------------------------------------------------------------------------

void TwoPhaseSolver::solve_tridiagonal(std::vector<double>& p) const {
    // Forward sweep
    c_prime_[0] = tri_.c[0] / tri_.b[0];
    d_prime_[0] = tri_.d[0] / tri_.b[0];

    for (int i = 1; i < n_; ++i) {
        double denom = tri_.b[i] - tri_.a[i] * c_prime_[i - 1];
        c_prime_[i] = tri_.c[i] / denom;
        d_prime_[i] = (tri_.d[i] - tri_.a[i] * d_prime_[i - 1]) / denom;
    }

    // Back substitution
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

    // 2. Compute density-dependent face resistances
    model_->compute_face_resistance(state, bc, fluid_, mesh, props_, R_face_);

    // 3. Assemble and solve pressure system
    model_->assemble_pressure_system(state, bc, mesh, props_, R_face_, dt, tri_);
    solve_tridiagonal(state.p);

    // Pressure floor: prevent sub-triple-point pressures that cause
    // property evaluation failures. The IAPWS-IF97 triple point is
    // 611.657 Pa; we use 700 Pa with a small margin.
    constexpr double p_floor = 700.0;  // Pa (just above triple point)
    for (int i = 0; i < n_; ++i) {
        if (state.p[i] < p_floor) state.p[i] = p_floor;
    }

    // 4. Update face velocities/flows from new pressures
    model_->update_velocities(state, bc, mesh, R_face_);

    // CFL check (explicit transport stability)
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

    // 5. Explicit transport update
    model_->update_transport(state, p_old, bc, mesh, props_, *recon_, dt, q_wall);
}

// ---------------------------------------------------------------------------
// Legacy step — wraps new step for backward compatibility
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

    // Convert legacy types to new types
    SolverState state = model_->make_state(p, h, mdot);
    BoundaryConditions new_bc;
    new_bc.p_in  = bc.p_in;
    new_bc.p_out = bc.p_out;
    new_bc.h_in  = bc.h_in;

    // Delegate
    step(state, new_bc, dt, q_wall);

    // Copy back
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
