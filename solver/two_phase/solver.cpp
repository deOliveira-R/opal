/**
 * solver.cpp — Semi-implicit staggered-mesh two-phase solver implementation.
 *
 * Semi-implicit pressure step derivation
 * ---------------------------------------
 * Mass conservation per cell:
 *
 *   V * d(rho_i)/dt = mdot[i] - mdot[i+1]
 *
 * Chain rule with h frozen at old value (operator split):
 *
 *   V * drho_dp_h_i / dt * (p_new[i] - p_old[i])
 *     = mdot[i](p_new) - mdot[i+1](p_new)
 *
 * Momentum (algebraic):
 *
 *   mdot[face] = (p_left - p_right) / R_face
 *
 * where R_face = f_D * dx / (2 * D_h * A^2 * rho_face).
 *
 * Substituting gives a tridiagonal system with variable coefficients:
 *
 *   a[i]*p[i-1] + b[i]*p[i] + c[i]*p[i+1] = d[i]
 *
 * Solved with Thomas algorithm (O(N)).
 *
 * Enthalpy update (explicit, donor-cell upwind):
 *
 *   rho_i * V * (h_new[i] - h_old[i]) / dt
 *     = mdot[i]*(h_face_in - h[i]) - mdot[i+1]*(h_face_out - h[i])
 *       + V * (p_new[i] - p_old[i]) / dt   [pressure work]
 *       + q_wall[i]
 */

#include "solver.hpp"
#include <cstring>
#include <algorithm>

namespace opal {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TwoPhaseSolver::TwoPhaseSolver(int N, double dx, double A_flow, double D_h,
                               double f_D, const FluidProperties& fluid)
    : n_(N), dx_(dx), A_(A_flow), D_h_(D_h), f_D_(f_D),
      V_(dx * A_flow), fluid_(fluid),
      props_(N), R_face_(N + 1),
      a_(N), b_(N), c_(N), d_(N),
      c_prime_(N), d_prime_(N)
{
    if (N < 1)      throw std::invalid_argument("N must be >= 1");
    if (dx <= 0)    throw std::invalid_argument("dx must be > 0");
    if (A_flow <= 0) throw std::invalid_argument("A_flow must be > 0");
    if (D_h <= 0)   throw std::invalid_argument("D_h must be > 0");
    if (f_D < 0)    throw std::invalid_argument("f_D must be >= 0");
}

// ---------------------------------------------------------------------------
// Public interface
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
    if (dt <= 0) throw std::invalid_argument("dt must be > 0");
    if (q_wall && static_cast<int>(q_wall->size()) != n_)
        throw std::invalid_argument("q_wall size mismatch");

    // Save old pressures for enthalpy update (pressure-work term)
    std::vector<double> p_old(p);

    // 1. Evaluate properties at old state
    evaluate_properties(p, h);

    // 2. Compute density-dependent face resistances
    compute_face_resistance(bc);

    // 3. Implicit pressure solve
    solve_pressure(p, bc, dt);

    // 4. Algebraic flow update from new pressures
    update_flows(p, mdot, bc);

    // 5. Explicit enthalpy update (donor-cell, forward Euler)
    update_enthalpy(h, p, p_old, mdot, bc, dt, q_wall);
}

std::vector<double> TwoPhaseSolver::solve(std::vector<double> p,
                                           std::vector<double> h,
                                           std::vector<double> mdot,
                                           const TwoPhaseBCs& bc,
                                           double dt, int n_steps,
                                           int stride,
                                           const std::vector<double>* q_wall) const
{
    if (stride < 1) throw std::invalid_argument("stride must be >= 1");

    int n_snap = (n_steps + stride - 1) / stride;
    int state_size = 3 * n_ + 1;  // p[N] + h[N] + mdot[N+1]
    std::vector<double> out;
    out.reserve(static_cast<size_t>(n_snap) * static_cast<size_t>(state_size));

    for (int s = 0; s < n_steps; ++s) {
        step(p, h, mdot, bc, dt, q_wall);
        if ((s + 1) % stride == 0 || s == n_steps - 1) {
            out.insert(out.end(), p.begin(), p.end());
            out.insert(out.end(), h.begin(), h.end());
            out.insert(out.end(), mdot.begin(), mdot.end());
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Private — property evaluation
// ---------------------------------------------------------------------------

void TwoPhaseSolver::evaluate_properties(const std::vector<double>& p,
                                          const std::vector<double>& h) const
{
    for (int i = 0; i < n_; ++i) {
        props_[i] = fluid_.evaluate(p[i], h[i]);
    }
}

void TwoPhaseSolver::compute_face_resistance(const TwoPhaseBCs& bc) const
{
    // R_face = f_D * dx / (2 * D_h * A^2 * rho_face)
    // rho_face: arithmetic average of adjacent cell densities, or BC density at boundary

    double geom = f_D_ * dx_ / (2.0 * D_h_ * A_ * A_);

    // Face 0 (inlet): average of inlet BC density and cell 0 density
    double rho_in = fluid_.evaluate(bc.p_in, bc.h_in).rho;
    double rho_face_0 = 0.5 * (rho_in + props_[0].rho);
    R_face_[0] = geom / rho_face_0;

    // Interior faces
    for (int i = 1; i < n_; ++i) {
        double rho_face = 0.5 * (props_[i - 1].rho + props_[i].rho);
        R_face_[i] = geom / rho_face;
    }

    // Face N (outlet): use last cell density (outflow)
    R_face_[n_] = geom / props_[n_ - 1].rho;
}

// ---------------------------------------------------------------------------
// Private — semi-implicit pressure solve
// ---------------------------------------------------------------------------

void TwoPhaseSolver::solve_pressure(std::vector<double>& p,
                                     const TwoPhaseBCs& bc,
                                     double dt) const
{
    // Build tridiagonal system: a[i]*p[i-1] + b[i]*p[i] + c[i]*p[i+1] = d[i]
    for (int i = 0; i < n_; ++i) {
        double alpha = V_ * props_[i].drho_dp_h / dt;
        double inv_R_left  = 1.0 / R_face_[i];
        double inv_R_right = 1.0 / R_face_[i + 1];

        a_[i] = (i > 0)      ? -inv_R_left  : 0.0;
        c_[i] = (i < n_ - 1) ? -inv_R_right : 0.0;
        b_[i] = alpha + inv_R_left + inv_R_right;
        d_[i] = alpha * p[i];

        // Boundary terms
        if (i == 0)      d_[i] += bc.p_in  * inv_R_left;
        if (i == n_ - 1) d_[i] += bc.p_out * inv_R_right;
    }

    // Thomas algorithm: forward sweep
    c_prime_[0] = c_[0] / b_[0];
    d_prime_[0] = d_[0] / b_[0];

    for (int i = 1; i < n_; ++i) {
        double denom = b_[i] - a_[i] * c_prime_[i - 1];
        c_prime_[i] = c_[i] / denom;
        d_prime_[i] = (d_[i] - a_[i] * d_prime_[i - 1]) / denom;
    }

    // Back substitution
    p[n_ - 1] = d_prime_[n_ - 1];
    for (int i = n_ - 2; i >= 0; --i) {
        p[i] = d_prime_[i] - c_prime_[i] * p[i + 1];
    }
}

// ---------------------------------------------------------------------------
// Private — algebraic flow update
// ---------------------------------------------------------------------------

void TwoPhaseSolver::update_flows(const std::vector<double>& p,
                                   std::vector<double>& mdot,
                                   const TwoPhaseBCs& bc) const
{
    mdot[0] = (bc.p_in - p[0]) / R_face_[0];
    for (int i = 1; i < n_; ++i) {
        mdot[i] = (p[i - 1] - p[i]) / R_face_[i];
    }
    mdot[n_] = (p[n_ - 1] - bc.p_out) / R_face_[n_];
}

// ---------------------------------------------------------------------------
// Private — explicit enthalpy update (donor-cell upwind)
// ---------------------------------------------------------------------------

void TwoPhaseSolver::update_enthalpy(std::vector<double>& h,
                                      const std::vector<double>& p_new,
                                      const std::vector<double>& p_old,
                                      const std::vector<double>& mdot,
                                      const TwoPhaseBCs& bc,
                                      double dt,
                                      const std::vector<double>* q_wall) const
{
    // Energy equation (from mass-conservation subtracted form):
    //
    //   rho_i * V * (h_new - h_old) / dt
    //     = mdot_in * (h_face_in - h_old)
    //     - mdot_out * (h_face_out - h_old)
    //     + V * (p_new - p_old) / dt     [pressure work]
    //     + q_wall
    //
    // Donor-cell upwind for face enthalpies:
    //   h_face = h_upstream (direction determined by sign of mdot)

    for (int i = 0; i < n_; ++i) {
        double rho_i = props_[i].rho;
        double h_old = h[i];

        // Inlet face enthalpy (face i)
        double h_face_in;
        if (mdot[i] >= 0.0) {
            // Flow enters cell from left
            h_face_in = (i == 0) ? bc.h_in : h[i - 1];
        } else {
            // Reverse flow: fluid leaves cell to the left
            h_face_in = h_old;
        }

        // Outlet face enthalpy (face i+1)
        double h_face_out;
        if (mdot[i + 1] >= 0.0) {
            // Flow leaves cell to the right
            h_face_out = h_old;
        } else {
            // Reverse flow: fluid enters from right
            h_face_out = (i == n_ - 1) ? h_old : h[i + 1];
        }

        // Enthalpy fluxes (relative to cell enthalpy)
        double flux = mdot[i] * (h_face_in - h_old)
                    - mdot[i + 1] * (h_face_out - h_old);

        // Pressure work
        double p_work = V_ * (p_new[i] - p_old[i]) / dt;

        // Wall heat
        double q = (q_wall != nullptr) ? (*q_wall)[i] : 0.0;

        // Forward Euler
        h[i] = h_old + dt / (rho_i * V_) * (flux + p_work + q);
    }
}

} // namespace opal
