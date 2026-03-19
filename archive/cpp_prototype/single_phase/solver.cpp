/**
 * solver.cpp — Semi-implicit staggered-mesh single-phase solver implementation.
 *
 * Semi-implicit pressure step derivation
 * ---------------------------------------
 * Substitute the algebraic momentum equation into the mass ODE at the *new*
 * time level (n+1):
 *
 *   C/dt * (p_new[i] - p[i]) = (p_left - p_new[i])/R - (p_new[i] - p_right)/R
 *
 * where p_left  = p_new[i-1]  for i > 0,   or  p_in  for i == 0
 *       p_right = p_new[i+1]  for i < N-1, or  p_out for i == N-1
 *
 * Rearranging gives the tridiagonal system  A * p_new = rhs :
 *
 *   diag    = C/dt + 2/R   (same for every cell)
 *   offdiag = -1/R          (sub and super, same)
 *
 *   rhs[0]     = C/dt * p[0]     + p_in / R
 *   rhs[i]     = C/dt * p[i]                     for 1 <= i <= N-2
 *   rhs[N-1]   = C/dt * p[N-1]  + p_out / R
 *
 * Solved with the standard Thomas (tridiagonal) algorithm.
 */

#include "solver.hpp"
#include <stdexcept>
#include <cstring>

namespace opal {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

SinglePhaseSolver::SinglePhaseSolver(int N, double R, double C,
                                     double rho, double Cp, double V)
    : n_(N), R_(R), C_(C), rho_(rho), Cp_(Cp), V_(V),
      c_prime_(N), d_prime_(N)
{
    if (N < 1)  throw std::invalid_argument("N must be >= 1");
    if (R <= 0) throw std::invalid_argument("R must be > 0");
    if (C <= 0) throw std::invalid_argument("C must be > 0");
    if (rho <= 0) throw std::invalid_argument("rho must be > 0");
    if (V <= 0)   throw std::invalid_argument("V must be > 0");
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void SinglePhaseSolver::step(std::vector<double>& p,
                             std::vector<double>& T,
                             std::vector<double>& mdot,
                             const BoundaryConditions& bc,
                             double dt) const
{
    if (static_cast<int>(p.size())    != n_)     throw std::invalid_argument("p size mismatch");
    if (static_cast<int>(T.size())    != n_)     throw std::invalid_argument("T size mismatch");
    if (static_cast<int>(mdot.size()) != n_ + 1) throw std::invalid_argument("mdot size mismatch");
    if (dt <= 0) throw std::invalid_argument("dt must be > 0");

    // Order matters: T uses old mdot → compute T before updating mdot
    // But we need new pressure before we can get new mdot.
    // Steps:
    //   1. Implicit pressure (uses old p, gives new p)
    //   2. Explicit mdot from new p
    //   3. Explicit T from old T + new mdot  (energy uses new-time flows)

    solve_pressure(p, bc, dt);
    update_flows(p, mdot, bc);
    update_temp(T, mdot, bc, dt);
}

std::vector<double> SinglePhaseSolver::solve(std::vector<double> p,
                                              std::vector<double> T,
                                              std::vector<double> mdot,
                                              const BoundaryConditions& bc,
                                              double dt, int n_steps,
                                              int stride) const
{
    if (stride < 1) throw std::invalid_argument("stride must be >= 1");

    int n_snap = (n_steps + stride - 1) / stride;
    int state_size = 3 * n_ + 1;  // p[N] + T[N] + mdot[N+1]
    std::vector<double> out;
    out.reserve(static_cast<size_t>(n_snap) * static_cast<size_t>(state_size));

    // Snapshot after every `stride` steps; always include the final state.
    for (int s = 0; s < n_steps; ++s) {
        step(p, T, mdot, bc, dt);
        if ((s + 1) % stride == 0 || s == n_steps - 1) {
            out.insert(out.end(), p.begin(), p.end());
            out.insert(out.end(), T.begin(), T.end());
            out.insert(out.end(), mdot.begin(), mdot.end());
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Private — semi-implicit steps
// ---------------------------------------------------------------------------

void SinglePhaseSolver::solve_pressure(std::vector<double>& p,
                                        const BoundaryConditions& bc,
                                        double dt) const
{
    const double alpha = C_ / dt;         // C/dt
    const double beta  = 1.0 / R_;        // 1/R
    const double diag  = alpha + 2.0 * beta;
    const double off   = -beta;

    // Thomas algorithm: forward sweep
    // a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    // a = off, b = diag, c = off  (uniform)
    // d[0]   = alpha*p[0]   + bc.p_in  * beta
    // d[i]   = alpha*p[i]                             (interior)
    // d[N-1] = alpha*p[N-1] + bc.p_out * beta

    // c_prime[0] = c[0] / b[0]
    c_prime_[0] = off / diag;
    // For N=1, cell 0 is both first and last: gets both p_in and p_out BCs
    double rhs0 = alpha * p[0] + bc.p_in * beta;
    if (n_ == 1) rhs0 += bc.p_out * beta;
    d_prime_[0] = rhs0 / diag;

    for (int i = 1; i < n_; ++i) {
        double denom = diag - off * c_prime_[i - 1];
        c_prime_[i] = off / denom;

        double rhs = (i == n_ - 1)
            ? alpha * p[i] + bc.p_out * beta
            : alpha * p[i];

        d_prime_[i] = (rhs - off * d_prime_[i - 1]) / denom;
    }

    // Back substitution
    p[n_ - 1] = d_prime_[n_ - 1];
    for (int i = n_ - 2; i >= 0; --i) {
        p[i] = d_prime_[i] - c_prime_[i] * p[i + 1];
    }
}

void SinglePhaseSolver::update_flows(const std::vector<double>& p,
                                      std::vector<double>& mdot,
                                      const BoundaryConditions& bc) const
{
    mdot[0] = (bc.p_in - p[0]) / R_;
    for (int i = 1; i < n_; ++i) {
        mdot[i] = (p[i - 1] - p[i]) / R_;
    }
    mdot[n_] = (p[n_ - 1] - bc.p_out) / R_;
}

void SinglePhaseSolver::update_temp(std::vector<double>& T,
                                     const std::vector<double>& mdot,
                                     const BoundaryConditions& bc,
                                     double dt) const
{
    // From ScalablePipe extracted equation:
    //   rho*V * der(T[i]) = mdot[i]*(T_in - T[i]) - mdot[i+1]*T[i]
    // Forward Euler:
    //   T_new[i] = T[i] + dt/(rho*V) * (mdot[i]*(T_in - T[i]) - mdot[i+1]*T[i])
    const double coeff = dt / (rho_ * V_);
    for (int i = 0; i < n_; ++i) {
        double dT = coeff * (mdot[i] * (bc.T_in - T[i]) - mdot[i + 1] * T[i]);
        T[i] += dT;
    }
}

} // namespace opal
