#pragma once
/**
 * reconstruction.hpp — Modular face-value reconstruction for advective fluxes.
 *
 * The solver calls recon.face_value() to compute the face enthalpy (or any
 * scalar) for the energy advection term.  Different reconstruction schemes
 * give different spatial accuracy:
 *
 *   DonorCell  — first-order upwind (Phase 2 default)
 *   MUSCL      — second-order TVD with pluggable slope limiter:
 *                limiters::minmod, van_leer, superbee, mc
 *
 * The solver stores a const reference to a FaceReconstruction object,
 * just like it stores a FluidProperties reference.  New schemes (WENO, PPM)
 * can be added without touching the solver code.  New slope limiters can
 * be added as one-line functions in the limiters namespace.
 *
 * Derived in: docs/math/derivations/muscl_reconstruction.py
 */

#include <cmath>
#include <algorithm>
#include <functional>

namespace opal {

// ---------------------------------------------------------------------------
// Slope limiters — pluggable via MUSCL constructor.
//
// A limiter phi(r) maps the upwind-to-downwind gradient ratio r to a
// flux-limiting coefficient.  phi=0 gives donor cell, phi=1 gives
// centered differences.  TVD limiters satisfy 0 <= phi(r) <= min(2r, 2).
// ---------------------------------------------------------------------------
namespace limiters {

/// Minmod: most diffusive TVD limiter.  phi(r) = max(0, min(r, 1))
inline double minmod(double r) {
    return std::max(0.0, std::min(r, 1.0));
}

/// Van Leer: smooth TVD limiter, good for void fronts.  phi(r) = (r+|r|)/(1+|r|)
inline double van_leer(double r) {
    return (r + std::abs(r)) / (1.0 + std::abs(r));
}

/// Superbee: least diffusive TVD limiter.  phi(r) = max(0, min(2r,1), min(r,2))
inline double superbee(double r) {
    return std::max({0.0, std::min(2.0 * r, 1.0), std::min(r, 2.0)});
}

/// MC (monotonized central): balanced between minmod and superbee.
/// phi(r) = max(0, min((1+r)/2, 2, 2r))
inline double mc(double r) {
    return std::max(0.0, std::min({(1.0 + r) / 2.0, 2.0, 2.0 * r}));
}

} // namespace limiters

/// Limiter function type: takes gradient ratio r, returns phi in [0, 2].
using LimiterFn = double(*)(double);

/**
 * Abstract base for face-value reconstruction.
 *
 * Convention: the face sits between cell_L (left) and cell_R (right).
 * cell_LL is the cell left of cell_L, cell_RR is right of cell_R.
 * mdot_face > 0 means flow goes L→R (upwind = cell_L).
 *
 * For boundary faces where cell_LL or cell_RR doesn't exist,
 * the caller passes cell_L or cell_R as a fallback (first-order).
 */
struct FaceReconstruction {
    virtual ~FaceReconstruction() = default;

    /// Number of ghost cells the scheme needs on each side of the domain.
    /// DonorCell: 1 (only uses the upwind cell; LL/RR are ignored)
    /// MUSCL:     2 (needs LL for gradient ratio; outer ghost must be
    ///               linearly extrapolated for second-order BCs)
    virtual int ghost_cells() const { return 1; }

    virtual double face_value(
        double cell_LL, double cell_L, double cell_R, double cell_RR,
        double mdot_face) const = 0;
};

/**
 * First-order upwind (donor cell).
 * h_face = h_upwind.  Ignores cell_LL and cell_RR.
 */
struct DonorCell : FaceReconstruction {
    int ghost_cells() const override { return 1; }
    double face_value(
        double /*cell_LL*/, double cell_L, double cell_R, double /*cell_RR*/,
        double mdot_face) const override
    {
        return (mdot_face >= 0.0) ? cell_L : cell_R;
    }
};

/**
 * MUSCL — second-order TVD reconstruction with pluggable slope limiter.
 *
 * Usage:
 *   MUSCL recon(limiters::minmod);    // most diffusive
 *   MUSCL recon(limiters::van_leer);  // smooth, good for void fronts
 *   MUSCL recon(limiters::superbee);  // least diffusive, sharpest fronts
 *   MUSCL recon(limiters::mc);        // balanced (monotonized central)
 */
struct MUSCL : FaceReconstruction {
    LimiterFn limiter_;

    explicit MUSCL(LimiterFn limiter = limiters::minmod) : limiter_(limiter) {}

    int ghost_cells() const override { return 2; }

    double face_value(
        double cell_LL, double cell_L, double cell_R, double cell_RR,
        double mdot_face) const override
    {
        if (mdot_face >= 0.0) {
            // Upwind = cell_L, gradient ratio r = (L-LL)/(R-L)
            double delta = cell_R - cell_L;
            if (std::abs(delta) < 1e-30) return cell_L;
            double r = (cell_L - cell_LL) / delta;
            double phi = limiter_(r);
            return cell_L + 0.5 * phi * delta;
        } else {
            // Upwind = cell_R, gradient ratio r = (R-RR)/(L-R)
            double delta = cell_L - cell_R;
            if (std::abs(delta) < 1e-30) return cell_R;
            double r = (cell_R - cell_RR) / delta;
            double phi = limiter_(r);
            return cell_R + 0.5 * phi * delta;
        }
    }
};

// ---------------------------------------------------------------------------
// Stencil builder: constructs the 4-cell stencil (LL, L, R, RR) for a face,
// with ghost cell extrapolation that matches the reconstruction scheme's order.
//
// face_idx: the face between cell face_idx-1 (left) and face_idx (right).
//   face_idx = 0   → inlet face (left side is ghost)
//   face_idx = N   → outlet face (right side is ghost)
//   face_idx = 1..N-1 → interior faces
//
// For n_ghost >= 2 (MUSCL): the outer ghost is linearly extrapolated from
// the BC and the first interior cell, giving a non-zero gradient ratio
// that enables second-order accuracy at boundaries.
//
// For n_ghost = 1 (donor cell): the outer ghost equals the inner ghost
// (constant extrapolation), preserving existing first-order behavior.
// ---------------------------------------------------------------------------

inline void build_stencil(
    const double* field, int N, int face_idx,
    double bc_left, double bc_right,
    int n_ghost,
    double& cell_LL, double& cell_L, double& cell_R, double& cell_RR)
{
    int iL = face_idx - 1;  // cell to the left of face
    int iR = face_idx;      // cell to the right of face

    // Inner neighbors (L and R)
    cell_L = (iL >= 0) ? field[iL] : bc_left;
    cell_R = (iR < N)  ? field[iR] : bc_right;

    if (n_ghost >= 2) {
        // Second-order: linear extrapolation for outer ghost
        if (iL >= 1)       cell_LL = field[iL - 1];
        else if (iL == 0)  cell_LL = 2.0 * bc_left - field[0];
        else                cell_LL = 2.0 * bc_left - cell_R; // face 0: both ghosts

        if (iR < N - 1)         cell_RR = field[iR + 1];
        else if (iR == N - 1)   cell_RR = 2.0 * bc_right - field[N - 1];
        else                     cell_RR = 2.0 * bc_right - cell_L;
    } else {
        // First-order: constant extrapolation (existing behavior)
        cell_LL = (iL >= 1)     ? field[iL - 1] : bc_left;
        cell_RR = (iR < N - 1)  ? field[iR + 1] : cell_R;
    }
}

} // namespace opal
