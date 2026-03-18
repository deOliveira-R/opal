#pragma once
/**
 * reconstruction.hpp — Modular face-value reconstruction for advective fluxes.
 *
 * The solver calls recon.face_value() to compute the face enthalpy (or any
 * scalar) for the energy advection term.  Different reconstruction schemes
 * give different spatial accuracy:
 *
 *   DonorCell     — first-order upwind (Phase 2 default)
 *   MUSCL_Minmod  — second-order TVD, most diffusive
 *   MUSCL_VanLeer — second-order TVD, smooth, good for void fronts
 *
 * The solver stores a const reference to a FaceReconstruction object,
 * just like it stores a FluidProperties reference.  New schemes (WENO, PPM)
 * can be added without touching the solver code.
 *
 * Derived in: docs/math/derivations/muscl_reconstruction.py
 */

#include <cmath>
#include <algorithm>

namespace opal {

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

    virtual double face_value(
        double cell_LL, double cell_L, double cell_R, double cell_RR,
        double mdot_face) const = 0;
};

/**
 * First-order upwind (donor cell).
 * h_face = h_upwind.  Ignores cell_LL and cell_RR.
 */
struct DonorCell : FaceReconstruction {
    double face_value(
        double /*cell_LL*/, double cell_L, double cell_R, double /*cell_RR*/,
        double mdot_face) const override
    {
        return (mdot_face >= 0.0) ? cell_L : cell_R;
    }
};

/**
 * MUSCL with minmod limiter — second-order TVD, most diffusive.
 * phi(r) = max(0, min(r, 1))
 */
struct MUSCL_Minmod : FaceReconstruction {
    double face_value(
        double cell_LL, double cell_L, double cell_R, double cell_RR,
        double mdot_face) const override
    {
        if (mdot_face >= 0.0) {
            // Upwind = cell_L, gradient ratio r = (L-LL)/(R-L)
            double delta = cell_R - cell_L;
            if (std::abs(delta) < 1e-30) return cell_L;
            double r = (cell_L - cell_LL) / delta;
            double phi = std::max(0.0, std::min(r, 1.0));
            return cell_L + 0.5 * phi * delta;
        } else {
            // Upwind = cell_R, gradient ratio r = (R-RR)/(L-R)
            double delta = cell_L - cell_R;
            if (std::abs(delta) < 1e-30) return cell_R;
            double r = (cell_R - cell_RR) / delta;
            double phi = std::max(0.0, std::min(r, 1.0));
            return cell_R + 0.5 * phi * delta;
        }
    }
};

/**
 * MUSCL with van Leer limiter — second-order TVD, smooth.
 * phi(r) = (r + |r|) / (1 + |r|)
 */
struct MUSCL_VanLeer : FaceReconstruction {
    double face_value(
        double cell_LL, double cell_L, double cell_R, double cell_RR,
        double mdot_face) const override
    {
        if (mdot_face >= 0.0) {
            double delta = cell_R - cell_L;
            if (std::abs(delta) < 1e-30) return cell_L;
            double r = (cell_L - cell_LL) / delta;
            double phi = (r + std::abs(r)) / (1.0 + std::abs(r));
            return cell_L + 0.5 * phi * delta;
        } else {
            double delta = cell_L - cell_R;
            if (std::abs(delta) < 1e-30) return cell_R;
            double r = (cell_R - cell_RR) / delta;
            double phi = (r + std::abs(r)) / (1.0 + std::abs(r));
            return cell_R + 0.5 * phi * delta;
        }
    }
};

} // namespace opal
