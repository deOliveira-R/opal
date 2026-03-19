#pragma once
/**
 * boundary_conditions.hpp — Boundary condition infrastructure.
 *
 * Each boundary face is a strategy object that encapsulates physics and
 * produces mathematical contributions. Time-aware. Extensible via subclassing.
 *
 * BoundaryFace subclasses:
 *   PressureFace  — specified pressure with inflow enthalpy
 *   WallFace      — closed wall (zero flux, no coupling)
 *   BreakFace     — pressure BC with critical flow limiter
 *   RampedBreak   — time-varying break opening (e.g., glass disk rupture)
 */

#include <cmath>
#include <algorithm>

namespace opal {

// Forward declarations
struct FluidProps;
struct MeshParams;
class FluidPackage;
class CriticalFlowModel;

// =========================================================================
// BoundaryFace strategy pattern
// =========================================================================

/// What a boundary face produces for the pressure equation.
struct FacePressureBC {
    enum Type { DIRICHLET, ZERO_FLUX };
    Type type = DIRICHLET;
    double p_boundary = 0.0;   ///< Pressure [Pa] (only when DIRICHLET)
};

/// What a boundary face produces for the transport equations.
struct FaceTransportBC {
    double h_l    = 0.0;   ///< Liquid enthalpy [J/kg] for ghost cells
    double h_v    = 0.0;   ///< Vapor enthalpy [J/kg]
    double h_mix  = 0.0;   ///< Mixture enthalpy [J/kg] (HEM)
    double alpha  = 0.0;   ///< Void fraction [-]
};

/**
 * Abstract boundary face — one per domain boundary.
 *
 * The solver calls these methods instead of reading a static struct.
 * Every method takes `double t` for time-dependent BCs; static BCs ignore it.
 */
class BoundaryFace {
public:
    virtual ~BoundaryFace() = default;

    /// Pressure coupling: Dirichlet (known p) or zero-flux (wall).
    virtual FacePressureBC pressure(double t) const = 0;

    /// Face density [kg/m³] for friction and resistance computation.
    /// @param adjacent_rho  density of the adjacent interior cell
    virtual double face_density(double adjacent_rho, double t) const = 0;

    /// Transport boundary values for ghost cells.
    virtual FaceTransportBC transport(double t) const = 0;

    /// Does this face need a critical flow check?
    virtual bool has_critical_flow() const { return false; }

    /// Critical flow discharge coefficient (time-dependent for ramped breaks).
    virtual double discharge_coefficient(double t) const { return 1.0; }

    /// Is this face a wall? (mdot forced to zero)
    virtual bool is_wall() const { return false; }
};

// =========================================================================
// Concrete BoundaryFace implementations
// =========================================================================

/**
 * Pressure BC: specified pressure with inflow enthalpy/void fraction.
 */
class PressureFace : public BoundaryFace {
public:
    PressureFace(double p, double h_l, double h_v = 0.0, double alpha = 0.0)
        : p_(p), h_l_(h_l), h_v_(h_v), alpha_(alpha) {}

    FacePressureBC pressure(double) const override {
        return {FacePressureBC::DIRICHLET, p_};
    }

    double face_density(double adjacent_rho, double) const override {
        // Average of BC-evaluated density and adjacent cell density.
        // For a true density, we'd need the FluidPackage — but the simple
        // average is used by the current solver and is adequate.
        // The caller (solver.cpp) can evaluate fluid.evaluate(p, h) if needed.
        return adjacent_rho;  // Will be refined by solver
    }

    FaceTransportBC transport(double) const override {
        return {h_l_, h_v_, h_l_, alpha_};  // h_mix = h_l for subcooled
    }

    double p()     const { return p_; }
    double h_l()   const { return h_l_; }
    double h_v()   const { return h_v_; }
    double alpha() const { return alpha_; }

private:
    double p_, h_l_, h_v_, alpha_;
};

/**
 * Wall: zero flux, no pressure coupling.
 */
class WallFace : public BoundaryFace {
public:
    WallFace(double h_l = 0.0, double h_v = 0.0)
        : h_l_(h_l), h_v_(h_v) {}

    FacePressureBC pressure(double) const override {
        return {FacePressureBC::ZERO_FLUX, 0.0};
    }

    double face_density(double adjacent_rho, double) const override {
        return adjacent_rho;  // wall: use adjacent cell density
    }

    FaceTransportBC transport(double) const override {
        return {h_l_, h_v_, h_l_, 0.0};
    }

    bool is_wall() const override { return true; }

private:
    double h_l_, h_v_;
};

/**
 * Break with critical flow limiter.
 * Pressure BC at p_back with discharge coefficient C_d.
 */
class BreakFace : public BoundaryFace {
public:
    BreakFace(double p_back, double C_d,
              double h_l = 0.0, double h_v = 0.0)
        : p_back_(p_back), C_d_(C_d), h_l_(h_l), h_v_(h_v) {}

    FacePressureBC pressure(double) const override {
        return {FacePressureBC::DIRICHLET, p_back_};
    }

    double face_density(double adjacent_rho, double) const override {
        return adjacent_rho;
    }

    FaceTransportBC transport(double) const override {
        return {h_l_, h_v_, h_l_, 0.0};
    }

    bool has_critical_flow() const override { return true; }

    double discharge_coefficient(double) const override { return C_d_; }

protected:
    double p_back_, C_d_, h_l_, h_v_;
};

/**
 * Time-ramped break: C_d goes from 0 to C_d_final over t_open seconds.
 * Models finite break opening time (e.g., glass disk rupture).
 */
class RampedBreak : public BreakFace {
public:
    RampedBreak(double p_back, double C_d_final, double t_open,
                double h_l = 0.0, double h_v = 0.0)
        : BreakFace(p_back, C_d_final, h_l, h_v), t_open_(t_open) {}

    double discharge_coefficient(double t) const override {
        if (t_open_ <= 0.0) return C_d_;
        return C_d_ * std::min(t / t_open_, 1.0);
    }

private:
    double t_open_;
};


// Legacy BoundaryConditions struct, BCType enum, and TwoPhaseBCs removed.
// All boundary conditions now use the BoundaryFace strategy hierarchy above.

} // namespace opal
