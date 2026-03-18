#pragma once
/**
 * boundary_conditions.hpp — Boundary condition infrastructure.
 *
 * Two layers:
 *   1. BoundaryFace strategy objects (new): encapsulate physics, produce
 *      mathematical contributions. Time-aware. Extensible via subclassing.
 *   2. BoundaryConditions struct (legacy): static bag of doubles, preserved
 *      for backward compatibility. The legacy step() constructs temporary
 *      BoundaryFace objects from this struct internally.
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


// =========================================================================
// Legacy structs (preserved for backward compatibility)
// =========================================================================

/// Boundary condition type for each face (legacy enum).
enum class BCType {
    PRESSURE,  ///< Specified pressure (default, existing behavior)
    WALL,      ///< Closed wall: mdot = 0, no pressure coupling
    BREAK,     ///< Break/rupture: pressure BC with critical flow limiter
};

/// Legacy boundary conditions struct. All existing tests use this.
/// The solver internally constructs BoundaryFace objects from these fields.
struct BoundaryConditions {
    // Pressure BCs (always needed)
    double p_in  = 0.0;   ///< Inlet pressure [Pa]
    double p_out = 0.0;   ///< Outlet pressure [Pa]

    // Enthalpy BCs
    double h_in  = 0.0;   ///< Mixture inlet enthalpy [J/kg] (HEM/4-eq)
    double h_l_in = 0.0;  ///< Liquid inlet enthalpy [J/kg] (5-eq/6-eq)
    double h_v_in = 0.0;  ///< Vapor inlet enthalpy [J/kg] (5-eq/6-eq)

    // Void fraction BC
    double alpha_in = 0.0; ///< Inlet void fraction [-] (4-eq+)

    // Velocity BCs (6-eq only)
    double v_l_in = 0.0;  ///< Inlet liquid velocity [m/s]
    double v_v_in = 0.0;  ///< Inlet vapor velocity [m/s]

    // Face BC types (default PRESSURE for backward compatibility)
    BCType bc_type_in  = BCType::PRESSURE;
    BCType bc_type_out = BCType::PRESSURE;

    // Break parameters (only used when bc_type == BREAK)
    double break_area_fraction = 1.0;  ///< Discharge coefficient C_d [-]
};

} // namespace opal
