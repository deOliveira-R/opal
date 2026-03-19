#pragma once
/**
 * fluid_package.hpp — Unified fluid property interface.
 *
 * FluidPackage combines FluidProperties (mixture EOS) and PhasicProperties
 * (saturation/phasic EOS) into a single interface. This enforces at the
 * type level that the same object provides both mixture and phasic properties,
 * preventing inconsistencies from using two different property objects.
 *
 * All concrete fluid implementations (SimpleFluid, IAPWSIF97) inherit from
 * FluidPackage. The solver and FlowModel both take const FluidPackage&.
 */

#include "properties.hpp"
#include "phasic_properties.hpp"

namespace opal {

class FluidPackage : public FluidProperties, public PhasicProperties {
public:
    virtual ~FluidPackage() = default;

    /// Minimum pressure for safe property evaluation [Pa].
    /// The solver clamps pressure to [p_min, p_max] after the tridiagonal solve.
    virtual double p_min() const { return 100.0; }

    /// Maximum pressure for safe property evaluation [Pa].
    virtual double p_max() const { return 100.0e6; }
};

} // namespace opal
