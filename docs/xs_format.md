# Cross-Section Data Format Specification

## Status: To Be Defined (Phase 5)

This document will specify OPAL's HDF5-based format for homogenized few-group neutron cross sections.

## Planned Content

### Data Requirements
- Few-group constants: D_g, Σ_a,g, Σ_f,g, νΣ_f,g, Σ_{s,g'→g}
- Parametrized by: T_fuel, T_mod, ρ_mod, boron concentration, burnup, rod insertion
- Per assembly type, per axial layer

### Format
- HDF5-based, self-describing
- Tabulated with interpolation metadata (linear, polynomial, or spline)
- Assembly geometry and mapping information

### Conversion Tools
- Serpent → OPAL
- OpenMC → OPAL
- DRAGON5 → OPAL
- Targeting open lattice codes that OPAL's users can actually access

### Validation
- Round-trip: generate XS → load in OPAL → verify eigenvalue matches lattice code
- Interpolation accuracy vs. direct lattice calculation at intermediate points
