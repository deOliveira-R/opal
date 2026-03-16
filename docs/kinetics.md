# Reactor Kinetics Module

Computes volumetric power distribution as function of reactor state. Two fidelity levels with common interface (`PartialKineticsModel`).

## Level 1: Point Kinetics (Phase 4)

~32 ODEs, trivially cheap. Same as RELAP5/TRACE/CATHARE built-in models.

Solves:
- 6-group delayed neutron precursor equations (6 ODEs)
- I-135 / Xe-135 / Sm-149 dynamics (3-4 ODEs)
- Decay heat per ANS 5.1 (23 exponential groups, 23 ODEs)

Reactivity:
```
ρ(t) = ρ_rod(z_rod) + Σ α_fuel,i·(T_fuel,i - T_ref) + Σ α_mod,i·(T_mod,i - T_ref)
      + α_void·Δα_void + α_boron·ΔC_boron + ρ_Xe(t) + ρ_Sm(t)
```

Power shape is prescribed (user-supplied axial/radial profiles, parametrized by rod position and burnup). This is the fundamental limitation.

Goes in standard DAE partition, not semi-implicit TH partition. Coupled via heat source (kinetics→TH) and reactivity feedback (TH→kinetics) each timestep.

**Adequate for:** LOCA, loss of flow, turbine trip, station blackout.
**Not adequate for:** asymmetric rod ejection, steam line break with asymmetric cooling, boron dilution. Need Level 2.

## Level 2: 3D Spatial Kinetics via Few-Group Neutron Diffusion (Phase 5)

Time-dependent few-group (2-4 groups) diffusion:
```
1/v_g · ∂φ_g/∂t = ∇·(D_g·∇φ_g) - Σ_r,g·φ_g + scatter + fission + precursors
```

Same physics as PARCS, DYN3D, SIMULATE-3K. These codes require institutional access most of the world cannot obtain.

### Quasi-Static Method

Factor flux: φ_g(r,t) = N(t) · S_g(r,t)

- **Amplitude N(t):** point kinetics ODEs with computed reactivity. Updated every TH step. DAE partition. Cheap.
- **Shape S_g(r,t):** full 3D diffusion solve. Updated at "macro" steps (0.1-1.0s or when reactivity changes). Sparse linear algebra, not DAE.

Three-phase loop:
1. Advance TH (semi-implicit)
2. Advance amplitude + precursors + controls (DAE)
3. Periodically re-solve 3D shape, update XS from TH state, update reactivity

Same coupling as TRACE/PARCS but internal to OPAL.

### Spatial Discretization
Finite difference on structured mesh. Start with Cartesian. Each node = one assembly × axial layer. Typical: 200 assemblies × 20 axial = 4000 nodes × 2 groups = 8000 unknowns + precursors.

### Eigenvalue Initialization
Reactor must start critical (k_eff=1.0). Power iteration + Wielandt shift or JFNK. Standalone step: converge TH steady state → solve eigenvalue → adjust boron/rods → start transient.

### Cross-Section Data
Homogenized few-group XS (D_g, Σ_a,g, Σ_f,g, νΣ_f,g, Σ_{s,g'→g}) parametrized by T_fuel, T_mod, ρ_mod, boron, burnup, rod insertion. From upstream lattice codes (Serpent, OpenMC, DRAGON5) run offline. OPAL defines HDF5 format with conversion tools targeting open lattice codes.

### Implementation
Diffusion solver in C++ via `external "C"`. Modelica handles quasi-static amplitude + coupling interface. Correct separation: Modelica describes system/coupling, C++ does linear algebra.

## Interface

```modelica
partial model PartialKineticsModel
  input Real T_fuel[:,:,:];
  input Real T_mod[:,:,:];
  input Real rho_mod[:,:,:];
  input Real boron_concentration;
  input Real rod_position[:];
  output Real q_vol[:,:,:];
  output Real total_power;
  output Real decay_heat_fraction;
end PartialKineticsModel;
```

Swapping Level 1 for Level 2 is a one-line `redeclare`. Rest of plant untouched.

## Phasing
Point kinetics first (Phase 4). Spatial diffusion when 3D vessel and semi-implicit solver are mature (Phase 5). Skip multi-node coupled point kinetics — go straight from Level 1 to Level 2.
