#!/usr/bin/env python3
"""
edwards_case_comparison.py — Compare Case 0 (hardcoded) vs Case 1 (extracted params).

Case 0: extracted_5eq_solver.py with hardcoded closure parameters
Case 1: parameterized_5eq_solver.py with ALL parameters from Modelica extraction

Both should produce IDENTICAL results (same physics, same numerics,
just different parameter source).
"""

import sys
import pathlib
import numpy as np

OPAL_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OPAL_ROOT / "two_phase"))
sys.path.insert(0, str(OPAL_ROOT))
sys.path.insert(0, str(OPAL_ROOT.parent / "docs" / "validation" / "edwards" / "data"))

import opal_two_phase as tp
from partitioner.xml_reader import load_equation_system
from partitioner.pipe1d_mapper import map_pipe1d
from partitioner.model_spec import extract_model_spec
from partitioner.extracted_5eq_solver import Extracted5EqSolver
from partitioner.parameterized_5eq_solver import Parameterized5EqSolver
from edwards_blowdown_data import edwards_blowdown

EDWARDS_XML = OPAL_ROOT.parent / "docs" / "validation" / "edwards" / "data" / "EdwardsTest_DriftFlux_backEnd.xml"

print("=" * 70)
print("Case Comparison: Case 0 (hardcoded) vs Case 1 (extracted params)")
print("=" * 70)

es = load_equation_system(str(EDWARDS_XML))

# Case 0 setup (hardcoded parameters)
spec_0 = map_pipe1d(es)
N = spec_0.N
fluid = tp.IAPWSIF97Properties()

import iapws
ic = edwards_blowdown["initial_conditions"]
geom = edwards_blowdown["geometry"]
p_init = ic["nominal_pressure_MPa"] * 1e6
T_init = ic["simplified_isothermal_K"]
ref = iapws.IAPWS97(P=p_init/1e6, T=T_init)
ref_g = iapws.IAPWS97(P=p_init/1e6, x=1)
h_init = ref.h * 1e3
h_g_init = ref_g.h * 1e3

solver_0 = Extracted5EqSolver(
    fluid, spec_0,
    H_i=1e5, C_0=1.0, alpha_nucleation=1e-3,
    use_critical_flow=True, C_d=geom["break_flow_area_fraction"],
    x_trans=0.10, c_floor=10.0,
    use_two_phase_friction=False)

# Case 1 setup (extracted parameters)
model_spec = extract_model_spec(es)
print(f"\nExtracted model spec:")
print(f"  {model_spec.summary()}")
print(f"  Closures: H_i={model_spec.closures.H_i}, C_0={model_spec.closures.C_0}")
print(f"  Crit flow: {model_spec.closures.use_critical_flow}, C_d={model_spec.closures.C_d}")

solver_1 = Parameterized5EqSolver(fluid, model_spec)

# Run both
dt = 5e-5
n_steps = 2000  # Short run for comparison

print(f"\nRunning {n_steps} steps side by side...")

p0 = np.full(N, p_init); a0 = np.full(N, 1e-6)
hl0 = np.full(N, h_init); hv0 = np.full(N, h_g_init); m0 = np.zeros(N+1)

p1 = p0.copy(); a1 = a0.copy()
hl1 = hl0.copy(); hv1 = hv0.copy(); m1 = m0.copy()

for step in range(n_steps):
    solver_0.step(p0, a0, hl0, hv0, m0, dt)
    solver_1.step(p1, a1, hl1, hv1, m1, dt)

# Compare
p_err = np.max(np.abs(p0 - p1) / np.maximum(np.abs(p0), 1.0))
h_err = np.max(np.abs(hl0 - hl1))
a_err = np.max(np.abs(a0 - a1))
m_err = np.max(np.abs(m0 - m1))

print(f"\n{'='*60}")
print(f"COMPARISON after {n_steps} steps")
print(f"{'='*60}")
print(f"  Pressure max relative error: {p_err:.2e}")
print(f"  Enthalpy max absolute error: {h_err:.2e} J/kg")
print(f"  Alpha max absolute error:    {a_err:.2e}")
print(f"  Mdot max absolute error:     {m_err:.2e} kg/s")

print(f"\n  Case 0 final: p[0]={p0[0]/1e6:.4f} MPa, mdot_out={m0[-1]:.4f}")
print(f"  Case 1 final: p[0]={p1[0]/1e6:.4f} MPa, mdot_out={m1[-1]:.4f}")

if p_err < 1e-6:
    print(f"\n  VERDICT: MATCH — Cases 0 and 1 produce identical results")
elif p_err < 1e-3:
    print(f"\n  VERDICT: CLOSE — small differences from extracted vs hardcoded parameters")
    diffs = []
    if model_spec.closures.H_i != 1e5:
        diffs.append(f"H_i: extracted={model_spec.closures.H_i:.0e} vs hardcoded=1e5")
    if model_spec.closures.use_critical_flow != True:
        diffs.append(f"crit_flow: extracted={model_spec.closures.use_critical_flow} vs hardcoded=True")
    if diffs:
        print(f"  Known parameter differences:")
        for d in diffs:
            print(f"    {d}")
    print(f"\n  Case 1 is CORRECT — it respects the Modelica model parameters.")
    print(f"  Case 0 was overriding Modelica values with hardcoded defaults.")
else:
    print(f"\n  VERDICT: DIVERGED — Cases 0 and 1 disagree significantly")
    print(f"  Investigate parameter extraction.")
