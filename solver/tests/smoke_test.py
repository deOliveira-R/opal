"""Quick smoke test — run directly, no pytest dependency."""
import sys, time
from pathlib import Path

OPAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(OPAL_ROOT / "solver" / "single_phase"))

import numpy as np
import opal_single_phase as sp

N = 5
R, C, rho, Cp, V = 1e4, 1e-9, 720.0, 5000.0, 0.01
p_in, p_out, T_in = 15.5e6, 15.4e6, 563.0

solver = sp.SinglePhaseSolver(N, R, C, rho, Cp, V)
bc     = sp.BoundaryConditions(p_in, p_out, T_in)
p0     = np.full(N, p_in)
T0     = np.full(N, T_in)
m0     = np.zeros(N + 1)

# --- 1. Single step ---
p_s, T_s, m_s = p0.copy(), T0.copy(), m0.copy()
solver.step(p_s, T_s, m_s, bc, 1e-3)
assert m_s[0] > 0, "inlet flow should be positive after 1 step"
print(f"[1] Single step:  mdot[0]={m_s[0]:.4f}  PASS")

# --- 2. H-P steady state ---
t0   = time.perf_counter()
hist = solver.solve(p0.copy(), T0.copy(), m0.copy(), bc, 1e-3, 10_000, stride=10_000)
dt_ms = (time.perf_counter()-t0)*1000
mdot_ss_analytical = (p_in - p_out) / ((N + 1) * R)
mdot_f   = hist[-1, 2*N:]
rel_err  = abs(mdot_f.mean() - mdot_ss_analytical) / mdot_ss_analytical
assert rel_err < 1e-4, f"H-P rel err = {rel_err:.2e}"
print(f"[2] H-P steady state: mdot={mdot_f.mean():.6g} (expected {mdot_ss_analytical:.6g})  rel_err={rel_err:.2e}  {dt_ms:.1f}ms  PASS")

# --- 3. Mass conservation over 50 steps ---
hist2 = solver.solve(p0.copy(), T0.copy(), m0.copy(), bc, 1e-3, 50, stride=1)
p_h   = hist2[:, :N]
m_h   = hist2[:, 2*N:]
# Conservation: C*(p[s+1]-p[s])/dt = mdot[s+1][0] - mdot[s+1][N]
dp    = np.diff(p_h, axis=0)
stored= (C * dp / 1e-3).sum(axis=1)
inflow= m_h[1:, 0] - m_h[1:, -1]    # mdot at the later snapshot
res   = np.abs(stored - inflow) / (np.abs(inflow).mean() + 1e-30)
assert res.max() < 1e-6, f"mass conservation residual = {res.max():.2e}"
print(f"[3] Mass conservation: max residual = {res.max():.2e}  PASS")

# --- 4. Partitioner round-trip ---
sys.path.insert(0, str(OPAL_ROOT))
from solver.partitioner.xml_reader  import load_equation_system
from solver.partitioner.grid_mapper import map_pipe_grid

xml = OPAL_ROOT / "feasibility/results/scale_N5_backEnd.xml"
es   = load_equation_system(xml)
grid = map_pipe_grid(es)
assert grid.N == 5
assert abs(grid.R - 1e4) < 1
assert abs(grid.p_in - 15.5e6) < 1
print(f"[4] Partitioner round-trip: N={grid.N}, R={grid.R:.3g}, p_in={grid.p_in:.6g}  PASS")

print("\nAll smoke tests PASSED")
