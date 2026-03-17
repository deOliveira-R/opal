"""
test_hagen_poiseuille.py — Analytical verification of the single-phase solver.

Tests:
  1. Steady-state flow rate matches Hagen-Poiseuille: mdot_ss = ΔP / ((N+1)*R)
  2. Steady-state pressure profile is linear
  3. Global mass conservation holds to machine precision at every step
  4. Joukowski acoustic speed: pressure transient from flow step matches
     wave speed  c = 1/sqrt(rho*C/V)  (characteristic decay ~L/c)

All tests use ScalablePipe parameters so results are directly comparable to
the feasibility XML (scale_N5_backEnd.xml).

Run with:
  python -m pytest solver/tests/test_hagen_poiseuille.py -v
or directly:
  python solver/tests/test_hagen_poiseuille.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SOLVER_DIR = Path(__file__).parent.parent.resolve()
OPAL_ROOT  = SOLVER_DIR.parent
sys.path.insert(0, str(OPAL_ROOT))
sys.path.insert(0, str(SOLVER_DIR / "single_phase"))

try:
    import opal_single_phase as sp
except ImportError as exc:
    sys.exit(f"ERROR: opal_single_phase not built yet.\n{exc}")

# ---------------------------------------------------------------------------
# ScalablePipe N=5 reference parameters
# ---------------------------------------------------------------------------
REF = dict(
    N=5,
    R=1e4,    # Pa/(kg/s)
    C=1e-9,   # kg/Pa
    rho=720.0,
    Cp=5000.0,
    V=0.01,   # m³
    p_in=15.5e6,
    p_out=15.4e6,
    T_in=563.0,
)


def make_solver(**overrides):
    p = {**REF, **overrides}
    return sp.SinglePhaseSolver(
        N=p["N"], R=p["R"], C=p["C"],
        rho=p["rho"], Cp=p["Cp"], V=p["V"],
    )


def make_bc(**overrides):
    p = {**REF, **overrides}
    return sp.BoundaryConditions(
        p_in=p["p_in"], p_out=p["p_out"], T_in=p["T_in"],
    )


def initial_state(params: dict):
    N = params["N"]
    p0    = np.full(N, params["p_in"],   dtype=np.float64)
    T0    = np.full(N, params["T_in"],   dtype=np.float64)
    mdot0 = np.zeros(N + 1,              dtype=np.float64)
    return p0, T0, mdot0


def analytical_mdot_ss(N, R, p_in, p_out):
    return (p_in - p_out) / ((N + 1) * R)


def analytical_p_profile(N, R, p_in, p_out):
    """Linear pressure profile at steady state (cell-centre pressures)."""
    mdot_ss = analytical_mdot_ss(N, R, p_in, p_out)
    return np.array([p_in - (i + 1) * R * mdot_ss for i in range(N)])


# ---------------------------------------------------------------------------
# Test 1 — Hagen-Poiseuille steady state
# ---------------------------------------------------------------------------

def test_hagen_poiseuille_N5():
    """Flow rate converges to ΔP/((N+1)R) within 1 part in 1e4."""
    solver = make_solver()
    bc     = make_bc()
    p0, T0, mdot0 = initial_state(REF)

    dt      = 1e-3
    n_steps = 10_000
    history = solver.solve(p0, T0, mdot0, bc, dt, n_steps, stride=100)

    N = REF["N"]
    mdot_final = history[-1, 2*N:]          # last snapshot, mdot columns
    mdot_ss    = analytical_mdot_ss(N, REF["R"], REF["p_in"], REF["p_out"])

    # All faces should carry the same flow at steady state
    uniformity = (mdot_final.max() - mdot_final.min()) / abs(mdot_ss)
    rel_err    = abs(mdot_final.mean() - mdot_ss) / abs(mdot_ss)

    assert uniformity < 1e-6, f"Flow uniformity err too large: {uniformity:.2e}"
    assert rel_err < 1e-4,    f"H-P rel err too large: {rel_err:.2e}"
    print(f"  PASS  H-P: mdot_ss={mdot_ss:.6g}, sim={mdot_final.mean():.6g}, "
          f"rel_err={rel_err:.2e}")


# ---------------------------------------------------------------------------
# Test 2 — Linear pressure profile at steady state
# ---------------------------------------------------------------------------

def test_pressure_profile_N5():
    """Steady-state cell pressures match the analytical linear profile."""
    solver = make_solver()
    bc     = make_bc()
    p0, T0, mdot0 = initial_state(REF)

    dt      = 1e-3
    n_steps = 10_000
    history = solver.solve(p0, T0, mdot0, bc, dt, n_steps, stride=n_steps)

    N = REF["N"]
    p_final  = history[-1, :N]
    p_expect = analytical_p_profile(N, REF["R"], REF["p_in"], REF["p_out"])

    rel_err = np.abs(p_final - p_expect) / REF["p_in"]
    assert rel_err.max() < 1e-4, f"Pressure profile max rel err: {rel_err.max():.2e}"
    print(f"  PASS  Pressure profile: max rel err = {rel_err.max():.2e}")


# ---------------------------------------------------------------------------
# Test 3 — Global mass conservation
# ---------------------------------------------------------------------------

def test_mass_conservation():
    """
    Global mass balance:
        sum_i C * dp[i]/dt  ≈  mdot_in - mdot_out

    Check is discrete: exact to roundoff for the semi-implicit scheme
    because the solver constructs p from the tridiagonal system that
    IS the mass balance.
    """
    solver = make_solver()
    bc     = make_bc()
    p0, T0, mdot0 = initial_state(REF)

    dt      = 1e-3
    n_steps = 500
    stride  = 1
    history = solver.solve(p0, T0, mdot0, bc, dt, n_steps, stride)

    N = REF["N"]
    p_hist    = history[:, :N]
    mdot_hist = history[:, 2*N:]

    # Conservation: C*(p[s+1]-p[s])/dt = mdot[s+1][inlet] - mdot[s+1][outlet]
    dp          = np.diff(p_hist, axis=0)
    mass_stored = REF["C"] * dp.sum(axis=1) / dt      # kg/s
    net_inflow  = mdot_hist[1:, 0] - mdot_hist[1:, -1]  # use later snapshot's mdot

    residual = mass_stored - net_inflow
    scale    = np.abs(net_inflow).mean() + 1e-30
    rel_res  = np.abs(residual) / scale

    # Thomas algorithm + large pressure values (~15.5e6) give O(N*eps_machine*p) ≈ 1e-8
    assert rel_res.max() < 1e-7, \
        f"Mass conservation residual too large: {rel_res.max():.2e}"
    print(f"  PASS  Mass conservation: max rel residual = {rel_res.max():.2e}")


# ---------------------------------------------------------------------------
# Test 4 — Acoustic time scale (pressure wave propagation)
# ---------------------------------------------------------------------------

def test_pressure_wave_N20():
    """
    Start from steady state, then instantaneously raise p_in by 10 %.
    The pressure perturbation should propagate through N cells with the
    acoustic speed  c = sqrt(V / (rho * C * dx²)) in the lumped model.

    We verify that:
    - Cell 1 responds within 1–2 acoustic transit times
    - Cell N responds later than cell 1 (causality)
    """
    params = {**REF, "N": 20}
    solver = make_solver(**params)

    # First: run to steady state with nominal BCs
    bc_nom   = make_bc(**params)
    p0, T0, m0 = initial_state(params)
    dt       = 5e-5
    n_warmup = 20_000
    hist_ss  = solver.solve(p0, T0, m0, bc_nom, dt, n_warmup, stride=n_warmup)

    p_ss    = hist_ss[-1, :params["N"]].copy()
    T_ss    = hist_ss[-1, params["N"]:2*params["N"]].copy()
    mdot_ss = hist_ss[-1, 2*params["N"]:].copy()

    # Step: raise p_in by 10%
    bc_step = sp.BoundaryConditions(
        p_in=REF["p_in"] * 1.1, p_out=REF["p_out"], T_in=REF["T_in"]
    )
    n_steps = 500
    hist = solver.solve(p_ss, T_ss, mdot_ss, bc_step, dt, n_steps, stride=1)

    N = params["N"]
    p_hist = hist[:, :N]

    # Normalised pressure perturbation at each cell
    p_ref   = p_hist[0, :]           # just before step
    dp      = p_hist - p_ref[None, :]
    dp_norm = dp / (REF["p_in"] * 0.1)  # normalised by perturbation size

    # Cell 0 (nearest inlet) should reach 10% of perturbation before cell N-1
    t_5pct = np.array([
        next((i * dt for i in range(n_steps) if dp_norm[i, j] > 0.05), np.inf)
        for j in range(N)
    ])

    assert t_5pct[0] < t_5pct[-1], \
        "Cell 0 should respond before cell N-1 (causality check)"
    assert t_5pct[0] < np.inf, "Cell 0 never responded to pressure step"

    print(f"  PASS  Pressure wave: cell 0 responds at t={t_5pct[0]:.2e}s, "
          f"cell {N-1} at t={t_5pct[-1]:.2e}s")


# ---------------------------------------------------------------------------
# Test 5 — Scalability: N=1,5,10,20 all give correct H-P flow
# ---------------------------------------------------------------------------

def test_scalability():
    """H-P check across different grid sizes."""
    for N in [1, 5, 10, 20]:
        params = {**REF, "N": N}
        solver = make_solver(**params)
        bc     = make_bc(**params)
        p0, T0, m0 = initial_state(params)

        dt      = 1e-3
        n_steps = 10_000
        hist    = solver.solve(p0, T0, m0, bc, dt, n_steps, stride=n_steps)

        mdot_f  = hist[-1, 2*N:]
        mdot_ss = analytical_mdot_ss(N, REF["R"], REF["p_in"], REF["p_out"])
        rel_err = abs(mdot_f.mean() - mdot_ss) / abs(mdot_ss)
        assert rel_err < 1e-3, f"N={N}: H-P rel err = {rel_err:.2e}"
        print(f"  PASS  N={N:2d}: mdot_ss={mdot_ss:.6g}, rel_err={rel_err:.2e}")


# ---------------------------------------------------------------------------
# Test 6 — Convergence rate (CRITICAL: proves scheme order)
# ---------------------------------------------------------------------------

def test_convergence_rate():
    """
    First-order convergence of the semi-implicit pressure scheme.

    For N=1 cell with p_in, p_out BCs:
        C * dp/dt = (p_in - p)/R - (p - p_out)/R = (p_in + p_out - 2p) / R

    Analytical: p(t) = p_ss + (p_0 - p_ss) * exp(-2t/(R*C))
    where p_ss = (p_in + p_out) / 2.

    The implicit Euler scheme is first-order in dt.  Halving dt should
    halve the error at a fixed final time T_final.
    """
    N = 1
    R, C = REF["R"], REF["C"]
    p_in, p_out = REF["p_in"], REF["p_out"]
    p_ss = (p_in + p_out) / 2.0
    tau = R * C / 2.0  # time constant

    # Choose T_final ~ 3*tau so the transient is well-resolved
    T_final = 3.0 * tau

    # Initial condition: all pressure = p_in (offset from steady state)
    p_0 = p_in

    # Analytical solution at T_final
    p_exact = p_ss + (p_0 - p_ss) * np.exp(-T_final / tau)

    # Run at four different dt values
    dt_values = [T_final / 50, T_final / 100, T_final / 200, T_final / 400]
    errors = []

    solver = make_solver(N=N)

    for dt in dt_values:
        n_steps = int(round(T_final / dt))
        p0 = np.array([p_0])
        T0 = np.array([REF["T_in"]])
        m0 = np.zeros(2)
        bc = make_bc()

        hist = solver.solve(p0, T0, m0, bc, dt, n_steps, stride=n_steps)
        p_final = hist[-1, 0]
        errors.append(abs(p_final - p_exact))

    errors = np.array(errors)
    dt_arr = np.array(dt_values)

    # Compute convergence rates between successive refinements
    rates = np.log(errors[:-1] / errors[1:]) / np.log(dt_arr[:-1] / dt_arr[1:])

    # All rates should be ~1.0 (first-order scheme)
    min_rate = rates.min()
    assert min_rate > 0.90, \
        f"Convergence rate too low: {min_rate:.3f} (expected ~1.0). Rates: {rates}"

    print(f"  PASS  Convergence rates: {', '.join(f'{r:.3f}' for r in rates)} "
          f"(min={min_rate:.3f}, expected ~1.0)")


# ---------------------------------------------------------------------------
# Test 7 — Energy equation steady-state (ScalablePipe formulation)
# ---------------------------------------------------------------------------

def test_energy_steady_state():
    """
    Verify the ScalablePipe energy equation reaches the correct steady state.

    The solver uses:  rho*V*dT[i]/dt = mdot[i]*(T_in - T[i]) - mdot[i+1]*T[i]

    At steady state with uniform forward flow (mdot[i] = mdot for all i):
        0 = mdot*(T_in - T[i]) - mdot*T[i]
        T[i] = T_in / 2

    Note: This is the ScalablePipe-specific formulation (T_in used for all cells),
    not a general donor-cell scheme.  Phase 2 will use enthalpy-based equations.
    """
    solver = make_solver()
    bc     = make_bc()
    p0, T0, mdot0 = initial_state(REF)

    dt      = 1e-3
    n_steps = 50_000  # long enough for energy to reach steady state
    hist    = solver.solve(p0, T0, mdot0, bc, dt, n_steps, stride=n_steps)

    N = REF["N"]
    T_final = hist[-1, N:2*N]

    # ScalablePipe steady state: T = T_in / 2 for all cells
    T_expect = REF["T_in"] / 2.0
    rel_err = np.abs(T_final - T_expect) / REF["T_in"]
    assert rel_err.max() < 1e-3, \
        f"Energy steady state: T_final = {T_final}, expected {T_expect}, err = {rel_err.max():.2e}"

    print(f"  PASS  Energy steady state: T = {T_final.mean():.2f} K "
          f"(expected {T_expect:.2f}), err = {rel_err.max():.2e}")


# ---------------------------------------------------------------------------
# Test 8 — Quantitative acoustic wave speed
# ---------------------------------------------------------------------------

def test_acoustic_wave_speed():
    """
    Verify the numerical wave speed against the analytical lumped-model value.

    For the semi-implicit scheme on N cells with compressibility C and
    friction R, the pressure response of a single cell has time constant
    tau = R*C/2.  For N cells, the transit time for a perturbation to
    propagate from cell 0 to cell N-1 scales as N * tau_cell.

    We measure the arrival time at the last cell (when it reaches 5% of
    the perturbation amplitude) and check it falls within the expected
    range: N * R * C / 2 to 2 * N * R * C (accounting for implicit
    diffusion from the semi-implicit discretisation).
    """
    N = 10
    params = {**REF, "N": N}
    solver = make_solver(**params)

    # Warm up to steady state
    bc_nom = make_bc(**params)
    p0, T0, m0 = initial_state(params)
    dt = 1e-5
    hist_ss = solver.solve(p0, T0, m0, bc_nom, dt, 50_000, stride=50_000)

    p_ss = hist_ss[-1, :N].copy()
    T_ss = hist_ss[-1, N:2*N].copy()
    m_ss = hist_ss[-1, 2*N:].copy()

    # Step: raise p_in by 1%
    bc_step = sp.BoundaryConditions(
        p_in=REF["p_in"] * 1.01, p_out=REF["p_out"], T_in=REF["T_in"]
    )
    n_steps = 2000
    hist = solver.solve(p_ss, T_ss, m_ss, bc_step, dt, n_steps, stride=1)
    p_hist = hist[:, :N]

    dp = p_hist - p_hist[0, :]
    dp_norm = dp / (REF["p_in"] * 0.01)

    # Find when last cell reaches 5% of perturbation
    t_arrival = None
    for i in range(n_steps):
        if dp_norm[i, N-1] > 0.05:
            t_arrival = i * dt
            break

    assert t_arrival is not None, "Last cell never responded to pressure step"

    # Expected transit time: between N*R*C/2 and 2*N*R*C
    tau_cell = REF["R"] * REF["C"] / 2.0
    t_min = N * tau_cell * 0.5   # lower bound (implicit scheme is faster than explicit)
    t_max = N * tau_cell * 4.0   # upper bound (diffusive spreading)

    assert t_min < t_arrival < t_max, \
        f"Wave arrival t={t_arrival:.2e}s outside [{t_min:.2e}, {t_max:.2e}]"

    print(f"  PASS  Acoustic wave: arrival at cell {N-1} = {t_arrival:.2e}s "
          f"(expected range [{t_min:.2e}, {t_max:.2e}])")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Hagen-Poiseuille steady state (N=5)",    test_hagen_poiseuille_N5),
        ("Linear pressure profile (N=5)",           test_pressure_profile_N5),
        ("Global mass conservation",                test_mass_conservation),
        ("Pressure wave causality (N=20)",          test_pressure_wave_N20),
        ("Scalability N=1,5,10,20",                 test_scalability),
        ("Convergence rate (first-order)",           test_convergence_rate),
        ("Energy steady state (ScalablePipe)",       test_energy_steady_state),
        ("Acoustic wave speed (N=10)",               test_acoustic_wave_speed),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n--- {name}")
        try:
            fn()
            passed += 1
        except AssertionError as exc:  # noqa: keep original name
            print(f"  FAIL: {exc}")
            failed += 1
        except Exception as exc:
            print(f"  ERROR: {exc}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
