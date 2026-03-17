"""
Numerical verification utilities for OPAL equation derivations.

Every derivation script should end with numerical verification:
generate random valid states, evaluate the derived equations, and
confirm conservation / consistency to machine precision.
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Optional, Tuple


def random_thermo_state(n_samples: int = 1, seed: int = 42) -> List[Dict]:
    """
    Generate random but thermodynamically plausible states for testing.

    Returns list of dicts mapping symbol names (strings) to float values.
    States are in PWR-relevant ranges:
        P:     1 - 17 MPa
        T_l:   300 - 620 K
        T_v:   373 - 650 K
        alpha: 0.0 - 1.0
        rho_l: 600 - 1000 kg/m³
        rho_v: 1 - 150 kg/m³
        h_l:   100e3 - 1600e3 J/kg
        h_v:   2500e3 - 2900e3 J/kg
        v_l:   -5 to 15 m/s
        v_v:   -5 to 30 m/s

    These are NOT self-consistent states (h and T are independent).
    For property verification, use random_consistent_state() which
    evaluates properties from (P, h) pairs.
    """
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n_samples):
        states.append({
            'P': rng.uniform(1e6, 17e6),
            'P_old': rng.uniform(1e6, 17e6),
            'T_l': rng.uniform(300, 620),
            'T_v': rng.uniform(373, 650),
            'T_sat': rng.uniform(373, 620),
            'alpha': rng.uniform(0.0, 1.0),
            'alpha_old': rng.uniform(0.0, 1.0),
            'rho_l': rng.uniform(600, 1000),
            'rho_v': rng.uniform(1, 150),
            'h_l': rng.uniform(100e3, 1600e3),
            'h_v': rng.uniform(2500e3, 2900e3),
            'v_l': rng.uniform(-5, 15),
            'v_v': rng.uniform(-5, 30),
            'dt': rng.uniform(1e-4, 0.1),
            'dx': rng.uniform(0.01, 1.0),
            'dA': rng.uniform(0.001, 0.1),
            'A_flow': rng.uniform(0.001, 0.1),
            'V_cell': rng.uniform(1e-4, 0.1),
            'D_h': rng.uniform(0.005, 0.05),
            'g': 9.81,
            'Gamma': rng.uniform(-10, 10),
            'q_wall': rng.uniform(0, 1e6),
            'q_vol': rng.uniform(0, 1e8),
            'F_wall': rng.uniform(0, 1e5),
            'F_i': rng.uniform(-1e4, 1e4),
            # Thermodynamic derivatives (plausible magnitudes)
            'drho_l_dP_h': rng.uniform(1e-7, 1e-5),
            'drho_v_dP_h': rng.uniform(1e-6, 1e-4),
            'drho_l_dh_P': rng.uniform(-1e-3, -1e-5),
            'drho_v_dh_P': rng.uniform(-1e-4, -1e-6),
        })
    return states


def check_conservation(
    residual_expr: sp.Expr,
    symbols_to_values: Optional[Dict] = None,
    n_samples: int = 100,
    tol: float = 1e-10,
    description: str = "conservation check",
    seed: int = 42,
) -> bool:
    """
    Numerically verify that a symbolic residual is zero.

    Args:
        residual_expr: SymPy expression that should evaluate to zero.
        symbols_to_values: optional fixed substitution dict. If None,
            uses random_thermo_state().
        n_samples: number of random samples to test.
        tol: absolute tolerance for zero check.
        description: label for reporting.
        seed: RNG seed.

    Returns:
        True if all samples pass, False otherwise.
        Also prints a PASS/FAIL summary.
    """
    # Get free symbols in the expression
    free_syms = residual_expr.free_symbols
    free_sym_names = {s.name for s in free_syms}

    # Build a lambdified evaluator
    sym_list = sorted(free_syms, key=lambda s: s.name)
    sym_names = [s.name for s in sym_list]

    try:
        f = sp.lambdify(sym_list, residual_expr, modules="numpy")
    except Exception as e:
        print(f"FAIL [{description}]: Could not lambdify expression: {e}")
        return False

    states = random_thermo_state(n_samples, seed)
    max_residual = 0.0
    n_fail = 0

    for state in states:
        try:
            values = []
            for name in sym_names:
                if name in state:
                    values.append(state[name])
                elif symbols_to_values and name in symbols_to_values:
                    values.append(symbols_to_values[name])
                else:
                    values.append(np.random.default_rng().uniform(0.1, 10.0))

            result = float(f(*values))
            max_residual = max(max_residual, abs(result))
            if abs(result) > tol:
                n_fail += 1
        except Exception as e:
            n_fail += 1

    if n_fail == 0:
        print(f"PASS [{description}]: max |residual| = {max_residual:.2e} "
              f"({n_samples} samples, tol={tol:.0e})")
        return True
    else:
        print(f"FAIL [{description}]: {n_fail}/{n_samples} samples exceeded "
              f"tol={tol:.0e}, max |residual| = {max_residual:.2e}")
        return False


def check_equation_pair(
    lhs: sp.Expr,
    rhs: sp.Expr,
    n_samples: int = 100,
    tol: float = 1e-10,
    description: str = "equation check",
) -> bool:
    """
    Verify that two expressions are numerically equal.
    Convenience wrapper around check_conservation(lhs - rhs).
    """
    return check_conservation(
        sp.expand(lhs - rhs),
        n_samples=n_samples,
        tol=tol,
        description=description,
    )


def check_discrete_conservation(
    cell_values_func,
    flux_func,
    source_func,
    n_cells: int = 10,
    n_samples: int = 50,
    tol: float = 1e-10,
    description: str = "discrete conservation",
) -> bool:
    """
    Verify discrete conservation: sum of accumulation = boundary fluxes + sources.

    For a closed system (no boundary flux), the sum of cell accumulation terms
    should equal the sum of source terms. For an open system, the net flux
    through boundaries must be accounted for.

    Args:
        cell_values_func: callable(state_array) → accumulation per cell
        flux_func: callable(state_array) → flux at each face
        source_func: callable(state_array) → source per cell
        n_cells: number of cells in test mesh
        n_samples: number of random tests
        tol: tolerance
        description: label

    Returns:
        True if conservation holds to tolerance.
    """
    rng = np.random.default_rng(42)
    n_fail = 0
    max_residual = 0.0

    for _ in range(n_samples):
        state = rng.uniform(0.1, 10.0, size=n_cells)

        accum = cell_values_func(state)
        fluxes = flux_func(state)
        sources = source_func(state)

        # Conservation: sum(accum) = flux_in - flux_out + sum(sources)
        total_accum = np.sum(accum)
        net_flux = fluxes[0] - fluxes[-1]  # flux_in(left) - flux_out(right)
        total_source = np.sum(sources)

        residual = total_accum - net_flux - total_source
        max_residual = max(max_residual, abs(residual))
        if abs(residual) > tol:
            n_fail += 1

    if n_fail == 0:
        print(f"PASS [{description}]: max |residual| = {max_residual:.2e}")
        return True
    else:
        print(f"FAIL [{description}]: {n_fail}/{n_samples} exceeded tol")
        return False
