"""Tests for opal_sympy library."""

import sympy as sp
import numpy as np
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


def test_symbols_exist():
    from opal_sympy.symbols import P, T, alpha, rho_l, rho_v, dt, dx
    assert all(isinstance(s, sp.Symbol) for s in [P, T, alpha, rho_l, rho_v, dt, dx])
    assert dt.is_positive
    assert rho_l.is_positive
    print("PASS [test_symbols_exist]")


def test_stencil_1d():
    from opal_sympy.stencil import center_1d, east_1d, west_1d
    from opal_sympy.symbols import i

    phi = sp.IndexedBase('phi')
    c = center_1d(phi)
    e = east_1d(phi)
    w = west_1d(phi)

    assert c == phi[i]
    assert e == phi[i + 1]
    assert w == phi[i - 1]
    print("PASS [test_stencil_1d]")


def test_stencil_3d():
    from opal_sympy.stencil import center_3d, east_3d, top_3d, bottom_3d
    from opal_sympy.symbols import i, j, k

    P = sp.IndexedBase('P')
    assert center_3d(P) == P[i, j, k]
    assert east_3d(P) == P[i + 1, j, k]
    assert top_3d(P) == P[i, j, k + 1]
    assert bottom_3d(P) == P[i, j, k - 1]
    print("PASS [test_stencil_3d]")


def test_laplacian_symmetry():
    """Laplacian stencil should be symmetric: swapping directions doesn't change structure."""
    from opal_sympy.stencil import laplacian_7pt
    from opal_sympy.symbols import dx, dy, dz

    phi = sp.IndexedBase('phi')
    lap = laplacian_7pt(phi, dx, dy, dz)

    # On uniform mesh (dx=dy=dz=h), coefficient of center should be -6/h^2
    h = sp.Symbol('h', positive=True)
    lap_uniform = lap.subs([(dx, h), (dy, h), (dz, h)])
    lap_expanded = sp.expand(lap_uniform)

    # Extract coefficient of the center term phi[i,j,k]
    from opal_sympy.symbols import i, j, k
    center_coeff = lap_expanded.coeff(phi[i, j, k])
    expected = sp.Rational(-6, 1) / h**2
    assert sp.simplify(center_coeff - expected) == 0, f"Got {center_coeff}, expected {expected}"
    print("PASS [test_laplacian_symmetry]")


def test_mixture_mass_conservation():
    """Gamma must cancel in mixture mass equation."""
    from opal_sympy.conservation import verify_mass_conservation
    assert verify_mass_conservation()
    print("PASS [test_mixture_mass_conservation]")


def test_codegen_modelica():
    from opal_sympy.codegen import to_modelica
    x, y = sp.symbols('x y')
    result = to_modelica(x**2 + 2*y, varname='z')
    assert 'z =' in result
    assert ';' in result
    print(f"PASS [test_codegen_modelica]: {result.strip()}")


def test_codegen_c():
    from opal_sympy.codegen import to_c
    x, y = sp.symbols('x y')
    result = to_c(x**2 + 2*y, 'compute_z', [x, y])
    assert 'double compute_z' in result
    assert 'return' in result
    print(f"PASS [test_codegen_c]")


def test_codegen_numpy():
    from opal_sympy.codegen import to_numpy
    x, y = sp.symbols('x y')
    f = to_numpy(x**2 + 2*y, [x, y])
    assert abs(f(3.0, 1.0) - 11.0) < 1e-12
    print("PASS [test_codegen_numpy]")


def test_thermo_linearization():
    from opal_sympy.thermo import linearize_around, drho_l_dP_h
    from opal_sympy.symbols import P, P_old, rho_l

    lin = linearize_around(rho_l, P, P_old, {rho_l: drho_l_dP_h})
    # Should be rho_l + drho_l_dP_h * (P - P_old)
    expected = rho_l + drho_l_dP_h * (P - P_old)
    assert sp.simplify(lin - expected) == 0
    print("PASS [test_thermo_linearization]")


def test_donor_cell_regularized():
    """Regularized donor cell should match upwind for large velocities."""
    from opal_sympy.stencil import donor_cell_regularized_1d
    from opal_sympy.codegen import to_numpy
    from opal_sympy.symbols import i

    phi = sp.IndexedBase('phi')
    v = sp.IndexedBase('v')
    eps = sp.Symbol('eps_donor', positive=True)

    flux_expr = donor_cell_regularized_1d(phi, v, eps)

    # Substitute concrete indices for evaluation
    flux_concrete = flux_expr.subs(i, 5)
    syms = [phi[5], phi[6], v[5], eps]
    f = sp.lambdify(syms, flux_concrete, modules='numpy')

    # Large positive velocity: should give v * phi_W
    phi_W, phi_E, v_pos, eps_val = 2.0, 5.0, 100.0, 1e-8
    result = f(phi_W, phi_E, v_pos, eps_val)
    expected = v_pos * phi_W
    assert abs(result - expected) / abs(expected) < 1e-6, f"Got {result}, expected {expected}"

    # Large negative velocity: should give v * phi_E
    v_neg = -100.0
    result = f(phi_W, phi_E, v_neg, eps_val)
    expected = v_neg * phi_E
    assert abs(result - expected) / abs(expected) < 1e-6
    print("PASS [test_donor_cell_regularized]")


def test_verify_conservation():
    """check_conservation should pass for a known-zero expression."""
    from opal_sympy.verify import check_conservation
    from opal_sympy.symbols import alpha, rho_l, rho_v

    # Mixture density definition minus itself = 0
    rho_m = (1 - alpha) * rho_l + alpha * rho_v
    residual = rho_m - (rho_l + alpha * (rho_v - rho_l))
    assert check_conservation(residual, description="rho_m identity")


if __name__ == '__main__':
    test_symbols_exist()
    test_stencil_1d()
    test_stencil_3d()
    test_laplacian_symmetry()
    test_mixture_mass_conservation()
    test_codegen_modelica()
    test_codegen_c()
    test_codegen_numpy()
    test_thermo_linearization()
    test_donor_cell_regularized()
    test_verify_conservation()
    print("\n══════════════════════════════════════")
    print("ALL TESTS PASSED")
    print("══════════════════════════════════════")
