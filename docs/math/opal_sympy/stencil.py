"""
Stencil helpers for structured mesh finite difference/volume discretizations.

Provides functions to build neighbor references on 1D, 2D, and 3D structured
meshes. The returned expressions use SymPy Indexed objects so that the
discretized equations are symbolic and can be manipulated, verified, and
code-generated.

Convention:
    - Cell-centered quantities: indexed at (i), (i,j), or (i,j,k)
    - Face quantities (staggered): east face of cell i = index (i) in face array
      (i.e., face i sits between cell i and cell i+1)

1D: east/west
2D: east/west/north/south
3D: east/west/north/south/top/bottom  (or +r/-r, +theta/-theta, +z/-z)
"""

import sympy as sp
from opal_sympy.symbols import i, j, k


def _make_indexed(base, *indices):
    """Create an Indexed expression from a base and index tuple."""
    if isinstance(base, sp.Indexed):
        return base.base[indices]
    if isinstance(base, sp.IndexedBase):
        return base[indices]
    # Assume it's a symbol — wrap it
    return sp.IndexedBase(base.name)[indices]


# ══════════════════════════════════════════════════════════════════════════
# 1D stencil (pipe, channel)
# ══════════════════════════════════════════════════════════════════════════

def center_1d(field):
    """Cell center value: field[i]"""
    return _make_indexed(field, i)

def east_1d(field):
    """East neighbor: field[i+1]"""
    return _make_indexed(field, i + 1)

def west_1d(field):
    """West neighbor: field[i-1]"""
    return _make_indexed(field, i - 1)

def face_east_1d(field):
    """East face value (staggered): field_face[i]"""
    return _make_indexed(field, i)

def face_west_1d(field):
    """West face value (staggered): field_face[i-1]"""
    return _make_indexed(field, i - 1)


def donor_cell_1d(field, velocity_face):
    """
    Donor-cell (upwind) advective flux at east face of cell i.

    If velocity > 0: use field[i]   (upwind = west cell)
    If velocity < 0: use field[i+1] (upwind = east cell)

    Returns a SymPy Piecewise, but for code generation this should be
    replaced with a smooth regularization. The symbolic form is for
    verification (showing it reduces to first-order upwind).

    For the regularized version used in actual OPAL code, see
    donor_cell_regularized_1d().
    """
    v = face_east_1d(velocity_face)
    phi_W = center_1d(field)
    phi_E = east_1d(field)
    return sp.Piecewise(
        (v * phi_W, v >= 0),
        (v * phi_E, True)
    )


def donor_cell_regularized_1d(field, velocity_face, epsilon=None):
    """
    Regularized donor-cell flux at east face, suitable for event-free
    Modelica code. Uses smooth absolute value:

        |v|_reg = sqrt(v^2 + ε^2)
        flux = v * (φ_W + φ_E)/2 + |v|_reg * (φ_W - φ_E)/2

    This is algebraically equivalent to upwind for |v| >> ε and
    smooth everywhere.
    """
    if epsilon is None:
        epsilon = sp.Symbol('eps_donor', positive=True)

    v = face_east_1d(velocity_face)
    phi_W = center_1d(field)
    phi_E = east_1d(field)
    v_abs = sp.sqrt(v**2 + epsilon**2)

    flux = v * (phi_W + phi_E) / 2 + v_abs * (phi_W - phi_E) / 2
    return sp.expand(flux)


# ══════════════════════════════════════════════════════════════════════════
# 3D stencil (vessel, r-theta-z)
# ══════════════════════════════════════════════════════════════════════════

def center_3d(field):
    """Cell center: field[i,j,k]"""
    return _make_indexed(field, i, j, k)

def east_3d(field):
    """field[i+1, j, k]  (+r direction)"""
    return _make_indexed(field, i + 1, j, k)

def west_3d(field):
    """field[i-1, j, k]  (-r direction)"""
    return _make_indexed(field, i - 1, j, k)

def north_3d(field):
    """field[i, j+1, k]  (+theta direction)"""
    return _make_indexed(field, i, j + 1, k)

def south_3d(field):
    """field[i, j-1, k]  (-theta direction)"""
    return _make_indexed(field, i, j - 1, k)

def top_3d(field):
    """field[i, j, k+1]  (+z direction)"""
    return _make_indexed(field, i, j, k + 1)

def bottom_3d(field):
    """field[i, j, k-1]  (-z direction)"""
    return _make_indexed(field, i, j, k - 1)


def laplacian_7pt(field, dx_sym, dy_sym, dz_sym):
    """
    7-point Laplacian stencil on uniform Cartesian mesh:
    ∇²φ ≈ (φ_E - 2φ_C + φ_W)/dx² + (φ_N - 2φ_C + φ_S)/dy² + (φ_T - 2φ_C + φ_B)/dz²

    For the diffusion equation discretization.
    """
    C = center_3d(field)
    E = east_3d(field)
    W = west_3d(field)
    N = north_3d(field)
    S = south_3d(field)
    T = top_3d(field)
    B = bottom_3d(field)

    return (
        (E - 2*C + W) / dx_sym**2
        + (N - 2*C + S) / dy_sym**2
        + (T - 2*C + B) / dz_sym**2
    )


def divergence_face_fluxes_3d(flux_r, flux_theta, flux_z, dr_sym, dtheta_sym, dz_sym):
    """
    Discrete divergence using face fluxes on a structured mesh:
    ∇·F ≈ (F_r[i] - F_r[i-1])/dr + (F_θ[j] - F_θ[j-1])/dθ + (F_z[k] - F_z[k-1])/dz

    Face fluxes are indexed at faces: flux_r[i] = flux at east face of cell i.
    """
    div_r = (_make_indexed(flux_r, i, j, k) - _make_indexed(flux_r, i - 1, j, k)) / dr_sym
    div_t = (_make_indexed(flux_theta, i, j, k) - _make_indexed(flux_theta, i, j - 1, k)) / dtheta_sym
    div_z = (_make_indexed(flux_z, i, j, k) - _make_indexed(flux_z, i, j, k - 1)) / dz_sym
    return div_r + div_t + div_z


# ══════════════════════════════════════════════════════════════════════════
# Aliases for readability in derivation scripts
# ══════════════════════════════════════════════════════════════════════════

# 1D shortcuts
center = center_1d
east = east_1d
west = west_1d

# 3D shortcuts available by explicit import:
#   from opal_sympy.stencil import center_3d, east_3d, west_3d, ...
