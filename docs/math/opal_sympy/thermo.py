"""
Thermodynamic derivative symbols and identities.

The semi-implicit pressure solve requires analytical derivatives of
thermodynamic properties. This module defines them symbolically and
provides identities for verification.

Naming: d{property}_d{variable}_{held_constant}
    drho_dP_h  = (∂ρ/∂P)|_h
    drho_dh_P  = (∂ρ/∂h)|_P
    dT_dP_h    = (∂T/∂P)|_h
"""

import sympy as sp

# ── Liquid-phase derivatives ──────────────────────────────────────────────
# Density derivatives (needed for pressure matrix)
drho_l_dP_h = sp.Symbol('drho_l_dP_h', real=True)   # (∂ρ_l/∂P)|_h
drho_l_dh_P = sp.Symbol('drho_l_dh_P', real=True)   # (∂ρ_l/∂h)|_P

# Temperature derivatives
dT_l_dP_h = sp.Symbol('dT_l_dP_h', real=True)       # (∂T_l/∂P)|_h
dT_l_dh_P = sp.Symbol('dT_l_dh_P', real=True)       # (∂T_l/∂h)|_P

# Enthalpy derivatives
dh_l_dP_T = sp.Symbol('dh_l_dP_T', real=True)       # (∂h_l/∂P)|_T
dh_l_dT_P = sp.Symbol('dh_l_dT_P', real=True)       # = c_p,l

# ── Vapor-phase derivatives ───────────────────────────────────────────────
drho_v_dP_h = sp.Symbol('drho_v_dP_h', real=True)
drho_v_dh_P = sp.Symbol('drho_v_dh_P', real=True)
dT_v_dP_h = sp.Symbol('dT_v_dP_h', real=True)
dT_v_dh_P = sp.Symbol('dT_v_dh_P', real=True)
dh_v_dP_T = sp.Symbol('dh_v_dP_T', real=True)
dh_v_dT_P = sp.Symbol('dh_v_dT_P', real=True)       # = c_p,v

# ── Saturation derivatives ────────────────────────────────────────────────
dT_sat_dP = sp.Symbol('dT_sat_dP', real=True)        # Clausius-Clapeyron
dh_sat_l_dP = sp.Symbol('dh_sat_l_dP', real=True)
dh_sat_v_dP = sp.Symbol('dh_sat_v_dP', real=True)
drho_sat_l_dP = sp.Symbol('drho_sat_l_dP', real=True)
drho_sat_v_dP = sp.Symbol('drho_sat_v_dP', real=True)

# ── Specific heats ────────────────────────────────────────────────────────
cp_l = sp.Symbol('cp_l', positive=True)
cp_v = sp.Symbol('cp_v', positive=True)
cv_l = sp.Symbol('cv_l', positive=True)
cv_v = sp.Symbol('cv_v', positive=True)

# ── Speed of sound ────────────────────────────────────────────────────────
c_sound_l = sp.Symbol('c_sound_l', positive=True)
c_sound_v = sp.Symbol('c_sound_v', positive=True)

# ── Transport properties ──────────────────────────────────────────────────
mu_l = sp.Symbol('mu_l', positive=True)    # dynamic viscosity
mu_v = sp.Symbol('mu_v', positive=True)
k_l = sp.Symbol('k_l', positive=True)      # thermal conductivity
k_v = sp.Symbol('k_v', positive=True)
sigma = sp.Symbol('sigma', positive=True)   # surface tension


def mixture_density(alpha_expr, rho_l_expr, rho_v_expr):
    """
    Mixture density: ρ_m = (1 - α)ρ_l + αρ_v

    >>> from opal_sympy.symbols import alpha, rho_l, rho_v
    >>> mixture_density(alpha, rho_l, rho_v)
    alpha*rho_v + rho_l*(1 - alpha)
    """
    return (1 - alpha_expr) * rho_l_expr + alpha_expr * rho_v_expr


def mixture_density_derivative_P(alpha_expr):
    """
    ∂ρ_m/∂P|_h = (1 - α)(∂ρ_l/∂P)|_h + α(∂ρ_v/∂P)|_h

    This is the diagonal contribution to the semi-implicit pressure matrix.
    """
    return (1 - alpha_expr) * drho_l_dP_h + alpha_expr * drho_v_dP_h


def linearize_around(expr, var, var_old, derivatives):
    """
    First-order Taylor expansion of expr around var = var_old.

    Args:
        expr: SymPy expression to linearize
        var: the variable being perturbed (e.g., P)
        var_old: the symbol for the old/reference value (e.g., P_old)
        derivatives: dict mapping symbols in expr to their derivatives w.r.t. var

    Returns:
        Linearized expression.

    Example:
        Linearize ρ_l(P) around P_old:
        >>> from opal_sympy.symbols import P, P_old, rho_l
        >>> linearize_around(rho_l, P, P_old, {rho_l: drho_l_dP_h})
        drho_l_dP_h*(P - P_old) + rho_l
    """
    result = expr
    for sym, deriv in derivatives.items():
        result = result + deriv * (var - var_old)
    return sp.expand(result)


def verify_maxwell_relation(expr1, expr2, subs_dict=None):
    """
    Check that two thermodynamic expressions are equal (Maxwell relation
    or other identity). Returns True if they simplify to the same thing.

    Args:
        expr1, expr2: SymPy expressions
        subs_dict: optional substitutions to apply before comparing
    """
    diff = sp.simplify(expr1 - expr2)
    if subs_dict:
        diff = diff.subs(subs_dict)
    return sp.simplify(diff) == 0
