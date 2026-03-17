"""
Standard conservation equation builders for the two-fluid model.

These return symbolic expressions in continuous (PDE) form. Discretization
is done separately using stencil helpers. This separation keeps the physics
definition clean and lets SymPy verify conservation properties before
any numerics are applied.

The two-fluid model solves separate conservation equations for liquid and
vapor phases, coupled through interfacial transfer terms.

Reference: Ishii & Hibiki, "Thermo-Fluid Dynamics of Two-Phase Flow" (2nd ed.)
"""

import sympy as sp
from opal_sympy.symbols import (
    P, alpha, rho_l, rho_v, h_l, h_v, v_l, v_v,
    t, z, g, A_flow, D_h,
    q_wall, q_vol, Gamma, F_wall, F_i,
)


# ══════════════════════════════════════════════════════════════════════════
# 1D Two-Fluid Conservation Equations (continuous form)
# ══════════════════════════════════════════════════════════════════════════
# These are functions of (z, t). Spatial derivatives use sp.diff(expr, z)
# or are left in flux form for the user to discretize.

def liquid_mass_1d():
    """
    Liquid mass conservation (1D, area-averaged):

    ∂/∂t[(1-α)ρ_l] + ∂/∂z[(1-α)ρ_l·v_l] = -Γ

    Returns: (accumulation_term, flux_term, source_term) as SymPy expressions.
    Equation is: accumulation = -flux + source (conservative form).

    Γ > 0 means evaporation (liquid mass sink, vapor mass source).
    """
    accum = (1 - alpha) * rho_l
    flux = (1 - alpha) * rho_l * v_l
    source = -Gamma
    return accum, flux, source


def vapor_mass_1d():
    """
    Vapor mass conservation (1D, area-averaged):

    ∂/∂t[α·ρ_v] + ∂/∂z[α·ρ_v·v_v] = +Γ

    Returns: (accumulation, flux, source)
    """
    accum = alpha * rho_v
    flux = alpha * rho_v * v_v
    source = Gamma
    return accum, flux, source


def mixture_mass_1d():
    """
    Mixture mass conservation (sum of liquid + vapor).
    Γ cancels — this is the constraint that total mass is conserved.

    Returns: (accumulation, flux, source)
    """
    l_acc, l_flux, l_src = liquid_mass_1d()
    v_acc, v_flux, v_src = vapor_mass_1d()
    return (
        sp.expand(l_acc + v_acc),
        sp.expand(l_flux + v_flux),
        sp.expand(l_src + v_src),  # should be zero
    )


def liquid_momentum_1d():
    """
    Liquid momentum conservation (1D, simplified):

    ∂/∂t[(1-α)ρ_l·v_l] + ∂/∂z[(1-α)ρ_l·v_l²]
        = -(1-α)·∂P/∂z - F_wall_l - F_i_l - (1-α)ρ_l·g - Γ·v_li

    Simplified: wall friction and interfacial drag returned as abstract
    symbols (F_wall, F_i). The momentum transfer at interface (Γ·v_li)
    uses the donor velocity convention.

    Returns: (accumulation, flux, pressure_term, friction, drag, gravity, interface_transfer)
    """
    dPdz = sp.Symbol('dPdz', real=True)  # placeholder for ∂P/∂z

    accum = (1 - alpha) * rho_l * v_l
    flux = (1 - alpha) * rho_l * v_l**2
    pressure = -(1 - alpha) * dPdz
    friction = -F_wall  # wall friction on liquid (to be closed)
    drag = -F_i         # interfacial drag on liquid
    gravity = -(1 - alpha) * rho_l * g

    return {
        'accumulation': accum,
        'advective_flux': flux,
        'pressure_gradient': pressure,
        'wall_friction': friction,
        'interfacial_drag': drag,
        'gravity': gravity,
    }


def liquid_energy_1d():
    """
    Liquid energy conservation (enthalpy form, 1D):

    ∂/∂t[(1-α)ρ_l·h_l] + ∂/∂z[(1-α)ρ_l·h_l·v_l]
        = (1-α)·∂P/∂t + q_wall_l + q_i_l - Γ·h_li

    Returns dict of terms.
    """
    dPdt = sp.Symbol('dPdt', real=True)
    q_wall_l = sp.Symbol('q_wall_l', real=True)  # wall-to-liquid heat
    q_i_l = sp.Symbol('q_i_l', real=True)        # interface-to-liquid heat

    return {
        'accumulation': (1 - alpha) * rho_l * h_l,
        'advective_flux': (1 - alpha) * rho_l * h_l * v_l,
        'pressure_work': (1 - alpha) * dPdt,
        'wall_heat': q_wall_l,
        'interfacial_heat': q_i_l,
        'phase_change_enthalpy': -Gamma * h_l,  # donor enthalpy convention
    }


def vapor_energy_1d():
    """Vapor energy conservation, same structure as liquid."""
    dPdt = sp.Symbol('dPdt', real=True)
    q_wall_v = sp.Symbol('q_wall_v', real=True)
    q_i_v = sp.Symbol('q_i_v', real=True)

    return {
        'accumulation': alpha * rho_v * h_v,
        'advective_flux': alpha * rho_v * h_v * v_v,
        'pressure_work': alpha * dPdt,
        'wall_heat': q_wall_v,
        'interfacial_heat': q_i_v,
        'phase_change_enthalpy': Gamma * h_v,  # donor enthalpy convention
    }


# ══════════════════════════════════════════════════════════════════════════
# Conservation verification helpers
# ══════════════════════════════════════════════════════════════════════════

def verify_mass_conservation():
    """
    Verify that Γ cancels when liquid + vapor mass equations are summed.
    Returns True if mixture mass equation has zero source term.
    """
    _, _, mix_source = mixture_mass_1d()
    return sp.simplify(mix_source) == 0


def verify_energy_interface_balance():
    """
    Verify that interfacial heat and phase change terms are consistent
    between liquid and vapor energy equations.

    The interface energy balance requires:
        q_i_l + q_i_v + Γ·h_fg = 0
    where h_fg = h_v - h_l (latent heat).

    Returns the interface energy residual (should be zero when properly closed).
    """
    l_en = liquid_energy_1d()
    v_en = vapor_energy_1d()

    # Sum of interfacial heat transfer terms
    q_i_sum = l_en['interfacial_heat'] + v_en['interfacial_heat']

    # Sum of phase change enthalpy terms
    phase_sum = l_en['phase_change_enthalpy'] + v_en['phase_change_enthalpy']

    return sp.expand(q_i_sum + phase_sum)


# ══════════════════════════════════════════════════════════════════════════
# Point kinetics equations
# ══════════════════════════════════════════════════════════════════════════

def point_kinetics_odes(n_groups=6):
    """
    Standard point kinetics equations with n delayed neutron groups.

    dN/dt = (ρ - β)/Λ · N + Σ_i λ_i · C_i
    dC_i/dt = β_i/Λ · N - λ_i · C_i    for i = 1..n_groups

    Returns: dict with 'dN_dt' and 'dC_dt' as SymPy expressions.
    'dC_dt' is a list of length n_groups.
    """
    from opal_sympy.symbols import (
        N_power, Lambda, rho_reac, beta_total, beta_i, lambda_i, C_i
    )

    # Create concrete indexed symbols for each group
    i_sym = sp.Symbol('i', integer=True, positive=True)

    # Power equation
    precursor_sum = sum(lambda_i[grp] * C_i[grp] for grp in range(1, n_groups + 1))
    dN_dt = (rho_reac - beta_total) / Lambda * N_power + precursor_sum

    # Precursor equations
    dC_dt = []
    for grp in range(1, n_groups + 1):
        dC_dt.append(beta_i[grp] / Lambda * N_power - lambda_i[grp] * C_i[grp])

    return {
        'dN_dt': dN_dt,
        'dC_dt': dC_dt,
        'n_groups': n_groups,
    }
