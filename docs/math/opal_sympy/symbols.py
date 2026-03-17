"""
Predefined symbols for OPAL thermal-hydraulic equations.

Naming convention:
    - Scalar fields: P, T, alpha, rho, h, u (internal energy), s (entropy)
    - Subscripts: _l (liquid), _v (vapor), _m (mixture), _sat (saturation)
    - Velocities: v (scalar speed), v_l, v_v
    - Geometry: dx, dA, V_cell, A_flow, D_h (hydraulic diameter)
    - Time: t, dt
    - Indices: i, j, k (spatial), n (time level)
    - Source terms: q_vol (volumetric heat), q_wall (wall heat flux)

All symbols are real. Physically positive quantities are declared positive.
"""

import sympy as sp

# ── Time ──────────────────────────────────────────────────────────────────
t = sp.Symbol('t', real=True)
dt = sp.Symbol('dt', positive=True)
n = sp.Symbol('n', integer=True, nonnegative=True)  # time level index

# ── Spatial indices ───────────────────────────────────────────────────────
i, j, k = sp.symbols('i j k', integer=True, positive=True)
Nr, Ntheta, Nz = sp.symbols('Nr Ntheta Nz', integer=True, positive=True)

# ── Pressure and temperature ──────────────────────────────────────────────
P = sp.Symbol('P', real=True)
P_old = sp.Symbol('P_old', real=True)  # previous time step
T = sp.Symbol('T', real=True)
T_l = sp.Symbol('T_l', real=True)
T_v = sp.Symbol('T_v', real=True)
T_sat = sp.Symbol('T_sat', real=True)
T_wall = sp.Symbol('T_wall', real=True)
T_fuel = sp.Symbol('T_fuel', real=True)

# ── Void fraction ─────────────────────────────────────────────────────────
alpha = sp.Symbol('alpha', real=True)  # not constrained to [0,1] for generality
alpha_old = sp.Symbol('alpha_old', real=True)

# ── Densities (always positive) ──────────────────────────────────────────
rho_l = sp.Symbol('rho_l', positive=True)
rho_v = sp.Symbol('rho_v', positive=True)
rho_m = sp.Symbol('rho_m', positive=True)

# ── Specific enthalpies and internal energies ─────────────────────────────
h_l = sp.Symbol('h_l', real=True)
h_v = sp.Symbol('h_v', real=True)
h_sat_l = sp.Symbol('h_sat_l', real=True)  # saturated liquid enthalpy
h_sat_v = sp.Symbol('h_sat_v', real=True)  # saturated vapor enthalpy
u_l = sp.Symbol('u_l', real=True)  # specific internal energy, liquid
u_v = sp.Symbol('u_v', real=True)  # specific internal energy, vapor

# ── Velocities ────────────────────────────────────────────────────────────
v_l = sp.Symbol('v_l', real=True)
v_v = sp.Symbol('v_v', real=True)
v_m = sp.Symbol('v_m', real=True)  # mixture velocity

# ── Geometry ──────────────────────────────────────────────────────────────
dx = sp.Symbol('dx', positive=True)
dy = sp.Symbol('dy', positive=True)
dz = sp.Symbol('dz', positive=True)
dr = sp.Symbol('dr', positive=True)
dtheta = sp.Symbol('dtheta', positive=True)
dA = sp.Symbol('dA', positive=True)       # flow area
A_flow = sp.Symbol('A_flow', positive=True)
V_cell = sp.Symbol('V_cell', positive=True)
D_h = sp.Symbol('D_h', positive=True)     # hydraulic diameter
L = sp.Symbol('L', positive=True)         # length
z = sp.Symbol('z', real=True)             # elevation coordinate

# ── Source terms ──────────────────────────────────────────────────────────
q_vol = sp.Symbol('q_vol', real=True)      # volumetric heat source [W/m^3]
q_wall = sp.Symbol('q_wall', real=True)    # wall heat flux [W/m^2]
Gamma = sp.Symbol('Gamma', real=True)      # interfacial mass transfer rate [kg/m^3/s]
                                            # positive = evaporation

# ── Friction and drag ────────────────────────────────────────────────────
f_D = sp.Symbol('f_D', nonnegative=True)   # Darcy friction factor
K_loss = sp.Symbol('K_loss', nonnegative=True)  # form loss coefficient
F_wall = sp.Symbol('F_wall', real=True)    # wall friction force per unit volume
F_i = sp.Symbol('F_i', real=True)          # interfacial drag force per unit volume

# ── Physical constants ────────────────────────────────────────────────────
g = sp.Symbol('g', positive=True)          # gravitational acceleration
g_earth = sp.Rational(981, 100)            # 9.81 m/s^2 as exact rational

# ── Kinetics ──────────────────────────────────────────────────────────────
N_power = sp.Symbol('N', positive=True)     # reactor power (amplitude)
Lambda = sp.Symbol('Lambda', positive=True) # prompt neutron generation time
rho_reac = sp.Symbol('rho_reac', real=True) # reactivity (not density!)
beta_total = sp.Symbol('beta', positive=True)  # total delayed neutron fraction
beta_i = sp.IndexedBase('beta_i')           # delayed neutron fraction, group i
lambda_i = sp.IndexedBase('lambda_i')       # decay constant, group i
C_i = sp.IndexedBase('C_i')                # precursor concentration, group i
N_groups = sp.Symbol('N_groups', integer=True, positive=True)

# ── Convenience: group commonly used symbols ──────────────────────────────

# All scalar TH state variables
th_state_vars = [P, alpha, T_l, T_v, rho_l, rho_v, h_l, h_v]

# All velocity variables (1D)
velocity_vars = [v_l, v_v]

# All geometry variables
geometry_vars = [dx, dA, A_flow, V_cell, D_h, L]
