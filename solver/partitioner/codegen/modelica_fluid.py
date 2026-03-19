"""
modelica_fluid.py — FluidPackage backed by OM-generated C code via ctypes.

Case 2 implementation: ALL property evaluation comes from Modelica source
compiled to C by OpenModelica's translateModel. Zero C++ dependency.

Drop-in replacement for the C++ FluidPackage used by Parameterized5EqSolver:
    fluid = ModelicaFluidPackage("opal_codegen_InlineTest.so")
    solver = Parameterized5EqSolver(fluid, spec)
"""

import ctypes
from pathlib import Path


class FluidProperties:
    """Return type matching C++ FluidProperties struct."""
    __slots__ = ('rho', 'drho_dp_h', 'drho_dh_p', 'T')


class PhasicProperties:
    """Return type matching C++ PhasicProperties struct."""
    __slots__ = ('T_sat', 'h_sat_l', 'h_sat_v', 'rho_l', 'rho_g', 'sigma',
                 'cp_l', 'cp_v', 'drho_l_dp', 'drho_v_dp')


class ModelicaFluidPackage:
    """FluidPackage backed by OM-generated C code via ctypes.

    Provides evaluate(p, h) and evaluate_phasic(p) matching the interface
    expected by Parameterized5EqSolver.
    """

    def __init__(self, so_path, p_min=700.0, p_max=21e6):
        """
        Args:
            so_path: Path to compiled .so from build_codegen.py
            p_min: Minimum valid pressure [Pa]
            p_max: Maximum valid pressure [Pa]
        """
        so_path = str(so_path)
        self.lib = ctypes.CDLL(so_path)
        self.p_min = p_min
        self.p_max = p_max
        self._setup_functions()

    def _setup_functions(self):
        """Configure ctypes function signatures."""
        D = ctypes.c_double

        def _bind(name, argtypes):
            fn = getattr(self.lib, name)
            fn.restype = D
            fn.argtypes = argtypes
            return fn

        # Mixture properties: f(p, h)
        self._rho_ph = _bind('opal_rho_ph', [D, D])
        self._drho_dp_h = _bind('opal_drho_dp_h', [D, D])
        self._drho_dh_p = _bind('opal_drho_dh_p', [D, D])
        self._T_ph = _bind('opal_T_ph', [D, D])

        # Saturation properties: f(p)
        self._T_sat = _bind('opal_T_sat', [D])
        self._h_f = _bind('opal_h_f', [D])
        self._h_g = _bind('opal_h_g', [D])
        self._h_fg = _bind('opal_h_fg', [D])
        self._rho_f = _bind('opal_rho_f', [D])
        self._rho_g = _bind('opal_rho_g', [D])

        # Optional: sigma may not be exported (depends on model)
        try:
            self._sigma_fn = _bind('opal_sigma', [D])
        except AttributeError:
            self._sigma_fn = None

    def evaluate(self, p, h):
        """Evaluate mixture properties at (p, h).

        Returns FluidProperties with .rho, .drho_dp_h, .drho_dh_p, .T
        """
        fp = FluidProperties()
        fp.rho = self._rho_ph(p, h)
        fp.drho_dp_h = self._drho_dp_h(p, h)
        fp.drho_dh_p = self._drho_dh_p(p, h)
        fp.T = self._T_ph(p, h)
        return fp

    def evaluate_phasic(self, p):
        """Evaluate saturation/phasic properties at pressure p.

        Returns PhasicProperties with .T_sat, .h_sat_l, .h_sat_v, .rho_l, .rho_g, .sigma
        """
        pp = PhasicProperties()
        pp.T_sat = self._T_sat(p)
        pp.h_sat_l = self._h_f(p)
        pp.h_sat_v = self._h_g(p)
        pp.rho_l = self._rho_f(p)
        pp.rho_g = self._rho_g(p)
        pp.sigma = self._sigma_fn(p) if self._sigma_fn else 0.06
        # These are not directly available from SimpleFluid's codegen
        # (not in PartialMedium interface). Set reasonable defaults.
        pp.cp_l = 4000.0
        pp.cp_v = 2000.0
        pp.drho_l_dp = 0.0
        pp.drho_v_dp = 0.0
        return pp
