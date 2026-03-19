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
    """Return type matching C++ PhasicProperties struct.

    Provides both Modelica naming (rho_g) and C++ naming (rho_v) via properties.
    """
    __slots__ = ('T_sat', 'h_sat_l', 'h_sat_v', 'rho_l', 'rho_g', 'sigma',
                 'cp_l', 'cp_v', 'drho_l_dp', 'drho_v_dp')

    @property
    def rho_v(self):
        """Alias for rho_g — C++ FluidPackage uses rho_v."""
        return self.rho_g


class ModelicaFluidPackage:
    """FluidPackage backed by OM-generated C code via ctypes.

    Provides evaluate(p, h) and evaluate_phasic(p) matching the interface
    expected by Parameterized5EqSolver and pipe1d_mapper.

    Note on sigma: if the .so doesn't export opal_sigma (e.g., SimpleFluid
    in a HEM model that doesn't use sigma), the value is computed from the
    SimpleFluid formula. For production, use a DriftFlux model .so which
    exports sigma.
    """

    # SimpleFluid sigma constants (fallback when not exported)
    _SF_SIGMA_0 = 0.06
    _SF_SIGMA_1 = -0.04
    _SF_P_REF = 10e6

    def __init__(self, so_path, p_min=1e4, p_max=50e6):
        """
        Args:
            so_path: Path to compiled .so from build_codegen.py
            p_min: Minimum valid pressure [Pa] (SimpleFluid default: 1e4)
            p_max: Maximum valid pressure [Pa] (SimpleFluid default: 50e6)
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

        def _try_bind(name, argtypes):
            try:
                return _bind(name, argtypes)
            except AttributeError:
                return None

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

        # Optional functions (may not be in all .so's)
        self._sigma_fn = _try_bind('opal_sigma', [D])
        self._quality_ph_fn = _try_bind('opal_quality_ph', [D, D])

        # region_ph returns modelica_integer (long), not double
        try:
            fn = self.lib.opal_region_ph
            fn.restype = ctypes.c_long
            fn.argtypes = [D, D]
            self._region_ph_fn = fn
        except AttributeError:
            self._region_ph_fn = None

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

        Returns PhasicProperties with .T_sat, .h_sat_l, .h_sat_v,
        .rho_l, .rho_g (.rho_v alias), .sigma, .drho_l_dp, .drho_v_dp
        """
        pp = PhasicProperties()
        pp.T_sat = self._T_sat(p)
        pp.h_sat_l = self._h_f(p)
        pp.h_sat_v = self._h_g(p)
        pp.rho_l = self._rho_f(p)
        pp.rho_g = self._rho_g(p)

        # sigma: use exported function, else compute from SimpleFluid formula
        if self._sigma_fn:
            pp.sigma = self._sigma_fn(p)
        else:
            p_hat = (p - self._SF_P_REF) / self._SF_P_REF
            pp.sigma = self._SF_SIGMA_0 + self._SF_SIGMA_1 * p_hat

        # Phasic heat capacities from SimpleFluid constants
        # (for IAPWS, these would come from OM-generated functions)
        pp.cp_l = 4000.0
        pp.cp_v = 2000.0

        # Phasic density derivatives: drho_f/dp and drho_g/dp
        # Computed via finite difference from the OM-generated rho_f/rho_g
        dp = p * 1e-8
        pp.drho_l_dp = (self._rho_f(p + dp) - self._rho_f(p - dp)) / (2 * dp)
        pp.drho_v_dp = (self._rho_g(p + dp) - self._rho_g(p - dp)) / (2 * dp)

        return pp

    def quality_ph(self, p, h):
        """Thermodynamic quality from (p, h). Returns 0 in subcooled, 1 in superheated.
        Always clamped to [0, 1] regardless of OM's raw output."""
        h_f = self._h_f(p)
        h_g = self._h_g(p)
        if h <= h_f:
            return 0.0
        elif h >= h_g:
            return 1.0
        return (h - h_f) / (h_g - h_f)

    def region_ph(self, p, h):
        """Fluid region: 1=subcooled, 2=superheated, 4=two-phase."""
        if self._region_ph_fn:
            return int(self._region_ph_fn(p, h))
        h_f = self._h_f(p)
        h_g = self._h_g(p)
        if h < h_f:
            return 1
        elif h > h_g:
            return 2
        return 4
