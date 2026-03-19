"""
equation_bridge.py — Python wrapper for the OM equation-level C bridge.

Provides a clean API over the compiled bridge .so:
  bridge = OMEquationBridge(so_path, info)
  bridge.set_state(p, h, mdot)
  bridge.evaluate()
  rho = bridge.get_rho_face()

This replaces BOTH the FluidPackage property calls AND the Python
reimplementations of face density, donor-cell, closures, etc.
The solver becomes a pure numerical engine.
"""

import ctypes
import numpy as np
from pathlib import Path

from .info_parser import ModelInfo


class OMEquationBridge:
    """Direct bridge to OM-generated per-equation C functions.

    All algebraic evaluation (properties, face densities, donor-cell enthalpies,
    closures) happens in a single C call to opal_bridge_evaluate(). The Python
    solver only needs to set state, evaluate, and read results.
    """

    def __init__(self, so_path, info: ModelInfo):
        """
        Args:
            so_path: Path to compiled opal_bridge_{Model}.so
            info: Parsed ModelInfo from _info.json
        """
        self.lib = ctypes.CDLL(str(so_path))
        self.info = info
        self.prefix = self._detect_prefix()
        self.N = self._detect_N()

        self._D = ctypes.c_double
        self._I = ctypes.c_int
        self._DP = ctypes.POINTER(ctypes.c_double)

        self._setup_signatures()

        # Cache variable group indices for fast access
        self._p_idx = info.vars_by_pattern(f'{self.prefix}.p', self.N)
        self._h_idx = info.vars_by_pattern(f'{self.prefix}.h', self.N)
        self._rho_face_idx = info.vars_by_pattern(f'{self.prefix}.rho_face', self.N + 1)
        self._h_face_idx = info.vars_by_pattern(f'{self.prefix}.h_face', self.N + 1)
        self._drho_dp_idx = info.vars_by_pattern(f'{self.prefix}.drho_dp', self.N)
        self._drho_dh_idx = info.vars_by_pattern(f'{self.prefix}.drho_dh', self.N)
        self._T_cell_idx = info.vars_by_pattern(f'{self.prefix}.T_cell', self.N)

        # mdot indices (states + dummy states, sorted by Modelica index)
        self._mdot_idx = sorted(
            [(int(name.split('[')[1].rstrip(']')), vi.index)
             for name, vi in info.all_vars.items()
             if f'{self.prefix}.mdot[' in name and vi.kind in ('state', 'dummy state')],
            key=lambda x: x[0]
        )

        # Pressure bounds (from SimpleFluid defaults — override via set_pressure_bounds)
        self.p_min = 1e4
        self.p_max = 50e6

    def _detect_prefix(self) -> str:
        for name in self.info.states:
            if '.p[' in name:
                return name.split('.p[')[0]
        raise ValueError("Cannot detect prefix")

    def _detect_N(self) -> int:
        return len([n for n in self.info.states if n.startswith(f'{self.prefix}.p[')])

    def _setup_signatures(self):
        """Configure ctypes function signatures."""
        D, I, DP = self._D, self._I, self._DP
        self.lib.opal_bridge_set_state.argtypes = [I, DP, I, DP, I, DP]
        self.lib.opal_bridge_set_params.argtypes = [I, DP]
        self.lib.opal_bridge_evaluate.argtypes = []
        self.lib.opal_bridge_get_rho_face.argtypes = [I, DP]
        self.lib.opal_bridge_get_h_face.argtypes = [I, DP]
        self.lib.opal_bridge_get_drho_dp.argtypes = [I, DP]
        self.lib.opal_bridge_get_drho_dh.argtypes = [I, DP]
        self.lib.opal_bridge_get_T_cell.argtypes = [I, DP]
        self.lib.opal_bridge_get_all_vars.argtypes = [I, DP]
        self.lib.opal_bridge_get_N.restype = I
        self.lib.opal_bridge_get_n_vars.restype = I
        self.lib.opal_bridge_get_n_params.restype = I

    def set_params_from_spec(self, spec):
        """Write parameters from a Pipe1DGridSpec or ExtractedModelSpec.

        Reads geometry and boundary parameters by name from the info map.
        """
        import math
        param_values = np.zeros(self.info.n_params)

        # Map spec attributes to parameter names
        param_map = {
            f'{self.prefix}.A_flow': spec.A_flow,
            f'{self.prefix}.D_h': spec.D_h,
            f'{self.prefix}.dx': spec.dx,
            f'{self.prefix}.f_D': spec.f_D,
            f'{self.prefix}.V_cell': spec.V_cell,
        }

        # Try to get L and D from spec
        if hasattr(spec, 'N'):
            N = spec.N
            L = spec.dx * N
            D = spec.D_h
            param_map[f'{self.prefix}.L'] = L
            param_map[f'{self.prefix}.D'] = D

        # Outlet pressure/enthalpy from boundary
        if hasattr(spec, 'p_out') and spec.p_out is not None:
            # Find the boundary parameter name (e.g., atm.p_set)
            for pname, pinfo in self.info.parameters.items():
                if pname.endswith('.p_set'):
                    param_values[pinfo.index] = spec.p_out
                if pname.endswith('.h_set') and hasattr(spec, 'h_out') and spec.h_out:
                    param_values[pinfo.index] = spec.h_out

        # Initial conditions
        if hasattr(spec, 'p0') and spec.p0:
            for pname, pinfo in self.info.parameters.items():
                if pname == f'{self.prefix}.p_init':
                    param_values[pinfo.index] = spec.p0[0]
        if hasattr(spec, 'h0') and spec.h0:
            for pname, pinfo in self.info.parameters.items():
                if pname == f'{self.prefix}.h_init':
                    param_values[pinfo.index] = spec.h0[0]

        # Write known parameters
        for pname, value in param_map.items():
            if pname in self.info.parameters:
                param_values[self.info.parameters[pname].index] = value

        self.lib.opal_bridge_set_params(
            self.info.n_params, param_values.ctypes.data_as(self._DP))

    def set_state(self, p: np.ndarray, h: np.ndarray, mdot: np.ndarray):
        """Write current state into the bridge arrays."""
        self.lib.opal_bridge_set_state(
            len(p), p.ctypes.data_as(self._DP),
            len(h), h.ctypes.data_as(self._DP),
            len(mdot), mdot.ctypes.data_as(self._DP),
        )

    def evaluate(self):
        """Call all algebraic OM equations in BLT order.

        After this call, all derived quantities (rho, rho_face, h_face,
        drho_dp, drho_dh, T_cell, etc.) are computed and readable.
        """
        self.lib.opal_bridge_evaluate()

    def get_rho_face(self) -> np.ndarray:
        """Face densities rho_face[1..N+1] (N+1 values, 0-indexed in output)."""
        out = np.zeros(self.N + 1)
        self.lib.opal_bridge_get_rho_face(self.N + 1, out.ctypes.data_as(self._DP))
        return out

    def get_drho_dp(self) -> np.ndarray:
        """Pressure derivative drho_dp[1..N] (N values, 0-indexed in output)."""
        out = np.zeros(self.N)
        self.lib.opal_bridge_get_drho_dp(self.N, out.ctypes.data_as(self._DP))
        return out

    def get_drho_dh(self) -> np.ndarray:
        """Enthalpy derivative drho_dh[1..N] (N values, 0-indexed in output)."""
        out = np.zeros(self.N)
        self.lib.opal_bridge_get_drho_dh(self.N, out.ctypes.data_as(self._DP))
        return out

    def get_h_face(self) -> np.ndarray:
        """Donor-cell face enthalpies h_face[1..N+1] (N+1 values, 0-indexed)."""
        out = np.zeros(self.N + 1)
        self.lib.opal_bridge_get_h_face(self.N + 1, out.ctypes.data_as(self._DP))
        return out

    def get_T_cell(self) -> np.ndarray:
        """Cell temperatures T_cell[1..N] (N values, 0-indexed)."""
        out = np.zeros(self.N)
        self.lib.opal_bridge_get_T_cell(self.N, out.ctypes.data_as(self._DP))
        return out

    def get_rho_cell(self) -> np.ndarray:
        """Cell densities rho[1..N] computed from rho_ph(p[i], h[i]).
        Uses the compiled media function directly (not from OM variables)."""
        out = np.zeros(self.N)
        try:
            self.lib.opal_bridge_get_rho_cell.argtypes = [self._I, self._DP]
            self.lib.opal_bridge_get_rho_cell(self.N, out.ctypes.data_as(self._DP))
        except AttributeError:
            # Fallback: compute from rho_face (boundary = cell density)
            rho_face = self.get_rho_face()
            out[0] = rho_face[0]
            out[-1] = rho_face[-1]
            for i in range(1, self.N - 1):
                out[i] = 2 * rho_face[i] - out[i - 1]
        return out

    def get_all_vars(self) -> np.ndarray:
        """Full variable array (for debugging)."""
        out = np.zeros(self.info.n_vars)
        self.lib.opal_bridge_get_all_vars(self.info.n_vars, out.ctypes.data_as(self._DP))
        return out
