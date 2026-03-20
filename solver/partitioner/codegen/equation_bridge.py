"""
equation_bridge.py — Python wrapper for the generic OM equation bridge.

The C bridge provides ONLY index-level access (set_vars/get_vars by integer index).
This Python wrapper adds name-to-index mapping using ModelInfo from _info.json.

    bridge = OMEquationBridge(so_path, info)
    bridge.set_state(p, h, mdot)       # maps variable names to indices
    bridge.evaluate()                   # calls OM equations in BLT order
    rho_face = bridge.get_rho_face()   # reads by index, returns numpy array
"""

import ctypes
import numpy as np
from pathlib import Path

from .info_parser import ModelInfo


class OMEquationBridge:
    """Generic bridge to OM-generated per-equation C functions.

    All name-to-index mapping happens here in Python, driven by ModelInfo.
    The C bridge is model-independent (only flat arrays + evaluate).
    """

    def __init__(self, so_path, info: ModelInfo):
        self.lib = ctypes.CDLL(str(so_path))
        self.info = info

        self._D = ctypes.c_double
        self._I = ctypes.c_int
        self._DP = ctypes.POINTER(ctypes.c_double)
        self._IP = ctypes.POINTER(ctypes.c_int)

        self._setup_signatures()

        # Detect model structure from info
        self.prefix = self._detect_prefix()
        self.N = self._detect_N()

        # Build index arrays for variable groups (computed once, reused every step)
        self._p_idx = self._c_indices(info.vars_by_pattern(f'{self.prefix}.p', self.N))
        self._h_idx = self._c_indices(info.vars_by_pattern(f'{self.prefix}.h', self.N))
        self._rho_face_idx = self._c_indices(info.vars_by_pattern(f'{self.prefix}.rho_face', self.N + 1))
        self._h_face_idx = self._c_indices(info.vars_by_pattern(f'{self.prefix}.h_face', self.N + 1))
        self._drho_dp_idx = self._c_indices(info.vars_by_pattern(f'{self.prefix}.drho_dp', self.N))
        self._drho_dh_idx = self._c_indices(info.vars_by_pattern(f'{self.prefix}.drho_dh', self.N))
        self._T_cell_idx = self._c_indices(info.vars_by_pattern(f'{self.prefix}.T_cell', self.N))

        # mdot: states + dummy states, sorted by Modelica index
        mdot_entries = sorted(
            [(int(name.split('[')[1].rstrip(']')), vi.index)
             for name, vi in info.all_vars.items()
             if f'{self.prefix}.mdot[' in name and vi.kind in ('state', 'dummy state')],
            key=lambda x: x[0]
        )
        self._mdot_idx = self._c_indices([idx for _, idx in mdot_entries])

        # Pressure bounds (defaults — can be overridden)
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
        """Configure ctypes function signatures for the generic C API."""
        D, I, DP, IP = self._D, self._I, self._DP, self._IP

        self.lib.opal_bridge_set_var.argtypes = [I, D]
        self.lib.opal_bridge_get_var.argtypes = [I]
        self.lib.opal_bridge_get_var.restype = D
        self.lib.opal_bridge_set_vars.argtypes = [I, IP, DP]
        self.lib.opal_bridge_get_vars.argtypes = [I, IP, DP]
        self.lib.opal_bridge_set_param.argtypes = [I, D]
        self.lib.opal_bridge_set_params.argtypes = [I, DP]
        self.lib.opal_bridge_evaluate.argtypes = []
        self.lib.opal_bridge_get_n_vars.restype = I
        self.lib.opal_bridge_get_n_params.restype = I

        # Media function: find the rho_ph wrapper (name varies by media package)
        self._rho_ph_fn = self._find_media_fn('rho_ph', [D, D])

    def _find_media_fn(self, suffix: str, argtypes: list):
        """Find a media function wrapper by suffix (e.g., 'rho_ph').

        The bridge exports media wrappers with names like:
          opal_bridge_rho_ph (SimpleFluid, short prefix)
          opal_bridge_WaterTest_pipe_Medium_rho_ph (Water, qualified prefix)
        We scan for any exported function ending in the suffix.
        """
        import subprocess
        D = self._D
        result = subprocess.run(['nm', '-gU', str(self.lib._name)],
                                capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                sym = parts[2].lstrip('_')
                if sym.startswith('opal_bridge_') and sym.endswith(f'_{suffix}'):
                    fn = getattr(self.lib, sym)
                    fn.restype = D
                    fn.argtypes = argtypes
                    return fn
        return None

    def _c_indices(self, indices: list[int]) -> ctypes.Array:
        """Convert Python index list to a ctypes int array (allocated once)."""
        return (ctypes.c_int * len(indices))(*indices)

    # ── State set/get (name-aware, index-driven) ──

    def set_state(self, p: np.ndarray, h: np.ndarray, mdot: np.ndarray):
        """Write current state into the bridge flat arrays."""
        self.lib.opal_bridge_set_vars(len(p), self._p_idx,
                                       p.ctypes.data_as(self._DP))
        self.lib.opal_bridge_set_vars(len(h), self._h_idx,
                                       h.ctypes.data_as(self._DP))
        self.lib.opal_bridge_set_vars(len(mdot), self._mdot_idx,
                                       mdot.ctypes.data_as(self._DP))

    def set_params_from_spec(self, spec):
        """Write parameters from a Pipe1DGridSpec."""
        param_values = np.zeros(self.info.n_params)

        # Map spec attributes to parameter names
        name_to_value = {
            f'{self.prefix}.A_flow': spec.A_flow,
            f'{self.prefix}.D_h': spec.D_h,
            f'{self.prefix}.dx': spec.dx,
            f'{self.prefix}.f_D': spec.f_D,
            f'{self.prefix}.V_cell': spec.V_cell,
        }
        if hasattr(spec, 'N'):
            name_to_value[f'{self.prefix}.L'] = spec.dx * spec.N
            name_to_value[f'{self.prefix}.D'] = spec.D_h

        for pname, pinfo in self.info.parameters.items():
            if pname in name_to_value:
                param_values[pinfo.index] = name_to_value[pname]
            elif pname.endswith('.p_set') and hasattr(spec, 'p_out') and spec.p_out:
                param_values[pinfo.index] = spec.p_out
            elif pname.endswith('.h_set') and hasattr(spec, 'h_out') and spec.h_out:
                param_values[pinfo.index] = spec.h_out
            elif pname == f'{self.prefix}.p_init' and hasattr(spec, 'p0') and spec.p0:
                param_values[pinfo.index] = spec.p0[0]
            elif pname == f'{self.prefix}.h_init' and hasattr(spec, 'h0') and spec.h0:
                param_values[pinfo.index] = spec.h0[0]

        self.lib.opal_bridge_set_params(
            self.info.n_params, param_values.ctypes.data_as(self._DP))

    def evaluate(self):
        """Call all algebraic OM equations in BLT order."""
        self.lib.opal_bridge_evaluate()

    # ── Named getters (Python maps names to indices, C does array[index]) ──

    def _get_array(self, c_indices, n: int) -> np.ndarray:
        """Generic getter: read n values at the given ctypes index array."""
        out = np.zeros(n)
        self.lib.opal_bridge_get_vars(n, c_indices, out.ctypes.data_as(self._DP))
        return out

    def get_rho_face(self) -> np.ndarray:
        return self._get_array(self._rho_face_idx, self.N + 1)

    def get_drho_dp(self) -> np.ndarray:
        return self._get_array(self._drho_dp_idx, self.N)

    def get_drho_dh(self) -> np.ndarray:
        return self._get_array(self._drho_dh_idx, self.N)

    def get_h_face(self) -> np.ndarray:
        return self._get_array(self._h_face_idx, self.N + 1)

    def get_T_cell(self) -> np.ndarray:
        return self._get_array(self._T_cell_idx, self.N)

    def get_rho_cell(self) -> np.ndarray:
        """Cell densities from rho_ph(p[i], h[i]) via bridge media wrapper."""
        if self._rho_ph_fn is None:
            raise RuntimeError("No rho_ph media function found in bridge .so")
        p_vals = self._get_array(self._p_idx, self.N)
        h_vals = self._get_array(self._h_idx, self.N)
        out = np.zeros(self.N)
        for i in range(self.N):
            out[i] = self._rho_ph_fn(p_vals[i], h_vals[i])
        return out

    def get_all_vars(self) -> np.ndarray:
        """Full variable array (for debugging)."""
        n = self.lib.opal_bridge_get_n_vars()
        indices = self._c_indices(list(range(n)))
        out = np.zeros(n)
        self.lib.opal_bridge_get_vars(n, indices, out.ctypes.data_as(self._DP))
        return out
