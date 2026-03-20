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
        # Use a generic pattern-based approach — works for HEM and DriftFlux
        self._var_groups = {}
        self._build_var_group('p', self.N)
        self._build_var_group('rho_face', self.N + 1)
        self._build_var_group('drho_dp', self.N)
        self._build_var_group('drho_dh', self.N)
        self._build_var_group('T_cell', self.N)

        # HEM-specific
        self._build_var_group('h', self.N)
        self._build_var_group('h_face', self.N + 1)

        # DriftFlux-specific (present if model uses Pipe1D_DriftFlux)
        for name in ['alpha', 'h_l', 'h_v', 'h_mix', 'rho_l', 'rho_v', 'rho_m',
                      'Gamma', 'q_i_l', 'q_i_v', 'V_gj', 'a_i', 'alpha_eff',
                      'T_l', 'T_sat_cell', 'h_sat_l', 'h_sat_v']:
            self._build_var_group(name, self.N)
        self._build_var_group('Phi2', self.N + 1)

        # mdot: states + dummy states, sorted by Modelica index
        mdot_entries = sorted(
            [(int(n.split('[')[1].rstrip(']')), vi.index)
             for n, vi in info.all_vars.items()
             if f'{self.prefix}.mdot[' in n and vi.kind in ('state', 'dummy state')],
            key=lambda x: x[0]
        )
        self._var_groups['mdot'] = self._c_indices([idx for _, idx in mdot_entries])

        # Backward compatibility aliases
        self._p_idx = self._var_groups.get('p')
        self._h_idx = self._var_groups.get('h')
        self._mdot_idx = self._var_groups.get('mdot')

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

        Uses the manifest JSON emitted by bridge_codegen.py alongside the .so.
        Falls back to ctypes probing with known naming patterns.
        """
        D = self._D

        # Try loading the manifest (emitted by bridge_codegen.py)
        manifest_path = Path(str(self.lib._name)).with_suffix('.json')
        if manifest_path.exists():
            import json
            manifest = json.loads(manifest_path.read_text())
            for name in manifest.get('media_wrappers', []):
                if name.endswith(f'_{suffix}'):
                    try:
                        fn = getattr(self.lib, name)
                        fn.restype = D
                        fn.argtypes = argtypes
                        return fn
                    except AttributeError:
                        continue

        # Fallback: try common patterns via ctypes probing
        candidates = [
            f'opal_bridge_{suffix}',
            f'opal_bridge_{self.info.model_name}_{self.prefix}_Medium_{suffix}',
        ]
        for name in candidates:
            try:
                fn = getattr(self.lib, name)
                fn.restype = D
                fn.argtypes = argtypes
                return fn
            except AttributeError:
                continue

        return None

    def _build_var_group(self, short_name: str, count: int):
        """Build a ctypes index array for a variable group if it exists in the model."""
        full_pattern = f'{self.prefix}.{short_name}'
        indices = self.info.vars_by_pattern(full_pattern, count)
        if indices:
            self._var_groups[short_name] = self._c_indices(indices)

    def _c_indices(self, indices: list[int]) -> ctypes.Array:
        """Convert Python index list to a ctypes int array (allocated once)."""
        return (ctypes.c_int * len(indices))(*indices)

    # ── State set/get (name-aware, index-driven) ──

    def set_state(self, p: np.ndarray, h: np.ndarray = None, mdot: np.ndarray = None,
                  alpha: np.ndarray = None, h_l: np.ndarray = None, h_v: np.ndarray = None):
        """Write current state into the bridge flat arrays.

        HEM: set_state(p, h=h, mdot=mdot)
        5-eq: set_state(p, mdot=mdot, alpha=alpha, h_l=h_l, h_v=h_v)
        """
        self._set_group('p', p)
        if mdot is not None:
            self._set_group('mdot', mdot)
        if h is not None:
            self._set_group('h', h)
        if alpha is not None:
            self._set_group('alpha', alpha)
        if h_l is not None:
            self._set_group('h_l', h_l)
        if h_v is not None:
            self._set_group('h_v', h_v)

    def _set_group(self, var_name: str, values: np.ndarray):
        """Write a numpy array to a variable group by name."""
        if var_name in self._var_groups:
            self.lib.opal_bridge_set_vars(
                len(values), self._var_groups[var_name],
                values.ctypes.data_as(self._DP))

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

    def get(self, var_name: str) -> np.ndarray:
        """Generic getter: read a variable group by short name (e.g., 'rho_face')."""
        if var_name not in self._var_groups:
            raise KeyError(f"Variable group '{var_name}' not found. "
                           f"Available: {sorted(self._var_groups.keys())}")
        c_idx = self._var_groups[var_name]
        n = len(c_idx)
        return self._get_array(c_idx, n)

    def has(self, var_name: str) -> bool:
        """Check if a variable group exists in this model."""
        return var_name in self._var_groups

    # Named getters for backward compatibility
    def get_rho_face(self) -> np.ndarray:
        return self.get('rho_face')

    def get_drho_dp(self) -> np.ndarray:
        return self.get('drho_dp')

    def get_drho_dh(self) -> np.ndarray:
        return self.get('drho_dh')

    def get_h_face(self) -> np.ndarray:
        return self.get('h_face')

    def get_T_cell(self) -> np.ndarray:
        return self.get('T_cell')

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
