"""
OPAL SymPy Library — CAS infrastructure for equation derivation and code generation.

This library exists because Claude cannot do symbolic math reliably.
Every equation in OPAL is derived here, verified here, and generated from here.

Usage:
    from opal_sympy import *
    # Gives you all symbols, stencil helpers, conservation builders, codegen, verify
"""

from opal_sympy.symbols import *
from opal_sympy.thermo import *
from opal_sympy.stencil import *
from opal_sympy.conservation import *
from opal_sympy.codegen import to_modelica, to_c, to_numpy
from opal_sympy.verify import random_thermo_state, check_conservation
