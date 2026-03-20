# OPAL Developer Guide: Adding New Modelica Components

## Architecture Overview

OPAL uses a three-layer architecture:

```
Layer 1: Modelica Component Library  →  ALL physics
Layer 2: OM Extraction Pipeline      →  equation structure + compiled C code
Layer 3: Python Semi-Implicit Solver →  ONLY numerical methods
```

The Cardinal Rule: **ALL physics lives in Modelica.** The solver provides only
numerical methods (Thomas algorithm, semi-implicit pressure solve, operator
splitting). If a physics change is needed, edit the `.mo` files — never the solver.

## How to Add a New Component

### Step 1: Define the Modelica Model

Create a new `.mo` file in the appropriate subdirectory of `library/`.

**File structure:**
```modelica
within library.Pipes;  // or library.Pumps, library.HeatExchangers, etc.
model MyComponent "Brief description"
  extends PartialPipe1D;  // if it's a pipe variant
  // OR: define own parameters, connectors, variables

  // Parameters (tunable from test model)
  parameter Real my_param = 1.0 "Description [units]";

  // State variables (OM computes der() of these)
  Real my_state[N](each start = 0, each fixed = true) "Description [units]";

  // Algebraic variables (computed from state every timestep)
  Real my_closure[N] "Description [units]";

equation
  // Physics equations go here
  for i in 1:N loop
    my_closure[i] = some_function(p[i], my_state[i]);
    der(my_state[i]) = ...;
  end for;

end MyComponent;
```

**Key rules:**
- Use `replaceable package Medium constrainedby library.Media.PartialMedium`
  for fluid properties (enables SimpleFluid ↔ Water swapping)
- All parameters should have documented units and reasonable defaults
- Variables that the solver needs must be `Real` (not `protected`) so OM
  exports them as named variables
- Use `max()`, `min()`, `noEvent()` for numerical guards — avoid events

### Step 2: Use the Medium Interface

The `PartialMedium` interface provides these functions:

| Function | Returns | Units |
|----------|---------|-------|
| `Medium.rho_ph(p, h)` | Density | kg/m³ |
| `Medium.T_ph(p, h)` | Temperature | K |
| `Medium.drho_dp_h(p, h)` | ∂ρ/∂p at const h | kg/(m³·Pa) |
| `Medium.drho_dh_p(p, h)` | ∂ρ/∂h at const p | kg/(m³·J/kg) |
| `Medium.T_sat(p)` | Saturation temperature | K |
| `Medium.h_f(p)` | Saturated liquid enthalpy | J/kg |
| `Medium.h_g(p)` | Saturated vapour enthalpy | J/kg |
| `Medium.h_fg(p)` | Latent heat | J/kg |
| `Medium.rho_f(p)` | Saturated liquid density | kg/m³ |
| `Medium.rho_g(p)` | Saturated vapour density | kg/m³ |
| `Medium.cp_f(p)` | Saturated liquid cp | J/(kg·K) |
| `Medium.k_f(p)` | Saturated liquid conductivity | W/(m·K) |
| `Medium.mu_f(p)` | Saturated liquid viscosity | Pa·s |
| `Medium.sigma(p)` | Surface tension | N/m |

**Available implementations:**
- `SimpleFluid` — linear properties, analytically verifiable (use for L0-L1 tests)
- `Water` — IAPWS-IF97 Regions 1, 2, 4 (production)

### Step 3: Handle the C_d_eff Pattern

If your component interacts with PartialPipe1D's critical flow, note that
`C_d_eff` is a variable (not a parameter) that must be set at the system level:

```modelica
equation
  pipe.C_d_eff = pipe.C_d;          // constant discharge coefficient
  // OR:
  pipe.C_d_eff = my_break.C_d;     // time-varying from RampedBreak
```

### Step 4: Create a Test Model

Create a test model in `feasibility/models/` that instantiates your component
with boundary conditions:

```modelica
model MyComponentTest
  library.Boundary.ClosedEnd closed_end;
  library.Pipes.MyComponent pipe(
    redeclare package Medium = library.Media.SimpleFluid,
    N=5, L=5.0, D=0.1);
  library.Boundary.PressureSource outlet(p_set=101325.0, h_set=800e3);
equation
  connect(closed_end.port, pipe.port_a);
  connect(pipe.port_b, outlet.port);
  pipe.C_d_eff = pipe.C_d;
end MyComponentTest;
```

### Step 5: Extract and Verify

**XML extraction (equation structure):**
```python
from partitioner.codegen.translate_model import translate_and_extract
so, info, xml = translate_and_extract('MyComponentTest')
```

**Verify variable availability:**
```python
from partitioner.codegen.info_parser import parse_info_json
info = parse_info_json(info_path)
# Check your variables exist
for name in sorted(info.all_vars.keys()):
    if 'my_closure' in name:
        print(f'{name}: idx={info.all_vars[name].index}')
```

**Bridge solver usage:**
```python
from partitioner.codegen.equation_bridge import OMEquationBridge
bridge = OMEquationBridge(so_path, info)
bridge.set_state(p, alpha=alpha, h_l=h_l, h_v=h_v, mdot=mdot)
bridge.evaluate()
my_vals = bridge.get('my_closure')  # Read computed values
```

### Step 6: Write Tests

Follow the QA methodology in `solver/tests/QA_AI_CODE_METHODOLOGY.md`:

**Level 0 (term verification):** Test each equation term against hand calculations
using SimpleFluid. Both sign AND magnitude must be verified.

**Level 1 (integration):** Run the full solver with your component and verify
conservation, stability, and physical behavior.

**Level 2 (validation):** Compare against experimental data or reference codes.

**Test file naming:** `solver/tests/test_<component_name>.py`

## Known Pitfalls

### OM Variable Elimination
OpenModelica may eliminate (inline) some variables during compilation. The bridge
handles this with -1 sentinel indices and nearest-neighbor gap filling. If your
variable is eliminated, it means OM substituted its defining expression directly
into the equations that use it. The value is still computed — just not stored
as a named variable.

**Workaround:** If a variable MUST be available in the bridge, ensure it appears
in at least two separate equations (OM won't inline variables used in multiple
places).

### OM Parameter Evaluation at Compile Time
OM evaluates `max(param_a, param_b)` at compile time when both are parameters.
This means runtime `set_params` CANNOT change the result. If you need runtime-
tunable parameter combinations, use a variable instead:
```modelica
// BAD: d_eff = max(d_b, d_b_min) — frozen at compile time
// GOOD: d_eff = d_b (set d_b >= d_b_min in the test model)
```

### Time-Dependent Variables
The bridge supports `time` through `opal_time`. OM generates
`data->localData[0]->timeValue` which the bridge_codegen rewrites to `opal_time`.
Call `bridge.set_time(t)` before `bridge.evaluate()` for models with
time-varying BCs (e.g., RampedBreak).

### Non-Conservative vs Conservative Forms
For void fraction transport, the Modelica model uses the non-conservative form
`rho_v * V * der(alpha)` because OM cannot efficiently differentiate through
the full EOS chain rule. The solver handles the conservative update
(`alpha*rho_v` product) in its explicit time-stepping. This split is intentional
and documented in `Pipe1D_DriftFlux.mo`.

### Semi-Implicit Scheme Requirements
The pressure equation needs:
- `drho_dp` evaluated at mixture enthalpy h_mix (for thermal compressibility)
- Friction treated semi-implicitly (implicit resistance `beta_eff = beta/(1+sigma)`)
- These are numerical method choices in the solver, not physics in Modelica

## File Checklist for a New Component

- [ ] `library/<Category>/<ComponentName>.mo` — the Modelica model
- [ ] `feasibility/models/<TestName>.mo` — test model with BCs
- [ ] `solver/tests/test_<component>.py` — L0 + L1 tests
- [ ] Verify extraction: `translate_and_extract()` succeeds
- [ ] Verify bridge: all needed variables available via `bridge.get()`
- [ ] Run full test suite: `pytest solver/tests/ -k "not slow"` — all pass
