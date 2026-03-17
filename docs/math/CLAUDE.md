# OPAL — Claude Code Instructions

Read OPAL_CLAUDE.md for full project context. This file governs how you work.

---

## Rule 1: Know When You Need SymPy

You can do straightforward algebra reliably — single-step substitutions,
well-known physics formulas, simple rearrangements. You have correctly
derived a full two-phase semi-implicit solver, SimpleFluid properties,
and Gibbs derivative chain rules without SymPy assistance.

**Where you reliably fail:**
- **Transcribing large coefficient tables** (IAPWS-IF97 has 34+43+20+34
  coefficients — you got one wrong). SymPy codegen eliminates hand-copying.
- **Multi-step chain rules with 4+ intermediates** where sign/factor errors
  compound silently. SymPy `diff()` is zero-risk.
- **Index bookkeeping in stencils** on structured meshes (off-by-one in
  3D indexing). Use `opal_sympy.stencil` helpers.
- **Counting and delimiter balancing.** Let Python count things.

**Use SymPy when:**
- Deriving through 4+ intermediate steps where a sign or factor error
  would be hard to catch by inspection
- Generating code from coefficient tables (IAPWS, cross-sections)
- Building discrete stencils on 2D/3D meshes
- You're not confident the result is correct and want a second opinion

**Write equations directly when:**
- The equation is a well-known physics formula (R = f*dx/(2*D*A²*ρ))
- It's a single substitution or rearrangement
- You're stating a conservation law or constitutive relation
- The result can be verified by a quick numerical spot-check

**Always verify numerically** regardless of how you derived it. The
`opal_sympy.verify` spot-checks are cheap and catch everything. This is
non-negotiable — every new equation gets tested at random valid states.

---

## Rule 2: The Derivation Workflow

Two paths depending on complexity:

### Path A: Direct (simple equations)

For well-known formulas and single-step algebra:
1. Write the equation directly in Modelica or C++
2. **Always** add a numerical spot-check (random states, compare to known solution or FD)
3. If it passes, done

### Path B: Full derivation (complex equations)

For multi-step derivations, chain rules, coefficient tables, or anything
where you're not confident:

```
Physics knowledge (you)
  → SymPy derivation script (you write, SymPy executes)
    → Symbolic verification (identities, conservation, limiting cases)
      → Code generation (SymPy codegen → Modelica/C++)
        → Numerical verification (NumPy spot-checks at random states)
```

Create a derivation script in `derivations/`:

1. **Import the OPAL SymPy library** (`from opal_sympy import *`)
2. **Define the physics symbolically** — conservation equations, constitutive relations
3. **Let SymPy do the algebra** — `diff()`, `simplify()`, `expand()`, `solve()`
4. **Verify symbolically** — conservation, limiting cases, symmetry
5. **Generate code** — `to_modelica()`, `to_c()`, `to_numpy()`
6. **Numerical verification** — random states, assert PASS/FAIL

### When to use Path B:

- Coefficient tables (IAPWS, cross-sections) — **always** use codegen
- Chain rules through 4+ intermediates — use `diff()`
- New discretization stencils (2D, 3D) — use stencil helpers
- Any equation where you hesitate about a sign or factor

### When derivation scripts exist, they are source of truth

For equations that went through Path B, the derivation script is authoritative.
If there's a discrepancy between the script and the Modelica/C++ code, the
script wins. Generated equation blocks should have a traceability comment:
```modelica
// Derived in: derivations/pressure_equation_linearization.py
```

Simple Path A equations don't need derivation scripts — the code itself and
its numerical test are sufficient documentation.

---

## Rule 3: Parenthesis and Syntax Discipline

You have a known, persistent weakness with delimiter balancing. Countermeasures:

1. **Never write expressions deeper than 3 levels of nesting.** Break into
   intermediate variables. `a = f(g(h(x, y), k(z)))` → NO. Use temps.

2. **After writing any code, run it immediately.** Do not write 200 lines and
   then run. Write a function, run it, confirm it parses, move on.

3. **For Modelica code:** always run through OpenModelica's parser as a
   syntax check before considering it done:
   ```python
   omc.sendExpression('loadFile("mymodel.mo")')
   omc.sendExpression('checkModel(MyModel)')
   ```

4. **For C++ code:** compile after every function. Don't accumulate.

5. **For Python code:** run after every logical block. Use the REPL.

---

## Rule 4: How to Use opal_sympy

The library lives at `opal_sympy/`. It provides:

### Predefined symbols
```python
from opal_sympy.symbols import *
# Gives you: P, T, alpha, rho_l, rho_v, h_l, h_v, v_l, v_v,
#            dt, dx, dA, V_cell, and all standard TH symbols
# All properly configured (real, positive where appropriate)
```

### Thermodynamic derivative symbols
```python
from opal_sympy.thermo import drho_dP_h, drho_dh_P, dT_dP_h, ...
# Named following OPAL convention: d{property}_d{variable}_{held_constant}
```

### Stencil helpers
```python
from opal_sympy.stencil import east, west, north, south, top, bottom, center
# For building finite difference/volume discretizations on structured meshes
# east(P) → P[i+1,j,k], center(P) → P[i,j,k], etc.
```

### Conservation law builders
```python
from opal_sympy.conservation import mass_equation, energy_equation, momentum_equation
# Returns symbolic expressions for the standard two-fluid conservation equations
# You then discretize/linearize these — SymPy does the algebra
```

### Code generation
```python
from opal_sympy.codegen import to_modelica, to_c, to_numpy
# to_modelica(expr, varname) → "varname = <modelica expression>;"
# to_c(expr, funcname) → C function string
# to_numpy(expr) → lambdified NumPy function for testing
```

### Numerical verification
```python
from opal_sympy.verify import random_thermo_state, check_conservation
# random_thermo_state() → dict of {symbol: value} at a valid thermodynamic state
# check_conservation(residual_expr, n_samples=1000) → runs spot checks, returns PASS/FAIL
```

### Adding to the library
When you need new shared functionality, add it to the appropriate submodule.
Every addition must have:
- A docstring explaining what it does and why
- At least one usage example in the docstring
- A test in `opal_sympy/tests/`

---

## Rule 5: Working With Equations in This Project

### When writing Modelica component equations:
1. Draft the physics in a `derivations/` script using SymPy
2. Verify symbolically and numerically in that script
3. Use `to_modelica()` to generate the equation block
4. Paste into the .mo file with the generation comment
5. Run through OpenModelica parser to confirm syntax
6. Run the component's test case to confirm physics

### When writing C++ solver routines:
1. Derive the discrete equations in `derivations/` using SymPy
2. Verify conservation of the discrete scheme numerically
3. Use `to_c()` to generate evaluation functions
4. Write the solver loop by hand (control flow is your strength)
5. Compile and test incrementally

### When debugging equation mismatches:
1. Go back to the derivation script
2. Print intermediate SymPy expressions
3. Compare against what's in the Modelica/C++ file
4. If they differ, regenerate from the derivation script — do NOT hand-edit the generated code

### When you need a new equation not in the library:
1. State what conservation law or constitutive relation you need
2. Find it in a reference (cite the reference in the derivation script)
3. Encode it in SymPy
4. Run the full pipeline

---

## Rule 6: What You Do Well (Lean Into These)

- Knowing which equations to write (physics judgment)
- Structuring code (architecture, interfaces, data flow)
- Reading and understanding existing code (OpenModelica internals)
- Writing test infrastructure
- Parsing and transforming structured data (XML extraction pipeline)
- Explaining what the math means after SymPy computes it
- Writing solver control flow (time stepping loops, convergence checks, partitioning logic)

---

## Rule 7: What You Do Badly (Compensate For These)

- Transcribing large coefficient tables (→ SymPy codegen, never hand-copy)
- Parenthesis balancing (→ write flat, run often)
- Index bookkeeping in 2D/3D stencils (→ opal_sympy.stencil helpers)
- Counting (→ let Python count things)
- Multi-step chain rules with 4+ intermediates (→ SymPy diff)
- Remembering earlier context in long sessions (→ derivation scripts are the memory, not chat history)

---

## Directory Structure for Math Infrastructure

```
opal/
├── CLAUDE.md                  # This file
├── OPAL_CLAUDE.md             # Project context
├── opal_sympy/                # CAS utility library
│   ├── __init__.py
│   ├── symbols.py             # Predefined TH symbols
│   ├── thermo.py              # Thermodynamic derivative symbols and relations
│   ├── stencil.py             # Structured mesh stencil helpers
│   ├── conservation.py        # Standard conservation equation builders
│   ├── codegen.py             # SymPy → Modelica/C++/NumPy emitters
│   ├── verify.py              # Numerical verification utilities
│   └── tests/
│       ├── test_symbols.py
│       ├── test_stencil.py
│       ├── test_codegen.py
│       └── test_verify.py
├── derivations/               # Equation derivation scripts (SOURCE OF TRUTH)
│   ├── README.md
│   ├── pressure_linearization.py
│   ├── donor_cell_advection_1d.py
│   ├── semi_implicit_momentum.py
│   ├── diffusion_operator_stencil.py
│   └── ...
├── feasibility/               # Extraction tests (per OPAL_CLAUDE.md)
├── solver/
├── library/
├── tests/
└── docs/
```
