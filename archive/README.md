# Archive — C++ Prototype Solver

This directory contains the C++ solver implementation that was built during Phases 1–3b
as a prototype to verify the numerical methods.

**DO NOT EDIT THESE FILES.** The production architecture is:
- Physics in Modelica (.mo files in `library/`)
- Extraction via OpenModelica
- Solver numerics in Python (`solver/partitioner/`)

The C++ code served its purpose: it proved the semi-implicit method works for two-phase
flow and provided 330 tests. The compiled .so files remain in `solver/two_phase/` and
`solver/single_phase/` for test compatibility and property evaluation.

## Contents

- `cpp_prototype/two_phase/` — Two-phase solver (HEM, 5-eq, closures, friction, MUSCL, etc.)
- `cpp_prototype/single_phase/` — Phase 1 single-phase solver

## If you need to rebuild

```bash
cd archive/cpp_prototype/two_phase
mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
make
cp opal_two_phase*.so ../../../../solver/two_phase/
```

## Why archived

The OPAL architecture requires ALL physics to live in Modelica. The C++ solver
reimplements physics (closures, friction, critical flow) that should come from
Modelica extraction. See `docs/architecture.md` "Cardinal Rule" section.
