# OPAL — Open Platform for Analytical Thermalhydraulics

An open thermal-hydraulic simulation platform for nuclear power plant analysis. Modelica front end, OpenModelica compiler, custom solver backend. No export controls. Target users: everyone who can't access RELAP5/TRACE.

## Architecture

**Option 4** — extract the equation system from OpenModelica, route subsystems to purpose-built solvers:

```
Modelica model
      │
      ▼
OpenModelica compiler
  (flattening, index reduction, BLT decomposition)
      │
      ▼
Equation extraction  (dumpXMLDAE / translateModelXML)
  → equations, variable list, incidence matrix, sparsity
      │
      ├─ two-phase primary ──→ semi-implicit staggered-mesh solver
      ├─ BOP / controls ─────→ IDA/DASSL
      └─ coupling ───────────→ partitioner
```

This buys purpose-built two-phase numerics where generic Modelica solvers fail, while keeping the Modelica ecosystem for non-nuclear components and real-time capability.

## Repository Layout

```
opal/
├── feasibility/       # Phase 0: extraction feasibility tests (complete)
│   ├── models/        # Modelica test models
│   ├── results/       # Generated XML, reports (gitignored)
│   ├── extraction_utils.py
│   ├── test_extraction.py               # Test 1: basic extraction
│   ├── test2_information_completeness.py
│   ├── test3_fluid_compat.py
│   ├── test4_scale.py
│   └── test5_3d_array.py
├── solver/            # Custom solver backend (C++)
├── library/           # OPAL Modelica component library
├── diagnostics/       # AI failure diagnosis
├── docs/              # Architecture, physics, design docs
└── external/
    ├── OpenModelica/  # OM compiler (git submodule)
    └── venv/          # Python environment (gitignored)
```

## Current Status

**Phase 0 complete.** All 5 extraction feasibility tests passed (2026-03-16):

| Test | Result | Key finding |
|------|--------|-------------|
| Basic extraction (6 OM APIs) | PASS | Full DAE, BLT, incidence matrix accessible |
| Information completeness | PASS | Events, stream connectors, 100% name traceability |
| MSL/Fluid compatibility | PASS | IF97 calls are opaque → own media package planned |
| Equation scaling (N=1–20 cells) | PASS | Near-linear time scaling |
| 3D array extraction (vessel mesh) | PASS | Array indices intact; for-loops unrolled (recoverable) |

**Next:** Phase 1 — single-phase solver coupling (extraction → solver pipeline).

## Build Path

| Phase | Milestone |
|-------|-----------|
| 1 | Single-phase solver coupling — proves extraction→solver pipeline |
| 2 | Two-phase solver plugin + oracle benchmarking |
| 3 | Multi-domain plant demo + real-time benchmark |
| 4 | Component library + point kinetics + own IAPWS-IF97 |
| 4.5 | 3D vessel component (Approach B monolithic) |
| 5 | 3D spatial kinetics (few-group neutron diffusion) |
| 6 | Real-time mode (fixed-step, bounded iterations) |

## Getting Started

### Prerequisites

- macOS (Apple Silicon) or Linux
- Homebrew (macOS) or system package manager
- CMake ≥ 3.14, Clang, Java

### Build OpenModelica

```bash
git submodule update --init --recursive external/OpenModelica
cd external/OpenModelica
cmake -S . -B build_cmake \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DOM_OMC_ENABLE_FORTRAN=OFF \
  -DOM_OMC_ENABLE_OPTIMIZATION=OFF \
  -DOM_OMC_ENABLE_MOO=OFF \
  -DOM_ENABLE_GUI_CLIENTS=OFF \
  -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build_cmake -j$(nproc)
```

### Python environment

```bash
python3 -m venv external/venv
external/venv/bin/pip install OMPython
```

### Run feasibility tests

```bash
cd feasibility
../external/venv/bin/python test_extraction.py
../external/venv/bin/python test2_information_completeness.py
../external/venv/bin/python test3_fluid_compat.py   # loads MSL (~60s)
../external/venv/bin/python test4_scale.py
../external/venv/bin/python test5_3d_array.py
```

## Design Philosophy

- **Open, no export controls** — accessible to the global nuclear engineering community
- **Modelica front end** — standard language, extensible by anyone
- **Custom numerics where it matters** — two-phase flow requires purpose-built solvers; everything else uses proven general solvers
- **Real-time capable** — target within 5× real time by Phase 3, hard real-time by Phase 6
- **Benchmarked** — every solver verified against analytical solutions and oracle code

## License

TBD.
