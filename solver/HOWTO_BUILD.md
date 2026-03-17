# Phase 1 — Build and Run

## Set up Python environment (first time only)

```bash
# Requires brew Python 3.13
brew install python@3.13
/opt/homebrew/bin/python3.13 -m venv external/venv
external/venv/bin/pip install -r external/requirements.txt
```

## Build the C++ extension

```bash
cd solver/single_phase
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cmake --install .   # puts .so in solver/single_phase/
```

## Run tests

```bash
# From OPAL root — PYTHONUNBUFFERED=1 is required (stdout is block-buffered
# by default when not a tty; the process may stay alive in cleanup
# after printing all results, which would appear as a hang).
PYTHONUNBUFFERED=1 external/venv/bin/python solver/tests/test_hagen_poiseuille.py
```

## Run end-to-end driver

```bash
PYTHONUNBUFFERED=1 external/venv/bin/python solver/phase1_driver.py
# Optional flags:
#   --dt 1e-3 --steps 5000 --stride 50
#   [XML_PATH]  (default: feasibility/results/scale_N5_backEnd.xml)
```

## Notes

- pybind11 is in the OPAL venv (`external/venv`) at version 3.0.2
- The extension is named `opal_single_phase`
- On macOS, running Python scripts that import local `.so` files requires
  `PYTHONUNBUFFERED=1` if you need to see output before Python exits cleanly.
