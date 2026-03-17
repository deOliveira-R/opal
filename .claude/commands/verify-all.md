Run ALL OPAL verification tests in sequence: IAPWS-IF97, SimpleFluid, and solver.

```bash
cd /Users/rodrigo/git/OPAL && echo "=== IAPWS-IF97 ===" && external/venv/bin/python library/Media/tests/verify_if97.py && echo "" && echo "=== SimpleFluid ===" && external/venv/bin/python library/Media/tests/verify_simple_fluid.py && echo "" && echo "=== Solver ===" && cd solver && ../external/venv/bin/python -m pytest tests/ -v
```
