Run all OPAL feasibility tests in sequence from the feasibility/ directory.

```bash
cd /Users/rodrigo/git/OPAL/feasibility
echo "=== Test 1: Basic Extraction ===" && ../external/venv/bin/python test_extraction.py
echo "=== Test 2: Information Completeness ===" && ../external/venv/bin/python test2_information_completeness.py
echo "=== Test 3: MSL/Fluid Compatibility ===" && ../external/venv/bin/python test3_fluid_compat.py
echo "=== Test 4: Equation Scaling ===" && ../external/venv/bin/python test4_scale.py
echo "=== Test 5: 3D Array Extraction ===" && ../external/venv/bin/python test5_3d_array.py
```

Note: Test 3 loads MSL (~60s). Test 1 must complete before Test 2 (Part A reads Test 1's XML output).
