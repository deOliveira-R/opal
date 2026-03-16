# OpenModelica Extraction: Anticipated Failure Modes

Ordered by likelihood and fix difficulty.

## FM1: For-loop unrolling (HIGH likelihood, Easy fix)

OM's backend works on scalar equations. Array loops unroll during flattening → hundreds of scalar equations, not loops.

**Fix:** Pattern-reconstruction pass in our extraction pipeline (Python, ~200-300 lines). Scan variable names for array index patterns, detect identical structures with shifted indices, reconstruct mesh topology. Our code, not a compiler patch.

## FM2: BLT scattering of vessel equations (MEDIUM-HIGH, Medium fix)

BLT may interleave vessel and pipe equations (shared pressure BCs at nozzles) instead of keeping the vessel contiguous.

**Fix A (preferred):** Extract before BLT (`flat` or `optimiser` level). Our pipeline does its own partitioning via variable name prefixes. We throw away OM's BLT work — we were going to re-sort anyway.

**Fix B (upstream):** Contribute `sourceComponent` attribute to OM's XML export. Compiler already carries this info internally (`DAE.Element` source info), just not serialized. ~50-100 lines in XML emitter. Most valuable single upstream contribution for OPAL.

## FM3: Opaque external function calls for fluid properties (CERTAIN, Already planned)

`Modelica.Media.Water.StandardWater` uses external C for IAPWS-IF97 → opaque calls in extracted XML.

**Fix:** Our own water property package in pure Modelica. IAPWS-IF97 is public. Pure-Modelica flattens to algebraic equations visible in extraction. Also provides analytical derivatives for pressure linearization.

## FM4: Equation count scaling (MEDIUM, Potentially hard)

3D vessel: 2500 cells × ~15 eq/cell = 37,500 + rest of plant. OM's Pantelides is O(n²) worst case.

**Fix Path 1:** Extract before index reduction. We know equation index structure a priori.

**Fix Path 2:** Vessel internals in C++ (Test 5 fallback). OM only sees external interface.

## FM5: Variable name mangling (LOW, Trivial)

Array indices lost during flattening.

**Fix:** Unlikely — OM preserves qualified names for plotting. If it happens, FM1 reconstruction handles it.
