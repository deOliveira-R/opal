Before writing or modifying physics code (Modelica .mo files, solver numerics, bridge pipeline), review these checklists. Apply them DURING implementation, not after.

## AI Failure Modes (check every term you write)

| # | Mode | What to check |
|---|------|--------------|
| FM1 | Sign flip | Hand-calc the expected sign at a reference state. `(a-b)` vs `(b-a)` |
| FM2 | Variable swap | Use an asymmetric state where `rho_l != rho_v`, `h_sat_l != h_sat_v` |
| FM3 | Missing negation | Test both evaporation AND condensation regimes |
| FM4 | Factor error | Verify exact magnitude, not just sign. Missing `2*`, area, volume |
| FM5 | Index error | Use non-uniform profile where `face[i]` vs `face[i+1]` gives different answers |
| FM6 | Convention drift | Trace sign convention from definition site to every usage site |
| FM7 | Regime overapplication | If a term should be active only in one regime, verify it is ZERO in the other |
| FM8 | Feedback loop | Trace dependency chain. If output feeds back to input with gain > 1, it's unstable |
| FM9 | Boundary/edge value | Test AT exact regime boundaries (h = h_f, alpha = 0, alpha = 1), not just near them |
| FM10 | Silent pipeline corruption | Verify the parameter/variable reaches the solver with the correct value |

## Pipeline Integrity (after any Modelica change)

- **P1 Parameter arrival:** Set parameter in .mo, verify bridge reads correct value (not 0/None from OM CSE)
- **P2 Codegen completeness:** Bridge .so compiles and loads without undefined symbols
- **P3 Variable completeness:** All solver-required variables exist at ALL mesh faces/cells. Check conservation identities at boundaries (mdot_v + mdot_l = mdot)
- **P4 Activation verification:** Run with feature ON and OFF — results must DIFFER. Identical results means the feature isn't reaching the solver
- **P5 Regime boundaries:** Property functions evaluated AT exact boundaries (h = h_f, h = h_g) give physically reasonable values

## Scheme Stability (when adding numerical corrections)

- **S1 Feedback loops:** For any new term that modifies a state variable, trace the dependency chain back to inputs. Compute loop gain. If > 1, demand rate limiter or self-limiting formulation
- **S2 Coupled corrections:** If scheme applies sequential corrections (Schur, predictor-corrector), verify the combined correction DAMPS rather than AMPLIFIES. Test at regime transitions
- **S3 Iteration convergence:** For iterative schemes, verify residual decreases each iteration — especially at property regime boundaries where drho_dp can jump 2400x
- **S4 Rate limiting:** Enhanced physics (100x H_i) needs self-limiting behavior. Verify enhancement decreases as state approaches equilibrium

## After Implementation

1. Write L0 tests for every new term (sign + magnitude at a hand-calculated reference state)
2. Spawn QA agent for full review
3. Run `/verify-solver` to confirm no regressions
