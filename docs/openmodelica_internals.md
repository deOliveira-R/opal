# OpenModelica Internals — OPAL's Documentation

This document captures our understanding of OpenModelica's internals relevant to OPAL's equation extraction use case. Official OM documentation is often stale or incomplete; this is what we've verified by reading the source.

## Status: To Be Populated

This document will be filled during feasibility testing (Phase 0) and extended as we work deeper with OM's compiler pipeline.

## Planned Sections

### Compiler Pipeline
- Parser → AST → SCode → DAE → Backend DAE → SimCode → C code
- Where each transformation happens in the source tree
- What information is preserved/lost at each stage

### XML Export Internals
- `dumpXMLDAE` implementation: which source files, what it serializes
- `translateModelXML` (CasADi export): format differences, what's included
- `instantiateModel`: what "flat Modelica" actually means in practice
- Gaps: what's available internally but not exported (source component info, sparsity)

### Flattening
- How inheritance and connections are resolved
- `stream` connector expansion — the actual algorithm, not the spec
- Array equation unrolling — where loops become scalar equations
- How `Modelica.Media` property models flatten (or don't)

### Index Reduction (Pantelides)
- Implementation location and algorithm
- Scaling behavior with equation count
- Which equation patterns cause worst-case behavior
- Whether it can be skipped (for equations we know are index-1)

### BLT Decomposition
- How equations are sorted into strongly connected components
- Whether source component information survives into BLT blocks
- How tearing variables are selected

### Potential Upstream Contributions
- `sourceComponent` attribute in XML export (FM2 fix)
- Sparsity pattern export
- Better array structure preservation in XML
- Documentation improvements
