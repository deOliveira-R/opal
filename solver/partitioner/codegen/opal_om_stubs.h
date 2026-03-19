/*
 * opal_om_stubs.h — Minimal type stubs for compiling OM-generated C code
 * without linking against the full OpenModelica runtime.
 *
 * OM's translateModel generates _functions.c with pure arithmetic
 * implementations of all Modelica functions. The only external dependencies
 * are types (modelica_real, threadData_t) and error handling (throwStreamPrint).
 *
 * This header stubs those out so we can compile _functions.c into a standalone
 * shared library with zero OM runtime dependency.
 */

#ifndef OPAL_OM_STUBS_H
#define OPAL_OM_STUBS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>

/* ── OM numeric types ── */
typedef double modelica_real;
typedef long modelica_integer;
typedef int modelica_boolean;
typedef const char* modelica_string;
typedef void* modelica_metatype;
typedef void* modelica_fnptr;

/* ── Thread context (never dereferenced for pure-Modelica functions) ── */
typedef struct {
    int lastEquationSolved;
} threadData_t;

/* ── Error handling ── */
static void throwStreamPrint(threadData_t *td, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "OPAL codegen error: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    abort();
}

/* ── DLL/visibility macros ── */
#define DLLDirection
#define DLLImport
#define DLLExport

/* ── Label macros ── */
#define OMC_LABEL_UNUSED

/* ── Boxed value macros (we never call boxptr_ functions from OPAL) ── */
static inline modelica_real mmc_unbox_real(modelica_metatype x) { (void)x; return 0.0; }
static inline modelica_metatype mmc_mk_rcon(modelica_real x) { (void)x; return NULL; }
static inline modelica_metatype mmc_mk_icon(modelica_integer x) { (void)x; return NULL; }

#define MMC_REFSTRUCTLIT(x) NULL
/* MMC_DEFSTRUCTLIT: OM uses this for boxed-value struct literals.
   Usage pattern: static const MMC_DEFSTRUCTLIT(name,sz,tag) {ptr, 0}};
   We define a compatible struct and open the initializer so the trailing
   {ptr, 0}}; completes the C initializer correctly. */
#define MMC_DEFSTRUCTLIT(name, sz, tag) \
    struct { long header; const void *data[sz]; } name = {0,

#endif /* OPAL_OM_STUBS_H */
