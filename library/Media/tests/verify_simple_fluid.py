"""
SimpleFluid Verification Tests
===============================
Verifies the synthetic test fluid against hand-computed values.

Five test groups:
  Test 1 — Property values at 9 known state points (3 per region)
  Test 2 — Derivative consistency (analytical vs. finite difference)
  Test 3 — Region boundary continuity
  Test 4 — Derivative sign checks
  Test 5 — Extraction transparency (OMPython / dumpXMLDAE)

Usage:
  external/venv/bin/python library/Media/tests/verify_simple_fluid.py
"""

import sys
import pathlib
import re
import math

OPAL_ROOT = pathlib.Path(__file__).resolve().parents[3]

# ==========================================================================
# SimpleFluid constants (must match SimpleFluid.mo exactly)
# ==========================================================================
p_ref   = 10.0e6

T_sat_0 = 400.0;  T_sat_1 = 20.0
h_f_0   = 800.0e3;  h_f_1 = 100.0e3
h_g_0   = 2800.0e3; h_g_1 = 50.0e3
rho_f_0 = 750.0;  rho_f_1 = 20.0
rho_g_0 = 40.0;   rho_g_1 = 5.0
A_L     = 6.25e-5
A_G     = 2.0e-5
cp_L    = 4000.0
cp_G    = 2000.0

# ==========================================================================
# Property functions (Python oracle — direct transcription of SimpleFluid.mo)
# ==========================================================================
def p_hat(p):
    return (p - p_ref) / p_ref

def T_sat(p):
    return T_sat_0 + T_sat_1 * p_hat(p)

def h_f(p):
    return h_f_0 + h_f_1 * p_hat(p)

def h_g(p):
    return h_g_0 + h_g_1 * p_hat(p)

def h_fg(p):
    return h_g(p) - h_f(p)

def rho_f(p):
    return rho_f_0 + rho_f_1 * p_hat(p)

def rho_g(p):
    return rho_g_0 + rho_g_1 * p_hat(p)

def region_ph(p, h):
    if h < h_f(p):
        return 1
    elif h > h_g(p):
        return 2
    else:
        return 4

def rho_ph(p, h):
    reg = region_ph(p, h)
    if reg == 1:
        return rho_f(p) + A_L * (h_f(p) - h)
    elif reg == 2:
        return rho_g(p) - A_G * (h - h_g(p))
    else:
        x  = (h - h_f(p)) / h_fg(p)
        vf = 1.0 / rho_f(p)
        vg = 1.0 / rho_g(p)
        v  = x * vg + (1.0 - x) * vf
        return 1.0 / v

def T_ph(p, h):
    reg = region_ph(p, h)
    if reg == 1:
        return T_sat(p) - (h_f(p) - h) / cp_L
    elif reg == 2:
        return T_sat(p) + (h - h_g(p)) / cp_G
    else:
        return T_sat(p)

def drho_dp_h(p, h):
    reg = region_ph(p, h)
    if reg == 1:
        return (rho_f_1 + A_L * h_f_1) / p_ref
    elif reg == 2:
        return (rho_g_1 + A_G * h_g_1) / p_ref
    else:
        rf   = rho_f(p)
        rg   = rho_g(p)
        hfv  = h_f(p)
        hgv  = h_g(p)
        hfgv = hgv - hfv
        x    = (h - hfv) / hfgv
        vf   = 1.0 / rf
        vg   = 1.0 / rg
        v    = x * vg + (1.0 - x) * vf
        rho2 = 1.0 / (v * v)

        drf_dp  = rho_f_1 / p_ref
        drg_dp  = rho_g_1 / p_ref
        dhf_dp  = h_f_1 / p_ref
        dhg_dp  = h_g_1 / p_ref
        dhfg_dp = dhg_dp - dhf_dp

        dvf_dp = -drf_dp / (rf * rf)
        dvg_dp = -drg_dp / (rg * rg)
        dx_dp  = (-dhf_dp - x * dhfg_dp) / hfgv
        dv_dp  = dx_dp * (vg - vf) + x * dvg_dp + (1.0 - x) * dvf_dp

        return -rho2 * dv_dp

def drho_dh_p(p, h):
    reg = region_ph(p, h)
    if reg == 1:
        return -A_L
    elif reg == 2:
        return -A_G
    else:
        rf   = rho_f(p)
        rg   = rho_g(p)
        hfgv = h_fg(p)
        rho_mix = rho_ph(p, h)
        return -rho_mix**2 * (1.0/rg - 1.0/rf) / hfgv


# ==========================================================================
# TEST 1: Property values at known state points
# ==========================================================================
def test_property_values():
    print("\n=== Test 1: Property Values at Known State Points ===")
    all_pass = True

    # Hand-computed reference values at p = p_ref (p_hat = 0)
    # Region 1: h = 400e3 (subcooled)
    #   rho = 750 + 6.25e-5*(800e3 - 400e3) = 750 + 25 = 775
    #   T   = 400 - (800e3 - 400e3)/4000 = 400 - 100 = 300
    # Region 4: h = 1800e3 (x = (1800e3 - 800e3)/2000e3 = 0.5)
    #   v = 0.5/40 + 0.5/750 = 0.0125 + 0.000667 = 0.013167
    #   rho = 1/0.013167 = 75.948...
    #   T = 400
    # Region 2: h = 3000e3 (superheated)
    #   rho = 40 - 2e-5*(3000e3 - 2800e3) = 40 - 4 = 36
    #   T   = 400 + (3000e3 - 2800e3)/2000 = 400 + 100 = 500

    cases = [
        # (label, p, h, expected_reg, expected_rho, expected_T)
        ("L1 p=10MPa h=400kJ/kg",  10.0e6, 400.0e3, 1,
         750.0 + 6.25e-5 * (800.0e3 - 400.0e3),
         400.0 - (800.0e3 - 400.0e3) / 4000.0),

        ("L2 p=12MPa h=600kJ/kg",  12.0e6, 600.0e3, 1,
         rho_f(12e6) + A_L * (h_f(12e6) - 600e3),
         T_sat(12e6) - (h_f(12e6) - 600e3) / cp_L),

        ("L3 p=8MPa h=200kJ/kg",   8.0e6, 200.0e3, 1,
         rho_f(8e6) + A_L * (h_f(8e6) - 200e3),
         T_sat(8e6) - (h_f(8e6) - 200e3) / cp_L),

        ("T1 p=10MPa h=1800kJ/kg", 10.0e6, 1800.0e3, 4,
         1.0 / (0.5/40.0 + 0.5/750.0),
         400.0),

        ("T2 p=10MPa h=h_f (x=0)", 10.0e6, 800.0e3, 4,
         750.0,   # rho_f at p_ref
         400.0),

        ("T3 p=10MPa h=h_g (x=1)", 10.0e6, 2800.0e3, 4,
         40.0,    # rho_g at p_ref
         400.0),

        ("G1 p=10MPa h=3000kJ/kg", 10.0e6, 3000.0e3, 2,
         40.0 - 2.0e-5 * (3000.0e3 - 2800.0e3),
         400.0 + (3000.0e3 - 2800.0e3) / 2000.0),

        ("G2 p=12MPa h=3200kJ/kg", 12.0e6, 3200.0e3, 2,
         rho_g(12e6) - A_G * (3200e3 - h_g(12e6)),
         T_sat(12e6) + (3200e3 - h_g(12e6)) / cp_G),

        ("G3 p=8MPa h=3500kJ/kg",  8.0e6, 3500.0e3, 2,
         rho_g(8e6) - A_G * (3500e3 - h_g(8e6)),
         T_sat(8e6) + (3500e3 - h_g(8e6)) / cp_G),
    ]

    for label, p, h, exp_reg, exp_rho, exp_T in cases:
        reg = region_ph(p, h)
        rho = rho_ph(p, h)
        T   = T_ph(p, h)

        reg_ok = (reg == exp_reg)
        rho_err = abs(rho - exp_rho) / (abs(exp_rho) + 1e-30)
        T_err   = abs(T - exp_T) / (abs(exp_T) + 1e-30)
        ok = reg_ok and rho_err < 1e-12 and T_err < 1e-12
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(f"  {label:30s}  reg={reg}  rho_err={rho_err:.2e}  T_err={T_err:.2e}  [{tag}]")

    print(f"\n  Test 1 overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ==========================================================================
# TEST 2: Derivative consistency (analytical vs. finite difference)
# ==========================================================================
def test_derivatives():
    print("\n=== Test 2: Derivative Consistency (analytical vs. finite difference) ===")
    all_pass = True

    # Single-phase tolerance: linear functions, but FD has cancellation error
    # ~rho * eps_machine / delta_rho ≈ 775 * 2.2e-16 / 5e-5 ≈ 3e-9
    tol_sp = 1e-7
    # Two-phase tolerance: nonlinear mixing → FD has O(eps^2) truncation
    tol_tp = 1e-6

    eps_p = 10.0    # Pa perturbation
    eps_h = 10.0    # J/kg perturbation

    points = [
        (10.0e6, 400.0e3, "L1"),
        (12.0e6, 600.0e3, "L2"),
        ( 8.0e6, 200.0e3, "L3"),
        (10.0e6, 1800.0e3, "T1"),
        (10.0e6, 1200.0e3, "T2"),
        (10.0e6, 2400.0e3, "T3"),
        (10.0e6, 3000.0e3, "G1"),
        (12.0e6, 3200.0e3, "G2"),
        ( 8.0e6, 3500.0e3, "G3"),
    ]

    for p, h, label in points:
        reg = region_ph(p, h)
        tol = tol_tp if reg == 4 else tol_sp

        a_dp = drho_dp_h(p, h)
        a_dh = drho_dh_p(p, h)

        fd_dp = (rho_ph(p + eps_p, h) - rho_ph(p - eps_p, h)) / (2.0 * eps_p)
        fd_dh = (rho_ph(p, h + eps_h) - rho_ph(p, h - eps_h)) / (2.0 * eps_h)

        err_dp = abs(a_dp - fd_dp) / (abs(fd_dp) + 1e-30)
        err_dh = abs(a_dh - fd_dh) / (abs(fd_dh) + 1e-30)

        ok = err_dp < tol and err_dh < tol
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(f"  {label} R{reg} p={p/1e6:.0f}MPa h={h/1e3:.0f}kJ/kg"
              f"  dp_err={err_dp:.2e}  dh_err={err_dh:.2e}  [{tag}]")

    print(f"\n  Test 2 overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ==========================================================================
# TEST 3: Region boundary continuity
# ==========================================================================
def test_boundary_continuity():
    print("\n=== Test 3: Region Boundary Continuity ===")
    all_pass = True
    eps = 1.0  # 1 J/kg offset from boundary

    for p in [8.0e6, 10.0e6, 12.0e6]:
        hfv = h_f(p)
        hgv = h_g(p)

        # Region 1/4 boundary
        rho_below = rho_ph(p, hfv - eps)   # Region 1
        rho_above = rho_ph(p, hfv + eps)   # Region 4 (x ~ 0)
        gap_f = abs(rho_below - rho_above)
        # Expected gap: O(eps) from the linear slope
        ok_f = gap_f < 0.01  # < 0.01 kg/m^3 for 1 J/kg offset
        all_pass = all_pass and ok_f

        # Region 4/2 boundary
        rho_below2 = rho_ph(p, hgv - eps)  # Region 4 (x ~ 1)
        rho_above2 = rho_ph(p, hgv + eps)  # Region 2
        gap_g = abs(rho_below2 - rho_above2)
        ok_g = gap_g < 0.01

        all_pass = all_pass and ok_g
        tag_f = "PASS" if ok_f else "FAIL"
        tag_g = "PASS" if ok_g else "FAIL"
        print(f"  p={p/1e6:.0f}MPa  h_f boundary: gap={gap_f:.2e} [{tag_f}]"
              f"   h_g boundary: gap={gap_g:.2e} [{tag_g}]")

    print(f"\n  Test 3 overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ==========================================================================
# TEST 4: Derivative sign checks
# ==========================================================================
def test_derivative_signs():
    print("\n=== Test 4: Derivative Sign Checks ===")
    all_pass = True

    points = [
        (10.0e6, 400.0e3),   # liquid
        (10.0e6, 1800.0e3),  # two-phase
        (10.0e6, 3000.0e3),  # vapour
        (8.0e6,  500.0e3),   # liquid, off-reference
        (12.0e6, 2000.0e3),  # two-phase, off-reference
        (8.0e6,  3500.0e3),  # vapour, off-reference
    ]

    for p, h in points:
        reg = region_ph(p, h)
        dp = drho_dp_h(p, h)
        dh = drho_dh_p(p, h)

        ok_dp = dp > 0   # density increases with pressure
        ok_dh = dh < 0   # density decreases with enthalpy
        ok = ok_dp and ok_dh
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(f"  R{reg} p={p/1e6:.0f}MPa h={h/1e3:.0f}kJ/kg"
              f"  dp={dp:+.3e} ({'>' if ok_dp else '<'}0)"
              f"  dh={dh:+.3e} ({'<' if ok_dh else '>'}0)  [{tag}]")

    print(f"\n  Test 4 overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ==========================================================================
# TEST 5: Roundtrip consistency & convenience functions
# ==========================================================================
def test_roundtrip_consistency():
    """Verify rho_ph(p, h_pT(p, T)) == rho_pT(p, T) and T_ph(p, h_pT(p, T)) == T."""
    print("\n=== Test 5: Roundtrip Consistency ===")
    all_pass = True
    tol = 1e-12

    def h_pT(p, T):
        """Enthalpy from (p, T) — matches SimpleFluid.mo h_pT."""
        T_s = T_sat(p)
        if T < T_s:
            return h_f(p) - cp_L * (T_s - T)
        else:
            return h_g(p) + cp_G * (T - T_s)

    def rho_pT_direct(p, T):
        return rho_ph(p, h_pT(p, T))

    # Test points spanning both regions (avoid exact saturation where T_ph is flat)
    points = [
        (10e6, 300.0, "Liquid 300K"),
        (10e6, 380.0, "Liquid 380K"),
        (8e6,  350.0, "Liquid 350K"),
        (12e6, 350.0, "Liquid 350K hi-p"),
        (10e6, 500.0, "Vapour 500K"),
        (10e6, 600.0, "Vapour 600K"),
        (8e6,  450.0, "Vapour 450K"),
        (12e6, 550.0, "Vapour 550K"),
    ]

    for p, T, label in points:
        h = h_pT(p, T)
        rho_via_ph = rho_ph(p, h)
        rho_via_pT = rho_pT_direct(p, T)
        T_back = T_ph(p, h)

        err_rho = abs(rho_via_ph - rho_via_pT) / (abs(rho_via_pT) + 1e-30)
        err_T = abs(T_back - T) / (abs(T) + 1e-30)
        ok = err_rho < tol and err_T < tol
        all_pass = all_pass and ok
        tag = "PASS" if ok else "FAIL"
        print(f"  {label:25s}  rho_err={err_rho:.2e}  T_err={err_T:.2e}  [{tag}]")

    print(f"\n  Test 5 overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ==========================================================================
# TEST 6: Extraction transparency (OMPython)
# ==========================================================================
def test_extraction_transparency():
    print("\n=== Test 5: Extraction Transparency (no OPAQUE in XML) ===")
    try:
        from OMPython import OMCSessionZMQ
    except ImportError:
        print("  SKIP — OMPython not available.")
        return None

    OM_HOME = OPAL_ROOT / "external" / "OpenModelica" / "build_cmake" / "install_cmake"
    if not (OM_HOME / "bin" / "omc").exists():
        print("  SKIP — OpenModelica not built.")
        return None

    try:
        omc = OMCSessionZMQ(omhome=str(OM_HOME))
    except Exception as e:
        print(f"  SKIP — OpenModelica session failed ({e}).")
        return None

    import tempfile

    lib_path = OPAL_ROOT / "library" / "Media"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        omc.sendExpression(f'cd("{tmpdir}")', parsed=False)

        # Copy Media package with within-clause stripped
        media_tmp = tmpdir / "Media"
        media_tmp.mkdir()

        # package.mo
        pkg_src = (lib_path / "package.mo").read_text()
        pkg_src = re.sub(r'within\s+OPAL\.library\s*;', '', pkg_src, count=1)
        (media_tmp / "package.mo").write_text(pkg_src)

        # SimpleFluid.mo
        sf_src = (lib_path / "SimpleFluid.mo").read_text()
        sf_src = re.sub(r'within\s+OPAL\.library\.', 'within ', sf_src, count=1)
        (media_tmp / "SimpleFluid.mo").write_text(sf_src)

        result = omc.sendExpression(f'loadFile("{media_tmp / "package.mo"}")', parsed=False)
        if 'false' in str(result).lower():
            err = omc.sendExpression('getErrorString()', parsed=False)
            print(f"  FAIL — could not load Media package: {err}")
            return False

        # Probe model
        probe_file = tmpdir / "SFProbe.mo"
        probe_file.write_text(
            'model SFProbe\n'
            '  parameter Real p = 10.0e6;\n'
            '  parameter Real h = 1.0e6;\n'
            '  Real rho_val;\n'
            '  Real drho_dp_val;\n'
            '  Real drho_dh_val;\n'
            'equation\n'
            '  rho_val       = Media.SimpleFluid.rho_ph(p, h);\n'
            '  drho_dp_val   = Media.SimpleFluid.drho_dp_h(p, h);\n'
            '  drho_dh_val   = Media.SimpleFluid.drho_dh_p(p, h);\n'
            'end SFProbe;\n'
        )
        omc.sendExpression(f'loadFile("{probe_file}")', parsed=False)

        xml_result = omc.sendExpression(
            'dumpXMLDAE(SFProbe, addMathMLCode=false)', parsed=False
        )

    xml_str = str(xml_result)
    xml_match = re.search(r'"([^"]+\.xml)"', xml_str)
    if xml_match:
        xml_file = pathlib.Path(xml_match.group(1))
        if xml_file.exists():
            xml_str = xml_file.read_text()

    if "OPAQUE" in xml_str:
        print("  FAIL — OPAQUE external call marker found in extracted XML.")
        idx = xml_str.find("OPAQUE")
        print("  ..." + xml_str[max(0, idx-100):idx+200] + "...")
        return False
    elif "true" not in str(xml_result).lower() and "xml" not in xml_str.lower():
        err = omc.sendExpression('getErrorString()', parsed=False)
        print(f"  FAIL — dumpXMLDAE did not produce output: {err}")
        return False
    else:
        print("  PASS — no OPAQUE markers in extracted XML.")
        return True


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    print("OPAL SimpleFluid Verification Tests")
    print("=" * 50)

    results = []
    results.append(("Property values",           test_property_values()))
    results.append(("Derivative consistency",     test_derivatives()))
    results.append(("Boundary continuity",        test_boundary_continuity()))
    results.append(("Derivative signs",           test_derivative_signs()))
    results.append(("Roundtrip consistency",       test_roundtrip_consistency()))
    results.append(("Extraction transparency",    test_extraction_transparency()))

    print("\n" + "=" * 50)
    print("SUMMARY")
    n_pass = n_fail = n_skip = 0
    for name, result in results:
        if result is True:
            tag = "PASS"; n_pass += 1
        elif result is False:
            tag = "FAIL"; n_fail += 1
        else:
            tag = "SKIP"; n_skip += 1
        print(f"  {name:35s}  {tag}")

    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    sys.exit(0 if n_fail == 0 else 1)
