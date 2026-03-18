#!/usr/bin/env python3
"""
clean_digitized_data.py — Clean up digitized Edwards-O'Brien experimental data.

Rules (from user):
  1. If any time is negative, offset ALL times by |most_negative| + 0.0002
  2. If any time is non-monotonic, interpolate it from its 2 neighbors
  3. If any pressure is negative, set to 0.5
  4. Process one file at a time, save as figN_clean.csv

Figure-to-gauge-station mapping:
  fig3 → GS-1 (x=3.927m, near break)
  fig4 → GS-2 (x=3.769m)
  fig5 → GS-3 (x=2.935m)
  fig6 → GS-4 (x=2.024m)
  fig7 → GS-5 (x=1.469m)
  fig8 → GS-6 (x=0.914m)
  fig9 → GS-7 (x=0.079m, near closed end)
"""

import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent

fig_to_gs = {
    "fig3": "GS-1",
    "fig4": "GS-2",
    "fig5": "GS-3",
    "fig6": "GS-4",
    "fig7": "GS-5",
    "fig8": "GS-6",
    "fig9": "GS-7",
}


def clean_file(fig_name):
    """Clean one digitized CSV file."""
    raw_path = data_dir / f"{fig_name}.csv"
    clean_path = data_dir / f"{fig_name}_clean.csv"
    gs_name = fig_to_gs[fig_name]

    raw = np.loadtxt(raw_path, delimiter=",")
    t = raw[:, 0].copy()
    p = raw[:, 1].copy()
    n = len(t)

    print(f"\n{'='*60}")
    print(f"{fig_name}.csv → {gs_name}  ({n} points)")
    print(f"{'='*60}")

    fixes = []

    # Rule 1: negative time → offset ALL points by |most_negative|,
    # then add 0.0001 to the first point only (so it's not exactly 0,
    # since t=0 is the rupture instant and the first data point is after it)
    t_min = t.min()
    if t_min < 0:
        offset = abs(t_min)
        print(f"  Most negative time: {t_min:.10f} s")
        print(f"  Offsetting all times by +{offset:.10f} s")
        t += offset
        # First point would now be exactly 0 — nudge it
        t[0] += 0.0001
        print(f"  First point nudged to {t[0]:.10f} s")
        fixes.append(f"time offset +{offset:.6f}, t[0] nudged +0.0001")

    # Rule 2: non-monotonic time → interpolate from neighbors
    # Run multiple passes since fixing one point can reveal another
    for _pass in range(5):
        fixed_any = False
        for i in range(1, n - 1):
            if t[i] <= t[i - 1]:
                t_old = t[i]
                t[i] = 0.5 * (t[i - 1] + t[i + 1])
                print(f"  Non-monotonic at index {i}: {t_old:.10f} → {t[i]:.10f}")
                fixes.append(f"t[{i}] interpolated")
                fixed_any = True
        # Check last point
        if n > 1 and t[-1] <= t[-2]:
            t_old = t[-1]
            t[-1] = t[-2] + 0.001
            print(f"  Non-monotonic at last index: {t_old:.10f} → {t[-1]:.10f}")
            fixes.append(f"t[{n-1}] adjusted")
            fixed_any = True
        if not fixed_any:
            break

    # Rule 3: negative pressure → set to 0.5
    for i in range(n):
        if p[i] < 0:
            print(f"  Negative pressure at index {i}: {p[i]:.4f} → 0.5 psia")
            p[i] = 0.5
            fixes.append(f"p[{i}] → 0.5")

    # Verify monotonicity
    dt = np.diff(t)
    if np.all(dt > 0):
        print(f"  ✓ Time is monotonically increasing")
    else:
        bad = np.where(dt <= 0)[0]
        print(f"  WARNING: Still non-monotonic at indices {bad}")

    # Save
    out = np.column_stack([t, p])
    np.savetxt(clean_path, out, delimiter=",", fmt="%.15g")
    print(f"  Saved → {clean_path.name}")

    # Summary
    p_MPa = p * 6894.76 / 1e6
    print(f"  t: [{t[0]*1e3:.3f}, {t[-1]*1e3:.1f}] ms")
    print(f"  p: [{p_MPa.min():.3f}, {p_MPa.max():.3f}] MPa")
    if fixes:
        print(f"  Fixes: {'; '.join(fixes)}")
    else:
        print(f"  No fixes needed")

    return t, p


# Process each file
for fig_name in sorted(fig_to_gs.keys()):
    clean_file(fig_name)
