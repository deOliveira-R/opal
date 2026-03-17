# Edwards-O'Brien Pipe Blowdown Test — OPAL Validation Data
#
# NRC Standard Problem 1
#
# Sources:
#   [1] Edwards, A.R. and O'Brien, T.P., "Studies of phenomena connected with the
#       depressurization of water reactors", J. British Nuclear Energy Society,
#       vol. 9, pp. 125-135, April 1970.
#   [2] Tomlinson, E.T. and Aumiller, D.L., "An Assessment of RELAP5-3D Using the
#       Edwards-O'Brien Blowdown Problem", B-T-3271, Bettis Atomic Power Lab, 1999.
#       (OSTI: https://www.osti.gov/servlets/purl/755356)
#   [3] Krotiuk, W.J., "PWR Steam Generator Internal Loading Following MSLB/FWLB",
#       SMSAB-02-05 Rev.1, NRC Office of Nuclear Regulatory Research, May 2004.
#       (NRC ML: ML080921008)
#   [4] Hendrie, J.M., AEC Direction and Guidance Letter on Comparative Analyses
#       of Standard Problems, USAEC, January 31, 1973.
#   [5] LA-6005-MS, Los Alamos, June 1975. (OSTI: https://www.osti.gov/servlets/purl/4139411)
#
# NOTE ON EXPERIMENTAL DATA:
#   The original Edwards & O'Brien (1970) paper published time-series data as
#   FIGURES ONLY (no tabulated data). No digitized dataset has been published
#   in the open literature. The pressure-vs-time and void-fraction-vs-time plots
#   must be digitized from the original paper or from the comparison plots in
#   sources [2], [3], or [5].
#
#   The data below provides everything needed to SET UP the problem. For
#   comparison against experiment, you need to digitize the plots from [1] or [2].
#   A plot digitizer (e.g., WebPlotDigitizer) applied to the OSTI report [2]
#   figures is the recommended path.

import json

edwards_blowdown = {

    "description": (
        "Edwards-O'Brien horizontal pipe blowdown experiment. "
        "A 4.096 m horizontal pipe filled with subcooled water at ~7 MPa "
        "is ruptured at one end by breaking a glass disk. The transient "
        "involves pressure rarefaction wave propagation, flashing onset, "
        "critical two-phase flow at the break, and void fraction wave "
        "propagation. Duration: 0.6 seconds. "
        "NRC Standard Problem 1."
    ),

    # ══════════════════════════════════════════════════════════════════
    # GEOMETRY
    # ══════════════════════════════════════════════════════════════════
    "geometry": {
        "pipe_length_m": 4.096,         # [1][2] 13.44 ft
        "pipe_inner_diameter_m": 0.073,  # [1][2] 2.88 in = 73.152 mm
        "pipe_flow_area_m2": 4.185e-3,   # pi/4 * 0.073^2
        "orientation": "horizontal",
        "closed_end": "left",            # x=0 is the sealed concrete abutment
        "break_end": "right",            # x=4.096m is the glass rupture disk
        "break_flow_area_fraction": 0.87, # 13% reduction from glass remnants [1][2]
        "break_opening_time_ms": 1.0,    # estimated disk rupture time [1][2]
    },

    # ══════════════════════════════════════════════════════════════════
    # GAUGE STATION LOCATIONS
    # Measured from closed end (x=0).
    # Source: [2] Figure 1, confirmed in [3] Table 3.1-2
    # ══════════════════════════════════════════════════════════════════
    "gauge_stations": {
        "GS-7": {"x_m": 0.079, "x_ft": 0.26, "measures": ["pressure"]},
        "GS-6": {"x_m": 0.914, "x_ft": 3.00, "measures": ["pressure"]},
        "GS-5": {"x_m": 1.469, "x_ft": 4.82, "measures": ["pressure", "temperature", "void_fraction"]},
        "GS-4": {"x_m": 2.024, "x_ft": 6.64, "measures": ["pressure"]},
        "GS-3": {"x_m": 2.935, "x_ft": 9.63, "measures": ["pressure"]},
        "GS-2": {"x_m": 3.769, "x_ft": 12.37, "measures": ["pressure"]},
        "GS-1": {"x_m": 3.927, "x_ft": 12.89, "measures": ["pressure"]},
    },

    # ══════════════════════════════════════════════════════════════════
    # INITIAL CONDITIONS — Standard Problem 1
    # Nominal: 7.0 MPa, ~514 K (465°F) — but NOT isothermal.
    # The actual experiment had a temperature gradient along the pipe.
    # Source: [2] Table 1 (Hendrie initial conditions [4])
    # ══════════════════════════════════════════════════════════════════
    "initial_conditions": {
        "nominal_pressure_MPa": 7.0,        # 1000 psig
        "nominal_pressure_psi": 1000,
        "nominal_temperature_K": 513.7,      # 465°F — nominal
        "subcooling_note": (
            "Water is subcooled at initial conditions. "
            "Saturation temperature at 7 MPa is ~559 K (547°F). "
            "Subcooling is ~45 K."
        ),
        "saturation_temperature_K_at_7MPa": 558.98,

        # Actual temperature profile along the pipe from [2] Table 1
        # (Hendrie data). NOT isothermal — varies by ~10°F along the pipe.
        # x measured from closed end, temperature in K and °F.
        "temperature_profile": [
            # x_ft, x_m,    T_F,   T_K   (from Bettis report Table 1)
            (0.26,  0.079,  447.5, 503.7),   # GS-7
            (0.78,  0.238,  448.4, 504.2),
            (1.30,  0.396,  449.4, 504.8),
            (1.82,  0.555,  450.3, 505.3),
            (2.385, 0.727,  451.4, 505.9),
            (3.00,  0.914,  452.5, 506.5),   # GS-6
            (3.615, 1.102,  451.7, 506.1),
            (4.215, 1.285,  450.8, 505.6),
            (4.82,  1.469,  450.0, 505.1),   # GS-5
            (5.425, 1.654,  449.2, 504.7),
            (6.025, 1.836,  448.3, 504.2),
            (6.64,  2.024,  447.5, 503.7),   # GS-4
            (7.255, 2.211,  448.0, 504.0),
            (7.82,  2.384,  448.5, 504.3),
            (8.34,  2.542,  448.9, 504.5),
            (8.94,  2.725,  449.4, 504.8),
            (9.63,  2.935,  450.0, 505.1),   # GS-3
            (10.32, 3.146,  446.5, 503.1),
            (10.92, 3.328,  443.4, 501.4),
            (11.44, 3.487,  440.8, 500.0),
            (11.905,3.628,  438.4, 498.6),
            (12.37, 3.769,  436.0, 497.3),   # GS-2
            (12.89, 3.927,  437.0, 497.8),   # GS-1
            (13.295,4.052,  437.8, 498.3),
        ],

        # Simplified isothermal initial conditions used by many codes
        # (from RELAP5-3D standard installation problem)
        "simplified_isothermal_K": 502.2,     # 444.3°F
        "simplified_isothermal_F": 444.3,
    },

    # ══════════════════════════════════════════════════════════════════
    # TRANSIENT DURATION AND MEASURED QUANTITIES
    # ══════════════════════════════════════════════════════════════════
    "transient": {
        "duration_s": 0.6,
        "measured_quantities": [
            "Pressure at GS-1 through GS-7 (all 7 stations)",
            "Temperature at GS-5 only",
            "Void fraction at GS-5 only",
        ],
        "break_flow_note": (
            "Break mass flow rate was NOT directly measured. "
            "It can only be inferred from pressure histories. "
            "See [2] for discussion of sensitivity."
        ),
    },

    # ══════════════════════════════════════════════════════════════════
    # KEY PHYSICAL PHENOMENA TO CAPTURE
    # (What the solver must get right to pass this validation)
    # ══════════════════════════════════════════════════════════════════
    "phenomena": [
        {
            "name": "Pressure rarefaction wave",
            "description": (
                "Initial depressurization propagates as a rarefaction wave "
                "from the break (x=4.096m) toward the closed end (x=0). "
                "Wave speed ~ speed of sound in subcooled water ~ 1000-1200 m/s. "
                "Wave should reach closed end in ~3-4 ms and reflect."
            ),
            "validation_check": (
                "Pressure drop at GS-1 (near break) should begin immediately. "
                "Pressure drop at GS-7 (near closed end) should be delayed by "
                "~3-4 ms. The arrival time at each gauge station tests the "
                "wave propagation speed."
            ),
        },
        {
            "name": "Flashing onset",
            "description": (
                "As pressure drops below saturation pressure (~2.14 MPa at the "
                "local temperature), water begins to flash. This occurs first "
                "near the break and propagates upstream."
            ),
            "validation_check": (
                "The pressure history should show a change in slope when "
                "flashing begins — the depressurization rate slows because "
                "vapor generation partially offsets the mass loss."
            ),
        },
        {
            "name": "Critical (choked) flow at break",
            "description": (
                "Flow at the break becomes choked. Mass flow rate depends "
                "on upstream conditions, not downstream (atmospheric) pressure."
            ),
            "validation_check": (
                "Pressure at GS-1 should not drop to atmospheric immediately — "
                "choked flow limits the depressurization rate."
            ),
        },
        {
            "name": "Void fraction evolution at GS-5",
            "description": (
                "Void fraction at the pipe midpoint starts at 0, begins "
                "increasing when the rarefaction wave arrives and pressure "
                "drops below local saturation, then increases gradually "
                "as the pipe empties."
            ),
            "validation_check": (
                "Void fraction at GS-5 should remain 0 for the first "
                "~2-3 ms (wave transit time), then begin increasing. "
                "Should reach ~0.5-0.7 by t=0.3s, approaching ~0.8-0.9 "
                "by t=0.6s."
            ),
        },
    ],

    # ══════════════════════════════════════════════════════════════════
    # APPROXIMATE VALIDATION TARGETS (extracted from published plots)
    # These are APPROXIMATE values read from figures in [2] and [3].
    # For rigorous validation, digitize the original plots.
    # ══════════════════════════════════════════════════════════════════
    "approximate_targets": {
        "GS-5_pressure": {
            "description": "Approximate pressure at GS-5 (pipe center) vs time",
            "units": {"time": "s", "pressure": "MPa"},
            "data_note": "Approximate values read from figures. Use digitized data for final validation.",
            "points": [
                # (time_s, pressure_MPa)  — approximate
                (0.000, 7.0),
                (0.002, 6.9),      # rarefaction wave arriving
                (0.005, 5.5),      # rapid depressurization
                (0.010, 4.2),
                (0.020, 3.0),
                (0.050, 2.2),      # approaching saturation pressure
                (0.100, 1.5),
                (0.200, 0.8),
                (0.300, 0.5),
                (0.400, 0.3),
                (0.500, 0.2),
                (0.600, 0.15),
            ],
        },
        "GS-5_void_fraction": {
            "description": "Approximate void fraction at GS-5 vs time",
            "units": {"time": "s", "void_fraction": "-"},
            "data_note": "Approximate values read from figures. Use digitized data for final validation.",
            "points": [
                # (time_s, void_fraction)  — approximate
                (0.000, 0.0),
                (0.010, 0.0),
                (0.020, 0.0),       # still subcooled
                (0.050, 0.05),      # flashing begins
                (0.100, 0.15),
                (0.200, 0.40),
                (0.300, 0.55),
                (0.400, 0.65),
                (0.500, 0.75),
                (0.600, 0.80),
            ],
        },
        "GS-7_pressure": {
            "description": "Approximate pressure at GS-7 (near closed end) vs time",
            "units": {"time": "s", "pressure": "MPa"},
            "data_note": "Approximate values read from figures. Pressure at closed end drops later than near break.",
            "points": [
                # (time_s, pressure_MPa) — approximate
                (0.000, 7.0),
                (0.003, 7.0),       # wave hasn't arrived yet
                (0.005, 6.5),       # wave arriving
                (0.010, 5.5),       # reflected wave
                (0.020, 4.5),
                (0.050, 3.0),
                (0.100, 2.0),
                (0.200, 1.0),
                (0.400, 0.4),
                (0.600, 0.2),
            ],
        },
    },

    # ══════════════════════════════════════════════════════════════════
    # MODELING GUIDANCE (from RELAP5/TRAC-M assessment reports)
    # ══════════════════════════════════════════════════════════════════
    "modeling_guidance": {
        "nodalization": (
            "20-24 equal-length volumes is standard. "
            "Node size ~0.05-0.11 m provides acceptable resolution of the "
            "pressure wave. Finer (161 nodes in [3]) gives smoother results "
            "but same general behavior. "
            "Node size comparable to pipe diameter (~0.073 m) is adequate."
        ),
        "time_step": (
            "Maximum time step 0.1 ms for semi-implicit scheme. "
            "CFL condition for acoustic waves: dt < dx/c_sound. "
            "For dx=0.1m, c=1200 m/s: dt < 0.083 ms."
        ),
        "critical_flow_model": (
            "Ransom-Trapp model with default discharge coefficients "
            "used in RELAP5/TRAC assessments. Henry-Fauske also tested — "
            "gives higher break flow, slightly different void fraction. "
            "Results not dramatically different between models."
        ),
        "heat_structures": (
            "Pipe wall heat structures can be neglected — transient is too "
            "short (0.6s) for significant wall heat transfer. "
            "Confirmed in [2]: removing heat structures had insignificant effect."
        ),
        "initial_conditions_sensitivity": (
            "CRITICAL: using isothermal vs actual temperature profile "
            "gives significantly different results, especially for break flow "
            "and void fraction near the break. Use the actual temperature "
            "profile from the temperature_profile table for accurate assessment. "
            "See [2] for detailed comparison."
        ),
    },

    # ══════════════════════════════════════════════════════════════════
    # DATA SOURCES FOR DIGITIZATION
    # (Where to get the actual experimental curves)
    # ══════════════════════════════════════════════════════════════════
    "digitization_sources": [
        {
            "source": "Bettis RELAP5-3D Assessment (1999)",
            "url": "https://www.osti.gov/servlets/purl/755356",
            "figures": [
                "Figure 3: Pressure vs time at GS-1 (experimental data points shown)",
                "Figure 4: Pressure vs time at GS-2",
                "Figure 5: Pressure vs time at GS-3",
                "Figure 6: Pressure vs time at GS-4",
                "Figure 7: Pressure vs time at GS-5",
                "Figure 8: Pressure vs time at GS-6",
                "Figure 9: Pressure vs time at GS-7",
                "Figure 14: Void fraction vs time at GS-5",
            ],
            "note": (
                "This report has the clearest overlay of experimental data "
                "with code predictions. Recommended for digitization."
            ),
        },
        {
            "source": "NRC SMSAB-02-05 Rev.1 (2004)",
            "url": "https://www.nrc.gov/docs/ML0809/ML080921008.pdf",
            "figures": [
                "Figures 3.1-1 through 3.1-9: TRAC-M vs experiment at all gauge stations",
            ],
        },
        {
            "source": "Original paper",
            "reference": (
                "Edwards, A.R. and O'Brien, T.P., J. British Nuclear Energy Society, "
                "vol. 9, pp. 125-135, 1970."
            ),
            "note": "Original source. May be hard to access. The assessment reports above reproduce the data.",
        },
    ],
}


if __name__ == "__main__":
    # Print summary
    print("=" * 70)
    print("EDWARDS-O'BRIEN BLOWDOWN TEST — OPAL VALIDATION DATA")
    print("=" * 70)

    g = edwards_blowdown["geometry"]
    print(f"\nGeometry:")
    print(f"  Pipe length:    {g['pipe_length_m']} m")
    print(f"  Inner diameter: {g['pipe_inner_diameter_m']*1000:.1f} mm")
    print(f"  Flow area:      {g['pipe_flow_area_m2']*1e4:.3f} cm²")
    print(f"  Orientation:    {g['orientation']}")
    print(f"  Break area:     {g['break_flow_area_fraction']*100:.0f}% of pipe area")

    ic = edwards_blowdown["initial_conditions"]
    print(f"\nInitial conditions (Standard Problem 1):")
    print(f"  Pressure:    {ic['nominal_pressure_MPa']} MPa ({ic['nominal_pressure_psi']} psig)")
    print(f"  Temperature: {ic['nominal_temperature_K']:.1f} K (nominal)")
    print(f"  Subcooling:  ~{ic['saturation_temperature_K_at_7MPa'] - ic['nominal_temperature_K']:.0f} K")

    print(f"\nGauge stations:")
    for name, gs in edwards_blowdown["gauge_stations"].items():
        print(f"  {name}: x = {gs['x_m']:.3f} m  ({', '.join(gs['measures'])})")

    print(f"\nTemperature profile ({len(ic['temperature_profile'])} points):")
    for x_ft, x_m, T_F, T_K in ic["temperature_profile"][:5]:
        print(f"  x = {x_m:.3f} m: T = {T_K:.1f} K ({T_F:.1f} °F)")
    print(f"  ... ({len(ic['temperature_profile'])-5} more points)")

    print(f"\nPhenomena to capture: {len(edwards_blowdown['phenomena'])}")
    for p in edwards_blowdown["phenomena"]:
        print(f"  - {p['name']}")

    print(f"\nDigitization sources: {len(edwards_blowdown['digitization_sources'])}")
    for s in edwards_blowdown["digitization_sources"]:
        print(f"  - {s['source']}")
        if "url" in s:
            print(f"    {s['url']}")

    print(f"\n{'=' * 70}")
    print("NOTE: Approximate validation targets are included but are read from")
    print("figures. For rigorous validation, digitize the experimental data from")
    print("the Bettis RELAP5-3D assessment report (OSTI link above).")
    print("Recommend: WebPlotDigitizer on Figures 3-9 and 14 of that report.")
    print(f"{'=' * 70}")
