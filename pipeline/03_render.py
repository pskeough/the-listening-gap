"""Stage 3 frequency-response renderer: 13 EQ parameters -> 256-point dB curve.

Purpose: this is the MEASURING INSTRUMENT of the study. Every distributional
comparison downstream (human vs. model curves, arm scoring, distance metrics)
is computed on the output of this module, so a bug here silently corrupts
every result in the paper. It must therefore be validated by unit tests with
known analytic answers BEFORE anything consumes it:
    python 03_render.py --selftest
writes the receipt file
    results/stats/render_validation.txt
and no downstream stage may run against an implementation whose receipt does
not end in ALL PASS.

Rigor rules encoded here:
  * Filter math is the RBJ Audio EQ Cookbook (Robert Bristow-Johnson,
    "Cookbook formulae for audio EQ biquad filter coefficients") — the same
    formulas the SAFE Equaliser plugin family uses — NOT an ad-hoc
    approximation. Shelves use the Q-based form (alpha = sin(w0)/(2*Q));
    peaking uses A = 10^(dB/40) so the curve peaks at exactly `gain` dB.
  * The response is evaluated exactly: H(z) at z = e^{j*2*pi*f/fs} with
    complex arithmetic, per filter, then summed in dB across the cascade.
    No sampled-impulse/FFT approximation, no interpolation.
  * The frequency grid is FIXED (256 log-spaced points, 20 Hz .. 20 kHz
    inclusive, fs = 48 kHz) and exposed as FREQ_GRID so every stage measures
    on identical axes. Tolerances in the self-tests are part of the spec:
    a failing test means fix the implementation, never widen the tolerance.
  * No randomness anywhere -> no seed needed; re-runs are byte-comparable.

Zero-dependency (stdlib only), matching the project's existing convention.
"""

import argparse
import cmath
import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_VALIDATION = BASE / "results" / "stats" / "render_validation.txt"

FS = 48000.0          # sample rate the plugin family runs at
SHELF_Q = 0.71        # fixed shelf Q per the plugin's parameter layout
N_POINTS = 256
F_LO, F_HI = 20.0, 20000.0

# Log-spaced measurement grid, endpoints exact: f_i = 20 * 1000^(i/255).
FREQ_GRID = [F_LO * (F_HI / F_LO) ** (i / (N_POINTS - 1)) for i in range(N_POINTS)]

PARAM_KEYS = [
    "low_shelf_gain", "low_shelf_freq",
    "band1_gain", "band1_freq", "band1_q",
    "band2_gain", "band2_freq", "band2_q",
    "band3_gain", "band3_freq", "band3_q",
    "high_shelf_gain", "high_shelf_freq",
]


# ---- RBJ cookbook coefficients (each returns (b0, b1, b2, a0, a1, a2)) -----

def peaking_coeffs(gain_db, f0, q):
    """RBJ peaking EQ. Exact +g/-g inverse pairs at equal f0, Q."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / FS
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    b0 = 1.0 + alpha * a_lin
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * a_lin
    a0 = 1.0 + alpha / a_lin
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / a_lin
    return b0, b1, b2, a0, a1, a2


def low_shelf_coeffs(gain_db, f0, q=SHELF_Q):
    """RBJ low shelf, Q-based form (alpha = sin(w0)/(2Q))."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / FS
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    sq = 2.0 * math.sqrt(a_lin) * alpha
    b0 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cw + sq)
    b1 = 2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cw)
    b2 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cw - sq)
    a0 = (a_lin + 1.0) + (a_lin - 1.0) * cw + sq
    a1 = -2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cw)
    a2 = (a_lin + 1.0) + (a_lin - 1.0) * cw - sq
    return b0, b1, b2, a0, a1, a2


def high_shelf_coeffs(gain_db, f0, q=SHELF_Q):
    """RBJ high shelf, Q-based form (alpha = sin(w0)/(2Q))."""
    a_lin = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / FS
    alpha = math.sin(w0) / (2.0 * q)
    cw = math.cos(w0)
    sq = 2.0 * math.sqrt(a_lin) * alpha
    b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cw + sq)
    b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cw)
    b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cw - sq)
    a0 = (a_lin + 1.0) - (a_lin - 1.0) * cw + sq
    a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cw)
    a2 = (a_lin + 1.0) - (a_lin - 1.0) * cw - sq
    return b0, b1, b2, a0, a1, a2


# ---- Exact evaluation ------------------------------------------------------

def biquad_mag_db(coeffs, f):
    """|H(e^{j*2*pi*f/fs})| in dB via direct complex evaluation."""
    b0, b1, b2, a0, a1, a2 = coeffs
    z1 = cmath.exp(-1j * 2.0 * math.pi * f / FS)   # z^-1 on the unit circle
    z2 = z1 * z1
    h = (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
    return 20.0 * math.log10(abs(h))


def render_curve(params):
    """13-key parameter dict -> list of 256 cascade magnitudes in dB.

    Cascade order (magnitudes multiply, so dB simply add; order is
    irrelevant to the magnitude response but fixed here for clarity):
    low shelf -> peaking 1 -> peaking 2 -> peaking 3 -> high shelf.
    """
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise KeyError(f"params missing required keys: {missing}")
    chain = [
        low_shelf_coeffs(params["low_shelf_gain"], params["low_shelf_freq"]),
        peaking_coeffs(params["band1_gain"], params["band1_freq"], params["band1_q"]),
        peaking_coeffs(params["band2_gain"], params["band2_freq"], params["band2_q"]),
        peaking_coeffs(params["band3_gain"], params["band3_freq"], params["band3_q"]),
        high_shelf_coeffs(params["high_shelf_gain"], params["high_shelf_freq"]),
    ]
    return [sum(biquad_mag_db(c, f) for c in chain) for f in FREQ_GRID]


# ---- Self-tests (tolerances are part of the spec; never widen them) --------

def _params(**overrides):
    """All-flat baseline: every gain 0 dB at plausible neutral frequencies."""
    p = {
        "low_shelf_gain": 0.0, "low_shelf_freq": 100.0,
        "band1_gain": 0.0, "band1_freq": 250.0, "band1_q": 1.0,
        "band2_gain": 0.0, "band2_freq": 1000.0, "band2_q": 1.0,
        "band3_gain": 0.0, "band3_freq": 4000.0, "band3_q": 1.0,
        "high_shelf_gain": 0.0, "high_shelf_freq": 10000.0,
    }
    p.update(overrides)
    return p


def _nearest_idx(f):
    return min(range(N_POINTS), key=lambda i: abs(FREQ_GRID[i] - f))


def run_selftests():
    """Run all unit tests. Returns (report_text, all_passed)."""
    lines = []
    results = []

    def check(name, expected, actual, passed):
        results.append(passed)
        lines.append(f"[{name}]")
        lines.append(f"  expected : {expected}")
        lines.append(f"  actual   : {actual}")
        lines.append(f"  {'PASS' if passed else 'FAIL'}")
        lines.append("")

    lines.append("RENDER VALIDATION — Stage 3 measuring-instrument self-test")
    lines.append(f"fs = {FS:.0f} Hz | grid = {N_POINTS} log points "
                 f"{F_LO:.0f}..{F_HI:.0f} Hz | shelf Q = {SHELF_Q}")
    lines.append("")

    # 1. FLAT: all gains 0 -> identity response everywhere.
    resp = render_curve(_params())
    worst = max(abs(v) for v in resp)
    check("FLAT", "max |dB| over grid < 0.01",
          f"max |dB| = {worst:.6f}", worst < 0.01)

    # 2. PEAK: +6 dB @ 1 kHz Q=1 on band2 only.
    resp = render_curve(_params(band2_gain=6.0, band2_freq=1000.0, band2_q=1.0))
    i1k = _nearest_idx(1000.0)
    at_peak = resp[i1k]
    at_lo, at_hi = resp[0], resp[-1]
    check("PEAK @1kHz",
          f"grid point nearest 1 kHz ({FREQ_GRID[i1k]:.1f} Hz) within 0.05 dB "
          "of +6; 20 Hz and 20 kHz within 0.1 dB of 0",
          f"peak = {at_peak:.4f} dB; 20 Hz = {at_lo:.4f} dB; 20 kHz = {at_hi:.4f} dB",
          abs(at_peak - 6.0) < 0.05 and abs(at_lo) < 0.1 and abs(at_hi) < 0.1)

    # 3. LOW SHELF: +6 dB @ 100 Hz.
    resp = render_curve(_params(low_shelf_gain=6.0, low_shelf_freq=100.0))
    at20 = resp[0]
    at10k = resp[_nearest_idx(10000.0)]
    check("LOW SHELF +6@100Hz",
          "20 Hz within 0.3 dB of +6; 10 kHz within 0.1 dB of 0",
          f"20 Hz = {at20:.4f} dB; 10 kHz = {at10k:.4f} dB",
          abs(at20 - 6.0) < 0.3 and abs(at10k) < 0.1)

    # 4. HIGH SHELF: -6 dB @ 10 kHz.
    resp = render_curve(_params(high_shelf_gain=-6.0, high_shelf_freq=10000.0))
    at20k = resp[-1]
    at100 = resp[_nearest_idx(100.0)]
    check("HIGH SHELF -6@10kHz",
          "20 kHz within 0.5 dB of -6; 100 Hz within 0.1 dB of 0",
          f"20 kHz = {at20k:.4f} dB; 100 Hz = {at100:.4f} dB",
          abs(at20k - (-6.0)) < 0.5 and abs(at100) < 0.1)

    # 5. INVERSE CANCELLATION: RBJ peaking +g and -g at equal f,Q are exact
    #    inverses (the +g numerator IS the -g denominator), so dB sums to 0.
    up = render_curve(_params(band1_gain=6.0, band1_freq=500.0, band1_q=2.0))
    dn = render_curve(_params(band1_gain=-6.0, band1_freq=500.0, band1_q=2.0))
    worst = max(abs(u + d) for u, d in zip(up, dn))
    check("INVERSE CANCELLATION",
          "max |sum of +6/-6 dB curves| < 0.05",
          f"max |dB| = {worst:.2e}", worst < 0.05)

    # 6. GRID SANITY.
    increasing = all(b > a for a, b in zip(FREQ_GRID, FREQ_GRID[1:]))
    ok = (len(FREQ_GRID) == N_POINTS and increasing
          and abs(FREQ_GRID[0] - F_LO) <= 1e-6 * F_LO
          and abs(FREQ_GRID[-1] - F_HI) <= 1e-6 * F_HI)
    check("GRID SANITY",
          "256 points, strictly increasing, 20.0 .. 20000.0 (rel tol 1e-6)",
          f"n = {len(FREQ_GRID)}; increasing = {increasing}; "
          f"first = {FREQ_GRID[0]!r}; last = {FREQ_GRID[-1]!r}", ok)

    all_passed = all(results)
    n_fail = results.count(False)
    lines.append("ALL PASS" if all_passed
                 else f"FAILURES: {n_fail} of {len(results)} tests failed")
    return "\n".join(lines), all_passed


# ---- CLI -------------------------------------------------------------------

def render_jsonl(in_path, out_path):
    """Add "response_db" (256 floats) to each JSONL record's "params"."""
    n = 0
    # utf-8-sig reads BOM and BOM-less files identically (Windows tools
    # sometimes prepend a BOM; plain utf-8 would reject the first record).
    with open(in_path, encoding="utf-8-sig") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["response_db"] = render_curve(rec["params"])
            fout.write(json.dumps(rec) + "\n")
            n += 1
    print(f"Rendered {n} curves -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true",
                    help="run unit tests and write results/stats/render_validation.txt")
    ap.add_argument("infile", nargs="?", help="input JSONL (records with 'params')")
    ap.add_argument("outfile", nargs="?", help="output JSONL (adds 'response_db')")
    args = ap.parse_args()

    if args.selftest:
        report, ok = run_selftests()
        print(report)
        OUT_VALIDATION.parent.mkdir(parents=True, exist_ok=True)
        OUT_VALIDATION.write_text(report + "\n", encoding="utf-8")
        print(f"Wrote {OUT_VALIDATION}")
        raise SystemExit(0 if ok else 1)

    if not (args.infile and args.outfile):
        ap.error("either --selftest or both infile and outfile are required")
    render_jsonl(args.infile, args.outfile)


if __name__ == "__main__":
    main()
