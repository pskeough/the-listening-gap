"""Stage 12 — consolidated BH-FDR over the ENTIRE study's test battery.

Collects every raw p-value from the stage receipts (7: core battery,
10: school-assignment GOFs, 11: method-selection pairwise) into ONE
Benjamini-Hochberg correction, so the manuscript's alpha=.05 claims are
corrected over everything that was actually run — not per-stage.

Extraction is from the receipt files (each parsed value remains traceable to
its receipt line); the full table is emitted with source, raw p, q, survives.
Data collection is CLOSED as of this stage (D-027): re-running after the
D/F arms land appends their tests and re-corrects; nothing else may be added.
"""

import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
STATS = BASE / "results" / "stats"
OUT = STATS / "consolidated_fdr_results.txt"

ALPHA = 0.05

SOURCES = {
    "stats_results.txt": re.compile(
        r"^(T\S+)\s+(.+?)\s{2,}(\d\.\d{3}e[+-]\d{2})", re.M),
    "mode_analysis_results.txt": re.compile(
        r"^\s{2}\[(\S+)\] (\S+)\s+(\S+)\s+n=\s*\d+\s+modes \S+\s+\(human [^)]*\)\s+chi2 p=(\d\.\d{2}e[+-]\d{2})", re.M),
    "method_selection_results.txt": re.compile(
        r"^\s{4}(\S+ vs \S+)\s+p=(\d\.\d{3}e[+-]\d{2})", re.M),
}


def main() -> None:
    tests = []
    txt = (STATS / "stats_results.txt").read_text(encoding="utf-8")
    for m in SOURCES["stats_results.txt"].finditer(txt):
        tests.append((f"S7:{m.group(1)}", m.group(2).strip(), float(m.group(3))))
    txt = (STATS / "mode_analysis_results.txt").read_text(encoding="utf-8")
    for m in SOURCES["mode_analysis_results.txt"].finditer(txt):
        tests.append((f"S10:{m.group(1)}", f"school GOF {m.group(2)} {m.group(3)}",
                      float(m.group(4))))
    txt = (STATS / "method_selection_results.txt").read_text(encoding="utf-8")
    for m in SOURCES["method_selection_results.txt"].finditer(txt):
        tests.append((f"S11", f"tail {m.group(1)}", float(m.group(2))))

    m_total = len(tests)
    order = sorted(range(m_total), key=lambda i: tests[i][2])
    qs = [None] * m_total
    prev = 1.0
    for rank_from_top in range(m_total - 1, -1, -1):
        i = order[rank_from_top]
        q = min(prev, tests[i][2] * m_total / (rank_from_top + 1))
        qs[i] = q
        prev = q

    n_surv = sum(1 for q in qs if q <= ALPHA)
    lines = [f"CONSOLIDATED BH-FDR — {m_total} tests across stages 7/10/11 "
             f"(alpha={ALPHA}); {n_surv} survive",
             "", f"{'source':<12}{'test':<52}{'raw p':<12}{'q':<12}survives"]
    for i in sorted(range(m_total), key=lambda i: tests[i][2]):
        src, name, p = tests[i]
        lines.append(f"{src:<12}{name[:50]:<52}{p:<12.3e}{qs[i]:<12.3e}"
                     f"{'YES' if qs[i] <= ALPHA else 'no'}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"consolidated: {m_total} tests, {n_surv} survive BH-FDR at {ALPHA}")
    for i in sorted(range(m_total), key=lambda i: tests[i][2]):
        src, name, p = tests[i]
        if qs[i] > ALPHA:
            print(f"  DIED: {src} {name} (raw p={p:.3e}, q={qs[i]:.3e})")


if __name__ == "__main__":
    main()
