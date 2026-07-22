"""Stage 11 — method selection: which system should the app actually use?

The head-to-head Patrick's product decision needs: deployed keyword matcher
(A'), embedding retrieval (A), tiny MLP (E), LLM zero-shot (B), LLM+RAG (C),
LLM+aggregate (C_AGG), against the FLAT null baseline (do-nothing EQ — which
is what the deployed system emits on 96% of real descriptors).

Reads results/metrics.jsonl ONLY (Stage 6 remains the sole eval reader).

  A. KNOWN-DESCRIPTOR TABLE: eval-descriptor centrality/HRP summary per arm.
  B. TAIL TABLE: median tail LSD per arm; for LLM arms, per-descriptor median
     of greedy P1 outputs across models (one value per descriptor per arm) so
     everything pairs on the same 60-descriptor sample.
  C. WIN MATRIX + paired Wilcoxon on the common tail descriptors (p-values
     feed the consolidated FDR battery at write-up).
  D. PROXIMITY GRADIENT: per-arm Spearman correlation between a tail
     descriptor's semantic proximity to the anchors (max cosine weight from
     arm_a.jsonl) and that arm's LSD — does each method degrade as
     descriptors leave the corpus's semantic neighborhood?
"""

import json
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scipy import stats as sps

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
OUT = RESULTS / "stats" / "method_selection_results.txt"
OUT_I = RESULTS / "stats" / "method_selection_interpretation.txt"


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    rows = [r for r in load_jsonl(RESULTS / "metrics.jsonl") if r.get("row_type") != "energy"]
    lines = ["METHOD SELECTION — Stage 11", ""]

    # ---- A. known-descriptor table -----------------------------------------
    lines.append("A. KNOWN DESCRIPTORS (eval split; centrality LSD dB / HRP):")
    for arm in ("A", "A'", "E", "FLAT", "B", "C", "C_AGG"):
        v = [(r["centrality_lsd"], r["hrp"]) for r in rows
             if r["arm"] == arm and "centrality_lsd" in r]
        if v:
            lsd = statistics.median(x[0] for x in v)
            hrp = statistics.median(x[1] for x in v)
            lines.append(f"  {arm:<6} n={len(v):>4}  LSD {lsd:6.3f}  HRP {hrp:6.1f}")
    lines.append("")

    # ---- B. tail per-arm values (one per descriptor per arm) ---------------
    tail_vals = defaultdict(dict)     # arm -> desc -> lsd
    for r in rows:
        if "tail_lsd" not in r:
            continue
        arm, d = r["arm"], r["descriptor"]
        if arm in ("A", "A'", "E", "FLAT"):
            tail_vals[arm][d] = r["tail_lsd"]
        elif r["sample_idx"] == 0 and r["prompt"] in ("p1_minimal", "n/a"):
            tail_vals[arm].setdefault(d, []).append(r["tail_lsd"])
    for arm in ("B", "C", "C_AGG"):
        tail_vals[arm] = {d: statistics.median(v) for d, v in tail_vals[arm].items()
                          if isinstance(v, list)}

    lines.append("B. TAIL (unseen descriptors; median LSD dB; lower = closer to humans):")
    for arm, dv in sorted(tail_vals.items(), key=lambda kv: statistics.median(kv[1].values())):
        lines.append(f"  {arm:<6} n_desc={len(dv):>3}  median LSD {statistics.median(dv.values()):6.3f}")
    lines.append("")

    # ---- C. head-to-head on common tail descriptors ------------------------
    common = set.intersection(*(set(dv) for dv in tail_vals.values()))
    lines.append(f"C. HEAD-TO-HEAD on {len(common)} common tail descriptors "
                 "(win % = row beats column; paired Wilcoxon p):")
    arms_order = ["FLAT", "A'", "A", "E", "B", "C", "C_AGG"]
    arms_present = [a for a in arms_order if a in tail_vals]
    header = "        " + "".join(f"{a:>10}" for a in arms_present)
    lines.append(header)
    pvals = []
    for a1 in arms_present:
        cells = []
        for a2 in arms_present:
            if a1 == a2:
                cells.append(f"{'--':>10}")
                continue
            wins = sum(1 for d in common if tail_vals[a1][d] < tail_vals[a2][d])
            cells.append(f"{100 * wins / len(common):>9.0f}%")
        lines.append(f"{a1:<8}" + "".join(cells))
    lines.append("")
    for a1, a2 in combinations(arms_present, 2):
        x = [tail_vals[a1][d] for d in sorted(common)]
        y = [tail_vals[a2][d] for d in sorted(common)]
        try:
            p = sps.wilcoxon(x, y).pvalue
        except ValueError:
            p = float("nan")
        pvals.append((f"{a1} vs {a2}", p,
                      statistics.median(x) - statistics.median(y)))
    lines.append("  pairwise Wilcoxon (raw p — feeds consolidated FDR at write-up):")
    for name, p, dmed in sorted(pvals, key=lambda t: t[1]):
        lines.append(f"    {name:<16} p={p:.3e}  median diff {dmed:+.3f} dB")
    lines.append("")

    # ---- D. proximity gradient ---------------------------------------------
    prox = {}
    for r in load_jsonl(RESULTS / "arm_a.jsonl"):
        if r["target_split"] == "tail_eval" and r.get("weights"):
            prox[r["descriptor"]] = max(r["weights"].values())
    lines.append("D. SEMANTIC-PROXIMITY GRADIENT (Spearman rho: proximity-to-anchors vs LSD):")
    for arm in arms_present:
        pairs = [(prox[d], tail_vals[arm][d]) for d in tail_vals[arm] if d in prox]
        if len(pairs) >= 20:
            rho, p = sps.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            lines.append(f"  {arm:<6} rho={rho:+.3f} (p={p:.2e}, n={len(pairs)})"
                         f"{'  <- degrades with distance' if rho < -0.15 else ''}")
    lines.append("")
    lines.append("NOTE: negative rho = arm does BETTER near the anchors (higher")
    lines.append("proximity -> lower LSD): corpus-bound knowledge. rho ~ 0 with a")
    lines.append("competitive median = knowledge independent of the corpus's")
    lines.append("semantic neighborhood.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_I.write_text(
        "METHOD SELECTION INTERPRETATION — Stage 11\n\n"
        "1. Section B/C answer the product question directly: the app should\n"
        "   ship whichever method wins the tail at acceptable runtime cost,\n"
        "   with FLAT as the 'is it better than nothing' gate.\n"
        "2. All pairwise p-values here are RAW; they join the consolidated\n"
        "   BH-FDR battery before any enters the manuscript.\n"
        "3. LLM tail values use greedy P1 medians across models — the cheap\n"
        "   deployable configuration, not best-case sampling.\n",
        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
