"""Stage 7 — inferential statistics battery + cost log.

Consumes results/metrics.jsonl (Stage 6 output) and results/receipts/calls.jsonl
(for the cost log). Never touches raw splits. Emits:
    results/stats/stats_results.txt        full battery: raw p, BH-FDR q, survives
    results/stats/stats_interpretation.txt what it means, incl. casualties
    results/stats/cost_log.txt             actual spend from receipts x D-022/25 prices

Battery (design v1.0 §7, adapted per D-018 fallback: entry/unit-level analysis,
descriptor as stratum — the descriptor-paired Wilcoxon died with the 2-descriptor
floor and is replaced by unit-paired contrasts + strata reporting):
  T1  B vs C on HRP           paired Wilcoxon signed-rank; pairs = identical
                              (model, prompt, descriptor, sample_idx) cells
  T2  B vs C on centrality    same pairing, LSD instead of HRP
  T3  B vs C on energy dist   paired by (model, prompt, descriptor) group
  T4  B vs C on tail LSD      paired by (model, descriptor)
  T5  RQ4 family divergence   Kruskal-Wallis on B HRP across models
  T6  prompt sensitivity      Kruskal-Wallis on B HRP across prompt variants
All p-values BH-FDR corrected at alpha=.05 across the full battery.
Effect sizes: Cliff's delta for pairwise contrasts; bootstrap (10k, seeded)
95% CIs on per-arm median HRP. Human split-half energy distance computed as
the interpretive reference for T3 magnitudes.
"""

import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from scipy import stats as sps

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
OUT = RESULTS / "stats" / "stats_results.txt"
OUT_I = RESULTS / "stats" / "stats_interpretation.txt"
OUT_COST = RESULTS / "stats" / "cost_log.txt"

ALPHA = 0.05
SEED = 42
N_BOOT = 10_000

PRICES = {  # $/M tokens (in, out) — receipted in D-022/D-023/D-025
    "openai/gpt-5.4-mini": (0.75, 4.5), "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "google/gemini-3.5-flash": (1.5, 9.0), "deepseek/deepseek-v4-flash": (0.098, 0.196),
    "qwen/qwen3-235b-a22b-2507": (0.09, 0.55), "z-ai/glm-4.7-flash": (0.0605, 0.4),
    "openai/gpt-5.5": (5.0, 30.0), "anthropic/claude-opus-4.8": (5.0, 25.0),
}


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def cliffs_delta(a: list, b: list) -> float:
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b)) if a and b else float("nan")


def boot_ci_median(vals: list, rng: random.Random) -> tuple:
    meds = []
    for _ in range(N_BOOT):
        meds.append(statistics.median(rng.choices(vals, k=len(vals))))
    meds.sort()
    return meds[int(0.025 * N_BOOT)], meds[int(0.975 * N_BOOT)]


def main() -> None:
    rows = load_jsonl(RESULTS / "metrics.jsonl")
    metric_rows = [r for r in rows if r.get("row_type") != "energy"]
    energy_rows = [r for r in rows if r.get("row_type") == "energy"]

    tests = []   # (id, description, p, extra)

    # ---- T1/T2 family: arm-pair contrasts on HRP and centrality ------------
    # Paired Wilcoxon on identical (model, prompt, descriptor, sample) cells
    # when the two arms share models; Mann-Whitney otherwise (D/F are local
    # models with no API counterpart cell).
    def cell_key(r):
        return (r["model"], r["prompt"], r["descriptor"], r["sample_idx"])
    units_by_arm = defaultdict(dict)
    for r in metric_rows:
        if "hrp" in r:
            units_by_arm[r["arm"]][cell_key(r)] = r
    ARM_PAIRS = [("B", "C"), ("B", "C_AGG"), ("C", "C_AGG"),
                 ("B", "D"), ("C", "F"), ("D", "F")]
    tno = 0
    for a1, a2 in ARM_PAIRS:
        u1, u2 = units_by_arm.get(a1, {}), units_by_arm.get(a2, {})
        if not u1 or not u2:
            continue
        common = sorted(set(u1) & set(u2))
        tno += 1
        if len(common) >= 10:
            h1 = [u1[k]["hrp"] for k in common]
            h2 = [u2[k]["hrp"] for k in common]
            stat = sps.wilcoxon(h1, h2)
            kind = f"paired, n={len(common)}"
        elif len(u1) >= 10 and len(u2) >= 10:
            h1 = [r["hrp"] for r in u1.values()]
            h2 = [r["hrp"] for r in u2.values()]
            stat = sps.mannwhitneyu(h1, h2, alternative="two-sided")
            kind = f"unpaired MW, n={len(h1)}/{len(h2)}"
        else:
            continue
        tests.append((f"T1.{tno}", f"{a1} vs {a2} HRP ({kind})", stat.pvalue,
                      f"median {a1}={statistics.median(h1):.1f} "
                      f"{a2}={statistics.median(h2):.1f} "
                      f"delta={cliffs_delta(h2, h1):+.3f}"))
    # T2: centrality LSD for the primary B-vs-C pair only (kept from v1 battery)
    u1, u2 = units_by_arm.get("B", {}), units_by_arm.get("C", {})
    common = sorted(set(u1) & set(u2))
    if len(common) >= 10:
        b_lsd = [u1[k]["centrality_lsd"] for k in common]
        c_lsd = [u2[k]["centrality_lsd"] for k in common]
        stat = sps.wilcoxon(b_lsd, c_lsd)
        tests.append(("T2", f"B vs C centrality LSD (paired, n={len(common)})", stat.pvalue,
                      f"median B={statistics.median(b_lsd):.3f} C={statistics.median(c_lsd):.3f} dB"))

    # ---- T3: B vs C energy distance ---------------------------------------
    e_b = {(r["model"], r["prompt"], r["descriptor"]): r["energy_dist_db"]
           for r in energy_rows if r["arm"] == "B"}
    e_c = {(r["model"], r["prompt"], r["descriptor"]): r["energy_dist_db"]
           for r in energy_rows if r["arm"] == "C"}
    e_common = sorted(set(e_b) & set(e_c))
    if len(e_common) >= 6:
        bb, cc = [e_b[k] for k in e_common], [e_c[k] for k in e_common]
        stat = sps.wilcoxon(bb, cc)
        tests.append(("T3", f"B vs C energy distance (paired groups, n={len(e_common)})",
                      stat.pvalue,
                      f"median B={statistics.median(bb):.3f} C={statistics.median(cc):.3f} dB"))

    # ---- T4: tail generalization B vs C -----------------------------------
    t_b = {(r["model"], r["descriptor"]): r["tail_lsd"]
           for r in metric_rows if r["arm"] == "B" and "tail_lsd" in r}
    t_c = {(r["model"], r["descriptor"]): r["tail_lsd"]
           for r in metric_rows if r["arm"] == "C" and "tail_lsd" in r}
    t_common = sorted(set(t_b) & set(t_c))
    if len(t_common) >= 10:
        bb, cc = [t_b[k] for k in t_common], [t_c[k] for k in t_common]
        stat = sps.wilcoxon(bb, cc)
        tests.append(("T4", f"B vs C tail LSD (paired, n={len(t_common)})", stat.pvalue,
                      f"median B={statistics.median(bb):.3f} C={statistics.median(cc):.3f} dB "
                      f"delta={cliffs_delta(cc, bb):+.3f}"))

    # ---- T5: family divergence on B HRP -----------------------------------
    by_model = defaultdict(list)
    for r in metric_rows:
        if r["arm"] == "B" and "hrp" in r:
            by_model[r["model"]].append(r["hrp"])
    groups = [v for v in by_model.values() if len(v) >= 10]
    if len(groups) >= 3:
        stat = sps.kruskal(*groups)
        tests.append(("T5", f"family divergence, B HRP ({len(groups)} models)", stat.pvalue,
                      " | ".join(f"{m.split('/')[-1]}:{statistics.median(v):.1f}"
                                 for m, v in sorted(by_model.items()) if len(v) >= 10)))

    # ---- T6: prompt sensitivity on B HRP ----------------------------------
    by_prompt = defaultdict(list)
    for r in metric_rows:
        if r["arm"] == "B" and "hrp" in r:
            by_prompt[r["prompt"]].append(r["hrp"])
    pgroups = [v for v in by_prompt.values() if len(v) >= 10]
    if len(pgroups) >= 2:
        stat = sps.kruskal(*pgroups)
        tests.append(("T6", f"prompt sensitivity, B HRP ({len(pgroups)} variants)", stat.pvalue,
                      " | ".join(f"{p}:{statistics.median(v):.1f}"
                                 for p, v in sorted(by_prompt.items()))))

    # ---- BH-FDR ------------------------------------------------------------
    ps = sorted(range(len(tests)), key=lambda i: tests[i][2])
    m = len(tests)
    qs = [None] * m
    prev = 1.0
    for rank_from_top in range(m - 1, -1, -1):
        i = ps[rank_from_top]
        q = min(prev, tests[i][2] * m / (rank_from_top + 1))
        qs[i] = q
        prev = q

    # ---- descriptive block -------------------------------------------------
    rng = random.Random(SEED)
    desc_lines = ["PER-ARM HRP (median [bootstrap 95% CI]):"]
    for arm in sorted({r["arm"] for r in metric_rows}):
        v = [r["hrp"] for r in metric_rows if r["arm"] == arm and "hrp" in r]
        if len(v) >= 3:
            lo, hi = boot_ci_median(v, rng)
            desc_lines.append(f"  {arm:<3} n={len(v):>4} median {statistics.median(v):6.1f} [{lo:.1f}, {hi:.1f}]")
        elif v:
            desc_lines.append(f"  {arm:<3} n={len(v):>4} median {statistics.median(v):6.1f} (too few for CI)")

    # Human split-half energy reference
    ref_lines = []
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("m06", BASE / "pipeline" / "06_metrics.py")
        m06 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m06)
        for d in ("warm", "bright"):
            cloud = [m06.render_curve(r["params"]) for r in m06.verify_split("eval_set")
                     if r["descriptor"] == d]
            rng2 = random.Random(SEED)
            rng2.shuffle(cloud)
            half = len(cloud) // 2
            e = m06.energy_components(cloud[:half], cloud[half:])["energy_dist_db"]
            ref_lines.append(f"  human split-half energy [{d}]: {e:.3f} dB")
    except Exception as ex:
        ref_lines.append(f"  (reference computation failed: {ex!r})")

    lines = ["STATS BATTERY — Stage 7", "",
             *desc_lines, "",
             "HUMAN ENERGY-DISTANCE REFERENCE (self-match floor for T3):",
             *ref_lines, "",
             f"TEST TABLE (BH-FDR alpha={ALPHA}, m={m} tests — ALL tests shown):",
             f"{'id':<4}{'test':<50}{'raw p':<12}{'q':<12}{'survives':<9}detail"]
    for i, (tid, desc, p, extra) in enumerate(tests):
        lines.append(f"{tid:<4}{desc:<50}{p:<12.3e}{qs[i]:<12.3e}"
                     f"{'YES' if qs[i] <= ALPHA else 'no':<9}{extra}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- cost log ----------------------------------------------------------
    tok = defaultdict(lambda: [0, 0, 0])   # model -> [in, out, calls]
    for r in load_jsonl(RESULTS / "receipts" / "calls.jsonl"):
        u = r.get("usage") or {}
        model = r.get("model", "?")
        tok[model][0] += u.get("prompt_tokens", 0) or 0
        tok[model][1] += u.get("completion_tokens", 0) or 0
        tok[model][2] += 1
    cost_lines = ["COST LOG (from receipts; includes retries and failed attempts)", ""]
    total = 0.0
    for model, (tin, tout, calls) in sorted(tok.items()):
        pin, pout = PRICES.get(model, (0.0, 0.0))
        c = tin / 1e6 * pin + tout / 1e6 * pout
        total += c
        cost_lines.append(f"  {model:<38} calls={calls:>5} in={tin:>9} out={tout:>9} ${c:.3f}")
    cost_lines += ["", f"  TOTAL ACTUAL SPEND: ${total:.2f}"]
    OUT_COST.write_text("\n".join(cost_lines) + "\n", encoding="utf-8")

    OUT_I.write_text(
        "STATS INTERPRETATION — Stage 7\n\n"
        "1. Unit-level analysis with descriptor as stratum is the pre-declared\n"
        "   D-018 fallback (2 descriptors killed descriptor-paired tests).\n"
        "2. The full test table is reported including non-survivors; a null on\n"
        "   T6 (prompt sensitivity) is EXPECTED and evidence of robustness.\n"
        "3. T3 magnitudes are interpretable only against the human split-half\n"
        "   reference: an arm whose energy distance approaches the human\n"
        "   self-match floor matches the human distribution about as well as\n"
        "   humans match themselves.\n"
        "4. HRP CIs are bootstrap-on-units; units within a model share draws,\n"
        "   so CIs are anti-conservative to that extent — stated in methods.\n"
        "5. Claim discipline: Arm A/A' HRP is reported as context (consensus\n"
        "   anchor), never as a contest winner (D-005).\n",
        encoding="utf-8")
    print(f"tests={m} -> {OUT.name}; total spend ${total:.2f} -> {OUT_COST.name}")


if __name__ == "__main__":
    main()
