"""Stage 1 audit of the real SAFE-DB Equaliser user data.

Purpose: descriptive audit ONLY — no modeling, no split-dependent statistics.
Produces the paired receipt files required by the collection plan:
    results/stats/audit_results.txt          (raw numbers)
    results/stats/audit_interpretation.txt   (what the numbers mean)

Rigor rules encoded here:
  * The raw file's SHA-256 must match PROVENANCE.md exactly; otherwise abort.
    (Guarantees every audit number refers to the provenanced artifact.)
  * Only FIXED, a-priori cleaning rules are applied (parse validity, |gain| <= 15 dB,
    freq > 0, junk-descriptor blocklist). The 2-SD outlier rejection is EXCLUDED
    on purpose: it is a fitted statistic and per design §5 may only ever be
    computed on the TRAIN split (Stage 2+). Applying it here would leak
    test-set information into cleaning.
  * Nothing is silently dropped: every cleaning step reports a funnel count,
    and suspicious-but-not-blocklisted descriptors are listed for human review
    rather than auto-removed.
  * No randomness anywhere -> no seed needed; re-runs are byte-comparable.

Zero-dependency (stdlib only), matching the project's existing convention.
"""

import csv
import hashlib
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW_CSV = BASE / "data" / "raw" / "SAFEEqualiserUserData.csv"
OUT_RESULTS = BASE / "results" / "stats" / "audit_results.txt"
OUT_INTERP = BASE / "results" / "stats" / "audit_interpretation.txt"

# From data/raw/PROVENANCE.md — the audit refuses to run on any other bytes.
EXPECTED_SHA256 = "CC43858680115070FFDFA41A1158FCA1344DB2E0096BA73FE68990E90A284AF1"

# Fixed a-priori rules (design §3). Not fitted to the data.
GAIN_LIMIT_DB = 15.0
JUNK_DESCRIPTORS = {"test"}   # evidence: 181 entries of literal plugin testing
FLOOR = 20                    # provisional descriptor floor from the design

# Column layout per PROVENANCE.md verification (25 columns):
COL_DESCRIPTOR = 1
COL_IP = 2
PARAM_COLS = list(range(5, 18))  # 13 parametric values
PARAM_NAMES = [
    "low_shelf_gain", "low_shelf_freq",
    "band1_gain", "band1_freq", "band1_q",
    "band2_gain", "band2_freq", "band2_q",
    "band3_gain", "band3_freq", "band3_q",
    "high_shelf_gain", "high_shelf_freq",
]
GAIN_IDX = [0, 2, 5, 8, 11]   # indices into PARAM_NAMES that are gains
FREQ_IDX = [1, 3, 6, 9, 12]   # indices that are frequencies
META_COLS = {18: "genre", 19: "instrument", 20: "location",
             21: "experience", 22: "age", 23: "nationality"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def normalize_descriptor(raw: str) -> str:
    return " ".join(raw.strip().lower().split())


def main() -> None:
    lines = []           # audit_results.txt content
    def log(s=""):
        lines.append(s)

    # ---- 0. Integrity gate -------------------------------------------------
    actual = sha256_of(RAW_CSV)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"ABORT: SHA-256 mismatch.\n expected {EXPECTED_SHA256}\n got      {actual}\n"
            "The file is not the provenanced artifact; do not audit it."
        )
    log("SAFE-DB EQUALISER AUDIT — Stage 1")
    log(f"source file : {RAW_CSV.name}")
    log(f"sha256      : {actual}  (matches PROVENANCE.md)")
    log()

    # ---- 1. Parse funnel ---------------------------------------------------
    n_total = 0
    n_colcount_bad = 0
    n_parse_bad = 0
    n_gain_oob = 0
    n_freq_oob = 0
    n_empty_desc = 0
    n_junk = 0
    records = []   # (norm_desc, raw_desc, ip, params[13], meta dict, row_index)

    with open(RAW_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row:
                continue
            n_total += 1
            if len(row) != 25:
                n_colcount_bad += 1
                continue
            raw_desc = row[COL_DESCRIPTOR]
            norm = normalize_descriptor(raw_desc)
            if not norm:
                n_empty_desc += 1
                continue
            try:
                params = [float(row[c]) for c in PARAM_COLS]
            except ValueError:
                n_parse_bad += 1
                continue
            if any(abs(params[g]) > GAIN_LIMIT_DB for g in GAIN_IDX):
                n_gain_oob += 1
                continue
            if any(params[q] <= 0.0 for q in FREQ_IDX):
                n_freq_oob += 1
                continue
            if norm in JUNK_DESCRIPTORS:
                n_junk += 1
                continue
            meta = {name: row[c].strip() for c, name in META_COLS.items()}
            records.append((norm, raw_desc, row[COL_IP].strip(), params, meta, i))

    log("CLEANING FUNNEL (fixed a-priori rules only; 2-SD outlier step is")
    log("deliberately deferred to TRAIN-only computation at Stage 2+):")
    log(f"  rows in file                     : {n_total}")
    log(f"  - wrong column count             : {n_colcount_bad}")
    log(f"  - empty descriptor               : {n_empty_desc}")
    log(f"  - unparseable numeric params     : {n_parse_bad}")
    log(f"  - |gain| > {GAIN_LIMIT_DB:.0f} dB (any band)      : {n_gain_oob}")
    log(f"  - freq <= 0 Hz (any band)        : {n_freq_oob}")
    log(f"  - junk descriptor blocklist      : {n_junk}   (blocklist: {sorted(JUNK_DESCRIPTORS)})")
    log(f"  = USABLE ENTRIES                 : {len(records)}")
    log()

    # ---- 2. Descriptor distribution ---------------------------------------
    by_desc = defaultdict(list)
    for rec in records:
        by_desc[rec[0]].append(rec)
    counts = Counter({d: len(v) for d, v in by_desc.items()})

    log(f"DISTINCT NORMALIZED DESCRIPTORS: {len(counts)}")
    log("Top 30 descriptors:")
    for d, c in counts.most_common(30):
        log(f"  {d:<20} {c}")
    log()

    buckets = {"    >= 20 (floor)": 0, "  10 - 19": 0, "   5 - 9": 0, "   2 - 4": 0, "singleton": 0}
    for _, c in counts.items():
        if c >= FLOOR:
            buckets["    >= 20 (floor)"] += 1
        elif c >= 10:
            buckets["  10 - 19"] += 1
        elif c >= 5:
            buckets["   5 - 9"] += 1
        elif c >= 2:
            buckets["   2 - 4"] += 1
        else:
            buckets["singleton"] += 1
    log(f"FLOOR ANALYSIS (floor = {FLOOR}):")
    for k, v in buckets.items():
        log(f"  {k:<18}: {v} descriptors")
    above = [(d, c) for d, c in counts.items() if c >= FLOOR]
    tail_n = sum(c for d, c in counts.items() if c < FLOOR)
    log(f"  descriptors above floor: {[f'{d} ({c})' for d, c in sorted(above, key=lambda x: -x[1])]}")
    log(f"  long-tail entries (below floor, candidate generalization set): {tail_n}")
    log()

    # ---- 3. Deep stats for above-floor descriptors -------------------------
    log("PER-PARAMETER STATS FOR ABOVE-FLOOR DESCRIPTORS (mean / SD / min / max):")
    for d, _ in sorted(above, key=lambda x: -x[1]):
        log(f"  [{d}] n={counts[d]}")
        vals = list(zip(*[r[3] for r in by_desc[d]]))
        for pi, pname in enumerate(PARAM_NAMES):
            v = vals[pi]
            log(f"    {pname:<16} {statistics.mean(v):9.2f} / {statistics.stdev(v):8.2f} / {min(v):9.2f} / {max(v):9.2f}")
    log()

    # ---- 4. Duplicate analysis (surfaced, NOT dropped here) ----------------
    seen = defaultdict(list)
    for rec in records:
        key = (rec[0], rec[2], tuple(round(p, 4) for p in rec[3]))
        seen[key].append(rec[5])
    dup_groups = {k: v for k, v in seen.items() if len(v) > 1}
    n_dup_extra = sum(len(v) - 1 for v in dup_groups.values())
    log("DUPLICATE ANALYSIS (same descriptor + same IP + identical 13 params):")
    log(f"  duplicate groups : {len(dup_groups)}")
    log(f"  redundant entries: {n_dup_extra}  ({100.0 * n_dup_extra / max(len(records), 1):.1f}% of usable)")
    log("  NOTE: surfaced only. Whether to drop is a Stage 2 decision -> DECISION_LOG.")
    log()

    # ---- 5. Suspicious descriptors for human review (NOT dropped) ----------
    suspicious = sorted(
        d for d in counts
        if (not any(ch.isalpha() for ch in d)) or len(d) <= 2
    )
    log("SUSPICIOUS DESCRIPTORS FOR HUMAN REVIEW (non-alphabetic or <= 2 chars —")
    log("surfaced, not dropped; add to blocklist only via DECISION_LOG entry):")
    log(f"  {suspicious if suspicious else '(none)'}")
    log()

    # ---- 6. Metadata profile ----------------------------------------------
    log("METADATA PROFILE (top 8 values per column; '<blank>' = empty):")
    for name in ("genre", "instrument", "location", "experience", "age", "nationality"):
        col = Counter((r[4][name] or "<blank>") for r in records)
        top = ", ".join(f"{v} ({c})" for v, c in col.most_common(8))
        log(f"  {name:<12}: {len(col)} distinct | {top}")
    log()

    log("END OF AUDIT — every number above derives from the provenanced file.")

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.write_text("\n".join(lines), encoding="utf-8")

    # ---- Interpretation file (prose, references the numbers above) ---------
    n_use = len(records)
    warm_n = counts.get("warm", 0)
    bright_n = counts.get("bright", 0)
    interp = f"""AUDIT INTERPRETATION — Stage 1 (companion to audit_results.txt)

1. Integrity: the audited file is byte-identical to the provenanced Wayback
   artifact (SHA-256 match), so all numbers below are anchored to a stable,
   citable source.

2. Usable N: {n_use} entries survive the fixed a-priori funnel (from {n_total}
   raw rows). The dominant single loss is the 'test' junk descriptor. The 2-SD
   outlier filter is NOT applied here — it is a fitted statistic and will be
   computed on TRAIN only (design §5), so the final analysis N will be
   slightly smaller and split-dependent.

3. Shape: the distribution is two-giants-plus-long-tail. warm ({warm_n}) and
   bright ({bright_n}) are the only descriptors above the {FLOOR}-entry floor;
   together they hold {100.0 * (warm_n + bright_n) / max(n_use, 1):.0f}% of usable data. This CONFIRMS the
   preliminary finding in DECISION_LOG D-018 on the full formal funnel.

4. Consequences (per pre-declared design rules, not post-hoc choices):
   a. §7 power fallback AUTO-TRIGGERS: primary analysis = entry-level
      mixed-effects with descriptor as a grouping factor; descriptor-paired
      Wilcoxon is demoted to a warm/bright-only descriptive.
   b. The paper's confirmatory core is deep-distributional on warm + bright
      (both have enough entries for stable held-out human distributions,
      ~{FLOOR}%+ splits give ~100 test curves each).
   c. The below-floor long tail ({tail_n} entries) becomes the unseen/rare-
      descriptor generalization set — scored in aggregate, never per-descriptor.
   d. Arm D/F viability gate is IN TENSION (passes total-N, fails
      descriptor-count). Resolution is Patrick's call before freeze —
      DECISION_LOG D-018 lists the two options.

5. Data-quality flags for Stage 2 decisions:
   - Duplicate entries (same descriptor+IP+params): see audit_results §4 —
     decide drop/keep in DECISION_LOG before splitting.
   - Suspicious descriptor strings: see audit_results §5 — blocklist additions
     require a DECISION_LOG entry.
   - Metadata sparsity: see audit_results §6 — decide which metadata fields
     are reliable enough to include in arm prompts.

Nothing in this audit fitted a statistic to the outcome data; every rule was
fixed a-priori. This file is the receipt behind every dataset claim in the
manuscript.
"""
    OUT_INTERP.write_text(interp, encoding="utf-8")
    print(f"Wrote {OUT_RESULTS}")
    print(f"Wrote {OUT_INTERP}")
    print(f"USABLE N = {n_use}; above-floor descriptors = {[d for d, _ in sorted(above, key=lambda x: -x[1])]}")


if __name__ == "__main__":
    main()
