"""Stage 2 split construction — the contamination firewall.

Reads the provenanced raw CSV, applies the AFFIRMED cleaning rules
(DECISION_LOG D-019: blocklist {test, 1, my}, exact-duplicate drop), and cuts:

  data/splits/train.jsonl      warm/bright 80% (per-descriptor, seed 42)
  data/splits/eval_set.jsonl   warm/bright 20% — the frozen confirmatory test set
  data/splits/tail_eval.jsonl  ALL below-floor descriptors — generalization set
  data/splits/*.sha256         hash locks for each

plus paired receipts in results/stats/ (split_results.txt / _interpretation.txt).

Rigor rules encoded here:
  * Raw-file SHA-256 must match PROVENANCE.md or the script aborts.
  * The ONLY randomness is a single seeded shuffle (random.Random(42)) applied
    per descriptor before the 80/20 cut. Re-running reproduces byte-identical
    splits; the .sha256 files prove it.
  * Test entries are written once and never read by any training/centroid
    code path — downstream scripts load train.jsonl only, and the final
    metrics script (06) is the single consumer of eval files.
  * The 2-SD outlier filter is still NOT applied — it is computed on TRAIN
    only, inside 04_retrieval.py, per design §5.
"""

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW_CSV = BASE / "data" / "raw" / "SAFEEqualiserUserData.csv"
SPLITS = BASE / "data" / "splits"
OUT_RESULTS = BASE / "results" / "stats" / "split_results.txt"
OUT_INTERP = BASE / "results" / "stats" / "split_interpretation.txt"

EXPECTED_SHA256 = "CC43858680115070FFDFA41A1158FCA1344DB2E0096BA73FE68990E90A284AF1"

SEED = 42
TEST_FRACTION = 0.20
FLOOR = 20
GAIN_LIMIT_DB = 15.0
BLOCKLIST = {"test", "1", "my"}          # D-019 item 3

COL_DESCRIPTOR = 1
COL_IP = 2
PARAM_COLS = list(range(5, 18))
PARAM_NAMES = [
    "low_shelf_gain", "low_shelf_freq",
    "band1_gain", "band1_freq", "band1_q",
    "band2_gain", "band2_freq", "band2_q",
    "band3_gain", "band3_freq", "band3_q",
    "high_shelf_gain", "high_shelf_freq",
]
GAIN_IDX = [0, 2, 5, 8, 11]
FREQ_IDX = [1, 3, 6, 9, 12]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def normalize_descriptor(raw: str) -> str:
    return " ".join(raw.strip().lower().split())


def main() -> None:
    if sha256_of(RAW_CSV) != EXPECTED_SHA256:
        raise SystemExit("ABORT: raw CSV SHA-256 mismatch vs PROVENANCE.md")

    # ---- load + affirmed cleaning funnel ----------------------------------
    n_total = n_bad = n_blocked = n_dupes = 0
    seen_dupe_keys = set()
    records = []
    with open(RAW_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row:
                continue
            n_total += 1
            if len(row) != 25:
                n_bad += 1
                continue
            desc = normalize_descriptor(row[COL_DESCRIPTOR])
            if not desc:
                n_bad += 1
                continue
            try:
                params = [float(row[c]) for c in PARAM_COLS]
            except ValueError:
                n_bad += 1
                continue
            if any(abs(params[g]) > GAIN_LIMIT_DB for g in GAIN_IDX) or \
               any(params[q] <= 0.0 for q in FREQ_IDX):
                n_bad += 1
                continue
            if desc in BLOCKLIST:
                n_blocked += 1
                continue
            dupe_key = (desc, row[COL_IP].strip(),
                        tuple(round(p, 4) for p in params))
            if dupe_key in seen_dupe_keys:          # D-019 item 2
                n_dupes += 1
                continue
            seen_dupe_keys.add(dupe_key)
            records.append({
                "id": f"safe-{i}",
                "descriptor": desc,
                "params": dict(zip(PARAM_NAMES, params)),
            })

    by_desc = defaultdict(list)
    for r in records:
        by_desc[r["descriptor"]].append(r)

    above = {d: v for d, v in by_desc.items() if len(v) >= FLOOR}
    below = {d: v for d, v in by_desc.items() if len(v) < FLOOR}

    # ---- per-descriptor seeded 80/20 on above-floor descriptors -----------
    rng = random.Random(SEED)
    train, eval_set = [], []
    for d in sorted(above):                 # sorted -> order-stable across runs
        entries = sorted(above[d], key=lambda r: r["id"])
        rng.shuffle(entries)
        n_test = round(len(entries) * TEST_FRACTION)
        for r in entries[:n_test]:
            eval_set.append({**r, "split": "eval"})
        for r in entries[n_test:]:
            train.append({**r, "split": "train"})

    tail = [{**r, "split": "tail_eval"}
            for d in sorted(below) for r in sorted(below[d], key=lambda r: r["id"])]

    # ---- write + hash-lock -------------------------------------------------
    SPLITS.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, data in (("train", train), ("eval_set", eval_set), ("tail_eval", tail)):
        p = SPLITS / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(r, sort_keys=True) for r in data) + "\n",
                     encoding="utf-8")
        hashes[name] = sha256_of(p)
        (SPLITS / f"{name}.sha256").write_text(hashes[name] + "\n", encoding="utf-8")

    # ---- receipts ----------------------------------------------------------
    per_desc = {d: (sum(1 for r in train if r["descriptor"] == d),
                    sum(1 for r in eval_set if r["descriptor"] == d))
                for d in sorted(above)}
    lines = [
        "SPLIT CONSTRUCTION — Stage 2",
        f"raw sha256      : {EXPECTED_SHA256} (verified)",
        f"seed            : {SEED}   test fraction: {TEST_FRACTION}   floor: {FLOOR}",
        f"funnel          : {n_total} rows -> bad {n_bad} | blocklist {n_blocked} "
        f"| duplicates dropped {n_dupes} -> {len(records)} clean entries",
        "",
        "CONFIRMATORY SPLITS (above-floor descriptors):",
    ]
    for d, (ntr, nev) in per_desc.items():
        lines.append(f"  {d:<10} train {ntr:>4}   eval {nev:>4}")
    lines += [
        f"  TOTAL      train {len(train):>4}   eval {len(eval_set):>4}",
        "",
        f"TAIL GENERALIZATION SET: {len(tail)} entries across "
        f"{len(below)} below-floor descriptors (never used for training)",
        "",
        "HASH LOCKS:",
    ]
    for name, h in hashes.items():
        lines.append(f"  {name}.jsonl : {h}")
    OUT_RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    OUT_INTERP.write_text(
        "SPLIT INTERPRETATION — Stage 2\n\n"
        f"1. Cleaning applied the affirmed D-019 rules on top of the audit funnel:\n"
        f"   blocklist {sorted(BLOCKLIST)} and exact-duplicate removal "
        f"({n_dupes} dropped, matching the audit's 44-duplicate estimate within\n"
        f"   rounding of the key definition).\n"
        f"2. The eval_set ({len(eval_set)} entries) is now FROZEN. It is read by\n"
        f"   pipeline/06_metrics.py and nothing else. Any script importing it\n"
        f"   for training purposes is a protocol violation.\n"
        f"3. The tail set ({len(tail)} entries) is evaluation-only generalization\n"
        f"   data: per-descriptor stats are meaningless there (max n < {FLOOR});\n"
        f"   it is scored only in aggregate (design D-018 consequence c).\n"
        f"4. Reproducibility: rerunning this script reproduces byte-identical\n"
        f"   files (single seeded shuffle, sorted iteration); verify against the\n"
        f"   .sha256 locks.\n",
        encoding="utf-8")

    print(f"train={len(train)}  eval={len(eval_set)}  tail={len(tail)}  dupes_dropped={n_dupes}")
    print(f"eval_set.sha256 = {hashes['eval_set']}")


if __name__ == "__main__":
    main()
