"""Stage 6 — metric computation. THE single consumer of the frozen eval sets.

Reads every arm's predictions plus the hash-verified eval/tail splits, renders
all curves through the validated Stage 3 instrument, and emits one metrics row
per prediction unit to results/metrics.jsonl, plus paired receipts.

Metrics (design v1.0 §6):
  * LSD          log-spectral distance: RMSE in dB between two rendered
                 256-point responses (the primary distance).
  * centrality   D(p, t) = median LSD from prediction p to the held-out human
                 curves of descriptor t (eval split only).
  * HRP          Human-Relative Percentile: percentile of D(p, t) within the
                 leave-one-out human-to-human centrality distribution {D(h_i)}.
                 ~50 = as central as a typical human; >95 = outside consensus.
  * energy_dist  distribution coverage for sampled arms (>= 2 cloud samples):
                 energy distance between the sample cloud and the human cloud
                 in rendered space (per-point distances normalized to dB via
                 /sqrt(256)); point-mass arms get n/a here by construction.
  * sign_agree / mag_err   per-band direction agreement vs TRAIN consensus
                 sign, and |gain - TRAIN consensus gain| (dB) — the derived-
                 binary + continuous decomposition (D-011).
  * tail_lsd     tail generalization: median LSD to the (1-4) human curves of
                 that tail descriptor; aggregate stats only ever reported
                 across descriptors (D-018.c).

Rigor: split hashes re-verified; renderer is imported from the validated
module (no reimplementation); every row carries enough keys to trace back to
its receipt.
"""

import hashlib
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
RESULTS = BASE / "results"
OUT_METRICS = RESULTS / "metrics.jsonl"
OUT_RESULTS = RESULTS / "stats" / "metrics_results.txt"
OUT_INTERP = RESULTS / "stats" / "metrics_interpretation.txt"

_spec = importlib.util.spec_from_file_location("render", BASE / "pipeline" / "03_render.py")
_render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_render)
render_curve = _render.render_curve

GAIN_KEYS = ["low_shelf_gain", "band1_gain", "band2_gain", "band3_gain", "high_shelf_gain"]
N_GRID_SQRT = 16.0   # sqrt(256): converts euclidean distance on the grid to dB RMSE


def verify_split(name: str) -> list:
    p = SPLITS / f"{name}.jsonl"
    expected = (SPLITS / f"{name}.sha256").read_text(encoding="utf-8-sig").strip()
    if hashlib.sha256(p.read_bytes()).hexdigest().upper() != expected:
        raise SystemExit(f"ABORT: {name}.jsonl hash mismatch vs lock")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def load_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def lsd(a: list, b: list) -> float:
    return (sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)) ** 0.5


def median_lsd_to(curve: list, cloud: list) -> float:
    return statistics.median(lsd(curve, h) for h in cloud)


def energy_components(cloud_a: list, cloud_b: list) -> dict:
    """Energy distance 2E|X-Y| - E|X-X'| - E|Y-Y'| (dB-normalized euclidean),
    plus the dispersion components: aa = model-cloud internal spread,
    bb = human-cloud internal spread. dispersion_ratio = aa/bb — the D-026
    under/over-dispersion diagnostic (ratio << 1: the model produces
    near-identical curves where humans genuinely vary)."""
    def euc(x, y):
        return (sum((p - q) ** 2 for p, q in zip(x, y)) ** 0.5) / N_GRID_SQRT
    ab = statistics.mean(euc(x, y) for x in cloud_a for y in cloud_b)
    aa = statistics.mean(euc(x, y) for i, x in enumerate(cloud_a)
                         for j, y in enumerate(cloud_a) if i < j) if len(cloud_a) > 1 else 0.0
    bb = statistics.mean(euc(x, y) for i, x in enumerate(cloud_b)
                         for j, y in enumerate(cloud_b) if i < j) if len(cloud_b) > 1 else 0.0
    return {"energy_dist_db": round(2 * ab - aa - bb, 4),
            "cloud_dispersion_db": round(aa, 4),
            "human_dispersion_db": round(bb, 4),
            "dispersion_ratio": round(aa / bb, 4) if bb > 0 else None}


def main() -> None:
    train = verify_split("train")
    eval_set = verify_split("eval_set")
    tail = verify_split("tail_eval")

    # Rendered human clouds
    eval_cloud = defaultdict(list)
    for r in eval_set:
        eval_cloud[r["descriptor"]].append(render_curve(r["params"]))
    tail_cloud = defaultdict(list)
    for r in tail:
        tail_cloud[r["descriptor"]].append(render_curve(r["params"]))

    # TRAIN consensus gains (sign/magnitude reference; TRAIN only)
    train_gains = defaultdict(lambda: defaultdict(list))
    for r in train:
        for k in GAIN_KEYS:
            train_gains[r["descriptor"]][k].append(r["params"][k])
    consensus = {d: {k: statistics.mean(v) for k, v in gains.items()}
                 for d, gains in train_gains.items()}

    # Human-to-human leave-one-out centrality distributions (per eval descriptor)
    hrp_ref = {}
    for d, cloud in eval_cloud.items():
        ds = []
        for i, h in enumerate(cloud):
            others = cloud[:i] + cloud[i + 1:]
            ds.append(median_lsd_to(h, others))
        hrp_ref[d] = sorted(ds)

    def hrp_of(dist: float, d: str) -> float:
        ref = hrp_ref[d]
        below = sum(1 for x in ref if x < dist)
        return 100.0 * below / len(ref)

    # ---- gather prediction units ------------------------------------------
    units = []   # (arm, model, prompt, descriptor, sample_idx, target_split, params)
    for name, arm in (("arm_a", "A"), ("arm_a_prime", "A'"), ("arm_e", "E"),
                      ("arm_flat", "FLAT")):
        for r in load_jsonl(RESULTS / f"{name}.jsonl"):
            units.append({"arm": arm, "model": "n/a", "prompt": "n/a",
                          "descriptor": r["descriptor"], "sample_idx": 0,
                          "target_split": r["target_split"], "params": r["params"]})
    for name, arm in (("arm_b", "B"), ("arm_c", "C"), ("arm_c_agg", "C_AGG"),
                      ("arm_d", "D"), ("arm_f", "F")):
        for r in load_jsonl(RESULTS / f"{name}.jsonl"):
            units.append({"arm": arm, "model": r["model"], "prompt": r["prompt"],
                          "descriptor": r["descriptor"], "sample_idx": r["sample_idx"],
                          "target_split": r["target_split"], "params": r["params"]})

    # ---- per-unit metrics --------------------------------------------------
    rows = []
    clouds_by_group = defaultdict(list)   # (arm, model, prompt, desc) -> rendered samples
    for u in units:
        curve = render_curve(u["params"])
        row = {k: u[k] for k in ("arm", "model", "prompt", "descriptor",
                                 "sample_idx", "target_split")}
        d = u["descriptor"]
        if u["target_split"] == "eval" and d in eval_cloud:
            dist = median_lsd_to(curve, eval_cloud[d])
            row["centrality_lsd"] = round(dist, 4)
            row["hrp"] = round(hrp_of(dist, d), 2)
            cons = consensus.get(d, {})
            agree, mag = [], []
            for k in GAIN_KEYS:
                c = cons.get(k)
                if c is None or abs(c) < 0.5:      # no meaningful consensus sign
                    continue
                agree.append((u["params"][k] > 0) == (c > 0))
                mag.append(abs(u["params"][k] - c))
            row["sign_agree_frac"] = round(sum(agree) / len(agree), 3) if agree else None
            row["mag_err_db_mean"] = round(statistics.mean(mag), 3) if mag else None
            clouds_by_group[(u["arm"], u["model"], u["prompt"], d)].append(curve)
        elif u["target_split"] == "tail_eval" and d in tail_cloud:
            row["tail_lsd"] = round(median_lsd_to(curve, tail_cloud[d]), 4)
        rows.append(row)

    # ---- distribution coverage per group (sampled arms only) ---------------
    energy_rows = []
    for (arm, model, prompt, d), cloud in clouds_by_group.items():
        if len(cloud) >= 5 and d in eval_cloud:
            comp = energy_components(cloud, eval_cloud[d])
            energy_rows.append({"arm": arm, "model": model, "prompt": prompt,
                                "descriptor": d, "n_samples": len(cloud), **comp})

    OUT_METRICS.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n" +
        "\n".join(json.dumps({**r, "row_type": "energy"}, sort_keys=True)
                  for r in energy_rows) + "\n", encoding="utf-8")

    # ---- receipts ----------------------------------------------------------
    n_eval_rows = sum(1 for r in rows if "centrality_lsd" in r)
    n_tail_rows = sum(1 for r in rows if "tail_lsd" in r)
    per_arm = defaultdict(list)
    for r in rows:
        if "hrp" in r:
            per_arm[r["arm"]].append(r["hrp"])
    lines = ["METRICS — Stage 6",
             f"prediction units: {len(rows)} ({n_eval_rows} eval-scored, {n_tail_rows} tail-scored)",
             f"energy-distance groups (n>=5 samples): {len(energy_rows)}",
             "",
             "HRP by arm (eval descriptors; median [IQR] across units):"]
    for arm in sorted(per_arm):
        v = sorted(per_arm[arm])
        med = statistics.median(v)
        q1, q3 = v[len(v) // 4], v[(3 * len(v)) // 4]
        lines.append(f"  {arm:<3} n={len(v):>4}  median HRP {med:6.1f}  [IQR {q1:.1f}-{q3:.1f}]")
    lines += ["", "Human reference: HRP is percentile vs leave-one-out human",
              "centrality; ~50 = typical human. Full distributions in metrics.jsonl.",
              f"eval descriptors: { {d: len(c) for d, c in eval_cloud.items()} }"]
    OUT_RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_INTERP.write_text(
        "METRICS INTERPRETATION — Stage 6\n\n"
        "1. This script is the frozen eval sets' only reader (design §5). Any\n"
        "   other consumer is a protocol violation.\n"
        "2. HRP interpretation guard (D-005): Arm A sits near HRP 0 BY\n"
        "   CONSTRUCTION (consensus anchor). Claim-bearing contrasts are B/C/E\n"
        "   vs the human range and vs each other; A is context, not competitor.\n"
        "3. energy_dist rows exist only for groups with >=5 cloud samples;\n"
        "   point-estimate arms are structurally excluded there and compete on\n"
        "   centrality/HRP only — stated in methods, not hidden.\n"
        "4. Inferential statistics live in 07_stats.py; nothing here is a\n"
        "   significance claim.\n", encoding="utf-8")
    print(f"metrics rows={len(rows)} energy_groups={len(energy_rows)} -> {OUT_METRICS.name}")
    for arm in sorted(per_arm):
        v = per_arm[arm]
        print(f"  {arm}: n={len(v)} median_hrp={statistics.median(v):.1f}")


if __name__ == "__main__":
    main()
