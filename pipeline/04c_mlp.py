"""Stage 4c — Arm E: frozen sentence-embedding → small MLP regressor.

The classic-ML floor and shippable-product candidate (design §4, D-013).
Trains on TRAIN only (hash-verified), 3 seeds, and predicts a 13-param curve
for every unique descriptor in eval and tail splits.

Honest modeling notes, stated up front:
  * TRAIN contains only two distinct descriptor strings (warm/bright), so the
    MLP's fit reduces to two conditional means; its scientific interest is
    (a) whether a trained model recovers the retrieval arm's behavior and
    (b) how it generalizes along the embedding manifold on the tail, where
    inputs it never saw land between and beyond the anchors.
  * Frequencies and Qs are regressed in log10 space (perceptually and
    physically the right scale); gains linear. Outputs clipped to the frozen
    output schema ranges.
  * Seeds 0/1/2; the emitted prediction is the seed-mean, per-seed spread is
    receipted (D-013 variance-reporting requirement).
"""

import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from sklearn.neural_network import MLPRegressor

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
OUT_E = BASE / "results" / "arm_e.jsonl"
OUT_RESULTS = BASE / "results" / "stats" / "arm_e_results.txt"
OUT_INTERP = BASE / "results" / "stats" / "arm_e_interpretation.txt"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS = [0, 1, 2]
PARAM_NAMES = [
    "low_shelf_gain", "low_shelf_freq",
    "band1_gain", "band1_freq", "band1_q",
    "band2_gain", "band2_freq", "band2_q",
    "band3_gain", "band3_freq", "band3_q",
    "high_shelf_gain", "high_shelf_freq",
]
LOG_KEYS = {k for k in PARAM_NAMES if k.endswith("_freq") or k.endswith("_q")}
CLIP = {**{k: (-12.0, 12.0) for k in PARAM_NAMES if k.endswith("_gain")},
        **{k: (20.0, 20000.0) for k in PARAM_NAMES if k.endswith("_freq")},
        **{k: (0.1, 10.0) for k in PARAM_NAMES if k.endswith("_q")}}


def verify_split(name: str) -> list:
    p = SPLITS / f"{name}.jsonl"
    expected = (SPLITS / f"{name}.sha256").read_text().strip()
    if hashlib.sha256(p.read_bytes()).hexdigest().upper() != expected:
        raise SystemExit(f"ABORT: {name}.jsonl hash mismatch vs lock")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def to_target(params: dict) -> list:
    return [math.log10(params[k]) if k in LOG_KEYS else params[k] for k in PARAM_NAMES]


def from_target(vec) -> dict:
    out = {}
    for i, k in enumerate(PARAM_NAMES):
        v = 10.0 ** vec[i] if k in LOG_KEYS else float(vec[i])
        lo, hi = CLIP[k]
        out[k] = min(hi, max(lo, v))
    return out


def main() -> None:
    train = verify_split("train")
    eval_set = verify_split("eval_set")
    tail = verify_split("tail_eval")

    model = TextEmbedding(EMBED_MODEL)
    train_descs = [r["descriptor"] for r in train]
    uniq_train = sorted(set(train_descs))
    uniq_vecs = dict(zip(uniq_train, (np.array(v) for v in model.embed(uniq_train))))
    X = np.stack([uniq_vecs[d] for d in train_descs])
    Y = np.array([to_target(r["params"]) for r in train])

    predict_descs = sorted({r["descriptor"] for r in eval_set} |
                           {r["descriptor"] for r in tail})
    pred_vecs = np.stack([np.array(v) for v in model.embed(predict_descs)])
    eval_names = {r["descriptor"] for r in eval_set}

    per_seed = []
    for seed in SEEDS:
        mlp = MLPRegressor(hidden_layer_sizes=(64,), activation="relu",
                           solver="adam", max_iter=2000, random_state=seed,
                           tol=1e-6)
        mlp.fit(X, Y)
        per_seed.append(mlp.predict(pred_vecs))
    stack = np.stack(per_seed)                       # (seeds, descs, 13)
    mean_pred = stack.mean(axis=0)
    seed_spread = stack.std(axis=0).mean(axis=1)     # per-descriptor mean SD

    records = []
    for i, d in enumerate(predict_descs):
        records.append({
            "descriptor": d,
            "target_split": "eval" if d in eval_names else "tail_eval",
            "params": from_target(mean_pred[i]),
            "seed_spread_mean_targetspace": round(float(seed_spread[i]), 5),
        })
    OUT_E.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
                     encoding="utf-8")

    OUT_RESULTS.write_text(
        "ARM E (embedding->MLP) — Stage 4c\n"
        f"train rows: {len(train)} (distinct descriptors: {uniq_train})\n"
        f"seeds: {SEEDS}  hidden: (64,)  targets: gains linear, freq/Q log10\n"
        f"predictions: {len(records)} descriptors "
        f"({sum(1 for r in records if r['target_split'] == 'eval')} eval, "
        f"{sum(1 for r in records if r['target_split'] == 'tail_eval')} tail)\n"
        f"cross-seed spread (target space): mean "
        f"{float(np.mean(seed_spread)):.5f}, max {float(np.max(seed_spread)):.5f}\n",
        encoding="utf-8")
    OUT_INTERP.write_text(
        "ARM E INTERPRETATION — Stage 4c\n\n"
        "1. With two distinct training inputs the MLP necessarily learns two\n"
        "   conditional means; on eval descriptors Arm E is expected to track\n"
        "   Arm A closely. Its informative behavior is on the tail, where unseen\n"
        "   descriptor embeddings land off-anchor and the network extrapolates.\n"
        "2. Cross-seed spread is receipted above; small spread = the fit is\n"
        "   data-determined, not initialization-determined.\n"
        "3. Model scale note for the product narrative: MiniLM (~90 MB, ONNX) +\n"
        "   a (64,) MLP head (~KB) — runs offline, no API.\n",
        encoding="utf-8")
    print(f"arm_e: {len(records)} predictions | seed spread mean "
          f"{float(np.mean(seed_spread)):.5f} max {float(np.max(seed_spread)):.5f}")


if __name__ == "__main__":
    main()
