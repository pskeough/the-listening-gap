"""Stage 4 — Arms A (embedding retrieval) and A′ (deployed keyword baseline).

Consumes TRAIN ONLY (hash-verified against the Stage 2 lock before use).
Emits one predicted curve per unique descriptor in the eval and tail splits:
    results/arm_a.jsonl        embedding-retrieval arm
    results/arm_a_prime.jsonl  deployed keyword-overlap arm (honesty baseline)
plus paired receipts in results/stats/.

Method per design v1.0:
  * TRAIN-only 2-SD outlier rejection per above-floor descriptor on the five
    band gains (the fitted filter deferred from Stages 1-2, now legitimate
    because it sees only TRAIN), then per-descriptor centroids.
  * Arm A, eval descriptors (warm/bright): prediction = own TRAIN centroid.
  * Arm A, tail descriptors: MiniLM (fastembed ONNX, pinned D-021) embedding
    of the descriptor vs. embeddings of the anchor descriptors; positive
    cosine similarities normalized to weights; weighted blend of anchor
    centroids. (With two anchors this is interpolation on the warm-bright
    axis — the honest capability of a 2-anchor codebook.)
  * Arm A′: faithful port of the shipped EqualizerAI matcher
    (embed_song_predictor.py): bag-of-words intersection of the descriptor
    against each profile vocabulary, +3.0 direct-match bonus, weights
    normalized; profiles blend from TRAIN centroids of the profile words
    (any n; ns reported); no match -> deployed flat-EQ fallback.
"""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from fastembed import TextEmbedding

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
OUT_A = BASE / "results" / "arm_a.jsonl"
OUT_AP = BASE / "results" / "arm_a_prime.jsonl"
OUT_RESULTS = BASE / "results" / "stats" / "retrieval_arms_results.txt"
OUT_INTERP = BASE / "results" / "stats" / "retrieval_arms_interpretation.txt"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # D-021
Z_LIMIT = 2.0
PARAM_NAMES = [
    "low_shelf_gain", "low_shelf_freq",
    "band1_gain", "band1_freq", "band1_q",
    "band2_gain", "band2_freq", "band2_q",
    "band3_gain", "band3_freq", "band3_q",
    "high_shelf_gain", "high_shelf_freq",
]
GAIN_KEYS = ["low_shelf_gain", "band1_gain", "band2_gain", "band3_gain", "high_shelf_gain"]

# Deployed system port — vocabularies copied verbatim from
# EmbeddingTestIsolated/embed_song_predictor.py (A-prime must match the app).
SONIC_DICTIONARY = {
    "warm": {"warm", "warmth", "cozy", "soft", "mellow", "smooth", "acoustic",
             "slow", "intimate", "analog", "vintage", "ballad", "folk",
             "indie", "singer-songwriter", "lo-fi", "jazz", "blues"},
    "bright": {"bright", "brightness", "crisp", "clear", "treble", "highs", "sharp",
               "synth", "synthesizer", "sparkle", "electronic", "pop", "disco",
               "dance", "techno", "house", "edm", "electro", "synthpop"},
    "muddy": {"muddy", "boxy", "bloated", "muffled", "boomy", "dark", "heavy"},
    "presence": {"presence", "vocal", "vocals", "mid", "mids", "voice", "lead",
                 "guitar", "singer", "songwriter", "intimate", "rock",
                 "alternative", "alternative rock", "indie", "metal", "grunge",
                 "hard rock"},
    "airy": {"air", "airy", "ambient", "dreamy", "reverb", "chill", "open",
             "breathable", "sparkle", "space", "psychedelic", "classical",
             "chillout", "instrumental"},
    "punchy": {"punchy", "punch", "dynamic", "groove", "funk", "kick", "drums",
               "snare", "beat", "electronic", "house", "dance", "club",
               "heavy bass", "hip-hop", "rap", "r&b", "drum and bass",
               "dubstep", "trap"},
}
FLAT_EQ = {
    "low_shelf_gain": 0.0, "low_shelf_freq": 120.0,
    "band1_gain": 0.0, "band1_freq": 250.0, "band1_q": 0.71,
    "band2_gain": 0.0, "band2_freq": 1000.0, "band2_q": 0.71,
    "band3_gain": 0.0, "band3_freq": 3500.0, "band3_q": 0.71,
    "high_shelf_gain": 0.0, "high_shelf_freq": 10000.0,
}


def verify_split(name: str) -> list:
    p = SPLITS / f"{name}.jsonl"
    expected = (SPLITS / f"{name}.sha256").read_text().strip()
    h = hashlib.sha256(p.read_bytes()).hexdigest().upper()
    if h != expected:
        raise SystemExit(f"ABORT: {name}.jsonl hash mismatch vs lock")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def centroid_of(entries: list) -> dict:
    return {k: statistics.mean(e["params"][k] for e in entries) for k in PARAM_NAMES}


def reject_outliers_2sd(entries: list) -> list:
    """TRAIN-only fitted filter: drop entries with any band-gain z-score > Z_LIMIT."""
    if len(entries) < 3:
        return entries
    means = {k: statistics.mean(e["params"][k] for e in entries) for k in GAIN_KEYS}
    sds = {k: statistics.stdev(e["params"][k] for e in entries) for k in GAIN_KEYS}
    kept = [e for e in entries
            if all(sds[k] == 0 or abs(e["params"][k] - means[k]) / sds[k] <= Z_LIMIT
                   for k in GAIN_KEYS)]
    return kept if kept else entries


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def blend(centroids: dict, weights: dict) -> dict:
    total = sum(weights.values())
    return {k: sum(centroids[d][k] * w for d, w in weights.items()) / total
            for k in PARAM_NAMES}


def main() -> None:
    train = verify_split("train")
    eval_set = verify_split("eval_set")
    tail = verify_split("tail_eval")

    by_desc = defaultdict(list)
    for r in train:
        by_desc[r["descriptor"]].append(r)

    # Anchor centroids (above-floor descriptors present in train)
    lines = ["RETRIEVAL ARMS — Stage 4", ""]
    anchors = {}
    for d, entries in sorted(by_desc.items(), key=lambda kv: -len(kv[1])):
        if len(entries) >= 16:      # 80% of the floor-20 -> above-floor in train
            kept = reject_outliers_2sd(entries)
            anchors[d] = centroid_of(kept)
            lines.append(f"anchor '{d}': train n={len(entries)}, after 2-SD rejection n={len(kept)}")

    eval_descs = sorted({r["descriptor"] for r in eval_set})
    tail_descs = sorted({r["descriptor"] for r in tail})

    # ---- Arm A -------------------------------------------------------------
    model = TextEmbedding(EMBED_MODEL)
    anchor_names = sorted(anchors)
    anchor_vecs = dict(zip(anchor_names, (list(v) for v in model.embed(anchor_names))))

    arm_a = []
    for d in eval_descs:
        arm_a.append({"descriptor": d, "target_split": "eval",
                      "params": anchors[d], "method": "own_centroid"})
    tail_vecs = dict(zip(tail_descs, (list(v) for v in model.embed(tail_descs))))
    example_weights = []
    for d in tail_descs:
        sims = {a: max(0.0, cosine(tail_vecs[d], anchor_vecs[a])) for a in anchor_names}
        if sum(sims.values()) == 0:
            params, method = dict(FLAT_EQ), "no_positive_similarity_flat"
        else:
            params, method = blend(anchors, sims), "cosine_blend"
        arm_a.append({"descriptor": d, "target_split": "tail_eval",
                      "params": params, "method": method,
                      "weights": {a: round(s, 4) for a, s in sims.items()}})
        if len(example_weights) < 5:
            example_weights.append(f"  '{d}' -> {sims}")

    # ---- Arm A′ ------------------------------------------------------------
    profile_centroids, profile_ns = {}, {}
    for p in SONIC_DICTIONARY:
        entries = by_desc.get(p, [])
        if entries:
            profile_centroids[p] = centroid_of(reject_outliers_2sd(entries))
            profile_ns[p] = len(entries)
    lines.append("")
    lines.append(f"A' profile centroids from TRAIN: { {p: n for p, n in profile_ns.items()} }")
    lines.append("A' profiles with zero TRAIN data (skipped): "
                 f"{sorted(set(SONIC_DICTIONARY) - set(profile_centroids))}")

    def a_prime_predict(desc: str) -> tuple:
        scores = {}
        for prof, vocab in SONIC_DICTIONARY.items():
            if prof not in profile_centroids:
                continue
            s = float(len({desc} & vocab))
            if prof == desc:
                s += 3.0
            if s > 0:
                scores[prof] = s
        if not scores:
            return dict(FLAT_EQ), "flat_fallback", {}
        return blend(profile_centroids, scores), "keyword_blend", scores

    arm_ap = []
    n_flat = 0
    for d, split in [(d, "eval") for d in eval_descs] + [(d, "tail_eval") for d in tail_descs]:
        params, method, scores = a_prime_predict(d)
        n_flat += method == "flat_fallback"
        arm_ap.append({"descriptor": d, "target_split": split, "params": params,
                       "method": method, "weights": scores})

    OUT_A.write_text("\n".join(json.dumps(r, sort_keys=True) for r in arm_a) + "\n",
                     encoding="utf-8")
    OUT_AP.write_text("\n".join(json.dumps(r, sort_keys=True) for r in arm_ap) + "\n",
                      encoding="utf-8")

    lines += ["", f"Arm A predictions : {len(arm_a)} ({len(eval_descs)} eval + {len(tail_descs)} tail)",
              f"Arm A' predictions: {len(arm_ap)}  (flat-fallback on {n_flat} descriptors)",
              "", "Example tail similarity weights (Arm A):", *example_weights]
    OUT_RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_INTERP.write_text(
        "RETRIEVAL ARMS INTERPRETATION — Stage 4\n\n"
        "1. On the confirmatory eval descriptors (warm/bright) Arm A is the own-\n"
        "   descriptor TRAIN centroid — the consensus anchor (HRP framing D-005).\n"
        "   A' converges toward A there because the descriptor directly matches its\n"
        "   profile word; the arms separate on the tail, where A interpolates the\n"
        "   warm-bright embedding axis and A' falls back to keyword hits or flat.\n"
        f"2. A' emitted the deployed flat-EQ fallback on {n_flat} descriptors — the\n"
        "   deployed system's honest behavior on vocabulary misses, reported not\n"
        "   hidden.\n"
        "3. Everything here consumed TRAIN only (hash-verified); the 2-SD filter is\n"
        "   fitted on TRAIN per design §5, closing the loop deferred since Stage 1.\n",
        encoding="utf-8")
    print(f"arm_a: {len(arm_a)} predictions | arm_a_prime: {len(arm_ap)} | "
          f"anchors: {anchor_names} | a_prime_flat: {n_flat}")


if __name__ == "__main__":
    main()
