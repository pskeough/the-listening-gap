"""Stage 10 — mode-aware model evaluation: which school did the models learn?

Connects the Stage 9 discovery (human "warm"/"bright" are mixtures of
discrete schools) to the model arms:

  A. SCHOOL ASSIGNMENT: refit the exact Stage 9 GMM (same seed/params ->
     identical modes), assign every model-generated eval curve to its nearest
     school, and compare model school-shares against human school-shares.
     Chi-square goodness-of-fit per arm x model x descriptor (expected =
     human school proportions). Tests "doctrine vs practice": do models
     reproduce the population mixture, or collapse onto one school — and
     WHICH school (the textbook one or the majority one)?
  B. CONTEXT-PULL COEFFICIENT: for each (model, prompt, descriptor), project
     (C_mean - B_greedy) onto (exemplar_mean - B_greedy) in rendered space.
     1.0 = model fully defers to provided context; 0.0 = ignores it; the
     numeric-domain cousin of instruction-deference. Same for C_AGG toward
     its provided centroid.
  C. BAND-ERROR ANATOMY: signed error (model greedy curve - TRAIN centroid)
     averaged over five frequency regions — where does model knowledge
     deviate systematically?

TRAIN-fitted structures only (GMM, PCA, centroids, exemplar sets); model
curves come from arm files; eval human curves are NOT read here (school
shares use TRAIN GMM labels).
"""

import hashlib
import importlib.util
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
RESULTS = BASE / "results"
OUT = RESULTS / "stats" / "mode_analysis_results.txt"
OUT_I = RESULTS / "stats" / "mode_analysis_interpretation.txt"

SEED = 42
K_BY_DESC = {"warm": 3, "bright": 4}   # Stage 9 BIC selections (receipted)
REGIONS = {"low(20-150)": (0, 37), "lowmid(150-600)": (37, 88),
           "mid(0.6-2k)": (88, 133), "highmid(2-6k)": (133, 173),
           "high(6-20k)": (173, 256)}

_spec = importlib.util.spec_from_file_location("render", BASE / "pipeline" / "03_render.py")
_render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_render)
render_curve = _render.render_curve


def verify_train() -> list:
    p = SPLITS / "train.jsonl"
    expected = (SPLITS / "train.sha256").read_text(encoding="utf-8-sig").strip()
    if hashlib.sha256(p.read_bytes()).hexdigest().upper() != expected:
        raise SystemExit("ABORT: train hash mismatch")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def load_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def main() -> None:
    train = verify_train()
    by_desc = defaultdict(list)
    for r in train:
        by_desc[r["descriptor"]].append(r)

    lines = ["MODE-AWARE MODEL EVALUATION — Stage 10", ""]

    # ---- fit TRAIN GMMs (identical spec to Stage 9) ------------------------
    gmm, pca, human_share, mode_desc = {}, {}, {}, {}
    for d, k in K_BY_DESC.items():
        curves = np.array([render_curve(r["params"]) for r in by_desc[d]])
        pca[d] = PCA(n_components=10, random_state=SEED).fit(curves)
        X = pca[d].transform(curves)
        gmm[d] = GaussianMixture(n_components=k, covariance_type="full",
                                 random_state=SEED, n_init=3).fit(X)
        labels = gmm[d].predict(X)
        share = np.bincount(labels, minlength=k) / len(labels)
        human_share[d] = share
        descs = []
        for m in range(k):
            mc = curves[labels == m].mean(axis=0)
            lo, hi = float(np.mean(mc[:43])), float(np.mean(mc[-64:]))
            descs.append(f"m{m}[{share[m]*100:.0f}% lo{lo:+.1f} hi{hi:+.1f}]")
        mode_desc[d] = "  ".join(descs)
        lines.append(f"[{d}] human school shares: {mode_desc[d]}")
    lines.append("")

    # ---- A. school assignment of model curves ------------------------------
    arms = {"B": load_jsonl(RESULTS / "arm_b.jsonl"),
            "C": load_jsonl(RESULTS / "arm_c.jsonl"),
            "C_AGG": load_jsonl(RESULTS / "arm_c_agg.jsonl"),
            "D": load_jsonl(RESULTS / "arm_d.jsonl"),
            "F": load_jsonl(RESULTS / "arm_f.jsonl")}
    lines.append("A. SCHOOL ASSIGNMENT (model curve -> nearest GMM mode; eval descriptors):")
    for arm, rows in arms.items():
        if not rows:
            lines.append(f"  [{arm}] no data (arm file absent)")
            continue
        per_md = defaultdict(list)
        for r in rows:
            if r.get("target_split") == "eval" and r["descriptor"] in K_BY_DESC:
                per_md[(r["model"], r["descriptor"])].append(r)
        for (model, d), rs in sorted(per_md.items()):
            X = pca[d].transform(np.array([render_curve(r["params"]) for r in rs]))
            labels = gmm[d].predict(X)
            k = K_BY_DESC[d]
            counts = np.bincount(labels, minlength=k)
            exp = human_share[d] * len(rs)
            # chi-square GOF only where expected counts are adequate
            ok_cells = (exp >= 1).all() and len(rs) >= 20
            if ok_cells:
                chi = sps.chisquare(counts, f_exp=exp)
                sig = f"chi2 p={chi.pvalue:.2e}"
            else:
                sig = "n too small for GOF"
            share_str = "/".join(f"{c}" for c in counts)
            lines.append(f"  [{arm}] {model.split('/')[-1]:<28} {d:<7} n={len(rs):>3} "
                         f"modes {share_str}  (human {'/'.join(f'{s*100:.0f}%' for s in human_share[d])})  {sig}")
    lines.append("")

    # ---- B. context-pull coefficient ---------------------------------------
    lines.append("B. CONTEXT-PULL (0=ignores provided context, 1=fully defers):")
    b_greedy = {}
    for r in arms["B"]:
        if r.get("target_split") == "eval" and r["sample_idx"] == 0:
            b_greedy[(r["model"], r["prompt"], r["descriptor"])] = \
                np.array(render_curve(r["params"]))
    # exemplar means: replicate harness selection (seed per descriptor)
    ex_mean = {}
    for d in K_BY_DESC:
        pool = sorted(by_desc[d], key=lambda r: r["id"])
        picks = random.Random(f"{SEED}:{d}").sample(pool, 5)
        ex_mean[d] = np.mean([render_curve(p["params"]) for p in picks], axis=0)
    cent = {r["descriptor"]: np.array(render_curve(r["params"]))
            for r in load_jsonl(RESULTS / "arm_a.jsonl") if r["target_split"] == "eval"}
    for arm, target in (("C", ex_mean), ("C_AGG", cent)):
        pulls = defaultdict(list)
        for r in arms[arm]:
            if r.get("target_split") != "eval" or r["descriptor"] not in K_BY_DESC:
                continue
            key = (r["model"], r["prompt"], r["descriptor"])
            if key not in b_greedy:
                continue
            b0 = b_greedy[key]
            tgt = target[r["descriptor"]]
            denom = float(np.dot(tgt - b0, tgt - b0))
            if denom < 1e-9:
                continue
            pull = float(np.dot(np.array(render_curve(r["params"])) - b0, tgt - b0)) / denom
            pulls[r["model"]].append(pull)
        for model, v in sorted(pulls.items()):
            lines.append(f"  [{arm}] {model.split('/')[-1]:<28} median pull "
                         f"{statistics.median(v):+.2f}  (n={len(v)})")
    lines.append("")

    # ---- C. band-error anatomy (B greedy vs TRAIN centroid) ----------------
    lines.append("C. BAND-ERROR ANATOMY (B greedy signed error vs TRAIN centroid, dB):")
    train_cent = {d: np.mean([render_curve(r["params"]) for r in by_desc[d]], axis=0)
                  for d in K_BY_DESC}
    for d in K_BY_DESC:
        errs = defaultdict(list)
        for (model, prompt, desc), curve in b_greedy.items():
            if desc == d:
                diff = curve - train_cent[d]
                for rname, (a, b) in REGIONS.items():
                    errs[rname].append(float(np.mean(diff[a:b])))
        row = "  ".join(f"{rn}:{statistics.mean(v):+.2f}" for rn, v in errs.items())
        lines.append(f"  [{d}] {row}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_I.write_text(
        "MODE ANALYSIS INTERPRETATION — Stage 10\n\n"
        "A. If model school-shares mismatch human shares (significant GOF),\n"
        "   models do not reproduce the population mixture. WHICH mode they\n"
        "   concentrate in is the doctrine-vs-practice question: concentration\n"
        "   in a minority 'textbook' school implies knowledge from written\n"
        "   doctrine rather than behavioral practice.\n"
        "B. Context-pull near 1.0 for C_AGG with lower values for C implies\n"
        "   models defer to authoritative-looking aggregates more than to\n"
        "   noisy individual examples; negative or >1 values flag overshoot.\n"
        "C. Systematic signed band errors show where model priors deviate\n"
        "   from practiced consensus (e.g. over-boosted bass on 'warm').\n"
        "All reference structures TRAIN-fitted; eval curves not read here.\n",
        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
