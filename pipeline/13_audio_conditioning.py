"""Stage 13 — is human 'disagreement' actually source-audio conditioning?

THE validity probe on the dispersion headline. Our central claim is that
models are ~8x under-dispersed relative to the human curve distribution. That
is only fair if the human spread is genuine person-to-person disagreement.
But SAFE engineers each EQ'd SPECIFIC source audio; their curves are
conditional responses warm(source). This stage decomposes the human curve
variance into:
    (a) explained by SOURCE (unprocessed) audio features  -> conditional signal
    (b) residual                                           -> genuine disagreement

Method (TRAIN only, warm/bright separately):
  * Join each TRAIN entry to its unprocessed (pre-EQ) 80-dim audio feature
    vector via the entry-ID mapping (rebuilt + verified here).
  * Render each EQ curve, PCA to components capturing >=95% variance.
  * Ridge regression, honest out-of-fold R^2 (5-fold cross_val_predict), of
    source features -> each PCA score. Aggregate R^2 = variance-weighted mean
    = fraction of rendered-curve variance a source-aware predictor can explain.
  * PERMUTATION CONTROL: shuffle the entry<->audio pairing and refit; R^2 must
    collapse to ~0. This proves any positive R^2 is real signal, not the
    overfitting of 80 predictors on a few hundred rows.
  * REFRAME: report the residual (genuine-disagreement) dispersion and compare
    the model cloud dispersion against THAT fairer target, not the raw human
    dispersion.
  * TARGET-CONVERGENCE: per-feature variance ratio processed/unprocessed —
    do humans pull varied sources toward a common output (ratio<1)?

Honest failure modes handled: zero-variance feature columns dropped;
standardization fit inside CV folds via Pipeline; if the audio join covers
too few TRAIN entries the stage reports and aborts rather than guessing.
"""

import csv
import hashlib
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
SPLITS = BASE / "data" / "splits"
RESULTS = BASE / "results"
OUT = RESULTS / "stats" / "audio_conditioning_results.txt"
OUT_I = RESULTS / "stats" / "audio_conditioning_interpretation.txt"

MAIN_SHA = "CC43858680115070FFDFA41A1158FCA1344DB2E0096BA73FE68990E90A284AF1"
AUDIO_SHA = "E28D909FA9FED943F8611AD446BE88B4CA3B87FD78815E1484926A9CEA1E969F"
SEED = 42

_spec = importlib.util.spec_from_file_location("render", BASE / "pipeline" / "03_render.py")
_render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_render)
render_curve = _render.render_curve
PARAM_NAMES = ["low_shelf_gain", "low_shelf_freq", "band1_gain", "band1_freq",
               "band1_q", "band2_gain", "band2_freq", "band2_q", "band3_gain",
               "band3_freq", "band3_q", "high_shelf_gain", "high_shelf_freq"]
N_GRID_SQRT = 16.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def verify_train():
    p = SPLITS / "train.jsonl"
    exp = (SPLITS / "train.sha256").read_text(encoding="utf-8-sig").strip()
    if sha(p) != exp:
        raise SystemExit("ABORT: train hash mismatch")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def build_join():
    if sha(RAW / "SAFEEqualiserUserData.csv") != MAIN_SHA:
        raise SystemExit("ABORT: main CSV hash mismatch")
    if sha(RAW / "SAFEEqualiserAudioFeatureData.csv") != AUDIO_SHA:
        raise SystemExit("ABORT: audio CSV hash mismatch")
    lineidx_to_eid = {}
    with open(RAW / "SAFEEqualiserUserData.csv", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.reader(f)):
            if len(row) == 25:
                lineidx_to_eid[f"safe-{i}"] = row[0].strip()
    eid_unproc = {}
    with open(RAW / "SAFEEqualiserAudioFeatureData.csv", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) == 82 and row[1].strip() == "unprocessed":
                try:
                    eid_unproc[row[0].strip()] = [float(x) for x in row[2:82]]
                except ValueError:
                    pass
    eid_proc = {}
    with open(RAW / "SAFEEqualiserAudioFeatureData.csv", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) == 82 and row[1].strip() == "processed":
                try:
                    eid_proc[row[0].strip()] = [float(x) for x in row[2:82]]
                except ValueError:
                    pass
    return lineidx_to_eid, eid_unproc, eid_proc


def cv_r2(Xf, Yscores, evr, rng_seed):
    """Variance-weighted out-of-fold R^2 across PCA components."""
    kf = KFold(n_splits=5, shuffle=True, random_state=rng_seed)
    r2s = []
    for j in range(Yscores.shape[1]):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        pred = cross_val_predict(model, Xf, Yscores[:, j], cv=kf)
        r2s.append(max(-1.0, r2_score(Yscores[:, j], pred)))
    r2s = np.array(r2s)
    weighted = float(np.sum(r2s * evr) / np.sum(evr))
    return weighted, r2s


def main():
    train = verify_train()
    lineidx_to_eid, unproc, proc = build_join()

    lines = ["AUDIO-CONDITIONING VARIANCE DECOMPOSITION — Stage 13 (TRAIN only)", ""]
    by_desc = defaultdict(list)
    for r in train:
        by_desc[r["descriptor"]].append(r)

    interp_nums = {}
    for d in ("warm", "bright"):
        rows = by_desc[d]
        joined = [(r, unproc[lineidx_to_eid[r["id"]]])
                  for r in rows
                  if lineidx_to_eid.get(r["id"]) in unproc]
        cov = len(joined) / len(rows)
        lines.append(f"[{d}] TRAIN n={len(rows)}, audio-joined n={len(joined)} ({cov*100:.0f}%)")
        if len(joined) < 50:
            lines.append("  too few joined entries; skipping decomposition")
            continue

        curves = np.array([render_curve(r["params"]) for r, _ in joined])
        Xraw = np.array([feats for _, feats in joined])
        # drop zero-variance audio columns
        keep = Xraw.std(axis=0) > 1e-9
        Xf = Xraw[:, keep]
        lines.append(f"  usable audio features: {int(keep.sum())}/80")

        pca = PCA(n_components=0.95, svd_solver="full", random_state=SEED)
        Y = pca.fit_transform(curves)
        evr = pca.explained_variance_
        total_curve_var = float(np.sum(pca.explained_variance_))

        real_w, real_per = cv_r2(Xf, Y, evr, SEED)
        # permutation control: shuffle audio rows
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(Xf))
        perm_w, _ = cv_r2(Xf[perm], Y, evr, SEED)

        # dispersion reframe (rendered dB space)
        centroid = curves.mean(axis=0)
        raw_disp = float(np.mean([np.linalg.norm(c - centroid) / N_GRID_SQRT for c in curves]))
        residual_frac = max(0.0, 1.0 - real_w)
        residual_disp = raw_disp * (residual_frac ** 0.5)

        interp_nums[d] = dict(real=real_w, perm=perm_w, raw_disp=raw_disp,
                              resid_disp=residual_disp)
        lines.append(f"  rendered-curve variance explained by SOURCE audio (out-of-fold):")
        lines.append(f"    real pairing   R^2 = {real_w:+.3f}")
        lines.append(f"    permuted (null) R^2 = {perm_w:+.3f}   <- must be ~0")
        lines.append(f"  raw human dispersion       : {raw_disp:.2f} dB")
        lines.append(f"  residual (genuine-disagreement) dispersion: {residual_disp:.2f} dB")

        # target convergence: variance ratio processed / unprocessed
        pj = [(unproc[lineidx_to_eid[r["id"]]], proc[lineidx_to_eid[r["id"]]])
              for r in rows if lineidx_to_eid.get(r["id"]) in unproc
              and lineidx_to_eid.get(r["id"]) in proc]
        U = np.array([u for u, _ in pj]); P = np.array([p for _, p in pj])
        keep2 = (U.std(axis=0) > 1e-9)
        ratios = P[:, keep2].std(axis=0) / U[:, keep2].std(axis=0)
        conv = float(np.median(ratios))
        lines.append(f"  target-convergence: median per-feature output/input "
                     f"variance ratio = {conv:.2f} "
                     f"({'converge' if conv < 1 else 'diverge'} on output)")
        interp_nums[d]["conv"] = conv
        lines.append("")

    # model dispersion for comparison (from metrics energy rows)
    er = [json.loads(l) for l in (RESULTS / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip() and json.loads(l).get("row_type") == "energy"]
    model_disp = {}
    for arm in ("B", "C", "C_AGG"):
        v = [r["cloud_dispersion_db"] for r in er if r["arm"] == arm]
        if v:
            model_disp[arm] = statistics.mean(v)

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    il = ["AUDIO-CONDITIONING INTERPRETATION — Stage 13", ""]
    for d, nv in interp_nums.items():
        signal = nv["real"] - nv["perm"]
        il.append(f"[{d}] Source audio explains R^2={nv['real']:.2f} of rendered-curve "
                  f"variance out-of-fold (permutation null {nv['perm']:+.2f}; "
                  f"net signal {signal:+.2f}).")
        il.append(f"  -> Of the {nv['raw_disp']:.2f} dB raw human dispersion, the "
                  f"genuine person-to-person residual after removing source "
                  f"conditioning is ~{nv['resid_disp']:.2f} dB.")
        if "conv" in nv:
            il.append(f"  -> Output/input variance ratio {nv['conv']:.2f}: humans "
                      f"{'DO' if nv['conv']<1 else 'do NOT'} converge on a common "
                      f"acoustic target across varied sources.")
    il.append("")
    il.append(f"Model cloud dispersions for comparison: {{" +
              ", ".join(f'{k}:{v:.2f}dB' for k, v in model_disp.items()) + "}")
    il.append("CLAIM DISCIPLINE: if source R^2 is small and the permutation null is")
    il.append("~0, the dispersion headline stands essentially unchanged and is now")
    il.append("bulletproofed against the 'it's just audio conditioning' objection.")
    il.append("If R^2 is large, the headline must be restated as models matching the")
    il.append("MARGINAL human distribution while being blind to the CONDITIONAL task,")
    il.append("and the model-vs-human dispersion gap must be quoted against the")
    il.append("residual dispersion, not the raw dispersion. Report whichever is true.")
    OUT_I.write_text("\n".join(il) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("--- model dispersions ---", model_disp)


if __name__ == "__main__":
    main()
