"""Stage 9 — structure of the human gold itself (TRAIN only).

Three questions about the humans, each with direct consequences for the
paper's claims and the app:

  A. MULTIMODALITY: is "warm" ONE consensus or several schools? GMM with
     BIC selection (k=1..4) on PCA-reduced rendered TRAIN curves. If k>1
     wins, the centroid is a mean of modes that possibly NOBODY holds — a
     validity check on our own centroid/C_AGG arms (self-critique before a
     reviewer does it).
  B. EXPERTISE: do experienced engineers agree more (variance shrink) and/or
     hold a different consensus than novices? Experience parsed from raw
     metadata (messy free text; parse rules stated below; unparseable rows
     reported, not hidden).
  C. CONTEXT: does "warm" mean something different on guitar vs full-mix
     stimuli? Instrument/stimulus labels from raw metadata; centroid shift
     in rendered space vs within-group spread.

Rigor: TRAIN split only (hash-verified) — the frozen eval set stays unread
by everything except 06. Metadata is re-joined from the provenanced raw CSV
by row id (splits deliberately carry no metadata). All parse rules are fixed
here, a-priori.
"""

import csv
import hashlib
import importlib.util
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

BASE = Path(__file__).resolve().parent.parent
RAW_CSV = BASE / "data" / "raw" / "SAFEEqualiserUserData.csv"
SPLITS = BASE / "data" / "splits"
OUT = BASE / "results" / "stats" / "human_structure_results.txt"
OUT_I = BASE / "results" / "stats" / "human_structure_interpretation.txt"

EXPECTED_SHA256 = "CC43858680115070FFDFA41A1158FCA1344DB2E0096BA73FE68990E90A284AF1"
SEED = 42

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


def load_metadata() -> dict:
    if hashlib.sha256(RAW_CSV.read_bytes()).hexdigest().upper() != EXPECTED_SHA256:
        raise SystemExit("ABORT: raw CSV hash mismatch")
    meta = {}
    with open(RAW_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for i, row in enumerate(csv.reader(f)):
            if row and len(row) == 25:
                meta[f"safe-{i}"] = {"genre": row[18].strip(), "instrument": row[19].strip(),
                                     "experience": row[21].strip()}
    return meta


def parse_experience(raw: str):
    """Fixed rule: first number in the string = years. '' or no number -> None."""
    m = re.search(r"\d+\.?\d*", raw)
    return float(m.group()) if m else None


def classify_context(instrument: str):
    """Fixed rule: 'guitar' substring -> guitar; lab stimulus labels
    (jazzN/metalN/etc pattern letter+digit) -> mix_stimulus; else None."""
    low = instrument.lower()
    if "guitar" in low:
        return "guitar"
    if re.fullmatch(r"[a-z]+\d", low):
        return "mix_stimulus"
    return None


def main() -> None:
    train = verify_train()
    meta = load_metadata()
    lines = ["HUMAN STRUCTURE — Stage 9 (TRAIN only, n=%d)" % len(train), ""]

    by_desc = defaultdict(list)
    for r in train:
        by_desc[r["descriptor"]].append(r)

    # ---- A. multimodality --------------------------------------------------
    lines.append("A. MULTIMODALITY (GMM, BIC selection, PCA-10 on rendered curves):")
    modes_found = {}
    for d in ("warm", "bright"):
        curves = np.array([render_curve(r["params"]) for r in by_desc[d]])
        pca = PCA(n_components=10, random_state=SEED)
        X = pca.fit_transform(curves)
        bics = {}
        for k in (1, 2, 3, 4):
            g = GaussianMixture(n_components=k, covariance_type="full",
                                random_state=SEED, n_init=3).fit(X)
            bics[k] = g.bic(X)
        best = min(bics, key=bics.get)
        modes_found[d] = best
        lines.append(f"  [{d}] n={len(curves)}  BIC: " +
                     "  ".join(f"k={k}:{v:,.0f}" for k, v in bics.items()) +
                     f"  -> best k={best}")
        if best > 1:
            g = GaussianMixture(n_components=best, covariance_type="full",
                                random_state=SEED, n_init=3).fit(X)
            labels = g.predict(X)
            for m in range(best):
                idx = [i for i, l in enumerate(labels) if l == m]
                mode_curves = curves[idx]
                mean_curve = mode_curves.mean(axis=0)
                lo = float(np.mean(mean_curve[:43]))     # ~20-200 Hz region
                hi = float(np.mean(mean_curve[-64:]))    # ~5k-20k region
                lines.append(f"    mode {m}: n={len(idx)} ({100*len(idx)/len(curves):.0f}%)"
                             f"  low-band avg {lo:+.1f} dB, high-band avg {hi:+.1f} dB")

    # ---- B. expertise ------------------------------------------------------
    lines.append("")
    lines.append("B. EXPERTISE (novice <=2 yrs vs experienced >=5 yrs; parse rule in docstring):")
    for d in ("warm", "bright"):
        groups = {"novice": [], "experienced": []}
        n_unparsed = 0
        for r in by_desc[d]:
            yrs = parse_experience(meta.get(r["id"], {}).get("experience", ""))
            if yrs is None:
                n_unparsed += 1
            elif yrs <= 2:
                groups["novice"].append(r)
            elif yrs >= 5:
                groups["experienced"].append(r)
        row = [f"  [{d}] unparsed/blank: {n_unparsed}"]
        cents = {}
        for gname, entries in groups.items():
            if len(entries) < 15:
                row.append(f"{gname}: n={len(entries)} (below floor, skipped)")
                continue
            curves = np.array([render_curve(r["params"]) for r in entries])
            spread = float(np.mean([np.linalg.norm(a - b) / 16.0
                                    for i, a in enumerate(curves)
                                    for b in curves[i + 1:]]))
            cents[gname] = curves.mean(axis=0)
            row.append(f"{gname}: n={len(entries)} spread={spread:.2f} dB")
        if len(cents) == 2:
            shift = float(np.linalg.norm(cents["novice"] - cents["experienced"]) / 16.0)
            row.append(f"consensus shift novice->experienced: {shift:.2f} dB")
        lines.append("  |  ".join(row))

    # ---- C. context --------------------------------------------------------
    lines.append("")
    lines.append("C. CONTEXT (guitar vs full-mix stimulus; classify rule in docstring):")
    for d in ("warm", "bright"):
        ctx = defaultdict(list)
        for r in by_desc[d]:
            c = classify_context(meta.get(r["id"], {}).get("instrument", ""))
            if c:
                ctx[c].append(r)
        row = [f"  [{d}]"]
        cents = {}
        for cname, entries in sorted(ctx.items()):
            if len(entries) < 15:
                row.append(f"{cname}: n={len(entries)} (below floor, skipped)")
                continue
            curves = np.array([render_curve(r["params"]) for r in entries])
            spread = float(np.mean([np.linalg.norm(a - b) / 16.0
                                    for i, a in enumerate(curves)
                                    for b in curves[i + 1:]]))
            cents[cname] = curves.mean(axis=0)
            row.append(f"{cname}: n={len(entries)} spread={spread:.2f} dB")
        if len(cents) == 2:
            ks = list(cents)
            shift = float(np.linalg.norm(cents[ks[0]] - cents[ks[1]]) / 16.0)
            row.append(f"context shift: {shift:.2f} dB")
        lines.append("  |  ".join(row))

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_I.write_text(
        "HUMAN STRUCTURE INTERPRETATION — Stage 9\n\n"
        f"A. Best-k by BIC: {modes_found}. If k>1 for a descriptor, the human\n"
        "   'consensus' is a mixture and the centroid is a compromise between\n"
        "   schools — this must be reported alongside the C_AGG/centroid arms\n"
        "   as a self-identified validity bound (a mean of modes may match no\n"
        "   individual school). If k=1, the centroid interpretation stands.\n"
        "B. Expertise contrasts are conditional on the messy self-reported\n"
        "   field (~half blank); treat as exploratory descriptives.\n"
        "C. Context shift vs within-group spread: a shift comparable to the\n"
        "   spread means descriptor meaning is context-dependent at the same\n"
        "   order as individual disagreement — relevant to the app's\n"
        "   one-curve-per-descriptor assumption.\n"
        "All computed on TRAIN only; eval remains untouched outside Stage 6.\n",
        encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
