"""Stage 14 — publication figures (visualization only; no new inferential tests).

Every plotted quantity is re-derived from metrics.jsonl + the hash-verified
frozen splits + the validated renderer — the SAME sources the stats used — so
a figure can never disagree with a reported number. Arms D/F are optional:
each figure degrades gracefully if they are absent, and re-running after the
LoRA lands folds them in automatically.

Figures:
  fig1_hrp_by_arm         headline: HRP distribution per arm vs the human band
  fig2_rag_effect         zero-shot B vs individual-RAG C vs centroid-RAG C_AGG
  fig3_band_anatomy       per-band |gain error| + sign-agreement (H3 decomp)
  fig4_dispersion         model-cloud spread vs human spread (under-dispersion)
  fig5_response_overlay   rendered mean curves: humans vs A vs B vs FLAT

Colorblind-safe Okabe-Ito palette. D/F slots pre-reserved so colors are stable
across the with-/without-LoRA reruns.
"""

import hashlib
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
RESULTS = BASE / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("render", BASE / "pipeline" / "03_render.py")
_render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_render)
render_curve = _render.render_curve
FREQ_GRID = np.array(_render.freq_grid()) if hasattr(_render, "freq_grid") else \
    np.logspace(np.log10(20), np.log10(20000), 256)

GAIN_KEYS = ["low_shelf_gain", "band1_gain", "band2_gain", "band3_gain", "high_shelf_gain"]
BAND_LABELS = ["low\nshelf", "band 1", "band 2", "band 3", "high\nshelf"]

# Okabe-Ito. D/F reserved so reruns keep colors stable.
COLOR = {
    "A": "#0072B2", "A'": "#56B4E9", "B": "#E69F00", "C": "#D55E00",
    "C_AGG": "#009E73", "E": "#CC79A7", "FLAT": "#555555",
    "D": "#F0E442", "F": "#000000", "human": "#999999",
}
ARM_ORDER = ["FLAT", "A", "A'", "E", "B", "C", "C_AGG", "D", "F"]
ARM_NICE = {"FLAT": "Flat\n(null)", "A": "A\nretrieval", "A'": "A'\nkeyword",
            "E": "E\nMLP", "B": "B\nLLM 0-shot", "C": "C\nRAG-indiv",
            "C_AGG": "C_AGG\nRAG-centroid", "D": "D\nLoRA", "F": "F\nRAFT"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def verify_split(name):
    p = SPLITS / f"{name}.jsonl"
    exp = (SPLITS / f"{name}.sha256").read_text(encoding="utf-8-sig").strip()
    if hashlib.sha256(p.read_bytes()).hexdigest().upper() != exp:
        raise SystemExit(f"ABORT: {name}.jsonl hash mismatch")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def load_jsonl(name):
    p = RESULTS / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l] if p.exists() else []


def load_metrics():
    rows, energy = [], []
    for l in (RESULTS / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if not l:
            continue
        r = json.loads(l)
        (energy if r.get("row_type") == "energy" else rows).append(r)
    return rows, energy


def boot_ci(vals, stat=np.median, n=2000, seed=42):
    if not vals:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    a = np.asarray(vals, float)
    bs = [stat(rng.choice(a, len(a), replace=True)) for _ in range(n)]
    return stat(a), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def present_arms(rows):
    have = {r["arm"] for r in rows if "hrp" in r}
    return [a for a in ARM_ORDER if a in have]


# ---------------------------------------------------------------- fig 1
def fig1_hrp_by_arm(rows):
    arms = present_arms(rows)
    data = {a: [r["hrp"] for r in rows if r.get("arm") == a and "hrp" in r] for a in arms}
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(arms)), 5))
    ax.axhspan(25, 75, color=COLOR["human"], alpha=0.18, zorder=0)
    ax.axhline(50, color=COLOR["human"], lw=1.4, ls="--", zorder=1,
               label="typical human (HRP 50)")
    for i, a in enumerate(arms):
        v = data[a]
        parts = ax.violinplot(v, positions=[i], widths=0.8, showmedians=False,
                              showextrema=False)
        for b in parts["bodies"]:
            b.set_facecolor(COLOR[a]); b.set_alpha(0.55); b.set_edgecolor(COLOR[a])
        med, lo, hi = boot_ci(v)
        ax.plot([i], [med], "o", color=COLOR[a], zorder=5)
        ax.plot([i, i], [lo, hi], color=COLOR[a], lw=2.4, zorder=5)
        ax.annotate(f"{med:.0f}", (i, med), textcoords="offset points",
                    xytext=(9, 0), fontsize=9, color=COLOR[a], fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([ARM_NICE[a] for a in arms], fontsize=8.5)
    ax.set_ylabel("Human-Relative Percentile  (lower = more central)")
    ax.set_ylim(-4, 104)
    ax.set_title("How central to human consensus is each method's EQ?\n"
                 "HRP 0 = closer than every human · 50 = typical human · 100 = outside all humans",
                 fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig1_hrp_by_arm.png"); plt.close(fig)
    return {a: boot_ci(data[a])[0] for a in arms}


# ---------------------------------------------------------------- fig 2
def fig2_rag_effect(rows):
    trio = [a for a in ("B", "C", "C_AGG") if any(r.get("arm") == a for r in rows)]
    if len(trio) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 4.6))
    labels = {"B": "B\nzero-shot\n(no context)", "C": "C\nRAG\nindividual curves",
              "C_AGG": "C_AGG\nRAG\ncentroid summary"}
    for i, a in enumerate(trio):
        v = [r["hrp"] for r in rows if r.get("arm") == a and "hrp" in r]
        med, lo, hi = boot_ci(v)
        ax.bar(i, med, color=COLOR[a], alpha=0.85, width=0.62)
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], color="black",
                    capsize=5, lw=1.4)
        ax.annotate(f"{med:.0f}", (i, med), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=10, fontweight="bold")
    b_med = boot_ci([r["hrp"] for r in rows if r.get("arm") == "B" and "hrp" in r])[0]
    ax.axhline(b_med, color=COLOR["B"], ls=":", lw=1.2, alpha=0.7)
    ax.set_xticks(range(len(trio))); ax.set_xticklabels([labels[a] for a in trio], fontsize=8.5)
    ax.set_ylabel("Median HRP  (lower = better)")
    ax.set_title("Does retrieval grounding help?\nFeeding real human curves as context",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig2_rag_effect.png"); plt.close(fig)
    return True


# ---------------------------------------------------------------- fig 3
def fig3_band_anatomy(train, eval_set):
    consensus = {}
    g = defaultdict(lambda: defaultdict(list))
    for r in train:
        for k in GAIN_KEYS:
            g[r["descriptor"]][k].append(r["params"][k])
    for d, gg in g.items():
        consensus[d] = {k: statistics.mean(v) for k, v in gg.items()}
    eval_desc = {r["descriptor"] for r in eval_set}

    arms_src = [("B", "arm_b"), ("A", "arm_a"), ("E", "arm_e")]
    err = {a: [[] for _ in GAIN_KEYS] for a, _ in arms_src}
    agree = {a: [[] for _ in GAIN_KEYS] for a, _ in arms_src}
    for a, src in arms_src:
        for r in load_jsonl(src):
            if r.get("target_split") != "eval" or r["descriptor"] not in consensus:
                continue
            if r["descriptor"] not in eval_desc:
                continue
            cons = consensus[r["descriptor"]]
            for bi, k in enumerate(GAIN_KEYS):
                c = cons[k]
                if abs(c) < 0.5:
                    continue
                err[a][bi].append(abs(r["params"][k] - c))
                agree[a][bi].append((r["params"][k] > 0) == (c > 0))
    present = [a for a, _ in arms_src if any(err[a][bi] for bi in range(5))]
    if not present:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(5); w = 0.8 / len(present)
    for j, a in enumerate(present):
        me = [np.mean(err[a][bi]) if err[a][bi] else 0 for bi in range(5)]
        ax1.bar(x + j * w, me, w, color=COLOR[a], alpha=0.85, label=a)
        ga = [100 * np.mean(agree[a][bi]) if agree[a][bi] else 0 for bi in range(5)]
        ax2.bar(x + j * w, ga, w, color=COLOR[a], alpha=0.85, label=a)
    off = w * (len(present) - 1) / 2
    for ax, ttl, yl in ((ax1, "Magnitude error per band", "mean |gain error| (dB)"),
                        (ax2, "Direction agreement per band", "sign-match with human consensus (%)")):
        ax.set_xticks(x + off); ax.set_xticklabels(BAND_LABELS, fontsize=8.5)
        ax.set_title(ttl, fontsize=10.5); ax.set_ylabel(yl); ax.legend(fontsize=8)
    ax2.axhline(50, color="gray", ls=":", lw=1, alpha=0.6)
    ax2.set_ylim(0, 105)
    fig.suptitle("Anatomy of the gap: direction (right) vs magnitude (off)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig3_band_anatomy.png"); plt.close(fig)
    return True


# ---------------------------------------------------------------- fig 4
def fig4_dispersion(energy):
    rows = [e for e in energy if e.get("dispersion_ratio") is not None]
    if not rows:
        return None
    arms = [a for a in ("B", "C", "C_AGG", "D", "F") if any(e["arm"] == a for e in rows)]
    descs = sorted({e["descriptor"] for e in rows})
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(arms)), 4.6))
    # human spread reference (same per descriptor regardless of arm)
    hum = {}
    for e in rows:
        hum[e["descriptor"]] = e["human_dispersion_db"]
    hbar = np.mean(list(hum.values()))
    ax.axhline(hbar, color=COLOR["human"], lw=1.6, ls="--",
               label=f"human spread ({hbar:.2f} dB)")
    x = np.arange(len(arms)); w = 0.8 / max(len(descs), 1)
    for j, d in enumerate(descs):
        vals = [np.mean([e["cloud_dispersion_db"] for e in rows
                         if e["arm"] == a and e["descriptor"] == d]) for a in arms]
        ax.bar(x + j * w, vals, w, alpha=0.85, label=f"{d}")
    ax.set_xticks(x + w * (len(descs) - 1) / 2)
    ax.set_xticklabels([ARM_NICE[a].replace("\n", " ") for a in arms], fontsize=8)
    ax.set_ylabel("within-cloud spread (dB)")
    ax.set_title("Do models vary like humans do?\nModel-cloud spread vs human spread",
                 fontsize=10.5)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig4_dispersion.png"); plt.close(fig)
    return True


# ---------------------------------------------------------------- fig 5
def fig5_response_overlay(train, eval_set):
    eval_by = defaultdict(list)
    for r in eval_set:
        eval_by[r["descriptor"]].append(np.array(render_curve(r["params"])))
    descs = [d for d in ("warm", "bright") if d in eval_by]
    if not descs:
        return None
    arm_a = {r["descriptor"]: r["params"] for r in load_jsonl("arm_a")
             if r.get("target_split") == "eval"}
    arm_b = defaultdict(list)
    for r in load_jsonl("arm_b"):
        if r.get("target_split") == "eval":
            arm_b[r["descriptor"]].append(np.array(render_curve(r["params"])))
    flat = np.array(render_curve({k: (0.0 if "gain" in k else v) for k, v in
                    {"low_shelf_gain": 0, "low_shelf_freq": 120, "band1_gain": 0,
                     "band1_freq": 250, "band1_q": 0.71, "band2_gain": 0,
                     "band2_freq": 1000, "band2_q": 0.71, "band3_gain": 0,
                     "band3_freq": 3500, "band3_q": 0.71, "high_shelf_gain": 0,
                     "high_shelf_freq": 10000}.items()}))
    fig, axes = plt.subplots(1, len(descs), figsize=(6.2 * len(descs), 4.4), squeeze=False)
    for ci, d in enumerate(descs):
        ax = axes[0][ci]
        H = np.vstack(eval_by[d]); hmean = H.mean(0); hsd = H.std(0)
        ax.fill_between(FREQ_GRID, hmean - hsd, hmean + hsd, color=COLOR["human"],
                        alpha=0.25, label="human ±1 SD")
        ax.plot(FREQ_GRID, hmean, color=COLOR["human"], lw=2.4, label="human mean")
        if d in arm_a:
            ax.plot(FREQ_GRID, render_curve(arm_a[d]), color=COLOR["A"], lw=2,
                    label="A retrieval")
        if arm_b.get(d):
            ax.plot(FREQ_GRID, np.vstack(arm_b[d]).mean(0), color=COLOR["B"], lw=2,
                    label="B LLM mean")
        ax.plot(FREQ_GRID, flat, color=COLOR["FLAT"], lw=1, ls=":", label="flat")
        ax.set_xscale("log"); ax.set_xlim(20, 20000)
        ax.set_xlabel("frequency (Hz)"); ax.set_title(f'"{d}"  (n={len(eval_by[d])} humans)')
        if ci == 0:
            ax.set_ylabel("gain (dB)"); ax.legend(fontsize=8)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    fig.suptitle("What each method actually drew (rendered frequency response)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGDIR / "fig5_response_overlay.png"); plt.close(fig)
    return True


def main():
    train = verify_split("train")
    eval_set = verify_split("eval_set")
    rows, energy = load_metrics()
    made = []
    medians = fig1_hrp_by_arm(rows); made.append("fig1_hrp_by_arm")
    if fig2_rag_effect(rows): made.append("fig2_rag_effect")
    if fig3_band_anatomy(train, eval_set): made.append("fig3_band_anatomy")
    if fig4_dispersion(energy): made.append("fig4_dispersion")
    if fig5_response_overlay(train, eval_set): made.append("fig5_response_overlay")
    (RESULTS / "stats" / "figures_manifest.txt").write_text(
        "FIGURES — Stage 14 (viz only, derived from metrics.jsonl + frozen splits)\n\n"
        + "\n".join(f"  {m}.png" for m in made)
        + "\n\nfig1 median HRP by arm:\n"
        + "\n".join(f"  {a:<6} {v:.1f}" for a, v in medians.items())
        + "\n\nArms D/F: "
        + ("present" if any(r.get("arm") in ("D", "F") for r in rows) else
           "ABSENT (LoRA pending) — rerun to fold in")
        + "\n", encoding="utf-8")
    print("figures:", ", ".join(made))
    print("median HRP:", {a: round(v, 1) for a, v in medians.items()})


if __name__ == "__main__":
    main()
