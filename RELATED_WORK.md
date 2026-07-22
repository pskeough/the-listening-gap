# Related Work & Novelty Scan — 2026-07-19

Product of a dedicated web-research agent (Sonnet) tasked with finding anything
resembling this study. Verdict summary at bottom. This file seeds the paper's
related-work section; every entry below was found and characterized this
session — verify citations again at writeup time.

## The closest competitor — MUST be cited and positioned against

**LLM2Fx** — Doh, Koo, Martínez-Ramírez, Liao, Nam, Mitsufuji (Sony AI),
WASPAA 2025, arXiv:2505.20770, github.com/SonyResearch/LLM2Fx.
Tests whether off-the-shelf LLMs (GPT-4o, Llama 3.x, Mistral-7B) can predict
EQ (6-band) and reverb parameters zero-shot from semantic descriptors, plus
three in-context boosts (DSP features, DSP code, few-shot examples).

Same research question as ours. The structural differences that keep our
study alive:

| Axis | LLM2Fx | Ours |
|---|---|---|
| Corpus | SocialFX (Northwestern; filtered to 273 usable, 7–9 descriptors) | SAFE-DB (QMUL/BCU; 1,519 usable; untouched by any LLM paper) |
| Metric | MMD between audio-embedding distributions | Rendered-curve log-spectral distance + energy distance + human-relative percentile |
| Training axis | NONE — prompting only | LoRA fine-tune (D), RAFT (F) — full 2×2 factorial |
| Retrieval baseline | none standalone | embedding-centroid retrieval (A) + deployed keyword baseline (A′) |
| Judge | n/a | none (objective gold) — stated as a design feature |

**Risk noted:** Sony AI is active in this line (LLM2Fx-Tools, ICLR 2026,
arXiv:2512.01559 — audio-input tool-calling with SFT). A fine-tuning
extension from them is plausible. Argues for executing fast.

## Adjacent cluster (same problem space, different question)

- **InstructFX2FX** (DAFx26, arXiv:2606.22005): multi-turn iterative
  refinement ("make it warmer") via LLM planner + CLAP optimization, SocialFX,
  MMD. Not single-shot knowledge probing.
- **Text2FX** (Northwestern, ICASSP 2025, arXiv:2409.18847): CLAP-gradient
  test-time optimization, listener-panel evaluation. Not an LLM-knowledge
  study.
- **Deng, Pardo, Pappas** (arXiv:2510.14249): do CLAP-family embeddings align
  with human timbre perception? Finds they often DON'T — supporting evidence
  for our hypothesis space. Representational probe only, no generation.
- **TimberAgent** (arXiv:2603.09332, 2026): Gram-matrix retrieval for guitar
  pedal presets from audio queries. Evidence retrieval-grounded DSP control is
  an active niche; low overlap.
- **MixAssist/MixParams** (arXiv:2507.06329): conversational mixing with
  LLM-as-judge evaluation — our explicit contrast case (we have no judge).
- **Word Embeddings for Automatic EQ** (Venkatesh et al., JAES 2022,
  arXiv:2202.08898): pre-LLM word2vec→regressor; closest ancestor of our
  Arm E; no LLM/RAG/fine-tune comparison.
- **flowEQ** (Steinmetz, ~2019-20): the only prior system confirmed to train
  on SAFE-DB Equaliser (β-VAE). No LLM-era follow-up found on SAFE-DB —
  the corpus has been idle since.

## 2×2 factorial precedent (methodology positioning)

Established on discrete/classification tasks only: Ovadia et al.
(arXiv:2312.05934) zero-shot vs FT vs RAG on QA; Soudani et al. (SIGIR-AP
2024) on low-popularity QA; a 2026 medical-MCQA paper (arXiv:2604.23801) runs
an explicit 2×2 at 4B scale. **No precedent found for the factorial on
continuous numeric regression against physical/perceptual gold, and no audio
application of RAFT (arXiv:2403.10131) exists at all.**

## Novelty verdict (agent's, endorsed)

- The exact study (SAFE-DB + training×retrieval factorial + rendered-curve
  distributional metrics) has NOT been done.
- The *question* ("do LLMs know descriptor→EQ?") HAS been asked — LLM2Fx,
  2025 — so the paper must not claim to ask it first.
- Threat level: MEDIUM-HIGH on framing, LOW-MEDIUM on design. Speed matters.

## Framing directives for the manuscript

1. Lead with corpus + metrics + training axis, positioned explicitly against
   LLM2Fx ("prompting-only, SocialFX, embedding-space MMD" vs. our
   "training×retrieval factorial, SAFE-DB, physically-rendered curve
   distances").
2. Present the 2×2 as the structural contribution, citing the QA-domain
   precedents to show the pattern, then the gap (never on continuous
   psychoacoustic regression; RAFT never in audio).
3. State the no-LLM-judge design as a deliberate methodological position,
   contrasting MixAssist.

## Open option (not yet decided): SocialFX as replication corpus

SAFE-DB and SocialFX are twin, era-matched, independently-collected
descriptor↔EQ corpora from different institutions. Running the frozen
pipeline on SocialFX as an external replication would be a major rigor
upgrade (and directly comparable to LLM2Fx's substrate). Adds scope; candidate
for v2/robustness rather than tomorrow's core. Decision pending.
