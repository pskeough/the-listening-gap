# Decision Log — Sonic_EQ

Every design decision with justification, dated. After design freeze (Stage 2),
this file is the only legitimate channel for protocol changes. Entries are
append-only; a reversed decision gets a new entry, never an edit.

---

## D-001 — 2026-07-19 — Scope: descriptor→curve only

**Decision:** The paper evaluates semantic descriptor (+ instrument/genre
metadata) → parametric EQ curve. The live Spotify/Last.fm pipeline, tag
harvesting, and EqualizerAPO deployment are out of scope.
**Justification:** The track→descriptor link in the deployed app is a separate,
untested mapping. Bundling it in would attach an unvalidated component to every
claim. Scoping to descriptor→curve makes the gold data (SAFE-DB) directly
applicable and keeps every claim testable against it.

## D-002 — 2026-07-19 — ⚠ Local SAFE CSV is synthetic mock; real data acquisition is a blocker

**Decision:** `EmbeddingTestIsolated/data/SAFEEqualiserUserData.csv` (150 rows) is
declared unusable for any reported result. Stage 0 = acquire real SAFE-DB.
**Justification:** Code inspection of `preprocess_safe.py` shows a
`generate_mock_dataset()` fallback that fabricates exactly this file when the real
one is missing. The CSV matches the mock signature precisely: 150 rows, sequential
IDs, `192.168.1.x` IPs, `hash_<int>` fields, uniform "United Kingdom /
Professional / British" metadata, and exactly the 6 hardcoded mock descriptors.
Publishing results on it would be publishing results on data our own script
invented. This is the single most important rigor catch in the project so far.
**Consequence:** All Ns, floors, and budget numbers in the design are provisional
until the Stage 1 audit of real data.

## D-003 — 2026-07-19 — Honesty arm A′: deployed system is keyword overlap, not embeddings

**Decision:** The paper's primary retrieval arm (A) is implemented as true
embedding-based retrieval; the deployed system's keyword-overlap weighting ships
as a separately-reported secondary arm (A′), described accurately.
**Justification:** `embed_song_predictor.py` computes bag-of-words intersection
("cosine-style" in a docstring, but not cosine, not embeddings). Describing the
deployed system as embedding retrieval would be a false methods claim. Reporting
both preserves the systems narrative and the truth.

## D-004 — 2026-07-19 — Evaluation in rendered frequency-response space

**Decision:** All primary metrics computed on rendered |H(f)| dB curves
(RBJ biquad cascade, 256-point log grid, 20 Hz–20 kHz, fs 48 kHz), not raw
parameters.
**Justification:** Parameter space is degenerate — distinct (freq, gain, Q)
vectors can be near-identical acoustically. Distances in parameter space punish
differences no listener could hear and are not defensible under review. The
rendering module is unit-tested against analytic filter responses before use
(instrument validation precedes measurement).

## D-005 — 2026-07-19 — Human-Relative Percentile as the mean-artifact control

**Decision:** Primary centrality claims are made on HRP (prediction's centrality
percentile within the leave-one-out human-to-human centrality distribution), not
raw distance-to-gold. Arm A's centrality is reported but framed as a consensus
anchor / upper bound, never as "beating" other arms on centrality.
**Justification:** A per-descriptor centroid minimizes expected squared distance
to held-out samples *by mathematical construction* — raw RMSE would manufacture
Arm A's win. HRP re-expresses every arm on the same human-consistency scale.
Complementary distribution-coverage metrics (energy distance) give the LLM arms a
genuine path to win (a point mass has zero variance; humans don't), keeping the
design falsifiable in both directions rather than rigged for retrieval.

## D-006 — 2026-07-19 — Per-descriptor stratified 80/20 split; train-only statistics; hash-locked eval set

**Decision:** Entry-level 80/20 within each descriptor, seed 42; all cleaning,
outlier rejection, centroids, PCA/covariance fitted on TRAIN only; frozen
`eval_set.jsonl` + SHA-256; secondary whole-descriptor holdout for
generalization.
**Justification:** SAFE-DB is both retrieval corpus and gold — without a split,
Arm A interpolates from the test distribution (circularity). Hash-locking follows
the established pipeline-hygiene convention (re-runs provably against identical
inputs).

## D-007 — 2026-07-19 — No LLM judge anywhere in the pipeline

**Decision:** All scoring is against objective human gold via deterministic
metrics. No model judges any output.
**Justification:** The gold is numeric human behavior, so judge-based scoring is
unnecessary — and its absence removes the judge-circularity / self-preference
attack surface entirely, which is the hardest-to-defend element of prior work.
State this explicitly in methods as a design strength.

## D-008 — 2026-07-19 — Cross-family model slate via OpenRouter; sampling protocol

**Decision:** 4 architecturally distinct families (provisional: GPT-5.x, Claude
Opus 4.x, Gemini, DeepSeek); per cell 1 greedy + 10 @ temp 0.7; ≥ 2 valid samples
for a clean cell; exact model IDs pinned at Stage 2 freeze.
**Justification:** Cross-family validation is the established pattern for
catching family-specific priors (PsychBench precedent: single-family studies miss
divergence entirely). Greedy gives the point estimate; the temp-0.7 cloud gives
the distribution estimate needed for coverage metrics (H5).

## D-009 — 2026-07-19 — Statistics: descriptor-paired nonparametrics + mixed-effects, BH-FDR

**Decision:** Wilcoxon signed-rank across descriptors (primary) + mixed-effects
convergent check; Cliff's delta + bootstrap CIs; Benjamini-Hochberg FDR α = 0.05
over the entire test battery with the full raw-p/adjusted-q table reported.
**Justification:** Descriptor is the natural paired unit; nonparametrics avoid
normality assumptions on distance distributions. BH over Bonferroni per standing
methodology (exploratory multi-test design; Bonferroni overcorrects). Reporting
tests that fail correction is claim-audit discipline.

## D-010 — 2026-07-19 — Memorization probe for public-data contamination

**Decision:** Dedicated probe batch testing whether models can reproduce SAFE-DB
rows/statistics; results reported regardless of outcome.
**Justification:** SAFE-DB is public since 2014 and plausibly in training
corpora. Bias direction is conservative for our headline hypotheses
(memorization *helps* the LLM arms), which should be stated — but measured, not
assumed.

## D-011 — 2026-07-19 — Direction/magnitude decomposition; binary only as derived threshold

**Decision:** Sign-agreement (binary) is computed and reported, but the primary
error measure is continuous magnitude error; the binary is explicitly a derived
threshold on the continuous scale.
**Justification:** Standing severity-over-binary posture: binary gates hide
moderate miscalibration, which is precisely where H3 predicts the interesting
failure lives ("knows the direction, misses the amount").

## D-012 — 2026-07-19 — Folder + receipt conventions

**Decision:** `Sonic_EQ/` layout: `data/raw` (read-only originals + provenance),
`data/splits`, `pipeline/` (numbered scripts), `results/stats` (paired
results/interpretation files), `results/receipts` (raw API responses, call/cost
logs). Every manuscript claim must trace to script + receipt.
**Justification:** Mirrors the claim-audit discipline from prior papers
(186-claim validation campaign), applied from day one instead of retrofitted
pre-submission.

## D-013 — 2026-07-19 — Arms D (fine-tuned small LLM) and E (embedding→MLP) added pre-freeze

**Decision:** Add a LoRA fine-tuned 1–3B open-weights arm (D) trained on TRAIN
human curves, and a classic-ML floor arm (E: frozen sentence-embedding → small
MLP). New hypotheses H6 (grounding beats scale) and H7 (exploratory D vs E).
Viability gate at Stage 1 audit: insufficient N demotes D to exploratory.
**Justification:** (Patrick's proposal, this session.) This completes the
canonical zero-shot vs. retrieval vs. fine-tune triangle with *objective* human
gold — the cleanest version of a comparison usually run with LLM judges. Arm E
pre-empts the inevitable reviewer question ("why an LLM for a 13-number
regression?") and doubles as the shippable product predictor (~MB, offline).
Compute-class asymmetry is reframed as the object of study (access-vs-scale),
with parameter counts and training cost reported per arm. Design not yet frozen,
so this is a legitimate scope addition, logged here rather than silently merged.
**Risks accepted:** fine-tune overfitting on small N (mitigated: dev slice,
early stopping, 3 seeds, variance reported; threat §8.9).

## D-014 — 2026-07-19 — Stage 0 sources resolved via live search

**Decision:** Canonical SAFE-DB source is `semanticaudio.co.uk/datasets/data/`
(~4 MB, per flowEQ README); fallbacks in order: GitHub mirrors of the CSV,
`safe-api` live dump, author contact. The CSV is confirmed **not** in the
`semanticaudio/SAFE` plugin repo (GitHub API tree check, 2026-07-19). The
official site refused HTTPS on 2026-07-19 — plain-HTTP/browser retrieval may be
required; provenance (URL, date, checksum) recorded either way.
**Justification:** Grounding Stage 0 in verified-today sources instead of
recalled ones; the HTTPS failure and repo absence are exactly the kind of detail
that wastes a day if discovered mid-execution.

## D-015 — 2026-07-19 — Product tag-acquisition direction: local audio-derived tags

**Decision:** For the *product* (out of paper scope, D-001): treat Spotify
acoustic APIs as permanently dead (deprecated 2024-11-27, no replacement as of
2026-07), keep Last.fm only as interim, and target a fully local chain — audio →
Essentia/CLAP tagger → descriptor → small local predictor (Arm D/E winner).
Documented with sources in TAG_ACQUISITION_2026.md; seeds Study 2 (end-to-end
audio→descriptor→curve against human gold).
**Justification:** A sellable app cannot depend on a legacy free API or a
platform that already removed its acoustic endpoints; local models (CLAP/MERT/
Essentia, benchmark-verified 2026) remove API cost, rate limits, and ToS risk,
and CLAP's zero-shot text scoring emits the SAFE descriptor vocabulary directly.

## D-016 — 2026-07-19 — Arm F (RAFT) added; design upgraded to a 2×2 factorial

**Decision:** Add Arm F: the same small open-weights model as Arm D, LoRA
fine-tuned RAFT-style — trained *with* retrieved TRAIN exemplars (plus
distractor exemplars from other descriptors) in context, evaluated with
retrieval at inference. Conditional on the same Stage 1 viability gate as
Arm D. This completes a 2×2 factorial: training {none, fine-tuned} × retrieval
{none, in-context} → B (none/none), C (none/RAG), D (FT/none), F (FT/RAG).
New hypothesis H8: F ≥ D and F ≥ C (training the model to *use* retrieved
human curves beats either mechanism alone).
**Justification:** (Patrick's proposal.) Converts an arm list into a factorial
design — main effects and interaction of grounding mechanisms become estimable,
which is a stronger contribution than pairwise comparisons. Direct
methodological continuity with the existing Train_LLM RAFT/CoT work, so
tooling and intuition carry over. Marginal cost is one more LoRA run per seed
on hardware already required for Arm D.
**Risks accepted:** doubles local training runs; if the Stage 1 gate demotes D,
F demotes with it (both exploratory or both confirmatory — never split, to
keep the factorial interpretable).

## D-017 — 2026-07-19 — Study-1 input unit reaffirmed: descriptor, never track

**Decision:** All Study 1 arms receive descriptor (+ instrument/genre metadata)
as input. No arm receives song/artist/album identity. Track-identity inputs
(now-playing via SMTC, screen capture, etc.) belong exclusively to Study 2 /
the app (APP_PIPELINE_BRAINSTORM.md).
**Justification:** Recurring design confusion resolved permanently: SAFE gold
is descriptor-level — no track-level gold EQ exists anywhere — so a
track-input arm would have nothing to be scored against (D-001, brainstorm §4).
Logged so the boundary survives future brainstorming sessions.

## D-018 — 2026-07-19 — Stage 0 COMPLETE: real SAFE-DB acquired via Wayback Machine

**Decision:** Real `SAFEEqualiserUserData.csv` (1,700 rows, 25 cols) retrieved
from the Internet Archive snapshot of its true host (`dmtlab.bcu.ac.uk`);
SHA-256 recorded; sanity gate passed; file locked read-only. Full chain in
`data/raw/PROVENANCE.md`.
**Justification:** Both original hosts are dead (semanticaudio.co.uk times out;
dmtlab.bcu.ac.uk gone from DNS). All three Wayback captures (2017–2020) share
one content digest, so the retrieved file is provably the final upstream
version — provenance is *stronger* than a live-site download would have been.
**Consequence — preliminary shape finding (formal Stage 1 audit to confirm):**
after normalization and dropping junk ("test", 181 entries): 1,519 usable
entries, but only TWO descriptors clear the ≥20 floor — warm (532) and bright
(504); nothing else reaches 10. This (a) auto-triggers the pre-declared §7
power fallback: entry-level mixed-effects becomes the primary analysis, not
descriptor-paired Wilcoxon; (b) reframes the paper as deep-distributional on
warm/bright + a 300-descriptor long-tail generalization set; (c) puts the
Arm D/F viability gate in tension — total-N criterion passes (1,519 ≥ 1,000),
descriptor-count criterion fails (2 < 15). Gate revision decision deferred to
Patrick at Stage 1 (options: keep gate as written → D/F exploratory; or revise
pre-freeze to entry-count basis since the descriptor-count criterion served the
now-abandoned descriptor-paired analysis). Matches the known SAFE-DB shape in
the literature (flowEQ used warm/bright subsets), which corroborates
authenticity.

## D-019 — 2026-07-19 — Patrick's affirmations (six items, given in session)

**Decision:** (1) Prompts are descriptor-only — genre/instrument metadata
excluded from all arm inputs (audit showed 30–56% blank + stimulus-label noise
like "jazz2"). (2) The 44 exact duplicates (same descriptor+IP+params) are
dropped at split time, count reported. (3) Blocklist extended to
{test, 1, my}. (4) Arm D/F viability gate KEPT AS WRITTEN → descriptor-count
criterion fails (2 < 15) → D and F are DEFERRED (exploratory, post-core; base
model pinned in D-021 so the spec is frozen even though execution waits).
(5) Core scope = Arms A, A′, B, C, E with a hard timebox: draft by end of
weekend or the project is logged to the research parking lot until September.
(6) OpenRouter spend for the B/C grid approved (tens of dollars).
**Justification:** Items 1–3 follow directly from Stage 1 audit evidence.
Item 4 reverses the earlier "revise the gate" lean after the career-context
scan: the LoRA arms are the multi-week component, and the cut-down core still
tests H6 via Arm E; keeping the pre-registered gate as written is also the
more conservative, more defensible reading. Items 5–6 implement the
bounded-pilot scope both scout reports converged on (novelty scan: move fast,
LLM2Fx is adjacent; career scan: don't displace Zhou outreach, Aug 8 drafts).

## D-020 — 2026-07-19 — Project home migrated to C:\Research\SonicEQ

**Decision:** Canonical project location is now `C:\Research\SonicEQ` (moved
from `C:\AI Coding Projects\Apps\EqualizerAI\Sonic_EQ`; a MOVED.md pointer
remains there and that copy is frozen). Raw-data SHA-256 verified identical
after copy.
**Justification:** This is a research project, not an app feature; it belongs
in the research workspace alongside the scaffold and other papers. App-side
docs (APP_PIPELINE_BRAINSTORM.md, TAG_ACQUISITION_2026.md) travel with it as
Study-2 seeds.

## D-021 — 2026-07-19 — Model pins (verified live against OpenRouter, 2026-07-19)

**Decision:** Frontier slate for Arms B/C: `openai/gpt-5.5`,
`anthropic/claude-opus-4.8`, `google/gemini-3.5-flash`,
`deepseek/deepseek-v4-pro`. Deferred Arm D/F base (spec frozen, execution
deferred): Qwen3-1.7B, LoRA, 3 seeds, local RTX 5060 Ti. Embedding model for
Arm A (tail retrieval) and Arm E: `sentence-transformers/all-MiniLM-L6-v2`
primary, nomic-embed-text-v1.5 as the Stage 7 robustness swap.
**Justification:** IDs pulled from the live OpenRouter model list this
session, not recalled. `gpt-5.5` chosen over the `gpt-5.6-luna/terra/sol`
variant lines because the variant semantics are undocumented and an ambiguous
pin is a reproducibility hazard; plain `-5.5` is the unambiguous flagship.
Gemini's newest listed text model is 3.5-flash (no 3.5-pro exists on the
router today). One model per family, four families, per D-008.

## D-022 — 2026-07-19 — PRE-DATA amendment: efficient-tier slate + flagship scale anchors

**Decision:** Amends D-021 BEFORE any Arm B/C API call was made (zero calls
receipted at amendment time). Confirmatory cross-family grid runs on each
family's efficient tier: `openai/gpt-5.4-mini` ($0.75/$4.5 per M in/out),
`anthropic/claude-haiku-4.5` ($1/$5), `google/gemini-3.5-flash` ($1.5/$9),
`deepseek/deepseek-v4-flash` ($0.098/$0.196) — prices pulled live from the
router this session. The two flagships `openai/gpt-5.5` ($5/$30) and
`anthropic/claude-opus-4.8` ($5/$25) run a small anchor subset only
(warm/bright × B/C × prompt P1 × full sampling = 88 calls) as a
within-family scale contrast. Projected total grid cost ≈ $8 (ceiling ~$15
with retries); actuals to be receipted per the cost-honesty rule.
**Justification:** Patrick's cost directive ("hella cheap"), applied the
defensible way: (a) amendment is pre-data and logged, not a silent mid-run
swap; (b) cross-family claims (RQ4) attach to the efficient tier, stated
plainly in methods; (c) the flagship anchors convert a cost cut into an
additional exploratory probe — does model scale change descriptor→EQ
knowledge within a family? — which the design previously lacked; (d) if
flagship and efficient tiers agree on warm/bright, the cheap-tier grid
inherits credibility; if they diverge, that divergence is itself a finding
and a stated limitation on RQ4 generality.

## D-023 — 2026-07-19 — ADDITIVE expansion: Qwen + GLM → 3 Western vs 3 Chinese families

**Decision:** Add `qwen/qwen3-235b-a22b-2507` ($0.09/$0.55 per M, live-priced)
and `z-ai/glm-4.7-flash` ($0.0605/$0.4) to the efficient-tier grid — same
cells as the other cheap models (~504 new cells, projected cost < $0.30).
Amendment is ADDITIVE and mid-collection but pre-analysis: the in-flight run
is untouched, no already-collected model is dropped, all collected data will
be reported, and the frozen eval set remains unread. Final slate: 3 Western
families (OpenAI, Anthropic, Google) vs 3 Chinese (DeepSeek, Qwen, GLM),
plus 2 Western flagship scale anchors.
**Justification:** (Patrick's directive on Chinese-model economics.) Beyond
cost, the balanced East/West slate upgrades RQ4 from "family divergence" to a
cross-cultural audio-priors probe — direct methodological continuity with
PsychBench's Eastern/Western divergence finding (GLM was in PsychBench's
validation set). "Warm/bright" are English pro-audio jargon; whether
Chinese-trained models reproduce the Western engineer consensus is a real
question either answer makes interesting. Dropping nothing and logging the
addition preserves the anti-cherry-picking guarantee of D-022.
**Addendum (same day, Patrick's call):** the East/West framing is DEMOTED —
it is not a focus of this paper and gets at most a sentence as an exploratory
observation. The added models are retained purely as family-coverage
robustness for RQ4. No headline, no dedicated analysis section.

## D-024 — 2026-07-19 — Run-1 failure diagnosis + execution fixes (receipted)

**Decision:** Run 1 completed 547/1096 cells. Receipt analysis attributes all
549 failures to three execution-level causes: (a) flagships gpt-5.5 and
opus-4.8 blocked wholesale by OpenRouter HTTP 402 (insufficient account
credits); (b) deepseek-v4-flash returned empty content in 746/751 attempts —
default hidden reasoning consumed the 1,500-token budget before any JSON;
(c) gemini-3.5-flash emitted valid JSON truncated at the token cap in
668/706 attempts (same reasoning-budget cause). haiku-4.5 and gpt-5.4-mini
were 252/252 clean. Fixes: `reasoning: {enabled: false}` for the affected
models (+ glm-4.7-flash preemptively), MAX_TOKENS 1500→4000 as safety net,
`--no-flagships` gate until credits are added. Failed cells re-run under the
resume logic; every failed attempt remains in receipts.
**Justification:** All three causes are response-format/account issues, not
arm-definition changes — the design is untouched. Documenting the 50%
failure-and-rerun honestly (rather than deleting receipts and pretending a
clean run) is the cost-honesty and failure-reporting rule in action.
Flagship anchor cells (88) stay pending until account top-up; if credits are
not added before analysis, the scale-anchor probe is reported as
not-collected, never quietly dropped from the methods.

## D-025 — 2026-07-19 — Flagship anchors CANCELLED; final slate is efficient-tier only

**Decision:** (Patrick's cost directive: "don't waste money, use cheap Chinese
models.") The flagship scale-anchor probe (gpt-5.5, opus-4.8; 88 cells, ~$4)
is cancelled, not deferred. Final slate = six efficient-tier production
models: gpt-5.4-mini, claude-haiku-4.5, gemini-3.5-flash, deepseek-v4-flash,
qwen3-235b-a22b-2507, glm-4.7-flash. Remaining collection cost ≈ $2.50
(Chinese models ≈ $0.30 of that); requires one minimum account top-up (~$5).
**Consequences, stated for the manuscript:** (a) the within-family scale
contrast from D-022 is dropped — no claim about flagship-tier behavior will
be made; every "LLM" claim is scoped to "production efficient-tier models";
(b) this is a wording constraint, not a validity problem — the preliminary
finding (hyper-centrality) is if anything MORE striking on cheap models;
(c) the 264 flagship 402-receipts remain in the receipt log, reported as
not-collected-by-design-change.

## D-026 — 2026-07-19 — Impact expansion (Patrick's call): four additions, incl. D/F un-deferral

**Decision:** (1) New arm **C-AGG**: identical to C but the in-context
grounding is the TRAIN consensus centroid (outlier-filtered mean, the exact
curve Arm A emits) instead of 5 individual exemplars — the mechanism probe
for the RAG-degradation finding ("aggregates vs instances"). Cheap-slate
models, P1 only. (2) **Variance-ratio/dispersion** metrics added to Stage 6
energy rows (cloud dispersion vs human dispersion). (3) **Memorization
probe** (D-010) implemented as pipeline/08_memorization_probe.py and run
before the draft. (4) **Arms D and F are UN-DEFERRED** — reversal of
D-019.4, justification: the goal changed from "quick bounded paper" to
"maximize field impact" (Patrick, this session); the marginal cost is zero
API dollars and overnight local GPU time now that the pipeline exists; the
factorial + RAFT-in-audio are the novelty scan's identified unclaimed
territory. D/F remain gate-labeled: with 2 above-floor descriptors they are
reported as the exploratory training axis, clearly flagged, not smuggled
into confirmatory FDR contrasts.
**Prompt-freeze note:** one NEW prompt file (agg_centroid_block.txt) is
added; all existing frozen prompt files are byte-unchanged and their hashes
unchanged in the regenerated manifest — additive, receipted.
**D-024 addendum (run 3 diagnosis):** gemini-3.5-flash rejects
`reasoning:{enabled:false}` with HTTP 400 "Reasoning is mandatory for this
endpoint" (receipted); its handling changes to `reasoning:{effort:low}` +
the 4,000-token cap. deepseek and glm confirmed fixed by the disable
(deepseek 0 failures in run 3 vs 746 in run 1).

## D-027 — 2026-07-19 — DATA COLLECTION CLOSED; consolidated FDR is the claims gate

**Decision:** No further arms, analyses, or exploratory tests may be added to
this study. The only permitted additions are (a) the in-flight D/F LoRA/RAFT
results (pre-registered exploratory training axis), whose tests append to the
consolidated battery on arrival, and (b) the write-up. The consolidated
BH-FDR table (Stage 12: 53 tests, 37 survive at alpha=.05) is the single
source of truth for which claims may be stated as significant.
**Justification:** The study has run 53 tests across 4 analysis stages. Every
additional exploratory pass grows the correction pool and weakens the whole
battery (garden of forking paths). The findings' remaining upside is in
synthesis, not addition. Notable post-consolidation casualties, to be
reported as such: T3 (B vs C energy distance, q=.052 — suggestive, not
claimable), GLM's individual warm GOFs (the mixture-failure claim rests on
the 22/24 surviving GOFs), and most tail pairwise contrasts — which is
consistent with the honest tail conclusion ("nothing significantly beats
flat"; the surviving tail claims are E-worse-than-FLAT and C-worse-than-
most). All headline claims survive consolidation.

## D-028 — 2026-07-19 — Audio-conditioning validity probe (the dispersion headline holds)

**Decision:** Acquired the SAFE audio-feature companion file (provenance in
data/raw/PROVENANCE_audiofeatures.md; SHA-256 recorded; 1,699/1,700 entries
join). Ran a pre-committed variance decomposition (Stage 13): does SOURCE
(pre-EQ) audio explain the human curve dispersion we headline as
"disagreement"? This is a validity probe on the dispersion claim, reported
whatever it showed — NOT a new confirmatory hypothesis (descriptive R^2 +
permutation control; does not enter the BH-FDR battery).
**Result:** Out-of-fold ridge R^2 of 78 source-audio features predicting the
rendered curve = +0.05 (warm) / -0.06 (bright), barely above the permutation
floor (-0.12 / -0.11). Residual "genuine disagreement" dispersion (~4.38 /
4.27 dB) is essentially identical to raw dispersion (~4.50 / 4.14 dB).
Target-convergence ratio 1.05 / 1.24 (humans do NOT converge on a common
acoustic output). Model cloud dispersion B=0.74, C=0.92, C_AGG=0.16 dB.
**Consequence:** The dispersion headline is STRENGTHENED, not threatened.
Human spread is genuine engineer-to-engineer disagreement, not source
conditioning, so the ~5-6x model under-dispersion holds against the fair
(residual) target. The sharpest reviewer objection ("your humans just had
different audio") is now pre-empted with a receipt. Honest caveat to state:
low R^2 could also mean the recorded summary features miss the relevant
acoustic dimension; we report the permutation-controlled number and the
caveat, and do not overclaim causation. This is the FINAL data addition
(D-027 closure holds); note the Stage 6 human-dispersion figure (6.27 dB,
mean pairwise) and the Stage 13 figure (4.50 dB, mean-to-centroid) are the
same distribution under two estimators (pairwise ≈ √2 × to-centroid) — the
manuscript will state one estimator consistently.

## D-029 — 2026-07-20 — Publication figures generated (viz only, no new tests)

**Decision:** `pipeline/14_figures.py` produces five figures into
`results/figures/` (fig1 HRP-by-arm, fig2 RAG effect, fig3 band anatomy,
fig4 dispersion, fig5 response overlay). Every plotted quantity is re-derived
from metrics.jsonl + the hash-verified frozen splits + the validated renderer
— the same sources the stats used — so figures cannot disagree with reported
numbers. Arms D/F slots are colour-reserved; figures degrade gracefully when
D/F are absent and fold them in on rerun.
**Justification:** Figures are the last unbuilt manuscript artifact and are
visualization only — no new inferential statistic is computed, so this does
not re-touch the test set as an analysis. Full-data fig1 medians: Flat 32,
A 2, A' 2, E 4, B 3, C 10, C_AGG 0 — confirming (a) zero-shot LLMs are
hyper-central, (b) individual-curve RAG (C=10) degrades vs zero-shot (B=3),
(c) the flat null at 32 sits inside the (very wide) human band, the honest
caveat that must accompany the hyper-centrality claim.

## D-030 — 2026-07-20 — Arm D/F LoRA queued as a GUARDED 2h scheduled fallback

**Decision:** Registered Windows one-time task `SonicEQ_LoRA_Fallback`
(fires ~02:07, 2h out) running `pipeline/run_lora_fallback.ps1`, which invokes
`09_finetune_lora.py` ONLY if `arm_d.jsonl`+`arm_f.jsonl` are not already
present. Logs to `results/receipts/lora_scheduled_run.log`.
**Justification:** Another session (StpTest) appeared to be live-training the
LoRA and was holding the GPU, so launching now would contend for VRAM and risk
double-producing D/F. A guarded, idempotent fallback honours Patrick's "queue
it for 2h" instruction while guaranteeing it no-ops if StpTest finishes first.
The 09_finetune_lora.py script is StpTest's artifact and was not modified by
this session.

## D-031 — 2026-07-20 — Arms D/F deferred; hung run killed; NOT in initial preprint

**Decision:** The full 3-seed D/F run hung at d_seed1 step 30/460 (CUDA context
wedged after earlier GPU contention; 100% util, zero progress for 20+ min,
~166 min wall). Process PID 10092 killed manually; GPU freed and verified
(410 MiB idle). Only surviving D output is the SMOKE-config d_seed0: 1 seed,
1 prompt, 2 samples/descriptor = 4 predictions. NOT used — it is under-sampled
vs design (3 seeds x 3 prompts x 11 samples) and the curves are pathological
(gains railed to +/-12 dB vs gentle human warm ~+3.8/-2.7 dB). Reporting it
would be an overclaim. D/F demoted to "preliminary / future work" for the
initial preprint.
**Justification:** D/F are the exploratory fine-tuning corner of the
knowledge-injection comparison; the confirmatory thesis (consensus encoding,
RAG degradation, mode collapse) is complete without them. Standing rule:
partial/failed runs reported, not deleted. To resume: needs EXCLUSIVE GPU
(the hang traced to VRAM contention with another local model). A clean single
full seed (~15 min) yields a reportable n=1-seed Arm D; 3 seeds for variance.
**Preliminary observation (not yet reportable, n=4):** the LoRA rail-slams
gains while the frozen-embedding MLP (Arm E) stays human-central — a potential
"generative fine-tune overshoots where a simple regressor doesn't" point if it
survives a clean run.
