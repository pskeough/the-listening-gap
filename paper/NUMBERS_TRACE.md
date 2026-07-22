# NUMBERS_TRACE — The Listening Gap draft v1.0 (2026-07-23)

Every number in `the_listening_gap_draft.md`, mapped to its receipt. "Derived" rows show the arithmetic and both inputs' receipts. Stats paths are relative to `C:\Research\SonicEQ\`.

| # | Number in draft | Where used | Source file | Line / context |
|---|---|---|---|---|
| 1 | 1,473 clean entries | Abstract, Related Work, 3.1 | results/stats/split_results.txt | line 4: funnel "-> 1473 clean entries" |
| 2 | warm n=532, bright n=504 | Abstract, 3.1 | results/stats/audit_results.txt | lines 18-19 (top descriptors); line 55 floor list |
| 3 | 20-entry floor | Abstract, 3.1 | results/stats/audit_results.txt | line 49 "FLOOR ANALYSIS (floor = 20)"; design §3 |
| 4 | 5 exemplars in Arm C | Abstract, 3.4, 4.3 | EXPERIMENTAL_DESIGN.md | §4 arm table, Arm C "n=5 TRAIN exemplar curves" |
| 5 | median HRP: B 3.0, C 10.5, C_AGG 0.0, A 2.4, A' 2.4, E 3.8, FLAT 32.2 | Abstract, Intro, 4.1, 5 | results/stats/metrics_results.txt | lines 6-12 (HRP by arm table); confirmed stats_results.txt lines 4-9 and figures_manifest.txt lines 10-16 |
| 6 | typical human = 50 | Abstract, Intro, 3.6, 4.1 | results/stats/metrics_results.txt | lines 14-15 "~50 = typical human" (definition of HRP percentile) |
| 7 | residual human dispersion 4.38 / 4.27 dB | Abstract, 4.1, 5 | results/stats/audio_conditioning_results.txt | line 9 (warm 4.38), line 18 (bright 4.27) |
| 8 | model cloud dispersion B 0.74, C 0.92, C_AGG 0.16 dB | Abstract, 4.1, 5 | results/stats/audio_conditioning_interpretation.txt | line 10 "{B:0.74dB, C:0.92dB, C_AGG:0.16dB}" (also DECISION_LOG D-028) |
| 9 | under-dispersion factor 5.9 (warm), 5.8 (bright) | Abstract, 4.1 | DERIVED | 4.38/0.74 = 5.92; 4.27/0.74 = 5.77 (inputs: rows 7-8) |
| 10 | 22 of 24 school GOF tests survive | Abstract, 4.2 | results/stats/consolidated_fdr_results.txt | count of S10 rows: 22 YES, 2 no (lines 41-42); corroborated by DECISION_LOG D-027 "22/24 surviving GOFs" |
| 11 | B vs C HRP q=1.240e-15 (T1.1), raw p=1.068e-16, n=396 | Abstract, 4.3 | results/stats/consolidated_fdr_results.txt | line 7 (S7:T1.1B); raw p also stats_results.txt line 17 |
| 12 | context-pull C_AGG median 1.00 all six models | Abstract, 4.3, 5 | results/stats/mode_analysis_results.txt | lines 53-58 (C_AGG pull +1.00, gpt +1.01) |
| 13 | 37 of 53 tests survive BH-FDR, alpha = 0.05 | Abstract, Intro, 3.7, 4, 7 | results/stats/consolidated_fdr_results.txt | line 1 header "53 tests ... 37 survive"; alpha in header |
| 14 | 16 tests fail correction | 3.7 | DERIVED | 53 - 37 = 16 (input: row 13) |
| 15 | six models / six families slate | Abstract, 3.5 | results/stats/harness_results.txt | lines 5-10 (model list); DECISION_LOG D-025 final slate |
| 16 | 13 parameters, 5-band | Intro, 3.3, 3.4 | EXPERIMENTAL_DESIGN.md | §3 "Parameter space (13 dims per entry)" table |
| 17 | SAFE-DB citation Stables et al. ISMIR 2014 | Intro, References | data/raw/PROVENANCE.md | Citation block, lines 27-29 |
| 18 | SHA-256 CC43858680115070FFDFA41A1158FCA1344DB2E0096BA73FE68990E90A284AF1 | 3.1 | data/raw/PROVENANCE.md | line 20; verified match in results/stats/audit_results.txt line 3 and split_results.txt line 2 |
| 19 | Wayback snapshot 2020-01-22, host dmtlab.bcu.ac.uk, original-bytes modifier | 3.1 | data/raw/PROVENANCE.md | lines 11-13 (retrieval URL with id_ modifier) |
| 20 | three captures 2017-2020 share one digest | 3.1 | data/raw/PROVENANCE.md | lines 14-17 (stability evidence, digest BJ6ZZ...) |
| 21 | 1,700 rows, 25 columns | 3.1 | data/raw/PROVENANCE.md | line 22; audit_results.txt line 7 (rows in file: 1700) |
| 22 | zero rows failed validity checks | 3.1 | results/stats/audit_results.txt | lines 8-12 (all funnel steps: 0) |
| 23 | blocklist {test, 1, my}; 183 removed; "test" = 181 | 3.1 | results/stats/split_results.txt line 4 (blocklist 183); audit_results.txt line 13 (test 181); DECISION_LOG D-019 item 3 (blocklist members) | |
| 24 | 44 exact duplicates dropped | 3.1 | results/stats/split_results.txt | line 4 "duplicates dropped 44"; audit_results.txt line 90 |
| 25 | 327 distinct descriptors; no other reaches 10 | 3.1 | results/stats/audit_results.txt | line 16 (327); line 52 (10-19 bucket: 0 descriptors) |
| 26 | tail 452 entries across 323 descriptors | 3.1, 4.6 | results/stats/split_results.txt | line 11 "TAIL GENERALIZATION SET: 452 entries across 323 below-floor descriptors" |
| 27 | metadata 30-56 percent blank | 3.1 | DECISION_LOG.md D-019 item 1 ("30-56% blank"); underlying counts audit_results.txt lines 98-103 | |
| 28 | 80/20 split, seed 42 | 3.2 | results/stats/split_results.txt | line 3 "seed 42, test fraction 0.2" |
| 29 | warm train 419 / eval 105; bright train 398 / eval 99; totals 817 / 204 | 3.2 | results/stats/split_results.txt | lines 7-9 (confirmatory splits table) |
| 30 | SHA-256 hash locks on train/eval/tail | 3.2 | results/stats/split_results.txt | lines 13-16 (HASH LOCKS block) |
| 31 | eval set read by exactly one script | 3.2 | results/stats/metrics_interpretation.txt | item 1 "This script is the frozen eval sets' only reader" |
| 32 | RBJ cascade, shelf Q 0.71, fs 48 kHz, 256-point grid 20 Hz-20 kHz | 3.3 | results/stats/render_validation.txt | line 2 (fs, grid, shelf Q); design §6.1 |
| 33 | all six render checks pass; inverse cancellation max 4.16e-15 dB | 3.3 | results/stats/render_validation.txt | line 26 (4.16e-15), line 34 "ALL PASS" (6 test blocks) |
| 34 | 2-SD rejection: warm 419 to 330, bright 398 to 322 | 3.4 | results/stats/retrieval_arms_results.txt | lines 3-4 |
| 35 | A' flat fallback on 312 tail descriptors | 3.4 | results/stats/retrieval_arms_results.txt | line 10 "flat-fallback on 312 descriptors" |
| 36 | Arm E: MiniLM, (64,) hidden, 3 seeds, 817 train rows | 3.4 | results/stats/arm_e_results.txt | lines 2-3; MiniLM named in arm_e_interpretation.txt item 3 |
| 37 | Arm E cross-seed spread mean 0.08167, max 0.12734 | 3.4 | results/stats/arm_e_results.txt | line 5 |
| 38 | 1 greedy + 10 at temp 0.7; 3 prompt variants | 3.5 | EXPERIMENTAL_DESIGN.md §4 sampling protocol; §8.6 (3 frozen variants); DECISION_LOG D-008 | |
| 39 | 2,092 grid cells; 2,003 ok; 1 failed all parse attempts | 3.5 | results/stats/harness_results.txt | line 2 |
| 40 | run-1: 549 failed cells; reasoning-token and credit causes; re-run receipted | 3.5 | DECISION_LOG.md | D-024 (549 failures, causes a-c, fixes, receipts kept) |
| 41 | flagship anchors cancelled pre-analysis | 3.5, 6 | DECISION_LOG.md | D-025 (cancelled, not deferred; efficient-tier-only slate) |
| 42 | total API spend $6.39 | 3.5 | results/stats/cost_log.txt | line 13 "TOTAL ACTUAL SPEND: $6.39" |
| 43 | human split-half energy floor: warm -0.034, bright 0.058 dB | 3.6 | results/stats/stats_results.txt | lines 12-13 |
| 44 | Wilcoxon signed-rank; bootstrap 95% CIs, 10,000 resamples | 3.7 | EXPERIMENTAL_DESIGN.md §7 (10,000 resamples); stats_results.txt (test table implements) | |
| 45 | pre-declared fallback: entry-level analysis, 2 descriptors above floor | 3.7 | results/stats/audit_interpretation.txt item 4a; stats_interpretation.txt item 1; design §7 power note | |
| 46 | no LLM judge | Intro, 3.7 | DECISION_LOG.md | D-007 |
| 47 | memorization probe: verbatim 0/6, statistical 0/6; awareness yes; public since 2014 | 3.7 | results/stats/memorization_results.txt | line 34 (AGGREGATE); Q1 awareness lines per model; "public since 2014" design §8.3 / D-010 |
| 48 | B CI [3.0, 3.8]; C CI [8.1, 13.3]; C_AGG CI [0.0, 4.8] | 4.1 | results/stats/stats_results.txt | lines 6-8 |
| 49 | n=396 (B, C), n=132 (C_AGG) | 4.1, 4.3 | results/stats/stats_results.txt | lines 6-8; metrics_results.txt lines 8-10 |
| 50 | "more central than 97 percent of individual engineers" | 4.1 | DERIVED | 100 - 3.0 = 97 (input: row 5, HRP definition row 6) |
| 51 | R^2 +0.051 (warm), -0.061 (bright); permutation floors -0.119 / -0.115; 78 features | 4.1, 6 | results/stats/audio_conditioning_results.txt | warm block lines 4-8 (R^2 +0.051, null -0.119, 78/80 usable); bright block lines 13-17 (-0.061, -0.115) |
| 52 | flat null HRP 32.2 inside human band caveat | 4.1 | results/stats/metrics_results.txt line 12; caveat framing DECISION_LOG D-029 | |
| 53 | GMM: warm k=3 (16/53/31%), bright k=4 (18/25/30/27%) | 4.2 | results/stats/human_structure_results.txt | lines 4-12 (BIC best k; mode shares) |
| 54 | gemini warm B 33/33 in one mode, chi2 p=1.17e-16 | 4.2 | results/stats/mode_analysis_results.txt | line 12 "[B] gemini-3.5-flash warm n=33 modes 0/0/33 chi2 p=1.17e-16" |
| 55 | claude-haiku bright B 32/33 in one mode | 4.2 | results/stats/mode_analysis_results.txt | line 7 "modes 0/0/32/1" |
| 56 | glm warm GOF failures at q=5.206e-02 (B and C) | 4.2 | results/stats/consolidated_fdr_results.txt | lines 41-42 |
| 57 | T2: median LSD B 3.929 vs C 4.103 dB; p=9.207e-18; q=1.627e-16 | 4.3 | stats_results.txt line 20 (medians, raw p); consolidated_fdr_results.txt line 6 (q) | |
| 58 | T1.2: n=132, q=2.742e-08 | 4.3 | results/stats/consolidated_fdr_results.txt | line 18; raw p stats_results.txt line 18 |
| 59 | T1.3: n=132, q=1.118e-19; median C 14.2 vs C_AGG 0.0 | 4.3 | consolidated_fdr_results.txt line 4 (q); stats_results.txt line 19 (medians 14.2 / 0.0) | |
| 60 | context-pull C range 0.51 (deepseek) to 0.87 (claude-haiku) | 4.3, 5 | results/stats/mode_analysis_results.txt | lines 47-52 (C pull values per model) |
| 61 | T3 fails: median B 2.235 vs C 2.477 dB; q=5.206e-02 | 4.3 | stats_results.txt line 21 (medians); consolidated_fdr_results.txt line 43 (q, "no") | |
| 62 | T5 family divergence q=1.802e-19; medians haiku 2.0, gemini 2.9, deepseek/gpt/qwen 3.0, glm 18.6 | 4.4 | consolidated_fdr_results.txt line 5 (q); stats_results.txt line 23 (per-model medians) | |
| 63 | T6 prompt sensitivity q=3.440e-05; p1 3.0, p2 persona 6.9, p3 3.0 | 4.4 | consolidated_fdr_results.txt line 25 (q); stats_results.txt line 24 (variant medians) | |
| 64 | band anatomy warm: low +2.07, lowmid -0.27, mid -1.46, highmid +0.56, high +0.85 dB | 4.5, 5 | results/stats/mode_analysis_results.txt | line 61 |
| 65 | band anatomy bright: low +1.06, lowmid +0.57, mid +0.08, highmid -2.23, high -1.72 dB | 4.5, 5 | results/stats/mode_analysis_results.txt | line 62 |
| 66 | T4 tail: n=359, B 6.030 vs C 6.183 dB, delta +0.096, p=9.429e-05, q=2.082e-04 | 4.6 | stats_results.txt line 22 (n, medians, delta, raw p); consolidated_fdr_results.txt line 27 (q) | |
| 67 | tail medians: A 5.422, A' 5.747, FLAT 5.747, E 5.831, C_AGG 5.938, B 6.117, C 6.285 dB | 4.6 | results/stats/method_selection_results.txt | lines 13-19 (section B) |
| 68 | 323 tail descriptors for A/A'/FLAT/E; 60 common for LLM arms and head-to-head | 4.6 | results/stats/method_selection_results.txt | lines 13-19 (n_desc per arm); line 21 (60 common descriptors) |
| 69 | surviving negative tail contrasts (A' vs E, FLAT vs E, A vs C, C vs C_AGG, A' vs C, FLAT vs C, A vs E, E vs C) | 4.6 | results/stats/consolidated_fdr_results.txt | lines 28-39 (S11 YES rows) |
| 70 | failing tail contrasts: FLAT vs B p=0.4616; FLAT vs A p=0.1294; A vs B p=0.1053; B vs C p=0.0668 | 4.6 | results/stats/method_selection_results.txt | lines 40-48 (raw p 4.616e-01, 1.294e-01, 1.053e-01, 6.680e-02); non-survival consolidated_fdr lines 44-52 |
| 71 | "nothing significantly beats flat on the tail" summary | 4.6, 5 | DECISION_LOG.md D-027 ("nothing significantly beats flat; surviving tail claims are E-worse-than-FLAT and C-worse-than-most"), grounded in rows 69-70 | |
| 72 | consensus anchor HRP 2.4 (Arm A) | 5 | results/stats/metrics_results.txt | line 6 |
| 73 | warm has "at least three identifiable schools" | 5 | results/stats/human_structure_results.txt | line 4 (best k=3) |
| 74 | D/F hang: step 30 of 460, ~166 min wall, CUDA wedge, GPU contention | 6 | DECISION_LOG.md | D-031 |
| 75 | smoke config: single seed, 4 predictions; gains railed +/-12 dB vs human warm ~+3.8/-2.7 dB | 6 | DECISION_LOG.md | D-031 (railed gains; human warm gentle values); underlying human means audit_results.txt lines 62/71 (band1_gain 3.76, high_shelf_gain -2.65 are the audited per-band values; D-031's ~+3.8/-2.7 is its own rounding, cited as the log states it) |
| 76 | CIs anti-conservative (shared draws) | 6 | results/stats/stats_interpretation.txt | item 4 |
| 77 | low-R^2 caveat wording | 6 | results/stats/audio_conditioning_interpretation.txt | CLAIM DISCIPLINE block; D-028 "honest caveat" |
| 78 | expertise metadata roughly half blank | 6 | results/stats/human_structure_results.txt | line 15 (warm unparsed/blank 258 of 419), line 16 (bright 249 of 398); framing human_structure_interpretation.txt item B |
| 79 | no listening test / no sounds-better claim | 6 | EXPERIMENTAL_DESIGN.md | §9 scope boundaries |
| 80 | LLM2Fx details (GPT-4o, Llama 3.x, Mistral-7B; 273 usable; 7-9 descriptors; MMD; arXiv:2505.20770) | 2 | RELATED_WORK.md | "closest competitor" block + comparison table |
| 81 | adjacent-work citations (arXiv IDs, venues) | 2, References | RELATED_WORK.md | adjacent cluster + factorial precedent sections (note: RELATED_WORK.md itself flags "verify citations again at writeup time") |
| 82 | figure filenames fig1-fig5 | Figures | results/figures/figures_manifest.txt | lines 3-7 |

## Numbers NOT traceable to a stats file

None. Every quantitative claim in the draft maps to a row above. Three rows are arithmetic derivations (9, 14, 50) with both inputs receipted. Row 75's "+3.8/-2.7 dB" and row 27's "30-56%" trace to DECISION_LOG entries rather than a `results/stats/` file; they are quoted as the log states them and flagged here for the pre-submission claim audit. Row 81's external citations come from RELATED_WORK.md, which itself instructs re-verification at write-up; they are literature pointers, not experimental results.
