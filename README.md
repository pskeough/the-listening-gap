# The Listening Gap
### Do Language Models Know What Warm Sounds Like?

Six production LLMs are asked to translate semantic audio descriptors ("warm", "bright")
into 5-band parametric EQ curves, graded against SAFE-DB — a crowdsourced corpus of real
audio-engineer settings collected in-plugin a decade before LLMs could have shaped it.
No LLM judge anywhere: the gold is numeric human behavior in rendered frequency-response space.

📄 **Working paper:** [`paper/the_listening_gap_draft.md`](paper/the_listening_gap_draft.md) · Patrick S. Keough · draft v1.0

## Headline findings

- **Models are hyper-central, not wrong:** median Human-Relative Percentile 3.0 where a
  typical engineer sits at 50 — the models collapse a genuinely contested practice onto
  its consensus curve, with 5.9×/5.8× less dispersion than real engineers (0.74 dB vs
  4.38/4.27 dB residual human disagreement).
- **RAG makes it worse:** grounding with 5 individual human exemplars *degrades* centrality
  (HRP 3.0 → 10.5, q=1.2e-15), while grounding with the consensus centroid snaps the model
  onto the anchor (HRP 0.0, context-pull 1.00). **Models defer to aggregates and are
  destabilized by instances** — the exportable design finding for numeric RAG.
- **Human practice is multimodal and models miss the mixture:** 3 warm / 4 bright "schools"
  in the data; models concentrate into single modes (22/24 goodness-of-fit tests significant).
- **On rare descriptors, nothing beats a flat curve** — deployed text-to-EQ tools serving
  open vocabulary are, on this evidence, decorating a no-op for most requests.
- 37/53 tests survive consolidated Benjamini–Hochberg FDR (α=.05); the 16 failures are
  reported, not hidden.

## Repository layout

```
paper/       working-paper draft + NUMBERS_TRACE.md (82 rows: every number → its receipt)
results/     stats outputs (results + interpretation pairs) and publication figures
pipeline/    seeded, numbered scripts 01–14 + byte-frozen prompts
EXPERIMENTAL_DESIGN.md   the pre-declared protocol
DECISION_LOG.md          31 dated design decisions, append-only
PROVENANCE.md            SAFE-DB acquisition trail (Wayback original-bytes, SHA-256 locked)
```

**Data note:** SAFE-DB itself is not redistributed here pending license review; PROVENANCE.md
documents the exact archived source and content hash so the corpus can be re-acquired and
byte-verified. Split files are hash-locked (SHA-256 in the pipeline).

## Context

Part of a research program auditing LLM behavior with psychometric method — this study
extends the program's recurring result (aggregate metrics hide fine-grained distortion;
"direction known, distribution lost") from clinical severity scores into decibels.
Program index: [Research_Collection_Patrick_Keough](https://github.com/pskeough/Research_Collection_Patrick_Keough).

## Authorship note

Draft prepared with AI assistance under the author's direction; all research questions,
experimental design, analysis decisions, and claims are the author's own, and every number
traces to a receipt in `results/stats/` (see the trace table).

## License

Code: Apache-2.0 · Paper text and figures: CC BY 4.0
