# Data Provenance — data/raw/

## SAFEEqualiserUserData.csv — REAL DATA, Stage 0 complete

- **Dataset:** SAFE Equaliser user data (SAFE-DB)
- **Original host:** `http://dmtlab.bcu.ac.uk/SAFE/data/SAFEEqualiserUserData.csv`
  (Birmingham City University DMT Lab — the actual file host; the better-known
  `semanticaudio.co.uk/datasets/data/` page only *linked* to it)
- **Both original hosts are dead as of 2026-07-19:** semanticaudio.co.uk times
  out on HTTP and HTTPS; dmtlab.bcu.ac.uk no longer resolves in DNS.
- **Retrieved from:** Internet Archive Wayback Machine, snapshot 2020-01-22:
  `https://web.archive.org/web/20200122215408id_/http://dmtlab.bcu.ac.uk/SAFE/data/SAFEEqualiserUserData.csv`
  (`id_` modifier = original bytes, no archive banner)
- **Stability evidence:** all three Wayback captures (2017-10-26, 2019-11-23,
  2020-01-22) share the identical content digest
  `BJ6ZZOZWLXI4F2UOMTNPZJHR26NJ7HMC` — the file was frozen upstream for at
  least 2.25 years before the host died. We hold the final version.
- **Retrieved:** 2026-07-19, by Patrick Keough (via Claude session)
- **SHA-256 (as downloaded):**
  `CC43858680115070FFDFA41A1158FCA1344DB2E0096BA73FE68990E90A284AF1`
- **Size / rows:** 479,130 bytes; 1,700 data rows; 25 columns (matches
  `preprocess_safe.py` parser assumptions: descriptor at index 1, params at
  indices 5–17)
- **License / usage terms:** none stated on the archived pages. The dataset was
  published openly for research use by its authors (ISMIR 2014 paper, below);
  no restrictions were attached at the source. Cite the paper.
- **Citation:** R. Stables, S. Enderby, B. De Man, G. Fazekas, J. D. Reiss,
  "SAFE: A system for the extraction and retrieval of semantic audio
  descriptors", ISMIR 2014.

### Real-vs-mock sanity gate — PASSED 2026-07-19

| Check | Mock signature | This file |
|---|---|---|
| Rows | exactly 150 | 1,700 ✓ |
| Distinct descriptors | exactly 6 | 346 (free-text, case variants, junk terms) ✓ |
| IPs | all `192.168.1.x` | varied real/institutional; zero `192.168.*` ✓ |
| Metadata | uniform UK/Professional/British | non-uniform ✓ |

### Preliminary shape (audit-lite; formal Stage 1 audit to confirm)

After trim/lowercase normalization and dropping the junk descriptor "test"
(181 entries): 1,519 usable entries. Distribution is two-giants-plus-long-tail:
**warm 532, bright 504**, no other descriptor ≥ 10, eleven descriptors at 5–9,
314 below 5. This matches the known shape of SAFE-DB in the literature (flowEQ
trained on the warm/bright subsets for the same reason) — reproducing it is
evidence the download is the genuine dataset.

### Also available in the same Wayback directory (not yet retrieved)

`SAFEEqualiserAudioFeatureData.csv` (spectral features around each EQ event),
equivalents for Compressor/Distortion/Reverb plugins, and `all.zip` (~4 MB
bundle) — relevant to Study 2; retrieve with same provenance discipline if
needed.

## Rules

- Files here are read-only after checksumming. Cleaning happens downstream in
  `pipeline/`, never in place.
- The synthetic mock CSV in `EmbeddingTestIsolated/data/` must never be copied
  into this directory (DECISION_LOG D-002).
