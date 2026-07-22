"""Stage 8 — memorization probe (D-010, D-026.3).

SAFE-DB has been public since 2014 and may be in model training corpora.
Because the headline finding is LLM hyper-centrality, "the model memorized
the dataset" is the strongest competing explanation. This probe measures it
directly with three escalating questions per model (greedy):

  Q1 STRUCTURAL   "Describe the SAFE dataset / SAFEEqualiserUserData.csv —
                  columns, layout." Tests dataset awareness.
  Q2 VERBATIM     "Reproduce the first rows of SAFEEqualiserUserData.csv."
                  Tests verbatim recall (scored against the real file).
  Q3 STATISTICAL  "For 'warm' in the SAFE equaliser dataset, what are the
                  mean low-shelf and high-shelf gains (dB)?" Tests whether
                  the model can recall the dataset's actual statistics
                  (scored vs TRAIN means; within +/-1 dB counts as recall).

Interpretation guide (written before data, to keep us honest):
  * Q1 yes + Q2 no + Q3 no  -> knows OF the dataset; hyper-centrality is not
    explained by verbatim/statistical memorization (weak contamination).
  * Q2 or Q3 yes            -> real contamination signal; the paper must
    weaken its knowledge claims accordingly and say so prominently.
  * All no                  -> no evidence of dataset-specific memorization
    (consensus knowledge more plausibly from general audio-engineering text).
All responses receipted; scoring is mechanical where possible and the raw
text is preserved for reader judgment.
"""

import hashlib
import importlib.util
import json
import re
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("h", BASE / "pipeline" / "05_llm_harness.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

OUT_RECEIPT = BASE / "results" / "receipts" / "memorization_probe.jsonl"
OUT_RESULTS = BASE / "results" / "stats" / "memorization_results.txt"

MODELS = H.CHEAP_MODELS
QUESTIONS = {
    "Q1_structural": (
        "Are you familiar with the SAFE project dataset from the Semantic Audio "
        "Feature Extraction plugins (Stables et al., ISMIR 2014)? Specifically "
        "SAFEEqualiserUserData.csv. If yes, describe its exact structure: number "
        "of columns, what each column contains, and the layout. If you are not "
        "certain, say so plainly."),
    "Q2_verbatim": (
        "Reproduce, verbatim, the first three data rows of the file "
        "SAFEEqualiserUserData.csv from the SAFE equaliser dataset. Output only "
        "the raw CSV rows. If you cannot recall them, say 'CANNOT RECALL'."),
    "Q3_statistical": (
        "In the SAFE equaliser user dataset (SAFEEqualiserUserData.csv), "
        "considering all entries labeled 'warm': what is the approximate mean "
        "LOW SHELF gain in dB, and the approximate mean HIGH SHELF gain in dB? "
        "Answer with two numbers and one sentence. If you cannot recall this "
        "dataset's statistics, say 'CANNOT RECALL'."),
}


def main() -> None:
    key = H.load_api_key()

    # ground truth for scoring
    train = [json.loads(l) for l in
             (BASE / "data" / "splits" / "train.jsonl").read_text(encoding="utf-8").splitlines() if l]
    warm = [r["params"] for r in train if r["descriptor"] == "warm"]
    true_ls = statistics.mean(p["low_shelf_gain"] for p in warm)
    true_hs = statistics.mean(p["high_shelf_gain"] for p in warm)
    raw_first_rows = (BASE / "data" / "raw" / "SAFEEqualiserUserData.csv") \
        .read_text(encoding="utf-8", errors="replace").splitlines()[:3]

    lines = ["MEMORIZATION PROBE — Stage 8",
             f"ground truth (TRAIN): warm low-shelf mean {true_ls:+.2f} dB, "
             f"high-shelf mean {true_hs:+.2f} dB", ""]
    receipts = []
    summary = {}
    for model in MODELS:
        verdicts = {}
        for qid, q in QUESTIONS.items():
            resp = H.api_call(key, model, q, 0.0)
            content = ""
            if "choices" in resp and resp["choices"]:
                content = (resp["choices"][0].get("message") or {}).get("content") or ""
            receipts.append({"model": model, "question": qid,
                             "response": content[:6000], "error": resp.get("error")})
            if qid == "Q2_verbatim":
                hit = any(row.strip() and row.strip()[:60] in content
                          for row in raw_first_rows)
                verdicts[qid] = "VERBATIM RECALL" if hit else (
                    "declined" if "CANNOT RECALL" in content.upper() else "no match")
            elif qid == "Q3_statistical":
                nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", content)][:4]
                close = any(abs(n - true_ls) <= 1.0 for n in nums) and \
                        any(abs(n - true_hs) <= 1.0 for n in nums)
                verdicts[qid] = "STATISTICAL RECALL" if close else (
                    "declined" if "CANNOT RECALL" in content.upper() else
                    f"no match (claimed: {nums})")
            else:
                aware = "25" in content or "columns" in content.lower()
                verdicts[qid] = f"awareness-signal={'yes' if aware else 'weak/no'} (read raw)"
        summary[model] = verdicts
        lines.append(f"[{model}]")
        for qid, v in verdicts.items():
            lines.append(f"  {qid}: {v}")
        lines.append("")

    n_verbatim = sum(1 for v in summary.values() if v["Q2_verbatim"] == "VERBATIM RECALL")
    n_stat = sum(1 for v in summary.values() if v["Q3_statistical"] == "STATISTICAL RECALL")
    lines.append(f"AGGREGATE: verbatim recall {n_verbatim}/{len(MODELS)} models; "
                 f"statistical recall {n_stat}/{len(MODELS)} models.")
    lines.append("Interpretation per the pre-registered guide in this script's docstring.")

    OUT_RECEIPT.write_text("\n".join(json.dumps(r) for r in receipts) + "\n", encoding="utf-8")
    OUT_RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"probe complete: verbatim {n_verbatim}/{len(MODELS)}, statistical {n_stat}/{len(MODELS)}")


if __name__ == "__main__":
    main()
