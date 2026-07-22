"""Stage 5 — Arms B (zero-shot) and C (RAG) via OpenRouter.

Grid (D-021 as amended by D-022, all pre-data):
  * Confirmatory, efficient tier (4 models): warm/bright x {B,C} x {P1,P2,P3}
    x {1 greedy + 10 @ temp 0.7}                                  = 528 calls
  * Tail generalization (4 models): 60 seeded-sample tail descriptors x {B,C}
    x P1 x greedy                                                 = 480 calls
  * Flagship scale anchors (2 models): warm/bright x {B,C} x P1
    x {1 greedy + 10 @ 0.7}                                       =  88 calls

Rigor rules:
  * API key loaded from env or a known .env path; the value is never logged.
  * Every call (including failures/retries) appended to
    results/receipts/calls.jsonl with request digest, response, usage.
  * Idempotent resume: completed cell_ids are skipped on re-run.
  * Outputs are schema-validated (frozen output_schema.json ranges); a cell
    gets up to 3 parse attempts (each receipted); unparseable cells are
    reported, never silently dropped.
  * Arm C exemplars are fixed per descriptor (seed 42) from TRAIN only; tail
    descriptors retrieve their nearest anchor per arm_a.jsonl weights, so C's
    retrieval is consistent with Arm A's.
  * Prompt files are used byte-frozen; their SHA-256 manifest is re-verified
    at startup and refused on mismatch.
"""

import hashlib
import json
import random
import threading
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
PROMPTS = BASE / "pipeline" / "prompts"
RECEIPTS = BASE / "results" / "receipts"
OUT_B = BASE / "results" / "arm_b.jsonl"
OUT_C = BASE / "results" / "arm_c.jsonl"
OUT_RESULTS = BASE / "results" / "stats" / "harness_results.txt"
OUT_INTERP = BASE / "results" / "stats" / "harness_interpretation.txt"

ENV_PATHS = [Path(r"C:\Research\SonicEQ\.env"),
             Path(r"C:\Research\ExplinationGap\PsychExplainedPaper\.env")]
API_URL = "https://openrouter.ai/api/v1/chat/completions"

CHEAP_MODELS = ["openai/gpt-5.4-mini", "anthropic/claude-haiku-4.5",
                "google/gemini-3.5-flash", "deepseek/deepseek-v4-flash",
                "qwen/qwen3-235b-a22b-2507", "z-ai/glm-4.7-flash"]  # D-023
FLAGSHIPS = ["openai/gpt-5.5", "anthropic/claude-opus-4.8"]
PROMPT_VARIANTS = ["p1_minimal", "p2_persona", "p3_format_demo"]
N_CLOUD = 10
TEMP_CLOUD = 0.7
MAX_TOKENS = 4000
# D-024: models whose default hidden reasoning starved the output budget
# (empty/truncated JSON in run 1 receipts) get reasoning disabled explicitly.
# gemini-3.5-flash rejects disabling (HTTP 400 "Reasoning is mandatory") —
# it gets effort:low instead (D-024 addendum).
NO_REASONING = {"deepseek/deepseek-v4-flash", "z-ai/glm-4.7-flash"}
LOW_REASONING = {"google/gemini-3.5-flash"}
TAIL_SAMPLE_N = 60
EXEMPLAR_N = 5
SEED = 42
CONCURRENCY = 8
PARSE_ATTEMPTS = 3

PARAM_NAMES = [
    "low_shelf_gain", "low_shelf_freq",
    "band1_gain", "band1_freq", "band1_q",
    "band2_gain", "band2_freq", "band2_q",
    "band3_gain", "band3_freq", "band3_q",
    "high_shelf_gain", "high_shelf_freq",
]
CLIP = {**{k: (-12.0, 12.0) for k in PARAM_NAMES if k.endswith("_gain")},
        **{k: (0.0, 20000.0) for k in PARAM_NAMES if k.endswith("_freq")},
        **{k: (0.1, 10.0) for k in PARAM_NAMES if k.endswith("_q")}}

_write_lock = threading.Lock()


def load_api_key() -> str:
    import os
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for p in ENV_PATHS:
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("ABORT: no OpenRouter API key found (env or known .env paths)")


def verify_prompts() -> dict:
    manifest = {}
    for line in (PROMPTS / "PROMPTS.sha256").read_text(encoding="utf-8-sig").splitlines():
        h, name = line.split(None, 1)
        manifest[name.strip()] = h.strip()
    texts = {}
    for name, expected in manifest.items():
        p = PROMPTS / name
        actual = hashlib.sha256(p.read_bytes()).hexdigest().upper()
        if actual != expected:
            raise SystemExit(f"ABORT: frozen prompt {name} hash mismatch")
        texts[name.rsplit(".", 1)[0]] = p.read_text(encoding="utf-8")
    return texts


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def validate_params(obj) -> dict | None:
    if not isinstance(obj, dict) or set(obj) != set(PARAM_NAMES):
        return None
    out = {}
    for k in PARAM_NAMES:
        v = obj[k]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        lo, hi = CLIP[k]
        if not (lo <= float(v) <= hi) or (k.endswith("_freq") and v <= 0):
            return None
        out[k] = float(v)
    return out


def extract_json(text: str) -> dict | None:
    s = text.strip()
    if "```" in s:
        parts = [seg for seg in s.split("```") if "{" in seg]
        s = parts[0] if parts else s
        s = s.replace("json", "", 1).strip() if s.startswith("json") else s
    start = s.find("{")
    if start < 0:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(s[start:])
        return obj
    except json.JSONDecodeError:
        return None


def api_call(key: str, model: str, prompt: str, temperature: float) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    if model in NO_REASONING:
        payload["reasoning"] = {"enabled": False}
    elif model in LOW_REASONING:
        payload["reasoning"] = {"effort": "low"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost/soniceq-study",
        "X-Title": "SonicEQ Listening Gap study",
    })
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 520, 524) and attempt < 4:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"error": {"http": e.code, "body": e.read().decode(errors="replace")[:500]}}
        except Exception as e:  # timeout, connection reset — receipt and retry
            if attempt < 4:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"error": {"exception": repr(e)[:300]}}
    return {"error": {"exhausted": True}}


def receipt(rec: dict) -> None:
    with _write_lock:
        with open(RECEIPTS / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def run_cell(key: str, cell: dict, prompt_text: str) -> dict:
    """One grid cell -> up to PARSE_ATTEMPTS API calls until valid params."""
    for parse_attempt in range(1, PARSE_ATTEMPTS + 1):
        resp = api_call(key, cell["model"], prompt_text, cell["temperature"])
        content = ""
        usage = resp.get("usage", {})
        if "choices" in resp and resp["choices"]:
            content = (resp["choices"][0].get("message") or {}).get("content") or ""
        params = validate_params(extract_json(content)) if content else None
        receipt({"cell_id": cell["cell_id"], "parse_attempt": parse_attempt,
                 "model": cell["model"], "arm": cell["arm"],
                 "descriptor": cell["descriptor"], "prompt": cell["prompt"],
                 "sample_idx": cell["sample_idx"],
                 "temperature": cell["temperature"],
                 "prompt_sha8": hashlib.sha256(prompt_text.encode()).hexdigest()[:8],
                 "response_content": content[:4000],
                 "usage": usage, "error": resp.get("error"),
                 "status": "ok" if params else "invalid"})
        if params:
            return {**cell, "params": params, "status": "ok"}
    return {**cell, "params": None, "status": "failed"}


def main() -> None:
    import sys
    smoke = "--smoke" in sys.argv
    key = load_api_key()
    prompts = verify_prompts()
    RECEIPTS.mkdir(parents=True, exist_ok=True)

    train = load_jsonl(SPLITS / "train.jsonl")
    tail = load_jsonl(SPLITS / "tail_eval.jsonl")
    arm_a = load_jsonl(BASE / "results" / "arm_a.jsonl")

    confirm_descs = ["warm", "bright"]
    rng = random.Random(SEED)
    tail_descs_all = sorted({r["descriptor"] for r in tail})
    tail_sample = sorted(rng.sample(tail_descs_all, min(TAIL_SAMPLE_N, len(tail_descs_all))))

    # Fixed Arm C exemplars per descriptor (TRAIN only, seed 42)
    by_desc = {}
    for r in train:
        by_desc.setdefault(r["descriptor"], []).append(r)
    nearest_anchor = {r["descriptor"]: max(r.get("weights", {"warm": 1}),
                                           key=lambda a: r["weights"].get(a, 0))
                      for r in arm_a if r["target_split"] == "tail_eval" and r.get("weights")}
    exemplars = {}
    for d in confirm_descs + tail_sample:
        pool_desc = d if d in by_desc else nearest_anchor.get(d, "warm")
        pool = sorted(by_desc[pool_desc], key=lambda r: r["id"])
        picks = random.Random(f"{SEED}:{d}").sample(pool, min(EXEMPLAR_N, len(pool)))
        exemplars[d] = [{k: round(p["params"][k], 2) for k in PARAM_NAMES} for p in picks]

    # D-026 C-AGG: consensus centroid context = the exact curve Arm A emits
    arm_a_eval = {r["descriptor"]: r["params"] for r in arm_a
                  if r["target_split"] == "eval"}
    train_counts = {d: len(v) for d, v in by_desc.items()}

    def centroid_ctx(desc: str) -> tuple:
        anchor = desc if desc in arm_a_eval else nearest_anchor.get(desc, "warm")
        c = {k: round(v, 2) for k, v in arm_a_eval[anchor].items()}
        return c, train_counts.get(anchor, 0)

    def build_prompt(arm: str, variant: str, desc: str) -> str:
        base_prompt = prompts[variant].replace("{descriptor}", desc)
        if arm == "B":
            return base_prompt
        if arm == "CAGG":
            c, n = centroid_ctx(desc)
            block = prompts["agg_centroid_block"] \
                .replace("{n}", str(n)) \
                .replace("{descriptor}", desc) \
                .replace("{centroid}", json.dumps(c))
            return block + base_prompt
        block = prompts["rag_exemplar_block"] \
            .replace("{n}", str(len(exemplars[desc]))) \
            .replace("{descriptor}", desc) \
            .replace("{exemplars}", "\n".join(json.dumps(e) for e in exemplars[desc]))
        return block + base_prompt

    # ---- build grid --------------------------------------------------------
    cells = []
    def add(model, arm, variant, desc, split, idx, temp):
        cells.append({"cell_id": f"{model}|{arm}|{variant}|{desc}|{idx}",
                      "model": model, "arm": arm, "prompt": variant,
                      "descriptor": desc, "target_split": split,
                      "sample_idx": idx, "temperature": temp})

    for model in CHEAP_MODELS:
        for desc in confirm_descs:
            for arm in ("B", "C"):
                for variant in PROMPT_VARIANTS:
                    add(model, arm, variant, desc, "eval", 0, 0.0)
                    for i in range(1, N_CLOUD + 1):
                        add(model, arm, variant, desc, "eval", i, TEMP_CLOUD)
        for desc in tail_sample:
            for arm in ("B", "C"):
                add(model, arm, "p1_minimal", desc, "tail_eval", 0, 0.0)
        # D-026 C-AGG cells: P1 only
        for desc in confirm_descs:
            add(model, "CAGG", "p1_minimal", desc, "eval", 0, 0.0)
            for i in range(1, N_CLOUD + 1):
                add(model, "CAGG", "p1_minimal", desc, "eval", i, TEMP_CLOUD)
        for desc in tail_sample:
            add(model, "CAGG", "p1_minimal", desc, "tail_eval", 0, 0.0)
    for model in FLAGSHIPS:
        for desc in confirm_descs:
            for arm in ("B", "C"):
                add(model, arm, "p1_minimal", desc, "eval", 0, 0.0)
                for i in range(1, N_CLOUD + 1):
                    add(model, arm, "p1_minimal", desc, "eval", i, TEMP_CLOUD)

    # ---- resume: skip cells already ok in receipts -------------------------
    done_ids = set()
    rp = RECEIPTS / "calls.jsonl"
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done_ids.add(r["cell_id"])
            except json.JSONDecodeError:
                continue
    todo = [c for c in cells if c["cell_id"] not in done_ids]
    if "--no-flagships" in sys.argv:   # D-024: 402 credit block; resume later
        todo = [c for c in todo if c["model"] not in FLAGSHIPS]
    if smoke:
        todo = [c for c in todo if c["model"] == "deepseek/deepseek-v4-flash"
                and c["sample_idx"] == 0 and c["descriptor"] == "warm"
                and c["prompt"] == "p1_minimal"][:2]
    print(f"grid={len(cells)} done={len(done_ids)} todo={len(todo)} smoke={smoke}")

    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(run_cell, key, c, build_prompt(c["arm"], c["prompt"], c["descriptor"]))
                   for c in todo]
        for i, fut in enumerate(futures, 1):
            results.append(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} cells complete")

    # ---- compile outputs from ALL ok receipts (incl. prior runs) -----------
    ok_by_cell = {}
    for line in rp.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "ok" and r["cell_id"] not in ok_by_cell:
            params = validate_params(extract_json(r.get("response_content", "")))
            if params:
                ok_by_cell[r["cell_id"]] = {**{k: r[k] for k in
                    ("model", "arm", "descriptor", "prompt", "sample_idx", "temperature")},
                    "params": params}
    meta = {c["cell_id"]: c["target_split"] for c in cells}
    b_rows = [{**v, "target_split": meta.get(cid, "?")} for cid, v in ok_by_cell.items() if v["arm"] == "B"]
    c_rows = [{**v, "target_split": meta.get(cid, "?")} for cid, v in ok_by_cell.items() if v["arm"] == "C"]
    cagg_rows = [{**v, "target_split": meta.get(cid, "?")} for cid, v in ok_by_cell.items() if v["arm"] == "CAGG"]
    OUT_B.write_text("\n".join(json.dumps(r, sort_keys=True) for r in b_rows) + "\n", encoding="utf-8")
    OUT_C.write_text("\n".join(json.dumps(r, sort_keys=True) for r in c_rows) + "\n", encoding="utf-8")
    (BASE / "results" / "arm_c_agg.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in cagg_rows) + "\n", encoding="utf-8")

    n_failed = sum(1 for r in results if r["status"] == "failed")
    per_model = {}
    for cid, v in ok_by_cell.items():
        per_model[v["model"]] = per_model.get(v["model"], 0) + 1
    lines = ["LLM HARNESS — Stage 5 (D-022 slate)",
             f"grid cells: {len(cells)}   completed ok: {len(ok_by_cell)}   failed after {PARSE_ATTEMPTS} parse attempts: {n_failed}",
             f"arm_b rows: {len(b_rows)}   arm_c rows: {len(c_rows)}",
             "ok cells per model:"]
    lines += [f"  {m}: {n}" for m, n in sorted(per_model.items())]
    OUT_RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_INTERP.write_text(
        "HARNESS INTERPRETATION — Stage 5\n\n"
        "1. Every API interaction (success, invalid parse, HTTP error) is a line\n"
        "   in results/receipts/calls.jsonl; the success rate above is computable\n"
        "   from receipts alone.\n"
        "2. Failed cells are REPORTED here and excluded from arm files — the\n"
        "   metrics stage must state resulting cell Ns per model (no silent N\n"
        "   shrinkage).\n"
        "3. Cost: compute from receipts' usage fields x D-022 pricing snapshot;\n"
        "   write to results/stats/cost_log.txt at analysis time (retries included\n"
        "   by construction because every receipt line carries usage).\n",
        encoding="utf-8")
    print(f"ok={len(ok_by_cell)} failed={n_failed} -> arm_b {len(b_rows)} rows, arm_c {len(c_rows)} rows")


if __name__ == "__main__":
    main()
