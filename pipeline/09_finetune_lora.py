"""Stage 5b — Arms D (LoRA SFT) and F (RAFT) on Qwen3-1.7B, local GPU.

Spec: D-013 (arm D added), D-016 (arm F / RAFT, 2x2 factorial), D-021 (base
model pin: Qwen3-1.7B, LoRA, 3 seeds, RTX 5060 Ti), D-026 (un-deferral).

Protocol rules enforced here:
  * Training data comes from data/splits/train.jsonl ONLY. This script never
    opens eval_set.jsonl or tail_eval.jsonl. Tail descriptor NAMES come from
    results/arm_a.jsonl (target_split=="tail_eval" rows), and the 60-name
    inference sample replicates the harness exactly:
    random.Random(42).sample(sorted(set), 60).
  * Dev slice for early stopping: 10% of TRAIN, per-descriptor stratified,
    seed 42 (rng = random.Random(f"42:dev:{descriptor}"), pool sorted by id,
    n_dev = round(0.10 * n)). The dev slice is identical for every run.
  * 3 seeds (0, 1, 2) per arm; the run seed governs LoRA init, dropout, and
    epoch shuffling only — the datasets are byte-identical across seeds.
  * Every run receipted: step losses, per-epoch dev loss, hyperparams,
    wall-clock, adapter SHA-256 -> results/receipts/finetune/.

Arm D example:  user msg = p1_minimal with {descriptor}; assistant completion
  = the row's 13-key params JSON, keys in train.jsonl order, values round(,2).
Arm F example:  same, but the user msg is prefixed with a rag_exemplar_block
  containing 5 exemplars: 3 same-descriptor + 2 distractor (other descriptor),
  drawn from TRAIN-minus-dev with the example's own row excluded, seeded per
  example (random.Random(f"42:raft:{row_id}")), shuffled by the same rng.
  Exemplar dicts use the harness's PARAM_NAMES order + round(,2), matching
  the Arm C / inference-time block format byte-for-byte in style.
Arm F inference: 5 same-descriptor TRAIN exemplars via the harness's exact
  selection logic (full-TRAIN pool sorted by id, random.Random(f"42:{desc}"),
  tail descriptors mapped to their nearest anchor from arm_a weights).

Spec-silent implementation choices (documented, receipted):
  * Chat template applied with enable_thinking=False (empty think block) for
    both training and inference; loss on completion tokens only.
  * AdamW weight_decay 0.0, no LR warmup, grad-clip 1.0; cosine horizon =
    the full 5-epoch step count (early stop just ends the schedule early).
  * Sampling at temperature 0.7 uses top_p=1.0 / top_k=0 so temperature is
    the only sampling transform (parity with the OpenRouter harness calls).
  * F training exemplar pools exclude dev rows (keeps dev loss an honest
    generalization signal) and exclude the example's own row (no label leak);
    inference pools use full TRAIN to match the harness's Arm C logic.

Usage:
  python 09_finetune_lora.py            # all 6 runs + compile outputs
  python 09_finetune_lora.py --smoke    # tiny plumbing test -> scratch dir
"""

import hashlib
import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
PROMPTS = BASE / "pipeline" / "prompts"
RESULTS = BASE / "results"
FT_RECEIPTS = RESULTS / "receipts" / "finetune"
MODELS_DIR = BASE / "models" / "lora"
OUT_D = RESULTS / "arm_d.jsonl"
OUT_F = RESULTS / "arm_f.jsonl"
OUT_RESULTS = RESULTS / "stats" / "finetune_results.txt"
OUT_INTERP = RESULTS / "stats" / "finetune_interpretation.txt"

MODEL_ID = "Qwen/Qwen3-1.7B"
SEED = 42                      # protocol seed (splits, exemplars, tail sample)
RUN_SEEDS = [0, 1, 2]
DEV_FRAC = 0.10
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
LR = 1e-4
BATCH = 8                      # effective batch; micro-batch backs off on OOM
MAX_EPOCHS = 5
PATIENCE = 1                   # stop after 1 epoch without dev improvement
GRAD_CLIP = 1.0
MAX_SEQ = 2048                 # assert-only cap; nothing is truncated
MAX_NEW_TOKENS = 512
N_CLOUD = 10
TEMP_CLOUD = 0.7
TAIL_SAMPLE_N = 60
EXEMPLAR_N = 5                 # inference-time exemplars (harness parity)
RAFT_SAME_N = 3
RAFT_DISTRACT_N = 2
PARSE_ATTEMPTS = 3
LOG_EVERY = 10

# Harness (05_llm_harness.py) parameter order — used for exemplar blocks.
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


# ---------------------------------------------------------------- data utils
def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]


def validate_params(obj):
    """Copied from 05_llm_harness.py — identical validation semantics."""
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


def extract_json(text: str):
    """Copied from 05_llm_harness.py — identical extraction semantics."""
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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts) -> int:
    return zlib.crc32(":".join(str(p) for p in parts).encode())


class Spec:
    """All protocol-level data, built once, identical for every run."""

    def __init__(self):
        self.train = load_jsonl(SPLITS / "train.jsonl")
        arm_a = load_jsonl(RESULTS / "arm_a.jsonl")

        # canonical completion key order = train.jsonl file order
        orders = {tuple(r["params"].keys()) for r in self.train}
        assert len(orders) == 1, "train.jsonl params key order inconsistent"
        self.completion_order = list(orders.pop())

        self.p1 = (PROMPTS / "p1_minimal.txt").read_text(encoding="utf-8")
        self.rag_block = (PROMPTS / "rag_exemplar_block.txt").read_text(encoding="utf-8")

        # descriptor pools
        self.by_desc = {}
        for r in self.train:
            self.by_desc.setdefault(r["descriptor"], []).append(r)
        self.descriptors = sorted(self.by_desc)          # ['bright', 'warm']
        assert len(self.descriptors) == 2

        # dev slice: 10% of TRAIN, per-descriptor stratified, seed 42
        self.dev_ids = set()
        for d in self.descriptors:
            pool = sorted(self.by_desc[d], key=lambda r: r["id"])
            n_dev = round(DEV_FRAC * len(pool))
            rng = random.Random(f"{SEED}:dev:{d}")
            self.dev_ids.update(r["id"] for r in rng.sample(pool, n_dev))
        self.train_rows = [r for r in self.train if r["id"] not in self.dev_ids]
        self.dev_rows = sorted((r for r in self.train if r["id"] in self.dev_ids),
                               key=lambda r: r["id"])

        # tail descriptor names from arm_a ONLY (never tail_eval.jsonl)
        tail_descs_all = sorted({r["descriptor"] for r in arm_a
                                 if r["target_split"] == "tail_eval"})
        rng = random.Random(SEED)
        self.tail_sample = sorted(rng.sample(tail_descs_all,
                                             min(TAIL_SAMPLE_N, len(tail_descs_all))))
        self.confirm_descs = ["warm", "bright"]

        # nearest anchor for tail descriptors (harness logic, verbatim)
        self.nearest_anchor = {
            r["descriptor"]: max(r.get("weights", {"warm": 1}),
                                 key=lambda a: r["weights"].get(a, 0))
            for r in arm_a if r["target_split"] == "tail_eval" and r.get("weights")}

        # inference-time exemplars (harness logic, verbatim: full-TRAIN pool)
        self.infer_exemplars = {}
        for d in self.confirm_descs + self.tail_sample:
            pool_desc = d if d in self.by_desc else self.nearest_anchor.get(d, "warm")
            pool = sorted(self.by_desc[pool_desc], key=lambda r: r["id"])
            picks = random.Random(f"{SEED}:{d}").sample(pool, min(EXEMPLAR_N, len(pool)))
            self.infer_exemplars[d] = [
                {k: round(p["params"][k], 2) for k in PARAM_NAMES} for p in picks]

        # RAFT training exemplar pools: TRAIN minus dev, per descriptor
        self.raft_pool = {}
        for d in self.descriptors:
            self.raft_pool[d] = sorted(
                (r for r in self.by_desc[d] if r["id"] not in self.dev_ids),
                key=lambda r: r["id"])

    # -------- prompt/completion builders ---------------------------------
    def completion_text(self, row) -> str:
        return json.dumps({k: round(row["params"][k], 2)
                           for k in self.completion_order})

    def p1_prompt(self, desc: str) -> str:
        return self.p1.replace("{descriptor}", desc)

    def fill_block(self, desc: str, exemplars: list) -> str:
        return (self.rag_block
                .replace("{n}", str(len(exemplars)))
                .replace("{descriptor}", desc)
                .replace("{exemplars}",
                         "\n".join(json.dumps(e) for e in exemplars)))

    def raft_train_prompt(self, row) -> str:
        d = row["descriptor"]
        other = [x for x in self.descriptors if x != d][0]
        rng = random.Random(f"{SEED}:raft:{row['id']}")
        same_pool = [r for r in self.raft_pool[d] if r["id"] != row["id"]]
        same = rng.sample(same_pool, RAFT_SAME_N)
        distract = rng.sample(self.raft_pool[other], RAFT_DISTRACT_N)
        picks = same + distract
        rng.shuffle(picks)   # presented shuffled, unlabeled
        ex = [{k: round(p["params"][k], 2) for k in PARAM_NAMES} for p in picks]
        return self.fill_block(d, ex) + self.p1_prompt(d)

    def build_examples(self, arm: str, rows: list) -> list:
        """-> [(prompt_text, completion_text), ...]"""
        out = []
        for row in rows:
            if arm == "d":
                prompt = self.p1_prompt(row["descriptor"])
            else:
                prompt = self.raft_train_prompt(row)
            out.append((prompt, self.completion_text(row)))
        return out

    def infer_prompt(self, arm: str, desc: str) -> str:
        if arm == "d":
            return self.p1_prompt(desc)
        return self.fill_block(desc, self.infer_exemplars[desc]) + self.p1_prompt(desc)

    def infer_cells(self) -> list:
        cells = []
        for desc in self.confirm_descs:
            cells.append((desc, "eval", 0, 0.0))
            for i in range(1, N_CLOUD + 1):
                cells.append((desc, "eval", i, TEMP_CLOUD))
        for desc in self.tail_sample:
            cells.append((desc, "tail_eval", 0, 0.0))
        return cells


# ---------------------------------------------------------------- torch side
def tokenize_examples(tok, examples):
    """-> list of dicts {input_ids, labels}; loss on completion tokens only."""
    feats = []
    for prompt, completion in examples:
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(chat, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        assert len(ids) <= MAX_SEQ, f"example exceeds MAX_SEQ: {len(ids)}"
        feats.append({"input_ids": ids,
                      "labels": [-100] * len(p_ids) + list(c_ids)})
    return feats


def collate(feats, pad_id):
    import torch
    width = max(len(f["input_ids"]) for f in feats)
    ids, labels, mask = [], [], []
    for f in feats:
        n = len(f["input_ids"])
        pad = width - n
        ids.append(f["input_ids"] + [pad_id] * pad)
        labels.append(f["labels"] + [-100] * pad)
        mask.append([1] * n + [0] * pad)
    return (torch.tensor(ids), torch.tensor(labels), torch.tensor(mask))


def dev_loss(model, feats, pad_id, micro):
    """Mean CE over completion tokens across the whole dev set."""
    import torch
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(feats), micro):
            ids, labels, mask = collate(feats[i:i + micro], pad_id)
            ids, labels, mask = ids.cuda(), labels.cuda(), mask.cuda()
            logits = model(input_ids=ids, attention_mask=mask).logits
            shift_logits = logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1), ignore_index=-100, reduction="sum")
            total += loss.item()
            count += (shift_labels != -100).sum().item()
    model.train()
    return total / max(count, 1)


def load_base(tok_only=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok_only:
        return tok, None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.cuda()
    return tok, model


def train_run(spec, arm, run_seed, log_path, adapter_dir, micro_bs, smoke=False):
    """One LoRA run. Returns receipt dict (dev losses, wall clock, etc.)."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    t0 = time.time()
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)

    tok, model = load_base()
    lcfg = LoraConfig(task_type="CAUSAL_LM", r=LORA_R, lora_alpha=LORA_ALPHA,
                      lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES)
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_rows = spec.train_rows[:24] if smoke else spec.train_rows
    dev_rows = spec.dev_rows[:8] if smoke else spec.dev_rows
    train_feats = tokenize_examples(tok, spec.build_examples(arm, train_rows))
    dev_feats = tokenize_examples(tok, spec.build_examples(arm, dev_rows))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    accum = max(1, BATCH // micro_bs)
    steps_per_epoch = math.ceil(len(train_feats) / (micro_bs * accum))
    max_epochs = 1 if smoke else MAX_EPOCHS
    total_steps = steps_per_epoch * max_epochs
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=LR, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, 0, total_steps)

    log = open(log_path, "w", encoding="utf-8")
    log.write(f"# {arm.upper()} seed {run_seed} — {MODEL_ID} LoRA "
              f"r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}\n"
              f"# targets={TARGET_MODULES}\n"
              f"# lr={LR} cosine no-warmup wd=0.0 clip={GRAD_CLIP} "
              f"batch={BATCH} (micro={micro_bs} x accum={accum})\n"
              f"# train_examples={len(train_feats)} dev_examples={len(dev_feats)} "
              f"steps/epoch={steps_per_epoch} max_epochs={max_epochs}\n"
              f"# trainable_params={trainable}\n")

    d0 = dev_loss(model, dev_feats, pad_id, micro_bs)
    log.write(f"epoch 0 (pre-training) dev_loss {d0:.6f}\n")
    log.flush()

    best = (float("inf"), -1)          # (dev_loss, epoch)
    dev_by_epoch = {0: d0}
    epochs_no_improve = 0
    gstep = 0
    model.train()
    for epoch in range(1, max_epochs + 1):
        order = list(range(len(train_feats)))
        random.Random(f"{run_seed}:epoch{epoch}").shuffle(order)
        for s in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            batch_idx = order[s * micro_bs * accum:(s + 1) * micro_bs * accum]
            if not batch_idx:
                continue
            step_loss = 0.0
            n_chunks = math.ceil(len(batch_idx) / micro_bs)
            for c in range(n_chunks):
                sub = [train_feats[j] for j in
                       batch_idx[c * micro_bs:(c + 1) * micro_bs]]
                ids, labels, mask = collate(sub, pad_id)
                ids, labels, mask = ids.cuda(), labels.cuda(), mask.cuda()
                out = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = out.loss / n_chunks
                loss.backward()
                step_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), GRAD_CLIP)
            opt.step()
            sched.step()
            gstep += 1
            if gstep % LOG_EVERY == 0 or s == steps_per_epoch - 1:
                log.write(f"step {gstep} epoch {epoch} "
                          f"lr {sched.get_last_lr()[0]:.3e} loss {step_loss:.6f}\n")
                log.flush()
        d = dev_loss(model, dev_feats, pad_id, micro_bs)
        dev_by_epoch[epoch] = d
        improved = d < best[0]
        log.write(f"epoch {epoch} dev_loss {d:.6f}"
                  f"{'  * new best' if improved else ''}\n")
        log.flush()
        if improved:
            best = (d, epoch)
            epochs_no_improve = 0
            model.save_pretrained(str(adapter_dir))
        else:
            epochs_no_improve += 1
            if epochs_no_improve > PATIENCE - 1:
                log.write(f"early stop after epoch {epoch} "
                          f"(no improvement over epoch {best[1]})\n")
                break

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    wall = time.time() - t0
    log.write(f"best dev_loss {best[0]:.6f} @ epoch {best[1]}\n"
              f"wall_clock_s {wall:.1f}  peak_vram_gb {peak_gb:.2f}\n")
    log.close()

    del model
    torch.cuda.empty_cache()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    return {"arm": arm, "seed": run_seed, "dev_loss_by_epoch": dev_by_epoch,
            "best_dev_loss": best[0], "best_epoch": best[1],
            "trainable_params": trainable, "train_wall_s": round(wall, 1),
            "peak_vram_gb": round(peak_gb, 2),
            "adapter_sha256": sha256_file(adapter_file) if adapter_file.exists() else None}


def run_inference(spec, arm, run_seed, adapter_dir, gen_receipts_path, smoke=False):
    """Greedy + sampled generation for one trained run. Returns (rows, receipt)."""
    import torch
    from peft import PeftModel

    t0 = time.time()
    tok, base = load_base()
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    model_name = f"local/qwen3-1.7b-lora-{arm}-seed{run_seed}"
    cells = spec.infer_cells()
    if smoke:
        cells = [c for c in cells if c[2] <= 1][:4]

    rows, n_failed, n_attempts = [], 0, 0
    gen_log = open(gen_receipts_path, "a", encoding="utf-8")
    for desc, split, idx, temp in cells:
        prompt = spec.infer_prompt(arm, desc)
        chat = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        in_ids = tok(chat, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        params = None
        for attempt in range(1, PARSE_ATTEMPTS + 1):
            n_attempts += 1
            torch.manual_seed(stable_seed(model_name, desc, idx, attempt))
            with torch.no_grad():
                if temp == 0.0:
                    out = model.generate(in_ids, max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=False, temperature=None,
                                         top_p=None, top_k=None,
                                         pad_token_id=tok.eos_token_id)
                else:
                    out = model.generate(in_ids, max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=True, temperature=temp,
                                         top_p=1.0, top_k=0,
                                         pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][in_ids.shape[1]:], skip_special_tokens=True)
            params = validate_params(extract_json(text))
            gen_log.write(json.dumps({
                "cell_id": f"{model_name}|{desc}|{idx}", "attempt": attempt,
                "descriptor": desc, "sample_idx": idx, "temperature": temp,
                "target_split": split, "output": text[:2000],
                "status": "ok" if params else "invalid"}, sort_keys=True) + "\n")
            gen_log.flush()
            if params:
                break
        if params:
            rows.append({"descriptor": desc, "model": model_name,
                         "params": params, "prompt": "p1_minimal",
                         "sample_idx": idx, "target_split": split,
                         "temperature": temp})
        else:
            n_failed += 1
    gen_log.close()
    wall = time.time() - t0
    del model, base
    torch.cuda.empty_cache()
    return rows, {"cells": len(cells), "ok": len(rows), "failed": n_failed,
                  "gen_attempts": n_attempts, "infer_wall_s": round(wall, 1)}


# -------------------------------------------------------------------- driver
def main():
    smoke = "--smoke" in sys.argv
    scratch = Path(sys.argv[sys.argv.index("--scratch") + 1]) \
        if "--scratch" in sys.argv else None
    receipts = (scratch / "finetune") if (smoke and scratch) else FT_RECEIPTS
    models_dir = (scratch / "models") if (smoke and scratch) else MODELS_DIR
    receipts.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    import torch
    spec = Spec()

    hw = {"gpu": torch.cuda.get_device_name(0),
          "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
          "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
          "torch": torch.__version__, "python": sys.version.split()[0]}
    import transformers, peft
    hw["transformers"] = transformers.__version__
    hw["peft"] = peft.__version__

    hyper = {"model": MODEL_ID, "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
             "lora_dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES,
             "lr": LR, "schedule": "cosine, no warmup, horizon = 5-epoch steps",
             "weight_decay": 0.0, "grad_clip": GRAD_CLIP, "batch": BATCH,
             "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
             "dev_frac": DEV_FRAC, "protocol_seed": SEED,
             "run_seeds": RUN_SEEDS, "loss": "CE on completion tokens only",
             "chat_template": "enable_thinking=False",
             "sampling": {"greedy": "do_sample=False",
                          "cloud": {"temperature": TEMP_CLOUD, "top_p": 1.0,
                                    "top_k": 0}},
             "raft": {"same_desc_exemplars": RAFT_SAME_N,
                      "distractor_exemplars": RAFT_DISTRACT_N,
                      "train_pool": "TRAIN minus dev, self-excluded",
                      "exemplar_seed": f"random.Random('{SEED}:raft:<row_id>')",
                      "infer_exemplars": "harness logic, full TRAIN, "
                                         f"random.Random('{SEED}:<desc>'), n={EXEMPLAR_N}"},
             "dev_slice": {"n_dev": len(spec.dev_rows),
                           "n_train": len(spec.train_rows),
                           "rule": f"per descriptor: random.Random('{SEED}:dev:<desc>')"
                                   f".sample(sorted_by_id, round({DEV_FRAC}*n))"},
             "completion_key_order": spec.completion_order,
             "hardware": hw, "smoke": smoke}
    (receipts / "hyperparams.json").write_text(
        json.dumps(hyper, indent=2), encoding="utf-8")

    runs = [("d", s) for s in RUN_SEEDS] + [("f", s) for s in RUN_SEEDS]
    if smoke:
        runs = [("d", 0), ("f", 0)]

    all_receipts, t_start = [], time.time()
    for arm, seed_ in runs:
        tag = f"{arm}_seed{seed_}"
        done_marker = receipts / f"run_{tag}.done.json"
        if done_marker.exists() and not smoke:
            all_receipts.append(json.loads(done_marker.read_text(encoding="utf-8")))
            print(f"[skip] {tag} already complete")
            continue
        print(f"[run] {tag} training...")
        adapter_dir = models_dir / tag
        adapter_dir.mkdir(parents=True, exist_ok=True)
        micro = BATCH
        while True:
            try:
                tr = train_run(spec, arm, seed_, receipts / f"training_log_{tag}.txt",
                               adapter_dir, micro, smoke=smoke)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                micro //= 2
                if micro < 1:
                    raise
                print(f"[oom] {tag}: retry with micro-batch {micro}")
        print(f"[run] {tag} best dev loss {tr['best_dev_loss']:.4f} "
              f"@ epoch {tr['best_epoch']} ({tr['train_wall_s']}s); inference...")
        rows, inf = run_inference(spec, arm, seed_, adapter_dir,
                                  receipts / "generations.jsonl", smoke=smoke)
        rows_path = receipts / f"rows_{tag}.jsonl"
        rows_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
            encoding="utf-8")
        rec = {**tr, **inf, "micro_batch": micro, "rows_file": str(rows_path)}
        done_marker.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        all_receipts.append(rec)
        print(f"[run] {tag} done: {inf['ok']}/{inf['cells']} cells ok, "
              f"{inf['failed']} failed")

    total_wall = time.time() - t_start
    if smoke:
        print(f"SMOKE OK in {total_wall:.0f}s — receipts in {receipts}")
        return

    # compile arm files (all seeds concatenated, run order)
    for arm, out_path in (("d", OUT_D), ("f", OUT_F)):
        lines = []
        for seed_ in RUN_SEEDS:
            lines.extend((receipts / f"rows_{arm}_seed{seed_}.jsonl")
                         .read_text(encoding="utf-8").splitlines())
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # checksums receipt
    with open(receipts / "checksums.txt", "w", encoding="utf-8") as f:
        for r in all_receipts:
            f.write(f"{r['arm']}_seed{r['seed']} adapter_model.safetensors "
                    f"sha256={r['adapter_sha256']}\n")
        f.write(f"arm_d.jsonl sha256={sha256_file(OUT_D)}\n")
        f.write(f"arm_f.jsonl sha256={sha256_file(OUT_F)}\n")

    # paired stats files
    n_cells = sum(r["cells"] for r in all_receipts)
    n_ok = sum(r["ok"] for r in all_receipts)
    n_failed = sum(r["failed"] for r in all_receipts)
    lines = ["ARMS D (LoRA SFT) + F (RAFT) — Stage 5b (D-013/D-016/D-021/D-026)",
             f"base model: {MODEL_ID}  LoRA r={LORA_R} alpha={LORA_ALPHA} "
             f"dropout={LORA_DROPOUT} on {'+'.join(TARGET_MODULES)}",
             f"train rows: {len(spec.train_rows)}  dev rows: {len(spec.dev_rows)} "
             f"(10% per-descriptor stratified, seed {SEED})",
             f"hardware: {hw['gpu']} ({hw['vram_gb']} GB, sm_{hw['capability'].replace('.','')}) "
             f"torch {hw['torch']} transformers {hw['transformers']} peft {hw['peft']}",
             ""]
    for r in all_receipts:
        lines.append(
            f"{r['arm'].upper()} seed {r['seed']}: best dev loss {r['best_dev_loss']:.4f} "
            f"@ epoch {r['best_epoch']} (of {max(r['dev_loss_by_epoch'])}), "
            f"cells {r['ok']}/{r['cells']} ok ({r['failed']} failed), "
            f"train {r['train_wall_s']}s + infer {r['infer_wall_s']}s")
    lines += ["",
              f"totals: {n_ok}/{n_cells} cells ok, {n_failed} failed "
              f"({100 * n_failed / n_cells:.2f}%), "
              f"wall clock {total_wall / 3600:.2f} h",
              f"arm_d.jsonl rows: {sum(r['ok'] for r in all_receipts if r['arm'] == 'd')}   "
              f"arm_f.jsonl rows: {sum(r['ok'] for r in all_receipts if r['arm'] == 'f')}"]
    OUT_RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")

    OUT_INTERP.write_text(
        "FINETUNE INTERPRETATION — Stage 5b\n\n"
        "1. Training data is train.jsonl ONLY; eval_set.jsonl and\n"
        "   tail_eval.jsonl were never opened by this stage. Tail descriptor\n"
        "   names came from arm_a.jsonl rows, and the 60-name sample\n"
        "   replicates the harness draw (random.Random(42) over the sorted\n"
        "   set), so D/F tail cells align with arms A/B/C cell-for-cell.\n"
        "2. Dev losses per seed are receipted in\n"
        "   results/receipts/finetune/training_log_*.txt with full step-loss\n"
        "   curves; adapters + SHA-256 in checksums.txt. Cross-seed dev-loss\n"
        "   spread is the overfitting-variance check promised in D-013.\n"
        "3. Every generation attempt (including invalid parses) is a line in\n"
        "   results/receipts/finetune/generations.jsonl; failed cells are\n"
        "   reported above and excluded from arm files — the metrics stage\n"
        "   must state resulting Ns (no silent shrinkage).\n"
        "4. Spec-silent choices (receipted in hyperparams.json): chat template\n"
        "   with empty think block; no LR warmup; weight decay 0; grad clip\n"
        "   1.0; sampling top_p=1/top_k=0 so temperature is the only\n"
        "   transform; F training exemplar pools exclude dev rows and the\n"
        "   example's own row (leak hygiene), while F inference exemplars\n"
        "   replicate the Arm C harness selection on full TRAIN.\n"
        "5. D/F remain gate-labeled exploratory (D-019.4 as amended by\n"
        "   D-026): with 2 above-floor descriptors they are the exploratory\n"
        "   training axis, not confirmatory FDR contrasts.\n",
        encoding="utf-8")
    print(f"DONE in {total_wall / 3600:.2f} h — arm_d + arm_f written")


if __name__ == "__main__":
    main()
