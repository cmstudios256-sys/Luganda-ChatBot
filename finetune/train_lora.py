#!/usr/bin/env python3
"""QLoRA supervised fine-tuning for a Luganda chat model.

The recipe follows what actually worked for Ugandan-language models (Sunbird's
Sunflower 14B/32B): LoRA r=16 on the linear layers, 4-bit base weights, and the
loss computed only on the assistant's tokens.

Works with any causal LM on the Hub. Recommended bases for Luganda, cheapest
first:
  Qwen/Qwen3-8B                 general, good multilingual tokenizer
  Sunbird/Sunflower-14B         already speaks 31 Ugandan languages  <- start here
  Sunbird/Sunflower-32B         best quality, needs ~24 GB+ with 4-bit LoRA

Quick plan check without a GPU (safe to run anywhere):
    python finetune/train_lora.py --config finetune/configs/lora_sunflower14b.yaml --dry-run

Real run (needs: pip install torch transformers peft trl datasets bitsandbytes accelerate pyyaml):
    python finetune/train_lora.py --config finetune/configs/lora_sunflower14b.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Config (tiny YAML subset, so we do not need pyyaml to plan a run)
# --------------------------------------------------------------------------- #
def _scalar(text: str) -> Any:
    text = text.strip()
    if text.lower() in {"true", "yes"}:
        return True
    if text.lower() in {"false", "no"}:
        return False
    if text.lower() in {"null", "none", "~", ""}:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(x) for x in inner.split(",")] if inner else []
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def load_config(path: Path) -> dict:
    try:
        import yaml  # type: ignore  # noqa: PLC0415

        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        pass
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key.strip()] = _scalar(value)
        else:
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
    return root


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


# --------------------------------------------------------------------------- #
# Dry run: validate the plan without torch
# --------------------------------------------------------------------------- #
def dry_run(cfg: dict) -> int:
    model = cfg.get("base_model", "?")
    tr = cfg.get("training", {})
    lora = cfg.get("lora", {})
    quant = cfg.get("quantization", {})
    data = cfg.get("data", {})

    train_path = ROOT / data.get("train", "out/train.jsonl")
    val_path = ROOT / data.get("val", "out/val.jsonl")
    n_train, n_val = count_lines(train_path), count_lines(val_path)
    epochs = float(tr.get("epochs", 3))
    batch = int(tr.get("batch_size", 2))
    accum = int(tr.get("grad_accum", 8))
    seq = int(tr.get("max_seq_len", 2048))

    words = 0
    if train_path.exists():
        for line in train_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for m in json.loads(line)["messages"]:
                    words += len(m["content"].split())
    tokens_est = int(words * 1.55)  # Luganda runs ~1.5-1.7 tokens/word on Qwen/Gemma BPE
    tokens_padded = n_train * seq
    steps_per_epoch = max(1, math.ceil(n_train / (batch * accum)))
    total_steps = int(steps_per_epoch * epochs)
    plan = {
        "base_model": model,
        "dataset": {"train": str(train_path), "val": str(val_path), "train_rows": n_train, "val_rows": n_val},
        "corpus": {"words": words, "tokens_estimate": tokens_est, "max_seq_len": seq},
        "schedule": {
            "epochs": epochs, "batch_size": batch, "grad_accum": accum,
            "effective_batch": batch * accum, "steps_per_epoch": steps_per_epoch,
            "total_optimizer_steps": total_steps,
            "approx_epochs_of_padding_tokens": round(tokens_padded / max(tokens_est, 1), 2),
        },
        "lora": {
            "r": lora.get("r", 16), "alpha": lora.get("alpha", 32),
            "dropout": lora.get("dropout", 0.05), "targets": lora.get("target_modules"),
            "trainable_pct_estimate": "0.5-1.5%",
        },
        "quantization": {
            "load_in_4bit": quant.get("load_in_4bit", True),
            "quant_type": quant.get("bnb_4bit_quant_type", "nf4"),
            "compute_dtype": quant.get("bnb_4bit_compute_dtype", "bfloat16"),
            "double_quant": quant.get("bnb_4bit_use_double_quant", True),
        },
        "loss": "assistant tokens only" if tr.get("train_on_responses_only", True) else "all tokens",
        "warnings": [],
    }
    if n_train < 500:
        plan["warnings"].append(
            f"{n_train} training rows is a smoke test, not training. 2k+ verified rows is a realistic floor."
        )
    if plan["schedule"]["approx_epochs_of_padding_tokens"] > 4:
        plan["warnings"].append("Most of your budget will be padding — lower max_seq_len or pack sequences.")
    if not (data.get("train")):
        plan["warnings"].append("no data.train in config; defaulted to out/train.jsonl")

    print(json.dumps(plan, indent=2, default=str))
    print("\nDry run OK. Nothing was downloaded and no GPU was touched.")
    print("Install the stack and drop --dry-run to train:")
    print("  pip install torch transformers peft trl datasets bitsandbytes accelerate pyyaml")
    return 0


# --------------------------------------------------------------------------- #
# Response-only masking collator
# --------------------------------------------------------------------------- #
def build_response_collator(tokenizer, max_len: int):
    """Mask every token that belongs to the prompt, so the loss is only computed
    on what the assistant says. This is what stops the model learning to echo
    instructions — a known failure mode for low-resource chat fine-tuning."""
    import torch  # noqa: PLC0415

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def render(messages: list[dict], upto: int) -> list[int]:
        prefix = tokenizer.apply_chat_template(
            messages[:upto], tokenize=False, add_generation_prompt=True
        )
        return tokenizer(prefix, add_special_tokens=False)["input_ids"]

    def collate(batch: list[dict]) -> dict:
        input_ids, labels, attn = [], [], []
        for example in batch:
            msgs = example["messages"]
            ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_special_tokens=False)
            ids = ids[:max_len]
            last_assistant = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
            prompt_len = len(render(msgs, last_assistant))
            lab = list(ids)
            for i in range(min(prompt_len, len(lab))):
                lab[i] = -100
            input_ids.append(ids)
            labels.append(lab)
            attn.append([1] * len(ids))
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

    return collate


# --------------------------------------------------------------------------- #
# Real training
# --------------------------------------------------------------------------- #
def train(cfg: dict, dry: bool = False) -> int:
    import torch  # noqa: PLC0415
    from datasets import load_dataset  # noqa: PLC0415
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    data_cfg = cfg.get("data", {})
    tr = cfg.get("training", {})
    lora = cfg.get("lora", {})
    quant = cfg.get("quantization", {})
    out_dir = Path(cfg.get("output_dir", "finetune/out/adapter"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = ROOT / data_cfg.get("train", "out/train.jsonl")
    val_file = ROOT / data_cfg.get("val", "out/val.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = None
    if quant.get("load_in_4bit", True):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=getattr(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16")),
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj",
                                                     "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": str(train_file), "validation": str(val_file)})
    collator = build_response_collator(tokenizer, int(tr.get("max_seq_len", 2048)))

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(tr.get("epochs", 3)),
        per_device_train_batch_size=int(tr.get("batch_size", 2)),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(tr.get("grad_accum", 8)),
        learning_rate=float(tr.get("lr", 1e-4)),
        lr_scheduler_type=tr.get("lr_scheduler", "cosine"),
        warmup_ratio=float(tr.get("warmup_ratio", 0.03)),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        logging_steps=int(tr.get("logging_steps", 10)),
        save_steps=int(tr.get("save_steps", 200)),
        eval_strategy="steps",
        eval_steps=int(tr.get("eval_steps", 100)),
        bf16=bool(tr.get("bf16", True)),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=[],
        seed=int(tr.get("seed", 42)),
        save_total_limit=2,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["validation"], data_collator=collator)

    if dry:
        stats = trainer.get_train_dataloader()
        print(f"dry-run: {len(stats)} batches ready, collator built, nothing trained")
        return 0

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"adapter saved -> {out_dir}")

    hub = cfg.get("hub", {})
    if hub.get("push_to_hub") and hub.get("repo_id"):
        model.push_to_hub(hub["repo_id"], private=bool(hub.get("private", True)))
        tokenizer.push_to_hub(hub["repo_id"], private=bool(hub.get("private", True)))
        print(f"pushed -> {hub['repo_id']}")

    if cfg.get("merge_and_export"):
        merged = model.merge_and_unload()
        target = cfg.get("merge_output_dir", str(out_dir) + "-merged")
        merged.save_pretrained(target, safe_serialization=True)
        tokenizer.save_pretrained(target)
        print(f"merged fp16 model -> {target}  (serve with vLLM, then point the app's LLM_BASE_URL at it)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "lora_sunflower14b.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no model, no GPU needed")
    ap.add_argument("--base-model", default=None, help="override base_model from the config")
    args = ap.parse_args()

    path = Path(args.config)
    if not path.exists():
        print(f"config not found: {path}")
        return 2
    cfg = load_config(path)
    if args.base_model:
        cfg["base_model"] = args.base_model
    if "base_model" not in cfg:
        print("config needs at least: base_model")
        return 2
    return dry_run(cfg) if args.dry_run else train(cfg)


if __name__ == "__main__":
    sys.exit(main())
