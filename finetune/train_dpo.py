#!/usr/bin/env python3
"""Stage 2: preference tuning (DPO) on human corrections.

SFT teaches the model what to say; DPO teaches it what *not* to say. The pairs
come from the app: a thumbs-down plus a native speaker's rewrite is a
(prompt, chosen, rejected) triple. Sunbird used the same shape — SFT, then a DPO
variant aimed at the failure modes that survived it (invented words, repeated
loops, drifting into English).

    python finetune/train_dpo.py --pairs finetune/data/exported/chats_dpo.jsonl --dry-run
    python finetune/train_dpo.py --pairs finetune/data/exported/chats_dpo.jsonl \
        --base Sunbird/Sunflower-14B --adapter finetune/out/yogera-sunflower14b-lora

Needs: pip install torch transformers peft trl datasets bitsandbytes accelerate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_pairs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if {"prompt", "chosen", "rejected"} <= set(rec):
            out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="finetune/data/exported/chats_dpo.jsonl")
    ap.add_argument("--base", default="Sunbird/Sunflower-14B")
    ap.add_argument("--adapter", default="", help="LoRA adapter from stage 1")
    ap.add_argument("--out", default="finetune/out/yogera-dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6, help="DPO wants a much smaller LR than SFT")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs = load_pairs(Path(args.pairs))
    print(f"pairs: {len(pairs)}  base: {args.base}  adapter: {args.adapter or '(none)'}  beta: {args.beta}")
    if not pairs:
        print("No preference pairs yet. Thumbs-down an answer in the app and give the correct Luganda,")
        print("then Export in the learning drawer. DPO needs at least a few hundred to do anything.")
        return 1
    if args.dry_run:
        lens = [(len(p["chosen"].split()), len(p["rejected"].split())) for p in pairs]
        print(f"avg chosen/rejected words: {sum(a for a, _ in lens)/len(lens):.1f}/"
              f"{sum(b for _, b in lens)/len(lens):.1f}")
        print("dry run OK — data shape is valid")
        return 0

    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig, PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: PLC0415
    from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.base, quantization_config=bnb,
                                                device_map="auto", torch_dtype=torch.bfloat16)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        model = PeftModel(model, cfg) if not hasattr(model, "peft_config") else model
        model.add_adapter(cfg)

    dataset = Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]} for p in pairs
    ])
    cfg = DPOConfig(
        output_dir=args.out, beta=args.beta, num_train_epochs=args.epochs,
        learning_rate=args.lr, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        bf16=True, gradient_checkpointing=True, optim="paged_adamw_8bit",
        logging_steps=10, save_steps=100, report_to=[], max_length=2048, max_prompt_length=768,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"DPO adapter saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
