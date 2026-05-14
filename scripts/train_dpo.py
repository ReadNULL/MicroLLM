from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.model import TransformerLM
from src.model.lora import (
    apply_lora_to_model,
    get_lora_params,
    get_lora_state_dict,
    load_lora_state_dict,
    print_trainable_params,
)
from src.tokenizer import BPETokenizer
from src.training import AdamW
from src.training.dpo import dpo_loss, compute_token_logps
from src.training.dpo_dataset import DPODataset
from src.training import load_model_state, load_checkpoint, save_checkpoint, masked_cross_entropy


EOS_TOKEN = "<|end|>"


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = config.get("tokenizer", {})
    model = config.get("model", {})
    optimizer = config.get("optimizer", {})
    training = config.get("training", {})
    data = config.get("data", {})
    logging = config.get("logging", {})
    lora_cfg = config.get("lora", {})
    dpo = config.get("dpo", {})

    defaults: dict[str, object] = {
        "vocab_path": Path(tokenizer["vocab_path"]) if tokenizer.get("vocab_path") else None,
        "merges_path": Path(tokenizer["merges_path"]) if tokenizer.get("merges_path") else None,
        "special_tokens": tokenizer.get("special_tokens"),
        "context_length": model.get("context_length"),
        "d_model": model.get("d_model"),
        "num_heads": model.get("num_heads"),
        "num_layers": model.get("num_layers"),
        "d_ff": model.get("d_ff"),
        "vocab_size": model.get("vocab_size"),
        "rope_theta": model.get("rope_theta"),
        "use_rms_norm": model.get("use_rms_norm"),
        "norm_mode": model.get("norm_mode"),
        "ffn_type": model.get("ffn_type"),
        "lr": optimizer.get("lr"),
        "weight_decay": optimizer.get("weight_decay"),
        "batch_size": training.get("batch_size"),
        "max_steps": training.get("max_steps"),
        "eval_interval": training.get("eval_interval"),
        "save_interval": training.get("save_interval"),
        "device": training.get("device"),
        "seed": training.get("seed"),
        "out_dir": Path(training["out_dir"]) if training.get("out_dir") else None,
        "init_checkpoint": Path(training["init_checkpoint"]) if training.get("init_checkpoint") else None,
        "resume": training.get("resume"),
        "train_data_path": Path(data["train_data_path"]) if data.get("train_data_path") else None,
        "valid_data_path": Path(data["valid_data_path"]) if data.get("valid_data_path") else None,
        "eos_token": data.get("eos_token"),
        "wandb_project": logging.get("wandb_project"),
        "run_name": logging.get("run_name"),
        "wandb_mode": logging.get("mode"),
        "use_lora": lora_cfg.get("enabled", False),
        "lora_r": lora_cfg.get("r", 8),
        "lora_alpha": lora_cfg.get("alpha", 16.0),
        "lora_targets": lora_cfg.get("targets"),
        "dpo_beta": dpo.get("beta", 0.1),
        "sft_weight": dpo.get("sft_weight", 0.0),
    }
    return {key: value for key, value in defaults.items() if value is not None}


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO training for MicroLM.")
    parser.add_argument("--config", type=str, default=defaults.get("config"))

    parser.add_argument("--vocab-path", type=Path, default=defaults.get("vocab_path"))
    parser.add_argument("--merges-path", type=Path, default=defaults.get("merges_path"))
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=defaults.get("special_tokens"),
        help="Special token reserved while loading the tokenizer.",
    )

    parser.add_argument("--context-length", type=int, default=defaults.get("context_length", 512))
    parser.add_argument("--d-model", type=int, default=defaults.get("d_model", 512))
    parser.add_argument("--num-heads", type=int, default=defaults.get("num_heads", 8))
    parser.add_argument("--num-layers", type=int, default=defaults.get("num_layers", 8))
    parser.add_argument("--d-ff", type=int, default=defaults.get("d_ff", 1344))
    parser.add_argument("--vocab-size", type=int, default=defaults.get("vocab_size", 6400))
    parser.add_argument("--rope-theta", type=float, default=defaults.get("rope_theta", 1000000.0))
    parser.add_argument("--use-rms-norm", action=argparse.BooleanOptionalAction, default=defaults.get("use_rms_norm", True))
    parser.add_argument("--norm-mode", type=str, default=defaults.get("norm_mode", "pre"), choices=["pre", "post"])
    parser.add_argument("--ffn-type", type=str, default=defaults.get("ffn_type", "swiglu"), choices=["swiglu", "silu"])

    parser.add_argument("--lr", type=float, default=defaults.get("lr", 5e-6))
    parser.add_argument("--weight-decay", type=float, default=defaults.get("weight_decay", 0.1))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 1))
    parser.add_argument("--max-steps", type=int, default=defaults.get("max_steps", 100))
    parser.add_argument("--eval-interval", type=int, default=defaults.get("eval_interval", 10))
    parser.add_argument("--save-interval", type=int, default=defaults.get("save_interval", 50))
    parser.add_argument("--device", type=str, default=defaults.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--out-dir", type=Path, default=defaults.get("out_dir", Path("outputs/dpo")))
    parser.add_argument("--init-checkpoint", type=Path, default=defaults.get("init_checkpoint"))
    parser.add_argument("--resume", action="store_true", default=defaults.get("resume", False))

    parser.add_argument("--train-data-path", type=Path, default=defaults.get("train_data_path"))
    parser.add_argument("--valid-data-path", type=Path, default=defaults.get("valid_data_path"))
    parser.add_argument("--eos-token", type=str, default=defaults.get("eos_token", EOS_TOKEN))

    parser.add_argument("--dpo-beta", type=float, default=defaults.get("dpo_beta", 0.1))
    parser.add_argument("--sft-weight", type=float, default=defaults.get("sft_weight", 0.0))

    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=defaults.get("use_lora", False))
    parser.add_argument("--lora-r", type=int, default=defaults.get("lora_r", 8))
    parser.add_argument("--lora-alpha", type=float, default=defaults.get("lora_alpha", 16.0))
    parser.add_argument("--lora-targets", type=str, nargs="+", default=defaults.get("lora_targets"))

    parser.add_argument("--wandb-project", type=str, default=defaults.get("wandb_project", "micro-lm"))
    parser.add_argument("--run-name", type=str, default=defaults.get("run_name"))
    parser.add_argument("--wandb-mode", type=str, default=defaults.get("wandb_mode", "disabled"), choices=["online", "offline", "disabled"])

    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()

    defaults = load_config_defaults(config_args.config)
    defaults["config"] = config_args.config
    parser = build_parser(defaults)
    return parser.parse_args(remaining)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_configured_tokenizer(args: argparse.Namespace) -> BPETokenizer:
    special_tokens = args.special_tokens or [EOS_TOKEN]
    if args.vocab_path is None or args.merges_path is None:
        raise ValueError("Tokenizer vocab/merges paths are required")
    return BPETokenizer.from_files(str(args.vocab_path), str(args.merges_path), special_tokens=special_tokens)


def build_model(args: argparse.Namespace, device: str) -> TransformerLM:
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rms_norm=bool(args.use_rms_norm),
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type,
        device=device,
    ).to(device)
    return model


def build_reference_model(args: argparse.Namespace, device: str) -> TransformerLM:
    ref_model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rms_norm=bool(args.use_rms_norm),
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type,
        device=device,
    ).to(device)
    return ref_model


def evaluate(
    model: TransformerLM,
    ref_model: TransformerLM,
    loader: DataLoader,
    device: str,
    beta: float,
) -> float:
    model.eval()
    ref_model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            chosen_loss_masks = batch["chosen_loss_masks"].to(device)
            rejected_loss_masks = batch["rejected_loss_masks"].to(device)

            chosen_logits = model(chosen_input_ids)
            rejected_logits = model(rejected_input_ids)
            ref_chosen_logits = ref_model(chosen_input_ids)
            ref_rejected_logits = ref_model(rejected_input_ids)

            policy_chosen_logps = compute_token_logps(chosen_logits, chosen_labels, chosen_loss_masks)
            policy_rejected_logps = compute_token_logps(rejected_logits, rejected_labels, rejected_loss_masks)
            ref_chosen_logps = compute_token_logps(ref_chosen_logits, chosen_labels, chosen_loss_masks)
            ref_rejected_logps = compute_token_logps(ref_rejected_logits, rejected_labels, rejected_loss_masks)

            loss, _, _ = dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta)
            total_loss += loss.item()
            total_count += 1

    model.train()
    return total_loss / total_count if total_count > 0 else float("nan")


def iter_batches(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    args = parse_args()
    if args.train_data_path is None or args.valid_data_path is None:
        raise ValueError("Both train-data-path and valid-data-path are required")
    if not args.train_data_path.exists():
        raise FileNotFoundError(f"Training data not found at {args.train_data_path}")
    if not args.valid_data_path.exists():
        raise FileNotFoundError(f"Validation data not found at {args.valid_data_path}")
    if args.init_checkpoint is None:
        raise ValueError("DPO training requires --init-checkpoint (SFT model)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    tokenizer = load_configured_tokenizer(args)
    model = build_model(args, args.device)
    ref_model = build_reference_model(args, args.device)

    load_model_state(str(args.init_checkpoint), model)
    load_model_state(str(args.init_checkpoint), ref_model)
    print(f"Loaded init checkpoint from {args.init_checkpoint}")

    if args.use_lora:
        apply_lora_to_model(model, r=args.lora_r, alpha=args.lora_alpha, target_names=args.lora_targets)
        print(f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}")
        print_trainable_params(model)
    else:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {num_params:,}")

    for param in ref_model.parameters():
        param.requires_grad = False

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "use_rms_norm": bool(args.use_rms_norm),
        "norm_mode": args.norm_mode,
        "ffn_type": args.ffn_type,
    }
    with (args.out_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    resolved_config = vars(args).copy()
    resolved_config["train_data_path"] = str(args.train_data_path)
    resolved_config["valid_data_path"] = str(args.valid_data_path)
    resolved_config["out_dir"] = str(args.out_dir)
    resolved_config["vocab_path"] = str(args.vocab_path) if args.vocab_path is not None else None
    resolved_config["merges_path"] = str(args.merges_path) if args.merges_path is not None else None
    resolved_config["init_checkpoint"] = str(args.init_checkpoint) if args.init_checkpoint is not None else None
    resolved_config["use_lora"] = args.use_lora
    resolved_config["lora_r"] = args.lora_r
    resolved_config["lora_alpha"] = args.lora_alpha
    resolved_config["lora_targets"] = args.lora_targets
    with (args.out_dir / "resolved_train_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2, ensure_ascii=False)

    train_ds = DPODataset(args.train_data_path, tokenizer=tokenizer, max_length=args.context_length, eos_token=args.eos_token)
    valid_ds = DPODataset(args.valid_data_path, tokenizer=tokenizer, max_length=args.context_length, eos_token=args.eos_token)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW(get_lora_params(model) if args.use_lora else model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ckpt_path = args.out_dir / "ckpt.pt"
    start_step = 0

    if args.resume and ckpt_path.exists():
        start_step = load_checkpoint(str(ckpt_path), model, optimizer)
        print(f"Resuming DPO from step {start_step}")
        if args.use_lora:
            lora_path = args.out_dir / "lora_adaptor.pt"
            if lora_path.exists():
                load_lora_state_dict(model, torch.load(lora_path, map_location=args.device, weights_only=True))
                print(f"Loaded LoRA adaptor from {lora_path}")

    wandb = None
    if args.wandb_mode != "disabled":
        import wandb as wandb_module
        wandb = wandb_module
        wandb.init(project=args.wandb_project, name=args.run_name, mode=args.wandb_mode, config=resolved_config)

    log_path = args.out_dir / "train_log.jsonl"
    train_iter = iter_batches(train_loader)

    for step in range(start_step, args.max_steps):
        model.train()
        batch = next(train_iter)
        chosen_input_ids = batch["chosen_input_ids"].to(args.device)
        chosen_labels = batch["chosen_labels"].to(args.device)
        rejected_input_ids = batch["rejected_input_ids"].to(args.device)
        rejected_labels = batch["rejected_labels"].to(args.device)
        chosen_loss_masks = batch["chosen_loss_masks"].to(args.device)
        rejected_loss_masks = batch["rejected_loss_masks"].to(args.device)

        chosen_logits = model(chosen_input_ids)
        rejected_logits = model(rejected_input_ids)

        with torch.no_grad():
            ref_chosen_logits = ref_model(chosen_input_ids)
            ref_rejected_logits = ref_model(rejected_input_ids)

        policy_chosen_logps = compute_token_logps(chosen_logits, chosen_labels, chosen_loss_masks)
        policy_rejected_logps = compute_token_logps(rejected_logits, rejected_labels, rejected_loss_masks)
        ref_chosen_logps = compute_token_logps(ref_chosen_logits, chosen_labels, chosen_loss_masks)
        ref_rejected_logps = compute_token_logps(ref_rejected_logits, rejected_labels, rejected_loss_masks)

        dpo_loss_val, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, args.dpo_beta
        )

        loss = dpo_loss_val
        if args.sft_weight > 0.0:
            sft_loss = masked_cross_entropy(chosen_logits, chosen_labels, chosen_loss_masks)
            loss = dpo_loss_val + args.sft_weight * sft_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        completed_step = step + 1

        if completed_step % args.eval_interval == 0 or completed_step == args.max_steps:
            val_loss = evaluate(model, ref_model, valid_loader, args.device, args.dpo_beta)
            print(f"Step {completed_step}: dpo_loss {dpo_loss_val.item():.4f}, val_loss {val_loss:.4f}, chosen_rewards {chosen_rewards.mean().item():.4f}, rejected_rewards {rejected_rewards.mean().item():.4f}")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(
                    {"step": completed_step, "dpo_loss": float(dpo_loss_val.item()), "val_loss": float(val_loss), "chosen_rewards": float(chosen_rewards.mean().item()), "rejected_rewards": float(rejected_rewards.mean().item())},
                    ensure_ascii=False,
                ) + "\n")
            if wandb is not None:
                wandb.log({"train/dpo_loss": dpo_loss_val.item(), "train/chosen_rewards": chosen_rewards.mean().item(), "train/rejected_rewards": rejected_rewards.mean().item(), "val/loss": val_loss, "step": completed_step})

        if completed_step % args.save_interval == 0 or completed_step == args.max_steps:
            save_checkpoint(model, optimizer, iteration=completed_step, out=str(ckpt_path))
            if args.use_lora:
                torch.save(get_lora_state_dict(model), args.out_dir / "lora_adaptor.pt")

    save_checkpoint(model, optimizer, iteration=args.max_steps, out=str(args.out_dir / "ckpt_final.pt"))
    if args.use_lora:
        torch.save(get_lora_state_dict(model), args.out_dir / "lora_adaptor.pt")

    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
