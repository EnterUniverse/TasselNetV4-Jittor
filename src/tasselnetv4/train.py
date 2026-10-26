"""Training and local count losses.

Adapted from tiny-smart/TasselNetV4 train_val.py (Apache-2.0).
"""
import argparse
import json
import math
from pathlib import Path

import jittor as jt
from jittor import nn

from .checkpoint import load_weights
from .data import PACDataset
from .evaluate import evaluate
from .model import AvgPool, build_model
from .runtime import load_config


def local_count_loss(model, outputs, density, gates, loss_type="L1", use_mask=False):
    if loss_type not in ("L1", "L2"):
        raise ValueError("loss must be L1 or L2")
    total = jt.zeros((1,))
    for idx, (block, stride) in enumerate(zip(model.blocksize_list, model.output_stride_list)):
        # Summed pixel density in each local window equals its target count.
        target = AvgPool(block, stride)(density.unsqueeze(1)).squeeze(1) * block ** 2
        difference = (outputs[str(block)] - target) * gates[:, idx].reshape(-1, 1, 1)
        error = difference.abs() if loss_type == "L1" else difference ** 2
        if use_mask:
            error = error * (jt.rand((1, *error.shape[-2:])) < 0.8).float32()
        # Normalize by the number of local windows.
        total = total + error.sum() / (error.shape[-2] * error.shape[-1])
    return total


def parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    unused = ("scale_embeds.", "v_y.", "density_proj.")
    for name, param in model.named_parameters():
        if param.is_stop_grad() or name.startswith(unused):
            continue
        (no_decay if param.ndim == 1 or name.endswith(".bias") else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def learning_rate(progress, cfg, base):
    warmup = cfg.get("warmup_epochs", 10)
    epochs = cfg.get("epochs", 400)
    if warmup and progress < warmup:
        return base * progress / warmup
    minimum = cfg.get("min_lr", 0)
    return minimum + (base - minimum) * 0.5 * (1 + math.cos(math.pi * (progress - warmup) / max(1, epochs - warmup)))


def main():
    parser = argparse.ArgumentParser(description="Train TasselNetV4.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="outputs/train")
    parser.add_argument("--checkpoint", help="Load converted or Jittor model weights")
    parser.add_argument("--encoder", help="Initialize encoder only from converted MAE .npz")
    parser.add_argument("--resume", help="Resume training from a Jittor checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--eval-every", type=int, default=0, help="0 disables optional validation")
    args = parser.parse_args()
    if sum(bool(v) for v in (args.checkpoint, args.encoder, args.resume)) > 1:
        parser.error("choose one of --checkpoint, --encoder, --resume")
    config = load_config(args.config, args.device)
    cfg = config["Optimizer"]
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if cfg["epochs"] < 1 or cfg.get("accum_iter", 1) < 1:
        parser.error("epochs and accum_iter must be positive")
    model = build_model(config["Model"])
    if args.checkpoint or args.encoder:
        print(load_weights(model, args.checkpoint or args.encoder, encoder_only=bool(args.encoder)))
    data_cfg = config["Dataset"]
    dataset = PACDataset(args.data, batch_size=data_cfg.get("batch_size", 8),
                         num_workers=data_cfg.get("num_workers", 0), shots=args.shots,
                         block_sizes=model.blocksize_list, augment=not args.no_augment)
    accum = cfg.get("accum_iter", 1)
    base_lr = cfg.get("lr")
    if base_lr is None:
        base_lr = cfg.get("blr", 0.0002) * dataset.batch_size * accum / 256
    optimizer = jt.optim.AdamW(parameter_groups(model, cfg.get("weight_decay", 0.05)),
                               lr=base_lr, betas=(0.9, 0.95))
    start_epoch = 0
    if args.resume:
        state = jt.load(args.resume)
        if state["config"]["Model"] != config["Model"]:
            raise ValueError("resume Model config does not match")
        load_weights(model, args.resume)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    batches = math.ceil(len(dataset.keys) / dataset.batch_size)
    valset = PACDataset(args.data, "val", shots=args.shots) if args.eval_every else None
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        losses = []
        for step, sample in enumerate(dataset):
            progress = epoch + step / batches
            optimizer.lr = learning_rate(progress, cfg, base_lr)
            outputs = model([sample["image"], sample["boxes"], sample["scale"], sample["size"]], test=False)
            loss = local_count_loss(model, outputs, sample["gt_density"], sample["size_embedding"],
                                    config["Loss"].get("loss", "L1"), config["Loss"].get("use_mask", False))
            value = float(loss.item())
            if not math.isfinite(value):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}, batch {step}")
            window = min(accum, batches - (step // accum) * accum)
            optimizer.backward(loss / window)
            if (step + 1) % accum == 0 or step + 1 == batches:
                optimizer.step()
                optimizer.zero_grad()
            losses.append(value)
        jt.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "config": config}, str(output / "last.pkl"))
        row = {"epoch": epoch, "loss": sum(losses) / len(losses), "lr": optimizer.lr}
        if valset is not None and (epoch + 1) % args.eval_every == 0:
            row["validation"] = evaluate(model, valset)["metrics"]
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
