"""Evaluate TasselNetV4 on PAC images."""
import argparse
import json
import math
from pathlib import Path

import jittor as jt
from jittor import nn
import numpy as np

from .checkpoint import load_weights
from .data import PACDataset
from .model import build_model
from .runtime import load_config


def predict_count(model, image, boxes, scales, size):
    """Center-pad normalized landscape image and sum non-overlapping tiles."""
    side = model.img_size
    h, w = image.shape[-2:]
    target_h, target_w = math.ceil(h / side) * side, math.ceil(w / side) * side
    pad_h, pad_w = target_h - h, target_w - w
    padded = nn.pad(image, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    count = 0.0
    with jt.no_grad():
        for y in range(0, target_h, side):
            for x in range(0, target_w, side):
                output, _ = model([padded[:, :, y:y + side, x:x + side], boxes, scales, size])
                count += float(output.sum().item())
    return count


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if not len(truth):
        raise ValueError("evaluation requires at least one sample")
    error = prediction - truth
    sst = float(((truth - truth.mean()) ** 2).sum())
    return {"n": len(truth), "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.square(error).mean())),
            "WCA": float(1 - np.abs(error).sum() / truth.sum()) if truth.sum() else None,
            "R2": float(1 - np.square(error).sum() / sst) if len(truth) > 1 and sst else None}


def evaluate(model, dataset):
    model.eval()
    rows = []
    for sample in dataset:
        count = predict_count(model, sample["image"], sample["boxes"], sample["scale"], sample["size"])
        row = {"id": sample["id"][0], "truth": float(sample["dots"].item()), "prediction": count}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"metrics": metrics([r["truth"] for r in rows], [r["prediction"] for r in rows]), "samples": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", nargs="+", default=["val"], choices=["val", "test"])
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output", default="outputs/evaluation.json")
    args = parser.parse_args()
    config = load_config(args.config, args.device)
    model = build_model(config["Model"])
    load_weights(model, args.checkpoint)
    dataset = PACDataset(args.data, args.split, shots=args.shots, block_sizes=model.blocksize_list)
    result = evaluate(model, dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
