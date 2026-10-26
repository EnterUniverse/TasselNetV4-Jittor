"""Count plants in one image using exemplar rectangles in original pixels."""
import argparse
import json
import math
from pathlib import Path

import jittor as jt
import numpy as np
from PIL import Image

from .checkpoint import load_weights
from .data import extract_exemplars, normalize
from .evaluate import predict_count
from .model import build_model
from .runtime import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--boxes", required=True, help="JSON file: [[x1,y1,x2,y2], ...] in original image pixels")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()
    config = load_config(args.config, args.device)
    model = build_model(config["Model"])
    if model.img_size != 384:
        raise ValueError("image preprocessing uses 384-pixel tiles")
    load_weights(model, args.checkpoint)
    model.eval()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    rects = np.asarray(json.loads(Path(args.boxes).read_text()), np.float32)
    if rects.ndim != 2 or rects.shape[1] != 4 or not len(rects) or not np.isfinite(rects).all():
        raise ValueError("boxes must be a nonempty Nx4 finite array")
    if np.any(rects[:, 2:] <= rects[:, :2]):
        raise ValueError("each box must have x2>x1 and y2>y1")
    if image.width < image.height:
        image = image.transpose(Image.Transpose.TRANSPOSE)
        rects = rects[:, [1, 0, 3, 2]]
    old_w, old_h = image.size
    new_w = max(384, math.ceil(old_w * 384 / old_h) // 16 * 16)
    image = image.resize((new_w, 384), Image.Resampling.BICUBIC)
    rects = np.rint(rects * [new_w / old_w, 384 / old_h, new_w / old_w, 384 / old_h]).astype(int)
    boxes, scales = extract_exemplars(image, rects, shots=len(rects))
    count = predict_count(model, jt.array(normalize(image)[None]), jt.array(boxes[None]),
                          jt.array(scales[None]), jt.array([scales.max() * 384]))
    result = {"image": args.image, "count": count, "shots": len(rects)}
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n")


if __name__ == "__main__":
    main()
