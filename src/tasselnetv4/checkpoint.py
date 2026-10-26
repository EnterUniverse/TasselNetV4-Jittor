"""Checkpoint loading and PyTorch-to-NumPy weight conversion."""
import argparse
import json
from pathlib import Path

import jittor as jt
import numpy as np


def read_arrays(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {k: archive[k] for k in archive.files if k != "__metadata__"}
    payload = jt.load(str(path))
    return payload.get("model", payload)


def load_weights(model, path, encoder_only=False):
    """Load model weights, checking parameter names and shapes."""
    arrays = read_arrays(path)
    target = model.state_dict()
    if encoder_only:
        prefix = ("blocks.", "norm.", "patch_embed.")
        arrays = {k: v for k, v in arrays.items() if k.startswith(prefix)}
    missing = sorted(set(target) - set(arrays))
    unexpected = sorted(set(arrays) - set(target))
    mismatched = [k for k in arrays if k in target and list(arrays[k].shape) != list(target[k].shape)]
    if mismatched or (not encoder_only and (missing or unexpected)):
        raise ValueError(f"incompatible checkpoint: missing={missing}, unexpected={unexpected}, shape={mismatched}. "
                         "Check Model.depth and the checkpoint configuration.")
    if encoder_only and not arrays:
        raise ValueError("no encoder weights matched")
    model.load_state_dict(arrays)
    return {"loaded": len(arrays), "missing": missing, "unexpected": unexpected}


def convert(source, destination):
    import torch
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if "model" in payload:
        payload = payload["model"]
    elif "state_dict" in payload:
        payload = payload["state_dict"]
    arrays = {}
    for key, value in payload.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"non-tensor checkpoint entry: {key}")
        name = key.removeprefix("module.")
        if name in arrays:
            raise ValueError(f"duplicate normalized parameter name: {name}")
        arrays[name] = value.detach().float().numpy()
    if not arrays:
        raise ValueError("empty checkpoint")
    depth = max((int(k.split(".")[1]) + 1 for k in arrays if k.startswith("blocks.")), default=0)
    arrays["__metadata__"] = np.array(json.dumps({"source": Path(source).name, "depth": depth,
                                                 "format": "tasselnetv4-numpy-v1"}))
    destination = Path(destination)
    if destination.suffix != ".npz":
        raise ValueError("destination must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **arrays)
    return {"tensors": len(arrays) - 1, "depth": depth, "destination": str(destination)}


def main():
    parser = argparse.ArgumentParser(description="Convert a PyTorch checkpoint to NumPy weights.")
    parser.add_argument("source", help="Original .pth checkpoint")
    parser.add_argument("destination", help="Output .npz weights")
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
