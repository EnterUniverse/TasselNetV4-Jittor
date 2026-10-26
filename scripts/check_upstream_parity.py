"""Compare model outputs with the original PyTorch implementation.

Requires torch, torchvision, timm==0.5.4, einops, thop, matplotlib, opencv-python.
"""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import sys

import jittor as jt
import numpy as np
import torch

from tasselnetv4.model import TasselNetV4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="Original TasselNetV4 source checkout")
    args = parser.parse_args()
    sys.path.insert(0, str(args.upstream.resolve()))
    path = args.upstream / "models/local_count_multihead/CntVit_3layers_scalepos_patchmat_withr_plain.py"
    spec = importlib.util.spec_from_file_location("tasselnetv4_torch_reference", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    torch.manual_seed(4)
    # The reference model requires 768 channels.
    with contextlib.redirect_stdout(io.StringIO()):
        original = reference.local_count_mutihead_loose(img_size=128, depth=2,
                    blocksize_list=[32, 64, 128], output_stride_list=[16, 16, 16]).eval()
    migrated = TasselNetV4(img_size=128, depth=2).eval()
    with torch.no_grad():
        # Nonconstant weights make the comparison sensitive to layout mistakes.
        for name, param in original.named_parameters():
            if param.requires_grad:
                if "norm" in name or ("counter" in name and param.ndim == 1):
                    param.uniform_(0.8, 1.2) if name.endswith("weight") else param.uniform_(-0.02, 0.02)
                else:
                    param.normal_(0, 0.02)
    arrays = {k: v.numpy() for k, v in original.state_dict().items()}
    target = migrated.state_dict()
    assert set(target) == set(arrays), (set(target) - set(arrays), set(arrays) - set(target))
    assert all(list(target[k].shape) == list(v.shape) for k, v in arrays.items())
    migrated.load_state_dict(arrays)
    rng = np.random.default_rng(9)
    errors = {}
    for shots in (1, 3):
        values = [rng.normal(size=(1, 3, 128, 128)).astype(np.float32),
                  rng.normal(size=(1, shots, 3, 64, 64)).astype(np.float32),
                  rng.uniform(0.05, 0.3, size=(1, shots, 2)).astype(np.float32),
                  np.array([20], np.float32)]
        torch_inputs = [torch.from_numpy(x) for x in values]
        jittor_inputs = [jt.array(x) for x in values]
        with torch.no_grad(), jt.no_grad():
            expected = original(torch_inputs, test=False)
            actual = migrated(jittor_inputs, test=False)
            for key in expected:
                a, b = actual[key].numpy(), expected[key].numpy()
                np.testing.assert_allclose(a, b, atol=3e-4, rtol=3e-4)
                errors[f"{shots}-shot/head-{key}"] = float(np.max(np.abs(a - b)))
            for size in (20, 40, 80):
                torch_inputs[-1] = torch.tensor([size], dtype=torch.float32)
                jittor_inputs[-1] = jt.array([size], dtype="float32")
                expected, expected_attn = original(torch_inputs)
                actual, actual_attn = migrated(jittor_inputs)
                np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=3e-4, rtol=3e-4)
                np.testing.assert_allclose(actual_attn.numpy(), expected_attn.numpy(), atol=3e-5, rtol=3e-4)
    import json
    print(json.dumps({"max_absolute_errors": errors, "parameter_keys": len(target)}, indent=2))


if __name__ == "__main__":
    main()
