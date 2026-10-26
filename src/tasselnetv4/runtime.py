"""Shared configuration, deterministic seeding, and device selection."""
import random
import jittor as jt
import numpy as np
import yaml


def load_config(path, device=None):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    selected = device or config.get("Training", {}).get("device", "cpu")
    if selected not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    jt.flags.use_cuda = int(selected == "cuda")
    seed = config.get("Training", {}).get("seed", 728)
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)
    return config
