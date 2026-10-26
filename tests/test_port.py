"""Model, data pipeline and checkpoint tests with synthetic inputs."""
import csv
import json

import jittor as jt
import numpy as np
from PIL import Image
import pytest

from tasselnetv4.checkpoint import load_weights
from tasselnetv4.data import PACDataset, point_map
from tasselnetv4.evaluate import metrics, predict_count
from tasselnetv4.model import TasselNetV4, accumulation_weights
from tasselnetv4.train import local_count_loss, parameter_groups


@pytest.fixture
def model():
    return TasselNetV4(img_size=128, embed_dim=32, depth=2, num_heads=4,
                       decoder_embed_dim=16, counter_dims=(32, 32), groups=4)


def inputs(shots=3, size=20):
    rng = np.random.default_rng(4)
    return [jt.array(rng.normal(size=(1, 3, 128, 128)).astype(np.float32)),
            jt.array(rng.normal(size=(1, shots, 3, 64, 64)).astype(np.float32)),
            jt.array(np.full((1, shots, 2), 0.1, np.float32)), jt.array([size])]


def test_overlap_normalizes_constant_density():
    for grid in (8, 24):
        for k in (2, 4, 8):
            weights = accumulation_weights(grid, k, 1)
            np.testing.assert_allclose((weights * k ** 2).sum(), grid ** 2, rtol=2e-6)
    with pytest.raises(ValueError):
        accumulation_weights(8, 2, 3)


def test_forward_all_heads_and_one_shot(model):
    model.eval()
    for shots in (1, 3):
        samples = inputs(shots)
        outputs = model(samples, test=False)
        assert {k: list(v.shape) for k, v in outputs.items()} == {"32": [1, 7, 7], "64": [1, 5, 5], "128": [1, 1, 1]}
        for size, shape in ((20, [1, 1, 7, 7]), (32, [1, 1, 5, 5]), (64, [1, 1, 1, 1]), (160, [1, 1, 1, 1])):
            result, attention = model(samples, size=size)
            assert list(result.shape) == shape
            assert list(attention.shape) == [1, 1, 8, 8]
            assert np.isfinite(result.numpy()).all()


def test_backward_updates_encoder_and_selected_head(model):
    model.train()
    samples = inputs()
    optimizer = jt.optim.AdamW(parameter_groups(model, 0.05), lr=1e-3)
    before_encoder = model.patch_embed.proj.weight.numpy().copy()
    before_head = model.counter["32"][10].weight.numpy().copy()
    outputs = model(samples, test=False)
    density = jt.ones((1, 128, 128)) / 128 ** 2
    loss = local_count_loss(model, outputs, density, jt.array([[1., 0., 0.]]))
    assert np.isfinite(loss.item())
    optimizer.step(loss)
    jt.sync_all()
    assert not np.array_equal(before_encoder, model.patch_embed.proj.weight.numpy())
    assert not np.array_equal(before_head, model.counter["32"][10].weight.numpy())


def test_checkpoint_strict_roundtrip(model, tmp_path):
    model.eval()
    original = model(inputs())[0].numpy()
    path = tmp_path / "weights.npz"
    state = model.state_dict(to="numpy")
    np.savez(path, **state)
    load_weights(model, path)
    np.testing.assert_allclose(model(inputs())[0].numpy(), original, atol=1e-6)
    del state["blocks.0.attn.qkv.weight"]
    np.savez(path, **state)
    with pytest.raises(ValueError, match="missing"):
        load_weights(model, path)


def make_dataset(root):
    with (root / "dataset.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "variety", "foldname"])
        for split in ("train", "val", "test"):
            writer.writerow([split, "plant", split])
            folder = root / "plant" / split
            (folder / "images").mkdir(parents=True)
            (folder / "labels").mkdir()
            Image.new("RGB", (512, 384), (80, 160, 50)).save(folder / "images" / "plant.png")
            with (folder / "labels" / "labels.csv").open("w", newline="") as labels:
                output = csv.writer(labels)
                output.writerow(["filename", "region_shape_attributes"])
                for box in range(3):
                    output.writerow(["plant.png", json.dumps({"name": "rect", "x": 10 + box * 30, "y": 10,
                                                              "width": 20, "height": 30})])
                for x, y in [(50, 60), (50, 60), (300, 200)]:
                    output.writerow(["plant.png", json.dumps({"name": "point", "cx": x, "cy": y})])


def test_dataset_batches_and_point_mass(tmp_path):
    make_dataset(tmp_path)
    assert point_map([[0, 0], [0, 0]], (2, 2)).sum() == 2
    dataset = PACDataset(tmp_path, augment=False)
    sample = next(iter(dataset))
    assert list(sample["image"].shape) == [1, 3, 384, 384]
    assert list(sample["boxes"].shape) == [1, 3, 3, 64, 64]
    assert np.isclose(sample["size_embedding"].sum().item(), 1)
    assert sample["gt_density"].sum().item() <= 3.001
    validation = next(iter(PACDataset(tmp_path, "val")))
    assert validation["dots"].item() == 3
    assert validation["gt_density"].sum().item() == 3
    # Exercise each augmentation family with synthetic PAC records.
    augmented = PACDataset(tmp_path)
    for seed in range(8):
        import random
        random.seed(seed)
        np.random.seed(seed)
        item = augmented[0]
        assert item["image"].shape == (3, 384, 384)
        assert np.isfinite(item["image"]).all()
        assert item["gt_density"].min() >= 0


def test_tiled_count_and_metrics(model):
    samples = inputs()
    count = predict_count(model, samples[0], *samples[1:])
    assert np.isfinite(count)
    assert metrics([1, 2], [1, 2])["R2"] == 1
    assert metrics([0], [0])["WCA"] is None


def test_training_entry_checkpoint_and_resume(tmp_path, monkeypatch):
    import sys
    import yaml
    from tasselnetv4.train import main
    make_dataset(tmp_path)
    cfg = {"Model": {"embed_dim": 32, "depth": 1, "num_heads": 4,
                     "decoder_embed_dim": 16, "counter_dims": [32, 32], "groups": 4},
           "Dataset": {"batch_size": 1, "num_workers": 0},
           "Loss": {"loss": "L1", "use_mask": False},
           "Optimizer": {"epochs": 1, "accum_iter": 2, "lr": 0.001, "warmup_epochs": 0},
           "Training": {"device": "cpu", "seed": 728}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    destination = tmp_path / "output"
    command = ["tasselnet-train", "--config", str(path), "--data", str(tmp_path),
               "--output", str(destination), "--no-augment"]
    monkeypatch.setattr(sys, "argv", command)
    main()
    assert (destination / "last.pkl").exists()
    monkeypatch.setattr(sys, "argv", command + ["--resume", str(destination / "last.pkl"), "--epochs", "2"])
    main()
    history = [json.loads(row) for row in (destination / "history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [0, 1]
    assert all(np.isfinite(row["loss"]) for row in history)


def test_pytorch_checkpoint_conversion(model, tmp_path):
    torch = pytest.importorskip("torch")
    from tasselnetv4.checkpoint import convert, read_arrays
    state = {"module." + key: torch.from_numpy(value.copy())
             for key, value in model.state_dict(to="numpy").items()}
    source, destination = tmp_path / "original.pth", tmp_path / "converted.npz"
    torch.save({"model": state}, source)
    result = convert(source, destination)
    assert result["depth"] == 2
    assert result["tensors"] == len(state)
    load_weights(model, destination)
    arrays = read_arrays(destination)
    assert "__metadata__" not in arrays
    for key, value in model.state_dict(to="numpy").items():
        np.testing.assert_array_equal(arrays[key], value)
