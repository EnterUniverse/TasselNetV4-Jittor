# TasselNetV4-Jittor

Jittor implementation of [TasselNetV4](https://arxiv.org/abs/2509.20857) for
plant counting. Given an image and a few exemplar boxes, the model estimates
the number of plants in the image.

Based on [tiny-smart/TasselNetV4](https://github.com/tiny-smart/TasselNetV4).

## Installation

Requires Linux, Python 3.9–3.12 and `g++`. Development and testing use Python 3.10.

```bash
git clone https://github.com/EnterUniverse/TasselNetV4-Jittor.git
cd TasselNetV4-Jittor
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Jittor compiles its runtime on first use. See the
[Jittor installation guide](https://cg.cs.tsinghua.edu.cn/jittor/) for CUDA setup.
Commands use CPU by default; add `--device cuda` to use a GPU.

## Pretrained weights

Download the original weights from the
[upstream project](docs/upstream/README.md#inference), then convert them:

```bash
pip install '.[conversion]'
tasselnet-convert original.pth checkpoints/tasselnetv4.npz
```

PyTorch is needed for conversion. Training and inference use Jittor.
The converter prints the encoder depth; set `Model.depth` in
`configs/default.yaml` to match it. The original training and test
configurations use depths 12 and 11 respectively.

## Inference

Create `boxes.json` with exemplar boxes in original image coordinates,
using `[x1, y1, x2, y2]`:

```json
[[40, 25, 80, 75], [180, 90, 220, 145], [290, 60, 330, 110]]
```

```bash
tasselnet-predict image.jpg --boxes boxes.json \
  --checkpoint checkpoints/tasselnetv4.npz
```

The command prints the predicted count as JSON. Use one box for one-shot
prediction. Images are resized and processed in 384×384 tiles.

## Dataset

Download [PAC-105 or PAC-Somalia](docs/upstream/README.md#prepare-dataset)
and arrange the files as follows:

```text
PAC/
├── dataset.csv                  # split,variety,foldname
└── aska/
    └── aska_fruit/
        ├── images/image1.png
        └── labels/aska.csv      # filename,region_shape_attributes,...
```

Label CSVs use VIA annotations: `rect` shapes for exemplar boxes and `point`
shapes for plant centers.

## Training

Set the batch size and optimizer options in `configs/default.yaml`, then run:

```bash
tasselnet-train --data /path/to/PAC --output outputs/train --device cuda
```

Use `--encoder converted_mae.npz` to initialize the encoder from MAE weights,
`--checkpoint model.npz` to load model weights, or `--resume outputs/train/last.pkl`
to resume training with optimizer state. Add `--shots 1` for one-shot training
and `--eval-every N` to validate every N epochs.

Checkpoints and training logs are saved under `--output`.

## Evaluation

```bash
tasselnet-evaluate --data /path/to/PAC \
  --checkpoint checkpoints/tasselnetv4.npz \
  --split val test --output outputs/evaluation.json --device cuda
```

Results include MAE, RMSE, WCA, R² and per-image predictions.
Use `--shots 1` for one-shot evaluation. Both training and evaluation accept
`--config path/to/config.yaml`. The scripts `train_val.py` and
`test_checkpoint.py` provide the same commands.

## Development

```bash
pip install -e '.[test]'
pytest -q
```

See [model and data details](docs/model.md) for configuration and preprocessing.

## Reference

Xiaonan Hu et al. *TasselNetV4: A vision foundation model for cross-scene,
cross-scale, and cross-species plant counting.* ISPRS Journal of Photogrammetry
and Remote Sensing, 2025.

[Paper](https://arxiv.org/abs/2509.20857) ·
[Original implementation](https://github.com/tiny-smart/TasselNetV4)

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
