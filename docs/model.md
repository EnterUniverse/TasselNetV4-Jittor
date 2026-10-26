# Model and data

## Model

The default configuration uses a 12-block, 768-dimensional encoder with
16-pixel patches, 384×384 query tiles and 64×64 exemplar crops. Exemplars
include a scale-position channel. Exemplar-to-query attention enhances the
features passed to the 32/64/128 local count heads, whose predictions use
inverse-overlap correction.

Attention uses Jittor matmul and softmax. Only the final attention matrix is
retained for prediction. Parameter names and layouts match the original model,
so converted linear weights do not need transposition. The unused
`scale_embeds`, `v_y` and `density_proj` parameters remain compatible with
existing checkpoints and are excluded from optimization.

The original training and test YAMLs specify depths 12 and 11. Set
`Model.depth` to the value printed by the checkpoint converter. The loader
checks parameter names and shapes. Legacy model options are accepted where
they match the implemented architecture; the unused `xenhance` field is ignored.

Head selection uses the largest exemplar side in resized image pixels.
The thresholds use strict `<` comparisons: size 32 selects the 64 head,
and size 64 selects the 128 head.

## Data and training

The data loader reads PAC split CSVs and VIA annotations. Portrait images and
their coordinates are transposed together. Exemplars are resized to 64×64;
when fewer boxes than the requested shot count are available, the last box
is repeated for batching. Missing exemplars and conflicting splits raise errors.

Augmentation includes random scaling and cropping, noise, color jitter, blur,
horizontal flips, self/cross-class mosaic and Gaussian density smoothing.
It uses Pillow, NumPy and SciPy, so random sampling and interpolation differ
from torchvision. Mosaic smoothing uses exemplar dimensions in the resized
source image.

Local count supervision and optimization use the following conventions:

- Points that land on the same pixel retain their combined count.
- Gaussian smoothing uses separate mean exemplar widths and heights.
- Local count targets use average pooling multiplied by window area.
- Gated head losses are summed before backpropagation.
- Gradient accumulation includes the final incomplete window.

AdamW excludes bias and normalization weights from decay. The default batch
size is 8 and can be changed in the configuration.

Evaluation center-pads images into disjoint 384×384 tiles and sums their local
count predictions. Metrics use annotation counts. Undefined WCA and R² values
are saved as JSON null. The default evaluation split is `val`.

## Checkpoints

The converter reads PyTorch tensor checkpoints with `weights_only=True` and
writes NumPy `.npz` arrays. It accepts raw state dictionaries, `model` and
`state_dict` wrappers, and DDP `module.` prefixes.

Jittor training checkpoints contain model weights, optimizer state, epoch and
configuration. Use `--checkpoint` to load model weights or `--resume` to restore
training state. Encoder initialization with `--encoder` loads matching patch,
transformer and normalization weights while keeping the fixed positions and
newly initialized count heads.

## CPU setup

If Jittor detects an incomplete CUDA installation on a CPU machine, disable
CUDA discovery for the process:

```bash
nvcc_path='' use_cuda=0 pytest -q
```
