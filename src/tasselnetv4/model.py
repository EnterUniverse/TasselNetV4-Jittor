"""TasselNetV4 encoder and local count heads.

Modified implementation of tiny-smart/TasselNetV4, Apache-2.0.
"""
from collections import OrderedDict
import math

import jittor as jt
from jittor import nn
import numpy as np


def sincos_position(dim, grid):
    if dim % 4:
        raise ValueError("embedding dimension must be divisible by 4")
    x, y = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32))
    omega = 1.0 / 10000 ** (np.arange(dim // 4, dtype=np.float64) / (dim // 4))
    components = []
    for axis in (x, y):
        phase = axis.reshape(-1, 1) * omega[None]
        components.extend((np.sin(phase), np.cos(phase)))
    return np.concatenate(components, axis=1).astype(np.float32)[None]


def accumulation_weights(grid, kernel, stride):
    """Normalize local count predictions by their spatial overlap."""
    starts = list(range(0, grid - kernel + 1, stride))
    coverage = np.zeros((grid, grid), np.float32)
    for y in starts:
        for x in starts:
            coverage[y:y + kernel, x:x + kernel] += 1
    if not starts or np.any(coverage == 0):
        raise ValueError("local windows must cover the entire image token grid")
    inverse = 1.0 / (coverage * kernel ** 2)
    weights = [[inverse[y:y + kernel, x:x + kernel].sum() for x in starts] for y in starts]
    return np.asarray(weights, np.float32)[None, None]


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        self.img_size = (img_size, img_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

    def execute(self, x):
        if tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(f"expected spatial size {self.img_size}, got {x.shape[-2:]}")
        x = self.proj(x)
        return x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)


class AvgPool(nn.Module):
    """Average pooling with support for a window covering the whole input."""
    def __init__(self, kernel, stride):
        self.kernel, self.stride = kernel, stride

    def execute(self, x):
        if x.shape[-2:] == [self.kernel, self.kernel]:
            return x.mean((2, 3), keepdims=True)
        return nn.avg_pool2d(x, self.kernel, stride=self.stride)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def execute(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def execute(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = nn.softmax(jt.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale, dim=-1)
        out = jt.matmul(attn, v).transpose(0, 2, 1, 3).reshape(b, n, c)
        return self.proj(out), attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def execute(self, x):
        out, attn = self.attn(self.norm1(x))
        x = x + out
        return x + self.mlp(self.norm2(x)), attn


class TasselNetV4(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, decoder_embed_dim=512, mlp_ratio=4,
                 blocksize_list=(32, 64, 128), output_stride_list=(16, 16, 16),
                 counter_dims=(512, 256), groups=32):
        if patch_size != 16 or img_size % patch_size or img_size < 128:
            raise ValueError("patch_size=16 and img_size >=128 divisible by 16 are required")
        if depth < 1 or embed_dim % num_heads:
            raise ValueError("depth must be positive and embed_dim divisible by num_heads")
        if (not blocksize_list or list(blocksize_list) != sorted(set(blocksize_list))
                or len(blocksize_list) != len(output_stride_list)):
            raise ValueError("block sizes must be sorted, unique, and paired with strides")
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.blocksize_list = tuple(blocksize_list)
        self.output_stride_list = tuple(output_stride_list)
        self.scale_embeds = nn.Linear(2, embed_dim)  # retained for checkpoint compatibility
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_exemplar = PatchEmbed(64, patch_size, in_chans + 1, embed_dim)
        self.pos_embed = jt.array(sincos_position(embed_dim, img_size // patch_size)).stop_grad()
        self.pos_embed_exemplar = jt.array(sincos_position(embed_dim, 4)).stop_grad()
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.v_y = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        self.density_proj = nn.Linear(decoder_embed_dim, decoder_embed_dim)
        counters = OrderedDict()
        # Recompute overlap weights instead of storing them in checkpoints.
        self._accumulation = {}
        c1, c2 = counter_dims
        for block, stride in zip(blocksize_list, output_stride_list):
            if block % 16 or stride % 16 or min(block, stride) <= 0:
                raise ValueError("block sizes and strides must be positive multiples of 16")
            k, s = block // 16, stride // 16
            self._accumulation[str(block)] = accumulation_weights(img_size // 16, k, s)
            counters[str(block)] = nn.Sequential(
                nn.Conv2d(embed_dim + 1, c1, 3, padding=1, bias=False),
                nn.GroupNorm(groups, c1), nn.ReLU(),
                nn.Conv2d(c1, c2, 3, padding=1, bias=False),
                nn.GroupNorm(groups, c2), nn.ReLU(),
                AvgPool(k, s), nn.Conv2d(c2, c2, 1),
                nn.GroupNorm(groups, c2), nn.ReLU(), nn.Conv2d(c2, 1, 1))
        self.counter = nn.Sequential(counters)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                bound = math.sqrt(6 / (module.weight.shape[0] + module.weight.shape[1]))
                jt.init.uniform_(module.weight, -bound, bound)
                if module.bias is not None:
                    jt.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                jt.init.constant_(module.weight, 0.01)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                jt.init.constant_(module.weight, 1)
                jt.init.constant_(module.bias, 0)

    def scale_embedding(self, boxes, scales):
        b, n, _, h, w = boxes.shape
        ramp_x = jt.linspace(0, 1, w).reshape(1, 1, 1, 1, w)
        ramp_y = jt.linspace(0, 1, h).reshape(1, 1, 1, h, 1)
        channel = (ramp_x * scales[:, :, 0].reshape(b, n, 1, 1, 1)
                   + ramp_y * scales[:, :, 1].reshape(b, n, 1, 1, 1))
        return jt.concat((boxes, channel), dim=2)

    def forward_encoder(self, image, boxes):
        b, n, c, h, w = boxes.shape
        y = self.patch_embed_exemplar(boxes.reshape(b * n, c, h, w)) + self.pos_embed_exemplar
        y = y.reshape(b, n * 16, self.embed_dim)
        x = self.patch_embed(image) + self.pos_embed
        length = x.shape[1]
        xy = jt.concat((x, y), dim=1)
        attn = None
        for block in self.blocks:
            xy, attn = block(xy)
        xy = self.norm(xy)
        return xy[:, :length], attn

    def select_head(self, size):
        for level in self.blocksize_list:
            if float(size) < level:
                return str(level)
        return str(self.blocksize_list[-1])

    def execute(self, samples, size=None, test=True, single=False):
        image, boxes, scales = samples[:3]
        if image.ndim != 4 or boxes.ndim != 5 or scales.ndim != 3:
            raise ValueError("expected image B,C,H,W; boxes B,N,3,64,64; scales B,N,2")
        b, n = boxes.shape[:2]
        if n < 1 or list(scales.shape) != [b, n, 2] or image.shape[0] != b:
            raise ValueError("at least one exemplar and matching image/scale batches are required")
        x, attn = self.forward_encoder(image, self.scale_embedding(boxes, scales))
        length = x.shape[1]
        grid = self.img_size // 16
        # Exemplar scales are measured relative to a 384-pixel query tile.
        weights = 16 / ((scales[:, :, 0] + scales[:, :, 1]) ** 2 * 384)
        exemplar_attention = attn.mean(dim=1)[:, length:, :length].reshape(b, n, 16, length)
        enhance = (exemplar_attention * weights.reshape(b, n, 1, 1)).sum(dim=2).mean(dim=1)
        enhance = enhance.reshape(b, 1, grid, grid)
        feature = jt.concat((x.transpose(0, 2, 1).reshape(b, self.embed_dim, grid, grid), enhance), dim=1)
        if not test:
            return {key: head(feature).squeeze(1) for key, head in self.counter.items()}
        if size is None:
            size = samples[3] if len(samples) > 3 else scales.max(dim=2).max(dim=1) * 384
        sizes = np.asarray(size.numpy() if isinstance(size, jt.Var) else size).reshape(-1)
        if sizes.size == 1:
            sizes = np.repeat(sizes, b)
        if sizes.size != b:
            raise ValueError("size must be scalar or contain one value per image")
        levels = [self.select_head(s) for s in sizes]
        if len(set(levels)) != 1:
            raise ValueError("inference batch must select one common head; use batch size 1")
        level = levels[0]
        result = self.counter[level](feature) * jt.array(self._accumulation[level]).stop_grad()
        if single:
            return result, enhance, attn
        return result, enhance


def build_model(config=None, **kwargs):
    """Build a model from configuration options."""
    options = dict(config or {})
    options.update(kwargs)
    name = options.pop("name", "local_count_mutihead_loose")
    if name not in ("local_count_mutihead_loose", "TasselNetV4", "mae_vit_base_patch16"):
        raise ValueError(f"unsupported model: {name}")
    legacy = {"mode": "GlobalAttention", "decodemode": "GlobalAttention",
              "similarityfunc": "PatchConv", "similaritymode": "OutputAdd"}
    for key, expected in legacy.items():
        if options.pop(key, expected) != expected:
            raise ValueError(f"only {key}={expected} is supported")
    for key in ("gamma", "mullayer", "norm_pix_loss", "xenhance", "test", "updown"):
        options.pop(key, None)  # unused by the upstream forward path
    return TasselNetV4(**options)
