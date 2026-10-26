"""PAC annotations and image augmentation.

Adapted from tiny-smart/TasselNetV4 datasets/plant.py (Apache-2.0).
"""
import csv
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import gaussian_filter
from jittor.dataset import Dataset

MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def normalize(image):
    return (np.asarray(image, np.float32).transpose(2, 0, 1) / 255 - MEAN) / STD


def resize_chw(array, size):
    """Antialiased bilinear resize of float channels; size is (height,width)."""
    return np.stack([np.asarray(Image.fromarray(c).resize((size[1], size[0]), Image.Resampling.BILINEAR))
                     for c in array]).astype(np.float32)


def point_map(points, shape):
    density = np.zeros(shape, np.float32)
    points = np.asarray(points).reshape(-1, 2)
    if len(points):
        xy = np.rint(points).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, shape[1] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, shape[0] - 1)
        np.add.at(density, (xy[:, 1], xy[:, 0]), 1)
    return density


class PACIndex:
    def __init__(self, root):
        self.root = Path(root)
        self.annotations = {}
        self.splits = {s: [] for s in ("train", "val", "test")}
        self.classes = {}
        with (self.root / "dataset.csv").open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            split = row["split"]
            if split not in self.splits:
                raise ValueError(f"unknown split {split}")
            folder = Path(row["variety"]) / row["foldname"]
            for path in sorted((self.root / folder / "images").iterdir()):
                if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
                    continue
                key = (folder / "images" / path.name).as_posix()
                if key in self.classes:
                    raise ValueError(f"duplicate image or split leakage: {key}")
                self.splits[split].append(key)
                self.classes[key] = folder.as_posix()
                self.annotations[key] = {"boxes": [], "points": []}
            for path in sorted((self.root / folder / "labels").glob("*.csv")):
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    for annotation in csv.DictReader(stream):
                        key = (folder / "images" / annotation["filename"]).as_posix()
                        if key not in self.annotations:
                            raise ValueError(f"annotation image missing: {key}")
                        shape = json.loads(annotation["region_shape_attributes"] or "{}")
                        if shape.get("name") == "rect":
                            x, y, w, h = (round(float(shape[k])) for k in ("x", "y", "width", "height"))
                            if min(w, h) <= 0:
                                raise ValueError(f"invalid exemplar in {key}")
                            self.annotations[key]["boxes"].append([x, y, x + w, y + h])
                        elif shape.get("name") == "point":
                            self.annotations[key]["points"].append([float(shape["cx"]), float(shape["cy"])])

    def load(self, key, training=False):
        annotation = self.annotations[key]
        with Image.open(self.root / key) as source:
            image = source.convert("RGB")
        points = np.asarray(annotation["points"], np.float32).reshape(-1, 2)
        boxes = np.asarray(annotation["boxes"], np.float32).reshape(-1, 4)
        if not len(boxes):
            raise ValueError(f"no exemplar rectangles for {key}")
        if image.width < image.height:
            image = image.transpose(Image.Transpose.TRANSPOSE)
            points = points[:, ::-1].copy()
            boxes = boxes[:, [1, 0, 3, 2]]
        old_w, old_h = image.size
        new_h = 384
        new_w = max(384, int(math.ceil(old_w * 384 / old_h)) // 16 * 16)
        if training and random.random() > 0.5:
            scale = random.uniform(1, 1.5)
            ratio = random.uniform(1, 1.2) if random.random() > 0.5 else 1
            new_h = int(new_h * scale) // 16 * 16
            new_w = int(new_w * scale * ratio) // 16 * 16
        image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        factor = np.array([new_w / old_w, new_h / old_h], np.float32)
        return image, points * factor, np.rint(boxes * np.tile(factor, 2)).astype(int)


def extract_exemplars(image, boxes, shots=3):
    normalized = normalize(image)
    crops, scales = [], []
    for x1, y1, x2, y2 in boxes[:shots]:
        x1, x2 = np.clip([x1, x2], 0, image.width - 1)
        y1, y2 = np.clip([y1, y2], 0, image.height - 1)
        if x2 < x1 or y2 < y1:
            raise ValueError("exemplar does not intersect image")
        crops.append(resize_chw(normalized[:, y1:y2 + 1, x1:x2 + 1], (64, 64)))
        scales.append([(x2 - x1 + 1) / 384, (y2 - y1 + 1) / 384])
    if not crops:
        raise ValueError("at least one exemplar is required")
    # Repeat the last exemplar to keep the batch shape consistent.
    while len(crops) < shots:
        crops.append(crops[-1].copy())
        scales.append(scales[-1].copy())
    return np.stack(crops), np.asarray(scales, np.float32)


def _crop_points(points, x, y, length, output):
    inside = ((points[:, 0] >= x) & (points[:, 0] < x + length)
              & (points[:, 1] >= y) & (points[:, 1] < y + length))
    return (points[inside] - [x, y]) * output / length


def mosaic(index, key, image, points):
    """Combine four image crops with blended seams."""
    blend = random.randint(10, 20)
    side = 192 + 2 * blend
    images, maps = [], []
    target_slot = random.randint(0, 3) if random.random() > 0.25 else random.randint(0, 4)
    for slot in range(4):
        other_key = key
        other_image, other_points = image, points
        if len(points) < 70 and slot != target_slot:
            other_key = random.choice(index.splits["train"])
            other_image, other_points, _ = index.load(other_key)
        length = random.randint(350, min(other_image.size))
        x = random.randint(0, other_image.width - length)
        y = random.randint(0, other_image.height - length)
        images.append(np.asarray(other_image.crop((x, y, x + length, y + length)).resize(
            (side, side), Image.Resampling.BILINEAR), np.float32) / 255)
        transformed = _crop_points(other_points, x, y, length, side)
        if index.classes[other_key] != index.classes[key]:
            transformed = np.empty((0, 2), np.float32)
        maps.append(point_map(transformed, (side, side)))

    def join(a, b, axis, do_blend):
        idx_a, idx_b = [slice(None)] * a.ndim, [slice(None)] * b.ndim
        idx_a[axis] = idx_b[axis] = slice(blend, side - blend)
        out = np.concatenate((a[tuple(idx_a)], b[tuple(idx_b)]), axis=axis)
        if do_blend:
            for offset in range(blend):
                left, right, a_edge, b_edge = ([slice(None)] * a.ndim for _ in range(4))
                left[axis], right[axis] = 191 - offset, 192 + offset
                a_edge[axis], b_edge[axis] = side - 1 - blend + offset, blend - offset
                alpha = (blend - offset) / (2 * blend)
                out[tuple(right)] = a[tuple(a_edge)] * alpha + out[tuple(right)] * (1 - alpha)
                out[tuple(left)] = b[tuple(b_edge)] * alpha + out[tuple(left)] * (1 - alpha)
        return out

    rgb = join(join(images[0], images[1], 0, True), join(images[2], images[3], 0, True), 1, True)
    density = join(join(maps[0], maps[1], 0, False), join(maps[2], maps[3], 0, False), 1, False)
    return (rgb.transpose(2, 0, 1) - MEAN) / STD, density


def photometric(image):
    for enhancer, extent in ((ImageEnhance.Brightness, 0.25), (ImageEnhance.Contrast, 0.15),
                             (ImageEnhance.Color, 0.15)):
        image = enhancer(image).enhance(random.uniform(1 - extent, 1 + extent))
    hsv = np.array(image.convert("HSV"))
    hsv[..., 0] = (hsv[..., 0].astype(int) + round(random.uniform(-0.15, 0.15) * 255)) % 256
    image = Image.fromarray(hsv, mode="HSV").convert("RGB")
    image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 2)))
    array = np.asarray(image, np.float32) / 255
    return np.clip(array + np.random.normal(0, 0.1, array.shape), 0, 1).astype(np.float32)


class PACDataset(Dataset):
    def __init__(self, root, split="train", batch_size=1, num_workers=0,
                 shots=3, block_sizes=(32, 64, 128), augment=True):
        super().__init__()
        if shots < 1:
            raise ValueError("shots must be positive")
        self.index = PACIndex(root)
        splits = [split] if isinstance(split, str) else split
        if "train" in splits and len(splits) > 1:
            raise ValueError("train and evaluation splits cannot be combined")
        self.training = splits == ["train"]
        self.keys = [key for s in splits for key in self.index.splits[s]]
        if not self.keys:
            raise ValueError(f"empty dataset split: {splits}")
        self.shots, self.block_sizes, self.augment = shots, tuple(block_sizes), augment
        self.set_attrs(total_len=len(self.keys), batch_size=batch_size if self.training else 1,
                       shuffle=self.training, num_workers=num_workers, drop_last=False)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image, points, rects = self.index.load(key, training=self.training and self.augment)
        boxes, scales = extract_exemplars(image, rects, self.shots)
        size = np.float32(scales.max() * 384)
        if self.training:
            probability = random.random() if self.augment else 1
            if probability < 0.25:
                pixels, density = mosaic(self.index, key, image, points)
            else:
                pixels = (photometric(image).transpose(2, 0, 1) - MEAN) / STD if probability < 0.4 else normalize(image)
                density = point_map(points, (image.height, image.width))
                if probability < 0.4 and random.random() > 0.5:
                    pixels, density = pixels[:, :, ::-1], density[:, ::-1]
                x = random.randint(0, image.width - 384)
                y = random.randint(0, image.height - 384)
                pixels, density = pixels[:, y:y + 384, x:x + 384], density[y:y + 384, x:x + 384]
            # Smooth point labels using the exemplar dimensions.
            width, height = scales[:, 0].mean() * 384, scales[:, 1].mean() * 384
            density = gaussian_filter(density, sigma=(height / 60, width / 60), mode="reflect")
            head_idx = next((i for i, block in enumerate(self.block_sizes) if size < block), len(self.block_sizes) - 1)
            embedding = np.zeros(len(self.block_sizes), np.float32)
            embedding[head_idx] = 1
        else:
            pixels = normalize(image)
            density = point_map(points, (image.height, image.width))
            embedding = np.zeros(len(self.block_sizes), np.float32)
        return {"image": np.ascontiguousarray(pixels, dtype=np.float32), "boxes": boxes,
                "scale": scales, "size": float(size), "size_embedding": embedding,
                "gt_density": density.astype(np.float32), "dots": float(len(points)), "id": key}
