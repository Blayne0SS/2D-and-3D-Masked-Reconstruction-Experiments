#!/usr/bin/env python3


from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import nibabel as nib
import numpy as np
import torch
from monai.metrics import SSIMMetric
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Default folder containing the aligned MRI volumes.
DATA_DIR = "/mnt/hpccs01/home/n10514821/X_ray_models/SYN_aligned_mri_cache_block"
METRIC_NAMES = ("masked_mae_01", "masked_mse_01", "masked_psnr_db", "hole_ssim_3d")


# Store the data, model and training settings in one place.

@dataclass
class Config:
    data_dir: str = DATA_DIR
    output: str = "mri_autoencoder_run"
    architecture: str = "recurrent"
    block_shape: tuple[int, int, int] = (48, 48, 48)
    mask_shape: tuple[int, int, int] = (21, 21, 21)
    variable_masks: bool = False
    mask_min: tuple[int, int, int] = (11, 11, 11)
    mask_max: tuple[int, int, int] = (25, 25, 25)
    context_margin: int = 2
    base_channels: int = 28
    dropout: float = 0.1
    epochs: int = 501
    batch_size: int = 4
    workers: int = 4
    learning_rate: float = 4e-5
    weight_decay: float = 1e-2
    ema_decay: float = 0.999
    seed: int = 42
    train_patches_per_volume: int = 1
    eval_cases_per_volume: int = 3
    foreground_threshold: float = 0.15
    minimum_foreground_fraction: float = 0.5
    subject_regex: str = r"(IXI\d+|sub-[A-Za-z0-9]+)"
    save_examples: int = 12
    patience: int = 0  # 0 disables early stopping; best checkpoint is always saved.
    checkpoint_every: int = 50
    amp: bool = True
    device: str = "auto"

    def validate(self) -> None:
        for name in ("block_shape", "mask_shape", "mask_min", "mask_max"):
            value = tuple(int(v) for v in getattr(self, name))
            setattr(self, name, value)
            if len(value) != 3 or min(value) < 3:
                raise ValueError(f"{name} needs three dimensions, each >=3.")
        if any(v % 8 for v in self.block_shape) or min(self.block_shape) < 16:
            raise ValueError("Block dimensions must be >=16 and divisible by 8.")
        if self.context_margin < 1:
            raise ValueError("context_margin must be >=1.")
        for lo, hi in zip(self.mask_min, self.mask_max):
            if lo > hi:
                raise ValueError("Every mask_min must be <= the corresponding mask_max.")
        active_max = self.mask_max if self.variable_masks else self.mask_shape
        if any(m + 2 * self.context_margin > b for m, b in zip(active_max, self.block_shape)):
            raise ValueError("Mask plus context margins must fit inside the block.")
        if self.architecture not in {"recurrent", "simple"}:
            raise ValueError("architecture must be recurrent or simple.")
        for name in ("base_channels", "epochs", "batch_size", "train_patches_per_volume", "eval_cases_per_volume", "checkpoint_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.workers < 0 or self.save_examples < 0 or self.patience < 0:
            raise ValueError("workers, save_examples and patience must be nonnegative.")
        if not 0 <= self.dropout < 1 or not 0 <= self.ema_decay < 1:
            raise ValueError("dropout and ema_decay must be in [0,1).")
        if not 0 < self.foreground_threshold < 1 or not 0 <= self.minimum_foreground_fraction <= 1:
            raise ValueError("Invalid foreground sampling settings.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings.")
        re.compile(self.subject_regex)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(name)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("This script supports CPU or CUDA devices.")
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return result


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Repeatable stochastic choices; CUDA kernels need not be bitwise identical.
    torch.backends.cudnn.benchmark = False


def identify_subject(path: Path, pattern: str) -> str:
    # Prefer the filename, then nearest enclosing subject directory.
    for part in (path.name, *reversed(path.parent.parts)):
        match = re.search(pattern, part, flags=re.IGNORECASE)
        if match:
            return (match.group(1) if match.lastindex else match.group(0)).lower()
    raise ValueError(f"No subject ID found in {path}; set --subject-regex explicitly.")


# Find MRI scans and split them by subject for training, validation and testing.

def discover_and_split(cfg: Config) -> tuple[dict, dict]:
    root = Path(cfg.data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Aligned MRI directory does not exist: {root}")
    paths = sorted({p.resolve() for p in root.rglob("*") if p.is_file() and
                    (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))})
    if not paths:
        raise RuntimeError(f"No .nii or .nii.gz scans found in {root}.")
    records = [{"path": str(p), "subject": identify_subject(p, cfg.subject_regex)} for p in paths]
    subjects = sorted({r["subject"] for r in records})
    n_train, n_val = int(0.7 * len(subjects)), int(0.2 * len(subjects))
    if min(n_train, n_val, len(subjects) - n_train - n_val) < 1:
        raise ValueError("At least five subjects are required for a 70/20/10 group split.")
    ordered = np.asarray(subjects)[np.random.default_rng(cfg.seed).permutation(len(subjects))]
    subject_splits = {"train": set(ordered[:n_train]), "val": set(ordered[n_train:n_train+n_val]),
                      "test": set(ordered[n_train+n_val:])}
    splits = {name: [r for r in records if r["subject"] in ids] for name, ids in subject_splits.items()}
    assert not (subject_splits["train"] & subject_splits["val"])
    assert not (subject_splits["train"] & subject_splits["test"])
    assert not (subject_splits["val"] & subject_splits["test"])

    reference = nib.load(records[0]["path"])
    if len(reference.shape) != 3:
        raise ValueError("Only 3D single-channel MRI volumes are supported.")
    geometry = {"shape_xyz": list(reference.shape), "affine": reference.affine.tolist()}
    for record in tqdm(records, desc="Checking aligned grids"):
        scan = nib.load(record["path"])
        check_geometry(scan, geometry)
    if any(v < b for v, b in zip(reference.shape[::-1], cfg.block_shape)):
        raise ValueError("A block is larger than the aligned volume.")
    return splits, geometry


def check_geometry(image: nib.spatialimages.SpatialImage, geometry: dict) -> None:
    if tuple(image.shape) != tuple(geometry["shape_xyz"]):
        raise ValueError(f"Grid shape mismatch for {image.get_filename()}: {image.shape}.")
    if not np.allclose(image.affine, geometry["affine"], atol=1e-4, rtol=0):
        raise ValueError(f"Affine mismatch for {image.get_filename()}; use a consistently aligned cache.")


# Load a NIfTI volume and return its image data and metadata.

def load_volume(path: str, allow_nonfinite: bool = False) -> tuple[np.ndarray, Any]:
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3D scan: {path}")
    # The transpose is reversed on export, preserving NIfTI orientation.
    volume = np.ascontiguousarray(image.get_fdata(dtype=np.float32).transpose(2, 1, 0))
    if not allow_nonfinite and not np.isfinite(volume).all():
        raise ValueError(f"Nonfinite values in a complete training/evaluation scan: {path}")
    return volume, image


def slices3(start, shape) -> tuple[slice, slice, slice]:
    return tuple(slice(int(a), int(a + b)) for a, b in zip(start, shape))


def random_mask_shape(cfg: Config, rng: np.random.Generator) -> tuple[int, int, int]:
    if not cfg.variable_masks:
        return cfg.mask_shape
    # Both odd and even sizes are correct because stop = start + size.
    return tuple(int(rng.integers(lo, hi + 1)) for lo, hi in zip(cfg.mask_min, cfg.mask_max))


# Choose the block and missing region used for one reconstruction case.

def sample_case(volume: np.ndarray, record: dict, cfg: Config,
                rng: np.random.Generator, case_index: int, candidates=None) -> dict:
    low, high = float(volume.min()), float(volume.max())
    if high <= low:
        raise ValueError(f"Constant scan: {record['path']}")
    threshold = low + cfg.foreground_threshold * (high - low)
    if candidates is None:
        candidates = np.argwhere(volume > threshold)
    if not len(candidates):
        raise ValueError(f"No foreground candidates: {record['path']}")
    mask_shape = np.asarray(random_mask_shape(cfg, rng))
    block_shape, volume_shape = np.asarray(cfg.block_shape), np.asarray(volume.shape)
    for _ in range(128):
        centre = candidates[int(rng.integers(len(candidates)))]
        jitter_limit = (block_shape - mask_shape) // 2 - cfg.context_margin
        jitter = np.asarray([rng.integers(-j, j + 1) for j in jitter_limit])
        block_start = np.clip(centre - block_shape // 2 + jitter, 0, volume_shape - block_shape)
        mask_start = np.clip(centre - block_start - mask_shape // 2, cfg.context_margin,
                             block_shape - mask_shape - cfg.context_margin)
        global_start = block_start + mask_start
        patch = volume[slices3(global_start, mask_shape)]
        if float((patch > threshold).mean()) >= cfg.minimum_foreground_fraction:
            return {**record, "case_index": int(case_index),
                    "block_start_zyx": block_start.tolist(),
                    "mask_start_zyx": mask_start.tolist(), "mask_shape_zyx": mask_shape.tolist()}
    raise RuntimeError(f"Could not sample a sufficiently foreground-filled hole: {record['path']}. "
                       "Inspect the volume or adjust --minimum-foreground-fraction.")


def fixed_cases(records: list[dict], cfg: Config, split_code: int) -> list[dict]:
    cases = []
    for i, record in enumerate(tqdm(records, desc="Preparing fixed evaluation masks")):
        volume, _ = load_volume(record["path"])
        threshold = volume.min() + cfg.foreground_threshold * (volume.max() - volume.min())
        candidates = np.argwhere(volume > threshold)
        for j in range(cfg.eval_cases_per_volume):
            rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, split_code, i, j]))
            cases.append(sample_case(volume, record, cfg, rng, j, candidates))
    return cases


def observed_bounds(volume: np.ndarray, hole_start, hole_shape) -> tuple[float, float]:
    """Min/max over known voxels only, using six disjoint views around the hole."""
    z, y, x = map(int, hole_start)
    ze, ye, xe = (int(a + b) for a, b in zip(hole_start, hole_shape))
    slabs = (volume[:z], volume[ze:], volume[z:ze, :y], volume[z:ze, ye:],
             volume[z:ze, y:ye, :x], volume[z:ze, y:ye, xe:])
    low, high = math.inf, -math.inf
    for slab in slabs:
        if slab.size:
            if not np.isfinite(slab).all():
                raise ValueError("Observed voxels contain NaN or infinity.")
            low, high = min(low, float(slab.min())), max(high, float(slab.max()))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("Observed voxels must contain a finite, nonconstant intensity range.")
    return low, high


# Create the observed block, keep mask and location information for the model.

def make_inputs(volume: np.ndarray, case: dict, cfg: Config) -> dict[str, torch.Tensor]:
    """No returned model input depends on hidden intensities for this fixed case."""
    block_start = np.asarray(case["block_start_zyx"], dtype=int)
    mask_start = np.asarray(case["mask_start_zyx"], dtype=int)
    mask_shape = np.asarray(case["mask_shape_zyx"], dtype=int)
    global_start = block_start + mask_start
    low, high = observed_bounds(volume, global_start, mask_shape)
    block_raw = volume[slices3(block_start, cfg.block_shape)]
    keep_mask = np.ones(cfg.block_shape, dtype=np.float32)
    keep_mask[slices3(mask_start, mask_shape)] = 0
    # Fill hidden values BEFORE arithmetic: 0 * NaN is still NaN.
    safe_raw = np.where(keep_mask.astype(bool), block_raw, low)
    observed = (safe_raw - low) * (2.0 / (high - low)) - 1.0
    observed[keep_mask == 0] = 0.0
    centre = global_start + (mask_shape - 1) / 2.0
    coords = centre / np.maximum(np.asarray(volume.shape) - 1, 1)
    mask_info = np.concatenate([coords, mask_shape / np.asarray(cfg.block_shape)])
    return {
        "observed": torch.from_numpy(np.ascontiguousarray(observed[None], dtype=np.float32)),
        "keep_mask": torch.from_numpy(keep_mask[None]),
        "coords": torch.tensor(coords, dtype=torch.float32),
        "mask_info": torch.tensor(mask_info, dtype=torch.float32),
        "bounds": torch.tensor([low, high], dtype=torch.float32),
    }


# Provide MRI blocks and masks to the training or evaluation loop.

class MRIPatchDataset(Dataset):
    def __init__(self, records: list[dict], cfg: Config, training: bool):
        self.records, self.cfg, self.training, self.epoch = records, cfg, training, 0

    def __len__(self) -> int:
        return len(self.records) * (self.cfg.train_patches_per_volume if self.training else 1)

    def __getitem__(self, index: int) -> dict:
        if self.training:
            record = self.records[index // self.cfg.train_patches_per_volume]
        else:
            record = self.records[index]
        volume, _ = load_volume(record["path"])
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.cfg.seed, self.epoch, index]))
            case = sample_case(volume, record, self.cfg, rng, index)
        else:
            case = record
        inputs = make_inputs(volume, case, self.cfg)
        low, high = inputs["bounds"].numpy().astype(np.float64)
        raw = volume[slices3(case["block_start_zyx"], self.cfg.block_shape)]
        target = ((raw - low) * (2.0 / (high - low)) - 1.0).astype(np.float32)
        return {**inputs, "target": torch.from_numpy(np.ascontiguousarray(target[None])),
                "record_index": index,
                "mask_start": torch.tensor(case["mask_start_zyx"], dtype=torch.int64),
                "mask_shape": torch.tensor(case["mask_shape_zyx"], dtype=torch.int64)}


def make_loader(dataset: Dataset, cfg: Config, shuffle: bool, device: torch.device,
                epoch: int = 0) -> DataLoader:
    # New workers see dataset.epoch each epoch; no stale persistent-worker state.
    kwargs = {"num_workers": cfg.workers, "pin_memory": device.type == "cuda"}
    if cfg.workers:
        kwargs.update(prefetch_factor=2, persistent_workers=False)
    generator = torch.Generator().manual_seed(cfg.seed + epoch)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False,
                      generator=generator, **kwargs)


def safe_groups(channels: int, maximum: int = 8) -> int:
    groups = min(maximum, channels)
    while channels % groups:
        groups -= 1
    return groups


class RecurrentConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, stride, 1),
                                  nn.GroupNorm(safe_groups(out_channels), out_channels), nn.SiLU())
        self.recurrent = nn.Sequential(nn.Conv3d(out_channels, out_channels, 3, padding=1),
                                       nn.GroupNorm(safe_groups(out_channels), out_channels), nn.SiLU())

    def forward(self, x):
        h = self.conv(x)
        for _ in range(2):
            h = h + self.recurrent(h)
        return h


class RecurrentResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim, downsample=False, dropout=0.1):
        super().__init__()
        stride = 2 if downsample else 1
        self.norm1 = nn.GroupNorm(safe_groups(in_channels), in_channels)
        self.conv1 = RecurrentConv3d(in_channels, out_channels, stride)
        self.condition_mlp = nn.Linear(condition_dim, out_channels)
        self.norm2 = nn.GroupNorm(safe_groups(out_channels), out_channels)
        self.conv2 = RecurrentConv3d(out_channels, out_channels)
        self.dropout = nn.Dropout3d(dropout)
        self.residual = (nn.Conv3d(in_channels, out_channels, 1, stride=stride)
                         if in_channels != out_channels or downsample else nn.Identity())

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.condition_mlp(condition)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.residual(x) + 0.3 * self.dropout(h)


class SimpleResidualBlock(nn.Module):
    """Two-convolution baseline, with the same conditioning interface."""
    def __init__(self, in_channels, out_channels, condition_dim, downsample=False, dropout=0.1):
        super().__init__()
        stride = 2 if downsample else 1
        self.norm1 = nn.GroupNorm(safe_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.condition_mlp = nn.Linear(condition_dim, out_channels)
        self.norm2 = nn.GroupNorm(safe_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout3d(dropout)
        self.residual = (nn.Conv3d(in_channels, out_channels, 1, stride=stride)
                         if in_channels != out_channels or downsample else nn.Identity())

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.condition_mlp(condition)[:, :, None, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.residual(x) + 0.3 * self.dropout(h)


class ProcessedSkipConnection(nn.Module):
    """The three residual skip convolutions from the original script."""
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv3d(channels, channels, 3, padding=1),
                          nn.GroupNorm(safe_groups(channels), channels), nn.SiLU())
            for _ in range(3)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        if channels % heads:
            raise ValueError("Attention channels must be divisible by head count.")
        self.heads, self.head_dim = heads, channels // heads
        self.norm = nn.GroupNorm(safe_groups(channels), channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, 1)
        self.projection = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        batch, channels, depth, height, width = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [item.reshape(batch, self.heads, self.head_dim, -1).transpose(-1, -2)
                   for item in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(-1, -2).reshape(batch, channels, depth, height, width)
        return x + 0.1 * self.projection(out)


class UpSample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1),
                                    nn.GroupNorm(safe_groups(out_channels), out_channels), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


# Define the 3D model that fills the missing region using the observed voxels.

class MaskedAutoencoder3D(nn.Module):
    def __init__(self, base=28, architecture="recurrent", dropout=0.1):
        super().__init__()
        condition_dim = 4 * base
        block = RecurrentResidualBlock if architecture == "recurrent" else SimpleResidualBlock

        def residual(a, b, down=False):
            return block(a, b, condition_dim, downsample=down, dropout=dropout)

        def attention(channels):
            return AttentionBlock(channels) if architecture == "recurrent" else nn.Identity()

        self.spatial_mlp = nn.Sequential(nn.Linear(3, condition_dim), nn.SiLU(),
                                         nn.Linear(condition_dim, condition_dim))
        self.mask_mlp = nn.Sequential(nn.Linear(6, condition_dim), nn.SiLU(),
                                      nn.Linear(condition_dim, condition_dim))
        self.initial = nn.Conv3d(2, base, 3, padding=1)
        self.down1 = residual(base, 2 * base, True)
        self.down2 = residual(2 * base, 4 * base, True)
        self.down2_attention = attention(4 * base)
        self.down3 = residual(4 * base, 8 * base, True)
        self.mid1 = residual(8 * base, 8 * base)
        self.mid_attention1 = attention(8 * base)
        self.mid_attention2 = attention(8 * base)
        self.mid2 = residual(8 * base, 8 * base)
        self.skip1 = ProcessedSkipConnection(2 * base) if architecture == "recurrent" else nn.Identity()
        self.skip2 = ProcessedSkipConnection(4 * base) if architecture == "recurrent" else nn.Identity()
        self.upsample1 = UpSample3D(8 * base, 8 * base)
        self.up1 = residual(12 * base, 4 * base)
        self.up1_attention = attention(4 * base)
        self.upsample2 = UpSample3D(4 * base, 4 * base)
        self.up2 = residual(6 * base, 2 * base)
        self.upsample3 = UpSample3D(2 * base, 2 * base)
        self.up3 = residual(3 * base, base)
        self.final = nn.Conv3d(base, 1, 1)  # Linear image-intensity output, not noise/velocity.

    def forward(self, observed, keep_mask, coords, mask_info):
        if observed.ndim != 5 or observed.shape[1] != 1 or observed.shape != keep_mask.shape:
            raise ValueError("Image and mask must both be [B,1,D,H,W].")
        if any(n % 8 for n in observed.shape[-3:]):
            raise ValueError("Spatial dimensions must be divisible by 8.")
        if not bool(((keep_mask == 0) | (keep_mask == 1)).all()):
            raise ValueError("The keep mask must be binary; soft masks are not supported.")
        # Defensive masking also protects callers who accidentally pass hidden values.
        observed = torch.where(keep_mask.bool(), observed, torch.zeros_like(observed))
        condition = self.spatial_mlp(coords) + self.mask_mlp(mask_info)
        x1 = self.initial(torch.cat([observed, keep_mask], dim=1))
        x2 = self.down1(x1, condition)
        x3 = self.down2_attention(self.down2(x2, condition))
        x4 = self.down3(x3, condition)
        middle = self.mid1(x4, condition)
        middle = self.mid_attention2(self.mid_attention1(middle))
        middle = self.mid2(middle, condition)
        x = self.up1(torch.cat([self.upsample1(middle), self.skip2(x3)], dim=1), condition)
        x = self.up1_attention(x)
        x = self.up2(torch.cat([self.upsample2(x), self.skip1(x2)], dim=1), condition)
        x = self.up3(torch.cat([self.upsample3(x), x1], dim=1), condition)
        return self.final(x)


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        source_state = model.state_dict()
        for name, value in self.model.state_dict().items():
            source = source_state[name]
            if value.is_floating_point():
                value.lerp_(source, 1.0 - self.decay)
            else:
                value.copy_(source)


def masked_l1_per_sample(prediction, target, keep_mask):
    missing = 1.0 - keep_mask.float()
    counts = missing.flatten(1).sum(1)
    if bool((counts <= 0).any()):
        raise ValueError("Every training sample must contain a missing region.")
    # torch.where also excludes any irrelevant NaN at observed target positions.
    error = torch.where(missing.bool(), (prediction.float() - target.float()).abs(), 0.0)
    return error.flatten(1).sum(1) / counts


@torch.no_grad()
def reconstruct_block(model, observed, keep_mask, coords, mask_info):
    """Inference accepts observed data only. There is no ground-truth argument."""
    model.eval()
    prediction = model(observed, keep_mask, coords, mask_info)
    return torch.where(keep_mask.bool(), observed, prediction)


# Calculate reconstruction metrics for each masked region.

class ReconstructionMetrics:
    def __init__(self):
        self.ssim_by_window = {}

    @torch.no_grad()
    def per_case(self, target, reconstruction, keep_mask, mask_starts, mask_shapes):
        target_01, reconstruction_01 = (target.float() + 1) / 2, (reconstruction.float() + 1) / 2
        output = []
        for i in range(target.shape[0]):
            missing = keep_mask[i] == 0
            error = reconstruction_01[i][missing] - target_01[i][missing]
            if not error.numel():
                raise ValueError("Cannot evaluate an empty hole.")
            mse, mae = error.square().mean(), error.abs().mean()
            shape = tuple(int(n) for n in mask_shapes[i])
            region = slices3(mask_starts[i], shape)
            truth_roi = target_01[(slice(i, i+1), slice(None), *region)]
            recon_roi = reconstruction_01[(slice(i, i+1), slice(None), *region)]
            hole_roi = keep_mask[(slice(i, i+1), slice(None), *region)]
            if not bool((hole_roi == 0).all()) or int(missing.sum()) != math.prod(shape):
                raise ValueError("SSIM ROI must match exactly one solid missing cuboid.")
            window = min(11, min(shape))
            window -= int(window % 2 == 0)
            if window < 3:
                raise ValueError("SSIM needs a missing region at least 3 voxels wide.")
            if window not in self.ssim_by_window:
                self.ssim_by_window[window] = SSIMMetric(spatial_dims=3, data_range=1.0, win_size=window)
            metric = self.ssim_by_window[window]
            # This API accumulates values; reset so evaluation cannot grow its buffer.
            similarity = float(metric(y_pred=recon_roi, y=truth_roi).item())
            metric.reset()
            values = {"masked_mae_01": float(mae), "masked_mse_01": float(mse),
                      "masked_psnr_db": float(-10 * torch.log10(mse.clamp_min(1e-10))),
                      "hole_ssim_3d": similarity}
            if not all(math.isfinite(v) for v in values.values()):
                raise FloatingPointError("Nonfinite reconstruction metric.")
            output.append(values)
        return output


def summarise_metrics(rows: list[dict]) -> tuple[dict, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject"]].append(row)
    if not grouped:
        raise ValueError("No cases were evaluated.")
    subjects = [{"subject": subject, "cases": len(cases),
                 **{name: float(np.mean([case[name] for case in cases])) for name in METRIC_NAMES}}
                for subject, cases in sorted(grouped.items())]
    summary = {"case_count": len(rows), "subject_count": len(subjects),
               "aggregation": "Mean within subject, then equal-weight mean across subjects",
               "psnr_zero_mse_ceiling_db": 100.0}
    for name in METRIC_NAMES:
        values = [row[name] for row in subjects]
        summary[name] = {"mean": float(np.mean(values)),
                         "std_between_subjects": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    return summary, subjects


def save_nifti_like(array_xyz: np.ndarray, reference, path: Path, start_xyz=(0, 0, 0),
                    binary: bool = False) -> None:
    """Preserve orientation/spacing; translate the affine when exporting a crop."""
    translation = np.eye(4)
    translation[:3, 3] = start_xyz
    affine = reference.affine @ translation
    header = reference.header.copy()
    header.set_data_dtype(np.uint8 if binary else np.float32)
    header.set_slope_inter(1.0, 0.0)
    header.set_intent("none")
    data = np.asarray(array_xyz, dtype=np.uint8 if binary else np.float32)
    image = nib.Nifti1Image(data, affine, header=header)
    image.set_sform(affine, code=int(reference.header["sform_code"]) or 2)
    qform, qcode = reference.get_qform(coded=True)
    if qcode:
        image.set_qform(qform @ translation, code=int(qcode))
    else:
        image.set_qform(None, code=0)
    image.header.set_slope_inter(1.0, 0.0)
    image.header["cal_min"], image.header["cal_max"] = float(data.min()), float(data.max())
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def save_case_example(case: dict, target, reconstruction, keep_mask, bounds,
                      folder: Path, index: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = nib.load(case["path"])
    target = target.detach().float().cpu().numpy()[0]
    reconstruction = reconstruction.detach().float().cpu().numpy()[0]
    mask = keep_mask.detach().float().cpu().numpy()[0]
    low, high = map(float, bounds)
    original_xyz = reference.get_fdata(dtype=np.float32)
    start_xyz = tuple(reversed(case["block_start_zyx"]))
    shape_xyz = tuple(reversed(target.shape))
    raw_target = original_xyz[slices3(start_xyz, shape_xyz)]
    pred_raw_xyz = (((reconstruction + 1) / 2) * (high - low) + low).transpose(2, 1, 0)
    mask_xyz = mask.transpose(2, 1, 0)
    raw_reconstruction = np.where(mask_xyz.astype(bool), raw_target, pred_raw_xyz)
    error_01 = np.where(mask == 0, np.abs(reconstruction - target) / 2, 0)
    prefix = f"case_{index:04d}_{case['subject']}"
    arrays = {"target": raw_target, "observed": np.where(mask_xyz, raw_target, 0),
              "reconstruction": raw_reconstruction, "keep_mask": mask_xyz,
              "error_01": error_01.transpose(2, 1, 0)}
    for label, data in arrays.items():
        save_nifti_like(data, reference, folder / f"{prefix}_{label}.nii.gz", start_xyz,
                        binary=label == "keep_mask")
    centre = np.asarray(case["mask_start_zyx"]) + np.asarray(case["mask_shape_zyx"]) // 2
    observed_01 = np.where(mask, (target + 1) / 2, 0)
    views = [(target + 1) / 2, observed_01, (reconstruction + 1) / 2, error_01]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    error_max = max(float(error_01.max()), 1e-6)
    for axis in range(3):
        for col, volume in enumerate(views):
            axes[axis, col].imshow(np.take(volume, int(centre[axis]), axis=axis), origin="lower",
                                   cmap="magma" if col == 3 else "gray", vmin=0,
                                   vmax=error_max if col == 3 else 1)
            axes[axis, col].set_title(f"{('Z', 'Y', 'X')[axis]}: {('Target', 'Observed', 'Reconstruction', 'Absolute error')[col]}")
            axes[axis, col].axis("off")
    fig.suptitle(f"{case['subject']} | error colour range 0–{error_max:.4f} (normalised units)")
    fig.tight_layout()
    fig.savefig(folder / f"{prefix}.png", dpi=150)
    plt.close(fig)


def input_tensors(batch: dict, device: torch.device) -> dict:
    return {name: batch[name].to(device, non_blocking=True)
            for name in ("observed", "keep_mask", "coords", "mask_info")}


class Precision:
    def __init__(self, cfg: Config, device: torch.device):
        self.enabled = cfg.amp and device.type == "cuda"
        self.dtype = torch.bfloat16 if self.enabled and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled and self.dtype == torch.float16)

    def context(self):
        return torch.autocast("cuda", dtype=self.dtype) if self.enabled else nullcontext()


# Train the model for one pass through the training data.

def train_epoch(model, ema, loader, optimizer, precision, device, epoch):
    model.train()
    total_mae, count = 0.0, 0
    progress = tqdm(loader, desc=f"Epoch {epoch + 1}: train")
    for batch in progress:
        inputs = input_tensors(batch, device)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with precision.context():
            prediction = model(**inputs)
        # Loss is reduced in float32, separately for each sample.
        per_sample = masked_l1_per_sample(prediction, target, inputs["keep_mask"])
        loss = per_sample.mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss.")
        scaler = precision.scaler
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                 error_if_nonfinite=not scaler.is_enabled())
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            ema.update(model)
        total_mae += float(per_sample.detach().sum()) / 2.0
        count += len(target)
        progress.set_postfix(masked_mae_01=f"{total_mae / count:.5f}")
    return total_mae / count


# Evaluate saved cases and calculate validation or test metrics.

@torch.no_grad()
def evaluate(model, cases: list[dict], cfg: Config, device: torch.device,
             export_folder: Path | None = None, export_count: int = 0):
    model.eval()
    dataset = MRIPatchDataset(cases, cfg, training=False)
    loader = make_loader(dataset, cfg, shuffle=False, device=device)
    metrics, rows = ReconstructionMetrics(), []
    # Evaluation uses float32, fixed inputs and the exact same one-pass inference.
    for batch in tqdm(loader, desc="Reconstruction evaluation"):
        inputs = input_tensors(batch, device)
        reconstruction = reconstruct_block(model, **inputs)
        target = batch["target"].to(device, non_blocking=True)
        values = metrics.per_case(target, reconstruction, inputs["keep_mask"],
                                  batch["mask_start"], batch["mask_shape"])
        for i, result in enumerate(values):
            index = int(batch["record_index"][i])
            case = cases[index]
            rows.append({"subject": case["subject"], "path": case["path"],
                         "case_index": case["case_index"],
                         "block_start_zyx": ",".join(map(str, case["block_start_zyx"])),
                         "mask_start_zyx": ",".join(map(str, case["mask_start_zyx"])),
                         "mask_shape_zyx": ",".join(map(str, case["mask_shape_zyx"])), **result})
            if export_folder is not None and index < export_count:
                save_case_example(case, target[i], reconstruction[i], inputs["keep_mask"][i],
                                  batch["bounds"][i], export_folder, index)
    if len(rows) != len(cases):
        raise RuntimeError("Evaluation did not process every case exactly once.")
    summary, subjects = summarise_metrics(rows)
    return summary, rows, subjects


def save_evaluation(output: Path, prefix: str, evaluation) -> None:
    summary, rows, subjects = evaluation
    write_json(output / f"{prefix}_summary.json", summary)
    write_csv(output / f"{prefix}_cases.csv", rows)
    write_csv(output / f"{prefix}_subjects.csv", subjects)


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def model_from_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("This is not a checkpoint from this masked autoencoder script.")
    cfg = Config(**checkpoint["config"])
    cfg.validate()
    model = MaskedAutoencoder3D(cfg.base_channels, cfg.architecture, cfg.dropout).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, cfg, checkpoint


def plot_history(history: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [row["train_masked_mae_01"] for row in history], label="Train (raw model)")
    axes[0].plot(epochs, [row["val_masked_mae_01"] for row in history], label="Validation (EMA)")
    axes[0].set_ylabel("Masked MAE, normalised units")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_masked_psnr_db"] for row in history])
    axes[1].set_ylabel("Validation masked PSNR (dB)")
    axes[2].plot(epochs, [row["val_hole_ssim_3d"] for row in history])
    axes[2].set_ylabel("Validation 3D SSIM inside hole")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "learning_curves.png", dpi=160)
    plt.close(fig)


# Train the model, select a checkpoint using validation results and test it.

def run_training(cfg: Config) -> None:
    cfg.validate()
    output = Path(cfg.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}. Choose a new --output directory.")
    cfg.output, cfg.data_dir = str(output), str(Path(cfg.data_dir).expanduser().resolve())
    # Validate source data before creating output files.
    splits, geometry = discover_and_split(cfg)
    val_cases, test_cases = fixed_cases(splits["val"], cfg, 101), fixed_cases(splits["test"], cfg, 202)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "splits.json", {"seed": cfg.seed, "geometry": geometry, **splits})
    write_json(output / "val_cases.json", val_cases)
    write_json(output / "test_cases.json", test_cases)
    device = select_device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    seed_everything(cfg.seed)
    print(f"Device: {device}; architecture: {cfg.architecture}")
    for split, records in splits.items():
        print(f"{split}: {len(records)} scans / {len({r['subject'] for r in records})} subjects")
    model = MaskedAutoencoder3D(cfg.base_channels, cfg.architecture, cfg.dropout).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ema = EMA(model, cfg.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                                          eta_min=min(1e-6, cfg.learning_rate))
    precision = Precision(cfg, device)
    training = MRIPatchDataset(splits["train"], cfg, training=True)
    history, best_score, stale_epochs = [], math.inf, 0

    for epoch in range(cfg.epochs):
        started = time.monotonic()
        training.epoch = epoch
        loader = make_loader(training, cfg, shuffle=True, device=device, epoch=epoch)
        lr = optimizer.param_groups[0]["lr"]
        train_mae = train_epoch(model, ema, loader, optimizer, precision, device, epoch)
        evaluation = evaluate(ema.model, val_cases, cfg, device)
        validation = evaluation[0]
        score = validation["masked_mae_01"]["mean"]
        row = {"epoch": epoch + 1, "learning_rate": lr, "train_masked_mae_01": train_mae,
               "val_masked_mae_01": score,
               "val_masked_psnr_db": validation["masked_psnr_db"]["mean"],
               "val_hole_ssim_3d": validation["hole_ssim_3d"]["mean"],
               "seconds": time.monotonic() - started}
        history.append(row)
        write_csv(output / "history.csv", history)
        scheduler.step()
        payload = {"format_version": 1, "config": asdict(cfg), "geometry": geometry,
                   "epoch": epoch + 1, "model_state": ema.model.state_dict(),
                   "validation": validation, "torch_version": str(torch.__version__)}
        if score < best_score:
            best_score, stale_epochs = score, 0
            save_checkpoint(output / "best.pt", payload)
            save_evaluation(output, "best_validation", evaluation)
        else:
            stale_epochs += 1
        stop = cfg.patience > 0 and stale_epochs >= cfg.patience
        if (epoch + 1) % cfg.checkpoint_every == 0 or epoch + 1 == cfg.epochs or stop:
            save_checkpoint(output / "last.pt", {
                **payload, "raw_model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(), "scaler_state": precision.scaler.state_dict(),
                "best_score": best_score, "history": history,
            })
            plot_history(history, output)
        print(f"Epoch {epoch+1}: train MAE={train_mae:.5f}, val MAE={score:.5f}, "
              f"val PSNR={row['val_masked_psnr_db']:.2f} dB, SSIM={row['val_hole_ssim_3d']:.4f}", flush=True)
        if stop:
            print(f"Early stopping after {stale_epochs} epochs without validation improvement.")
            break

    # Select by validation only. Test cases have never influenced the optimizer/checkpoint selection.
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    print(f"Evaluating best validation checkpoint (epoch {checkpoint['epoch']}) on test subjects.")
    test = evaluate(model, test_cases, cfg, device, output / "test_outputs", cfg.save_examples)
    save_evaluation(output, "test", test)
    plot_history(history, output)
    print(json.dumps(test[0], indent=2))
    print(f"Run saved to {output}")


# Test the best checkpoint using the cases saved with the training run.

def run_saved_test(run: Path, device_name: str, workers: int | None, save_examples: int | None):
    device = select_device(device_name)
    model, cfg, checkpoint = model_from_checkpoint(run / "best.pt", device)
    if workers is not None:
        cfg.workers = workers
    if save_examples is not None:
        cfg.save_examples = save_examples
    cfg.validate()
    cases = read_json(run / "test_cases.json")
    for path in {case["path"] for case in cases}:
        check_geometry(nib.load(path), checkpoint["geometry"])
    evaluation = evaluate(model, cases, cfg, device, run / "test_outputs", cfg.save_examples)
    save_evaluation(run, "test", evaluation)
    print(json.dumps(evaluation[0], indent=2))


def case_from_keep_mask(keep_mask: np.ndarray, cfg: Config) -> dict:
    if keep_mask.ndim != 3 or not np.isfinite(keep_mask).all():
        raise ValueError("Expected a finite 3D keep mask.")
    if not np.all((keep_mask == 0) | (keep_mask == 1)):
        raise ValueError("Keep mask must be binary: 1 observed / 0 missing.")
    missing = np.argwhere(keep_mask == 0)
    if not len(missing):
        raise ValueError("The keep mask has no missing voxels.")
    start, end = missing.min(0), missing.max(0) + 1
    shape = end - start
    if int(np.prod(shape)) != len(missing):
        raise ValueError("This model's inference command supports one solid cuboid hole.")
    block_shape = np.asarray(cfg.block_shape)
    if np.any(np.asarray(keep_mask.shape) < block_shape):
        raise ValueError("Volume is smaller than the configured block.")
    centre = (start + end - 1) // 2
    block_start = np.clip(centre - block_shape // 2, 0, np.asarray(keep_mask.shape) - block_shape)
    mask_start = start - block_start
    if np.any(mask_start < cfg.context_margin) or np.any(
            block_shape - mask_start - shape < cfg.context_margin):
        raise ValueError("The missing section must fit inside one block with the configured context margin.")
    return {"case_index": 0, "subject": "inference", "block_start_zyx": block_start.tolist(),
            "mask_start_zyx": mask_start.tolist(), "mask_shape_zyx": shape.tolist()}


# Use a checkpoint to fill a mask in one input MRI volume.

def run_inference(checkpoint_path: Path, input_path: Path, mask_path: Path,
                  output_path: Path, device_name: str):
    if output_path.resolve() in {input_path.resolve(), mask_path.resolve(), checkpoint_path.resolve()}:
        raise ValueError("Choose a separate output path; input files must not be overwritten.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")
    device = select_device(device_name)
    model, cfg, checkpoint = model_from_checkpoint(checkpoint_path, device)
    volume, reference = load_volume(str(input_path), allow_nonfinite=True)
    check_geometry(reference, checkpoint["geometry"])
    mask_volume, mask_image = load_volume(str(mask_path))
    check_geometry(mask_image, checkpoint["geometry"])
    case = {**case_from_keep_mask(mask_volume, cfg), "path": str(input_path)}
    features = make_inputs(volume, case, cfg)
    inputs = {key: value.unsqueeze(0).to(device) for key, value in features.items()
              if key in {"observed", "keep_mask", "coords", "mask_info"}}
    reconstruction = reconstruct_block(model, **inputs)[0, 0].cpu().numpy()
    low, high = map(float, features["bounds"])
    reconstruction_raw = (reconstruction + 1) / 2 * (high - low) + low
    if not np.isfinite(reconstruction_raw).all():
        raise FloatingPointError("Nonfinite prediction; output was not saved.")
    completed = volume.copy()
    global_start = np.asarray(case["block_start_zyx"]) + np.asarray(case["mask_start_zyx"])
    completed[slices3(global_start, case["mask_shape_zyx"])] = reconstruction_raw[
        slices3(case["mask_start_zyx"], case["mask_shape_zyx"])]
    # Only the unknown cuboid was changed; original observed intensities are copied exactly.
    assert np.array_equal(completed[mask_volume == 1], volume[mask_volume == 1])
    save_nifti_like(completed.transpose(2, 1, 0), reference, output_path)
    write_json(Path(str(output_path) + ".json"), {
        "checkpoint": str(checkpoint_path.resolve()), "checkpoint_epoch": checkpoint["epoch"],
        "input": str(input_path.resolve()), "keep_mask": str(mask_path.resolve()),
        "case": case, "observed_intensity_min": low, "observed_intensity_max": high,
        "training_mask_shape_zyx": list(cfg.mask_shape), "trained_with_variable_masks": cfg.variable_masks,
    })
    print(f"Saved reconstruction: {output_path}")


# Run small synthetic checks without needing the MRI dataset.

def smoke_test() -> None:
    """Focused CPU checks; no MRI dataset or trained checkpoint required."""
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    seed_everything(7)
    cfg = Config(block_shape=(16, 16, 16), mask_shape=(5, 5, 5),
                 mask_min=(3, 3, 3), mask_max=(7, 7, 7), base_channels=4,
                 batch_size=2, workers=0, dropout=0, amp=False)
    cfg.validate()
    rng = np.random.default_rng(7)
    volume = rng.uniform(0, 100, (32, 40, 48)).astype(np.float32)
    case = {"subject": "synthetic", "path": "unused", "case_index": 0,
            "block_start_zyx": [4, 6, 8], "mask_start_zyx": [5, 4, 6],
            "mask_shape_zyx": [5, 5, 5]}
    features = make_inputs(volume, case, cfg)
    hidden = slices3(np.asarray(case["block_start_zyx"]) + case["mask_start_zyx"], case["mask_shape_zyx"])
    changed = volume.copy()
    changed[hidden] = np.nan
    other = make_inputs(changed, case, cfg)
    for key in features:
        assert torch.equal(features[key], other[key]), f"Hidden content influenced {key}."
    assert features["observed"].shape == (1, 16, 16, 16)
    assert set(features["keep_mask"].unique().tolist()) == {0.0, 1.0}
    assert int((features["keep_mask"] == 0).sum()) == 125
    print("PASS: binary mask, tensor shape, and hidden-target-independent normalisation.")

    inputs = {key: features[key][None] for key in ("observed", "keep_mask", "coords", "mask_info")}
    for architecture in ("recurrent", "simple"):
        model = MaskedAutoencoder3D(4, architecture, dropout=0).eval()
        reconstructed = reconstruct_block(model, **inputs)
        contaminated = dict(inputs)
        contaminated["observed"] = inputs["observed"].clone()
        contaminated["observed"][inputs["keep_mask"] == 0] = float("nan")
        repeat = reconstruct_block(model, **contaminated)
        torch.testing.assert_close(reconstructed, repeat, rtol=0, atol=0)
        assert torch.equal(reconstructed[inputs["keep_mask"] == 1],
                           inputs["observed"][inputs["keep_mask"] == 1])
        low, high = map(float, features["bounds"])
        target_np = (volume[slices3(case["block_start_zyx"], cfg.block_shape)] - low) * (2 / (high-low)) - 1
        target = torch.from_numpy(target_np[None, None].astype(np.float32))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.final.weight.detach().clone()
        model.train()
        loss = masked_l1_per_sample(model(**inputs), target, inputs["keep_mask"]).mean()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert not torch.equal(before, model.final.weight)
        print(f"PASS: {architecture} forward/backward, one-pass inference, observed-voxel preservation, leakage check.")

    masks = torch.ones(2, 1, 16, 16, 16)
    starts, shapes = [(3, 3, 3), (4, 5, 6)], [(5, 5, 5), (6, 4, 5)]
    for i in range(2):
        masks[(i, 0, *slices3(starts[i], shapes[i]))] = 0
    target, prediction = torch.zeros_like(masks), torch.zeros_like(masks)
    prediction[0][masks[0] == 0] = 1
    prediction[1][masks[1] == 0] = 3
    torch.testing.assert_close(masked_l1_per_sample(prediction, target, masks), torch.tensor([1., 3.]))
    prediction.zero_()
    prediction[1][masks[1] == 0] = 0.4
    values = ReconstructionMetrics().per_case(target, prediction, masks, starts, shapes)
    assert values[0]["masked_psnr_db"] == 100.0
    assert abs(values[0]["hole_ssim_3d"] - 1.0) < 1e-6
    assert abs(values[1]["masked_mse_01"] - 0.04) < 1e-6
    assert abs(values[1]["masked_psnr_db"] - (-10 * math.log10(0.04))) < 1e-4
    prediction[masks == 1] = 99
    unchanged = ReconstructionMetrics().per_case(target, prediction, masks, starts, shapes)
    for first, second in zip(values, unchanged):
        assert first == second
    print("PASS: per-sample losses/metrics, even-sized masks, hole-only SSIM, no exterior contribution.")

    cfg.variable_masks, cfg.mask_min, cfg.mask_max = True, (3, 4, 5), (7, 6, 7)
    sampled = np.asarray([random_mask_shape(cfg, rng) for _ in range(100)])
    assert np.all(sampled >= cfg.mask_min) and np.all(sampled <= cfg.mask_max)
    assert np.array_equal(sampled.min(0), cfg.mask_min) and np.array_equal(sampled.max(0), cfg.mask_max)
    print("PASS: variable-mask limits are respected, including inclusive bounds.")

    with tempfile.TemporaryDirectory(prefix="mri_ae_smoke_") as folder:
        folder = Path(folder)
        affine = np.array([[0, -1.2, 0, 10], [1.3, 0, 0, 20], [0, 0, 2, 30], [0, 0, 0, 1.]])
        reference = nib.Nifti1Image(volume.transpose(2, 1, 0), affine)
        reference.set_qform(affine, code=1)
        start_xyz = tuple(reversed(case["block_start_zyx"]))
        mask_xyz = features["keep_mask"][0].numpy().transpose(2, 1, 0)
        save_nifti_like(mask_xyz, reference, folder / "mask.nii.gz", start_xyz, binary=True)
        saved = nib.load(folder / "mask.nii.gz")
        assert np.array_equal(saved.get_fdata(), mask_xyz)
        np.testing.assert_allclose(saved.affine @ np.array([0, 0, 0, 1]),
                                   reference.affine @ np.array([*start_xyz, 1]), atol=1e-5)
        errors = np.zeros((16, 16, 16), dtype=np.float32)
        save_nifti_like(errors, reference, folder / "error.nii.gz", start_xyz)
        assert np.count_nonzero(nib.load(folder / "error.nii.gz").get_fdata()) == 0
        checkpoint = {"format_version": 1, "config": asdict(cfg), "geometry": {
            "shape_xyz": list(reference.shape), "affine": affine.tolist()}, "epoch": 0,
            "model_state": MaskedAutoencoder3D(cfg.base_channels, cfg.architecture, cfg.dropout).state_dict()}
        save_checkpoint(folder / "checkpoint.pt", checkpoint)
        restored, _, _ = model_from_checkpoint(folder / "checkpoint.pt", torch.device("cpu"))
        assert not restored.training
    print("PASS: NIfTI orientation/offset, binary and error values, and checkpoint reload.")
    print("All synthetic smoke checks passed. These checks do not establish MRI reconstruction accuracy.")


# Set up the command line options for training, testing and inference.

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="3D masked MRI autoencoder; see the source docstring for examples.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("smoke-test", help="Run synthetic CPU correctness checks.")
    train = commands.add_parser("train", help="Train from scratch, select on validation, evaluate held-out test subjects.")
    defaults = Config()
    train.add_argument("--data-dir", default=defaults.data_dir, help="Directory of aligned 3D T1 scans.")
    train.add_argument("--output", default=defaults.output, help="New, empty run directory.")
    train.add_argument("--architecture", choices=("recurrent", "simple"), default=defaults.architecture)
    for name in ("block_shape", "mask_shape", "mask_min", "mask_max"):
        train.add_argument("--" + name.replace("_", "-"), nargs=3, type=int,
                           default=getattr(defaults, name), metavar=("D", "H", "W"))
    train.add_argument("--variable-masks", action="store_true", help="Sample each dimension inclusively between mask-min/max.")
    for name in ("context_margin", "base_channels", "epochs", "batch_size", "workers", "seed",
                 "train_patches_per_volume", "eval_cases_per_volume", "save_examples", "patience", "checkpoint_every"):
        train.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "weight_decay", "ema_decay", "dropout", "foreground_threshold", "minimum_foreground_fraction"):
        train.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    train.add_argument("--subject-regex", default=defaults.subject_regex,
                       help="First capture group (or whole match) identifies one subject across all their scans.")
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    train.add_argument("--no-amp", dest="amp", action="store_false", default=True)

    test = commands.add_parser("test", help="Re-evaluate best.pt on the run's saved test masks.")
    test.add_argument("--run", type=Path, default=Path(defaults.output))
    test.add_argument("--device", default="auto")
    test.add_argument("--workers", type=int, default=None)
    test.add_argument("--save-examples", type=int, default=None)

    infer = commands.add_parser("infer", help="Fill one cuboid in a real incomplete aligned scan.")
    infer.add_argument("--checkpoint", type=Path, required=True)
    infer.add_argument("--input", type=Path, required=True)
    infer.add_argument("--mask", type=Path, required=True, help="Full-volume binary KEEP mask: 1 known / 0 missing.")
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--device", default="auto")
    return parser


def automatic_training_config() -> Config:
    """Use editable defaults and choose an unused run folder when necessary."""
    cfg = Config()
    base = Path(cfg.output).expanduser()
    candidate, number = base, 2
    while candidate.exists() and (not candidate.is_dir() or any(candidate.iterdir())):
        candidate = base.with_name(f"{base.name}_{number:03d}")
        number += 1
    cfg.output = str(candidate)
    return cfg


# Choose automatic training or run the command selected by the user.

def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else list(argv)
    if not arguments:
        cfg = automatic_training_config()
        print("Automatic mode: train, validate, then test the best checkpoint.", flush=True)
        print(f"Output directory: {Path(cfg.output).resolve()}", flush=True)
        # run_training already reloads best.pt and evaluates the held-out test set.
        run_training(cfg)
        return

    args = build_parser().parse_args(arguments)
    if args.command == "smoke-test":
        smoke_test()
    elif args.command == "train":
        run_training(Config(**{k: v for k, v in vars(args).items() if k != "command"}))
    elif args.command == "test":
        run_saved_test(args.run, args.device, args.workers, args.save_examples)
    elif args.command == "infer":
        run_inference(args.checkpoint, args.input, args.mask, args.output, args.device)


if __name__ == "__main__":
    main()
