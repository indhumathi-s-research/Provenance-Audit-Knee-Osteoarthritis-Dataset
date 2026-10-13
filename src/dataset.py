"""PyTorch Dataset + transform pipeline driven by experiment config."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocessing import apply_clahe, crop_roi, read_image_rgb, resize_cv, to_tensor_normalized

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class KneeImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: dict, train: bool = False, cache_dir=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.train = train
        self.size = tuple(cfg["preprocessing"]["image_size"])
        self.contrast = cfg["preprocessing"].get("contrast", "none")
        self.clahe_clip = cfg["preprocessing"].get("clahe_clip", 2.0)
        self.clahe_grid = tuple(cfg["preprocessing"].get("clahe_grid", [8, 8]))
        self.roi = cfg["roi"]["enabled"]
        self.roi_margin_h = cfg["roi"].get("margin_h", 0.18)
        self.roi_margin_w = cfg["roi"].get("margin_w", 0.22)
        self.augment = cfg["training"].get("augmentation", "none")
        self.normalize = cfg["preprocessing"].get("normalize", "raw")
        if self.normalize == "imagenet":
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        else:
            self.mean, self.std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
        self._cache = None
        if cache_dir:
            self._cache = self._load_cache(cache_dir)

    def _cache_key(self) -> str:
        img_fp = hashlib.md5("|".join(self.df["image_path"].tolist()).encode("utf-8")).hexdigest()[:10]
        return (
            f"{self.size[0]}x{self.size[1]}_{self.contrast}_"
            f"{self.clahe_clip}_{'_'.join(map(str, self.clahe_grid))}_"
            f"roi{int(self.roi)}_{self.roi_margin_h}_{self.roi_margin_w}_{img_fp}"
        )

    def _load_cache(self, cache_dir: str) -> np.ndarray | None:
        """Memmapped array of static (pre-augmentation) images, decoded once."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"cache_{self._cache_key()}.npy"
        if path.exists():
            return np.load(path, mmap_mode="r")
        t0 = time.time()
        h, w = self.size
        arr = np.zeros((len(self.df), h, w, 3), dtype=np.uint8)
        for i in range(len(self.df)):
            img = read_image_rgb(self.df.iloc[i]["image_path"])
            img = self._static(img)
            arr[i] = img
        np.save(path, arr)
        print(f"[cache] decoded {len(self.df)} images in {time.time()-t0:.1f}s -> {path}", flush=True)
        return np.load(path, mmap_mode="r")

    def _static(self, img: np.ndarray) -> np.ndarray:
        if self.contrast == "clahe":
            img = apply_clahe(img, self.clahe_clip, self.clahe_grid)
        if self.roi:
            img = crop_roi(img, self.roi_margin_h, self.roi_margin_w)
        img = resize_cv(img, self.size)
        return np.ascontiguousarray(img)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
        angle = np.random.uniform(-6.0, 6.0)
        h, w = img.shape[:2]
        m = cv2_getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2_warpAffine(img, m, (w, h))
        return img

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self._cache is not None:
            img = np.ascontiguousarray(self._cache[idx])
        else:
            img = read_image_rgb(row["image_path"])
            img = self._static(img)
        if self.train and self.augment == "light":
            img = self._augment(img)
        x = to_tensor_normalized(img, self.mean, self.std)
        y = int(row["kl_label"])
        return x, y, row["image_path"], row["patient_id"], row.get("knee_id", "")


def cv2_getRotationMatrix2D(center, angle, scale):
    import cv2
    return cv2.getRotationMatrix2D(center, angle, scale)


def cv2_warpAffine(img, m, size):
    import cv2
    return cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def class_weights_from_df(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights computed from a training-only frame."""
    counts = np.bincount(df["kl_label"].astype(int).to_numpy(), minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = total / (num_classes * np.maximum(counts, 1.0))
    return torch.as_tensor(weights, dtype=torch.float32)
