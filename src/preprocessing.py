"""Preprocessing: grayscale handling, CLAHE contrast, ROI crop, resize, normalize."""
from __future__ import annotations

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def apply_clahe(img: np.ndarray, clip: float = 2.0, grid: tuple = (8, 8)) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    out = clahe.apply(gray.astype(np.uint8))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB).astype(np.float32)


def crop_roi(img: np.ndarray, margin_h: float = 0.18, margin_w: float = 0.22) -> np.ndarray:
    h, w = img.shape[:2]
    top, bot = int(h * margin_h), int(h * (1 - margin_h))
    left, right = int(w * margin_w), int(w * (1 - margin_w))
    return img[top:bot, left:right]


def resize_cv(img: np.ndarray, size: tuple) -> np.ndarray:
    return cv2.resize(img, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)


def to_tensor_normalized(img: np.ndarray, mean, std, scale: float = 255.0) -> torch.Tensor:
    img = img.astype(np.float32) / scale
    img = (img - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
    return t
