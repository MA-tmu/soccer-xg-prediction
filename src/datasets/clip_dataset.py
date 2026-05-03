# src/datasets/clip_dataset.py

import csv
import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.transforms import (
    flow_to_tensor,
    resize_flow,
    resize_rgb,
    rgb_to_tensor_imagenet,
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _list_image_files(clip_dir: str) -> List[str]:
    names = [n for n in os.listdir(clip_dir) if n.lower().endswith(IMAGE_EXTS)]
    names.sort()
    return [os.path.join(clip_dir, n) for n in names]


def _read_rgb_image(path: str) -> np.ndarray:
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _load_rgb_clip(clip_dir: str, frames_rgb: int, img_size: int) -> torch.Tensor:
    files = _list_image_files(clip_dir)
    if len(files) == 0:
        raise RuntimeError(f"No images found in clip_dir: {clip_dir}")

    picked = files[:frames_rgb]
    if len(picked) < frames_rgb:
        picked = picked + [picked[-1]] * (frames_rgb - len(picked))

    frames = []
    for fp in picked:
        img = _read_rgb_image(fp)
        img = resize_rgb(img, img_size)
        frames.append(rgb_to_tensor_imagenet(img))

    return torch.stack(frames, dim=0)  # (T,3,H,W)


def _ensure_flow_T_HW_2(arr: np.ndarray) -> np.ndarray:
    """
    Accept:
      (T,H,W,2) or (T,2,H,W)
    Return:
      (T,H,W,2) float32
    """
    if arr.ndim != 4:
        raise RuntimeError(f"Unexpected flow ndim={arr.ndim}, shape={arr.shape}")

    if arr.shape[-1] == 2:
        out = arr
    elif arr.shape[1] == 2:
        out = np.transpose(arr, (0, 2, 3, 1))
    else:
        raise RuntimeError(f"Unexpected flow shape={arr.shape}, expected (...,2) or (T,2,H,W)")

    return out.astype(np.float32)


def _load_flow_clip(clip_dir: str, flow_file: str, flows_per_clip: int, img_size: int) -> torch.Tensor:
    path = os.path.join(clip_dir, flow_file)
    if not os.path.isfile(path):
        raise RuntimeError(f"Flow file not found: {path}")

    arr = np.load(path)
    arr = _ensure_flow_T_HW_2(arr)  # (T,H,W,2)
    if arr.shape[0] == 0:
        raise RuntimeError(f"Empty flow array in: {path}")

    picked = arr[:flows_per_clip]
    if picked.shape[0] < flows_per_clip:
        last = picked[-1:]
        pad = np.repeat(last, flows_per_clip - picked.shape[0], axis=0)
        picked = np.concatenate([picked, pad], axis=0)

    frames = []
    for i in range(flows_per_clip):
        flow_uv = picked[i]
        flow_uv = resize_flow(flow_uv, img_size)
        frames.append(flow_to_tensor(flow_uv))  # (2,H,W)

    return torch.stack(frames, dim=0)  # (T,2,H,W)


class ClipDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        mode: str,
        img_size: int,
        frames_rgb: int,
        flows_per_clip: int,
        flow_file: str,
    ):
        self.mode = str(mode)
        self.img_size = int(img_size)
        self.frames_rgb = int(frames_rgb)
        self.flows_per_clip = int(flows_per_clip)
        self.flow_file = str(flow_file)

        if self.mode not in ("rgb", "flow"):
            raise ValueError(f"mode must be 'rgb' or 'flow', got: {self.mode}")

        self.rows: List[Tuple[str, str, int]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                sid = str(row["id"])
                path = str(row.get("path", row.get("clip_dir", "")))
                if path == "":
                    raise RuntimeError("CSV must have column 'path' (or 'clip_dir').")
                label = int(row["label"])
                self.rows.append((sid, path, label))

        if len(self.rows) == 0:
            raise RuntimeError(f"Empty csv: {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        sid, clip_dir, label = self.rows[idx]

        if self.mode == "rgb":
            x = _load_rgb_clip(clip_dir, self.frames_rgb, self.img_size)
        else:
            x = _load_flow_clip(clip_dir, self.flow_file, self.flows_per_clip, self.img_size)

        # IMPORTANT:
        # return scalar label (not (1,)) to avoid (B,1) issues in eval metrics
        y = torch.tensor(float(label), dtype=torch.float32)  # scalar
        return x, y, sid
