# src/datasets/transforms.py

import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_rgb(img_rgb: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)


def rgb_to_tensor_imagenet(img_rgb: np.ndarray) -> torch.Tensor:
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # (3,H,W)
    mean = IMAGENET_MEAN[:, None, None]
    std = IMAGENET_STD[:, None, None]
    x = (x - mean) / std
    return torch.from_numpy(x).float()


def resize_flow(flow_uv: np.ndarray, size: int) -> np.ndarray:
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    u2 = cv2.resize(u, (size, size), interpolation=cv2.INTER_LINEAR)
    v2 = cv2.resize(v, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.stack([u2, v2], axis=-1).astype(np.float32)


def flow_to_tensor(flow_uv: np.ndarray) -> torch.Tensor:
    x = np.transpose(flow_uv, (2, 0, 1))  # (2,H,W)
    return torch.from_numpy(x).float()
