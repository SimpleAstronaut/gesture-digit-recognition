"""
关键点数据集与在线几何数据增强

增强全部作用于原始 (21, 3) 关键点空间，随后再计算 88 维几何特征，
保证角度、距离与坐标之间严格几何一致。
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features import extract_features


def augment_landmarks(
    landmarks: np.ndarray,
    handedness: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """对单帧 (21, 3) 关键点施加第一人称视角友好的在线增强。"""
    pts = landmarks.astype(np.float32).copy()

    # 1. 随机水平镜像（中国单手数字 0-9 左右手语义对称）
    if rng.random() < 0.5:
        pts[:, 0] = -pts[:, 0]
        handedness = 1.0 - handedness

    # 2. 随机面内旋转 (±25°，模拟头部或手腕倾斜)
    angle_z = rng.uniform(-25.0, 25.0) * (np.pi / 180.0)
    cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
    rot_z = np.array(
        [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    pts = pts @ rot_z.T

    # 3. 轻微 3D 俯仰/偏航扰动 (±12°，模拟不同拍摄视角)
    angle_y = rng.uniform(-12.0, 12.0) * (np.pi / 180.0)
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    rot_y = np.array(
        [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]],
        dtype=np.float32,
    )
    pts = pts @ rot_y.T

    # 4. 各向异性微缩放（模拟不同手型胖瘦与摄像头畸变）
    scale = rng.uniform(0.90, 1.10, size=(1, 3)).astype(np.float32)
    pts *= scale

    # 5. 高斯抖动（模拟 MediaPipe 检出误差与运动模糊）
    palm_span = float(np.linalg.norm(pts[9] - pts[0])) + 1e-4
    noise_std = rng.uniform(0.0, 0.025) * palm_span
    pts += rng.normal(0.0, noise_std, size=pts.shape).astype(np.float32)

    # 6. 局部遮挡扰动（20% 概率对 1-2 个非手腕点施加较大抖动）
    if rng.random() < 0.2:
        k = int(rng.integers(1, 3))
        idx = rng.choice(np.arange(1, 21), size=k, replace=False)
        pts[idx] += rng.normal(0.0, 0.06 * palm_span, size=(k, 3)).astype(np.float32)

    return pts, handedness


class HandLandmarkDataset(Dataset):
    def __init__(
        self,
        landmarks: np.ndarray,
        handedness: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.landmarks = np.asarray(landmarks, dtype=np.float32)
        self.handedness = np.asarray(handedness, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pts = self.landmarks[idx]
        hand = float(self.handedness[idx])
        if self.augment:
            pts, hand = augment_landmarks(pts, hand, self.rng)
        feat = extract_features(pts, handedness=hand)
        return torch.from_numpy(feat), torch.tensor(self.labels[idx], dtype=torch.long)
