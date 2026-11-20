"""
MediaPipe Hands 封装模块（用于 PC 采集、OpenAoE 抽帧与 PC 实时推理）

基于 MediaPipe Tasks API (HandLandmarker)，兼容 Python 3.14 / mediapipe>=0.10.30。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (conn.start, conn.end) for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS
]


@dataclass
class HandDetection:
    landmarks: np.ndarray          # shape (21, 3), 归一化图像坐标 (x, y, z)
    handedness: float              # 1.0 = Right, 0.0 = Left
    handedness_label: str          # "Right" / "Left"
    score: float                   # 手部检测置信度
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) 像素框


class HandExtractor:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,  # 保留参数以兼容旧调用方（Tasks 模型不区分 complexity）
        model_path: Optional[str | Path] = None,
    ) -> None:
        del model_complexity
        resolved_model = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        if not resolved_model.is_file():
            raise FileNotFoundError(
                f"未找到 MediaPipe 手部模型文件: {resolved_model}。"
                "请先下载 hand_landmarker.task 并放置于该路径。"
            )

        running_mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=resolved_model.read_bytes()),
            running_mode=running_mode,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._static_image_mode = static_image_mode
        self._last_timestamp_ms = 0
        self._landmarker: Optional[vision.HandLandmarker] = (
            vision.HandLandmarker.create_from_options(options)
        )

    def extract(self, bgr_image: np.ndarray) -> List[HandDetection]:
        """从 BGR 图像中提取检测到的手部 21 关键点。"""
        if self._landmarker is None:
            raise RuntimeError("HandExtractor 已关闭，无法继续提取关键点。")

        h, w = bgr_image.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._static_image_mode:
            results = self._landmarker.detect(mp_image)
        else:
            ts_ms = time.monotonic_ns() // 1_000_000
            if ts_ms <= self._last_timestamp_ms:
                ts_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = ts_ms
            results = self._landmarker.detect_for_video(mp_image, ts_ms)

        detections: List[HandDetection] = []
        if not results.hand_landmarks:
            return detections

        handedness_list = results.handedness or []
        for idx, hand_lms in enumerate(results.hand_landmarks):
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms],
                dtype=np.float32,
            )
            if idx < len(handedness_list) and handedness_list[idx]:
                cls_info = handedness_list[idx][0]
                label = cls_info.category_name or cls_info.display_name or "Right"
                score = float(cls_info.score) if cls_info.score is not None else 1.0
            else:
                label = "Right"
                score = 1.0

            xs = (pts[:, 0] * w).astype(int)
            ys = (pts[:, 1] * h).astype(int)
            pad = 16
            x1 = max(0, int(xs.min()) - pad)
            y1 = max(0, int(ys.min()) - pad)
            x2 = min(w - 1, int(xs.max()) + pad)
            y2 = min(h - 1, int(ys.max()) + pad)

            detections.append(
                HandDetection(
                    landmarks=pts,
                    handedness=1.0 if label == "Right" else 0.0,
                    handedness_label=label,
                    score=score,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections

    def draw_landmarks(self, bgr_image: np.ndarray, landmarks: np.ndarray) -> None:
        """在图像上原地绘制 21 关键点与连线。"""
        h, w = bgr_image.shape[:2]
        pts_px = [(int(x * w), int(y * h)) for x, y, _ in landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(bgr_image, pts_px[start_idx], pts_px[end_idx], (0, 220, 120), 2)
        for px, py in pts_px:
            cv2.circle(bgr_image, (px, py), 4, (0, 120, 255), -1)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "HandExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
