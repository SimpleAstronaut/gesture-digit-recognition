"""
视频流时序防抖后处理器（滑动窗口投票 + 置信度阈值）
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Optional


class PredictionSmoother:
    def __init__(
        self,
        window_size: int = 7,
        min_consensus: int = 5,
        confidence_threshold: float = 0.75,
        other_class_id: int = 10,
    ) -> None:
        self.window = deque(maxlen=window_size)
        self.min_consensus = min_consensus
        self.confidence_threshold = confidence_threshold
        self.other_class_id = other_class_id
        self.stable_class: Optional[int] = None

    def update(self, class_id: Optional[int], confidence: float) -> Optional[int]:
        """
        输入当前帧预测结果，返回平滑后的稳定类别（若为非数字或未达阈值则返回 None）。
        """
        if class_id is None or confidence < self.confidence_threshold:
            self.window.append(self.other_class_id)
        else:
            self.window.append(class_id)

        if len(self.window) < self.min_consensus:
            return self.stable_class

        top_cls, count = Counter(self.window).most_common(1)[0]
        if count >= self.min_consensus:
            self.stable_class = None if top_cls == self.other_class_id else top_cls
        return self.stable_class

    def reset(self) -> None:
        self.window.clear()
        self.stable_class = None
