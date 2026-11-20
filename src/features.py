"""
21-Landmark 几何不变特征工程（专为中国单手数字 0-9 优化）

输出 88 维特征向量:
  - [0:63]   相对手腕平移 + 手掌尺度归一化后的 21 个 (x, y, z) 坐标
  - [63:77]  14 个关节弯曲夹角 & 相邻手指张角（余弦值，区分 6/8/9 等弯曲与张开）
  - [77:86]  9 个归一化距离特征（5 指尖到手腕 + 拇指尖到其余 4 指尖，区分 0/7 捏合）
  - [86:88]  左右手标志 (0/1) + 手掌朝向法向量 z 分量（区分手心/手背）
"""

from __future__ import annotations

import numpy as np

# MediaPipe 21 关键点索引常量
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# 10 个指关节弯曲角三元组 (A, B, C) -> 计算向量 BA 与 BC 的夹角余弦
BEND_ANGLE_TRIPLETS = [
    (THUMB_CMC, THUMB_MCP, THUMB_IP),
    (THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP),
    (INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP),
    (RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP),
    (PINKY_PIP, PINKY_DIP, PINKY_TIP),
]

# 4 个相邻手指张角 (向量: MCP -> TIP 之间的夹角余弦)
SPREAD_VECTOR_PAIRS = [
    ((THUMB_MCP, THUMB_TIP), (INDEX_MCP, INDEX_TIP)),
    ((INDEX_MCP, INDEX_TIP), (MIDDLE_MCP, MIDDLE_TIP)),
    ((MIDDLE_MCP, MIDDLE_TIP), (RING_MCP, RING_TIP)),
    ((RING_MCP, RING_TIP), (PINKY_MCP, PINKY_TIP)),
]

FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]


def _cosine_angle(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-6) -> float:
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return 0.0
    return float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    将 (21, 3) 关键点以手腕为原点平移，并以手掌尺度（手腕到中指 MCP 距离）归一化。
    """
    pts = np.asarray(landmarks, dtype=np.float32).reshape(21, 3).copy()
    wrist = pts[WRIST].copy()
    pts -= wrist

    # 以手腕到中指根部的距离作为尺度基准，若退化则退回最大范数
    palm_scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if palm_scale < 1e-4:
        palm_scale = float(np.max(np.linalg.norm(pts, axis=1))) + 1e-6
    pts /= palm_scale
    return pts


def extract_features(landmarks: np.ndarray, handedness: float = 1.0) -> np.ndarray:
    """
    从原始 (21, 3) 关键点提取 88 维几何特征。

    Args:
        landmarks: shape (21, 3) 的浮点数组，来自 MediaPipe 或 MANO 21 点投影。
        handedness: 1.0 表示右手，0.0 表示左手。

    Returns:
        shape (88,) 的 float32 特征向量。
    """
    pts = normalize_landmarks(landmarks)

    # 1. 归一化坐标 (63 维)
    coord_feat = pts.reshape(-1)

    # 2. 指关节弯曲夹角 (10 维) + 相邻手指张角 (4 维) = 14 维
    bend_angles = [
        _cosine_angle(pts[a] - pts[b], pts[c] - pts[b])
        for a, b, c in BEND_ANGLE_TRIPLETS
    ]
    spread_angles = [
        _cosine_angle(pts[p1[1]] - pts[p1[0]], pts[p2[1]] - pts[p2[0]])
        for p1, p2 in SPREAD_VECTOR_PAIRS
    ]
    angle_feat = np.array(bend_angles + spread_angles, dtype=np.float32)

    # 3. 距离特征 (9 维): 5 指尖到手腕 + 拇指尖到其余 4 指尖（对 0 和 7 极度敏感）
    tip_to_wrist = [float(np.linalg.norm(pts[tip])) for tip in FINGERTIPS]
    thumb_to_others = [
        float(np.linalg.norm(pts[THUMB_TIP] - pts[tip]))
        for tip in FINGERTIPS[1:]
    ]
    dist_feat = np.array(tip_to_wrist + thumb_to_others, dtype=np.float32)

    # 4. 手掌法向量 z 分量（区分掌心/手背）+ 左右手标志 (2 维)
    v_index = pts[INDEX_MCP] - pts[WRIST]
    v_pinky = pts[PINKY_MCP] - pts[WRIST]
    normal = np.cross(v_index, v_pinky)
    normal_norm = float(np.linalg.norm(normal))
    palm_facing_z = float(normal[2] / normal_norm) if normal_norm > 1e-6 else 0.0
    meta_feat = np.array([float(handedness), palm_facing_z], dtype=np.float32)

    feat = np.concatenate([coord_feat, angle_feat, dist_feat, meta_feat], axis=0)
    return feat.astype(np.float32)
