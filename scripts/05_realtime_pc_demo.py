"""
【PC 端】中国单手数字实时识别 Demo（ONNX Runtime + MediaPipe + OpenCV）

特点:
  - 无需安装 PyTorch，普通笔记本 CPU 即可满帧运行
  - 内置滑动窗口防抖与非数字手势过滤

用法:
    python scripts/05_realtime_pc_demo.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import extract_features
from src.hand_extractor import HandExtractor
from src.smoother import PredictionSmoother


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def main() -> None:
    parser = argparse.ArgumentParser(description="PC 端中国单手数字实时识别")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--flip", action="store_true", help="镜像翻转画面")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    onnx_path = Path(cfg["paths"]["onnx_model"])
    if not onnx_path.exists():
        raise FileNotFoundError(f"未找到 ONNX 模型: {onnx_path}，请先在 AutoDL 导出并拷贝至此路径。")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    print(f"[Info] ONNX Runtime 已加载 ({sess.get_providers()[0]})")

    inf_cfg = cfg["inference"]
    smoother = PredictionSmoother(
        window_size=inf_cfg["smooth_window"],
        min_consensus=inf_cfg["min_consensus"],
        confidence_threshold=inf_cfg["confidence_threshold"],
        other_class_id=10,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}")

    prev_t = time.perf_counter()
    fps = 0.0

    with HandExtractor(
        static_image_mode=False,
        max_num_hands=inf_cfg["max_num_hands"],
        min_detection_confidence=inf_cfg["min_detection_confidence"],
        min_tracking_confidence=inf_cfg["min_tracking_confidence"],
    ) as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            detections = extractor.extract(frame)
            raw_cls, raw_conf = None, 0.0

            if detections:
                det = detections[0]
                extractor.draw_landmarks(frame, det.landmarks)
                feat = extract_features(det.landmarks, handedness=det.handedness)[None, :]
                logits = sess.run(None, {input_name: feat})[0][0]
                probs = softmax(logits)
                raw_cls = int(np.argmax(probs))
                raw_conf = float(probs[raw_cls])

                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
                raw_name = cfg["classes"][raw_cls]
                cv2.putText(
                    frame,
                    f"{raw_name} ({raw_conf * 100:.1f}%)",
                    (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 180),
                    2,
                )

            stable_cls = smoother.update(raw_cls, raw_conf)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            # 顶部 HUD 面板
            display_digit = str(cfg["classes"][stable_cls]) if stable_cls is not None else "--"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 68), (25, 25, 25), -1)
            cv2.putText(
                frame,
                f"Digit: {display_digit}",
                (18, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25,
                (0, 230, 255) if stable_cls is not None else (160, 160, 160),
                3,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  |  Press [q] to quit",
                (frame.shape[1] - 280, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

            cv2.imshow("Chinese Hand Digit Recognition (PC ONNX)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
