"""
【PC 端】中国单手数字 0-9 摄像头快速采集工具

用法:
    python scripts/01_collect_pc_data.py --output data/processed/pc_collected.npz

按键说明:
    0 - 9   : 切换当前采集的数字标签 (0~9)
    o       : 切换为 Other (非数字手势, label=10)
    Space   : 开始 / 暂停 连续采集
    s       : 立即保存当前已采集数据到磁盘
    q / ESC : 保存并退出
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hand_extractor import HandExtractor


def load_existing(path: Path) -> tuple[list[np.ndarray], list[float], list[int]]:
    if not path.exists():
        return [], [], []
    data = np.load(path)
    return (
        list(data["landmarks"]),
        list(data["handedness"].astype(float)),
        list(data["labels"].astype(int)),
    )


def save_dataset(
    path: Path,
    landmarks: list[np.ndarray],
    handedness: list[float],
    labels: list[int],
) -> None:
    if not landmarks:
        print("[Warn] 当前没有可保存的数据。")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        landmarks=np.asarray(landmarks, dtype=np.float32),
        handedness=np.asarray(handedness, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
    )
    counts = dict(sorted(Counter(labels).items()))
    print(f"[Saved] 共保存 {len(labels)} 条样本 -> {path} | 各类别分布: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PC 端中国单手数字关键点采集工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/pc_collected.npz",
        help="输出 .npz 文件路径",
    )
    parser.add_argument("--flip", action="store_true", help="是否镜像翻转画面（前置自拍视角可开启）")
    args = parser.parse_args()

    out_path = Path(args.output)
    landmarks_buf, handedness_buf, labels_buf = load_existing(out_path)
    if labels_buf:
        print(f"[Info] 已加载历史数据 {len(labels_buf)} 条，将继续追加。")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.camera}，请检查设备连接。")

    current_label = 0
    recording = False
    print("=== 采集启动：按 0-9 选数字，o 选 Other，空格开始/暂停录制，q 保存退出 ===")

    with HandExtractor(static_image_mode=False, max_num_hands=1) as extractor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            detections = extractor.extract(frame)
            if detections:
                det = detections[0]
                extractor.draw_landmarks(frame, det.landmarks)
                x1, y1, x2, y2 = det.bbox
                box_color = (0, 0, 255) if recording else (0, 220, 120)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                if recording:
                    landmarks_buf.append(det.landmarks)
                    handedness_buf.append(det.handedness)
                    labels_buf.append(current_label)

            # 绘制 HUD 状态栏
            cls_name = "Other(10)" if current_label == 10 else str(current_label)
            cls_count = sum(1 for y in labels_buf if y == current_label)
            status_text = "REC" if recording else "PAUSED"
            status_color = (0, 0, 255) if recording else (200, 200, 200)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (25, 25, 25), -1)
            cv2.putText(
                frame,
                f"Target: [{cls_name}]  Count: {cls_count}  Total: {len(labels_buf)}  [{status_text}]",
                (16, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                status_color,
                2,
            )
            cv2.putText(
                frame,
                "Keys: [0-9] Digit | [o] Other | [Space] Rec/Pause | [s] Save | [q] Quit",
                (16, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 220, 255),
                1,
            )

            cv2.imshow("Chinese Hand Digit Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord(" "):
                recording = not recording
            elif key == ord("s"):
                save_dataset(out_path, landmarks_buf, handedness_buf, labels_buf)
            elif key == ord("o"):
                current_label = 10
                recording = False
            elif ord("0") <= key <= ord("9"):
                current_label = key - ord("0")
                recording = False

    cap.release()
    cv2.destroyAllWindows()
    save_dataset(out_path, landmarks_buf, handedness_buf, labels_buf)


if __name__ == "__main__":
    main()
