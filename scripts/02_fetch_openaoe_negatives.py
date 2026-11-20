"""
【AutoDL 端】按需拉取 ModelScope OpenAoE-2000h 子集，提取第一人称日常手部关键点作为负样本 (Class 10: Other)

核心特性:
  1. 绝不全量下载 OpenAoE-2000h，仅按 --max-videos 逐个拉取视频分片
  2. 每处理完一个视频立即删除原始大文件（默认），防止 AutoDL 50GB 数据盘爆盘
  3. 仅保存轻量 21 关键点 (.npz)，1 万条负样本不到 2 MB
  4. 也支持 --local-video-dir 处理你手动下载好的本地第一人称视频目录

用法示例 (AutoDL 推荐):
    python scripts/02_fetch_openaoe_negatives.py \
        --cache-dir /root/autodl-tmp/openaoe_cache \
        --output data/processed/openaoe_negatives.npz \
        --max-videos 30 \
        --max-samples 8000 \
        --sample-stride 15
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hand_extractor import HandExtractor

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def list_modelscope_videos(dataset_id: str, max_videos: int) -> List[str]:
    """通过 ModelScope HubApi 列出数据集中的视频文件路径（不触发下载）。"""
    from modelscope.hub.api import HubApi

    api = HubApi()
    video_files: List[str] = []

    def _walk(root_path: str = "") -> None:
        if len(video_files) >= max_videos:
            return
        files = api.get_dataset_files(
            repo_id=dataset_id,
            root_path=root_path,
            recursive=False,
        )
        for item in files:
            if len(video_files) >= max_videos:
                return
            ftype = item.get("Type", "")
            fpath = item.get("Path", "")
            if ftype == "tree":
                _walk(fpath)
            elif Path(fpath).suffix.lower() in VIDEO_EXTS:
                video_files.append(fpath)

    print(f"[Info] 正在检索 ModelScope 数据集 {dataset_id} 的视频分片列表...")
    _walk("")
    print(f"[Info] 检索完成，计划按需拉取 {len(video_files)} 个视频分片。")
    return video_files


def extract_from_video(
    video_path: Path,
    extractor: HandExtractor,
    sample_stride: int,
    max_per_video: int,
) -> tuple[list[np.ndarray], list[float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], []

    lms_list: list[np.ndarray] = []
    hand_list: list[float] = []
    frame_idx = 0

    while len(lms_list) < max_per_video:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            detections = extractor.extract(frame)
            for det in detections:
                lms_list.append(det.landmarks)
                hand_list.append(det.handedness)
                if len(lms_list) >= max_per_video:
                    break
        frame_idx += 1

    cap.release()
    return lms_list, hand_list


def iter_local_videos(local_dir: Path) -> Iterable[Path]:
    for p in sorted(local_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def main() -> None:
    parser = argparse.ArgumentParser(description="按需拉取 OpenAoE-2000h 子集并提取负样本关键点")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="inclusionAI/OpenAoE-2000h",
        help="ModelScope 数据集 ID",
    )
    parser.add_argument(
        "--local-video-dir",
        type=str,
        default="",
        help="若已手动下载部分视频，可指定本地目录直接抽帧（跳过 ModelScope 下载）",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/root/autodl-tmp/openaoe_cache",
        help="临时视频缓存目录（建议放在 AutoDL 数据盘）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/openaoe_negatives.npz",
        help="输出负样本 .npz 路径",
    )
    parser.add_argument("--max-videos", type=int, default=30, help="最多拉取的视频数")
    parser.add_argument("--max-samples", type=int, default=8000, help="目标负样本总帧数")
    parser.add_argument("--max-per-video", type=int, default=300, help="单个视频最多抽取手部样本数")
    parser.add_argument("--sample-stride", type=int, default=15, help="抽帧步长（30fps 下 15 约每秒 2 帧）")
    parser.add_argument("--keep-videos", action="store_true", help="处理完后保留原始视频（默认自动删除防爆盘）")
    args = parser.parse_args()

    all_landmarks: list[np.ndarray] = []
    all_handedness: list[float] = []

    with HandExtractor(static_image_mode=False, max_num_hands=2) as extractor:
        if args.local_video_dir:
            videos = list(iter_local_videos(Path(args.local_video_dir)))[: args.max_videos]
            for vpath in tqdm(videos, desc="Extracting local videos"):
                if len(all_landmarks) >= args.max_samples:
                    break
                lms, hands = extract_from_video(
                    vpath, extractor, args.sample_stride, args.max_per_video
                )
                all_landmarks.extend(lms)
                all_handedness.extend(hands)
        else:
            from modelscope.hub.file_download import dataset_file_download

            cache_dir = Path(args.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            remote_videos = list_modelscope_videos(args.dataset_id, args.max_videos)

            for remote_path in tqdm(remote_videos, desc="Fetching & Extracting OpenAoE"):
                if len(all_landmarks) >= args.max_samples:
                    break
                local_file = Path(
                    dataset_file_download(
                        dataset_id=args.dataset_id,
                        file_path=remote_path,
                        cache_dir=str(cache_dir),
                    )
                )
                try:
                    lms, hands = extract_from_video(
                        local_file, extractor, args.sample_stride, args.max_per_video
                    )
                    all_landmarks.extend(lms)
                    all_handedness.extend(hands)
                finally:
                    if not args.keep_videos and local_file.exists():
                        os.remove(local_file)

    if not all_landmarks:
        print("[Warn] 未提取到任何手部关键点，请检查视频源或 ModelScope 网络/认证。")
        return

    all_landmarks = all_landmarks[: args.max_samples]
    all_handedness = all_handedness[: args.max_samples]
    labels = np.full(len(all_landmarks), 10, dtype=np.int64)  # Class 10 = Other

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        landmarks=np.asarray(all_landmarks, dtype=np.float32),
        handedness=np.asarray(all_handedness, dtype=np.float32),
        labels=labels,
    )
    print(f"[Done] 已生成 {len(labels)} 条第一人称负样本 -> {out_path}")


if __name__ == "__main__":
    main()
