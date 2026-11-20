"""
【AutoDL RTX 3080 Ti 端】手势数字分类器训练与评估脚本

功能:
  - 自动合并 data/processed/*.npz（含自采中国数字 + 公开数字 + OpenAoE 负样本）
  - 自动按类别逆频率加权，解决负样本比例不平衡问题
  - 支持 AMP (FP16)、CosineAnnealingLR、Label Smoothing、Early Stopping
  - 输出完整分类报告、混淆矩阵与 OpenAoE 负样本误触发率 (FPR)

用法:
    python scripts/03_train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import HandLandmarkDataset
from src.model import HandDigitResMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_all_npz(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"在 {data_dir} 中未找到任何 .npz 数据文件。请先运行 01_collect_pc_data.py 或 02_fetch_openaoe_negatives.py。"
        )
    all_lms, all_hands, all_labels = [], [], []
    for f in npz_files:
        d = np.load(f)
        all_lms.append(d["landmarks"])
        all_hands.append(d["handedness"])
        all_labels.append(d["labels"])
        print(f"[Data] 加载 {f.name}: {len(d['labels'])} 条")

    landmarks = np.concatenate(all_lms, axis=0)
    handedness = np.concatenate(all_hands, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return landmarks, handedness, labels


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return macro_f1, y_true, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="训练中国单手数字分类器")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] 使用计算设备: {device}")

    data_dir = Path(cfg["paths"]["data_dir"])
    landmarks, handedness, labels = load_all_npz(data_dir)
    print(f"[Data] 合并后总样本数: {len(labels)} | 类别分布: {dict(sorted(Counter(labels.tolist()).items()))}")

    val_ratio = cfg["train"]["val_ratio"]
    test_ratio = cfg["train"]["test_ratio"]

    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_ratio, stratify=labels, random_state=cfg["project"]["seed"]
    )
    rel_val = val_ratio / (1.0 - test_ratio)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=rel_val,
        stratify=labels[idx_trainval],
        random_state=cfg["project"]["seed"],
    )

    train_ds = HandLandmarkDataset(
        landmarks[idx_train], handedness[idx_train], labels[idx_train], augment=True
    )
    val_ds = HandLandmarkDataset(
        landmarks[idx_val], handedness[idx_val], labels[idx_val], augment=False
    )
    test_ds = HandLandmarkDataset(
        landmarks[idx_test], handedness[idx_test], labels[idx_test], augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    num_classes = cfg["model"]["num_classes"]
    model = HandDigitResMLP(
        input_dim=cfg["features"]["input_dim"],
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=num_classes,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # 计算平滑逆频率类别权重，防止 OpenAoE 负样本过多淹没数字类
    train_counts = np.bincount(labels[idx_train], minlength=num_classes).astype(np.float32)
    class_weights = np.where(train_counts > 0, 1.0 / np.sqrt(train_counts + 1.0), 0.0)
    if class_weights.sum() > 0:
        class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=cfg["train"]["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-5
    )
    use_amp = bool(cfg["train"]["use_amp"] and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = Path(cfg["paths"]["best_model_pt"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    patience = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(y)

        scheduler.step()
        avg_loss = total_loss / len(train_ds)
        val_f1, _, _ = evaluate(model, val_loader, device)

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "val_macro_f1": best_f1,
                },
                ckpt_path,
            )
            tag = " *"
        else:
            patience += 1
            tag = ""

        if epoch == 1 or epoch % 5 == 0 or tag:
            print(
                f"Epoch {epoch:03d}/{cfg['train']['epochs']} | "
                f"TrainLoss: {avg_loss:.4f} | Val Macro-F1: {val_f1:.4f}{tag}"
            )

        if patience >= cfg["train"]["early_stop_patience"]:
            print(f"[EarlyStop] 连续 {patience} 轮无提升，提前结束训练。")
            break

    # 加载最佳权重在测试集上做最终评估
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    print("\n================ 测试集最终评估 ================")
    print(f"Test Macro-F1: {test_f1:.4f}")
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [str(cfg["classes"][i]) for i in present_labels]
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=target_names, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=present_labels))

    # 专项指标：日常非数字手势 (Class 10) 被误触发为 0-9 的比例
    neg_mask = y_true == 10
    if neg_mask.any():
        fpr = float((y_pred[neg_mask] != 10).mean())
        print(f"\n[关键指标] 非数字手势误触发率 (FPR on Other/OpenAoE): {fpr * 100:.2f}%")


if __name__ == "__main__":
    main()
