"""
导出 ONNX 模型并校验 PyTorch 与 ONNX Runtime 数值一致性

用法:
    python scripts/04_export_onnx.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import HandDigitResMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(cfg["paths"]["best_model_pt"])
    onnx_path = Path(cfg["paths"]["onnx_model"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到训练权重: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HandDigitResMLP(
        input_dim=cfg["features"]["input_dim"],
        hidden_dims=cfg["model"]["hidden_dims"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, cfg["features"]["input_dim"], dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )
    print(f"[Export] ONNX 模型已导出 -> {onnx_path}")

    # 数值一致性校验
    with torch.no_grad():
        pt_out = model(dummy_input).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"features": dummy_input.numpy()})[0]
    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"[Verify] PyTorch vs ONNXRuntime 最大绝对误差: {max_diff:.2e}")
    assert max_diff < 1e-4, f"导出误差过大: {max_diff}"
    print("[Verify] 校验通过！可直接拷贝至 PC 端运行。")


if __name__ == "__main__":
    main()
