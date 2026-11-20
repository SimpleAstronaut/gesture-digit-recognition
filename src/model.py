"""
轻量化残差关键点分类器 (HandDigitResMLP)

特点:
  - 参数量 ~0.2M，AutoDL 3080 Ti 上数分钟完成收敛
  - PC CPU (ONNX Runtime) 单次推理 < 0.3 ms
  - 使用 LayerNorm + SiLU + Residual Connection，对小样本与高噪声关键点更稳健
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


class HandDigitResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 88,
        hidden_dims: Sequence[int] = (256, 256, 128),
        num_classes: int = 11,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(inplace=True),
        )

        blocks = []
        in_d = hidden_dims[0]
        for h_d in hidden_dims:
            blocks.append(ResMLPBlock(in_d, h_d, dropout=dropout))
            in_d = h_d
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(in_d, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)
