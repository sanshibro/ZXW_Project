"""
BNCT 478 keV 复杂辐射场 · 光子作用事例甄别与散射角信息解析网络 (BNCTEventNet)
(Inverse Reconstruction Package - Stage 1 Event Discrimination & Kinematics Parsing)

五分类目标定义:
  Class 0: 有效 478 keV 单小孔准直投影事例 (Valid 478 keV Pinhole)
  Class 1: 有效 478 keV 正向全吸收康普顿事例 A -> B (Valid 478 keV Compton A->B)
  Class 2: 有效 478 keV 逆向反跳康普顿事例 B -> A (Valid 478 keV Compton B->A)
  Class 3: BNCT 非目标多能量射线干扰事例 (511 keV / 2223 keV / 杂散谱)
  Class 4: 散射光子逃逸、准直孔壁散射或电子学暗噪声伪事例 (Escape / Wall-Scatter)
同时输出双点康普顿散射角余弦残差修正量 delta_cos_theta，缓解 478 keV 低反冲电子能量测角展宽。
"""

from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class _ResMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class BNCTEventNet(nn.Module):
    def __init__(self, in_dim: int = 18, hidden_dim: int = 96, num_classes: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.res1 = _ResMLPBlock(hidden_dim)
        self.res2 = _ResMLPBlock(hidden_dim)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 48),
            nn.GELU(),
            nn.Linear(48, num_classes),
        )
        # 散射角余弦残差微调头 (限制在 [-0.25, +0.25] 物理合理修正区间)
        self.cos_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.res2(self.res1(self.stem(x)))
        logits = self.cls_head(h)
        delta_cos = 0.25 * self.cos_head(h).squeeze(-1)
        return logits, delta_cos
