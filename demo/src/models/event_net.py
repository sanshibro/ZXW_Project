"""
Stage-1 Lightweight Photon Interaction Event Discrimination & Sequence Parsing Network (LiteEventNet).
- Parameters: ~33K (ultra-compact, microsecond inference, FPGA/edge friendly)
- Tasks:
  1) 4-Class Event Sanitization & Sequence Ordering:
     0: Invalid / Escape / Mask-scattered / Random Coincidence Background
     1: Clean Collimated Single-Site Photoelectric Event
     2: Forward-Ordered Full-Absorption Compton Event (Hit A -> Hit B)
     3: Reverse-Ordered Full-Absorption Compton Event (Hit B -> Hit A)
  2) Incident Gamma Energy Regression (keV)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics.digitizer import (
    DigitizedEvents,
    CLS_VALID_COLLIMATED,
    CLS_COMPTON_FWD_1TO2,
    CLS_COMPTON_REV_2TO1,
)


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + h


class LiteEventNet(nn.Module):
    def __init__(self, in_features: int = 16, hidden_dim: int = 96, num_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.block1 = ResMLPBlock(hidden_dim, dropout=0.05)
        self.block2 = ResMLPBlock(hidden_dim, dropout=0.05)
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.ene_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.block2(self.block1(h))
        logits = self.cls_head(h)
        # Predict energy in units of 1000 keV, mapped strictly positive
        e0_pred_kev = F.softplus(self.ene_head(h)).squeeze(-1) * 1000.0
        return logits, e0_pred_kev


@dataclass
class SanitizedEventPack:
    pix_ix_col: np.ndarray
    pix_iy_col: np.ndarray
    e_col_kev: np.ndarray
    r1_comp_mm: np.ndarray
    e1_comp_kev: np.ndarray
    r2_comp_mm: np.ndarray
    e2_comp_kev: np.ndarray
    pred_classes: np.ndarray
    probs_4cls: np.ndarray
    pred_e0_kev: np.ndarray


@torch.no_grad()
def sanitize_and_parse_events(
    model: LiteEventNet,
    dig: DigitizedEvents,
    device: torch.device | str = "cpu",
    compton_conf_thresh: float = 0.42,
    collimated_conf_thresh: float = 0.45,
) -> SanitizedEventPack:
    """
    Run Stage-1 LiteEventNet on DigitizedEvents to:
    1. Reject photon escape, mask scatter, and random coincidence background;
    2. Extract clean collimated hits;
    3. Extract clean Compton pairs and automatically correct reversed hit ordering (B -> A)!
    """
    model.eval()
    x_t = torch.from_numpy(dig.features_16d).to(device)
    logits, pred_e0 = model(x_t)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    pred_cls = np.argmax(probs, axis=-1)
    pred_e0_np = pred_e0.cpu().numpy()

    # Valid collimated single-site hits in Layer 0
    col_mask = (
        (pred_cls == CLS_VALID_COLLIMATED)
        & (probs[:, CLS_VALID_COLLIMATED] >= collimated_conf_thresh)
        & (dig.n_hits == 1)
        & (dig.layer1 == 0)
    )
    ix_c = dig.pix_ix1[col_mask]
    iy_c = dig.pix_iy1[col_mask]
    e_c = dig.e1_dig_kev[col_mask]

    # Forward-ordered Compton events (Hit A -> Hit B)
    fwd_mask = (
        (pred_cls == CLS_COMPTON_FWD_1TO2)
        & (probs[:, CLS_COMPTON_FWD_1TO2] >= compton_conf_thresh)
        & (dig.n_hits == 2)
    )
    # Reverse-ordered Compton events (Hit B -> Hit A -> unswap!)
    rev_mask = (
        (pred_cls == CLS_COMPTON_REV_2TO1)
        & (probs[:, CLS_COMPTON_REV_2TO1] >= compton_conf_thresh)
        & (dig.n_hits == 2)
    )

    r1_c = np.concatenate([dig.r1_dig_mm[fwd_mask], dig.r2_dig_mm[rev_mask]], axis=0)
    e1_c = np.concatenate([dig.e1_dig_kev[fwd_mask], dig.e2_dig_kev[rev_mask]], axis=0)
    r2_c = np.concatenate([dig.r2_dig_mm[fwd_mask], dig.r1_dig_mm[rev_mask]], axis=0)
    e2_c = np.concatenate([dig.e2_dig_kev[fwd_mask], dig.e1_dig_kev[rev_mask]], axis=0)

    return SanitizedEventPack(
        pix_ix_col=ix_c,
        pix_iy_col=iy_c,
        e_col_kev=e_c,
        r1_comp_mm=r1_c,
        e1_comp_kev=e1_c,
        r2_comp_mm=r2_c,
        e2_comp_kev=e2_c,
        pred_classes=pred_cls,
        probs_4cls=probs,
        pred_e0_kev=pred_e0_np,
    )
