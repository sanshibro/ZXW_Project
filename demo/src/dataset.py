"""
Feature projection helpers and PyTorch Dataset classes for Dual-Mode Gamma Reconstruction.
"""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from .physics.geometry_and_sources import CameraGeometry
from .baselines.traditional_recon import decode_collimated_projection, compton_simple_backprojection
from .models.event_net import SanitizedEventPack


def build_dual_branch_tensors(
    geom: CameraGeometry,
    sanitized: SanitizedEventPack,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert Stage-1 AI-sanitized photon events into 2D physics-projected input tensors
    for Stage-2 LiteDualGammaNet:
      x_col:  (3, H, W) -> [low_E_MURA (<=240 keV), full_E_MURA, log_lowE_count_norm]
      x_comp: (3, H, W) -> [compton_SBP, compton_SBP_sharpened_sq, log_comp_count_norm]
    """
    h = geom.grid_size
    ix = sanitized.pix_ix_col
    iy = sanitized.pix_iy_col
    ec = sanitized.e_col_kev

    col_low = decode_collimated_projection(geom, ix, iy, ec, energy_max_kev=240.0)
    col_all = decode_collimated_projection(geom, ix, iy, ec, energy_max_kev=None)
    n_low = int(np.sum(ec <= 240.0)) if ec.size > 0 else 0
    col_stat_plane = np.full((h, h), min(np.log1p(n_low) / 9.0, 1.0), dtype=np.float32)
    x_col = np.stack([col_low, col_all, col_stat_plane], axis=0)

    sbp, sbp_sq = compton_simple_backprojection(
        geom,
        sanitized.r1_comp_mm,
        sanitized.e1_comp_kev,
        sanitized.r2_comp_mm,
        sanitized.e2_comp_kev,
    )
    n_cmp = sanitized.r1_comp_mm.shape[0]
    cmp_stat_plane = np.full((h, h), min(np.log1p(n_cmp) / 6.5, 1.0), dtype=np.float32)
    x_comp = np.stack([sbp, sbp_sq, cmp_stat_plane], axis=0)

    return x_col, x_comp


class DualGammaImageDataset(Dataset):
    def __init__(
        self,
        x_col: np.ndarray,
        x_comp: np.ndarray,
        y_gt: np.ndarray,
        augment: bool = False,
    ):
        self.x_col = x_col.astype(np.float32)
        self.x_comp = x_comp.astype(np.float32)
        self.y_gt = y_gt[:, None, :, :].astype(np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return self.y_gt.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xc = self.x_col[idx]
        xp = self.x_comp[idx]
        yg = self.y_gt[idx]

        if self.augment:
            # Random 90-deg rotations & flips preserve isotropic FOV physics!
            k = np.random.randint(0, 4)
            if k > 0:
                xc = np.rot90(xc, k, axes=(1, 2)).copy()
                xp = np.rot90(xp, k, axes=(1, 2)).copy()
                yg = np.rot90(yg, k, axes=(1, 2)).copy()
            if np.random.rand() < 0.5:
                xc = np.flip(xc, axis=2).copy()
                xp = np.flip(xp, axis=2).copy()
                yg = np.flip(yg, axis=2).copy()

        return torch.from_numpy(xc), torch.from_numpy(xp), torch.from_numpy(yg)
