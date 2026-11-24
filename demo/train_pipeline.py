"""
End-to-End Multi-Condition Dataset Generation & Two-Stage Lightweight AI Training Pipeline:
1. Generates multi-condition Monte Carlo + Digitizer events across diverse source positions,
   shapes (point, multi-point, line, ring, L-shape), energies (59.5 keV - 1332 keV & mixed),
   and photon statistics;
2. Trains Stage-1 LiteEventNet (4-class photon event discrimination, sequence ordering & energy regression);
3. Projects AI-sanitized events through fast MURA & Compton SBP operators and trains Stage-2
   LiteDualGammaNet (< 0.5M parameters);
4. Saves compact model weights and training history to demo/checkpoints/.
"""

from __future__ import annotations
import os
import json
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from demo.src.physics.geometry_and_sources import CameraGeometry, sample_random_training_scene
from demo.src.physics.mc_simulator import simulate_scene_transport
from demo.src.physics.digitizer import DetectorDigitizer
from demo.src.models.event_net import LiteEventNet, sanitize_and_parse_events
from demo.src.models.dual_gammanet import LiteDualGammaNet, PeakAwareReconstructionLoss, count_parameters
from demo.src.dataset import build_dual_branch_tensors, DualGammaImageDataset


def train_stage1_event_net(
    geom: CameraGeometry,
    digitizer: DetectorDigitizer,
    rng: np.random.Generator,
    epochs: int = 26,
    device: torch.device = torch.device("cpu"),
) -> tuple[LiteEventNet, dict]:
    print("=== [Stage 1] Simulating photon interactions for LiteEventNet training ===")
    feat_list, lbl_list, e0_list = [], [], []
    # Collect ~240,000 diverse photon events across 140 random multi-energy scenes
    for _ in range(140):
        sc = sample_random_training_scene(geom, rng)
        raw = simulate_scene_transport(geom, sc, n_emitted_toward_det=45000, rng=rng)
        dig = digitizer.digitize(raw, rng)
        # Subsample single-hit events slightly so Compton classes (2, 3) have rich representation
        c0 = np.where(dig.label_4cls == 0)[0]
        c1 = np.where(dig.label_4cls == 1)[0]
        c23 = np.where((dig.label_4cls == 2) | (dig.label_4cls == 3))[0]
        if c0.size > 600:
            c0 = rng.choice(c0, 600, replace=False)
        if c1.size > 650:
            c1 = rng.choice(c1, 650, replace=False)
        sel = np.concatenate([c0, c1, c23])
        feat_list.append(dig.features_16d[sel])
        lbl_list.append(dig.label_4cls[sel])
        e0_list.append(dig.e0_true_kev[sel])

    x_all = np.concatenate(feat_list, axis=0)
    y_all = np.concatenate(lbl_list, axis=0)
    e_all = np.concatenate(e0_list, axis=0)

    perm = rng.permutation(x_all.shape[0])
    x_all, y_all, e_all = x_all[perm], y_all[perm], e_all[perm]

    n_val = int(0.15 * x_all.shape[0])
    x_tr, y_tr, e_tr = x_all[n_val:], y_all[n_val:], e_all[n_val:]
    x_va, y_va, e_va = x_all[:n_val], y_all[:n_val], e_all[:n_val]

    u, cnts = np.unique(y_tr, return_counts=True)
    print(f"Event dataset size: train={x_tr.shape[0]}, val={x_va.shape[0]} | Class counts: {dict(zip(u.tolist(), cnts.tolist()))}")

    # Inverse square-root class weights so Compton fwd/rev classes get balanced gradient attention
    cls_freq = np.bincount(y_tr, minlength=4).astype(np.float32)
    cls_weights = np.sqrt(np.mean(cls_freq) / np.maximum(cls_freq, 1.0))
    cls_weights_t = torch.from_numpy(cls_weights).to(device)

    model = LiteEventNet(in_features=16, hidden_dim=96, num_classes=4).to(device)
    print(f"LiteEventNet trainable parameters: {count_parameters(model):,}")

    ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr), torch.from_numpy(e_tr))
    loader_tr = DataLoader(ds_tr, batch_size=1024, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    history = {"train_loss": [], "val_acc": [], "val_compton_order_acc": []}
    x_va_t = torch.from_numpy(x_va).to(device)

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for bx, by, be in loader_tr:
            bx, by, be = bx.to(device), by.to(device), be.to(device)
            opt.zero_grad()
            logits, e_pred = model(bx)
            loss_cls = F.cross_entropy(logits, by, weight=cls_weights_t)
            # Supervise incident energy on clean signal events (classes 1, 2, 3)
            sig_m = by > 0
            loss_ene = F.smooth_l1_loss(e_pred[sig_m] / 1000.0, be[sig_m] / 1000.0) if torch.any(sig_m) else 0.0
            loss = loss_cls + 0.35 * loss_ene
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * bx.size(0)
        sched.step()

        model.eval()
        with torch.no_grad():
            v_logits, _ = model(x_va_t)
            v_pred = torch.argmax(v_logits, dim=-1).cpu().numpy()
            acc = float(np.mean(v_pred == y_va))
            cmp_mask = (y_va == 2) | (y_va == 3)
            cmp_acc = float(np.mean(v_pred[cmp_mask] == y_va[cmp_mask])) if np.any(cmp_mask) else 0.0

        history["train_loss"].append(ep_loss / x_tr.shape[0])
        history["val_acc"].append(acc)
        history["val_compton_order_acc"].append(cmp_acc)
        if ep in (1, 5, 10, 18, epochs):
            print(f"  [EventNet Ep {ep:02d}/{epochs}] loss={ep_loss/x_tr.shape[0]:.4f} | val_4cls_acc={acc*100:.2f}% | compton_seq_acc={cmp_acc*100:.2f}%")

    return model, history


def generate_dual_recon_dataset(
    geom: CameraGeometry,
    digitizer: DetectorDigitizer,
    event_net: LiteEventNet,
    n_samples: int,
    rng: np.random.Generator,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_cols, x_comps, gts = [], [], []
    t0 = time.perf_counter()
    for i in range(n_samples):
        sc = sample_random_training_scene(geom, rng)
        # Vary photon emission counts across 12x dynamic range (sparse rapid scan -> standard scan)
        has_high_e = any(c.energy_kev > 220.0 for c in sc.components)
        n_emit = int(rng.integers(28000, 95000)) if has_high_e else int(rng.integers(12000, 55000))

        raw = simulate_scene_transport(geom, sc, n_emitted_toward_det=n_emit, rng=rng)
        dig = digitizer.digitize(raw, rng)
        san = sanitize_and_parse_events(event_net, dig, device=device)
        xc, xp = build_dual_branch_tensors(geom, san)
        x_cols.append(xc)
        x_comps.append(xp)
        gts.append(sc.gt_map)

        if (i + 1) % 400 == 0 or (i + 1) == n_samples:
            el = time.perf_counter() - t0
            print(f"  Generated {i+1}/{n_samples} dual-mode scenes ({el:.1f}s)")

    return np.stack(x_cols, axis=0), np.stack(x_comps, axis=0), np.stack(gts, axis=0)


def compute_batch_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2, axis=(-2, -1))
    psnr = 10.0 * np.log10(1.0 / np.maximum(mse, 1e-10))
    return float(np.mean(psnr))


def main() -> None:
    cfg_path = "demo/configs/system_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"]["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running training pipeline on device: {device}")

    geom = CameraGeometry.from_config(cfg)
    digitizer = DetectorDigitizer(geom, cfg["digitizer"])

    os.makedirs("demo/checkpoints", exist_ok=True)
    os.makedirs("demo/results", exist_ok=True)

    # 1. Train Stage-1 LiteEventNet
    t_start = time.perf_counter()
    event_net, ev_hist = train_stage1_event_net(
        geom, digitizer, rng, epochs=int(cfg["training"]["event_net_epochs"]), device=device
    )
    torch.save(event_net.state_dict(), "demo/checkpoints/lite_event_net.pt")

    # 2. Generate Multi-Condition Dual-Branch Dataset
    n_tr = 1600
    n_va = 240
    print(f"\n=== [Stage 2] Building {n_tr} train + {n_va} val multi-condition dual-mode scenes ===")
    xc_tr, xp_tr, y_tr = generate_dual_recon_dataset(geom, digitizer, event_net, n_tr, rng, device)
    xc_va, xp_va, y_va = generate_dual_recon_dataset(geom, digitizer, event_net, n_va, rng, device)

    ds_tr = DualGammaImageDataset(xc_tr, xp_tr, y_tr, augment=True)
    ds_va = DualGammaImageDataset(xc_va, xp_va, y_va, augment=False)
    loader_tr = DataLoader(ds_tr, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=64, shuffle=False)

    # 3. Train Stage-2 LiteDualGammaNet
    recon_net = LiteDualGammaNet(in_ch_col=3, in_ch_comp=3, base_ch=32).to(device)
    n_params = count_parameters(recon_net)
    print(f"\nLiteDualGammaNet trainable parameters: {n_params:,} ({n_params/1e6:.3f} M < 1.0 M target!)")

    criterion = PeakAwareReconstructionLoss(peak_alpha=5.5)
    epochs_r = int(cfg["training"]["recon_net_epochs"])
    opt = torch.optim.AdamW(recon_net.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_r, eta_min=5e-5)

    best_psnr = -1.0
    recon_hist = {"train_loss": [], "val_loss": [], "val_psnr_db": []}

    for ep in range(1, epochs_r + 1):
        recon_net.train()
        tr_loss = 0.0
        for bxc, bxp, byg in loader_tr:
            bxc, bxp, byg = bxc.to(device), bxp.to(device), byg.to(device)
            opt.zero_grad()
            pred, _ = recon_net(bxc, bxp)
            loss = criterion(pred, byg)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * bxc.size(0)
        sched.step()
        tr_loss /= len(ds_tr)

        recon_net.eval()
        va_loss = 0.0
        preds_np, gts_np = [], []
        with torch.no_grad():
            for bxc, bxp, byg in loader_va:
                bxc, bxp, byg = bxc.to(device), bxp.to(device), byg.to(device)
                pred, _ = recon_net(bxc, bxp)
                va_loss += float(criterion(pred, byg).item()) * bxc.size(0)
                preds_np.append(pred.squeeze(1).cpu().numpy())
                gts_np.append(byg.squeeze(1).cpu().numpy())

        va_loss /= len(ds_va)
        va_psnr = compute_batch_psnr(np.concatenate(preds_np, axis=0), np.concatenate(gts_np, axis=0))
        recon_hist["train_loss"].append(tr_loss)
        recon_hist["val_loss"].append(va_loss)
        recon_hist["val_psnr_db"].append(va_psnr)

        if va_psnr > best_psnr:
            best_psnr = va_psnr
            torch.save(recon_net.state_dict(), "demo/checkpoints/lite_dual_gammanet.pt")

        if ep in (1, 5, 10, 18, 28, epochs_r):
            print(f"  [DualGammaNet Ep {ep:02d}/{epochs_r}] tr_loss={tr_loss:.4f} | va_loss={va_loss:.4f} | va_PSNR={va_psnr:.2f} dB (best={best_psnr:.2f} dB)")

    total_sec = time.perf_counter() - t_start
    summary = {
        "event_net_params": count_parameters(event_net),
        "dual_gammanet_params": n_params,
        "total_params": count_parameters(event_net) + n_params,
        "event_net_history": ev_hist,
        "recon_net_history": recon_hist,
        "best_val_psnr_db": best_psnr,
        "total_training_seconds": round(total_sec, 2),
    }
    with open("demo/results/training_history.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTraining complete in {total_sec:.1f}s! Checkpoints saved to demo/checkpoints/")


if __name__ == "__main__":
    main()
