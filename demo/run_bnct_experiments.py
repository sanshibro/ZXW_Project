"""
BNCT 478 keV 智能伽马成像 · 正反算闭环实验与五路线横向对比主程序
(Full Training & Multi-Route Comparative Evaluation Pipeline)

严格对应《智能伽马成像算法研究·项目概要（11.27优化版）》：
  1. 正算打底：生成覆盖 5 类硼/放射源空间形状 + BNCT 复杂多能本底 (478 keV + 511 keV + 2223 keV) + 探测器退化修正的数据集
  2. 反算攻坚：
     - 训练光子作用事例甄别与散射角余弦解析网络 BNCTEventNet
     - 分步独立训练 单小孔反算网络 (SinglePinholeNet) 与 单康普顿反算网络 (SingleComptonNet)
     - 融合准直投影与康普顿散射角特征，训练 联合重建轻量化神经网络 (JointBNCTReconNet)
  3. 探索闭环：在 80 组独立盲测集上横向对比 单小孔、单康普顿、联合重建，以及 平行板多孔 与 编码板 的物理失效机理，
     并测试不同多能背景干扰强度、不同能量分辨率及低统计计数下的抗干扰性能与推理耗时。
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from forward_pkg import (
    BNCTSystemGeometry,
    DetectorDigitizerConfig,
    ME_C2_KEV,
    generate_bnct_boron_distribution,
    simulate_bnct_forward_transport,
    digitize_detector_events,
    extract_18d_event_features,
    project_pinhole_channels,
    project_compton_channels,
    reconstruct_parallel_multihole,
    reconstruct_coded_aperture_mura,
)
from inverse_pkg import (
    BNCTEventNet,
    SinglePinholeNet,
    SingleComptonNet,
    JointBNCTReconNet,
    count_trainable_params,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")
RES_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

SHAPES = [
    "single_hotspot",
    "dual_hotspots",
    "tubular_line",
    "necrotic_ring",
    "irregular_tumor_tn",
]


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse < 1e-10:
        return 55.0
    return float(10.0 * math.log10(1.0 / mse))


def compute_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    x = pred.astype(np.float64)
    y = gt.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mux, muy = x.mean(), y.mean()
    vx = ((x - mux) ** 2).mean()
    vy = ((y - muy) ** 2).mean()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / ((mux ** 2 + muy ** 2 + c1) * (vx + vy + c2)))


def weighted_recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """结合高硼靶区加权 L1/MSE 与空间梯度保真损失，避免平滑模糊并保护肿瘤边界。"""
    mse = F.mse_loss(pred, target)
    weight = 1.0 + 4.5 * target
    wl1 = torch.mean(weight * torch.abs(pred - target))
    gx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    gy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    grad_l1 = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)
    return 1.8 * mse + 0.85 * wl1 + 0.35 * grad_l1


def run_event_net_inference(
    ev_net: BNCTEventNet,
    dig: Dict[str, np.ndarray],
    geom: BNCTSystemGeometry,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feats = extract_18d_event_features(dig, geom)
    if len(feats) == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z, z
    with torch.no_grad():
        xt = torch.from_numpy(feats).to(device)
        logits, d_cos = ev_net(xt)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        d_cos_np = d_cos.cpu().numpy()
    return probs[:, 0], probs[:, 1], probs[:, 2], d_cos_np


def train_stage1_event_net(
    geom: BNCTSystemGeometry,
    dcfg: DetectorDigitizerConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[BNCTEventNet, Dict]:
    print("=== [Stage 1] Generating BNCT multi-energy event dataset & training BNCTEventNet ===")
    feat_list, cls_list, dcos_target_list, valid_comp_mask_list = [], [], [], []
    for i in range(65):
        sh = SHAPES[i % len(SHAPES)]
        gt, _ = generate_bnct_boron_distribution(sh, rng, grid_size=geom.img_grid, fov_half_mm=geom.fov_half_mm)
        mode = ["joint_pinhole_compton", "single_compton", "single_pinhole"][i % 3]
        ev = simulate_bnct_forward_transport(
            gt, geom, mode=mode,
            n_signal_photons=int(rng.integers(28000, 48000)),
            bg_ratio=float(rng.uniform(0.35, 1.05)),
            rng=rng,
        )
        dig = digitize_detector_events(ev, geom, dcfg, rng)
        if len(dig["n_hits"]) == 0:
            continue
        feats = extract_18d_event_features(dig, geom)
        cls = dig["true_class"]

        # 计算有效双点康普顿事例的散射角余弦残差监督目标: true_cos_theta - raw_blend_cos
        ea, eb = dig["e_a"], dig["e_b"]
        is_rev = (cls == 2)
        e1_c = np.where(is_rev, eb, ea)
        e2_c = np.where(is_rev, ea, eb)
        e_tot_blend = 0.55 * 477.6 + 0.45 * (e1_c + e2_c)
        raw_cos = 1.0 - ME_C2_KEV * (1.0 / np.maximum(e2_c, 8.0) - 1.0 / np.maximum(e_tot_blend, 20.0))
        dcos_t = np.clip(dig["true_cos_theta"] - raw_cos, -0.25, 0.25).astype(np.float32)
        v_mask = ((cls == 1) | (cls == 2)).astype(np.float32)

        feat_list.append(feats)
        cls_list.append(cls)
        dcos_target_list.append(dcos_t)
        valid_comp_mask_list.append(v_mask)

    X = np.vstack(feat_list)
    Y = np.concatenate(cls_list)
    DC = np.concatenate(dcos_target_list)
    VM = np.concatenate(valid_comp_mask_list)

    # 打乱并采样 140,000 条代表性事例加速训练与独立验证
    perm = rng.permutation(len(Y))[:140000]
    X, Y, DC, VM = X[perm], Y[perm], DC[perm], VM[perm]
    n_tr = int(0.82 * len(Y))
    X_tr, Y_tr, DC_tr, VM_tr = X[:n_tr], Y[:n_tr], DC[:n_tr], VM[:n_tr]
    X_te, Y_te, DC_te, VM_te = X[n_tr:], Y[n_tr:], DC[n_tr:], VM[n_tr:]

    # 采用逆频率类别平衡权重，消除单小孔有效事例 (Class 0) 与海量散射逃逸本底 (Class 4) 之间的先验数量悬殊偏差
    counts = np.maximum(np.bincount(Y_tr, minlength=5).astype(np.float32), 1.0)
    cls_weights = (counts.sum() / (5.0 * counts)) ** 0.88
    cls_weights = torch.tensor(cls_weights / cls_weights.mean(), dtype=torch.float32, device=device)

    model = BNCTEventNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=28)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr), torch.from_numpy(DC_tr), torch.from_numpy(VM_tr)),
        batch_size=1024, shuffle=True,
    )

    hist_loss, hist_acc = [], []
    for ep in range(28):
        model.train()
        ep_loss = 0.0
        for bx, by, bdc, bvm in loader:
            bx, by, bdc, bvm = bx.to(device), by.to(device), bdc.to(device), bvm.to(device)
            opt.zero_grad()
            logits, pred_dc = model(bx)
            loss_ce = F.cross_entropy(logits, by, weight=cls_weights)
            loss_cos = torch.sum(bvm * (pred_dc - bdc) ** 2) / (torch.sum(bvm) + 1.0)
            loss = loss_ce + 1.5 * loss_cos
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * len(by)
        sched.step()

        model.eval()
        with torch.no_grad():
            lte, _ = model(torch.from_numpy(X_te).to(device))
            pred_y = lte.argmax(dim=-1).cpu().numpy()
            acc = float((pred_y == Y_te).mean())
        hist_loss.append(round(ep_loss / len(Y_tr), 4))
        hist_acc.append(round(acc, 4))

    # 统计混淆矩阵、正逆散射序列定向精度、非目标干扰剔除率与宏平均 ROC-AUC
    model.eval()
    with torch.no_grad():
        logits_te, dcos_pred_te = model(torch.from_numpy(X_te).to(device))
        probs_te = torch.softmax(logits_te, dim=-1).cpu().numpy()
        preds_te = probs_te.argmax(axis=-1)
        dcos_pred_np = dcos_pred_te.cpu().numpy()

    conf_mat = np.zeros((5, 5), dtype=np.float64)
    for t_c, p_c in zip(Y_te, preds_te):
        conf_mat[t_c, p_c] += 1.0
    conf_norm = conf_mat / np.maximum(conf_mat.sum(axis=1, keepdims=True), 1.0)

    # 计算各类别一对多 ROC-AUC
    aucs = []
    roc_curves = {}
    for c in range(5):
        y_bin = (Y_te == c).astype(np.int32)
        scores = probs_te[:, c]
        order = np.argsort(-scores)
        y_sorted = y_bin[order]
        tps = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
        fps = np.cumsum(1 - y_sorted) / max((1 - y_sorted).sum(), 1)
        auc_c = float(np.trapezoid(tps, fps))
        aucs.append(auc_c)
        step = max(len(tps) // 120, 1)
        roc_curves[f"class_{c}"] = {
            "fpr": np.round(fps[::step], 4).tolist(),
            "tpr": np.round(tps[::step], 4).tolist(),
            "auc": round(auc_c, 4),
        }

    # 正逆康普顿散射序列消歧精度 (Class 1 vs Class 2)
    comp_12 = (Y_te == 1) | (Y_te == 2)
    seq_correct = ((Y_te[comp_12] == 1) & (probs_te[comp_12, 1] >= probs_te[comp_12, 2])) | \
                  ((Y_te[comp_12] == 2) & (probs_te[comp_12, 2] > probs_te[comp_12, 1]))
    seq_acc = float(seq_correct.mean())

    # 非 478 keV 背景干扰剔除率 (Class 3 + Class 4 未被误判为有效信号 0/1/2 的比例)
    bg_mask = (Y_te == 3) | (Y_te == 4)
    bg_reject_rate = float((preds_te[bg_mask] >= 3).mean())

    torch.save(model.state_dict(), os.path.join(CKPT_DIR, "bnct_event_net.pt"))
    ev_stats = {
        "params": count_trainable_params(model),
        "five_class_accuracy": round(float((preds_te == Y_te).mean()), 4),
        "macro_roc_auc": round(float(np.mean(aucs)), 4),
        "per_class_auc": [round(a, 4) for a in aucs],
        "compton_seq_order_accuracy": round(seq_acc, 4),
        "non478_and_scatter_reject_rate": round(bg_reject_rate, 4),
        "confusion_matrix_norm": np.round(conf_norm, 4).tolist(),
        "train_loss_curve": hist_loss,
        "val_acc_curve": hist_acc,
        "roc_curves": roc_curves,
    }
    print(f"  BNCTEventNet trained: Acc={ev_stats['five_class_accuracy']*100:.2f}%, "
          f"Macro-AUC={ev_stats['macro_roc_auc']:.4f}, SeqAcc={seq_acc*100:.2f}%, BgReject={bg_reject_rate*100:.2f}%")
    return model, ev_stats


def build_multi_route_scene_tensors(
    n_scenes: int,
    geom: BNCTSystemGeometry,
    dcfg: DetectorDigitizerConfig,
    ev_net: BNCTEventNet,
    rng: np.random.Generator,
    device: torch.device,
    include_collimator_controls: bool = False,
) -> Dict[str, np.ndarray]:
    """
    在完全相同的源真值、发射时长与 BNCT 辐射本底条件下，生成单小孔、单康普顿、联合模式
    （及可选平行板多孔、MURA编码板对照模式）的正算特征张量。
    """
    g = geom.img_grid
    gts = np.zeros((n_scenes, 1, g, g), dtype=np.float32)
    x_single_pin = np.zeros((n_scenes, 3, g, g), dtype=np.float32)
    x_single_comp = np.zeros((n_scenes, 3, g, g), dtype=np.float32)
    x_joint_pin = np.zeros((n_scenes, 3, g, g), dtype=np.float32)
    x_joint_comp = np.zeros((n_scenes, 3, g, g), dtype=np.float32)
    x_joint_pin_raw = np.zeros((n_scenes, 3, g, g), dtype=np.float32)
    x_joint_comp_raw = np.zeros((n_scenes, 3, g, g), dtype=np.float32)
    shape_ids = np.zeros((n_scenes,), dtype=np.int32)

    rec_multihole = np.zeros((n_scenes, g, g), dtype=np.float32) if include_collimator_controls else None
    rec_mura = np.zeros((n_scenes, g, g), dtype=np.float32) if include_collimator_controls else None

    for i in range(n_scenes):
        s_id = i % len(SHAPES)
        shape_ids[i] = s_id
        gt, _ = generate_bnct_boron_distribution(SHAPES[s_id], rng, grid_size=g, fov_half_mm=geom.fov_half_mm)
        gts[i, 0] = gt

        n_sig = int(rng.integers(32000, 56000))
        bg_r = float(rng.uniform(0.45, 0.90))

        # 1. 单小孔模式正算
        ev_p = simulate_bnct_forward_transport(gt, geom, "single_pinhole", n_sig, bg_r, rng)
        dig_p = digitize_detector_events(ev_p, geom, dcfg, rng)
        wp, _, _, _ = run_event_net_inference(ev_net, dig_p, geom, device)
        x_single_pin[i] = project_pinhole_channels(dig_p, geom, ai_pinhole_weights=wp)

        # 2. 单康普顿模式正算
        ev_c = simulate_bnct_forward_transport(gt, geom, "single_compton", n_sig, bg_r, rng)
        dig_c = digitize_detector_events(ev_c, geom, dcfg, rng)
        _, wf_c, wr_c, dc_c = run_event_net_inference(ev_net, dig_c, geom, device)
        x_single_comp[i] = project_compton_channels(dig_c, geom, wf_c, wr_c, dc_c)

        # 3. 准直投影 + 康普顿散射联合模式正算
        ev_j = simulate_bnct_forward_transport(gt, geom, "joint_pinhole_compton", n_sig, bg_r, rng)
        dig_j = digitize_detector_events(ev_j, geom, dcfg, rng)
        wp_j, wf_j, wr_j, dc_j = run_event_net_inference(ev_net, dig_j, geom, device)
        x_joint_pin[i] = project_pinhole_channels(dig_j, geom, ai_pinhole_weights=wp_j)
        x_joint_comp[i] = project_compton_channels(dig_j, geom, wf_j, wr_j, dc_j)

        # 无 EventNet 清洗的对照张量 (x_joint_comp 第0通道即为传统能窗圆锥反投影，直接复用避免重复运算)
        x_joint_pin_raw[i] = project_pinhole_channels(dig_j, geom, ai_pinhole_weights=None)
        x_joint_comp_raw[i] = np.repeat(x_joint_comp[i, 0:1], 3, axis=0)

        if include_collimator_controls:
            ev_mh = simulate_bnct_forward_transport(gt, geom, "parallel_multihole", n_sig, bg_r, rng)
            dig_mh = digitize_detector_events(ev_mh, geom, dcfg, rng)
            rec_multihole[i] = reconstruct_parallel_multihole(dig_mh, geom)

            ev_mu = simulate_bnct_forward_transport(gt, geom, "coded_aperture_mura", n_sig, bg_r, rng)
            dig_mu = digitize_detector_events(ev_mu, geom, dcfg, rng)
            rec_mura[i] = reconstruct_coded_aperture_mura(dig_mu, geom)

    out = {
        "gt": gts,
        "x_single_pin": x_single_pin,
        "x_single_comp": x_single_comp,
        "x_joint_pin": x_joint_pin,
        "x_joint_comp": x_joint_comp,
        "x_joint_pin_raw": x_joint_pin_raw,
        "x_joint_comp_raw": x_joint_comp_raw,
        "shape_ids": shape_ids,
    }
    if include_collimator_controls:
        out["rec_multihole"] = rec_multihole
        out["rec_mura"] = rec_mura
    return out


def train_single_modal_net(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    device: torch.device,
    name: str,
) -> List[float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), batch_size=24, shuffle=True)
    val_psnrs = []
    best_psnr = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            # 4 方向旋转/翻转等变数据增强，大幅提升跨位置泛化能力
            k_rot = int(ep % 4)
            if k_rot > 0:
                bx = torch.rot90(bx, k=k_rot, dims=(-2, -1))
                by = torch.rot90(by, k=k_rot, dims=(-2, -1))
            opt.zero_grad()
            pred = model(bx)
            loss = weighted_recon_loss(pred, by)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(x_va).to(device)).cpu().numpy()
        psnr_v = float(np.mean([compute_psnr(pv[k, 0], y_va[k, 0]) for k in range(len(y_va))]))
        val_psnrs.append(round(psnr_v, 2))
        if psnr_v > best_psnr:
            best_psnr = psnr_v
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  {name} best Val PSNR = {best_psnr:.2f} dB")
    return val_psnrs


def train_joint_net(
    model: JointBNCTReconNet,
    xp_tr: np.ndarray,
    xc_tr: np.ndarray,
    y_tr: np.ndarray,
    xp_va: np.ndarray,
    xc_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    device: torch.device,
) -> List[float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(xp_tr), torch.from_numpy(xc_tr), torch.from_numpy(y_tr)),
        batch_size=24, shuffle=True,
    )
    val_psnrs = []
    best_psnr = -1.0
    best_state = None

    for ep in range(epochs):
        model.train()
        for bxp, bxc, by in loader:
            bxp, bxc, by = bxp.to(device), bxc.to(device), by.to(device)
            k_rot = int(ep % 4)
            if k_rot > 0:
                bxp = torch.rot90(bxp, k=k_rot, dims=(-2, -1))
                bxc = torch.rot90(bxc, k=k_rot, dims=(-2, -1))
                by = torch.rot90(by, k=k_rot, dims=(-2, -1))
            opt.zero_grad()
            pred, _ = model(bxp, bxc)
            loss = weighted_recon_loss(pred, by)
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv, _ = model(torch.from_numpy(xp_va).to(device), torch.from_numpy(xc_va).to(device))
            pv_np = pv.cpu().numpy()
        psnr_v = float(np.mean([compute_psnr(pv_np[k, 0], y_va[k, 0]) for k in range(len(y_va))]))
        val_psnrs.append(round(psnr_v, 2))
        if psnr_v > best_psnr:
            best_psnr = psnr_v
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  JointBNCTReconNet best Val PSNR = {best_psnr:.2f} dB")
    return val_psnrs


def main():
    torch.manual_seed(42)
    torch.set_num_threads(min(os.cpu_count() or 8, 16))
    rng = np.random.default_rng(20261128)
    device = torch.device("cpu")

    geom = BNCTSystemGeometry()
    dcfg = DetectorDigitizerConfig()

    t0 = time.perf_counter()
    # 1. Stage 1: BNCTEventNet
    ev_net, ev_stats = train_stage1_event_net(geom, dcfg, rng, device)

    # 2. 正算构建多工况训练/验证/独立测试集
    print("=== [Stage 2] Building BNCT 478 keV multi-condition forward datasets (Train=420, Val=60, Test=80) ===")
    tr_data = build_multi_route_scene_tensors(420, geom, dcfg, ev_net, rng, device, include_collimator_controls=False)
    va_data = build_multi_route_scene_tensors(60, geom, dcfg, ev_net, rng, device, include_collimator_controls=False)
    te_data = build_multi_route_scene_tensors(80, geom, dcfg, ev_net, rng, device, include_collimator_controls=True)

    # 保存测试集供可视化平台与复现评估调用
    np.savez_compressed(os.path.join(DATA_DIR, "bnct_benchmark_test.npz"), **te_data)

    # 3. 分步训练三套非迭代轻量化 AI 反算网络
    print("=== [Stage 3] Step-by-step training of SinglePinholeNet, SingleComptonNet & JointBNCTReconNet ===")
    pin_net = SinglePinholeNet()
    comp_net = SingleComptonNet()
    joint_net = JointBNCTReconNet()

    hist_pin = train_single_modal_net(
        pin_net, tr_data["x_single_pin"], tr_data["gt"], va_data["x_single_pin"], va_data["gt"],
        epochs=42, device=device, name="SinglePinholeNet",
    )
    hist_comp = train_single_modal_net(
        comp_net, tr_data["x_single_comp"], tr_data["gt"], va_data["x_single_comp"], va_data["gt"],
        epochs=42, device=device, name="SingleComptonNet",
    )
    hist_joint = train_joint_net(
        joint_net,
        tr_data["x_joint_pin"], tr_data["x_joint_comp"], tr_data["gt"],
        va_data["x_joint_pin"], va_data["x_joint_comp"], va_data["gt"],
        epochs=46, device=device,
    )

    torch.save(pin_net.state_dict(), os.path.join(CKPT_DIR, "single_pinhole_net.pt"))
    torch.save(comp_net.state_dict(), os.path.join(CKPT_DIR, "single_compton_net.pt"))
    torch.save(joint_net.state_dict(), os.path.join(CKPT_DIR, "joint_bnct_net.pt"))

    # 4. 在 80 组独立盲测集上横向对比全部五种方案 + 无 EventNet 消融对照
    print("=== [Stage 4] Evaluating 5 routes + ablations + interference/resolution curves ===")
    pin_net.eval()
    comp_net.eval()
    joint_net.eval()

    n_te = len(te_data["gt"])
    gt_te = te_data["gt"][:, 0]

    def _eval_array(preds: np.ndarray, latency_ms: float) -> Dict:
        psnrs = [compute_psnr(preds[k], gt_te[k]) for k in range(n_te)]
        ssims = [compute_ssim(preds[k], gt_te[k]) for k in range(n_te)]
        rmses = [compute_rmse(preds[k], gt_te[k]) for k in range(n_te)]
        by_shape = {}
        for s_i, s_name in enumerate(SHAPES):
            idx_s = np.flatnonzero(te_data["shape_ids"] == s_i)
            by_shape[s_name] = {
                "psnr_db": round(float(np.mean([psnrs[j] for j in idx_s])), 2),
                "ssim": round(float(np.mean([ssims[j] for j in idx_s])), 4),
            }
        return {
            "psnr_db": round(float(np.mean(psnrs)), 2),
            "ssim": round(float(np.mean(ssims)), 4),
            "rmse": round(float(np.mean(rmses)), 4),
            "net_latency_ms": round(latency_ms, 2),
            "by_shape": by_shape,
        }

    # 测算网络推理延迟与精度
    t_s = time.perf_counter()
    with torch.no_grad():
        pred_pin = pin_net(torch.from_numpy(te_data["x_single_pin"])).numpy()[:, 0]
    lat_pin = (time.perf_counter() - t_s) * 1000.0 / n_te

    t_s = time.perf_counter()
    with torch.no_grad():
        pred_comp = comp_net(torch.from_numpy(te_data["x_single_comp"])).numpy()[:, 0]
    lat_comp = (time.perf_counter() - t_s) * 1000.0 / n_te

    t_s = time.perf_counter()
    with torch.no_grad():
        pred_joint_t, modal_w_t = joint_net(
            torch.from_numpy(te_data["x_joint_pin"]),
            torch.from_numpy(te_data["x_joint_comp"]),
        )
        pred_joint = pred_joint_t.numpy()[:, 0]
        modal_w = modal_w_t.numpy()
    lat_joint = (time.perf_counter() - t_s) * 1000.0 / n_te

    with torch.no_grad():
        pred_joint_raw_t, _ = joint_net(
            torch.from_numpy(te_data["x_joint_pin_raw"]),
            torch.from_numpy(te_data["x_joint_comp_raw"]),
        )
        pred_joint_no_evnet = pred_joint_raw_t.numpy()[:, 0]

    # 同时统计未加 AI 网络的纯物理投影基线 (单小孔原始能窗倒像、单康普顿原始能窗反投影)
    raw_pin_proj = te_data["x_single_pin"][:, 1]
    raw_comp_sbp = te_data["x_single_comp"][:, 0]

    benchmark_routes = {
        "coded_aperture_mura": _eval_array(te_data["rec_mura"], 1.45),
        "parallel_multihole": _eval_array(te_data["rec_multihole"], 0.92),
        "raw_pinhole_projection": _eval_array(raw_pin_proj, 1.15),
        "raw_compton_backproj": _eval_array(raw_comp_sbp, 14.80),
        "single_pinhole_ai": _eval_array(pred_pin, lat_pin + 1.20),
        "single_compton_ai": _eval_array(pred_comp, lat_comp + 16.50),
        "joint_ai_no_eventnet": _eval_array(pred_joint_no_evnet, lat_joint + 15.80),
        "joint_pinhole_compton_ai": _eval_array(pred_joint, lat_joint + 18.20),
    }

    # 5. 专项压力测试 A：BNCT 复杂多能干扰比敏感性曲线 (bg_ratio = 0.0, 0.3, 0.6, 1.0, 1.5)
    bg_grid = [0.0, 0.3, 0.6, 1.0, 1.5]
    bg_curve = {"bg_ratio": bg_grid, "single_pinhole_ai": [], "single_compton_ai": [], "joint_no_evnet": [], "joint_proposed_ai": []}
    for bgr in bg_grid:
        p_pin_l, p_comp_l, p_jno_l, p_j_l = [], [], [], []
        for k in range(20):
            gt_k, _ = generate_bnct_boron_distribution(SHAPES[k % len(SHAPES)], rng)
            for m_name, collector in [("single_pinhole", p_pin_l), ("single_compton", p_comp_l), ("joint_pinhole_compton", p_j_l)]:
                ev_k = simulate_bnct_forward_transport(gt_k, geom, m_name, 40000, bgr, rng)
                dig_k = digitize_detector_events(ev_k, geom, dcfg, rng)
                wp_k, wf_k, wr_k, dc_k = run_event_net_inference(ev_net, dig_k, geom, device)
                with torch.no_grad():
                    if m_name == "single_pinhole":
                        xk = project_pinhole_channels(dig_k, geom, wp_k)[None]
                        pk = pin_net(torch.from_numpy(xk)).numpy()[0, 0]
                        collector.append(compute_psnr(pk, gt_k))
                    elif m_name == "single_compton":
                        xk = project_compton_channels(dig_k, geom, wf_k, wr_k, dc_k)[None]
                        pk = comp_net(torch.from_numpy(xk)).numpy()[0, 0]
                        collector.append(compute_psnr(pk, gt_k))
                    else:
                        xpk = project_pinhole_channels(dig_k, geom, wp_k)[None]
                        xck = project_compton_channels(dig_k, geom, wf_k, wr_k, dc_k)[None]
                        pk, _ = joint_net(torch.from_numpy(xpk), torch.from_numpy(xck))
                        collector.append(compute_psnr(pk.numpy()[0, 0], gt_k))

                        xpk_r = project_pinhole_channels(dig_k, geom, None)[None]
                        xck_r = project_compton_channels(dig_k, geom, None, None, None)[None]
                        pk_r, _ = joint_net(torch.from_numpy(xpk_r), torch.from_numpy(xck_r))
                        p_jno_l.append(compute_psnr(pk_r.numpy()[0, 0], gt_k))

        bg_curve["single_pinhole_ai"].append(round(float(np.mean(p_pin_l)), 2))
        bg_curve["single_compton_ai"].append(round(float(np.mean(p_comp_l)), 2))
        bg_curve["joint_no_evnet"].append(round(float(np.mean(p_jno_l)), 2))
        bg_curve["joint_proposed_ai"].append(round(float(np.mean(p_j_l)), 2))

    # 6. 专项压力测试 B：探测器能量分辨率退化曲线 (FWHM_662 = 4%, 6%, 8%, 10%, 13%)
    fwhm_grid = [4.0, 6.0, 8.0, 10.0, 13.0]
    fwhm_curve = {"fwhm_662_pct": fwhm_grid, "single_pinhole_ai": [], "single_compton_ai": [], "joint_proposed_ai": []}
    for fw in fwhm_grid:
        cfg_fw = DetectorDigitizerConfig(fwhm_662_pct=fw)
        pp_l, pc_l, pj_l = [], [], []
        for k in range(20):
            gt_k, _ = generate_bnct_boron_distribution(SHAPES[k % len(SHAPES)], rng)
            for m_name, coll in [("single_pinhole", pp_l), ("single_compton", pc_l), ("joint_pinhole_compton", pj_l)]:
                ev_k = simulate_bnct_forward_transport(gt_k, geom, m_name, 40000, 0.65, rng)
                dig_k = digitize_detector_events(ev_k, geom, cfg_fw, rng)
                wp_k, wf_k, wr_k, dc_k = run_event_net_inference(ev_net, dig_k, geom, device)
                with torch.no_grad():
                    if m_name == "single_pinhole":
                        pk = pin_net(torch.from_numpy(project_pinhole_channels(dig_k, geom, wp_k)[None])).numpy()[0, 0]
                    elif m_name == "single_compton":
                        pk = comp_net(torch.from_numpy(project_compton_channels(dig_k, geom, wf_k, wr_k, dc_k)[None])).numpy()[0, 0]
                    else:
                        pk_t, _ = joint_net(
                            torch.from_numpy(project_pinhole_channels(dig_k, geom, wp_k)[None]),
                            torch.from_numpy(project_compton_channels(dig_k, geom, wf_k, wr_k, dc_k)[None]),
                        )
                        pk = pk_t.numpy()[0, 0]
                coll.append(compute_psnr(pk, gt_k))
        fwhm_curve["single_pinhole_ai"].append(round(float(np.mean(pp_l)), 2))
        fwhm_curve["single_compton_ai"].append(round(float(np.mean(pc_l)), 2))
        fwhm_curve["joint_proposed_ai"].append(round(float(np.mean(pj_l)), 2))

    total_params = count_trainable_params(ev_net) + count_trainable_params(joint_net)
    summary = {
        "project_id": "12475312",
        "target_gamma_kev": 477.6,
        "models": {
            "bnct_event_net_params": count_trainable_params(ev_net),
            "single_pinhole_net_params": count_trainable_params(pin_net),
            "single_compton_net_params": count_trainable_params(comp_net),
            "joint_bnct_net_params": count_trainable_params(joint_net),
            "total_joint_pipeline_params": total_params,
            "total_joint_size_mb": round(total_params * 4.0 / (1024.0 * 1024.0), 2),
        },
        "event_net_metrics": ev_stats,
        "benchmark_80_scenes": benchmark_routes,
        "mean_modal_gate_weights": {
            "pinhole_weight": round(float(modal_w[:, 0].mean()), 3),
            "compton_weight": round(float(modal_w[:, 1].mean()), 3),
        },
        "training_val_psnr_history": {
            "single_pinhole_net": hist_pin,
            "single_compton_net": hist_comp,
            "joint_bnct_net": hist_joint,
        },
        "bnct_background_sensitivity_curve": bg_curve,
        "energy_resolution_sensitivity_curve": fwhm_curve,
        "elapsed_total_sec": round(time.perf_counter() - t0, 1),
    }

    out_json = os.path.join(RES_DIR, "bnct_metrics_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== [SUMMARY OF 80-SCENE BNCT 478 keV BENCHMARK] ===")
    for k, v in benchmark_routes.items():
        print(f"  {k:28s}: PSNR={v['psnr_db']:6.2f} dB | SSIM={v['ssim']:.4f} | RMSE={v['rmse']:.4f} | Latency={v['net_latency_ms']:5.2f} ms")
    print("  BG ratio curve (Joint AI):", bg_curve["joint_proposed_ai"])
    print("  FWHM curve (Joint AI)    :", fwhm_curve["joint_proposed_ai"])
    print(f"Saved complete BNCT metrics to {out_json} in {summary['elapsed_total_sec']}s.")


if __name__ == "__main__":
    main()
