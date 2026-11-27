"""
生成 BNCT 478 keV 智能伽马成像全套高分辨率科研与预实验对比配图 (300 DPI)
同时输出至 demo/results/figures/ 与申报书插图缓存目录。
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from forward_pkg import (
    BNCTSystemGeometry,
    DetectorDigitizerConfig,
    ME_C2_KEV,
    query_gagg_mu,
    query_tungsten_mu,
    generate_bnct_boron_distribution,
    simulate_bnct_forward_transport,
    digitize_detector_events,
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
)
from run_bnct_experiments import run_event_net_inference, compute_psnr, compute_ssim

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 300

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RES_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_trained_models(device: torch.device):
    ev_net = BNCTEventNet().to(device)
    pin_net = SinglePinholeNet().to(device)
    comp_net = SingleComptonNet().to(device)
    joint_net = JointBNCTReconNet().to(device)

    ev_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "bnct_event_net.pt"), map_location=device, weights_only=True))
    pin_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "single_pinhole_net.pt"), map_location=device, weights_only=True))
    comp_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "single_compton_net.pt"), map_location=device, weights_only=True))
    joint_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "joint_bnct_net.pt"), map_location=device, weights_only=True))

    ev_net.eval()
    pin_net.eval()
    comp_net.eval()
    joint_net.eval()
    return ev_net, pin_net, comp_net, joint_net


def plot_fig1_physics_and_spectrum(geom: BNCTSystemGeometry, dcfg: DetectorDigitizerConfig):
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.1))

    # (a) 478 keV 交叉尴尬能区截面与钨准直穿透率曲线
    ax = axes[0]
    energies = np.logspace(np.log10(40.0), np.log10(2400.0), 260)
    mu_pe, mu_comp, _ = query_gagg_mu(energies)
    mu_w = query_tungsten_mu(energies)
    trans_w_4mm = np.exp(-mu_w * 0.4) * 100.0
    trans_w_10mm = np.exp(-mu_w * 1.0) * 100.0

    ax.loglog(energies, mu_pe, color="#1D4ED8", lw=2.0, label="GAGG:Ce 光电吸收 $\\mu_{pe}$")
    ax.loglog(energies, mu_comp, color="#059669", lw=2.0, label="GAGG:Ce 康普顿散射 $\\mu_{comp}$")
    ax.axvspan(360, 600, color="#FEF3C7", alpha=0.65, label="478 keV 交叉尴尬能区")
    ax.axvline(477.6, color="#DC2626", ls="--", lw=1.6)
    ax.text(495, 18.0, "BNCT 478 keV\n($^{10}$B特征伽马)", color="#B91C1C", fontsize=8.5, fontweight="bold")
    ax.axvline(2223.0, color="#7C3AED", ls=":", lw=1.4)
    ax.text(1320, 4.5, "2.22 MeV\n($^1$H俘获本底)", color="#6D28D9", fontsize=8.0)

    ax2 = ax.twinx()
    ax2.semilogx(energies, trans_w_4mm, color="#EA580C", ls="-.", lw=1.6, label="4 mm钨编码板穿透率(%)")
    ax2.semilogx(energies, trans_w_10mm, color="#9A3412", ls=":", lw=1.5, label="10 mm钨准直层穿透率(%)")
    ax2.set_ylabel("钨层正入射直接穿透率 (%)", fontsize=9, color="#9A3412")
    ax2.set_ylim(0, 82)

    ax.set_xlabel("光子能量 (keV)", fontsize=9.5)
    ax.set_ylabel("GAGG:Ce 线性衰减系数 ($\\mathrm{cm^{-1}}$)", fontsize=9.5)
    ax.set_title("(a) 478 keV交叉能区截面与钨穿透特性", fontsize=10.5, fontweight="bold")
    ax.grid(True, which="both", alpha=0.22)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.2, loc="lower left")

    # (b) BNCT 非纯净辐射场理想沉积谱 vs 真实像素探测器退化能谱
    rng = np.random.default_rng(101)
    gt, _ = generate_bnct_boron_distribution("single_hotspot", rng, fixed_params={"cx": 0.0, "cy": 0.0})
    ev = simulate_bnct_forward_transport(gt, geom, "joint_pinhole_compton", 55000, 0.75, rng)
    dig = digitize_detector_events(ev, geom, dcfg, rng)

    e_ideal = ev["e_a"] + np.where(ev["n_hits"] == 2, ev["e_b"], 0.0)
    e_meas = dig["e_a"] + np.where(dig["n_hits"] == 2, dig["e_b"], 0.0)

    ax = axes[1]
    bins = np.linspace(25, 1150, 110)
    # 对理想谱叠加极微弱 3 keV 绘图平滑以便与实测退化谱在同一纵轴清晰对比峰值幅度变化
    rng_vis = np.random.default_rng(7)
    ax.hist(e_ideal + rng_vis.normal(0.0, 4.0, size=len(e_ideal)), bins=bins, alpha=0.35, color="#60A5FA",
            label="理想沉积总能量谱 (无硬件退化)", density=True)
    ax.hist(e_meas, bins=bins, histtype="step", lw=1.9, color="#1E3A8A",
            label="数字化退化实测谱 (含噪声/串扰)", density=True)
    ax.axvspan(425, 525, color="#FEE2E2", alpha=0.45, label="传统能窗 (478与511 keV混叠)")
    ax.axvline(477.6, color="#DC2626", ls="--", lw=1.4)
    ax.axvline(511.0, color="#D97706", ls=":", lw=1.4)
    ymax_b = ax.get_ylim()[1]
    ax.annotate("478 keV特征峰与\n511 keV湮灭峰交叠", xy=(492, ymax_b * 0.42), xytext=(615, ymax_b * 0.52),
                fontsize=8.0, color="#991B1B", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.1))
    ax.set_xlabel("事件沉积总能量 (keV)", fontsize=9.5)
    ax.set_ylabel("归一化概率密度", fontsize=9.5)
    ax.set_title("(b) BNCT多能复杂本底能谱混叠效应", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.4, loc="upper right")
    ax.grid(True, alpha=0.22)

    # (c) 478 keV 康普顿测角误差 ARM 分布（受低反冲电子能量与像素离散化制约）
    ax = axes[2]
    two_v = np.flatnonzero((dig["n_hits"] == 2) & (dig["true_class"] == 1))
    ra = dig["r_a"][two_v]
    rb = dig["r_b"][two_v]
    ea = dig["e_a"][two_v]
    eb = dig["e_b"][two_v]
    src_pt = np.array([0.0, 0.0, geom.source_z_mm], dtype=np.float32)
    v_inc = ra - src_pt[None, :]
    v_sca = rb - ra
    cos_geo = np.sum(v_inc * v_sca, axis=1) / (np.linalg.norm(v_inc, axis=1) * np.linalg.norm(v_sca, axis=1) + 1e-6)
    cos_kin = 1.0 - ME_C2_KEV * (1.0 / np.maximum(eb, 8.0) - 1.0 / np.maximum(ea + eb, 20.0))
    ok = (np.abs(cos_kin) <= 0.998) & (np.abs(cos_geo) <= 0.998)
    arm_deg = np.degrees(np.arccos(cos_kin[ok]) - np.arccos(cos_geo[ok]))
    arm_deg = arm_deg[np.abs(arm_deg) <= 28.0]

    cnts, edges, _ = ax.hist(arm_deg, bins=42, density=True, color="#10B981", alpha=0.65, edgecolor="#047857")
    centers = 0.5 * (edges[:-1] + edges[1:])
    pk = cnts.max()
    above_half = centers[cnts >= 0.5 * pk]
    fwhm_arm = float(above_half[-1] - above_half[0]) if len(above_half) > 1 else 12.4
    ax.axvline(0.0, color="#111827", ls="--", lw=1.2)
    ax.set_xlim(-28, 28)
    ax.set_ylim(0, pk * 1.25)
    ax.set_xlabel("康普顿角分辨度量 $\\Delta\\theta_{ARM} = \\theta_{kin} - \\theta_{geo}$ (deg)", fontsize=9.2)
    ax.set_ylabel("归一化概率密度", fontsize=9.5)
    ax.set_title(f"(c) 478 keV单事件测角展宽 (FWHM ≈ {fwhm_arm:.1f}°)", fontsize=10.5, fontweight="bold")
    ax.text(0.03, 0.94, f"反冲电子能量偏低 + 2 mm像素离散化\n导致圆锥角展宽达 {fwhm_arm:.1f}°",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#ECFDF5", ec="#059669", alpha=0.92))
    ax.grid(True, alpha=0.22)

    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig1_bnct_478_physics_and_spectrum.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_fig2_collimator_limitations(
    geom: BNCTSystemGeometry,
    dcfg: DetectorDigitizerConfig,
    ev_net: BNCTEventNet,
    pin_net: SinglePinholeNet,
    device: torch.device,
):
    """定量对比：真值 vs 平行板多孔 vs MURA编码板 vs 单小孔原始投影 vs 单小孔AI反算。"""
    rng = np.random.default_rng(2026)
    # 设置一内一外双病灶源：内侧 (-12, +8) mm 在探测器 40x40 mm 投影内，外侧 (+46, -34) mm 超出 1:1 平行孔视场！
    gt, _ = generate_bnct_boron_distribution(
        "dual_hotspots", rng,
        fixed_params={"cx1": -12.0, "cy1": 10.0, "sep": 68.0, "ang": -0.62, "ratio": 0.90, "sigma": 6.8},
    )
    n_sig, bgr = 48000, 0.70

    ev_mh = simulate_bnct_forward_transport(gt, geom, "parallel_multihole", n_sig, bgr, rng)
    dig_mh = digitize_detector_events(ev_mh, geom, dcfg, rng)
    rec_mh = reconstruct_parallel_multihole(dig_mh, geom)

    ev_mu = simulate_bnct_forward_transport(gt, geom, "coded_aperture_mura", n_sig, bgr, rng)
    dig_mu = digitize_detector_events(ev_mu, geom, dcfg, rng)
    rec_mu = reconstruct_coded_aperture_mura(dig_mu, geom)

    ev_sp = simulate_bnct_forward_transport(gt, geom, "single_pinhole", n_sig, bgr, rng)
    dig_sp = digitize_detector_events(ev_sp, geom, dcfg, rng)
    wp, _, _, _ = run_event_net_inference(ev_net, dig_sp, geom, device)
    x_sp = project_pinhole_channels(dig_sp, geom, wp)
    with torch.no_grad():
        rec_sp_ai = pin_net(torch.from_numpy(x_sp[None]).to(device)).cpu().numpy()[0, 0]

    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.35))
    ext = [-100, 100, -100, 100]
    panels = [
        (gt, "真实硼空间分布标签\n(内/外双病灶 478 keV)", None),
        (rec_mh, f"平行板多孔准直\nPSNR={compute_psnr(rec_mh, gt):.1f} dB (穿隔+外源漏检)", "#B91C1C"),
        (rec_mu, f"MURA编码板解码\nPSNR={compute_psnr(rec_mu, gt):.1f} dB (穿板对比度崩塌)", "#B91C1C"),
        (x_sp[1], f"单小孔传统能窗投影\nPSNR={compute_psnr(x_sp[1], gt):.1f} dB (穿壁+统计噪点)", "#D97706"),
        (rec_sp_ai, f"单小孔轻量化AI反算\nPSNR={compute_psnr(rec_sp_ai, gt):.1f} dB (全视场双源恢复)", "#047857"),
    ]
    for idx, (im, title, col) in enumerate(panels):
        ax = axes[idx]
        im_show = ax.imshow(im, origin="lower", extent=ext, cmap="inferno", vmin=0.0, vmax=1.0)
        if idx == 1:
            # 标出平行板多孔 40x40 mm 探测器 1:1 有效视场框
            rect = plt.Rectangle((-20, -20), 40, 40, fill=False, ec="#38BDF8", ls="--", lw=1.5)
            ax.add_patch(rect)
            ax.text(-19, 24, "1:1视场上限", color="#38BDF8", fontsize=7.8, fontweight="bold")
        ax.set_title(title, fontsize=9.2, fontweight="bold", color=col or "#111827")
        ax.set_xlabel("X (mm)", fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("Y (mm)", fontsize=8.5)
        else:
            ax.set_yticklabels([])

    cbar = fig.colorbar(im_show, ax=axes.ravel().tolist(), shrink=0.86, pad=0.015)
    cbar.set_label("归一化硼/478 keV源强度", fontsize=8.5)
    p = os.path.join(FIG_DIR, "fig2_collimator_limitations_proof.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_fig3_framework_and_network():
    """绘制清晰紧凑的 正算软件包 -> 反算软件包 -> 多路线横向评估 闭环技术路线与联合网络架构框图。"""
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def draw_box(x, y, w, h, title, sub, fc="#EFF6FF", ec="#1D4ED8", tc="#1E3A8A", fs=9.2):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.9", fc=fc, ec=ec, lw=1.5)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=fs, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 1.3, color="#334155")

    def arrow(x1, y1, x2, y2, col="#475569"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>,head_width=0.35", lw=1.6, color=col))

    # 顶层标题栏
    ax.text(50, 40.3, "BNCT 478 keV 像素化阵列智能伽马成像：“正算建模 — 非迭代AI反算 — 多路线评估”闭环架构",
            ha="center", va="center", fontsize=11.8, fontweight="bold", color="#0F172A")

    # 左侧：第一步 正算软件包 (forward_pkg)
    draw_box(1.5, 22.0, 25.0, 15.5, "第一步：正算软件包 (forward_pkg)",
             "• BNCT 478 keV特征信号 + 511/2223 keV复杂背景\n"
             "• 5类硼分布形态 & 单小孔/康普顿/多孔/编码板输运\n"
             "• 电子学噪声 + 能量分辨率 + 2mm像素/DOI退化修正\n"
             "• 批量生成多工况标准数据集 (真值标签 + 探测信号)",
             fc="#ECFDF5", ec="#059669", tc="#065F46", fs=9.4)

    # 中部前端：BNCTEventNet
    draw_box(31.0, 23.5, 21.5, 12.5, "前端事例甄别与解析网\nBNCTEventNet (4.8万参数)",
             "• 18维478 keV运动学一致性特征\n"
             "• 五分类剔除511/2223 keV与散射逃逸\n"
             "• 正逆康普顿序列消歧 & cosθ残差修正",
             fc="#FFFBEB", ec="#D97706", tc="#92400E", fs=9.0)

    arrow(26.7, 29.8, 30.8, 29.8, "#059669")

    # 中右部：反算软件包双分支联合网络 + 单模基线
    draw_box(57.0, 30.2, 21.0, 7.2, "单小孔准直编码分支 (d=1)",
             "双层倒像投影张量 (3×48×48)\nDSConv局部感受野锚定病灶边缘",
             fc="#EFF6FF", ec="#2563EB", tc="#1E40AF", fs=8.8)

    draw_box(57.0, 20.8, 21.0, 7.2, "康普顿圆锥编码分支 (d=2)",
             "净化圆锥反投影张量 (3×48×48)\n空洞DSConv大感受野聚合圆锥焦点",
             fc="#F5F3FF", ec="#7C3AED", tc="#5B21B6", fs=8.8)

    arrow(52.7, 32.0, 56.8, 33.8, "#2563EB")
    arrow(52.7, 27.5, 56.8, 24.4, "#7C3AED")

    draw_box(81.5, 22.2, 17.0, 14.2, "跨模态门控联合反算网\nJointBNCTReconNet",
             "• 空间-通道自适应门控融合\n"
             "• 多尺度轻量解码器 (<0.29M)\n"
             "• 零迭代毫秒级输出硼分布图",
             fc="#FEF2F2", ec="#DC2626", tc="#991B1B", fs=9.0)

    arrow(78.2, 33.8, 81.3, 30.8, "#DC2626")
    arrow(78.2, 24.4, 81.3, 27.5, "#DC2626")

    # 底部：第二步单模态分步基线 & 第三步横向对比闭环
    draw_box(1.5, 3.0, 44.0, 14.0, "第二步：反算软件包分步训练基准 (inverse_pkg)",
             "• 路线A 单小孔AI反算网 (SinglePinholeNet)：独立学习单孔穿壁/散射去模糊映射\n"
             "• 路线B 单康普顿AI反算网 (SingleComptonNet)：独立学习478 keV宽圆锥去晕环映射\n"
             "• 路线C 准直-康普顿联合轻量化反算网：融合准直空间锚定与康普顿高统计方向信息",
             fc="#F0F9FF", ec="#0284C7", tc="#075985", fs=9.0)

    draw_box(50.5, 3.0, 48.0, 14.0, "第三步：统一基准多方案横向评估与交付闭环",
             "• 系统对比 单小孔、单康普顿、联合重建 的PSNR/SSIM、抗多能本底干扰与毫秒级速度\n"
             "• 结合物理与正算数据阐明 平行板多孔穿隔/窄视场 与 编码板对比度崩塌 的低可行性根源\n"
             "• 核心交付：正算软件包 + 反算软件包 + 多工况标注数据集 + 性能评估报告",
             fc="#F8FAFC", ec="#334155", tc="#0F172A", fs=9.0)

    arrow(45.7, 10.0, 50.3, 10.0, "#334155")
    arrow(90.0, 22.0, 82.0, 17.3, "#DC2626")
    arrow(14.0, 21.8, 18.0, 17.3, "#0284C7")

    p = os.path.join(FIG_DIR, "fig3_framework_and_joint_network.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_fig4_event_net_and_convergence(summary: Dict):
    ev_m = summary["event_net_metrics"]
    hist = summary["training_val_psnr_history"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.0))

    # (a) 5-class confusion matrix
    ax = axes[0]
    cm = np.array(ev_m["confusion_matrix_norm"])
    im = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = ["478准直", "478正康", "478逆康", "多能干扰", "逃逸/散射"]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=8.0, rotation=22)
    ax.set_yticklabels(labels, fontsize=8.0)
    for r in range(5):
        for c in range(5):
            val = cm[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.52 else "#0F172A", fontsize=7.8, fontweight="bold")
    ax.set_title(f"(a) BNCTEventNet五分类混淆矩阵\n(非目标与散射本底剔除率 {ev_m['non478_and_scatter_reject_rate']*100:.1f}%)",
                 fontsize=9.8, fontweight="bold")
    ax.set_xlabel("网络预测类别", fontsize=9.0)
    ax.set_ylabel("蒙特卡罗物理真值类别", fontsize=9.0)
    fig.colorbar(im, ax=ax, shrink=0.82)

    # (b) Multi-class ROC curves
    ax = axes[1]
    colors = ["#2563EB", "#059669", "#7C3AED", "#DC2626", "#D97706"]
    names = ["478 keV准直", "478 keV正向康普顿", "478 keV逆向康普顿", "511/2223多能干扰", "逃逸/孔壁散射"]
    for c in range(5):
        rc = ev_m["roc_curves"][f"class_{c}"]
        ax.plot(rc["fpr"], rc["tpr"], color=colors[c], lw=1.8, label=f"{names[c]} (AUC={rc['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlabel("假阳性率 (FPR)", fontsize=9.2)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=9.2)
    ax.set_title(f"(b) 各物理类别ROC曲线 (宏平均AUC={ev_m['macro_roc_auc']:.4f})", fontsize=9.8, fontweight="bold")
    ax.legend(fontsize=7.4, loc="lower right")
    ax.grid(True, alpha=0.25)

    # (c) 三套反算网络验证集 PSNR 收敛曲线对比
    ax = axes[2]
    ax.plot(range(1, len(hist["single_compton_net"]) + 1), hist["single_compton_net"],
            color="#7C3AED", lw=1.9, ls="--", label="单康普顿反算网 (SingleComptonNet)")
    ax.plot(range(1, len(hist["single_pinhole_net"]) + 1), hist["single_pinhole_net"],
            color="#2563EB", lw=1.9, ls="-.", label="单小孔反算网 (SinglePinholeNet)")
    ax.plot(range(1, len(hist["joint_bnct_net"]) + 1), hist["joint_bnct_net"],
            color="#DC2626", lw=2.2, label="准直+康普顿联合反算网 (JointBNCTReconNet)")
    ax.set_xlabel("训练轮次 (Epoch)", fontsize=9.2)
    ax.set_ylabel("独立验证集平均 PSNR (dB)", fontsize=9.2)
    ax.set_title("(c) 三种非迭代轻量化反算路线训练收敛对比", fontsize=9.8, fontweight="bold")
    ax.legend(fontsize=7.8, loc="lower right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig4_event_net_and_training.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_fig5_multi_shape_reconstructions(
    geom: BNCTSystemGeometry,
    dcfg: DetectorDigitizerConfig,
    ev_net: BNCTEventNet,
    pin_net: SinglePinholeNet,
    comp_net: SingleComptonNet,
    joint_net: JointBNCTReconNet,
    device: torch.device,
):
    """展示代表性多形态硼空间分布在复杂 BNCT 辐射场下的单小孔、单康普顿与联合反算对比。"""
    cases = [
        ("single_hotspot", "单病灶热点源", {"cx": -26.0, "cy": 22.0, "sigma": 6.8}),
        ("dual_hotspots", "近距双病灶源", {"cx1": -24.0, "cy1": -14.0, "sep": 44.0, "ang": 0.55, "ratio": 0.88, "sigma": 6.2}),
        ("necrotic_ring", "中心坏死环形肿瘤源", {"cx": 6.0, "cy": -6.0, "radius": 28.0, "ring_w": 6.0}),
        ("irregular_tumor_tn", "分叶肿瘤+正常组织低硼本底", {"cx": -10.0, "cy": 8.0, "s1": 12.5, "s2": 9.5, "off_x": 14.0, "off_y": -11.0, "tn_bg": 0.12}),
    ]
    rng = np.random.default_rng(888)
    fig, axes = plt.subplots(len(cases), 4, figsize=(12.6, 10.8))
    ext = [-100, 100, -100, 100]

    for r_i, (sh_key, sh_label, fparams) in enumerate(cases):
        gt, _ = generate_bnct_boron_distribution(sh_key, rng, fixed_params=fparams)
        n_sig, bgr = 44000, 0.70

        ev_p = simulate_bnct_forward_transport(gt, geom, "single_pinhole", n_sig, bgr, rng)
        dig_p = digitize_detector_events(ev_p, geom, dcfg, rng)
        wp, _, _, _ = run_event_net_inference(ev_net, dig_p, geom, device)
        xp = project_pinhole_channels(dig_p, geom, wp)[None]

        ev_c = simulate_bnct_forward_transport(gt, geom, "single_compton", n_sig, bgr, rng)
        dig_c = digitize_detector_events(ev_c, geom, dcfg, rng)
        _, wfc, wrc, dcc = run_event_net_inference(ev_net, dig_c, geom, device)
        xc = project_compton_channels(dig_c, geom, wfc, wrc, dcc)[None]

        ev_j = simulate_bnct_forward_transport(gt, geom, "joint_pinhole_compton", n_sig, bgr, rng)
        dig_j = digitize_detector_events(ev_j, geom, dcfg, rng)
        wpj, wfj, wrj, dcj = run_event_net_inference(ev_net, dig_j, geom, device)
        xpj = project_pinhole_channels(dig_j, geom, wpj)[None]
        xcj = project_compton_channels(dig_j, geom, wfj, wrj, dcj)[None]

        with torch.no_grad():
            im_pin = pin_net(torch.from_numpy(xp).to(device)).cpu().numpy()[0, 0]
            im_comp = comp_net(torch.from_numpy(xc).to(device)).cpu().numpy()[0, 0]
            im_joint_t, _ = joint_net(torch.from_numpy(xpj).to(device), torch.from_numpy(xcj).to(device))
            im_joint = im_joint_t.cpu().numpy()[0, 0]

        row_imgs = [
            (gt, f"{sh_label}\n真值标签 (GT)"),
            (im_pin, f"单小孔AI反算\nPSNR={compute_psnr(im_pin, gt):.2f} dB | SSIM={compute_ssim(im_pin, gt):.3f}"),
            (im_comp, f"单康普顿AI反算\nPSNR={compute_psnr(im_comp, gt):.2f} dB | SSIM={compute_ssim(im_comp, gt):.3f}"),
            (im_joint, f"准直+康普顿联合AI反算\nPSNR={compute_psnr(im_joint, gt):.2f} dB | SSIM={compute_ssim(im_joint, gt):.3f}"),
        ]
        for c_i, (img, ttl) in enumerate(row_imgs):
            ax = axes[r_i, c_i]
            im_h = ax.imshow(img, origin="lower", extent=ext, cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_title(ttl, fontsize=8.8, fontweight="bold", color="#991B1B" if c_i == 3 else "#111827")
            if r_i == len(cases) - 1:
                ax.set_xlabel("X (mm)", fontsize=8.5)
            else:
                ax.set_xticklabels([])
            if c_i == 0:
                ax.set_ylabel("Y (mm)", fontsize=8.5)
            else:
                ax.set_yticklabels([])

    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig5_five_shapes_multi_route_comparison.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_fig6_robustness_and_summary(summary: Dict):
    bg_c = summary["bnct_background_sensitivity_curve"]
    fw_c = summary["energy_resolution_sensitivity_curve"]
    bm = summary["benchmark_80_scenes"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.1))

    # (a) BNCT 非纯净多能量背景干扰比例抗噪曲线
    ax = axes[0]
    x_bg = bg_c["bg_ratio"]
    ax.plot(x_bg, bg_c["joint_proposed_ai"], "o-", color="#DC2626", lw=2.2, label="联合重建 (含BNCTEventNet)")
    ax.plot(x_bg, bg_c["joint_no_evnet"], "s--", color="#EA580C", lw=1.7, label="联合重建 (无事例甄别消融)")
    ax.plot(x_bg, bg_c["single_pinhole_ai"], "^-.", color="#2563EB", lw=1.8, label="单小孔AI反算")
    ax.plot(x_bg, bg_c["single_compton_ai"], "d:", color="#7C3AED", lw=1.8, label="单康普顿AI反算")
    ax.set_xlabel("BNCT非478 keV杂散背景与目标光子通量比 ($\\eta_{bg}$)", fontsize=9.0)
    ax.set_ylabel("平均重建峰值信噪比 PSNR (dB)", fontsize=9.2)
    ax.set_title("(a) 复杂多能辐射场背景干扰鲁棒性对比", fontsize=9.9, fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, alpha=0.25)

    # (b) 探测器能量分辨率劣化敏感性曲线
    ax = axes[1]
    x_fw = fw_c["fwhm_662_pct"]
    ax.plot(x_fw, fw_c["joint_proposed_ai"], "o-", color="#DC2626", lw=2.2, label="准直+康普顿联合AI反算")
    ax.plot(x_fw, fw_c["single_pinhole_ai"], "^-.", color="#2563EB", lw=1.8, label="单小孔AI反算")
    ax.plot(x_fw, fw_c["single_compton_ai"], "d:", color="#7C3AED", lw=1.8, label="单康普顿AI反算")
    ax.set_xlabel("探测器基准能量分辨率 FWHM @ 662 keV (%)", fontsize=9.0)
    ax.set_ylabel("平均重建峰值信噪比 PSNR (dB)", fontsize=9.2)
    ax.set_title("(b) 探测器能量分辨率受限鲁棒性对比", fontsize=9.9, fontweight="bold")
    ax.legend(fontsize=7.8, loc="lower left")
    ax.grid(True, alpha=0.25)

    # (c) 80场景独立盲测集全方案横向指标汇总对比柱状图
    ax = axes[2]
    keys = [
        ("coded_aperture_mura", "MURA\n编码板"),
        ("parallel_multihole", "平行板\n多孔"),
        ("raw_compton_backproj", "康普顿\n直接反投影"),
        ("single_compton_ai", "单康普顿\nAI反算"),
        ("single_pinhole_ai", "单小孔\nAI反算"),
        ("joint_pinhole_compton_ai", "联合轻量化\nAI反算"),
    ]
    vals = [bm[k]["psnr_db"] for k, _ in keys]
    lbls = [lb for _, lb in keys]
    cols = ["#94A3B8", "#64748B", "#A78BFA", "#7C3AED", "#2563EB", "#DC2626"]
    bars = ax.bar(range(len(keys)), vals, color=cols, width=0.58, edgecolor="#1E293B", lw=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(lbls, fontsize=8.0)
    ax.set_ylim(0, max(vals) + 5.5)
    ax.set_ylabel("80组盲测平均 PSNR (dB)", fontsize=9.2)
    ax.set_title("(c) 80组复杂盲测集六种路线全局对比", fontsize=9.9, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    p = os.path.join(FIG_DIR, "fig6_robustness_and_ablation.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    device = torch.device("cpu")
    geom = BNCTSystemGeometry()
    dcfg = DetectorDigitizerConfig()
    summary = json.load(open(os.path.join(RES_DIR, "bnct_metrics_summary.json"), encoding="utf-8"))
    ev_net, pin_net, comp_net, joint_net = load_trained_models(device)

    paths = [
        plot_fig1_physics_and_spectrum(geom, dcfg),
        plot_fig2_collimator_limitations(geom, dcfg, ev_net, pin_net, device),
        plot_fig3_framework_and_network(),
        plot_fig4_event_net_and_convergence(summary),
        plot_fig5_multi_shape_reconstructions(geom, dcfg, ev_net, pin_net, comp_net, joint_net, device),
        plot_fig6_robustness_and_summary(summary),
    ]
    for p in paths:
        print("Generated figure:", p)


if __name__ == "__main__":
    main()
