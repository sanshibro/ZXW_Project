"""
BNCT 478 keV 智能伽马成像算法研究 · 正算与反算双软件包交互式实验平台
依托课题：四川大学大学生创新训练计划（课题编号 12475312，指导教师：韩纪锋研究员）
"""

from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import time
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from forward_pkg import (
    BNCTSystemGeometry,
    DetectorDigitizerConfig,
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
)
from run_bnct_experiments import compute_psnr, compute_ssim, compute_rmse

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RES_DIR, "figures")

st.set_page_config(
    page_title="BNCT 478 keV 智能伽马成像正反算实验台",
    page_icon="⚛️",
    layout="wide",
)


@st.cache_resource
def load_bnct_models():
    device = torch.device("cpu")
    ev_net = BNCTEventNet().to(device)
    pin_net = SinglePinholeNet().to(device)
    comp_net = SingleComptonNet().to(device)
    joint_net = JointBNCTReconNet().to(device)

    ev_p = os.path.join(CKPT_DIR, "bnct_event_net.pt")
    if os.path.exists(ev_p):
        ev_net.load_state_dict(torch.load(ev_p, map_location=device, weights_only=True))
        pin_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "single_pinhole_net.pt"), map_location=device, weights_only=True))
        comp_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "single_compton_net.pt"), map_location=device, weights_only=True))
        joint_net.load_state_dict(torch.load(os.path.join(CKPT_DIR, "joint_bnct_net.pt"), map_location=device, weights_only=True))

    ev_net.eval()
    pin_net.eval()
    comp_net.eval()
    joint_net.eval()
    return ev_net, pin_net, comp_net, joint_net, device


def infer_event_weights(ev_net, dig, geom, device):
    feats = extract_18d_event_features(dig, geom)
    if len(feats) == 0:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z, z
    with torch.no_grad():
        logits, dcos = ev_net(torch.from_numpy(feats).to(device))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs[:, 0], probs[:, 1], probs[:, 2], dcos.cpu().numpy()


st.title("⚛️ BNCT 478 keV 特征伽马智能成像 · 正算与反算双软件包实验平台")
st.caption(
    "四川大学大学生创新训练计划（课题编号 `12475312` · 指导教师：韩纪锋研究员） | "
    "聚焦硼中子俘获治疗（BNCT）478 keV 交叉能区与复杂多能本底下的单小孔、单康普顿及联合轻量化神经网络重建"
)

ev_net, pin_net, comp_net, joint_net, device = load_bnct_models()
summary_path = os.path.join(RES_DIR, "bnct_metrics_summary.json")
summary = json.load(open(summary_path, encoding="utf-8")) if os.path.exists(summary_path) else None

with st.sidebar:
    st.header("正算物理工况参数设置")
    shape_map = {
        "分叶肿瘤 + 正常组织低硼本底 (T/N)": "irregular_tumor_tn",
        "近距双病灶热点源": "dual_hotspots",
        "中心坏死环形肿瘤硼分布": "necrotic_ring",
        "单病灶点状硼富集源": "single_hotspot",
        "浸润条带/管状分布源": "tubular_line",
    }
    shape_label = st.selectbox("硼 (10B) 空间分布几何形态", list(shape_map.keys()), index=0)
    n_photons = st.slider("478 keV 目标发射光子数", min_value=15000, max_value=80000, value=45000, step=5000)
    bg_ratio = st.slider("BNCT 非单一能量杂散背景比 (511/2223 keV)", min_value=0.0, max_value=1.5, value=0.65, step=0.05)
    fwhm_662 = st.slider("像素阵列基准能量分辨率 (% @ 662 keV)", min_value=4.0, max_value=13.0, value=7.0, step=0.5)
    elec_noise = st.slider("电子学读出噪声 (keV)", min_value=1.5, max_value=10.0, value=4.5, step=0.5)
    seed_val = st.number_input("蒙特卡罗随机种子", min_value=1, max_value=99999, value=2026)
    show_controls = st.checkbox("同步运行平行板多孔与MURA编码板对照实验", value=True)
    run_btn = st.button("🚀 运行正算模拟与多路线AI反算对比", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs([
    "🔬 实时正算模拟与反算路线横向对比",
    "📊 80组独立盲测集与抗干扰性能总览",
    "📦 正算软件包 (forward_pkg) 与反算软件包 (inverse_pkg) 架构",
])

with tab1:
    if run_btn or "last_bnct_run" not in st.session_state:
        rng = np.random.default_rng(int(seed_val))
        geom = BNCTSystemGeometry()
        dcfg = DetectorDigitizerConfig(fwhm_662_pct=float(fwhm_662), elec_noise_kev=float(elec_noise))
        gt, meta = generate_bnct_boron_distribution(shape_map[shape_label], rng)

        t0 = time.perf_counter()
        ev_p = simulate_bnct_forward_transport(gt, geom, "single_pinhole", int(n_photons), float(bg_ratio), rng)
        dig_p = digitize_detector_events(ev_p, geom, dcfg, rng)

        ev_c = simulate_bnct_forward_transport(gt, geom, "single_compton", int(n_photons), float(bg_ratio), rng)
        dig_c = digitize_detector_events(ev_c, geom, dcfg, rng)

        ev_j = simulate_bnct_forward_transport(gt, geom, "joint_pinhole_compton", int(n_photons), float(bg_ratio), rng)
        dig_j = digitize_detector_events(ev_j, geom, dcfg, rng)
        fwd_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        wp, _, _, _ = infer_event_weights(ev_net, dig_p, geom, device)
        xp = project_pinhole_channels(dig_p, geom, wp)[None]
        with torch.no_grad():
            rec_pin = pin_net(torch.from_numpy(xp)).numpy()[0, 0]

        _, wfc, wrc, dcc = infer_event_weights(ev_net, dig_c, geom, device)
        xc = project_compton_channels(dig_c, geom, wfc, wrc, dcc)[None]
        with torch.no_grad():
            rec_comp = comp_net(torch.from_numpy(xc)).numpy()[0, 0]

        t_j0 = time.perf_counter()
        wpj, wfj, wrj, dcj = infer_event_weights(ev_net, dig_j, geom, device)
        xpj = project_pinhole_channels(dig_j, geom, wpj)[None]
        xcj = project_compton_channels(dig_j, geom, wfj, wrj, dcj)[None]
        with torch.no_grad():
            rec_joint_t, mw_t = joint_net(torch.from_numpy(xpj), torch.from_numpy(xcj))
            rec_joint = rec_joint_t.numpy()[0, 0]
            mw = mw_t.numpy()[0]
        joint_ms = (time.perf_counter() - t_j0) * 1000.0

        rec_mh, rec_mura = None, None
        if show_controls:
            dig_mh = digitize_detector_events(
                simulate_bnct_forward_transport(gt, geom, "parallel_multihole", int(n_photons), float(bg_ratio), rng),
                geom, dcfg, rng
            )
            rec_mh = reconstruct_parallel_multihole(dig_mh, geom)
            dig_mu = digitize_detector_events(
                simulate_bnct_forward_transport(gt, geom, "coded_aperture_mura", int(n_photons), float(bg_ratio), rng),
                geom, dcfg, rng
            )
            rec_mura = reconstruct_coded_aperture_mura(dig_mu, geom)

        st.session_state["last_bnct_run"] = {
            "gt": gt, "dig_j": dig_j, "rec_pin": rec_pin, "rec_comp": rec_comp,
            "rec_joint": rec_joint, "rec_mh": rec_mh, "rec_mura": rec_mura,
            "mw": mw, "fwd_ms": fwd_ms, "joint_ms": joint_ms,
        }

    res = st.session_state["last_bnct_run"]
    gt = res["gt"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("联合轻量化重建 PSNR", f"{compute_psnr(res['rec_joint'], gt):.2f} dB",
              f"+{compute_psnr(res['rec_joint'], gt) - compute_psnr(res['rec_pin'], gt):.2f} dB vs 单小孔")
    c2.metric("联合重建结构相似性 SSIM", f"{compute_ssim(res['rec_joint'], gt):.4f}",
              f"+{compute_ssim(res['rec_joint'], gt) - compute_ssim(res['rec_comp'], gt):.4f} vs 单康普顿")
    c3.metric("联合网络单帧反算耗时", f"{res['joint_ms']:.1f} ms", "非迭代神经网络一步出图")
    c4.metric("跨模态门控权重 (准直 : 康普顿)", f"{res['mw'][0]:.2f} : {res['mw'][1]:.2f}", "自适应双通道协同")

    ncols = 6 if res["rec_mh"] is not None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 3.3))
    ext = [-100, 100, -100, 100]
    items = [(gt, "真实硼空间分布 (GT)")]
    if res["rec_mh"] is not None:
        items.append((res["rec_mh"], f"平行板多孔 (对照)\nPSNR={compute_psnr(res['rec_mh'], gt):.1f} dB"))
        items.append((res["rec_mura"], f"MURA编码板 (对照)\nPSNR={compute_psnr(res['rec_mura'], gt):.1f} dB"))
    items.extend([
        (res["rec_pin"], f"单小孔AI反算\nPSNR={compute_psnr(res['rec_pin'], gt):.2f} dB"),
        (res["rec_comp"], f"单康普顿AI反算\nPSNR={compute_psnr(res['rec_comp'], gt):.2f} dB"),
        (res["rec_joint"], f"联合轻量化AI反算\nPSNR={compute_psnr(res['rec_joint'], gt):.2f} dB"),
    ])
    for ax, (im, ttl) in zip(axes, items):
        ax.imshow(im, origin="lower", extent=ext, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(ttl, fontsize=9.2, fontweight="bold")
        ax.set_xlabel("X (mm)", fontsize=8)
    axes[0].set_ylabel("Y (mm)", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    if summary is not None:
        st.subheader("80 组独立复杂工况盲测集横向评估结果")
        bm = summary["benchmark_80_scenes"]
        rows = [
            {"评估路线": "MURA 编码板准直解码（可行性对照）", **bm["coded_aperture_mura"]},
            {"评估路线": "平行板多孔准直投影（可行性对照）", **bm["parallel_multihole"]},
            {"评估路线": "单康普顿轻量化 AI 反算 (SingleComptonNet)", **bm["single_compton_ai"]},
            {"评估路线": "单小孔轻量化 AI 反算 (SinglePinholeNet)", **bm["single_pinhole_ai"]},
            {"评估路线": "联合重建（去除前端事例甄别消融）", **bm["joint_ai_no_eventnet"]},
            {"评估路线": "准直+康普顿联合轻量化 AI 反算（完整方案）", **bm["joint_pinhole_compton_ai"]},
        ]
        st.dataframe(
            [{
                "成像方案路线": r["评估路线"],
                "平均 PSNR (dB)": r["psnr_db"],
                "平均 SSIM": r["ssim"],
                "平均 RMSE": r["rmse"],
                "单帧耗时 (ms)": r["net_latency_ms"],
            } for r in rows],
            use_container_width=True,
        )

    for fn, cap in [
        ("fig2_collimator_limitations_proof.png", "平行板多孔与编码板在 BNCT 478 keV 复杂场下的局限性机理验证"),
        ("fig5_five_shapes_multi_route_comparison.png", "不同硼/放射源形状下单小孔、单康普顿与联合反算对比"),
        ("fig6_robustness_and_ablation.png", "多能本底干扰与能量分辨率退化鲁棒性曲线"),
    ]:
        fp = os.path.join(FIG_DIR, fn)
        if os.path.exists(fp):
            st.image(fp, caption=cap, use_container_width=True)

with tab3:
    st.markdown("""
### 双软件包工程目录结构
1. **`forward_pkg/`（正算软件包）**
   - `bnct_physics_and_collimators.py`：封装 `GAGG:Ce` 与钨合金截面插值、5 类硼浓度分布生成、BNCT `478 keV` 特征伽马与 `511 keV / 2223 keV` 复杂本底抽样，以及单小孔、单康普顿、联合几何、平行板多孔、MURA 编码板五种输运模型。
   - `detector_digitizer.py`：模拟电子学噪声、平方根能量分辨率展宽、相邻像素串扰、`2.0 mm` 像素离散化与轴向 DOI 位置模糊，并提供无迭代快速几何投影接口。
2. **`inverse_pkg/`（反算软件包）**
   - `bnct_event_net.py`：轻量残差网络 `BNCTEventNet`（`4.8万`参数），基于 18 维运动学判据剔除非 478 keV 杂散干扰与孔壁散射，恢复康普顿正逆散射序列并微调散射角余弦。
   - `recon_networks.py`：提供单小孔反算网 `SinglePinholeNet`、单康普顿反算网 `SingleComptonNet` 及跨模态门控联合重建网 `JointBNCTReconNet`（全链路总参数量 `< 0.30 M`，零传统统计迭代）。
""")
