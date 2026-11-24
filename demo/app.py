"""
Streamlit Interactive Workbench:
基于像素化阵列探测器的准直孔与康普顿散射智能伽马成像仿真与重建平台
Run from repo root:
  streamlit run demo/app.py
"""

from __future__ import annotations
import os
import sys
import copy
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
import streamlit as st

from demo.src.physics.geometry_and_sources import (
    CameraGeometry,
    SourceComponent,
    SourceScene,
    render_ground_truth_map,
)
from demo.src.physics.mc_simulator import simulate_scene_transport
from demo.src.physics.digitizer import DetectorDigitizer
from demo.src.baselines.traditional_recon import build_collimated_system_matrix
from demo.evaluate_benchmark import reconstruct_all_methods, calc_psnr, calc_ssim_2d
from demo.src.models.event_net import LiteEventNet
from demo.src.models.dual_gammanet import LiteDualGammaNet

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(
    page_title="双模智能伽马成像仿真与重建平台",
    page_icon="☢️",
    layout="wide",
)


@st.cache_resource
def load_models_and_base_config():
    with open("demo/configs/system_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ev_net = LiteEventNet(in_features=16, hidden_dim=96, num_classes=4).to(device)
    ev_net.load_state_dict(torch.load("demo/checkpoints/lite_event_net.pt", map_location=device))
    ev_net.eval()

    rc_net = LiteDualGammaNet(in_ch_col=3, in_ch_comp=3, base_ch=32).to(device)
    rc_net.load_state_dict(torch.load("demo/checkpoints/lite_dual_gammanet.pt", map_location=device))
    rc_net.eval()
    return cfg, ev_net, rc_net, device


st.title("☢️ 基于像素化阵列探测器的准直孔与康普顿散射智能伽马成像系统")
st.caption("四川大学大创课题原型演示系统 | 物理输运蒙特卡罗 + 真实探测器退化修正 (Digitizer) + 两级轻量化 AI 联合重建 vs 传统迭代算法")

base_cfg, event_net, recon_net, device = load_models_and_base_config()

# ==================== Sidebar Controls ====================
st.sidebar.header("1. 放射源场景预设与工况配置")
preset = st.sidebar.selectbox(
    "选择典型测试工况",
    [
        "宽能段双核素同场 (Am-241 59.5keV + Cs-137 661.7keV)",
        "低能单点源 Am-241 (59.5 keV - 准直主导)",
        "中高能单点源 Cs-137 (661.7 keV - 康普顿主导)",
        "近距双点源角分辨测试 (12° 间隔, Tc-99m 140 keV)",
        "复杂扩展圆环源 (Ba-133 356 keV)",
        "L形扩展面源 (Cs-137 661.7 keV)",
        "自定义单/双放射源配置",
    ],
)

n_photons = st.sidebar.slider("入射探测器立体角模拟光子数", min_value=10000, max_value=130000, value=75000, step=5000)

st.sidebar.header("2. 真实探测器物理退化修正参数")
fwhm_662 = st.sidebar.slider("662 keV 处能量分辨率 FWHM (%)", min_value=3.0, max_value=14.0, value=7.0, step=0.5) / 100.0
doi_sigma = st.sidebar.slider("晶体作用深度 (DOI) 误差 σ_z (mm)", min_value=0.2, max_value=3.0, value=1.2, step=0.2)
trig_thresh = st.sidebar.slider("单通道电子学触发阈值 (keV)", min_value=10.0, max_value=35.0, value=18.0, step=2.0)
bg_frac = st.sidebar.slider("环境本底与偶然符合污染比例 (%)", min_value=0, max_value=25, value=8, step=2) / 100.0
mlem_iters = st.sidebar.slider("传统联合 LM-MLEM 对比迭代轮数", min_value=10, max_value=60, value=30, step=5)

# Custom scene controls if chosen
custom_comps = []
if preset == "自定义单/双放射源配置":
    st.sidebar.subheader("自定义源 1")
    e_c1 = st.sidebar.selectbox("源1 核素能量 (keV)", [59.5, 140.5, 356.0, 661.7, 1332.5], index=0)
    x_c1 = st.sidebar.slider("源1 水平角 θ_x (°)", -25.0, 25.0, -14.0, 1.0)
    y_c1 = st.sidebar.slider("源1 俯仰角 θ_y (°)", -25.0, 25.0, 10.0, 1.0)
    custom_comps.append(SourceComponent("point", float(e_c1), 1.0, {"cx": x_c1, "cy": y_c1}))
    if st.sidebar.checkbox("启用源 2 (构成双源混合场)", value=True):
        e_c2 = st.sidebar.selectbox("源2 核素能量 (keV)", [59.5, 140.5, 356.0, 661.7, 1332.5], index=3)
        x_c2 = st.sidebar.slider("源2 水平角 θ_x (°)", -25.0, 25.0, 13.0, 1.0)
        y_c2 = st.sidebar.slider("源2 俯仰角 θ_y (°)", -25.0, 25.0, -10.0, 1.0)
        custom_comps.append(SourceComponent("point", float(e_c2), 1.0, {"cx": x_c2, "cy": y_c2}))

# Build dynamic config
cfg_run = copy.deepcopy(base_cfg)
cfg_run["digitizer"]["energy_res_ref_fwhm"] = fwhm_662
cfg_run["digitizer"]["doi_sigma_mm"] = doi_sigma
cfg_run["digitizer"]["trigger_threshold_keV"] = trig_thresh
cfg_run["digitizer"]["random_bg_fraction"] = bg_frac

geom = CameraGeometry.from_config(cfg_run)
digitizer = DetectorDigitizer(geom, cfg_run["digitizer"])
a_coll = build_collimated_system_matrix(geom, rep_energy_kev=120.0)

if preset.startswith("宽能段双核素同场"):
    comps = [
        SourceComponent("point", 59.5, 0.92, {"cx": -15.0, "cy": 12.0}),
        SourceComponent("point", 661.7, 1.00, {"cx": 14.0, "cy": -11.0}),
    ]
elif preset.startswith("低能单点源"):
    comps = [SourceComponent("point", 59.5, 1.0, {"cx": -12.0, "cy": 10.0})]
elif preset.startswith("中高能单点源"):
    comps = [SourceComponent("point", 661.7, 1.0, {"cx": 14.0, "cy": -11.0})]
elif preset.startswith("近距双点源"):
    comps = [
        SourceComponent("point", 140.5, 1.0, {"cx": -6.0, "cy": 2.0}),
        SourceComponent("point", 140.5, 1.0, {"cx": 6.0, "cy": 2.0}),
    ]
elif preset.startswith("复杂扩展圆环源"):
    comps = [SourceComponent("ring", 356.0, 1.0, {"cx": 0.0, "cy": 0.0, "radius": 11.0})]
elif preset.startswith("L形扩展面源"):
    comps = [SourceComponent("l_shape", 661.7, 1.0, {"cx": -9.0, "cy": -9.0, "arm": 17.0})]
else:
    comps = custom_comps

scene = SourceScene(preset, comps, *render_ground_truth_map(geom, comps))

# Run live MC simulation + Digitization + Reconstruction
rng = np.random.default_rng(1024)
t_mc0 = time.perf_counter()
raw_ev = simulate_scene_transport(geom, scene, n_photons, rng, bg_fraction=bg_frac)
dig_ev = digitizer.digitize(raw_ev, rng)
t_mc_ms = (time.perf_counter() - t_mc0) * 1000.0

res = reconstruct_all_methods(geom, a_coll, dig_ev, event_net, recon_net, device, mlem_iters=mlem_iters)
san = res["sanitized"]
mw = res["modal_weights"]
speedup = res["t_mlem_ms"] / max(res["t_ai_ms"], 0.05)

# Top KPI Metrics Row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("触发数字化总事例数", f"{dig_ev.n_hits.size:,}", f"MC模拟 {t_mc_ms:.1f} ms")
k2.metric("AI甄别有效准直 / 康普顿数", f"{san.pix_ix_col.size:,} / {san.r1_comp_mm.shape[0]:,}", f"剔除废事例 {int(np.sum(san.pred_classes == 0)):,}")
k3.metric("本文双模 AI 重建 PSNR", f"{calc_psnr(res['ai'], scene.gt_map):.2f} dB", f"SSIM = {calc_ssim_2d(res['ai'], scene.gt_map):.3f}")
k4.metric("双模 AI 单帧推理耗时", f"{res['t_ai_ms']:.2f} ms", f"较 LM-MLEM 提速 {speedup:.0f}×")
k5.metric("跨模态自适应门控权重", f"准直 {mw[0]*100:.0f}% : 康普顿 {mw[1]*100:.0f}%", "物理可解释动态融合")

# Main 5-Panel Comparison Figure
extent = [-geom.fov_half_deg, geom.fov_half_deg, -geom.fov_half_deg, geom.fov_half_deg]
fig, axes = plt.subplots(1, 5, figsize=(17.0, 3.8))
panels = [
    ("真实放射源分布 (GT)", scene.gt_map, ""),
    ("纯准直解码 (MURA)", res["mura"], f"PSNR: {calc_psnr(res['mura'], scene.gt_map):.1f} dB"),
    ("纯康普顿反投影 (SBP)", res["sbp"], f"PSNR: {calc_psnr(res['sbp'], scene.gt_map):.1f} dB"),
    (f"传统联合 LM-MLEM ({mlem_iters}轮)", res["jmlem"], f"PSNR: {calc_psnr(res['jmlem'], scene.gt_map):.1f} dB | {res['t_mlem_ms']:.1f} ms"),
    ("本文轻量化双模 AI", res["ai"], f"PSNR: {calc_psnr(res['ai'], scene.gt_map):.1f} dB | {res['t_ai_ms']:.1f} ms"),
]
for j, (tit, im_d, sub) in enumerate(panels):
    axes[j].imshow(im_d, cmap="inferno", origin="lower", extent=extent, vmin=0.0, vmax=1.0)
    axes[j].set_title(tit, fontsize=10.5, fontweight="bold")
    axes[j].set_xlabel("水平角 θ_x (°)")
    if j == 0:
        axes[j].set_ylabel("俯仰角 θ_y (°)")
    if sub:
        axes[j].text(0.04, 0.94, sub, transform=axes[j].transAxes, color="white", fontsize=8, va="top", bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6))
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Bottom Diagnostics Row: Energy Spectrum + Event Breakdown
col_left, col_right = st.columns(2)
with col_left:
    st.subheader("探测器理想沉积能谱 vs Digitizer 退化修正后实测能谱")
    fig_e, ax_e = plt.subplots(figsize=(7.2, 3.3))
    e_raw = raw_ev.e1_kev + raw_ev.e2_kev
    e_dig = dig_ev.e1_dig_kev + dig_ev.e2_dig_kev
    bins = np.linspace(10, max(float(np.percentile(e_dig, 99.5)), 200.0), 110)
    ax_e.hist(e_raw, bins=bins, alpha=0.45, color="#4C78A8", label="MC 理想沉积能量")
    ax_e.hist(e_dig, bins=bins, histtype="step", lw=1.6, color="#E45756", label="Digitizer 展宽+噪声+阈值后能量")
    ax_e.set_yscale("log")
    ax_e.set_xlabel("沉积总能量 (keV)")
    ax_e.set_ylabel("事件计数 (log)")
    ax_e.legend()
    ax_e.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig_e)
    plt.close(fig_e)

with col_right:
    st.subheader("第一级 LiteEventNet 光子作用事例四分类解析统计")
    fig_c, ax_c = plt.subplots(figsize=(7.2, 3.3))
    c_labels = ["逃逸/穿透/本底\n(已自动剔除)", "有效准直光电\n(送入准直分支)", "正序康普顿 A→B\n(送入康普顿分支)", "逆序康普顿 B→A\n(AI自动翻转修复)"]
    cnts = [int(np.sum(san.pred_classes == k)) for k in range(4)]
    bars = ax_c.bar(c_labels, cnts, color=["#7F7F7F", "#4C78A8", "#54A24B", "#F58518"], width=0.52)
    for b, v in zip(bars, cnts):
        ax_c.text(b.get_x() + b.get_width() / 2, v + max(cnts) * 0.02, f"{v:,}", ha="center", fontsize=9)
    ax_c.set_ylabel("事件数")
    ax_c.set_ylim(0, max(cnts) * 1.18 + 10)
    ax_c.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig_c)
    plt.close(fig_c)
