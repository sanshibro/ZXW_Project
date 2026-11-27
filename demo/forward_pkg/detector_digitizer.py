"""
BNCT 像素化阵列探测器真实响应数字化修正与正算特征投影模块
(Pixelated Detector Digitizer & Physical Feature Projectors)

模拟实际探测器三大核心退化机制：
  1. 电子学读出暗噪声与闪烁光子统计涨落（能量分辨率受限，478 keV 与 511 keV 峰部分交叠）
  2. 离散像素尺寸量化、相邻晶体光学串扰与晶体厚度方向作用深度（DOI）模糊（位置分辨率受限）
  3. 电子学下限触发阈值截断与符合窗内偶然暗计数注入
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch

from .bnct_physics_and_collimators import BNCTSystemGeometry, ME_C2_KEV, build_rank19_mura_mask


@dataclass
class DetectorDigitizerConfig:
    fwhm_662_pct: float = 7.0       # 662 keV 处基准能量分辨率 (%) -> 478 keV 处约为 8.25% (~39.4 keV FWHM)
    elec_noise_kev: float = 4.5     # 电子学等效基线均方根噪声 sigma_elec (keV)
    crosstalk_frac: float = 0.028   # SiPM/晶体阵列相邻通道光学串扰系数 (2.8%)
    threshold_kev: float = 20.0     # 单通道触发阈值 (keV)
    dark_rate_frac: float = 0.015   # 电子学偶然暗噪声伪事例比例


def digitize_detector_events(
    ideal_events: Dict[str, np.ndarray],
    geom: BNCTSystemGeometry,
    cfg: DetectorDigitizerConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """将理想蒙特卡罗连续交互事件转化为贴近真实像素化阵列探测器的数字化读出信号。"""
    n = len(ideal_events["n_hits"])
    if n == 0:
        return dict(ideal_events)

    r_a = ideal_events["r_a"].copy()
    r_b = ideal_events["r_b"].copy()
    e_a = ideal_events["e_a"].copy()
    e_b = ideal_events["e_b"].copy()
    l_a = ideal_events["layer_a"].copy()
    l_b = ideal_events["layer_b"].copy()
    n_hits = ideal_events["n_hits"].copy()
    t_cls = ideal_events["true_class"].copy()
    t_cos = ideal_events["true_cos_theta"].copy()

    # 1. 相邻通道光学串扰导致的能量微弱亏损与邻近像素重心漂移
    e_a_ct = e_a * (1.0 - cfg.crosstalk_frac * rng.uniform(0.5, 1.5, size=n))
    e_b_ct = np.where(n_hits == 2, e_b * (1.0 - cfg.crosstalk_frac * rng.uniform(0.5, 1.5, size=n)), 0.0)

    # 2. 平方根能量非线性高斯展宽 + 电子学常数热噪声: sigma(E) = sqrt( (sigma_662 * sqrt(E/662))^2 + sigma_elec^2 )
    sig_ref = (cfg.fwhm_662_pct / 100.0) * 661.7 / 2.35482
    sig_a = np.sqrt((sig_ref * np.sqrt(np.maximum(e_a_ct, 1.0) / 661.7)) ** 2 + cfg.elec_noise_kev ** 2)
    sig_b = np.sqrt((sig_ref * np.sqrt(np.maximum(e_b_ct, 1.0) / 661.7)) ** 2 + cfg.elec_noise_kev ** 2)

    e_a_meas = np.maximum(e_a_ct + rng.normal(0.0, sig_a), 0.0)
    e_b_meas = np.where(n_hits == 2, np.maximum(e_b_ct + rng.normal(0.0, sig_b), 0.0), 0.0)

    # 3. 位置分辨率退化：像素节距 (2.0 mm) 网格中心离散化 + 串扰跳变 + 晶体层中心深度 (DOI) 固化
    def _quantize_xyz(r_xyz: np.ndarray, layer_id: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 约 4% 概率因光学串扰偏移至相邻半个像素
        jitter_x = np.where(rng.random(len(r_xyz)) < 0.04, rng.choice([-1.0, 1.0], size=len(r_xyz)) * 0.5 * geom.pix_pitch_mm, 0.0)
        jitter_y = np.where(rng.random(len(r_xyz)) < 0.04, rng.choice([-1.0, 1.0], size=len(r_xyz)) * 0.5 * geom.pix_pitch_mm, 0.0)
        ix = np.clip(np.floor((r_xyz[:, 0] + jitter_x + geom.det_half_mm) / geom.pix_pitch_mm).astype(np.int32), 0, geom.n_pix - 1)
        iy = np.clip(np.floor((r_xyz[:, 1] + jitter_y + geom.det_half_mm) / geom.pix_pitch_mm).astype(np.int32), 0, geom.n_pix - 1)
        xq = -geom.det_half_mm + (ix + 0.5) * geom.pix_pitch_mm
        yq = -geom.det_half_mm + (iy + 0.5) * geom.pix_pitch_mm
        # 无双端 DOI 读出时，z 轴位置退化为所在层晶体几何中心 (z=0 或 z=-layer_gap_mm)
        zq = np.where(layer_id == 0, 0.0, np.where(layer_id == 1, -geom.layer_gap_mm, 0.0))
        return np.column_stack([xq, yq, zq]).astype(np.float32), ix, iy

    ra_q, ixa, iya = _quantize_xyz(r_a, l_a)
    rb_q, ixb, iyb = _quantize_xyz(r_b, l_b)

    # 4. 电子学触发下限阈值截断 (E_th = 20 keV)
    pass_a = e_a_meas >= cfg.threshold_kev
    pass_b = (n_hits == 2) & (e_b_meas >= cfg.threshold_kev)

    # 若双点事例中 B 未过阈值但 A 过阈值，则退化为单点伪事例 (Class 4)
    demote = (n_hits == 2) & pass_a & (~pass_b)
    if np.any(demote):
        n_hits[demote] = 1
        e_b_meas[demote] = 0.0
        rb_q[demote] = 0.0
        l_b[demote] = -1
        t_cls[demote] = np.where(t_cls[demote] == 3, 3, 4)

    valid_trig = pass_a & ((n_hits == 1) | pass_b)

    # 5. 注入少量电子学热暗噪声/偶然符合伪触发 (Class 4)
    n_dark = int(round(valid_trig.sum() * cfg.dark_rate_frac))
    ra_v = ra_q[valid_trig]
    rb_v = rb_q[valid_trig]
    ea_v = e_a_meas[valid_trig].astype(np.float32)
    eb_v = e_b_meas[valid_trig].astype(np.float32)
    la_v = l_a[valid_trig]
    lb_v = l_b[valid_trig]
    nh_v = n_hits[valid_trig]
    cls_v = t_cls[valid_trig]
    cos_v = t_cos[valid_trig]
    ix_v, iy_v = ixa[valid_trig], iya[valid_trig]

    if n_dark > 0:
        ix_d = rng.integers(0, geom.n_pix, size=n_dark, dtype=np.int32)
        iy_d = rng.integers(0, geom.n_pix, size=n_dark, dtype=np.int32)
        xd = -geom.det_half_mm + (ix_d + 0.5) * geom.pix_pitch_mm
        yd = -geom.det_half_mm + (iy_d + 0.5) * geom.pix_pitch_mm
        ra_d = np.column_stack([xd, yd, np.zeros(n_dark)]).astype(np.float32)
        rb_d = np.zeros((n_dark, 3), dtype=np.float32)
        ea_d = rng.uniform(cfg.threshold_kev, 95.0, size=n_dark).astype(np.float32)
        eb_d = np.zeros(n_dark, dtype=np.float32)
        la_d = np.zeros(n_dark, dtype=np.int32)
        lb_d = np.full(n_dark, -1, dtype=np.int32)
        nh_d = np.ones(n_dark, dtype=np.int32)
        cls_d = np.full(n_dark, 4, dtype=np.int64)
        cos_d = np.zeros(n_dark, dtype=np.float32)

        ra_v = np.vstack([ra_v, ra_d])
        rb_v = np.vstack([rb_v, rb_d])
        ea_v = np.concatenate([ea_v, ea_d])
        eb_v = np.concatenate([eb_v, eb_d])
        la_v = np.concatenate([la_v, la_d])
        lb_v = np.concatenate([lb_v, lb_d])
        nh_v = np.concatenate([nh_v, nh_d])
        cls_v = np.concatenate([cls_v, cls_d])
        cos_v = np.concatenate([cos_v, cos_d])
        ix_v = np.concatenate([ix_v, ix_d])
        iy_v = np.concatenate([iy_v, iy_d])

    return {
        "r_a": ra_v,
        "e_a": ea_v,
        "layer_a": la_v,
        "r_b": rb_v,
        "e_b": eb_v,
        "layer_b": lb_v,
        "n_hits": nh_v,
        "pix_ix_a": ix_v,
        "pix_iy_a": iy_v,
        "true_class": cls_v,
        "true_cos_theta": cos_v,
    }


def extract_18d_event_features(dig: Dict[str, np.ndarray], geom: BNCTSystemGeometry) -> np.ndarray:
    """
    为每个数字化触发事例提取 18 维物理与 478 keV 康普顿运动学判别特征（供 BNCTEventNet 使用）。
    """
    n = len(dig["n_hits"])
    if n == 0:
        return np.zeros((0, 18), dtype=np.float32)

    ra = dig["r_a"]
    rb = dig["r_b"]
    ea = dig["e_a"]
    eb = dig["e_b"]
    la = dig["layer_a"].astype(np.float32)
    lb = dig["layer_b"].astype(np.float32)
    is_two = (dig["n_hits"] == 2).astype(np.float32)
    esum = ea + eb

    # 针对 BNCT 478 keV 目标全能峰的相对能量偏离度与 511 keV 邻峰区分特征
    dev_478 = (esum - 477.6) / 45.0
    dev_511 = (esum - 511.0) / 45.0
    above_540 = (esum > 545.0).astype(np.float32)

    # 正向假设 (A->B) 与逆向假设 (B->A) 下的康普顿散射角余弦值
    # 公式: cos(theta) = 1 - m_e c^2 * (1 / E2 - 1 / (E1 + E2))
    safe_eb = np.maximum(eb, 5.0)
    safe_ea = np.maximum(ea, 5.0)
    safe_sum = np.maximum(esum, 10.0)
    cos_ab = np.where(is_two > 0.5, 1.0 - ME_C2_KEV * (1.0 / safe_eb - 1.0 / safe_sum), 0.0)
    cos_ba = np.where(is_two > 0.5, 1.0 - ME_C2_KEV * (1.0 / safe_ea - 1.0 / safe_sum), 0.0)
    valid_ab = ((np.abs(cos_ab) <= 1.02) & (is_two > 0.5)).astype(np.float32)
    valid_ba = ((np.abs(cos_ba) <= 1.02) & (is_two > 0.5)).astype(np.float32)

    # 若固定 E0 = 477.6 keV 先验，由首点沉积能量反推的理想散射角余弦与实测之差（强力识别逃逸与非478干扰！）
    e_scat_478_ab = np.maximum(477.6 - ea, 5.0)
    cos_478_prior_ab = np.where(is_two > 0.5, 1.0 - ME_C2_KEV * (1.0 / e_scat_478_ab - 1.0 / 477.6), 0.0)
    kin_consistency_478 = np.clip(np.abs(cos_ab - cos_478_prior_ab), 0.0, 3.0)

    dist_ab = np.where(is_two > 0.5, np.linalg.norm(rb - ra, axis=1) / 45.0, 0.0)
    dz_ab = np.where(is_two > 0.5, (ra[:, 2] - rb[:, 2]) / max(geom.layer_gap_mm, 1.0), 0.0)

    feats = np.column_stack([
        ra[:, 0] / geom.det_half_mm,
        ra[:, 1] / geom.det_half_mm,
        ra[:, 2] / max(geom.layer_gap_mm, 1.0),
        ea / 600.0,
        la,
        rb[:, 0] / geom.det_half_mm,
        rb[:, 1] / geom.det_half_mm,
        rb[:, 2] / max(geom.layer_gap_mm, 1.0),
        eb / 600.0,
        lb,
        is_two,
        esum / 600.0,
        np.clip(dev_478, -5.0, 8.0),
        np.clip(dev_511, -5.0, 8.0),
        above_540,
        np.clip(cos_ab, -2.5, 2.5),
        valid_ab - valid_ba,
        kin_consistency_478 + 0.25 * dist_ab + 0.15 * dz_ab,
    ]).astype(np.float32)
    return feats


def project_pinhole_channels(
    dig: Dict[str, np.ndarray],
    geom: BNCTSystemGeometry,
    ai_pinhole_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将单小孔准直通道信号（综合前层 z=0 与后层 z=-gap 单像素吸收事件，各自按精确焦距倒像比例翻转对齐）
    映射为 3 通道 (3, 48, 48) 空间分布输入张量：
      Ch 0: 全能谱单点投影（含非目标能量穿壁本底）
      Ch 1: 传统 478 keV 能窗 ([425, 525] keV) 截取倒像投影（受 511 keV 交叠与孔壁散射影响）
      Ch 2: BNCTEventNet 智能甄别加权净化的准直投影通道
    """
    n_pix = geom.n_pix
    src_coords = np.linspace(-geom.fov_half_mm, geom.fov_half_mm, geom.img_grid, dtype=np.float32)
    out = np.zeros((3, geom.img_grid, geom.img_grid), dtype=np.float32)

    for layer_idx, z_det in [(0, 0.0), (1, -geom.layer_gap_mm)]:
        sel = (dig["n_hits"] == 1) & (dig["layer_a"] == layer_idx)
        if not np.any(sel):
            continue
        ix = dig["pix_ix_a"][sel]
        iy = dig["pix_iy_a"][sel]
        ea = dig["e_a"][sel]

        det_raw = np.zeros((n_pix, n_pix), dtype=np.float32)
        det_ewin = np.zeros((n_pix, n_pix), dtype=np.float32)
        det_clean = np.zeros((n_pix, n_pix), dtype=np.float32)

        np.add.at(det_raw, (iy, ix), 1.0)
        in_win = (ea >= 425.0) & (ea <= 525.0)
        np.add.at(det_ewin, (iy[in_win], ix[in_win]), 1.0)
        if ai_pinhole_weights is not None:
            np.add.at(det_clean, (iy, ix), ai_pinhole_weights[sel])
        else:
            in_narrow = (ea >= 440.0) & (ea <= 510.0)
            np.add.at(det_clean, (iy[in_narrow], ix[in_narrow]), 1.0)

        # 第 layer_idx 层与准直孔平面 z_col 的轴向焦距 f_L = z_col - z_det
        f_layer = geom.collimator_z_mm - z_det
        inv_scale = (geom.source_z_mm - geom.collimator_z_mm) / f_layer
        det_xy = -src_coords / inv_scale
        u = (det_xy + geom.det_half_mm) / geom.pix_pitch_mm - 0.5

        for ch_i, dmap in enumerate([det_raw, det_ewin, det_clean]):
            out[ch_i] += _bilinear_resample_2d(dmap, u, u, zero_outside=True)

    for ch_i in range(3):
        mx = float(out[ch_i].max())
        if mx > 1e-6:
            out[ch_i] /= mx
    return out


def project_compton_channels(
    dig: Dict[str, np.ndarray],
    geom: BNCTSystemGeometry,
    ai_fwd_weights: Optional[np.ndarray] = None,
    ai_rev_weights: Optional[np.ndarray] = None,
    ai_delta_cos: Optional[np.ndarray] = None,
    max_cones: int = 1600,
) -> np.ndarray:
    """
    将双层符合康普顿散射事例在目标源平面 z = z_s 上做快速矢量化圆锥投影，生成 (3, 48, 48) 特征张量：
      Ch 0: 传统 478 keV 能窗 ([425, 530] keV) 简单圆锥反投影（含 511/2223 散射混入与 B->A 反序伪圆锥）
      Ch 1: BNCTEventNet 剔除杂散能量 + 翻转校正 B->A 反序 + 散射角余弦微调后的纯净圆锥投影
      Ch 2: 立体角与圆锥周长 1/max(sin(theta), 0.2) 归一化加权圆锥密度图
    """
    g = geom.img_grid
    out = np.zeros((3, g, g), dtype=np.float32)
    two = np.flatnonzero(dig["n_hits"] == 2)
    if len(two) == 0:
        return out

    coords = np.linspace(-geom.fov_half_mm, geom.fov_half_mm, g, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    vox = np.stack([xx, yy, np.full_like(xx, geom.source_z_mm)], axis=-1)  # (G, G, 3)

    ra = dig["r_a"][two]
    rb = dig["r_b"][two]
    ea = dig["e_a"][two]
    eb = dig["e_b"][two]
    esum = ea + eb

    # --- Ch 0: 传统能窗康普顿反投影 (假设前层为首点或直接 r_a -> r_b) ---
    in_win = (esum >= 425.0) & (esum <= 530.0)
    idx_w = np.flatnonzero(in_win)[:max_cones]
    if len(idx_w) > 0:
        # 传统规则：若 layer_a == 1 且 layer_b == 0 则按层号交换，但无法识别同层或背散射 B->A！
        la_w = dig["layer_a"][two[idx_w]]
        lb_w = dig["layer_b"][two[idx_w]]
        sw_naive = (la_w == 1) & (lb_w == 0)
        r1_n = np.where(sw_naive[:, None], rb[idx_w], ra[idx_w])
        r2_n = np.where(sw_naive[:, None], ra[idx_w], rb[idx_w])
        e2_n = np.where(sw_naive, ea[idx_w], eb[idx_w])
        es_n = esum[idx_w]
        cos_n = 1.0 - ME_C2_KEV * (1.0 / np.maximum(e2_n, 8.0) - 1.0 / np.maximum(es_n, 15.0))
        valid_n = np.abs(cos_n) <= 0.995
        if np.any(valid_n):
            out[0] = _fast_cone_backproject(
                vox, r1_n[valid_n], r2_n[valid_n], cos_n[valid_n],
                weights=np.ones(valid_n.sum(), dtype=np.float32), sigma_cos=0.095
            )

    # --- Ch 1 & Ch 2: BNCTEventNet 清洗、正逆散射序列恢复与散射角余弦解析修正 ---
    if ai_fwd_weights is not None and ai_rev_weights is not None:
        wf = ai_fwd_weights[two]
        wr = ai_rev_weights[two]
        w_tot = wf + wr
        use_rev = wr > wf
        keep = np.flatnonzero(w_tot > 0.28)
        if len(keep) > max_cones:
            top_k = np.argsort(-w_tot[keep])[:max_cones]
            keep = keep[top_k]
        if len(keep) > 0:
            rev_k = use_rev[keep]
            r1_c = np.where(rev_k[:, None], rb[keep], ra[keep])
            r2_c = np.where(rev_k[:, None], ra[keep], rb[keep])
            e1_c = np.where(rev_k, eb[keep], ea[keep])
            e2_c = np.where(rev_k, ea[keep], eb[keep])
            # 融合实测双点能量和与 477.6 keV 特征能谱约束计算散射角余弦，并叠加 AI 余弦残差修正
            e_tot_blend = 0.55 * 477.6 + 0.45 * (e1_c + e2_c)
            cos_c = 1.0 - ME_C2_KEV * (1.0 / np.maximum(e2_c, 8.0) - 1.0 / np.maximum(e_tot_blend, 20.0))
            if ai_delta_cos is not None:
                cos_c = cos_c + ai_delta_cos[two[keep]]
            ok = np.abs(cos_c) <= 0.995
            if np.any(ok):
                wk = w_tot[keep][ok].astype(np.float32)
                cos_ok = cos_c[ok]
                sin_ok = np.sqrt(np.maximum(1.0 - cos_ok * cos_ok, 0.04))
                out[1], out[2] = _fast_cone_backproject_dual(
                    vox, r1_c[ok], r2_c[ok], cos_ok, w1=wk, w2=wk / sin_ok, s1=0.075, s2=0.065
                )
    else:
        out[1] = out[0].copy()
        out[2] = out[0].copy()

    for c in range(3):
        mx = float(out[c].max())
        if mx > 1e-6:
            out[c] /= mx
    return out


def reconstruct_parallel_multihole(dig: Dict[str, np.ndarray], geom: BNCTSystemGeometry) -> np.ndarray:
    """平行板多孔准直器 1:1 投影成像（用于定量展示穿隔模糊与 40x40 mm 狭窄视场截断）。"""
    single = (dig["n_hits"] == 1) & (dig["layer_a"] == 0) & (dig["e_a"] >= 425.0) & (dig["e_a"] <= 525.0)
    det_map = np.zeros((geom.n_pix, geom.n_pix), dtype=np.float32)
    if np.any(single):
        np.add.at(det_map, (dig["pix_iy_a"][single], dig["pix_ix_a"][single]), 1.0)
    # 平行孔无缩像放大率 (1:1)，探测器 [-20, +20] mm 仅能覆盖源平面中心 [-20, +20] mm 区域！
    src_coords = np.linspace(-geom.fov_half_mm, geom.fov_half_mm, geom.img_grid, dtype=np.float32)
    u = (src_coords + geom.det_half_mm) / geom.pix_pitch_mm - 0.5
    img = _bilinear_resample_2d(det_map, u, u, zero_outside=True)
    mx = float(img.max())
    return img / mx if mx > 1e-6 else img


def reconstruct_coded_aperture_mura(dig: Dict[str, np.ndarray], geom: BNCTSystemGeometry) -> np.ndarray:
    """Rank-19 MURA 编码板周期互相关解码（用于定量展示 478/2223 keV 穿板与厚孔暗角导致的解码失效）。"""
    _, g_mat = build_rank19_mura_mask()
    single = (dig["n_hits"] == 1) & (dig["layer_a"] == 0) & (dig["e_a"] >= 425.0) & (dig["e_a"] <= 525.0)
    det_map = np.zeros((geom.n_pix, geom.n_pix), dtype=np.float32)
    if np.any(single):
        np.add.at(det_map, (dig["pix_iy_a"][single], dig["pix_ix_a"][single]), 1.0)
    d19 = det_map[:19, :19]
    d_zero = d19 - float(d19.mean())
    rec19 = np.fft.ifft2(np.conj(np.fft.fft2(d_zero)) * np.fft.fft2(g_mat)).real
    rec19 = np.fft.fftshift(rec19).astype(np.float32)
    rec19 = np.maximum(rec19 - np.median(rec19), 0.0)
    u = np.linspace(0.0, 18.0, geom.img_grid, dtype=np.float32)
    img = _bilinear_resample_2d(rec19, u, u, zero_outside=False)
    mx = float(img.max())
    return img / mx if mx > 1e-6 else img


@torch.no_grad()
def _fast_cone_backproject(
    vox_xyz: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    cos_theta: np.ndarray,
    weights: np.ndarray,
    sigma_cos: float = 0.08,
) -> np.ndarray:
    """基于 PyTorch 多线程 AVX 矢量化加速的康普顿散射圆锥体素高斯核快速投影算子。"""
    axis = r1 - r2
    anorm = np.linalg.norm(axis, axis=1, keepdims=True)
    valid = anorm[:, 0] > 1.0
    if not np.any(valid):
        return np.zeros(vox_xyz.shape[:2], dtype=np.float32)
    r1_t = torch.from_numpy(r1[valid].astype(np.float32))
    axis_u = torch.from_numpy((axis[valid] / anorm[valid]).astype(np.float32))
    cos_t = torch.from_numpy(cos_theta[valid].astype(np.float32)).unsqueeze(1)
    w_t = torch.from_numpy(weights[valid].astype(np.float32)).unsqueeze(0)

    g1, g2, _ = vox_xyz.shape
    vox_flat = torch.from_numpy(vox_xyz.reshape(-1, 3).astype(np.float32))  # (M, 3)
    dv = vox_flat.unsqueeze(0) - r1_t.unsqueeze(1)                         # (N, M, 3)
    inv_d = torch.rsqrt(torch.sum(dv * dv, dim=-1) + 1e-6)
    cos_geo = torch.bmm(dv, axis_u.unsqueeze(-1)).squeeze(-1) * inv_d      # (N, M)
    diff = cos_geo - cos_t
    ker = torch.exp(-(diff * diff) * (1.0 / (2.0 * sigma_cos * sigma_cos)))
    acc = torch.matmul(w_t, ker).reshape(g1, g2)
    return acc.numpy()


@torch.no_grad()
def _fast_cone_backproject_dual(
    vox_xyz: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    cos_theta: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    s1: float = 0.075,
    s2: float = 0.065,
) -> Tuple[np.ndarray, np.ndarray]:
    """单次共享几何余弦计算同步输出纯净圆锥图与周长归一化圆锥图。"""
    axis = r1 - r2
    anorm = np.linalg.norm(axis, axis=1, keepdims=True)
    valid = anorm[:, 0] > 1.0
    g1, g2, _ = vox_xyz.shape
    if not np.any(valid):
        z = np.zeros((g1, g2), dtype=np.float32)
        return z, z
    r1_t = torch.from_numpy(r1[valid].astype(np.float32))
    axis_u = torch.from_numpy((axis[valid] / anorm[valid]).astype(np.float32))
    cos_t = torch.from_numpy(cos_theta[valid].astype(np.float32)).unsqueeze(1)
    w1_t = torch.from_numpy(w1[valid].astype(np.float32)).unsqueeze(0)
    w2_t = torch.from_numpy(w2[valid].astype(np.float32)).unsqueeze(0)

    vox_flat = torch.from_numpy(vox_xyz.reshape(-1, 3).astype(np.float32))
    dv = vox_flat.unsqueeze(0) - r1_t.unsqueeze(1)
    inv_d = torch.rsqrt(torch.sum(dv * dv, dim=-1) + 1e-6)
    cos_geo = torch.bmm(dv, axis_u.unsqueeze(-1)).squeeze(-1) * inv_d
    d2 = (cos_geo - cos_t).square()
    acc1 = torch.matmul(w1_t, torch.exp(-d2 * (1.0 / (2.0 * s1 * s1)))).reshape(g1, g2)
    acc2 = torch.matmul(w2_t, torch.exp(-d2 * (1.0 / (2.0 * s2 * s2)))).reshape(g1, g2)
    return acc1.numpy(), acc2.numpy()


def _bilinear_resample_2d(src: np.ndarray, ux: np.ndarray, uy: np.ndarray, zero_outside: bool = True) -> np.ndarray:
    h, w = src.shape
    xx, yy = np.meshgrid(ux, uy)
    inside = (xx >= -0.5) & (xx <= w - 0.5) & (yy >= -0.5) & (yy <= h - 0.5)
    x0 = np.floor(xx).astype(np.int32)
    y0 = np.floor(yy).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xx - x0
    wy = yy - y0
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)
    val = (
        (1.0 - wx) * (1.0 - wy) * src[y0c, x0c]
        + wx * (1.0 - wy) * src[y0c, x1c]
        + (1.0 - wx) * wy * src[y1c, x0c]
        + wx * wy * src[y1c, x1c]
    )
    if zero_outside:
        val = np.where(inside, val, 0.0)
    return val.astype(np.float32)
