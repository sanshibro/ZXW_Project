"""
BNCT 478 keV 特征伽马成像 · 正算软件包核心物理与准直器输运模块
(Forward Simulation Package: BNCT 478 keV + Multi-Energy Background + 5 Geometries)

依托背景：硼中子俘获治疗 (BNCT) 中 10B(n, alpha)7Li* 发射 477.6 keV (478 keV) 瞬发特征伽马射线，
同时存在 1H(n, gamma)2H 的 2223 keV 强穿透伽马、511 keV 正电子湮灭伽马及连续散射本底。
支持五种几何体制统一正算对比：
  1. single_pinhole       : 厚钨双圆锥刀口单小孔准直（含刀口穿壁与孔壁小角散射）
  2. single_compton       : 无准直双层像素化 GAGG:Ce 阵列康普顿散射（含多普勒展宽与逃逸）
  3. joint_pinhole_compton: 前端厚钨单小孔 + 双层像素化阵列联合探测体制
  4. parallel_multihole   : 平行板多孔准直（对照验证：穿隔透射严重 + 1:1无缩像视场受限）
  5. coded_aperture_mura  : 钨编码板准直（对照验证：478/2223 keV穿板致对比度崩塌 + 厚孔暗角解码失真）
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

ME_C2_KEV = 510.99895  # 电子静止能量 (keV)

# NIST XCOM 线性衰减系数表 (单位: cm^-1)
# GAGG:Ce (Gd3Al2Ga3O12, 密度 6.63 g/cm^3): [能量 keV, 光电 mu_pe, 康普顿 mu_comp, 电子对 mu_pair]
_GAGG_XCOM_TABLE = np.array([
    [30.0,   78.50, 1.12, 0.00],
    [50.0,   21.80, 1.08, 0.00],
    [59.5,   48.20, 1.05, 0.00],  # Gd K-edge (50.2 keV) 跃升
    [80.0,   22.60, 0.99, 0.00],
    [100.0,  12.40, 0.94, 0.00],
    [150.0,   4.12, 0.84, 0.00],
    [200.0,   1.85, 0.76, 0.00],
    [300.0,   0.61, 0.65, 0.00],
    [400.0,   0.28, 0.57, 0.00],
    [477.6,   0.18, 0.525, 0.00], # BNCT 478 keV 特征伽马射线（典型交叉尴尬能区）
    [511.0,   0.15, 0.508, 0.00], # 正电子湮灭干扰线
    [661.7,   0.082, 0.445, 0.00],
    [1000.0,  0.032, 0.362, 0.00],
    [1500.0,  0.015, 0.292, 0.012],
    [2223.0,  0.008, 0.236, 0.038], # 1H(n,g)2H 氢俘获高能背景伽马
    [2500.0,  0.007, 0.220, 0.048],
], dtype=np.float64)

# 钨合金准直材料 (W-Ni-Fe, 密度 18.0 g/cm^3): [能量 keV, 总线性衰减系数 mu_tot (cm^-1)]
_TUNGSTEN_XCOM_TABLE = np.array([
    [30.0,   395.0],
    [50.0,   102.0],
    [69.5,    39.0],
    [70.0,   192.0],  # W K-edge (69.5 keV)
    [100.0,   78.5],
    [150.0,   27.4],
    [200.0,   13.6],
    [300.0,    5.62],
    [400.0,    3.38],
    [477.6,    2.56],  # 478 keV 自由程 ~3.9 mm，薄板易穿透、厚板孔壁散射显著
    [511.0,    2.34],
    [661.7,    1.76],
    [1000.0,   1.22],
    [1500.0,   0.92],
    [2223.0,   0.78],  # 2.223 MeV 自由程达 12.8 mm，极易穿透准直器形成强本底
    [2500.0,   0.76],
], dtype=np.float64)


def query_gagg_mu(energy_kev: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对数插值查询 GAGG:Ce 晶体的光电、康普顿与总线性衰减系数 (cm^-1)。"""
    e = np.clip(np.asarray(energy_kev, dtype=np.float64), 30.0, 2500.0)
    log_e = np.log(e)
    log_tab_e = np.log(_GAGG_XCOM_TABLE[:, 0])
    mu_pe = np.exp(np.interp(log_e, log_tab_e, np.log(np.maximum(_GAGG_XCOM_TABLE[:, 1], 1e-5))))
    mu_comp = np.exp(np.interp(log_e, log_tab_e, np.log(_GAGG_XCOM_TABLE[:, 2])))
    mu_pair = np.interp(e, _GAGG_XCOM_TABLE[:, 0], _GAGG_XCOM_TABLE[:, 3])
    return mu_pe, mu_comp, mu_pe + mu_comp + mu_pair


def query_tungsten_mu(energy_kev: np.ndarray) -> np.ndarray:
    """对数插值查询钨合金准直材料的总线性衰减系数 (cm^-1)。"""
    e = np.clip(np.asarray(energy_kev, dtype=np.float64), 30.0, 2500.0)
    return np.exp(np.interp(np.log(e), np.log(_TUNGSTEN_XCOM_TABLE[:, 0]), np.log(_TUNGSTEN_XCOM_TABLE[:, 1])))


@dataclass
class BNCTSystemGeometry:
    """BNCT 像素化阵列准直/康普顿成像系统几何配置参数。"""
    n_pix: int = 20                     # 20x20 像素化闪烁体阵列
    pix_pitch_mm: float = 2.0           # 像素中心节距 2.0 mm -> 敏感面 40 mm x 40 mm
    d1_thick_mm: float = 5.0            # 第一层 GAGG:Ce 厚度 5.0 mm (中心 z = 0)
    d2_thick_mm: float = 14.0           # 第二层 GAGG:Ce 厚度 14.0 mm
    layer_gap_mm: float = 24.0          # 前后两层中心间距 24.0 mm (第二层中心 z = -24.0 mm)
    collimator_z_mm: float = 36.0       # 准直器中心平面 z = +36.0 mm
    pinhole_thick_mm: float = 12.0      # 单小孔钨板厚度 12.0 mm
    pinhole_diam_mm: float = 2.6        # 双圆锥刀口中心孔径 2.6 mm
    pinhole_cone_deg: float = 37.0      # 双圆锥张角半角 37 deg
    source_z_mm: float = 180.0          # BNCT 目标肿瘤/水模中心平面距离 z_s = +180.0 mm
    fov_half_mm: float = 100.0          # 目标平面成像视场 [-100 mm, +100 mm]^2
    img_grid: int = 48                  # 重建空间分布网格 48 x 48

    @property
    def det_half_mm(self) -> float:
        return 0.5 * self.n_pix * self.pix_pitch_mm

    @property
    def magnification(self) -> float:
        return self.collimator_z_mm / self.source_z_mm  # M = 36 / 180 = 0.20


def build_rank19_mura_mask() -> Tuple[np.ndarray, np.ndarray]:
    """构造 19x19 素数阶 MURA 编码板开孔矩阵与平衡解码算子（用于编码板可行性对照验证）。"""
    p = 19
    qr = set((k * k) % p for k in range(1, p))
    c = np.array([1 if i in qr else -1 for i in range(p)], dtype=np.int32)
    a = np.zeros((p, p), dtype=np.float32)
    for i in range(p):
        for j in range(p):
            if i == 0:
                a[i, j] = 0.0
            elif j == 0:
                a[i, j] = 1.0
            else:
                a[i, j] = 1.0 if (c[i] * c[j] == 1) else 0.0
    g = np.where(a > 0.5, 1.0, -1.0).astype(np.float32)
    g[0, 0] = 1.0
    return a, g


def generate_bnct_boron_distribution(
    shape_type: str,
    rng: np.random.Generator,
    grid_size: int = 48,
    fov_half_mm: float = 100.0,
    fixed_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    生成 BNCT 治疗区不同位置与几何形态的 10B（478 keV 特征伽马源）空间分布真值标签 (48x48)。
    支持：单病灶热点、近距双病灶、浸润条带/管状源、环形坏死肿瘤源、含正常组织本底的不规则团块源。
    """
    coords = np.linspace(-fov_half_mm, fov_half_mm, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(coords, coords)
    img = np.zeros((grid_size, grid_size), dtype=np.float32)
    meta: Dict = {"shape_type": shape_type}
    p = fixed_params or {}

    if shape_type == "single_hotspot":
        cx = float(p.get("cx", rng.uniform(-58.0, 58.0)))
        cy = float(p.get("cy", rng.uniform(-58.0, 58.0)))
        sigma = float(p.get("sigma", rng.uniform(5.8, 8.5)))
        img = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        meta.update({"cx": cx, "cy": cy, "sigma": sigma})

    elif shape_type == "dual_hotspots":
        cx1 = float(p.get("cx1", rng.uniform(-50.0, -12.0)))
        cy1 = float(p.get("cy1", rng.uniform(-42.0, 42.0)))
        sep = float(p.get("sep", rng.uniform(32.0, 54.0)))
        ang = float(p.get("ang", rng.uniform(-0.7, 0.7)))
        cx2 = np.clip(cx1 + sep * math.cos(ang), -65.0, 65.0)
        cy2 = np.clip(cy1 + sep * math.sin(ang), -65.0, 65.0)
        ratio = float(p.get("ratio", rng.uniform(0.65, 1.0)))
        sigma = float(p.get("sigma", rng.uniform(5.5, 7.2)))
        img = np.exp(-((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2.0 * sigma ** 2)) + \
              ratio * np.exp(-((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2.0 * sigma ** 2))
        meta.update({"cx1": cx1, "cy1": cy1, "cx2": float(cx2), "cy2": float(cy2), "sep_mm": sep})

    elif shape_type == "tubular_line":
        cx = float(p.get("cx", rng.uniform(-25.0, 25.0)))
        cy = float(p.get("cy", rng.uniform(-25.0, 25.0)))
        length = float(p.get("length", rng.uniform(52.0, 88.0)))
        theta = float(p.get("theta", rng.uniform(0.0, math.pi)))
        width = float(p.get("width", rng.uniform(5.2, 7.0)))
        dx, dy = math.cos(theta), math.sin(theta)
        proj = (xx - cx) * dx + (yy - cy) * dy
        proj_clip = np.clip(proj, -0.5 * length, 0.5 * length)
        dist2 = (xx - (cx + proj_clip * dx)) ** 2 + (yy - (cy + proj_clip * dy)) ** 2
        img = np.exp(-dist2 / (2.0 * width ** 2))
        meta.update({"cx": cx, "cy": cy, "length": length, "theta": theta})

    elif shape_type == "necrotic_ring":
        cx = float(p.get("cx", rng.uniform(-28.0, 28.0)))
        cy = float(p.get("cy", rng.uniform(-28.0, 28.0)))
        radius = float(p.get("radius", rng.uniform(24.0, 36.0)))
        ring_w = float(p.get("ring_w", rng.uniform(5.2, 7.2)))
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img = np.exp(-((rr - radius) ** 2) / (2.0 * ring_w ** 2))
        meta.update({"cx": cx, "cy": cy, "radius": radius})

    elif shape_type == "irregular_tumor_tn":
        # 模拟真实 BNCT 临床场景：不规则分叶状肿瘤高硼区 + 周围正常脑/软组织低硼底色 (T/N = 4:1)
        cx = float(p.get("cx", rng.uniform(-32.0, 32.0)))
        cy = float(p.get("cy", rng.uniform(-32.0, 32.0)))
        s1 = float(p.get("s1", rng.uniform(10.0, 15.0)))
        s2 = float(p.get("s2", rng.uniform(7.5, 11.5)))
        off_x = float(p.get("off_x", rng.uniform(10.0, 18.0) * rng.choice([-1.0, 1.0])))
        off_y = float(p.get("off_y", rng.uniform(8.0, 16.0) * rng.choice([-1.0, 1.0])))
        tumor = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * s1 ** 2)) + \
                0.78 * np.exp(-((xx - cx - off_x) ** 2 + (yy - cy - off_y) ** 2) / (2.0 * s2 ** 2))
        tumor = tumor / max(float(tumor.max()), 1e-6)
        # 正常组织椭圆水模范围缓慢变化的低浓度本底 (约 1/4.2 峰值强度)
        tissue_mask = np.exp(-((xx / 78.0) ** 2 + (yy / 78.0) ** 2) ** 2)
        tn_bg_level = float(p.get("tn_bg", rng.uniform(0.08, 0.16)))
        img = tumor + tn_bg_level * tissue_mask
        meta.update({"cx": cx, "cy": cy, "tn_ratio": round(1.0 / max(tn_bg_level, 1e-3), 2)})

    else:
        raise ValueError(f"Unknown BNCT source shape: {shape_type}")

    img = np.clip(img, 0.0, None)
    mx = float(img.max())
    if mx > 1e-8:
        img = img / mx
    return img.astype(np.float32), meta


def _sample_klein_nishina_cos_theta(e0_kev: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """矢量化 Kahn/Carlson 拒绝采样精确生成康普顿散射角余弦 cos(theta)。"""
    n = len(e0_kev)
    alpha = e0_kev / ME_C2_KEV
    cos_out = np.zeros(n, dtype=np.float64)
    active = np.arange(n)
    while len(active) > 0:
        m = len(active)
        a = alpha[active]
        r1 = rng.random(m)
        r2 = rng.random(m)
        r3 = rng.random(m)
        p_branch = (1.0 + 2.0 * a) / (9.0 + 2.0 * a)
        b1 = r1 <= p_branch
        x = np.empty(m, dtype=np.float64)
        acc = np.zeros(m, dtype=bool)
        if np.any(b1):
            x1 = 1.0 + 2.0 * a[b1] * r2[b1]
            acc[b1] = r3[b1] <= 4.0 * (1.0 / x1 - 1.0 / (x1 * x1))
            x[b1] = x1
        b2 = ~b1
        if np.any(b2):
            x2 = (1.0 + 2.0 * a[b2]) / (1.0 + 2.0 * a[b2] * r2[b2])
            cos_tmp = 1.0 - (x2 - 1.0) / a[b2]
            acc[b2] = r3[b2] <= 0.5 * (cos_tmp * cos_tmp + 1.0 / x2)
            x[b2] = x2
        done = active[acc]
        cos_out[done] = 1.0 - (x[acc] - 1.0) / alpha[done]
        active = active[~acc]
    return np.clip(cos_out, -1.0, 1.0)


def simulate_bnct_forward_transport(
    gt_boron_map: np.ndarray,
    geom: BNCTSystemGeometry,
    mode: str,
    n_signal_photons: int,
    bg_ratio: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """
    运行 BNCT 478 keV + 多能量复杂干扰场在指定准直/康普顿模式下的三维蒙特卡罗输运模拟。

    参数:
      mode: 'single_pinhole' | 'single_compton' | 'joint_pinhole_compton' |
            'parallel_multihole' | 'coded_aperture_mura'
      n_signal_photons: 朝向探测器有效接收立体角内发射的 478 keV 目标特征伽马光子数
      bg_ratio: 复杂辐射场非目标光子数与 478 keV 目标光子数之比（含 511 keV、2223 keV 及连续散射本底）
    返回:
      理想连续空间坐标与真实物理标签事件字典（供 DetectorDigitizer 叠加硬件噪声与分辨率退化）
    """
    grid_n = gt_boron_map.shape[0]
    coords = np.linspace(-geom.fov_half_mm, geom.fov_half_mm, grid_n, dtype=np.float64)
    cell_step = 2.0 * geom.fov_half_mm / max(grid_n - 1, 1)

    # 1. 抽样 478 keV 目标特征伽马射线初始起点与能量
    weights = np.maximum(gt_boron_map.ravel().astype(np.float64), 0.0)
    w_sum = weights.sum()
    if w_sum <= 1e-12:
        weights = np.ones_like(weights) / weights.size
    else:
        weights = weights / w_sum

    idx_sig = rng.choice(weights.size, size=n_signal_photons, p=weights)
    iy_s, ix_s = np.divmod(idx_sig, grid_n)
    xs_sig = coords[ix_s] + rng.uniform(-0.5 * cell_step, 0.5 * cell_step, size=n_signal_photons)
    ys_sig = coords[iy_s] + rng.uniform(-0.5 * cell_step, 0.5 * cell_step, size=n_signal_photons)
    e0_sig = np.full(n_signal_photons, 477.6, dtype=np.float64)
    is_target_sig = np.ones(n_signal_photons, dtype=bool)

    # 2. 抽样 BNCT 复杂辐射场非单一能量背景光子 (2223 keV H俘获 + 511 keV 湮灭 + 90-360 keV 散射连续谱)
    n_bg = int(round(n_signal_photons * max(bg_ratio, 0.0)))
    if n_bg > 0:
        xs_bg = rng.uniform(-geom.fov_half_mm * 1.15, geom.fov_half_mm * 1.15, size=n_bg)
        ys_bg = rng.uniform(-geom.fov_half_mm * 1.15, geom.fov_half_mm * 1.15, size=n_bg)
        bg_type_u = rng.random(n_bg)
        e0_bg = np.empty(n_bg, dtype=np.float64)
        # 42% 为 2.223 MeV 氢俘获伽马及其大角度水模一次散射高能光子，33% 为 511 keV 湮灭光子，25% 为 90-380 keV 连续散射谱
        m_h = bg_type_u < 0.42
        m_511 = (bg_type_u >= 0.42) & (bg_type_u < 0.75)
        m_cont = ~(m_h | m_511)
        e0_bg[m_h] = np.where(rng.random(m_h.sum()) < 0.65, 2223.0, rng.uniform(680.0, 1850.0, size=m_h.sum()))
        e0_bg[m_511] = 511.0
        e0_bg[m_cont] = rng.uniform(90.0, 385.0, size=m_cont.sum())
        is_target_bg = np.zeros(n_bg, dtype=bool)

        xs = np.concatenate([xs_sig, xs_bg])
        ys = np.concatenate([ys_sig, ys_bg])
        e0 = np.concatenate([e0_sig, e0_bg])
        is_target_478 = np.concatenate([is_target_sig, is_target_bg])
    else:
        xs, ys, e0, is_target_478 = xs_sig, ys_sig, e0_sig, is_target_sig

    n_tot = len(e0)
    zs = np.full(n_tot, geom.source_z_mm, dtype=np.float64)

    # 针对含单小孔的模式采用双区域分层立体角重要性无偏抽样：
    # 50% 样本定向瞄准准直孔及刀口邻域 (半径 7.5 mm)，50% 覆盖全探测器敏感面，并记录精确立体角权重避免稀疏欠采样
    det_lim = geom.det_half_mm * 1.04
    if mode in ("single_pinhole", "joint_pinhole_compton"):
        use_ap_focus = rng.random(n_tot) < 0.52
        # 在准直平面 z_col 处瞄准中心半径 7.5 mm 圆域，反推 z=0 平面落点
        r_ap = 7.5 * np.sqrt(rng.random(n_tot))
        phi_ap = rng.uniform(0.0, 2.0 * math.pi, size=n_tot)
        x_ap = r_ap * np.cos(phi_ap)
        y_ap = r_ap * np.sin(phi_ap)
        scale_to_d1 = geom.source_z_mm / (geom.source_z_mm - geom.collimator_z_mm)
        tx_ap = xs + (x_ap - xs) * scale_to_d1
        ty_ap = ys + (y_ap - ys) * scale_to_d1

        tx_full = rng.uniform(-det_lim, det_lim, size=n_tot)
        ty_full = rng.uniform(-det_lim, det_lim, size=n_tot)
        tx = np.where(use_ap_focus, tx_ap, tx_full)
        ty = np.where(use_ap_focus, ty_ap, ty_full)
        # 面积比权重校正，保持小孔透过率与全探测器穿透本底的真实物理比例
        area_ap_proj = math.pi * (7.5 * scale_to_d1) ** 2
        area_full = (2.0 * det_lim) ** 2
        omega_weight = np.where(use_ap_focus, area_ap_proj / (0.52 * area_full), 1.0 / 0.48)
        omega_weight = np.clip(omega_weight, 0.08, 2.5)
    else:
        tx = rng.uniform(-det_lim, det_lim, size=n_tot)
        ty = rng.uniform(-det_lim, det_lim, size=n_tot)
        omega_weight = np.ones(n_tot, dtype=np.float64)

    dx = tx - xs
    dy = ty - ys
    dz = 0.0 - zs
    inv_norm = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
    ux, uy, uz = dx * inv_norm, dy * inv_norm, dz * inv_norm  # uz < 0

    # 3. 准直器平面输运（穿孔、刀口穿壁、孔壁散射、穿隔或编码板穿透）
    survive_col = np.ones(n_tot, dtype=bool)
    wall_scattered = np.zeros(n_tot, dtype=bool)
    passed_clean_aperture = np.zeros(n_tot, dtype=bool)
    coinc_only_zone = np.zeros(n_tot, dtype=bool)
    mu_w_cm = query_tungsten_mu(e0)  # cm^-1
    mu_w_mm = 0.1 * mu_w_cm          # mm^-1

    t_col = (geom.collimator_z_mm - zs) / uz
    x_col = xs + ux * t_col
    y_col = ys + uy * t_col
    cos_inc = np.abs(uz)

    if mode == "single_compton":
        # 无机械准直器，全部光子直达第一层像素化闪烁体阵列
        passed_clean_aperture[:] = False

    elif mode in ("single_pinhole", "joint_pinhole_compton"):
        r_col = np.sqrt(x_col * x_col + y_col * y_col)
        r0 = 0.5 * geom.pinhole_diam_mm
        cot_cone = 1.0 / math.tan(math.radians(geom.pinhole_cone_deg))

        if mode == "joint_pinhole_compton":
            # 联合探测几何：中心区为 12.0 mm 厚钨双圆锥单小孔提供准直投影锚定；
            # 外围散射环区 (r_col > 8.5 mm) 配置 2.0 mm 低能散射滤片并启用双层符合触发（单层外环事件硬件门控抑制）
            coinc_only_zone = r_col > 8.5
            path_w_mm = np.where(
                r_col <= r0,
                0.0,
                np.where(
                    coinc_only_zone,
                    2.0 / np.maximum(cos_inc, 0.30),
                    np.minimum(geom.pinhole_thick_mm / np.maximum(cos_inc, 0.25), 2.0 * (r_col - r0) * cot_cone),
                ),
            )
        else:
            path_w_mm = np.where(
                r_col <= r0,
                0.0,
                np.minimum(geom.pinhole_thick_mm / np.maximum(cos_inc, 0.25), 2.0 * (r_col - r0) * cot_cone),
            )
        trans_prob = np.exp(-mu_w_mm * path_w_mm)
        u_col = rng.random(n_tot)
        survive_col = u_col < trans_prob
        passed_clean_aperture = survive_col & (r_col <= r0 * 1.18)

        # 刀口附近 (r0 < r_col < r0 + 2.8 mm) 存在明显孔壁小角康普顿散射效应
        near_wall = (r_col > r0) & (r_col < r0 + 2.8) & (~survive_col)
        scatter_wall_prob = 0.22 * np.exp(-0.25 * (r_col - r0))
        do_wall_scat = near_wall & (rng.random(n_tot) < scatter_wall_prob)
        if np.any(do_wall_scat):
            survive_col[do_wall_scat] = True
            wall_scattered[do_wall_scat] = True
            # 孔壁小角散射偏折 (8~24 deg) 并伴随康普顿能量损失
            theta_w = rng.uniform(math.radians(8.0), math.radians(24.0), size=do_wall_scat.sum())
            phi_w = rng.uniform(0.0, 2.0 * math.pi, size=do_wall_scat.sum())
            e_old = e0[do_wall_scat]
            e0[do_wall_scat] = e_old / (1.0 + (e_old / ME_C2_KEV) * (1.0 - np.cos(theta_w)))
            ux[do_wall_scat] = np.clip(ux[do_wall_scat] + np.sin(theta_w) * np.cos(phi_w), -0.85, 0.85)
            uy[do_wall_scat] = np.clip(uy[do_wall_scat] + np.sin(theta_w) * np.sin(phi_w), -0.85, 0.85)
            uz[do_wall_scat] = -np.sqrt(np.maximum(1.0 - ux[do_wall_scat] ** 2 - uy[do_wall_scat] ** 2, 0.08))
            # 起点重设至准直孔平面
            xs[do_wall_scat] = x_col[do_wall_scat]
            ys[do_wall_scat] = y_col[do_wall_scat]
            zs[do_wall_scat] = geom.collimator_z_mm

    elif mode == "parallel_multihole":
        # 平行板多孔准直器：孔节距 2.0 mm，孔宽 1.35 mm，钨隔片厚 0.65 mm，孔长 12.0 mm
        pitch = geom.pix_pitch_mm
        hole_half = 0.675
        L_col = 12.0
        z_top = geom.collimator_z_mm + 0.5 * L_col
        z_bot = geom.collimator_z_mm - 0.5 * L_col
        x_top = xs + ux * ((z_top - zs) / uz)
        y_top = ys + uy * ((z_top - zs) / uz)
        x_bot = xs + ux * ((z_bot - zs) / uz)
        y_bot = ys + uy * ((z_bot - zs) / uz)

        in_footprint = (np.abs(x_top) <= geom.det_half_mm) & (np.abs(y_top) <= geom.det_half_mm)
        ix_top = np.floor((x_top + geom.det_half_mm) / pitch)
        iy_top = np.floor((y_top + geom.det_half_mm) / pitch)
        ix_bot = np.floor((x_bot + geom.det_half_mm) / pitch)
        iy_bot = np.floor((y_bot + geom.det_half_mm) / pitch)
        loc_x = np.abs(((x_top + geom.det_half_mm) % pitch) - 0.5 * pitch)
        loc_y = np.abs(((y_top + geom.det_half_mm) % pitch) - 0.5 * pitch)

        same_hole = in_footprint & (ix_top == ix_bot) & (iy_top == iy_bot) & (loc_x <= hole_half) & (loc_y <= hole_half)
        # 斜射穿过若干层 0.65 mm 薄钨隔片（穿隔效应在 478 keV 与 2223 keV 下极其严重！）
        n_septa = np.abs(ix_bot - ix_top) + np.abs(iy_bot - iy_top)
        septa_frac = np.where(same_hole, 0.0, np.clip(0.32 + 0.08 * np.minimum(n_septa, 5.0), 0.32, 0.72))
        path_w_mm = np.where(in_footprint, septa_frac * (L_col / np.maximum(cos_inc, 0.25)), L_col * 1.2)
        survive_col = rng.random(n_tot) < np.exp(-mu_w_mm * path_w_mm)
        passed_clean_aperture = survive_col & same_hole
        wall_scattered = survive_col & (~same_hole)

    elif mode == "coded_aperture_mura":
        # Rank-19 钨编码板：厚度 6.0 mm，单元尺寸 2.0 mm (深宽比 3:1，斜射孔壁自遮挡严重 + 478/2223 keV 高穿透)
        mask_a, _ = build_rank19_mura_mask()
        tiled = np.tile(mask_a, (2, 2))[:37, :37]  # 37x37 镶嵌阵列
        elem_mm = 2.0 * (1.0 - geom.collimator_z_mm / geom.source_z_mm)  # 近场匹配 1.60 mm
        half_m = 0.5 * 37 * elem_mm
        t_mask = 6.0
        ix_m = np.floor((x_col + half_m) / elem_mm).astype(np.int32)
        iy_m = np.floor((y_col + half_m) / elem_mm).astype(np.int32)
        valid_m = (ix_m >= 0) & (ix_m < 37) & (iy_m >= 0) & (iy_m < 37)
        is_open = np.zeros(n_tot, dtype=bool)
        is_open[valid_m] = tiled[iy_m[valid_m], ix_m[valid_m]] > 0.5

        # 厚编码孔斜射暗角截断因子 (vignetting): tan(theta) * t_mask / elem_mm
        tan_x = np.abs(ux / uz)
        tan_y = np.abs(uy / uz)
        open_vig_x = np.clip(1.0 - tan_x * (t_mask / elem_mm) * 0.55, 0.08, 1.0)
        open_vig_y = np.clip(1.0 - tan_y * (t_mask / elem_mm) * 0.55, 0.08, 1.0)
        clean_open = is_open & (rng.random(n_tot) < (open_vig_x * open_vig_y))

        path_w_mm = np.where(clean_open, 0.0, t_mask / np.maximum(cos_inc, 0.30))
        survive_col = rng.random(n_tot) < np.exp(-mu_w_mm * path_w_mm)
        passed_clean_aperture = survive_col & clean_open
        wall_scattered = survive_col & (~clean_open)

    else:
        raise ValueError(f"Unsupported simulation mode: {mode}")

    # 仅保留穿过准直器存活的光子继续与双层像素化 GAGG:Ce 阵列作用
    keep = np.flatnonzero(survive_col)
    if len(keep) == 0:
        return _empty_event_dict()

    xs, ys, zs = xs[keep], ys[keep], zs[keep]
    ux, uy, uz = ux[keep], uy[keep], uz[keep]
    e0 = e0[keep]
    is_target_478 = is_target_478[keep]
    wall_scattered = wall_scattered[keep]
    passed_clean_aperture = passed_clean_aperture[keep]
    coinc_only_zone = coinc_only_zone[keep]

    # 4. 第一层 GAGG:Ce 像素阵列输运 (中心 z = 0, 厚度 d1_thick_mm)
    z1_top = 0.5 * geom.d1_thick_mm
    t_d1 = (z1_top - zs) / uz
    x1_in = xs + ux * t_d1
    y1_in = ys + uy * t_d1
    in_d1 = (np.abs(x1_in) < geom.det_half_mm) & (np.abs(y1_in) < geom.det_half_mm)

    idx_d1 = np.flatnonzero(in_d1)
    if len(idx_d1) == 0:
        return _empty_event_dict()

    x1_in, y1_in = x1_in[idx_d1], y1_in[idx_d1]
    ux, uy, uz = ux[idx_d1], uy[idx_d1], uz[idx_d1]
    e0 = e0[idx_d1]
    is_target_478 = is_target_478[idx_d1]
    wall_scattered = wall_scattered[idx_d1]
    passed_clean_aperture = passed_clean_aperture[idx_d1]
    coinc_only_zone = coinc_only_zone[idx_d1]

    m = len(e0)
    mu_pe1, mu_comp1, mu_tot1 = query_gagg_mu(e0)
    mu_tot1_mm = 0.1 * mu_tot1
    max_step1 = geom.d1_thick_mm / np.maximum(np.abs(uz), 0.15)
    free_path1 = rng.exponential(scale=1.0 / np.maximum(mu_tot1_mm, 1e-5))
    hit_d1 = free_path1 < max_step1

    # 未在第一层作用的光子直飞第二层 GAGG:Ce (中心 z = -layer_gap_mm)
    # 先收集：Type A (第一层单点光电/逃逸) 与 Type B (第一层康普顿散射 -> 第二层作用)
    events_r1, events_e1, events_l1 = [], [], []
    events_r2, events_e2, events_l2 = [], [], []
    events_nhits, events_true_cls, events_true_cos = [], [], []
    events_e0_orig = []

    # --- 分支 4A: 在第一层发生相互作用 ---
    id_h1 = np.flatnonzero(hit_d1)
    if len(id_h1) > 0:
        fp1 = free_path1[id_h1]
        xh1 = x1_in[id_h1] + ux[id_h1] * fp1
        yh1 = y1_in[id_h1] + uy[id_h1] * fp1
        zh1 = z1_top + uz[id_h1] * fp1
        eh0 = e0[id_h1]
        pe_prob1 = mu_pe1[id_h1] / np.maximum(mu_tot1[id_h1], 1e-6)
        is_pe1 = rng.random(len(id_h1)) < pe_prob1

        # (i) 第一层光电全吸收单点事例（仅非符合门控区记录单层触发）
        pe_mask = np.flatnonzero(is_pe1 & (~coinc_only_zone[id_h1]))
        if len(pe_mask) > 0:
            k = id_h1[pe_mask]
            n_pe = len(pe_mask)
            r1 = np.column_stack([xh1[pe_mask], yh1[pe_mask], zh1[pe_mask]])
            e1 = eh0[pe_mask]
            # 真值类别定义:
            # 0 = 有效 478 keV 准直单点投影事例
            # 3 = 非目标能量干扰事例 (511 / 2223 / 散射谱)
            # 4 = 孔壁散射/穿壁杂散本底事例
            cls_pe = np.where(
                ~is_target_478[k],
                3,
                np.where(passed_clean_aperture[k] & (~wall_scattered[k]), 0, 4),
            )
            events_r1.append(r1)
            events_e1.append(e1)
            events_l1.append(np.zeros(n_pe, dtype=np.int32))
            events_r2.append(np.zeros((n_pe, 3), dtype=np.float64))
            events_e2.append(np.zeros(n_pe, dtype=np.float64))
            events_l2.append(np.full(n_pe, -1, dtype=np.int32))
            events_nhits.append(np.ones(n_pe, dtype=np.int32))
            events_true_cls.append(cls_pe)
            events_true_cos.append(np.zeros(n_pe, dtype=np.float64))
            events_e0_orig.append(eh0[pe_mask])

        # (ii) 第一层发生康普顿散射
        cp_mask = np.flatnonzero(~is_pe1)
        if len(cp_mask) > 0:
            k = id_h1[cp_mask]
            n_cp = len(cp_mask)
            r1 = np.column_stack([xh1[cp_mask], yh1[cp_mask], zh1[cp_mask]])
            e_inc = eh0[cp_mask]
            cos_th = _sample_klein_nishina_cos_theta(e_inc, rng)
            sin_th = np.sqrt(np.maximum(1.0 - cos_th * cos_th, 0.0))
            phi_s = rng.uniform(0.0, 2.0 * math.pi, size=n_cp)

            # 计入 GAGG:Ce 束缚电子壳层多普勒能量展宽 (Hartree-Fock 冲量近似等效动量涨落)
            e_scat_ideal = e_inc / (1.0 + (e_inc / ME_C2_KEV) * (1.0 - cos_th))
            doppler_sigma_kev = 2.6 * np.sqrt(np.maximum(1.0 - cos_th, 0.05)) * (e_inc / 477.6) ** 0.65
            e_scat = np.clip(e_scat_ideal + rng.normal(0.0, doppler_sigma_kev), 8.0, e_inc - 2.0)
            e_dep1 = e_inc - e_scat

            # 旋转散射光子方向矢量 d_in -> d_scat
            dx0, dy0, dz0 = ux[k], uy[k], uz[k]
            perp = np.sqrt(np.maximum(dx0 * dx0 + dy0 * dy0, 1e-8))
            e1x = np.where(perp > 1e-3, -dy0 / perp, 1.0)
            e1y = np.where(perp > 1e-3, dx0 / perp, 0.0)
            e2x = dy0 * 0.0 - dz0 * e1y
            e2y = dz0 * e1x - dx0 * 0.0
            e2z = dx0 * e1y - dy0 * e1x
            sx = cos_th * dx0 + sin_th * (np.cos(phi_s) * e1x + np.sin(phi_s) * e2x)
            sy = cos_th * dy0 + sin_th * (np.cos(phi_s) * e1y + np.sin(phi_s) * e2y)
            sz = cos_th * dz0 + sin_th * (np.sin(phi_s) * e2z)
            snorm = 1.0 / np.sqrt(sx * sx + sy * sy + sz * sz)
            sx, sy, sz = sx * snorm, sy * snorm, sz * snorm

            # 追踪散射光子飞向第二层 GAGG:Ce (中心 z = -layer_gap_mm, 顶面 z2_top)
            z2_top = -geom.layer_gap_mm + 0.5 * geom.d2_thick_mm
            go_down = sz < -0.08
            t_to_d2 = np.where(go_down, (z2_top - r1[:, 2]) / np.minimum(sz, -1e-4), -1.0)
            x2_in = r1[:, 0] + sx * t_to_d2
            y2_in = r1[:, 1] + sy * t_to_d2
            reach_d2 = go_down & (t_to_d2 > 0.0) & (np.abs(x2_in) < geom.det_half_mm) & (np.abs(y2_in) < geom.det_half_mm)

            mu_pe2, mu_comp2, mu_tot2 = query_gagg_mu(e_scat)
            mu_tot2_mm = 0.1 * mu_tot2
            max_step2 = geom.d2_thick_mm / np.maximum(np.abs(sz), 0.12)
            fp2 = rng.exponential(scale=1.0 / np.maximum(mu_tot2_mm, 1e-5))
            hit_d2 = reach_d2 & (fp2 < max_step2)

            # 4A-(ii)-1: 散射光子未命中第二层而逃逸 -> 形成单层不完全沉积逃逸本底 (Class 4，仅非符合门控区触发单层)
            esc1 = np.flatnonzero((~hit_d2) & (~coinc_only_zone[k]))
            if len(esc1) > 0:
                n_e1 = len(esc1)
                events_r1.append(r1[esc1])
                events_e1.append(e_dep1[esc1])
                events_l1.append(np.zeros(n_e1, dtype=np.int32))
                events_r2.append(np.zeros((n_e1, 3), dtype=np.float64))
                events_e2.append(np.zeros(n_e1, dtype=np.float64))
                events_l2.append(np.full(n_e1, -1, dtype=np.int32))
                events_nhits.append(np.ones(n_e1, dtype=np.int32))
                events_true_cls.append(np.where(is_target_478[k[esc1]], 4, 3))
                events_true_cos.append(cos_th[esc1])
                events_e0_orig.append(e_inc[esc1])

            # 4A-(ii)-2: 命中第二层形成双作用点符合事例！
            coinc = np.flatnonzero(hit_d2)
            if len(coinc) > 0:
                n_c = len(coinc)
                r2 = np.column_stack([
                    x2_in[coinc] + sx[coinc] * fp2[coinc],
                    y2_in[coinc] + sy[coinc] * fp2[coinc],
                    z2_top + sz[coinc] * fp2[coinc],
                ])
                # 在 14 mm 厚 GAGG:Ce 第二层内考虑一次光电吸收或二次散射后吸收 vs 部分能量逃逸
                full_abs_prob = np.clip((mu_pe2[coinc] + 0.58 * mu_comp2[coinc]) / np.maximum(mu_tot2[coinc], 1e-6), 0.25, 0.92)
                is_full_abs = rng.random(n_c) < full_abs_prob
                esc_frac = rng.uniform(0.30, 0.78, size=n_c)
                e_dep2 = np.where(is_full_abs, e_scat[coinc], e_scat[coinc] * esc_frac)

                # 在电子学读出中，约 26% 的双点事例因时间抖动或按能量大小排序导致触发槽 A/B 颠倒 (模拟 B->A 序列歧义！)
                swap = rng.random(n_c) < 0.26
                ra = np.where(swap[:, None], r2, r1[coinc])
                rb = np.where(swap[:, None], r1[coinc], r2)
                ea = np.where(swap, e_dep2, e_dep1[coinc])
                eb = np.where(swap, e_dep1[coinc], e_dep2)
                la = np.where(swap, 1, 0).astype(np.int32)
                lb = np.where(swap, 0, 1).astype(np.int32)

                # 真值类别:
                # 1 = 有效 478 keV 全吸收正向康普顿 (A->B)
                # 2 = 有效 478 keV 全吸收逆向康普顿 (B->A，需翻转散射轴)
                # 3 = 非 478 keV (511 / 2223 keV 等) 复杂辐射场双层干扰事例
                # 4 = 478 keV 逃逸不完全沉积或孔壁散射畸变双点事例
                valid_478_comp = is_target_478[k[coinc]] & is_full_abs & (~wall_scattered[k[coinc]])
                cls_c = np.where(
                    ~is_target_478[k[coinc]],
                    3,
                    np.where(valid_478_comp, np.where(swap, 2, 1), 4),
                )
                events_r1.append(ra)
                events_e1.append(ea)
                events_l1.append(la)
                events_r2.append(rb)
                events_e2.append(eb)
                events_l2.append(lb)
                events_nhits.append(np.full(n_c, 2, dtype=np.int32))
                events_true_cls.append(cls_c)
                events_true_cos.append(cos_th[coinc])
                events_e0_orig.append(e_inc[coinc])

    # --- 分支 4B: 穿透第一层直达第二层，部分发生背向康普顿反跳回第一层 (真实物理 B->A 反散射事例！) ---
    id_pass1 = np.flatnonzero(~hit_d1)
    if len(id_pass1) > 0:
        z2_top = -geom.layer_gap_mm + 0.5 * geom.d2_thick_mm
        t2 = (z2_top - z1_top) / uz[id_pass1]
        x2p = x1_in[id_pass1] + ux[id_pass1] * t2
        y2p = y1_in[id_pass1] + uy[id_pass1] * t2
        in_d2 = (np.abs(x2p) < geom.det_half_mm) & (np.abs(y2p) < geom.det_half_mm)
        k2 = id_pass1[in_d2]
        if len(k2) > 0:
            x2p, y2p = x2p[in_d2], y2p[in_d2]
            e_p = e0[k2]
            mu_pe2, mu_comp2, mu_tot2 = query_gagg_mu(e_p)
            fp2 = rng.exponential(scale=1.0 / np.maximum(0.1 * mu_tot2, 1e-5))
            max_s2 = geom.d2_thick_mm / np.maximum(np.abs(uz[k2]), 0.15)
            hit2 = fp2 < max_s2
            kh = np.flatnonzero(hit2)
            if len(kh) > 0:
                kk = k2[kh]
                r2_hit = np.column_stack([
                    x2p[kh] + ux[kk] * fp2[kh],
                    y2p[kh] + uy[kk] * fp2[kh],
                    z2_top + uz[kk] * fp2[kh],
                ])
                # (i) 第二层 (14 mm 厚吸收体) 单点光电/多次散射全能吸收 -> 极大地增强穿过小孔的 478 keV 准直投影效率！
                pe2_prob = np.clip((mu_pe2[kh] + 0.42 * mu_comp2[kh]) / np.maximum(mu_tot2[kh], 1e-6), 0.18, 0.85)
                u_d2_fate = rng.random(len(kh))
                is_pe2 = u_d2_fate < pe2_prob
                ip2 = np.flatnonzero(is_pe2 & (~coinc_only_zone[kk]))
                if len(ip2) > 0:
                    kp2 = kk[ip2]
                    np2 = len(ip2)
                    cls_p2 = np.where(
                        ~is_target_478[kp2],
                        3,
                        np.where(passed_clean_aperture[kp2] & (~wall_scattered[kp2]), 0, 4),
                    )
                    events_r1.append(r2_hit[ip2])
                    events_e1.append(e_p[kh[ip2]])
                    events_l1.append(np.ones(np2, dtype=np.int32))
                    events_r2.append(np.zeros((np2, 3), dtype=np.float64))
                    events_e2.append(np.zeros(np2, dtype=np.float64))
                    events_l2.append(np.full(np2, -1, dtype=np.int32))
                    events_nhits.append(np.ones(np2, dtype=np.int32))
                    events_true_cls.append(cls_p2)
                    events_true_cos.append(np.zeros(np2, dtype=np.float64))
                    events_e0_orig.append(e_p[kh[ip2]])

                # (ii) 约 9% 大角度背散射光子反跳回第一层形成真实物理后->前 (B->A) 双点事例
                back_prob = 0.09 * (mu_comp2[kh] / np.maximum(mu_tot2[kh], 1e-6))
                do_back = (~is_pe2) & (rng.random(len(kh)) < back_prob)
                ib = np.flatnonzero(do_back)
                if len(ib) > 0:
                    kb = kk[ib]
                    nb = len(ib)
                    cos_b = rng.uniform(-0.92, -0.35, size=nb)  # 大角度背散射
                    e_ib = e_p[kh[ib]]
                    e_back = e_ib / (1.0 + (e_ib / ME_C2_KEV) * (1.0 - cos_b))
                    e_d2 = e_ib - e_back
                    r1_back = np.column_stack([
                        np.clip(r2_hit[ib, 0] + rng.normal(0.0, 5.5, size=nb), -geom.det_half_mm + 1.0, geom.det_half_mm - 1.0),
                        np.clip(r2_hit[ib, 1] + rng.normal(0.0, 5.5, size=nb), -geom.det_half_mm + 1.0, geom.det_half_mm - 1.0),
                        rng.uniform(-0.5 * geom.d1_thick_mm, 0.5 * geom.d1_thick_mm, size=nb),
                    ])
                    # 读出默认将前层记为槽 A，后层记为槽 B -> 因此真实物理顺序为 B -> A (Class 2)！
                    cls_b = np.where(is_target_478[kb] & (~wall_scattered[kb]), 2, np.where(~is_target_478[kb], 3, 4))
                    events_r1.append(r1_back)
                    events_e1.append(e_back)
                    events_l1.append(np.zeros(nb, dtype=np.int32))
                    events_r2.append(r2_hit[ib])
                    events_e2.append(e_d2)
                    events_l2.append(np.ones(nb, dtype=np.int32))
                    events_nhits.append(np.full(nb, 2, dtype=np.int32))
                    events_true_cls.append(cls_b)
                    events_true_cos.append(cos_b)
                    events_e0_orig.append(e_ib)

    if not events_nhits:
        return _empty_event_dict()

    return {
        "r_a": np.vstack(events_r1).astype(np.float32),
        "e_a": np.concatenate(events_e1).astype(np.float32),
        "layer_a": np.concatenate(events_l1).astype(np.int32),
        "r_b": np.vstack(events_r2).astype(np.float32),
        "e_b": np.concatenate(events_e2).astype(np.float32),
        "layer_b": np.concatenate(events_l2).astype(np.int32),
        "n_hits": np.concatenate(events_nhits).astype(np.int32),
        "true_class": np.concatenate(events_true_cls).astype(np.int64),
        "true_cos_theta": np.concatenate(events_true_cos).astype(np.float32),
        "orig_e0": np.concatenate(events_e0_orig).astype(np.float32),
    }


def _empty_event_dict() -> Dict[str, np.ndarray]:
    return {
        "r_a": np.zeros((0, 3), dtype=np.float32),
        "e_a": np.zeros((0,), dtype=np.float32),
        "layer_a": np.zeros((0,), dtype=np.int32),
        "r_b": np.zeros((0, 3), dtype=np.float32),
        "e_b": np.zeros((0,), dtype=np.float32),
        "layer_b": np.zeros((0,), dtype=np.int32),
        "n_hits": np.zeros((0,), dtype=np.int32),
        "true_class": np.zeros((0,), dtype=np.int64),
        "true_cos_theta": np.zeros((0,), dtype=np.float32),
        "orig_e0": np.zeros((0,), dtype=np.float32),
    }
