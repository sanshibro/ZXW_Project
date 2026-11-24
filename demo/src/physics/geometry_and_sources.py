"""
Dual-mode gamma camera geometry (Gottesman-Fenimore Rank-17 MURA mask / Pinhole +
Dual-layer 17x17 pixelated GAGG:Ce array) and multi-condition radioactive source generators.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


# Common radioisotopes used in nuclear security, industry, and nuclear medicine
ISOTOPE_LIBRARY: dict[str, list[tuple[float, float]]] = {
    "Am-241 (59.5 keV)": [(59.5, 1.0)],
    "Tc-99m (140.5 keV)": [(140.5, 1.0)],
    "Ba-133 (356.0 keV)": [(356.0, 0.85), (81.0, 0.15)],
    "Cs-137 (661.7 keV)": [(661.7, 1.0)],
    "Co-60 (1173/1332 keV)": [(1173.2, 0.50), (1332.5, 0.50)],
}


def build_mura_pattern(p: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct exact Gottesman & Fenimore (1989) prime-rank MURA base aperture A (p x p),
    balanced decoding kernel G (p x p), and antisymmetric 2x2 cyclic mosaic (2p-1 x 2p-1).
    """
    # Compute quadratic residues mod p
    residues = set((k * k) % p for k in range(1, p))
    c = np.array([1 if (i % p) in residues else -1 for i in range(p)], dtype=np.int32)

    a = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            if i == 0:
                a[i, j] = 0.0
            elif j == 0:
                a[i, j] = 1.0
            else:
                a[i, j] = 1.0 if (c[i] * c[j] == 1) else 0.0

    # Centered cyclic shift so (0,0) feature sits at array center (p//2, p//2)
    shift = p // 2
    a_centered = np.roll(np.roll(a, shift, axis=0), shift, axis=1)

    g = np.where(a == 1.0, 1.0, -1.0)
    g[0, 0] = 1.0
    g_centered = np.roll(np.roll(g, shift, axis=0), shift, axis=1)

    # Tile into (2p-1) x (2p-1) cyclic mosaic centered around origin
    mosaic_size = 2 * p - 1
    half_m = mosaic_size // 2
    idx = (np.arange(-half_m, half_m + 1)) % p
    mosaic = a[np.ix_(idx, idx)]
    return a_centered, g_centered, mosaic


@dataclass
class CameraGeometry:
    fov_half_deg: float
    grid_size: int
    source_dist_mm: float
    collimator_mode: str
    mura_rank: int
    mask_pitch_mm: float
    mask_thickness_mm: float
    mask_z_mm: float
    pinhole_radius_mm: float
    n_pix: int
    pix_pitch_mm: float
    scat_thick_mm: float
    abs_thick_mm: float
    abs_z_mm: float
    mura_base: np.ndarray
    mura_decoder_g: np.ndarray
    mura_mosaic: np.ndarray
    angle_grid_deg: np.ndarray  # shape (grid_size,)
    pixel_centers_mm: np.ndarray  # shape (n_pix,)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CameraGeometry":
        sys_c = cfg["system"]
        col_c = cfg["collimator"]
        det_c = cfg["detector"]

        p = int(col_c["mura_rank"])
        a_base, g_dec, mosaic = build_mura_pattern(p)

        fov_half = float(sys_c["fov_half_angle_deg"])
        g_size = int(sys_c["image_grid_size"])
        src_d = float(sys_c["source_distance_mm"])
        mask_z = float(col_c["mask_to_scatterer_mm"])
        pix_pitch = float(det_c["pixel_pitch_mm"])
        # Near-field MURA magnification-matched pitch: d_mask = d_det * (1 - f / z_s)
        matched_mask_pitch = pix_pitch * (1.0 - mask_z / src_d)

        n_pix = int(det_c["n_pixels_xy"])
        pix_centers = (np.arange(n_pix, dtype=np.float64) - 0.5 * (n_pix - 1)) * pix_pitch
        ang_1d = np.linspace(-fov_half, fov_half, g_size, dtype=np.float64)

        return cls(
            fov_half_deg=fov_half,
            grid_size=g_size,
            source_dist_mm=src_d,
            collimator_mode=str(col_c["mode"]),
            mura_rank=p,
            mask_pitch_mm=matched_mask_pitch,
            mask_thickness_mm=float(col_c["mask_thickness_mm"]),
            mask_z_mm=mask_z,
            pinhole_radius_mm=float(col_c["pinhole_radius_mm"]),
            n_pix=n_pix,
            pix_pitch_mm=pix_pitch,
            scat_thick_mm=float(det_c["scatterer_thickness_mm"]),
            abs_thick_mm=float(det_c["absorber_thickness_mm"]),
            abs_z_mm=-float(det_c["layer_gap_mm"]),
            mura_base=a_base,
            mura_decoder_g=g_dec,
            mura_mosaic=mosaic,
            angle_grid_deg=ang_1d,
            pixel_centers_mm=pix_centers,
        )

    def fov_meshgrid_deg(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (TX, TY) meshgrid in degrees of shape (grid_size, grid_size), row=y, col=x."""
        tx, ty = np.meshgrid(self.angle_grid_deg, self.angle_grid_deg, indexing="xy")
        return tx, ty

    def fov_unit_vectors(self) -> np.ndarray:
        """Return unit vectors pointing from (0,0,0) to every FOV grid pixel, shape (H, W, 3)."""
        tx_deg, ty_deg = self.fov_meshgrid_deg()
        tx = np.tan(np.deg2rad(tx_deg))
        ty = np.tan(np.deg2rad(ty_deg))
        vz = np.ones_like(tx)
        vec = np.stack([tx, ty, vz], axis=-1)
        vec /= np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec


@dataclass
class SourceComponent:
    shape_type: str  # "point", "line", "ring", "l_shape"
    energy_kev: float
    weight: float
    params: dict[str, Any]


@dataclass
class SourceScene:
    scene_name: str
    components: list[SourceComponent]
    gt_map: np.ndarray  # Ground truth smooth intensity map (H, W), normalized max=1
    low_e_gt_map: np.ndarray
    high_e_gt_map: np.ndarray


def render_ground_truth_map(geom: CameraGeometry, components: list[SourceComponent], psf_sigma_deg: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render continuous normalized Ground Truth spatial distribution maps on the FOV grid."""
    tx, ty = geom.fov_meshgrid_deg()
    gt_total = np.zeros_like(tx, dtype=np.float64)
    gt_low = np.zeros_like(tx, dtype=np.float64)
    gt_high = np.zeros_like(tx, dtype=np.float64)

    for comp in components:
        st = comp.shape_type
        p = comp.params
        if st == "point":
            d2 = (tx - p["cx"]) ** 2 + (ty - p["cy"]) ** 2
            comp_map = np.exp(-0.5 * d2 / (psf_sigma_deg ** 2))
        elif st == "line":
            x0, y0, x1, y1 = p["x0"], p["y0"], p["x1"], p["y1"]
            dx, dy = x1 - x0, y1 - y0
            seg_len2 = max(dx * dx + dy * dy, 1e-6)
            t_proj = np.clip(((tx - x0) * dx + (ty - y0) * dy) / seg_len2, 0.0, 1.0)
            px = x0 + t_proj * dx
            py = y0 + t_proj * dy
            d2 = (tx - px) ** 2 + (ty - py) ** 2
            comp_map = np.exp(-0.5 * d2 / (psf_sigma_deg ** 2))
        elif st == "ring":
            r = np.sqrt((tx - p["cx"]) ** 2 + (ty - p["cy"]) ** 2)
            d2 = (r - p["radius"]) ** 2
            comp_map = np.exp(-0.5 * d2 / ((0.9 * psf_sigma_deg) ** 2))
        elif st == "l_shape":
            cx, cy, arm = p["cx"], p["cy"], p["arm"]
            # Vertical bar from (cx, cy) to (cx, cy + arm) + Horizontal bar from (cx, cy) to (cx + arm, cy)
            t_v = np.clip((ty - cy) / arm, 0.0, 1.0)
            d2_v = (tx - cx) ** 2 + (ty - (cy + t_v * arm)) ** 2
            t_h = np.clip((tx - cx) / arm, 0.0, 1.0)
            d2_h = (tx - (cx + t_h * arm)) ** 2 + (ty - cy) ** 2
            comp_map = np.exp(-0.5 * np.minimum(d2_v, d2_h) / (psf_sigma_deg ** 2))
        else:
            raise ValueError(f"Unknown shape_type: {st}")

        gt_total += comp.weight * comp_map
        if comp.energy_kev <= 220.0:
            gt_low += comp.weight * comp_map
        else:
            gt_high += comp.weight * comp_map

    max_val = np.max(gt_total)
    if max_val > 1e-12:
        gt_total /= max_val
        gt_low /= max_val
        gt_high /= max_val
    return gt_total.astype(np.float32), gt_low.astype(np.float32), gt_high.astype(np.float32)


def sample_source_emissions(
    geom: CameraGeometry,
    scene: SourceScene,
    n_photons: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample 3D emission coordinates r_s (N, 3) in mm and initial photon energies E0 (N,) in keV
    from a multi-component SourceScene.
    """
    weights = np.array([c.weight for c in scene.components], dtype=np.float64)
    weights /= np.sum(weights)
    comp_ids = rng.choice(len(scene.components), size=n_photons, p=weights)

    theta_x = np.empty(n_photons, dtype=np.float64)
    theta_y = np.empty(n_photons, dtype=np.float64)
    energies = np.empty(n_photons, dtype=np.float64)

    for idx, comp in enumerate(scene.components):
        mask = comp_ids == idx
        m = int(np.sum(mask))
        if m == 0:
            continue
        energies[mask] = comp.energy_kev
        p = comp.params
        st = comp.shape_type
        if st == "point":
            theta_x[mask] = p["cx"] + rng.normal(0.0, 0.35, size=m)
            theta_y[mask] = p["cy"] + rng.normal(0.0, 0.35, size=m)
        elif st == "line":
            t = rng.random(m)
            theta_x[mask] = p["x0"] + t * (p["x1"] - p["x0"]) + rng.normal(0.0, 0.35, size=m)
            theta_y[mask] = p["y0"] + t * (p["y1"] - p["y0"]) + rng.normal(0.0, 0.35, size=m)
        elif st == "ring":
            phi = rng.uniform(0.0, 2.0 * np.pi, size=m)
            rad = p["radius"] + rng.normal(0.0, 0.35, size=m)
            theta_x[mask] = p["cx"] + rad * np.cos(phi)
            theta_y[mask] = p["cy"] + rad * np.sin(phi)
        elif st == "l_shape":
            arm = p["arm"]
            is_vert = rng.random(m) < 0.5
            t = rng.random(m) * arm
            tx_loc = np.where(is_vert, 0.0, t)
            ty_loc = np.where(is_vert, t, 0.0)
            theta_x[mask] = p["cx"] + tx_loc + rng.normal(0.0, 0.35, size=m)
            theta_y[mask] = p["cy"] + ty_loc + rng.normal(0.0, 0.35, size=m)

    zs = np.full(n_photons, geom.source_dist_mm, dtype=np.float64)
    xs = zs * np.tan(np.deg2rad(theta_x))
    ys = zs * np.tan(np.deg2rad(theta_y))
    pos_mm = np.stack([xs, ys, zs], axis=-1)
    return pos_mm, energies


def sample_random_training_scene(geom: CameraGeometry, rng: np.random.Generator) -> SourceScene:
    """
    Procedurally sample diverse multi-condition scenes covering:
    - Single point sources (low, med, high energy)
    - Dual / triple point sources (close & wide separations, single or mixed isotopes)
    - Extended shapes: Line, Ring, L-shape
    - Mixed wide-band dual-isotope scenes (e.g. 59.5 keV + 662 keV)
    """
    lim = geom.fov_half_deg * 0.76
    energy_pool = [59.5, 81.0, 122.0, 140.5, 245.0, 356.0, 511.0, 661.7, 834.8, 1173.2, 1332.5]

    mode_roll = rng.random()
    components: list[SourceComponent] = []

    if mode_roll < 0.28:
        # Single point source
        e0 = float(rng.choice(energy_pool))
        cx, cy = float(rng.uniform(-lim, lim)), float(rng.uniform(-lim, lim))
        components.append(SourceComponent("point", e0, 1.0, {"cx": cx, "cy": cy}))
        name = f"single_point_{int(e0)}keV"
    elif mode_roll < 0.56:
        # Dual / triple point sources (possibly mixed low + high energy!)
        n_pts = 2 if rng.random() < 0.8 else 3
        mixed = rng.random() < 0.55
        base_e = float(rng.choice(energy_pool))
        cx0, cy0 = float(rng.uniform(-lim * 0.7, lim * 0.7)), float(rng.uniform(-lim * 0.7, lim * 0.7))
        for k in range(n_pts):
            if k == 0:
                cx, cy = cx0, cy0
            else:
                sep = float(rng.uniform(7.5, 26.0))
                ang = float(rng.uniform(0, 2 * np.pi))
                cx = float(np.clip(cx0 + sep * np.cos(ang), -lim, lim))
                cy = float(np.clip(cy0 + sep * np.sin(ang), -lim, lim))
            if mixed:
                ek = float(rng.choice([59.5, 140.5])) if (k % 2 == 0) else float(rng.choice([356.0, 661.7, 1173.2]))
            else:
                ek = base_e
            w = float(rng.uniform(0.55, 1.0))
            components.append(SourceComponent("point", ek, w, {"cx": cx, "cy": cy}))
        name = "multi_point_mixed" if mixed else "multi_point_mono"
    elif mode_roll < 0.73:
        # Line source
        e0 = float(rng.choice(energy_pool))
        cx, cy = float(rng.uniform(-lim * 0.5, lim * 0.5)), float(rng.uniform(-lim * 0.5, lim * 0.5))
        length = float(rng.uniform(12.0, 24.0))
        ang = float(rng.uniform(0, np.pi))
        dx, dy = 0.5 * length * np.cos(ang), 0.5 * length * np.sin(ang)
        components.append(SourceComponent("line", e0, 1.0, {"x0": cx - dx, "y0": cy - dy, "x1": cx + dx, "y1": cy + dy}))
        name = f"line_{int(e0)}keV"
    elif mode_roll < 0.88:
        # Ring source
        e0 = float(rng.choice(energy_pool))
        cx, cy = float(rng.uniform(-lim * 0.45, lim * 0.45)), float(rng.uniform(-lim * 0.45, lim * 0.45))
        rad = float(rng.uniform(7.5, 13.5))
        components.append(SourceComponent("ring", e0, 1.0, {"cx": cx, "cy": cy, "radius": rad}))
        name = f"ring_{int(e0)}keV"
    else:
        # L-shape extended source (optionally accompanied by a point hotspot)
        e0 = float(rng.choice(energy_pool))
        cx, cy = float(rng.uniform(-lim * 0.55, lim * 0.25)), float(rng.uniform(-lim * 0.55, lim * 0.25))
        arm = float(rng.uniform(11.0, 18.0))
        components.append(SourceComponent("l_shape", e0, 1.0, {"cx": cx, "cy": cy, "arm": arm}))
        name = f"l_shape_{int(e0)}keV"

    gt, gt_l, gt_h = render_ground_truth_map(geom, components)
    return SourceScene(name, components, gt, gt_l, gt_h)
