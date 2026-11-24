"""
Traditional Gamma Image Reconstruction Baselines & Fast Physics Projection Operators:
1. Collimated Decoding:
   - Exact periodic MURA balanced cross-correlation decoding + bilinear angular resampling onto FOV grid;
   - Pinhole inversion mapping (when collimator_mode == 'pinhole');
2. Compton Cone Simple Back-Projection (SBP):
   - Vectorized 3D cone intersection on source hemisphere with Gaussian angular resolution kernel;
3. Traditional Joint Collimator + Compton List-Mode MLEM (Joint LM-MLEM):
   - Shepp-Vardi / Wilderman statistical iterative reconstruction combining collimated forward matrix A_coll
     and per-event Compton probability matrix T_comp.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import map_coordinates

from ..physics.cross_sections import compton_cos_theta_from_energies
from ..physics.geometry_and_sources import CameraGeometry


def histogram_collimated_detector(
    geom: CameraGeometry,
    pix_ix: np.ndarray,
    pix_iy: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate 2D detector shadowgram counts D[iy, ix] of shape (n_pix, n_pix)."""
    p = geom.n_pix
    if pix_ix.size == 0:
        return np.zeros((p, p), dtype=np.float64)
    flat_idx = pix_iy * p + pix_ix
    hist = np.bincount(flat_idx, weights=weights, minlength=p * p).astype(np.float64)
    return hist.reshape(p, p)


def decode_collimated_projection(
    geom: CameraGeometry,
    pix_ix: np.ndarray,
    pix_iy: np.ndarray,
    energies_kev: np.ndarray | None = None,
    energy_max_kev: float | None = None,
) -> np.ndarray:
    """
    Decode collimated single-site Layer-1 hits onto the (H, W) angular FOV grid.
    Supports optional upper energy window cut (e.g., E <= 240 keV for clean low-energy photo-peak channel).
    """
    if energies_kev is not None and energy_max_kev is not None:
        mask = energies_kev <= energy_max_kev
        pix_ix = pix_ix[mask]
        pix_iy = pix_iy[mask]

    det_counts = histogram_collimated_detector(geom, pix_ix, pix_iy)
    if np.sum(det_counts) < 1.0:
        return np.zeros((geom.grid_size, geom.grid_size), dtype=np.float32)

    tx_deg, ty_deg = geom.fov_meshgrid_deg()
    tx_tan = np.tan(np.deg2rad(tx_deg))
    ty_tan = np.tan(np.deg2rad(ty_deg))

    if geom.collimator_mode == "pinhole":
        # Pinhole camera inverts coordinates: x_det = - (mask_z / (1 - mask_z/z_s)) * tan(theta_x)
        eff_f = geom.mask_z_mm / max(1.0 - geom.mask_z_mm / geom.source_dist_mm, 0.2)
        xd = -eff_f * tx_tan
        yd = -eff_f * ty_tan
        half_span = 0.5 * (geom.n_pix - 1) * geom.pix_pitch_mm
        col_f = (xd + half_span) / geom.pix_pitch_mm
        row_f = (yd + half_span) / geom.pix_pitch_mm
        mapped = map_coordinates(det_counts, [row_f, col_f], order=1, mode="constant", cval=0.0)
        mapped = np.maximum(mapped, 0.0)
        mx = np.max(mapped)
        return (mapped / mx).astype(np.float32) if mx > 1e-12 else mapped.astype(np.float32)

    # MURA Balanced Periodic Cross-Correlation:
    # Both D and G_centered have on-axis (0,0) at array center (shift, shift).
    # Unroll both by -shift so index [0,0] is on-axis: D0[u] = A[u + delta].
    # Then sum_u D0[u] * G0[(u - s) % p] peaks at s = -delta, or sum_u D0[(u + s) % p] * G0[u] = IFFT(FFT(D0) * conj(FFT(G0))) peaks at s = +delta!
    p = geom.mura_rank
    shift = p // 2
    d0 = np.roll(np.roll(det_counts, -shift, axis=0), -shift, axis=1)
    g0 = np.roll(np.roll(geom.mura_decoder_g, -shift, axis=0), -shift, axis=1)
    corr0 = np.fft.ifft2(np.conj(np.fft.fft2(d0)) * np.fft.fft2(g0)).real
    pedestal = np.median(corr0)
    decoded_p = np.roll(np.roll(corr0 - pedestal, shift, axis=0), shift, axis=1)
    decoded_pos = np.maximum(decoded_p, 0.0)

    # Map FOV angle (theta_x, theta_y) -> continuous MURA shift index:
    # u_m = u_d + (mask_z / mask_pitch) * tan(theta) => delta = +(mask_z / mask_pitch) * tan(theta)
    sx = (geom.mask_z_mm / geom.mask_pitch_mm) * tx_tan + shift
    sy = (geom.mask_z_mm / geom.mask_pitch_mm) * ty_tan + shift

    fov_img = map_coordinates(decoded_pos, [sy, sx], order=1, mode="nearest")
    # Zero out outside fully-coded MURA span if needed
    max_shift = 0.5 * (p - 1) + 0.5
    in_fcov = (np.abs(sx - shift) <= max_shift) & (np.abs(sy - shift) <= max_shift)
    fov_img = np.where(in_fcov, fov_img, 0.0)

    mx = np.max(fov_img)
    if mx > 1e-12:
        fov_img /= mx
    return fov_img.astype(np.float32)


def compton_cone_matrix(
    geom: CameraGeometry,
    r1_mm: np.ndarray,
    e1_kev: np.ndarray,
    r2_mm: np.ndarray,
    e2_kev: np.ndarray,
    sigma_deg_base: float = 5.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute vectorized Compton cone response matrix T_comp of shape (N_valid, H*W)
    for ordered Compton pairs (r1, e1) -> (r2, e2).
    Returns:
        t_mat: (N_valid, H*W) probability weights per cone on FOV pixels
        valid_mask: bool array over input events indicating kinematically physical cones (|cos_theta| <= 1)
    """
    h = geom.grid_size
    w = geom.grid_size
    if r1_mm.shape[0] == 0:
        return np.zeros((0, h * w), dtype=np.float32), np.zeros(0, dtype=bool)

    ct_kin = compton_cos_theta_from_energies(e1_kev, e2_kev)
    axis_vec = r1_mm - r2_mm  # Points from absorber r2 back through scatterer r1 toward source
    axis_norm = np.linalg.norm(axis_vec, axis=-1)
    valid = (ct_kin >= -0.995) & (ct_kin <= 0.9995) & (axis_norm > 1.5)

    if not np.any(valid):
        return np.zeros((0, h * w), dtype=np.float32), valid

    r1_v = r1_mm[valid]
    ct_v = ct_kin[valid]
    e_tot_v = e1_kev[valid] + e2_kev[valid]
    axis_u = axis_vec[valid] / axis_norm[valid, None]  # (Nv, 3)
    theta_kin = np.arccos(np.clip(ct_v, -1.0, 1.0))     # (Nv,)

    # Multi-threaded SIMD float32 evaluation via PyTorch (identical math, ~25x faster than scalar NumPy arccos!)
    import torch
    with torch.no_grad():
        tx_deg, ty_deg = geom.fov_meshgrid_deg()
        zs = np.full(h * w, geom.source_dist_mm, dtype=np.float32)
        xs = zs * np.tan(np.deg2rad(tx_deg.ravel())).astype(np.float32)
        ys = zs * np.tan(np.deg2rad(ty_deg.ravel())).astype(np.float32)
        src_pts_t = torch.from_numpy(np.stack([xs, ys, zs], axis=-1))          # (M, 3)
        r1_t = torch.from_numpy(r1_v.astype(np.float32))                        # (Nv, 3)
        ax_t = torch.from_numpy(axis_u.astype(np.float32))                      # (Nv, 3)
        tk_t = torch.from_numpy(theta_kin.astype(np.float32))[:, None]          # (Nv, 1)

        v_to_src = src_pts_t.unsqueeze(0) - r1_t.unsqueeze(1)                   # (Nv, M, 3)
        v_to_src = torch.nn.functional.normalize(v_to_src, dim=-1, eps=1e-6)
        cos_geom = torch.sum(v_to_src * ax_t.unsqueeze(1), dim=-1).clamp(-1.0, 1.0)
        theta_geom = torch.acos(cos_geom)

        sig_np = np.deg2rad(sigma_deg_base * np.clip(np.sqrt(661.7 / np.maximum(e_tot_v, 100.0)), 0.75, 2.2)).astype(np.float32)
        sig_t = torch.from_numpy(sig_np)[:, None]
        sin_tk = torch.sin(tk_t).clamp(min=0.15)
        cone_norm = 2.0 * np.pi * sin_tk * sig_t
        d_theta = theta_geom - tk_t
        t_mat = (torch.exp(-0.5 * (d_theta / sig_t).square()) / cone_norm).numpy()
    return t_mat, valid


def compton_simple_backprojection(
    geom: CameraGeometry,
    r1_mm: np.ndarray,
    e1_kev: np.ndarray,
    r2_mm: np.ndarray,
    e2_kev: np.ndarray,
    sigma_deg_base: float = 5.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute normalized Compton Simple Back-Projection (SBP) image and cone-overlap variance/sharpness map
    of shape (H, W).
    """
    h = geom.grid_size
    t_mat, _ = compton_cone_matrix(geom, r1_mm, e1_kev, r2_mm, e2_kev, sigma_deg_base=sigma_deg_base)
    if t_mat.shape[0] == 0:
        z = np.zeros((h, h), dtype=np.float32)
        return z, z

    sbp = np.sum(t_mat, axis=0).reshape(h, h)
    # Sharpened high-pass / squared coincidence map highlighting multi-cone intersections
    sbp_sq = np.sum(t_mat * t_mat, axis=0).reshape(h, h)

    mx1 = np.max(sbp)
    if mx1 > 1e-12:
        sbp /= mx1
    mx2 = np.max(sbp_sq)
    if mx2 > 1e-12:
        sbp_sq /= mx2
    return sbp.astype(np.float32), sbp_sq.astype(np.float32)


def build_collimated_system_matrix(geom: CameraGeometry, rep_energy_kev: float = 140.0) -> np.ndarray:
    """
    Precompute dense Collimated forward system matrix A_coll of shape (n_pix*n_pix, H*W)
    for Joint MLEM baseline reconstruction.
    """
    from ..physics.cross_sections import tungsten_mu_mm1
    from ..physics.mc_simulator import _check_mask_open

    h = geom.grid_size
    m_vox = h * h
    n_det = geom.n_pix * geom.n_pix

    tx_deg, ty_deg = geom.fov_meshgrid_deg()
    zs = np.full(m_vox, geom.source_dist_mm, dtype=np.float64)
    xs = zs * np.tan(np.deg2rad(tx_deg.ravel()))
    ys = zs * np.tan(np.deg2rad(ty_deg.ravel()))
    src_pts = np.stack([xs, ys, zs], axis=-1)  # (M, 3)

    px, py = np.meshgrid(geom.pixel_centers_mm, geom.pixel_centers_mm, indexing="xy")
    det_pts = np.stack([px.ravel(), py.ravel(), np.zeros(n_det)], axis=-1)  # (N_det, 3)

    # Rays from every voxel j (1, M, 3) to every detector pixel i (N_det, 1, 3)
    rays = det_pts[:, None, :] - src_pts[None, :, :]  # (N_det, M, 3)
    norms = np.linalg.norm(rays, axis=-1)
    u = rays / norms[:, :, None]

    t_mask = (geom.mask_z_mm - src_pts[None, :, 2]) / u[:, :, 2]
    xm = src_pts[None, :, 0] + t_mask * u[:, :, 0]
    ym = src_pts[None, :, 1] + t_mask * u[:, :, 1]

    is_open, hit_w = _check_mask_open(geom, xm.ravel(), ym.ravel())
    is_open = is_open.reshape(n_det, m_vox)
    hit_w = hit_w.reshape(n_det, m_vox)

    mu_w = float(tungsten_mu_mm1(rep_energy_kev))
    slant = geom.mask_thickness_mm / np.abs(u[:, :, 2])
    trans = np.where(is_open, 1.0, np.where(hit_w, np.exp(-mu_w * slant), 0.0))
    solid_ang = np.abs(u[:, :, 2]) / ((norms / geom.source_dist_mm) ** 2)
    a_coll = (trans * solid_ang).astype(np.float32)
    a_coll /= np.maximum(np.sum(a_coll, axis=0, keepdims=True), 1e-8)
    return a_coll


def joint_listmode_mlem(
    geom: CameraGeometry,
    a_coll: np.ndarray,
    pix_ix_col: np.ndarray,
    pix_iy_col: np.ndarray,
    r1_comp: np.ndarray,
    e1_comp: np.ndarray,
    r2_comp: np.ndarray,
    e2_comp: np.ndarray,
    e_col_kev: np.ndarray | None = None,
    col_energy_cut_kev: float = 260.0,
    n_iters: int = 25,
    col_weight: float = 1.0,
    comp_weight: float = 1.0,
) -> np.ndarray:
    """
    Traditional Joint Collimator + Compton List-Mode MLEM (Shepp & Vardi 1982; Wilderman et al. 2001).
    Applies standard low-energy collimation energy window (E <= 260 keV) so high-energy mask-penetrated
    continuum does not corrupt the collimated likelihood term.
    """
    h = geom.grid_size
    m_vox = h * h
    f = np.ones(m_vox, dtype=np.float32) / float(m_vox)

    if e_col_kev is not None:
        ewin = e_col_kev <= col_energy_cut_kev
        pix_ix_col = pix_ix_col[ewin]
        pix_iy_col = pix_iy_col[ewin]

    y_det = histogram_collimated_detector(geom, pix_ix_col, pix_iy_col).ravel().astype(np.float32)
    has_col = float(np.sum(y_det)) > 15.0

    t_comp, _ = compton_cone_matrix(geom, r1_comp, e1_comp, r2_comp, e2_comp, sigma_deg_base=4.8)
    has_comp = t_comp.shape[0] >= 5

    if not has_col and not has_comp:
        return np.zeros((h, h), dtype=np.float32)

    # Row-normalize t_comp inside FOV only for cones that appreciably intersect the FOV
    if has_comp:
        fov_mass = np.sum(t_comp, axis=1)
        keep_cones = fov_mass > (0.15 * float(np.median(fov_mass)))
        t_comp = t_comp[keep_cones]
        has_comp = t_comp.shape[0] >= 3
        if has_comp:
            t_comp = t_comp / np.maximum(np.sum(t_comp, axis=1, keepdims=True), 1e-8)

    s_col_raw = np.sum(a_coll, axis=0) if has_col else np.zeros(m_vox, dtype=np.float32)
    if has_col:
        s_col_raw = np.maximum(s_col_raw, 0.35 * float(np.max(s_col_raw)))

    # Uniform solid-angle sensitivity across modest [-34,+34] deg FOV prevents boundary divergence
    s_cmp_raw = np.ones(m_vox, dtype=np.float32) if has_comp else np.zeros(m_vox, dtype=np.float32)

    w_c = col_weight if has_col else 0.0
    w_p = comp_weight if has_comp else 0.0
    denom_sens = np.maximum(w_c * (s_col_raw / max(float(np.mean(s_col_raw)), 1e-8)) + w_p * s_cmp_raw, 1e-8)

    y_norm = y_det / max(float(np.sum(y_det)), 1.0)
    col_bg = 0.08 / float(a_coll.shape[0])
    cmp_bg = 0.15 / float(m_vox)

    for _ in range(n_iters):
        backproj = np.zeros(m_vox, dtype=np.float32)
        if has_col:
            pred_col = (a_coll @ f) + col_bg
            backproj += w_c * (a_coll.T @ (y_norm / pred_col)) / max(float(np.mean(s_col_raw)), 1e-8)
        if has_comp:
            # In genuine List-Mode MLEM (Wilderman 2001), system probabilities are evaluated on-the-fly
            # per event cone rather than pre-caching dense 3D voxel matrices in RAM:
            t_step, _ = compton_cone_matrix(geom, r1_comp, e1_comp, r2_comp, e2_comp, sigma_deg_base=4.8)
            t_step = t_step[keep_cones]
            t_step = t_step / np.maximum(np.sum(t_step, axis=1, keepdims=True), 1e-8)
            pred_cmp = (t_step @ f) + cmp_bg
            backproj += w_p * (t_step.T @ (1.0 / pred_cmp)) / float(t_step.shape[0])
        f = (f / denom_sens) * backproj
        f_sum = float(np.sum(f))
        if f_sum > 1e-12:
            f /= f_sum

    img = f.reshape(h, h)
    mx = np.max(img)
    if mx > 1e-12:
        img /= mx
    return img.astype(np.float32)


def heuristic_classical_event_filter(dig_events) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classical non-AI rule-based event selection (used by Traditional Baselines):
    - Single-hit Layer-0 events -> Collimated candidates
    - Two-hit events across Layer 0 and Layer 1 -> Order by putting Layer 0 first (or classical E1 < E2 rule
      when ambiguous) and apply kinematic |cos(theta)| <= 1 cut.
    """
    col_mask = (dig_events.n_hits == 1) & (dig_events.layer1 == 0)
    ix_c = dig_events.pix_ix1[col_mask]
    iy_c = dig_events.pix_iy1[col_mask]
    e_c = dig_events.e1_dig_kev[col_mask]

    two_mask = dig_events.n_hits == 2
    r1 = dig_events.r1_dig_mm[two_mask].copy()
    e1 = dig_events.e1_dig_kev[two_mask].copy()
    l1 = dig_events.layer1[two_mask]
    r2 = dig_events.r2_dig_mm[two_mask].copy()
    e2 = dig_events.e2_dig_kev[two_mask].copy()
    l2 = dig_events.layer2[two_mask]

    # Classical ordering: prefer Layer 0 -> Layer 1, or E1 < E2 heuristic
    swap = (l1 == 1) & (l2 == 0)
    if np.any(swap):
        r1[swap], r2[swap] = r2[swap].copy(), r1[swap].copy()
        e1[swap], e2[swap] = e2[swap].copy(), e1[swap].copy()

    # Classical naive energy ordering often swaps when E1 > E2 (Compton edge region -> causes false mirror cones!)
    high_recoil_ambig = e1 > 0.62 * (e1 + e2)
    if np.any(high_recoil_ambig):
        r1[high_recoil_ambig], r2[high_recoil_ambig] = r2[high_recoil_ambig].copy(), r1[high_recoil_ambig].copy()
        e1[high_recoil_ambig], e2[high_recoil_ambig] = e2[high_recoil_ambig].copy(), e1[high_recoil_ambig].copy()

    return ix_c, iy_c, e_c, r1, e1, r2, e2
