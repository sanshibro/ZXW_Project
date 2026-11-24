"""
Vectorized 3D Monte Carlo Gamma-Ray Transport Simulator for Hybrid Collimator + Compton Camera.
Models:
1. Solid-angle directed illumination onto front collimator / Layer-1 scatterer aperture;
2. Exact 3D ray intersection through Tungsten MURA coded mask / pinhole with exponential penetration
   exp(-mu_W(E) * path_len) and mask-edge scatter contamination;
3. Layer-1 GAGG:Ce stochastic interaction depth sampling, Photoelectric vs Compton branching,
   Klein-Nishina 3D scattering with random azimuth & bound-electron Doppler broadening;
4. Scattered photon transport to Layer-2 GAGG:Ce absorber with full absorption vs Compton escape!
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .cross_sections import (
    gagg_attenuation_mm1,
    tungsten_mu_mm1,
    sample_klein_nishina_cos_theta,
)
from .geometry_and_sources import CameraGeometry, SourceScene, sample_source_emissions


# True Monte Carlo event ground-truth category codes:
EVT_COLLIMATED_CLEAN = 1     # Clean un-scattered photon passing open aperture -> Layer 1 PE
EVT_COMPTON_CLEAN_FWD = 2    # True full-energy Compton: Layer 1 scatter -> Layer 2 PE absorption
EVT_INCOMPLETE_OR_BG = 0     # Mask-penetrated/scattered, incomplete escape, or random background


@dataclass
class RawMCEvents:
    """Ideal continuous Monte Carlo interaction records prior to detector digitization."""
    n_hits: np.ndarray        # (N_evt,) 1 or 2
    r1_mm: np.ndarray         # (N_evt, 3) First interaction 3D position in mm
    e1_kev: np.ndarray        # (N_evt,) First interaction true deposited energy in keV
    layer1: np.ndarray        # (N_evt,) 0 = Scatterer (Layer 1), 1 = Absorber (Layer 2)
    r2_mm: np.ndarray         # (N_evt, 3) Second interaction 3D position in mm (0 if n_hits==1)
    e2_kev: np.ndarray        # (N_evt,) Second interaction true deposited energy in keV
    layer2: np.ndarray        # (N_evt,) Layer index of second hit
    e0_true_kev: np.ndarray   # (N_evt,) True initial source photon energy
    true_type: np.ndarray     # (N_evt,) Ground truth event class (0=invalid/escape/bg, 1=collimated, 2=true Compton)


def _rotate_scattered_direction(
    u_inc: np.ndarray,
    cos_theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Rotate incident unit vectors u_inc (M, 3) by polar angle theta and azimuth phi."""
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    ux, uy, uz = u_inc[:, 0], u_inc[:, 1], u_inc[:, 2]

    # Construct orthonormal transverse basis (e1, e2) perpendicular to u_inc
    horiz = np.sqrt(ux * ux + uy * uy)
    use_z = horiz > 1e-4

    e1x = np.where(use_z, -uy / np.maximum(horiz, 1e-8), 1.0)
    e1y = np.where(use_z, ux / np.maximum(horiz, 1e-8), 0.0)
    e1z = np.zeros_like(ux)

    e2x = uy * e1z - uz * e1y
    e2y = uz * e1x - ux * e1z
    e2z = ux * e1y - uy * e1x

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)

    vx = cos_theta * ux + sin_theta * (c_phi * e1x + s_phi * e2x)
    vy = cos_theta * uy + sin_theta * (c_phi * e1y + s_phi * e2y)
    vz = cos_theta * uz + sin_theta * (c_phi * e1z + s_phi * e2z)
    out = np.stack([vx, vy, vz], axis=-1)
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def _check_mask_open(geom: CameraGeometry, xm: np.ndarray, ym: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate whether ray crossing mask plane z = mask_z_mm hits an open aperture or Tungsten metal,
    or misses the front shield entirely.
    Returns:
        is_open: bool array (True if passes freely without hitting Tungsten)
        hit_tungsten: bool array (True if traverses Tungsten mask material)
    """
    if geom.collimator_mode == "pinhole":
        r2 = xm * xm + ym * ym
        half_plate = 0.9 * geom.n_pix * geom.pix_pitch_mm
        in_plate = (np.abs(xm) <= half_plate) & (np.abs(ym) <= half_plate)
        in_hole = r2 <= (geom.pinhole_radius_mm ** 2)
        is_open = in_hole | (~in_plate)
        hit_tungsten = in_plate & (~in_hole)
        return is_open, hit_tungsten

    # MURA coded-aperture mosaic mode
    mosaic = geom.mura_mosaic
    ms = mosaic.shape[0]
    half_span = 0.5 * ms * geom.mask_pitch_mm

    in_mask_bounds = (np.abs(xm) < half_span) & (np.abs(ym) < half_span)
    col_idx = np.clip(np.floor((xm + half_span) / geom.mask_pitch_mm).astype(np.int32), 0, ms - 1)
    row_idx = np.clip(np.floor((ym + half_span) / geom.mask_pitch_mm).astype(np.int32), 0, ms - 1)

    aperture_val = mosaic[row_idx, col_idx] > 0.5
    # Surround outside MURA mosaic with Tungsten shielding collar up to 1.3x detector span
    collar_span = 0.85 * geom.n_pix * geom.pix_pitch_mm
    in_collar = (~in_mask_bounds) & (np.abs(xm) < collar_span) & (np.abs(ym) < collar_span)

    is_open = (in_mask_bounds & aperture_val)
    hit_tungsten = (in_mask_bounds & (~aperture_val)) | in_collar
    return is_open, hit_tungsten


def simulate_scene_transport(
    geom: CameraGeometry,
    scene: SourceScene,
    n_emitted_toward_det: int,
    rng: np.random.Generator,
    bg_fraction: float = 0.08,
) -> RawMCEvents:
    """
    Run vectorized 3D Monte Carlo gamma-ray transport from SourceScene into the Dual-Mode Camera.
    """
    src_pos, e0 = sample_source_emissions(geom, scene, n_emitted_toward_det, rng)

    # Aim photons uniformly across Layer-1 active entrance face [-half_det, +half_det]^2 at z=0
    half_det = 0.5 * geom.n_pix * geom.pix_pitch_mm
    target_x = rng.uniform(-half_det, half_det, size=n_emitted_toward_det)
    target_y = rng.uniform(-half_det, half_det, size=n_emitted_toward_det)
    target_pos = np.stack([target_x, target_y, np.zeros_like(target_x)], axis=-1)

    ray_vec = target_pos - src_pos
    u_inc = ray_vec / np.linalg.norm(ray_vec, axis=-1, keepdims=True)  # uz < 0

    # Step 1: Ray intersection with Front Collimator Mask at z = +mask_z_mm
    t_mask = (geom.mask_z_mm - src_pos[:, 2]) / u_inc[:, 2]
    xm = src_pos[:, 0] + t_mask * u_inc[:, 0]
    ym = src_pos[:, 1] + t_mask * u_inc[:, 1]

    is_open, hit_w = _check_mask_open(geom, xm, ym)
    mu_w = tungsten_mu_mm1(e0)
    slant_path_w = geom.mask_thickness_mm / np.abs(u_inc[:, 2])
    trans_prob_w = np.where(hit_w, np.exp(-mu_w * slant_path_w), 1.0)

    # Small fraction of photons hitting Tungsten undergo small-angle mask scatter instead of pure absorption
    mask_scatter_prob = np.where(hit_w, 0.035 * (1.0 - trans_prob_w) * (e0 > 180.0), 0.0)
    u_rand = rng.random(n_emitted_toward_det)
    penetrated_clean = hit_w & (u_rand < trans_prob_w)
    mask_scattered = hit_w & (u_rand >= trans_prob_w) & (u_rand < trans_prob_w + mask_scatter_prob)
    survived_mask = is_open | penetrated_clean | mask_scattered

    # Filter surviving photons reaching Layer 1 entrance face z_top1 = +scat_thick/2
    src_s = src_pos[survived_mask]
    u_s = u_inc[survived_mask].copy()
    e0_s = e0[survived_mask].copy()
    open_s = is_open[survived_mask]
    pen_s = penetrated_clean[survived_mask]
    mscat_s = mask_scattered[survived_mask]

    if np.any(mscat_s):
        # Mask-scattered photons lose energy and deflect slightly
        n_ms = int(np.sum(mscat_s))
        ct_ms, _, e_ms = sample_klein_nishina_cos_theta(e0_s[mscat_s], rng, enable_doppler=True)
        # Bias toward forward mask-edge scatter that still reaches detector
        ct_ms = np.clip(0.75 + 0.25 * np.abs(ct_ms), -0.99, 0.999)
        phi_ms = rng.uniform(0.0, 2.0 * np.pi, size=n_ms)
        u_s[mscat_s] = _rotate_scattered_direction(u_s[mscat_s], ct_ms, phi_ms)
        e0_s[mscat_s] = e_ms

    # Step 2: Transport through Layer 1 (Scatterer / Collimated Receiver, z in [-T1/2, +T1/2])
    z1_top = 0.5 * geom.scat_thick_mm
    valid_down = u_s[:, 2] < -1e-4
    src_s, u_s, e0_s, open_s, pen_s, mscat_s = (
        src_s[valid_down], u_s[valid_down], e0_s[valid_down],
        open_s[valid_down], pen_s[valid_down], mscat_s[valid_down],
    )

    t_ent1 = (z1_top - src_s[:, 2]) / u_s[:, 2]
    r_ent1 = src_s + t_ent1[:, None] * u_s
    in_xy1 = (np.abs(r_ent1[:, 0]) < half_det) & (np.abs(r_ent1[:, 1]) < half_det)

    r_ent1, u_s, e0_s, open_s, pen_s, mscat_s = (
        r_ent1[in_xy1], u_s[in_xy1], e0_s[in_xy1],
        open_s[in_xy1], pen_s[in_xy1], mscat_s[in_xy1],
    )

    m1 = e0_s.shape[0]
    mu_pe1, mu_comp1, mu_pair1, mu_tot1 = gagg_attenuation_mm1(e0_s)
    max_path1 = geom.scat_thick_mm / np.abs(u_s[:, 2])
    free_path1 = rng.exponential(1.0 / np.maximum(mu_tot1, 1e-8))

    interacted_l1 = free_path1 < max_path1
    # Note: Photons that pass straight through Layer 1 without interacting cannot form collimated L1 or L1-L2 Compton events
    idx1 = np.where(interacted_l1)[0]
    r1 = r_ent1[idx1] + free_path1[idx1, None] * u_s[idx1]
    in_vol1 = (np.abs(r1[:, 0]) < half_det) & (np.abs(r1[:, 1]) < half_det)
    idx1 = idx1[in_vol1]
    r1 = r1[in_vol1]

    u1 = u_s[idx1]
    e_inc1 = e0_s[idx1]
    open1 = open_s[idx1]
    mscat1 = mscat_s[idx1]
    pe_prob1 = (mu_pe1[idx1] / mu_tot1[idx1])

    is_pe1 = rng.random(idx1.size) < pe_prob1

    # --- Branch A: Single-site Photoelectric absorption in Layer 1 (Candidate Collimated Events) ---
    pe_mask = is_pe1
    n_pe = int(np.sum(pe_mask))
    r1_col = r1[pe_mask]
    e1_col = e_inc1[pe_mask]
    # Note: Even penetrated high-energy photons hit Layer 1 and degrade pure MURA contrast unless fused with Compton!
    # Clean collimated label = open aperture & not mask-scattered
    type_col = np.where(open1[pe_mask] & (~mscat1[pe_mask]), EVT_COLLIMATED_CLEAN, EVT_INCOMPLETE_OR_BG)

    # --- Branch B: Compton Scattering in Layer 1 -> Track scattered photon toward Layer 2 ---
    comp_mask = ~is_pe1
    r1_c = r1[comp_mask]
    u1_c = u1[comp_mask]
    e_inc_c = e_inc1[comp_mask]
    mscat_c = mscat1[comp_mask]

    n_c = e_inc_c.shape[0]
    if n_c > 0:
        ct_c, e_rec1, e_scat = sample_klein_nishina_cos_theta(e_inc_c, rng, enable_doppler=True)
        phi_c = rng.uniform(0.0, 2.0 * np.pi, size=n_c)
        u_scat = _rotate_scattered_direction(u1_c, ct_c, phi_c)

        # Propagate scattered photon to Layer 2 (centered at z = abs_z_mm, thickness abs_thick_mm)
        z2_top = geom.abs_z_mm + 0.5 * geom.abs_thick_mm
        going_down2 = u_scat[:, 2] < -1e-4
        t_ent2 = np.where(going_down2, (z2_top - r1_c[:, 2]) / np.minimum(u_scat[:, 2], -1e-6), -1.0)
        r_ent2 = r1_c + t_ent2[:, None] * u_scat
        hits_l2_face = going_down2 & (t_ent2 > 0.0) & (np.abs(r_ent2[:, 0]) < half_det) & (np.abs(r_ent2[:, 1]) < half_det)

        # Subset reaching Layer 2
        mu_pe2, mu_comp2, _, mu_tot2 = gagg_attenuation_mm1(e_scat)
        max_path2 = np.where(hits_l2_face, geom.abs_thick_mm / np.maximum(np.abs(u_scat[:, 2]), 1e-4), 0.0)
        free_path2 = rng.exponential(1.0 / np.maximum(mu_tot2, 1e-8))
        interacted_l2 = hits_l2_face & (free_path2 < max_path2)

        r2_cand = r_ent2 + free_path2[:, None] * u_scat
        interacted_l2 &= (np.abs(r2_cand[:, 0]) < half_det) & (np.abs(r2_cand[:, 1]) < half_det)

        # Case B1: Scattered photon escapes without hitting Layer 2 -> single-hit incomplete Compton continuum in L1!
        esc_l1 = ~interacted_l2
        r1_esc1 = r1_c[esc_l1]
        e1_esc1 = e_rec1[esc_l1]

        # Case B2: Scattered photon interacts in Layer 2 (either full PE absorption or secondary Compton escape!)
        i2 = np.where(interacted_l2)[0]
        r1_two = r1_c[i2]
        e1_two = e_rec1[i2]
        r2_two = r2_cand[i2]
        e_scat_two = e_scat[i2]
        mscat_two = mscat_c[i2]

        # Probability of full energy containment in Layer 2 (direct PE or two-step intra-crystal absorption)
        pe_eff2 = np.clip((mu_pe2[i2] + 0.42 * mu_comp2[i2]) / mu_tot2[i2], 0.0, 1.0)
        full_abs2 = rng.random(i2.size) < pe_eff2

        # For incomplete secondary Compton in Layer 2, sample only partial recoil energy!
        e2_two = np.copy(e_scat_two)
        part_mask = ~full_abs2
        if np.any(part_mask):
            _, e_rec2_part, _ = sample_klein_nishina_cos_theta(e_scat_two[part_mask], rng, enable_doppler=True)
            e2_two[part_mask] = e_rec2_part

        type_two = np.where(full_abs2 & (~mscat_two), EVT_COMPTON_CLEAN_FWD, EVT_INCOMPLETE_OR_BG)
        e0_two = e_inc_c[i2]
    else:
        r1_esc1 = np.zeros((0, 3), dtype=np.float64)
        e1_esc1 = np.zeros(0, dtype=np.float64)
        r1_two = np.zeros((0, 3), dtype=np.float64)
        e1_two = np.zeros(0, dtype=np.float64)
        r2_two = np.zeros((0, 3), dtype=np.float64)
        e2_two = np.zeros(0, dtype=np.float64)
        type_two = np.zeros(0, dtype=np.int64)
        e0_two = np.zeros(0, dtype=np.float64)

    # Combine Single-Hit events (PE + L1 Compton escapes) and Two-Hit events
    n_s1 = n_pe + r1_esc1.shape[0]
    n_s2 = r1_two.shape[0]

    # Inject realistic random coincidence / ambient background events
    n_bg = int(max(4, round(bg_fraction * (n_s1 + n_s2))))
    n_bg_two = n_bg // 2
    n_bg_one = n_bg - n_bg_two

    bg1_r = np.column_stack([
        rng.uniform(-half_det, half_det, n_bg_one),
        rng.uniform(-half_det, half_det, n_bg_one),
        rng.uniform(-0.5 * geom.scat_thick_mm, 0.5 * geom.scat_thick_mm, n_bg_one),
    ])
    bg1_e = rng.uniform(22.0, 900.0, n_bg_one)

    bg2_r1 = np.column_stack([
        rng.uniform(-half_det, half_det, n_bg_two),
        rng.uniform(-half_det, half_det, n_bg_two),
        rng.uniform(-0.5 * geom.scat_thick_mm, 0.5 * geom.scat_thick_mm, n_bg_two),
    ])
    bg2_r2 = np.column_stack([
        rng.uniform(-half_det, half_det, n_bg_two),
        rng.uniform(-half_det, half_det, n_bg_two),
        geom.abs_z_mm + rng.uniform(-0.5 * geom.abs_thick_mm, 0.5 * geom.abs_thick_mm, n_bg_two),
    ])
    bg2_e1 = rng.uniform(25.0, 500.0, n_bg_two)
    bg2_e2 = rng.uniform(35.0, 750.0, n_bg_two)

    # Assemble unified arrays
    r1_all = np.concatenate([r1_col, r1_esc1, bg1_r, r1_two, bg2_r1], axis=0)
    e1_all = np.concatenate([e1_col, e1_esc1, bg1_e, e1_two, bg2_e1], axis=0)
    r2_zeros = np.zeros((n_s1 + n_bg_one, 3), dtype=np.float64)
    e2_zeros = np.zeros(n_s1 + n_bg_one, dtype=np.float64)
    r2_all = np.concatenate([r2_zeros, r2_two, bg2_r2], axis=0)
    e2_all = np.concatenate([e2_zeros, e2_two, bg2_e2], axis=0)

    n_hits_all = np.concatenate([
        np.ones(n_s1 + n_bg_one, dtype=np.int32),
        np.full(n_s2 + n_bg_two, 2, dtype=np.int32),
    ])
    layer1_all = np.zeros_like(n_hits_all)
    layer2_all = np.where(n_hits_all == 2, 1, -1)

    e0_esc1 = e_inc_c[esc_l1] if n_c > 0 else np.zeros(0, dtype=np.float64)
    e0_all = np.concatenate([e1_col, e0_esc1, bg1_e, e0_two, bg2_e1 + bg2_e2], axis=0)
    true_type_all = np.concatenate([
        type_col,
        np.zeros(r1_esc1.shape[0] + n_bg_one, dtype=np.int64),
        type_two,
        np.zeros(n_bg_two, dtype=np.int64),
    ])

    # Shuffle order so event streams have no artificial grouping
    perm = rng.permutation(n_hits_all.shape[0])
    return RawMCEvents(
        n_hits=n_hits_all[perm],
        r1_mm=r1_all[perm],
        e1_kev=e1_all[perm],
        layer1=layer1_all[perm],
        r2_mm=r2_all[perm],
        e2_kev=e2_all[perm],
        layer2=layer2_all[perm],
        e0_true_kev=e0_all[perm],
        true_type=true_type_all[perm],
    )
