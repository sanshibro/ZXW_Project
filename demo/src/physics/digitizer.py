"""
Realistic Pixelated Array Detector Response Digitizer (GATE-style Digitizer Chain):
1. Pixel discretization onto 17x17 crystal array centers + optical/charge crosstalk shift;
2. Depth-of-Interaction (DOI) axial Gaussian uncertainty;
3. Energy resolution Gaussian broadening: FWHM(E)/E = sqrt(a^2/E + b^2 + (2.355*sigma_elec/E)^2);
4. Electronic trigger threshold truncation (drops sub-threshold recoil hits);
5. Random hit-order ambiguity in list-mode readout (50% random swap of (hit A, hit B) on 2-hit events
   so the AI EventNet genuinely learns sequence ordering!).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

from .cross_sections import compton_cos_theta_from_energies
from .geometry_and_sources import CameraGeometry
from .mc_simulator import RawMCEvents, EVT_COLLIMATED_CLEAN, EVT_COMPTON_CLEAN_FWD, EVT_INCOMPLETE_OR_BG


# Digitized 4-class labels for Stage-1 LiteEventNet:
CLS_INVALID_ESCAPE_BG = 0     # Incomplete energy deposition, mask scatter, or random background
CLS_VALID_COLLIMATED = 1      # Valid single-site collimated event
CLS_COMPTON_FWD_1TO2 = 2      # Valid Compton event where Hit A is scatterer & Hit B is absorber
CLS_COMPTON_REV_2TO1 = 3      # Valid Compton event where Hit A/B were swapped (Hit B scattered first!)


@dataclass
class DigitizedEvents:
    n_hits: np.ndarray        # (N,) 1 or 2 after threshold truncation
    pix_ix1: np.ndarray       # (N,) Pixel x-index [0..n_pix-1] for Hit A
    pix_iy1: np.ndarray       # (N,) Pixel y-index [0..n_pix-1] for Hit A
    r1_dig_mm: np.ndarray     # (N, 3) Digitized 3D position of Hit A
    e1_dig_kev: np.ndarray    # (N,) Digitized energy of Hit A
    layer1: np.ndarray        # (N,) Layer ID of Hit A (0 or 1)
    r2_dig_mm: np.ndarray     # (N, 3) Digitized 3D position of Hit B
    e2_dig_kev: np.ndarray    # (N,) Digitized energy of Hit B
    layer2: np.ndarray        # (N,) Layer ID of Hit B (0, 1, or -1)
    e0_true_kev: np.ndarray   # (N,) True initial gamma energy (for training supervision)
    label_4cls: np.ndarray    # (N,) 4-class target label [0, 1, 2, 3]
    features_16d: np.ndarray  # (N, 16) Extracted physical & kinematic feature matrix for LiteEventNet


class DetectorDigitizer:
    def __init__(self, geom: CameraGeometry, dig_cfg: dict[str, Any]):
        self.geom = geom
        self.ref_fwhm = float(dig_cfg["energy_res_ref_fwhm"])
        self.b_const = float(dig_cfg["constant_term_b"])
        self.elec_noise_kev = float(dig_cfg["electronic_noise_keV"])
        self.threshold_kev = float(dig_cfg["trigger_threshold_keV"])
        self.doi_sigma_mm = float(dig_cfg["doi_sigma_mm"])
        self.crosstalk = float(dig_cfg["optical_crosstalk_frac"])

        # Calibrate statistical term 'a' so FWHM(661.7 keV) / 661.7 == ref_fwhm
        e_ref = 661.7
        noise_term_ref = (2.35482 * self.elec_noise_kev / e_ref) ** 2
        a_sq = max(1e-6, (self.ref_fwhm ** 2 - self.b_const ** 2 - noise_term_ref) * e_ref)
        self.a_stat = float(np.sqrt(a_sq))

    def energy_sigma_kev(self, e_true_kev: np.ndarray) -> np.ndarray:
        """Compute 1-sigma Gaussian energy resolution in keV as a function of deposited energy."""
        e_pos = np.maximum(e_true_kev, 1.0)
        fwhm_rel_sq = (self.a_stat ** 2) / e_pos + (self.b_const ** 2) + (2.35482 * self.elec_noise_kev / e_pos) ** 2
        return (e_pos * np.sqrt(fwhm_rel_sq)) / 2.35482

    def _quantize_position(
        self,
        r_mm: np.ndarray,
        layer_id: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Discretize (x,y) to pixel centers with SiPM optical crosstalk jitter, and blur z by DOI sigma."""
        n = r_mm.shape[0]
        half_span = 0.5 * self.geom.n_pix * self.geom.pix_pitch_mm

        # Optical crosstalk induces slight Anger centroid pull toward neighboring pixels
        xt_jitter = rng.normal(0.0, self.crosstalk * self.geom.pix_pitch_mm * 3.5, size=(n, 2))
        x_eff = r_mm[:, 0] + xt_jitter[:, 0]
        y_eff = r_mm[:, 1] + xt_jitter[:, 1]

        ix = np.clip(np.floor((x_eff + half_span) / self.geom.pix_pitch_mm).astype(np.int32), 0, self.geom.n_pix - 1)
        iy = np.clip(np.floor((y_eff + half_span) / self.geom.pix_pitch_mm).astype(np.int32), 0, self.geom.n_pix - 1)

        xq = self.geom.pixel_centers_mm[ix]
        yq = self.geom.pixel_centers_mm[iy]

        # Blur axial z within crystal layer bounds
        zq = r_mm[:, 2] + rng.normal(0.0, self.doi_sigma_mm, size=n)
        z_l1_lim = 0.5 * self.geom.scat_thick_mm
        z_l2_min = self.geom.abs_z_mm - 0.5 * self.geom.abs_thick_mm
        z_l2_max = self.geom.abs_z_mm + 0.5 * self.geom.abs_thick_mm

        zq = np.where(
            layer_id == 0,
            np.clip(zq, -z_l1_lim, z_l1_lim),
            np.clip(zq, z_l2_min, z_l2_max),
        )
        return np.column_stack([xq, yq, zq]), ix, iy

    def digitize(self, raw: RawMCEvents, rng: np.random.Generator) -> DigitizedEvents:
        n0 = raw.n_hits.shape[0]
        if n0 == 0:
            raise ValueError("Empty RawMCEvents passed to digitizer.")

        # 1. Smear energies with realistic energy resolution + electronic noise
        sig1 = self.energy_sigma_kev(raw.e1_kev)
        e1_sm = raw.e1_kev + rng.normal(0.0, sig1)

        is_two = raw.n_hits == 2
        sig2 = self.energy_sigma_kev(np.where(is_two, raw.e2_kev, 1.0))
        e2_sm = np.where(is_two, raw.e2_kev + rng.normal(0.0, sig2), 0.0)

        # 2. Quantize 3D interaction coordinates
        r1_q, ix1, iy1 = self._quantize_position(raw.r1_mm, raw.layer1, rng)
        r2_q, ix2, iy2 = self._quantize_position(raw.r2_mm, np.maximum(raw.layer2, 0), rng)

        # 3. Apply electronic trigger threshold E_th
        trig1 = e1_sm >= self.threshold_kev
        trig2 = is_two & (e2_sm >= self.threshold_kev)

        # Events where neither hit triggers are completely unseen by readout
        keep = trig1 | trig2
        trig1, trig2 = trig1[keep], trig2[keep]
        r1_q, e1_sm, ix1, iy1, l1 = r1_q[keep], e1_sm[keep], ix1[keep], iy1[keep], raw.layer1[keep].copy()
        r2_q, e2_sm, ix2, iy2, l2 = r2_q[keep], e2_sm[keep], ix2[keep], iy2[keep], raw.layer2[keep].copy()
        t_type = raw.true_type[keep].copy()
        e0_t = raw.e0_true_kev[keep].copy()

        m = r1_q.shape[0]
        both_trig = trig1 & trig2
        only_hit2 = (~trig1) & trig2

        # If Hit 1 was below threshold (e.g., very small-angle scatter) and only Hit 2 triggered in Layer 2,
        # promote Hit 2 to slot 1 -> becomes a contaminated single-hit event (invalid for collimated Layer 1!)
        if np.any(only_hit2):
            r1_q[only_hit2] = r2_q[only_hit2]
            e1_sm[only_hit2] = e2_sm[only_hit2]
            ix1[only_hit2] = ix2[only_hit2]
            iy1[only_hit2] = iy2[only_hit2]
            l1[only_hit2] = l2[only_hit2]
            t_type[only_hit2] = EVT_INCOMPLETE_OR_BG

        n_hits_dig = np.where(both_trig, 2, 1).astype(np.int32)
        r2_q[~both_trig] = 0.0
        e2_sm[~both_trig] = 0.0
        l2[~both_trig] = -1

        # Build 4-class ground truth label before random hit-order shuffling
        labels = np.full(m, CLS_INVALID_ESCAPE_BG, dtype=np.int64)
        labels[(n_hits_dig == 1) & (l1 == 0) & (t_type == EVT_COLLIMATED_CLEAN)] = CLS_VALID_COLLIMATED
        labels[(n_hits_dig == 2) & (t_type == EVT_COMPTON_CLEAN_FWD)] = CLS_COMPTON_FWD_1TO2

        # 4. Simulate arbitrary arrival ordering in list-mode coincidence packets (swap Hit A <-> Hit B with p=0.5)
        swap_mask = (n_hits_dig == 2) & (rng.random(m) < 0.5)
        if np.any(swap_mask):
            r1_tmp = r1_q[swap_mask].copy()
            r1_q[swap_mask] = r2_q[swap_mask]
            r2_q[swap_mask] = r1_tmp

            e1_tmp = e1_sm[swap_mask].copy()
            e1_sm[swap_mask] = e2_sm[swap_mask]
            e2_sm[swap_mask] = e1_tmp

            l1_tmp = l1[swap_mask].copy()
            l1[swap_mask] = l2[swap_mask]
            l2[swap_mask] = l1_tmp

            ix_tmp, iy_tmp = ix1[swap_mask].copy(), iy1[swap_mask].copy()
            ix1[swap_mask], iy1[swap_mask] = ix2[swap_mask], iy2[swap_mask]
            ix2[swap_mask], iy2[swap_mask] = ix_tmp, iy_tmp

            # Update forward Compton label -> reverse-ordered Compton label!
            rev_true = swap_mask & (labels == CLS_COMPTON_FWD_1TO2)
            labels[rev_true] = CLS_COMPTON_REV_2TO1

        feats = build_event_features(r1_q, e1_sm, l1, r2_q, e2_sm, l2, n_hits_dig)
        return DigitizedEvents(
            n_hits=n_hits_dig,
            pix_ix1=ix1,
            pix_iy1=iy1,
            r1_dig_mm=r1_q.astype(np.float32),
            e1_dig_kev=e1_sm.astype(np.float32),
            layer1=l1.astype(np.int32),
            r2_dig_mm=r2_q.astype(np.float32),
            e2_dig_kev=e2_sm.astype(np.float32),
            layer2=l2.astype(np.int32),
            e0_true_kev=e0_t.astype(np.float32),
            label_4cls=labels,
            features_16d=feats,
        )


def build_event_features(
    r1: np.ndarray,
    e1: np.ndarray,
    l1: np.ndarray,
    r2: np.ndarray,
    e2: np.ndarray,
    l2: np.ndarray,
    n_hits: np.ndarray,
) -> np.ndarray:
    """
    Construct 16 normalized physics-informed features per photon event for Stage-1 LiteEventNet:
    [0..2]   r1 / 35.0
    [3]      e1 / 1000.0
    [4]      layer1 (0 or 1)
    [5..7]   r2 / 35.0
    [8]      e2 / 1000.0
    [9]      layer2 (-1, 0, or 1)
    [10]     is_two_hit (0 or 1)
    [11]     (e1 + e2) / 1000.0
    [12]     ||r2 - r1|| / 50.0  (0 if single hit)
    [13]     (z1 - z2) / 35.0    (positive if hit A is in front scatterer and hit B in back absorber!)
    [14]     clipped kinematic cos(theta_{A->B})
    [15]     kinematic validity flag |cos(theta_{A->B})| <= 1 and |cos(theta_{B->A})|
    """
    is_two = (n_hits == 2).astype(np.float32)
    e_sum = e1 + e2
    dr = np.where(is_two[:, None] > 0.5, r2 - r1, 0.0)
    dist = np.linalg.norm(dr, axis=-1)
    dz = np.where(is_two > 0.5, r1[:, 2] - r2[:, 2], 0.0)

    ct_ab = np.where(is_two > 0.5, compton_cos_theta_from_energies(e1, e2), 0.0)
    ct_ba = np.where(is_two > 0.5, compton_cos_theta_from_energies(e2, e1), 0.0)

    valid_ab = ((ct_ab >= -1.0) & (ct_ab <= 1.0) & (is_two > 0.5)).astype(np.float32)
    valid_ba = ((ct_ba >= -1.0) & (ct_ba <= 1.0) & (is_two > 0.5)).astype(np.float32)

    feats = np.column_stack([
        r1[:, 0] / 35.0,
        r1[:, 1] / 35.0,
        r1[:, 2] / 35.0,
        e1 / 1000.0,
        l1.astype(np.float32),
        r2[:, 0] / 35.0,
        r2[:, 1] / 35.0,
        r2[:, 2] / 35.0,
        e2 / 1000.0,
        l2.astype(np.float32),
        is_two,
        e_sum / 1000.0,
        dist / 50.0,
        dz / 35.0,
        np.clip(ct_ab, -2.0, 2.0),
        valid_ab - valid_ba,
    ])
    return feats.astype(np.float32)
