"""
NIST XCOM-calibrated photon interaction cross-sections for GAGG:Ce and Tungsten (W),
exact Klein-Nishina differential Compton angular sampling, and Doppler broadening.
"""

from __future__ import annotations
import numpy as np

MEC2_KEV = 511.0  # Electron rest mass energy in keV

# NIST XCOM tabulated log-log control points: Energy (keV) from 20 keV to 3000 keV
_E_GRID_KEV = np.array(
    [20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0, 511.0, 662.0, 800.0, 1000.0, 1332.0, 2000.0, 3000.0],
    dtype=np.float64,
)

# GAGG:Ce (Gd3Al2Ga3O12:Ce, density = 6.63 g/cm^3, Z_eff ~ 54, Gd K-edge at 50.2 keV)
# Linear attenuation coefficients mu (cm^-1) calibrated to NIST XCOM mixture rule:
_GAGG_MU_PE_CM1 = np.array(
    [145.0, 48.5, 22.0, 12.1, 41.8, 19.8, 10.9, 3.55, 1.62, 0.54, 0.25, 0.135, 0.068, 0.042, 0.025, 0.013, 0.006, 0.003],
    dtype=np.float64,
)
_GAGG_MU_COMP_CM1 = np.array(
    [0.95, 0.98, 0.99, 0.98, 0.97, 0.93, 0.89, 0.80, 0.73, 0.63, 0.56, 0.505, 0.448, 0.410, 0.368, 0.318, 0.258, 0.206],
    dtype=np.float64,
)
_GAGG_MU_PAIR_CM1 = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006, 0.032, 0.075],
    dtype=np.float64,
)

# Tungsten Alloy W (density = 19.3 g/cm^3, Z = 74, W K-edge at 69.5 keV)
# Total linear attenuation coefficient mu_total (cm^-1):
_W_MU_TOT_CM1 = np.array(
    [820.0, 435.0, 205.0, 114.0, 70.5, 150.0, 85.6, 30.5, 15.1, 6.25, 3.70, 2.52, 1.80, 1.48, 1.23, 1.01, 0.85, 0.78],
    dtype=np.float64,
)


def gagg_attenuation_mm1(energy_kev: np.ndarray | float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return linear attenuation coefficients (mu_pe, mu_comp, mu_pair, mu_tot) in mm^-1 for GAGG:Ce.
    Note: 1 cm^-1 = 0.1 mm^-1.
    """
    e = np.clip(np.asarray(energy_kev, dtype=np.float64), 20.0, 3000.0)
    log_e = np.log(e)
    log_grid = np.log(_E_GRID_KEV)

    mu_pe = 0.1 * np.exp(np.interp(log_e, log_grid, np.log(_GAGG_MU_PE_CM1)))
    mu_comp = 0.1 * np.exp(np.interp(log_e, log_grid, np.log(_GAGG_MU_COMP_CM1)))
    mu_pair = 0.1 * np.interp(e, _E_GRID_KEV, _GAGG_MU_PAIR_CM1)
    mu_tot = mu_pe + mu_comp + mu_pair
    return mu_pe, mu_comp, mu_pair, mu_tot


def tungsten_mu_mm1(energy_kev: np.ndarray | float) -> np.ndarray:
    """Return total linear attenuation coefficient in mm^-1 for Tungsten collimator mask."""
    e = np.clip(np.asarray(energy_kev, dtype=np.float64), 20.0, 3000.0)
    return 0.1 * np.exp(np.interp(np.log(e), np.log(_E_GRID_KEV), np.log(_W_MU_TOT_CM1)))


def sample_klein_nishina_cos_theta(
    energy_kev: np.ndarray,
    rng: np.random.Generator,
    enable_doppler: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized exact Kahn / rejection sampler for Klein-Nishina differential cross-section,
    plus bound-electron Doppler broadening (mimicking Geant4 G4EmLivermorePhysics).

    Args:
        energy_kev: Incident photon energies E0 (shape [N]).
        rng: NumPy random Generator.
        enable_doppler: Whether to include atomic bound-electron momentum broadening.

    Returns:
        cos_theta: Sampled scattering angle cosine (shape [N]).
        e_recoil_kev: Energy deposited to recoil electron E1 (shape [N]).
        e_scattered_kev: Scattered photon energy E2 (shape [N]).
    """
    e0 = np.asarray(energy_kev, dtype=np.float64)
    n = e0.shape[0]
    alpha = e0 / MEC2_KEV

    cos_theta = np.empty(n, dtype=np.float64)
    remaining = np.arange(n)

    # Vectorized Kahn's rejection loop (converges in ~4-5 iterations for N=10^5)
    while remaining.size > 0:
        m = remaining.size
        a = alpha[remaining]
        r1 = rng.random(m)
        r2 = rng.random(m)
        r3 = rng.random(m)

        p_branch = (1.0 + 2.0 * a) / (9.0 + 2.0 * a)
        branch1 = r1 <= p_branch

        x = np.empty(m, dtype=np.float64)
        accept = np.zeros(m, dtype=bool)

        if np.any(branch1):
            x1 = 1.0 + 2.0 * a[branch1] * r2[branch1]
            x[branch1] = x1
            inv_x1 = 1.0 / x1
            # Rejection test for branch 1
            accept[branch1] = r3[branch1] <= 4.0 * (inv_x1 - inv_x1 * inv_x1)

        b2 = ~branch1
        if np.any(b2):
            x2 = (1.0 + 2.0 * a[b2]) / (1.0 + 2.0 * a[b2] * r2[b2])
            x[b2] = x2
            ct2 = 1.0 - (x2 - 1.0) / a[b2]
            accept[b2] = r3[b2] <= 0.5 * (ct2 * ct2 + 1.0 / x2)

        acc_idx = remaining[accept]
        cos_theta[acc_idx] = np.clip(1.0 - (x[accept] - 1.0) / alpha[acc_idx], -1.0, 1.0)
        remaining = remaining[~accept]

    # Ideal Compton scattered photon energy
    e_scat_ideal = e0 / (1.0 + alpha * (1.0 - cos_theta))

    if enable_doppler:
        # Ribberfors / impulse approximation for bound electrons in GAGG:Ce (effective Hartree-Fock pz width)
        # Typical Compton profile width p_z / (m_e c) ~ 0.012 for valence/outer shells + 0.045 inner shells
        sin_half = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - cos_theta)))
        q_transfer_kev = np.sqrt(np.maximum(1.0, e0 * e0 + e_scat_ideal * e_scat_ideal - 2.0 * e0 * e_scat_ideal * cos_theta))
        pz_over_mc = np.where(rng.random(n) < 0.78, rng.normal(0.0, 0.011, size=n), rng.normal(0.0, 0.036, size=n))
        doppler_shift_kev = (e_scat_ideal / e0) * q_transfer_kev * pz_over_mc * (0.65 + 0.35 * sin_half)
        e_scattered = np.clip(e_scat_ideal + doppler_shift_kev, 2.0, e0 - 0.5)
    else:
        e_scattered = e_scat_ideal

    e_recoil = np.maximum(0.0, e0 - e_scattered)
    return cos_theta, e_recoil, e_scattered


def compton_cos_theta_from_energies(e1_kev: np.ndarray, e2_kev: np.ndarray) -> np.ndarray:
    """Compute kinematic cos(theta) assuming photon scattered with E1 first and absorbed with E2 second."""
    e_tot = np.maximum(e1_kev + e2_kev, 1e-6)
    e2_safe = np.maximum(e2_kev, 1e-6)
    return 1.0 - MEC2_KEV * (1.0 / e2_safe - 1.0 / e_tot)
