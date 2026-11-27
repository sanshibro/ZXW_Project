"""BNCT 478 keV 智能伽马成像 · 正算软件包 (forward_pkg)"""
from .bnct_physics_and_collimators import (
    BNCTSystemGeometry,
    ME_C2_KEV,
    query_gagg_mu,
    query_tungsten_mu,
    build_rank19_mura_mask,
    generate_bnct_boron_distribution,
    simulate_bnct_forward_transport,
)
from .detector_digitizer import (
    DetectorDigitizerConfig,
    digitize_detector_events,
    extract_18d_event_features,
    project_pinhole_channels,
    project_compton_channels,
    reconstruct_parallel_multihole,
    reconstruct_coded_aperture_mura,
)
