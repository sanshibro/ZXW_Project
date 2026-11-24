"""
Bridge script converting standard Geant4 SteppingAction/SensitiveDetector CSV hit logs
into RawMCEvents -> DetectorDigitizer -> LiteEventNet / LiteDualGammaNet format.

Expected CSV columns:
event_id, layer_id, x_mm, y_mm, z_mm, edep_keV, e0_true_keV, true_event_type
"""

from __future__ import annotations
import argparse
import os
import sys
import csv
from collections import defaultdict
import numpy as np
import yaml

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from demo.src.physics.geometry_and_sources import CameraGeometry
from demo.src.physics.mc_simulator import RawMCEvents
from demo.src.physics.digitizer import DetectorDigitizer


def load_geant4_hits_csv(csv_path: str) -> RawMCEvents:
    grouped: dict[int, list[tuple[int, float, float, float, float, float, int]]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            eid = int(row["event_id"])
            grouped[eid].append((
                int(row["layer_id"]),
                float(row["x_mm"]),
                float(row["y_mm"]),
                float(row["z_mm"]),
                float(row["edep_keV"]),
                float(row.get("e0_true_keV", 0.0)),
                int(row.get("true_event_type", 0)),
            ))

    n_ev = len(grouped)
    n_hits = np.zeros(n_ev, dtype=np.int32)
    r1 = np.zeros((n_ev, 3), dtype=np.float64)
    e1 = np.zeros(n_ev, dtype=np.float64)
    l1 = np.zeros(n_ev, dtype=np.int32)
    r2 = np.zeros((n_ev, 3), dtype=np.float64)
    e2 = np.zeros(n_ev, dtype=np.float64)
    l2 = np.full(n_ev, -1, dtype=np.int32)
    e0 = np.zeros(n_ev, dtype=np.float64)
    ttype = np.zeros(n_ev, dtype=np.int64)

    for idx, hits in enumerate(grouped.values()):
        h0 = hits[0]
        l1[idx], r1[idx], e1[idx] = h0[0], [h0[1], h0[2], h0[3]], h0[4]
        e0[idx], ttype[idx] = h0[5], h0[6]
        if len(hits) >= 2:
            h1 = hits[1]
            n_hits[idx] = 2
            l2[idx], r2[idx], e2[idx] = h1[0], [h1[1], h1[2], h1[3]], h1[4]
        else:
            n_hits[idx] = 1

    return RawMCEvents(n_hits, r1, e1, l1, r2, e2, l2, e0, ttype)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Geant4 CSV hits to DigitizedEvents NPZ.")
    parser.add_argument("--input", required=True, help="Path to Geant4 hit CSV file")
    parser.add_argument("--config", default="demo/configs/system_config.yaml")
    parser.add_argument("--output", required=True, help="Output NPZ file path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    geom = CameraGeometry.from_config(cfg)
    digitizer = DetectorDigitizer(geom, cfg["digitizer"])
    raw = load_geant4_hits_csv(args.input)
    dig = digitizer.digitize(raw, np.random.default_rng(42))
    np.savez_compressed(
        args.output,
        features_16d=dig.features_16d,
        label_4cls=dig.label_4cls,
        r1_dig_mm=dig.r1_dig_mm,
        e1_dig_kev=dig.e1_dig_kev,
        r2_dig_mm=dig.r2_dig_mm,
        e2_dig_kev=dig.e2_dig_kev,
    )
    print(f"Converted {dig.n_hits.size} Geant4 events -> {args.output}")


if __name__ == "__main__":
    main()
