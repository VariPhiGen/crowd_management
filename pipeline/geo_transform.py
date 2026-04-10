"""
geo_transform.py — Affine mapping from floor (x, y) metres to (lat, lng).

Given 3+ reference points where both floor coordinates and GPS coordinates
are known, compute a least-squares affine transform and apply it to every
row of a fused-crossings CSV, adding ``latitude`` and ``longitude`` columns.

Includes two spatial-quality corrections:

  1. **Edge jitter** — crossing coordinates snap to integer grid positions
     (e.g. crossing_y = 73.0 on edge y_73).  A small uniform random offset
     (±half the grid spacing, default ±0.5 m) is added to the snapped axis
     before the affine transform.  This breaks the parallel-line artefact
     without changing aggregate statistics.

  2. **Density weight** — cameras detect more crossings near their mounting
     position (perspective resolution gradient).  A per-camera density
     profile is estimated from the data and each row receives a ``weight``
     column = 1 / relative_density.  When used for spatial analysis the
     client should multiply counts by ``weight`` to get uniform coverage.
"""
from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Grid spacing in metres (edges are placed every GRID_STEP_M along each axis).
GRID_STEP_M = 1.0

# Half-width of uniform jitter applied to the snapped coordinate.
_JITTER_HALF = GRID_STEP_M * 0.5

# Bandwidth (in metres) for the Gaussian KDE used in density estimation.
_DENSITY_BW_M = 5.0

# Minimum density weight (caps very sparse regions to avoid extreme weights).
_MIN_WEIGHT = 0.1
_MAX_WEIGHT = 10.0


@dataclass
class GeoRefPoint:
    floor_x: float
    floor_y: float
    lat: float
    lng: float


# ───────────────────────────────────────────────────────────────────────────
#  Validation
# ───────────────────────────────────────────────────────────────────────────

def _validate_points(pts: list[GeoRefPoint]) -> list[str]:
    """Return a list of human-readable validation errors (empty = OK)."""
    errors: list[str] = []
    if len(pts) < 3:
        errors.append(f"Need at least 3 reference points, got {len(pts)}.")
        return errors

    for i, p in enumerate(pts, 1):
        if not (-90 <= p.lat <= 90):
            errors.append(f"Point {i}: latitude {p.lat} outside [-90, 90].")
        if not (-180 <= p.lng <= 180):
            errors.append(f"Point {i}: longitude {p.lng} outside [-180, 180].")

    coords = np.array([[p.floor_x, p.floor_y] for p in pts])
    centered = coords - coords.mean(axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s[-1] < 1e-6:
        errors.append("Reference points are (nearly) collinear — need non-collinear points.")

    return errors


# ───────────────────────────────────────────────────────────────────────────
#  Affine fitting
# ───────────────────────────────────────────────────────────────────────────

def fit_affine(pts: list[GeoRefPoint]) -> tuple[np.ndarray, np.ndarray]:
    """Fit affine transform: floor (x,y) -> (lat, lng).

    Returns (A, residuals) where A is a (2, 3) matrix such that:
        [lat]   =  A  @  [x]
        [lng]             [y]
                          [1]

    *residuals* is a (N, 2) array of per-point reprojection errors in
    (lat, lng) space.
    """
    n = len(pts)
    B = np.zeros((n, 3))
    L = np.zeros((n, 2))
    for i, p in enumerate(pts):
        B[i] = [p.floor_x, p.floor_y, 1.0]
        L[i] = [p.lat, p.lng]

    result, _, _, _ = np.linalg.lstsq(B, L, rcond=None)
    A = result.T  # shape (2, 3)

    predicted = B @ result  # (N, 2)
    residuals = predicted - L

    return A, residuals


def reprojection_error_m(residuals: np.ndarray, ref_lat: float) -> float:
    """Convert lat/lng residuals to approximate metres (Haversine approx).

    Uses 1° lat ≈ 111_320 m, 1° lng ≈ 111_320 * cos(lat) m.
    Returns RMS error in metres across all points.
    """
    lat_scale = 111_320.0
    lng_scale = 111_320.0 * np.cos(np.radians(ref_lat))
    m_err = np.column_stack([residuals[:, 0] * lat_scale,
                              residuals[:, 1] * lng_scale])
    return float(np.sqrt(np.mean(m_err ** 2)))


# ───────────────────────────────────────────────────────────────────────────
#  Density weight computation
# ───────────────────────────────────────────────────────────────────────────

def _compute_camera_density_profiles(
    rows: list[tuple[str, str, float, float]],
) -> dict[str, np.ndarray]:
    """Build a per-camera density profile along the primary (Y) axis.

    For each camera, bins crossing_y into 1-m buckets and smooths with a
    Gaussian kernel.  Returns {camera_id: array_of_density_per_y_bin}.
    The density values are normalised so their mean = 1.0 per camera.
    """
    cam_ys: dict[str, list[float]] = defaultdict(list)
    for cam_id, edge_id, cx, cy in rows:
        if cam_id.startswith("fused:"):
            continue
        cam_ys[cam_id].append(cy)

    y_global_min = min(v for vs in cam_ys.values() for v in vs) if cam_ys else 0
    y_global_max = max(v for vs in cam_ys.values() for v in vs) if cam_ys else 100
    bin_min = int(np.floor(y_global_min)) - 2
    bin_max = int(np.ceil(y_global_max)) + 2
    n_bins = bin_max - bin_min + 1
    bin_centers = np.arange(bin_min, bin_max + 1, dtype=np.float64) + 0.5

    gauss = np.exp(-0.5 * ((np.arange(n_bins) - n_bins // 2) / _DENSITY_BW_M) ** 2)
    gauss /= gauss.sum()

    profiles: dict[str, np.ndarray] = {}
    for cam_id, ys in cam_ys.items():
        hist = np.zeros(n_bins, dtype=np.float64)
        for y_val in ys:
            idx = int(round(y_val - bin_min))
            if 0 <= idx < n_bins:
                hist[idx] += 1
        smoothed = np.convolve(hist, gauss, mode="same")
        mean_d = smoothed.mean()
        if mean_d > 0:
            smoothed /= mean_d
        smoothed = np.clip(smoothed, 1e-6, None)
        profiles[cam_id] = smoothed

    profiles["_bin_min"] = np.array([bin_min])
    return profiles


def _lookup_weight(
    profiles: dict[str, np.ndarray],
    cam_id: str,
    cy: float,
) -> float:
    """Look up the inverse-density weight for a crossing at (cam_id, cy)."""
    base_cam = cam_id
    if cam_id.startswith("fused:"):
        parts = cam_id.replace("fused:", "").split("+")
        base_cam = parts[0] if parts else cam_id

    profile = profiles.get(base_cam)
    if profile is None:
        return 1.0

    bin_min = int(profiles["_bin_min"][0])
    idx = int(round(cy - bin_min))
    idx = max(0, min(idx, len(profile) - 1))
    density = profile[idx]
    weight = 1.0 / density
    return float(np.clip(weight, _MIN_WEIGHT, _MAX_WEIGHT))


# ───────────────────────────────────────────────────────────────────────────
#  CSV transform with jitter + weight
# ───────────────────────────────────────────────────────────────────────────

def _is_snapped(val: float) -> bool:
    """True if val is within 0.01 of an integer (snapped to an edge)."""
    return abs(val - round(val)) < 0.01


def transform_csv(
    input_path: str,
    output_path: str,
    affine: np.ndarray,
    apply_jitter: bool = True,
    apply_weight: bool = True,
    rng_seed: int = 42,
) -> int:
    """Read a fused-crossings CSV, add lat/lng/weight columns, write output.

    When *apply_jitter* is True the snapped coordinate of each crossing
    (crossing_x for vertical edges, crossing_y for horizontal edges) is
    perturbed by U(-0.5, +0.5) m before the affine transform.

    When *apply_weight* is True a ``weight`` column (inverse local density)
    is appended so downstream analysis can normalise for the camera
    proximity bias.

    Returns the number of data rows written.
    """
    rng = np.random.default_rng(rng_seed)

    # ── First pass: collect (cam_id, edge_id, cx, cy) for density profiles ──
    meta_rows: list[tuple[str, str, float, float]] = []
    if apply_weight:
        with open(input_path, "r", newline="") as fin:
            reader = csv.reader(fin)
            header = next(reader)
            cx_idx = header.index("crossing_x")
            cy_idx = header.index("crossing_y")
            eid_idx = header.index("edge_id")
            cam_idx = header.index("camera_id")
            for row in reader:
                try:
                    meta_rows.append((
                        row[cam_idx], row[eid_idx],
                        float(row[cx_idx]), float(row[cy_idx]),
                    ))
                except (ValueError, IndexError):
                    pass
        logger.info("Density profiling: scanned %d rows", len(meta_rows))

    profiles = _compute_camera_density_profiles(meta_rows) if apply_weight else {}

    # ── Second pass: transform + write ───────────────────────────────────────
    rows_written = 0
    with open(input_path, "r", newline="") as fin, \
         open(output_path, "w", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        cx_idx = header.index("crossing_x")
        cy_idx = header.index("crossing_y")
        eid_idx = header.index("edge_id")
        cam_idx = header.index("camera_id")

        out_header = header + ["latitude", "longitude"]
        if apply_weight:
            out_header.append("weight")
        writer.writerow(out_header)

        for row in reader:
            try:
                x = float(row[cx_idx])
                y = float(row[cy_idx])
            except (ValueError, IndexError):
                extras = ["", ""]
                if apply_weight:
                    extras.append("")
                writer.writerow(row + extras)
                rows_written += 1
                continue

            jx, jy = x, y
            if apply_jitter:
                edge_id = row[eid_idx] if eid_idx < len(row) else ""
                if edge_id.startswith("x_") and _is_snapped(x):
                    jx = x + rng.uniform(-_JITTER_HALF, _JITTER_HALF)
                elif edge_id.startswith("y_") and _is_snapped(y):
                    jy = y + rng.uniform(-_JITTER_HALF, _JITTER_HALF)

            coords = affine @ np.array([jx, jy, 1.0])
            lat_s = f"{coords[0]:.7f}"
            lng_s = f"{coords[1]:.7f}"

            extras = [lat_s, lng_s]
            if apply_weight:
                cam = row[cam_idx] if cam_idx < len(row) else ""
                w = _lookup_weight(profiles, cam, y)
                extras.append(f"{w:.4f}")

            writer.writerow(row + extras)
            rows_written += 1

    return rows_written


# ───────────────────────────────────────────────────────────────────────────
#  Main entry point
# ───────────────────────────────────────────────────────────────────────────

def run_geo_transform(
    ref_points: list[GeoRefPoint],
    input_csv: str,
    output_csv: str,
    apply_jitter: bool = True,
    apply_weight: bool = True,
) -> dict:
    """Full pipeline: validate → fit → transform (with jitter + weight) → report.

    Returns a dict with keys:
        ok (bool), error (str|None),
        reprojection_error_m (float), rows (int),
        output_path (str), affine (list),
        parquet_name (str|None), parquet_size_mb (float|None),
        corrections (dict)
    """
    errors = _validate_points(ref_points)
    if errors:
        return {"ok": False, "error": "; ".join(errors)}

    if not os.path.isfile(input_csv):
        return {"ok": False, "error": f"Input CSV not found: {input_csv}"}

    affine, residuals = fit_affine(ref_points)
    ref_lat = np.mean([p.lat for p in ref_points])
    rms_m = reprojection_error_m(residuals, ref_lat)

    if rms_m > 50.0:
        return {
            "ok": False,
            "error": (
                f"Reprojection error is {rms_m:.1f} m — too large. "
                "Check that floor_x/y and lat/lng are correct."
            ),
        }

    rows = transform_csv(
        input_csv, output_csv, affine,
        apply_jitter=apply_jitter,
        apply_weight=apply_weight,
    )

    parquet_path = output_csv.replace(".csv", ".parquet")
    try:
        import pandas as pd
        df = pd.read_csv(output_csv, low_memory=False)
        df.to_parquet(parquet_path, index=False, engine="pyarrow")
        parquet_name = Path(parquet_path).name
        parquet_size_mb = round(os.path.getsize(parquet_path) / (1024 * 1024), 1)
        logger.info("Saved parquet: %s (%.1f MB)", parquet_path, parquet_size_mb)
    except Exception as exc:
        logger.warning("Parquet save failed (CSV still OK): %s", exc)
        parquet_name = None
        parquet_size_mb = None

    logger.info(
        "Geo-transform: %d rows, reprojection %.2f m, jitter=%s, weight=%s, saved %s",
        rows, rms_m, apply_jitter, apply_weight, output_csv,
    )

    return {
        "ok": True,
        "error": None,
        "reprojection_error_m": round(rms_m, 3),
        "rows": rows,
        "output_path": output_csv,
        "affine": affine.tolist(),
        "parquet_name": parquet_name,
        "parquet_size_mb": parquet_size_mb,
        "corrections": {
            "jitter_applied": apply_jitter,
            "jitter_half_m": _JITTER_HALF if apply_jitter else 0,
            "density_weight_applied": apply_weight,
        },
    }
