"""
geo_transform.py — Affine mapping from floor (x, y) metres to (lat, lng).

Given 3+ reference points where both floor coordinates and GPS coordinates
are known, compute a least-squares affine transform and apply it to every
row of a fused-crossings CSV, adding ``latitude`` and ``longitude`` columns.
"""
from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeoRefPoint:
    floor_x: float
    floor_y: float
    lat: float
    lng: float


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

    # lstsq solves  B @ X = L  in least-squares sense
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


def transform_csv(
    input_path: str,
    output_path: str,
    affine: np.ndarray,
) -> int:
    """Read a fused-crossings CSV, add lat/lng columns, write to output.

    Returns the number of data rows written.
    """
    rows_written = 0

    with open(input_path, "r", newline="") as fin, \
         open(output_path, "w", newline="") as fout:

        reader = csv.reader(fin)
        writer = csv.writer(fout)

        header = next(reader)
        try:
            cx_idx = header.index("crossing_x")
            cy_idx = header.index("crossing_y")
        except ValueError as exc:
            raise ValueError(
                f"CSV missing required column: {exc}. "
                f"Header: {header}"
            ) from exc

        writer.writerow(header + ["latitude", "longitude"])

        for row in reader:
            try:
                x = float(row[cx_idx])
                y = float(row[cy_idx])
            except (ValueError, IndexError):
                writer.writerow(row + ["", ""])
                rows_written += 1
                continue

            coords = affine @ np.array([x, y, 1.0])
            lat_s = f"{coords[0]:.7f}"
            lng_s = f"{coords[1]:.7f}"
            writer.writerow(row + [lat_s, lng_s])
            rows_written += 1

    return rows_written


def run_geo_transform(
    ref_points: list[GeoRefPoint],
    input_csv: str,
    output_csv: str,
) -> dict:
    """Full pipeline: validate → fit → transform → report.

    Returns a dict with keys:
        ok (bool), error (str|None),
        reprojection_error_m (float), rows (int),
        output_path (str), affine (list)
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

    rows = transform_csv(input_csv, output_csv, affine)

    # Also save as Parquet (much smaller, faster to load)
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
        "Geo-transform: %d rows, reprojection %.2f m, saved %s",
        rows, rms_m, output_csv,
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
    }
