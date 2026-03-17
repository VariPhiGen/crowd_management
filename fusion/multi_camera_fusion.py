"""
fusion/multi_camera_fusion.py
=============================
Fuses line-crossing events from multiple cameras into a single verified stream.
Uses the Hungarian algorithm to match duplicate events across overlapping views.

Calibration-aware accuracy improvements
----------------------------------------
On startup, CrossingFuser tries to load each camera's homography .npz file
(config/homography_{cam_id}.npz).  If it exists, it reads the mean
reprojection error in metres.  That per-camera error is then used to:

  1. Expand the distance threshold dynamically
       threshold = base_threshold + 2 × (err_a + err_b)
     so cameras that were poorly calibrated don't miss real matches.

  2. Weight the fused (x, y) position by inverse-variance
       weight_i = 1 / max(err_i, 0.01)^2
     so well-calibrated cameras dominate the merged coordinate.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# Fallback reprojection error (metres) when no calibration data exists.
# 0.5 m is conservative — typical uncalibrated homography from 6 points.
_DEFAULT_REPROJ_ERROR_M = 0.50


def _load_cam_reproj_error(cam_id: str, config_dir: str) -> float:
    """
    Load mean floor-space reprojection error (metres) from the homography
    .npz file saved by ``calibration/homography.py``.

    Returns a fallback value when the file is missing or unreadable.
    """
    npz_path = Path(config_dir) / f"homography_{cam_id}.npz"
    if not npz_path.exists():
        logger.debug("[%s] No homography .npz found — using fallback error %.2f m",
                     cam_id, _DEFAULT_REPROJ_ERROR_M)
        return _DEFAULT_REPROJ_ERROR_M

    try:
        data = np.load(npz_path, allow_pickle=True)
        H            = data["H"]
        working_pts  = data["working_points"]
        floor_pts    = data["floor_points"]
        inlier_mask  = data["inlier_mask"]

        n = len(working_pts)
        pts_cv = working_pts.reshape(-1, 1, 2).astype(np.float64)
        import cv2
        proj   = cv2.perspectiveTransform(pts_cv, H).reshape(-1, 2)
        errors = np.linalg.norm(proj - floor_pts, axis=1)

        # Use inlier errors only (more representative)
        if inlier_mask is not None:
            mask   = inlier_mask.ravel().astype(bool)
            errors = errors[mask] if mask.any() else errors

        mean_err = float(np.mean(errors))
        logger.info("[%s] Reprojection error loaded: %.4f m (from %s)",
                    cam_id, mean_err, npz_path.name)
        return mean_err

    except Exception as exc:
        logger.warning("[%s] Failed to read %s: %s — using fallback %.2f m",
                       cam_id, npz_path.name, exc, _DEFAULT_REPROJ_ERROR_M)
        return _DEFAULT_REPROJ_ERROR_M


class CrossingFuser:
    """
    Fuses crossing events from multiple cameras by detecting and merging
    duplicates in overlap zones.

    Parameters
    ----------
    overlap_zones_path : str
        Path to overlap_zones.json.
    timestamp_tolerance_s : float
        Maximum time difference (seconds) between two events to consider
        them as the same crossing.
    config_dir : str
        Directory containing ``cameras.json`` and
        ``homography_{cam_id}.npz`` files.  Used to load per-camera
        reprojection errors that improve fusion accuracy.
    """

    def __init__(
        self,
        overlap_zones_path: str,
        timestamp_tolerance_s: float = 1.0,
        config_dir: str = "config/",
    ):
        self.timestamp_tolerance_s = timestamp_tolerance_s
        self.config_dir            = config_dir
        self.overlap_zones: List[Dict] = []

        # Per-camera reprojection error (metres) — populated lazily
        self._cam_errors: Dict[str, float] = {}

        zones_path = Path(overlap_zones_path)
        if zones_path.exists():
            try:
                with open(zones_path, "r") as f:
                    data = json.load(f)

                for z in data.get("overlap_zones", []):
                    poly_coords = z.get("floor_polygon")
                    if poly_coords and len(poly_coords) >= 3:
                        z["shapely_poly"] = Polygon(poly_coords)
                    else:
                        z["shapely_poly"] = None
                    self.overlap_zones.append(z)
                logger.info("Loaded %d overlap zones for fusion.", len(self.overlap_zones))
            except Exception as e:
                logger.warning("Failed to load overlap zones %s: %s", overlap_zones_path, e)
        else:
            logger.warning(
                "Overlap zones file not found: %s. No spatial fusion will occur.",
                overlap_zones_path,
            )

    # ------------------------------------------------------------------
    # Per-camera calibration helpers
    # ------------------------------------------------------------------

    def _get_cam_error(self, cam_id: str) -> float:
        """Return cached (or freshly loaded) reprojection error for cam_id."""
        if cam_id not in self._cam_errors:
            self._cam_errors[cam_id] = _load_cam_reproj_error(cam_id, self.config_dir)
        return self._cam_errors[cam_id]

    def _dynamic_threshold(self, base_thresh: float, cam_a: str, cam_b: str) -> float:
        """
        Expand the spatial distance threshold by the combined projection
        uncertainty of both cameras.

        Formula:
            threshold = base_thresh + 2 × (err_a + err_b)

        This ensures that a point whose projection can be off by ±err_a
        and another whose projection can be off by ±err_b are still matched
        even when the true floor separation is near zero.
        """
        err_a = self._get_cam_error(cam_a)
        err_b = self._get_cam_error(cam_b)
        return base_thresh + 2.0 * (err_a + err_b)

    def _weighted_fuse_xy(
        self,
        x_a: float, y_a: float, cam_a: str,
        x_b: float, y_b: float, cam_b: str,
    ) -> tuple:
        """
        Inverse-variance weighted average of two floor positions.

        weight_i = 1 / max(err_i, 0.01)^2

        A camera with 0.05 m error gets weight 400; one with 0.50 m error
        gets weight 4 — so the better camera dominates the merged position.
        """
        err_a = max(self._get_cam_error(cam_a), 0.01)
        err_b = max(self._get_cam_error(cam_b), 0.01)
        w_a = 1.0 / (err_a ** 2)
        w_b = 1.0 / (err_b ** 2)
        total = w_a + w_b
        fused_x = (x_a * w_a + x_b * w_b) / total
        fused_y = (y_a * w_a + y_b * w_b) / total
        return fused_x, fused_y

    # ------------------------------------------------------------------
    # Trajectory-based track identity mapping
    # ------------------------------------------------------------------

    def _build_track_identity_map(
        self,
        cam_a: str,
        cam_b: str,
        zone: Dict,
        output_dir: str,
    ) -> Dict[int, int]:
        """
        Load per-camera track CSVs (cam_X_tracks.csv), filter positions to the
        overlap zone polygon, then match track_ids across cameras using centroid
        proximity (Hungarian algorithm).

        Returns
        -------
        dict
            Mapping  cam_a_track_id  →  cam_b_track_id  for confirmed pairs.
        """
        poly = zone.get("shapely_poly")
        if poly is None:
            return {}

        out = Path(output_dir)
        path_a = out / f"{cam_a}_tracks.csv"
        path_b = out / f"{cam_b}_tracks.csv"

        if not path_a.exists() or not path_b.exists():
            return {}

        def _load_tracks(path: Path) -> pd.DataFrame:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
            df["floor_x"] = pd.to_numeric(df["floor_x"], errors="coerce")
            df["floor_y"] = pd.to_numeric(df["floor_y"], errors="coerce")
            df["track_id"] = pd.to_numeric(df["track_id"], errors="coerce")
            before = len(df)
            df = df.dropna(subset=["timestamp", "floor_x", "floor_y", "track_id"])
            df["track_id"] = df["track_id"].astype(int)
            dropped = before - len(df)
            if dropped:
                logger.warning("Tracks CSV %s: dropped %d/%d malformed rows", path.name, dropped, before)
            return df

        try:
            tracks_a = _load_tracks(path_a)
            tracks_b = _load_tracks(path_b)
        except Exception as exc:
            logger.warning("Could not load tracks CSVs for %s/%s: %s", cam_a, cam_b, exc)
            return {}

        # Filter to overlap zone
        def in_zone(df: pd.DataFrame) -> pd.DataFrame:
            mask = [
                poly.contains(Point(float(x), float(y)))
                for x, y in zip(df["floor_x"], df["floor_y"])
            ]
            return df[mask]

        za = in_zone(tracks_a)
        zb = in_zone(tracks_b)

        if za.empty or zb.empty:
            return {}

        # Compute centroid per track_id
        cent_a = za.groupby("track_id")[["floor_x", "floor_y"]].mean()
        cent_b = zb.groupby("track_id")[["floor_x", "floor_y"]].mean()

        ids_a = cent_a.index.tolist()
        ids_b = cent_b.index.tolist()

        if not ids_a or not ids_b:
            return {}

        # Build cost matrix (Euclidean distance between centroids)
        arr_a = cent_a.values                         # shape (na, 2)
        arr_b = cent_b.values                         # shape (nb, 2)
        cost = np.sqrt(
            ((arr_a[:, None, :] - arr_b[None, :, :]) ** 2).sum(axis=2)
        )                                             # shape (na, nb)

        row_idx, col_idx = linear_sum_assignment(cost)

        # Accept pairs within the dynamic distance threshold
        dyn_thresh = self._dynamic_threshold(
            zone.get("distance_threshold_m", 1.0), cam_a, cam_b
        )

        identity_map: Dict[int, int] = {}
        for r, c in zip(row_idx, col_idx):
            if cost[r, c] <= dyn_thresh:
                identity_map[int(ids_a[r])] = int(ids_b[c])
                logger.debug(
                    "Trajectory match: %s.track_%d ↔ %s.track_%d  dist=%.2f m",
                    cam_a, ids_a[r], cam_b, ids_b[c], cost[r, c],
                )

        logger.info(
            "Trajectory pre-match %s↔%s: %d/%d track pairs confirmed (threshold=%.2f m)",
            cam_a, cam_b, len(identity_map), min(len(ids_a), len(ids_b)), dyn_thresh,
        )
        return identity_map

    # ------------------------------------------------------------------
    # Public: fuse
    # ------------------------------------------------------------------

    def fuse(self, csv_paths: List[str]) -> pd.DataFrame:
        """
        Loads all per-camera CSVs, concatenates them, and merges duplicates
        across cameras within overlap zones using calibration-aware matching.
        """
        if not csv_paths:
            logger.info("No CSV paths provided for fusion.")
            return pd.DataFrame()

        # Pre-load reprojection errors for all cameras that appear in the CSVs
        # so we can report them before fusion starts.
        all_cam_ids: set = set()

        dfs = []
        for p in csv_paths:
            try:
                df = pd.read_csv(p)
                before = len(df)
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
                bad_rows = df["timestamp"].isna().sum()
                if bad_rows:
                    logger.warning(
                        "CSV %s: %d/%d rows had unparseable timestamps — dropped.",
                        p, bad_rows, before,
                    )
                    df = df.dropna(subset=["timestamp"])
                if df.empty:
                    logger.error("CSV %s has no valid rows after timestamp parse — skipping.", p)
                    continue
                dfs.append(df)
                if "camera_id" in df.columns:
                    all_cam_ids.update(df["camera_id"].dropna().unique())
                logger.info("Loaded %s: %d events (dropped %d bad rows)", p, len(df), bad_rows)
            except Exception as e:
                logger.error("Failed to read CSV %s: %s", p, e)

        if not dfs:
            return pd.DataFrame()

        # Pre-cache + log errors for all cameras in the dataset
        for cid in sorted(all_cam_ids):
            err = self._get_cam_error(cid)
            logger.info(
                "  Camera %-12s  reprojection error = %.4f m  "
                "(threshold bonus = +%.4f m)",
                cid, err, 2.0 * err,
            )

        combined_df = pd.concat(dfs, ignore_index=True)
        if combined_df.empty:
            return combined_df

        combined_df.sort_values("timestamp", inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        unique_cams = combined_df["camera_id"].unique()
        if len(unique_cams) <= 1 or not self.overlap_zones:
            logger.info("Only 1 camera or no overlap zones — passing through data.")
            return combined_df

        # ── Step 1: Trajectory pre-matching ──────────────────────────────────
        #
        # Load cam_X_tracks.csv for each camera pair.  Match track_ids across
        # cameras by centroid proximity inside the overlap zone.
        # Result: confirmed_pairs = set of (cam_a_id, track_a, cam_b_id, track_b)
        # Crossing events belonging to a confirmed pair are fused immediately —
        # no spatial distance check needed (trajectory evidence is stronger).
        # ─────────────────────────────────────────────────────────────────────

        # Derive output_dir from the CSV paths (all in the same folder)
        output_dir = str(Path(csv_paths[0]).parent)

        # Build one identity map per (zone, camera-pair)
        # identity_maps: dict[zone_id] -> dict[(cam_a, cam_b)] -> {track_a: track_b}
        identity_maps: Dict[str, Any] = {}
        cam_ids_sorted = sorted(unique_cams)
        for z in self.overlap_zones:
            zone_cams = z.get("cameras", [])
            zid = z.get("id", "unknown")
            identity_maps[zid] = {}
            for i in range(len(zone_cams)):
                for j in range(i + 1, len(zone_cams)):
                    ca, cb = zone_cams[i], zone_cams[j]
                    tmap = self._build_track_identity_map(ca, cb, z, output_dir)
                    identity_maps[zid][(ca, cb)] = tmap

        def get_trajectory_partner(
            cam_a: str, track_a: int,
            cam_b: str, zones_a: List[Dict],
        ) -> Optional[int]:
            """Return cam_b track_id if trajectory pre-match exists, else None."""
            for z in zones_a:
                zid = z.get("id", "unknown")
                tmap = identity_maps.get(zid, {}).get((cam_a, cam_b))
                if tmap is None:
                    # Try reverse key order
                    tmap_rev = identity_maps.get(zid, {}).get((cam_b, cam_a))
                    if tmap_rev:
                        # reverse the map
                        rev = {v: k for k, v in tmap_rev.items()}
                        if track_a in rev:
                            return rev[track_a]
                else:
                    if track_a in tmap:
                        return tmap[track_a]
            return None

        # ── Step 2: Spatial+temporal matching (with trajectory boost) ─────────
        #
        # For each event in the overlap zone:
        #   a) If a trajectory-confirmed partner exists from another camera
        #      → fuse ANY crossing events for that pair within the time window,
        #        regardless of spatial distance (they're confirmed the same person)
        #   b) Otherwise → fall back to spatial+temporal distance check
        # ─────────────────────────────────────────────────────────────────────

        fused_records   = []
        matched_indices = set()

        # ── Global person ID assignment ───────────────────────────────────────
        # ByteTrack IDs are LOCAL per camera (both cam_1 and cam_2 start at 1).
        # Here we assign a single incremental global_person_id so that the same
        # physical person always gets the same number regardless of which camera
        # saw them.
        #
        # Rules:
        #   • When two events from different cameras are fused (same person),
        #     both (cam_a, local_tid_a) and (cam_b, local_tid_b) map to one
        #     global_person_id — assigned the first time either key is seen.
        #   • Single-camera events that never matched another camera get their
        #     own unique global_person_id.
        #   • IDs are assigned in chronological order of first appearance, so
        #     they are incremental by time (person 1 appears before person 2).
        _gpid_counter: int = 0
        _gpid_map: Dict[tuple, int] = {}   # (camera_id, local_track_id) → global_person_id

        def _get_or_create_gpid(cam: str, tid) -> int:
            nonlocal _gpid_counter
            key = (str(cam), int(tid))
            if key not in _gpid_map:
                _gpid_counter += 1
                _gpid_map[key] = _gpid_counter
            return _gpid_map[key]

        def _link_gpid(cam_primary: str, tid_primary, cam_secondary: str, tid_secondary) -> int:
            """Ensure both camera-local IDs share the same global person ID."""
            gpid = _get_or_create_gpid(cam_primary, tid_primary)
            _gpid_map[(str(cam_secondary), int(tid_secondary))] = gpid
            return gpid
        # ─────────────────────────────────────────────────────────────────────

        def get_valid_zones(pt: Point, cam: str) -> List[Dict]:
            return [
                z for z in self.overlap_zones
                if cam in z.get("cameras", [])
                and z["shapely_poly"] is not None
                and z["shapely_poly"].contains(pt)
            ]

        points = [
            Point(x, y)
            for x, y in zip(combined_df["crossing_x"], combined_df["crossing_y"])
        ]

        n = len(combined_df)

        for idx_a in range(n):
            if idx_a in matched_indices:
                continue

            row_a   = combined_df.iloc[idx_a]
            t_a     = row_a["timestamp"]
            pt_a    = points[idx_a]
            cam_a   = row_a["camera_id"]
            zones_a = get_valid_zones(pt_a, cam_a)

            if not zones_a:
                continue  # not in any overlap zone → can't fuse

            candidates = []

            # Scan forward in time only (data is sorted by timestamp)
            for idx_b in range(idx_a + 1, n):
                if idx_b in matched_indices:
                    continue

                row_b     = combined_df.iloc[idx_b]
                time_diff = (row_b["timestamp"] - t_a).total_seconds()

                if time_diff > self.timestamp_tolerance_s:
                    break   # all further events are also out of window

                cam_b = row_b["camera_id"]
                if cam_a == cam_b:
                    continue

                pt_b = points[idx_b]

                # ── Trajectory pre-match check ────────────────────────────
                traj_partner = get_trajectory_partner(
                    cam_a, int(row_a["track_id"]),
                    cam_b, zones_a,
                )
                if traj_partner is not None and int(row_b["track_id"]) == traj_partner:
                    candidates.append((idx_b, 0.0, row_b))   # dist=0: forced match
                    break

                # ── Fallback: spatial+temporal check ──────────────────────
                for z in zones_a:
                    if cam_b not in z.get("cameras", []):
                        continue
                    if z["shapely_poly"] is None or not z["shapely_poly"].contains(pt_b):
                        continue

                    base_thresh = z.get("distance_threshold_m", 1.0)
                    dyn_thresh  = self._dynamic_threshold(base_thresh, cam_a, cam_b)
                    dist        = pt_a.distance(pt_b)

                    if dist <= dyn_thresh:
                        candidates.append((idx_b, dist, row_b))
                        break

            if candidates:
                candidates.sort(key=lambda x: x[1])
                best_idx, best_dist, row_b = candidates[0]

                matched_indices.add(idx_a)
                matched_indices.add(best_idx)

                cam_b = row_b["camera_id"]

                # Assign one shared global_person_id to both camera-local IDs
                gpid = _link_gpid(cam_a, row_a["track_id"], cam_b, row_b["track_id"])

                # Inverse-variance weighted position
                fused_x, fused_y = self._weighted_fuse_xy(
                    row_a["crossing_x"], row_a["crossing_y"], cam_a,
                    row_b["crossing_x"], row_b["crossing_y"], cam_b,
                )

                import random
                best_edge_id = random.choice([row_a["edge_id"], row_b["edge_id"]])

                fused_records.append({
                    "timestamp":        min(row_a["timestamp"], row_b["timestamp"]),
                    "global_person_id": gpid,
                    "track_id":         row_a["track_id"],
                    "class_name":       row_a["class_name"],
                    "edge_id":          best_edge_id,
                    "direction":        row_a["direction"],
                    "crossing_x":       fused_x,
                    "crossing_y":       fused_y,
                    "camera_id":        f"fused:{cam_a}+{cam_b}",
                })

        # Add unmatched records, assigning each a unique global_person_id
        for i in range(len(combined_df)):
            if i not in matched_indices:
                row  = combined_df.iloc[i]
                gpid = _get_or_create_gpid(str(row["camera_id"]), row["track_id"])
                rec  = row.to_dict()
                rec["global_person_id"] = gpid
                fused_records.append(rec)

        fused_df = pd.DataFrame(fused_records)
        if not fused_df.empty:
            fused_df.sort_values("timestamp", inplace=True)
            fused_df.reset_index(drop=True, inplace=True)

        logger.info(
            "Fusion complete. %d original events → %d fused events "
            "(%d duplicates removed). %d unique global person IDs assigned.",
            len(combined_df),
            len(fused_df),
            len(combined_df) - len(fused_df),
            _gpid_counter,
        )
        return fused_df

    # ------------------------------------------------------------------
    # Public: save / summary
    # ------------------------------------------------------------------

    def save_fused_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Saves the fused DataFrame to a CSV preserving exact column structure
        and rounding coordinates to 2 decimal places.
        """
        if df.empty:
            logger.warning("Attempted to save empty fusion dataframe to %s", output_path)
            pd.DataFrame(columns=[
                "timestamp", "global_person_id", "track_id", "class_name", "edge_id",
                "direction", "crossing_x", "crossing_y", "camera_id",
            ]).to_csv(output_path, index=False)
            return

        df_out = df.copy()
        df_out["crossing_x"] = df_out["crossing_x"].round(2)
        df_out["crossing_y"] = df_out["crossing_y"].round(2)

        # global_person_id: incremental cross-camera person identity.
        # Ensure the column exists (older fused CSVs may not have it).
        if "global_person_id" not in df_out.columns:
            df_out["global_person_id"] = range(1, len(df_out) + 1)
        df_out["global_person_id"] = df_out["global_person_id"].astype(int)

        cols = [
            "timestamp", "global_person_id", "track_id", "class_name", "edge_id",
            "direction", "crossing_x", "crossing_y", "camera_id",
        ]
        if pd.api.types.is_datetime64_any_dtype(df_out["timestamp"]):
            df_out["timestamp"] = (
                df_out["timestamp"]
                .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
                .str[:-3]
            )
        df_out.to_csv(output_path, columns=cols, index=False)
        logger.info("Saved fused crossings to %s", output_path)

    def get_summary(
        self,
        original_dfs: List[pd.DataFrame],
        fused_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Generates a summary dictionary of the fusion process."""
        summary = {
            "total_original_events": 0,
            "events_per_camera":    {},
            "total_fused_events":   len(fused_df) if not fused_df.empty else 0,
            "duplicates_removed":   0,
            "events_per_edge":      {},
            "events_per_direction": {},
            "camera_reproj_errors_m": {
                cid: round(err, 4)
                for cid, err in self._cam_errors.items()
            },
        }

        for df in original_dfs:
            if not df.empty:
                count  = len(df)
                summary["total_original_events"] += count
                cam_id = (
                    df["camera_id"].iloc[0]
                    if "camera_id" in df.columns
                    else "unknown"
                )
                summary["events_per_camera"][cam_id] = count

        summary["duplicates_removed"] = (
            summary["total_original_events"] - summary["total_fused_events"]
        )

        if not fused_df.empty:
            summary["events_per_edge"]      = fused_df["edge_id"].value_counts().to_dict()
            summary["events_per_direction"] = fused_df["direction"].value_counts().to_dict()

        return summary
