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

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
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
        distance_threshold_override_m: float | None = None,
    ):
        self.timestamp_tolerance_s         = timestamp_tolerance_s
        self.distance_threshold_override_m = distance_threshold_override_m
        self.config_dir                    = config_dir
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
            df = pd.read_csv(path, on_bad_lines="skip")
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

        # Filter to overlap zone — vectorised via shapely.contains_xy (140x faster
        # than a Python loop with individual Point objects for millions of rows).
        def in_zone(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            import shapely as _shp
            mask = _shp.contains_xy(poly, df["floor_x"].values, df["floor_y"].values)
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

        # Greedy nearest-neighbour track matching via KD-tree.
        #
        # The old Hungarian (linear_sum_assignment) approach required building
        # an N×M cost matrix which OOM-crashes for large track sets (millions
        # of unique track IDs from a long recording).  Instead:
        #   1. Build a KD-tree on arr_b.
        #   2. For each A-track find its single nearest B-track (k=1)
        #      within dyn_thresh.
        #   3. Greedily assign closest pairs first (sort by distance).
        #      Each B-track can only be claimed once.
        #
        # Greedy NN is O(N log N) and uses O(N) extra memory — safe for
        # millions of centroids — while still finding the correct match in
        # the vast majority of cases (two cameras tracking the same person
        # will have very similar centroid positions).
        arr_a = cent_a.values                         # shape (na, 2)
        arr_b = cent_b.values                         # shape (nb, 2)

        base_d = (self.distance_threshold_override_m
                  if self.distance_threshold_override_m is not None
                  else zone.get("distance_threshold_m", 1.0))
        dyn_thresh = self._dynamic_threshold(base_d, cam_a, cam_b)

        logger.info(
            "  Track centroids: %d for %s, %d for %s (threshold=%.2f m)",
            len(ids_a), cam_a, len(ids_b), cam_b, dyn_thresh,
        )

        tree_b = cKDTree(arr_b)
        # k=1 → find the single nearest B-centroid for every A-centroid
        distances, nearest_b = tree_b.query(arr_a, k=1,
                                             distance_upper_bound=dyn_thresh)

        # Build (distance, a_idx, b_idx) tuples for valid pairs, sort by dist
        valid_pairs = [
            (float(distances[i]), i, int(nearest_b[i]))
            for i in range(len(ids_a))
            if distances[i] <= dyn_thresh
        ]
        valid_pairs.sort(key=lambda x: x[0])

        identity_map: Dict[int, int] = {}
        used_b: set = set()
        for dist, ia, ib in valid_pairs:
            if ib in used_b:
                continue
            identity_map[int(ids_a[ia])] = int(ids_b[ib])
            used_b.add(ib)
            logger.debug(
                "Trajectory match: %s.track_%d ↔ %s.track_%d  dist=%.2f m",
                cam_a, ids_a[ia], cam_b, ids_b[ib], dist,
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
                df = pd.read_csv(p, on_bad_lines="skip")
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
        logger.info("Step 1: building track identity maps …")
        identity_maps: Dict[str, Any] = {}
        cam_ids_sorted = sorted(unique_cams)
        for z in self.overlap_zones:
            zone_cams = z.get("cameras", [])
            zid = z.get("id", "unknown")
            identity_maps[zid] = {}
            for i in range(len(zone_cams)):
                for j in range(i + 1, len(zone_cams)):
                    ca, cb = zone_cams[i], zone_cams[j]
                    logger.info("  Building track map %s ↔ %s (zone: %s) …", ca, cb, zid)
                    tmap = self._build_track_identity_map(ca, cb, z, output_dir)
                    identity_maps[zid][(ca, cb)] = tmap
        logger.info("Step 1 done. Starting Step 2: zone membership precompute …")

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

        # Pre-compute zone membership for every crossing event using vectorised
        # shapely.contains_xy — avoids millions of per-row Python contain() calls.
        import shapely as _shp

        xs_all = combined_df["crossing_x"].values
        ys_all = combined_df["crossing_y"].values
        cam_all = combined_df["camera_id"].values

        # zone_membership[zid] = boolean array (len = n)
        zone_membership: Dict[str, Any] = {}
        for z in self.overlap_zones:
            poly_z = z.get("shapely_poly")
            if poly_z is None:
                zone_membership[z["id"]] = None
                continue
            zone_membership[z["id"]] = _shp.contains_xy(poly_z, xs_all, ys_all)

        def get_valid_zones(idx: int, cam: str) -> List[Dict]:
            """Return overlap zones that contain the point at position idx."""
            return [
                z for z in self.overlap_zones
                if cam in z.get("cameras", [])
                and zone_membership.get(z["id"]) is not None
                and zone_membership[z["id"]][idx]
            ]

        n = len(combined_df)

        # ── Pre-extract all hot columns as numpy arrays ────────────────────
        ts_ns     = combined_df["timestamp"].values.astype("int64")  # ns since epoch
        tol_ns    = int(self.timestamp_tolerance_s * 1_000_000_000)
        track_arr = combined_df["track_id"].values
        cx_arr    = combined_df["crossing_x"].values.astype("float64")
        cy_arr    = combined_df["crossing_y"].values.astype("float64")
        cam_arr   = combined_df["camera_id"].values                   # numpy string array

        # ── Force-flush logging so nohup log shows progress in real-time ──
        _handlers = logging.getLogger().handlers + logger.handlers
        def _flush_log() -> None:
            for h in _handlers:
                h.flush()

        # ── Zone pre-filter ────────────────────────────────────────────────
        import functools, operator, random as _random
        valid_zone_masks = [m for m in zone_membership.values() if m is not None]
        in_any_zone = (functools.reduce(operator.or_, valid_zone_masks)
                       if valid_zone_masks else np.zeros(n, dtype=bool))
        logger.info(
            "Zone pre-filter: %d / %d events inside an overlap zone (%.1f%%)",
            int(in_any_zone.sum()), n, in_any_zone.sum() * 100.0 / max(n, 1),
        )
        _flush_log()

        # ── Identify active camera pairs (those with data) ─────────────────
        present_cams = set(combined_df["camera_id"].unique())
        active_pairs = [
            (ca, cb, z)
            for z in self.overlap_zones
            for i, ca in enumerate(z.get("cameras", []))
            for j, cb in enumerate(z.get("cameras", []))
            if i < j and ca in present_cams and cb in present_cams
        ]
        logger.info("Active camera pairs: %s",
                    [(ca, cb, z["id"]) for ca, cb, z in active_pairs])
        _flush_log()

        # ── VECTORISED MATCHING: per camera-pair, binary-search sliding window
        #
        # For each camera pair (ca, cb) in each overlap zone:
        #   1. Extract zone events for ca (n_a events) and cb (n_b events)
        #      as plain numpy arrays — NO pandas .iloc in the hot loop.
        #   2. Both arrays are already sorted by timestamp (combined_df sorted).
        #   3. For each ca event i_a:
        #        • Use searchsorted to find cb events in [t_a, t_a+tol] → O(log n_b)
        #        • Vectorised numpy distance to all window cb events → O(window_size)
        #        • Trajectory pre-match check → O(window_size) Python
        #        • Greedy nearest-neighbour assignment
        #
        # Complexity: O(n_a × (log n_b + W)) where W ≈ 5 events/camera/second.
        # For n_a=2M and W=5: ~10M operations vs 4.8M×48=230M in the old loop.
        # ─────────────────────────────────────────────────────────────────────

        for pair_idx, (ca, cb, zone) in enumerate(active_pairs):
            logger.info("── Camera pair %d/%d: %s ↔ %s (zone: %s) ──",
                        pair_idx + 1, len(active_pairs), ca, cb, zone["id"])
            _flush_log()

            # Build per-camera zone index arrays (in combined_df positions)
            ca_zone_mask = in_any_zone & (cam_arr == ca)
            cb_zone_mask = in_any_zone & (cam_arr == cb)
            ca_idx = np.where(ca_zone_mask)[0]   # positions in combined_df
            cb_idx = np.where(cb_zone_mask)[0]

            if len(ca_idx) == 0 or len(cb_idx) == 0:
                logger.info("  Skipping — no zone events for one or both cameras.")
                continue

            logger.info("  %s: %d zone events | %s: %d zone events",
                        ca, len(ca_idx), cb, len(cb_idx))
            _flush_log()

            # Pre-extract numeric arrays (no pandas overhead in hot loop)
            ts_a   = ts_ns[ca_idx];   ts_b   = ts_ns[cb_idx]
            cx_a   = cx_arr[ca_idx];  cy_a   = cy_arr[ca_idx]
            cx_b   = cx_arr[cb_idx];  cy_b   = cy_arr[cb_idx]
            tid_a  = track_arr[ca_idx].astype("int64")
            tid_b  = track_arr[cb_idx].astype("int64")

            # Matched flags (indexed in local ca/cb arrays)
            a_matched = np.zeros(len(ca_idx), dtype=bool)
            b_matched = np.zeros(len(cb_idx), dtype=bool)

            # Trajectory identity map for this zone/pair
            tmap = (identity_maps.get(zone["id"], {}).get((ca, cb))
                    or identity_maps.get(zone["id"], {}).get((cb, ca))
                    or {})

            base_thresh = (self.distance_threshold_override_m
                           if self.distance_threshold_override_m is not None
                           else zone.get("distance_threshold_m", 1.0))
            dyn_thresh = self._dynamic_threshold(base_thresh, ca, cb)

            _log_every_a = max(len(ca_idx) // 20, 10_000)
            n_pair_fused = 0

            for i_a in range(len(ca_idx)):
                if i_a % _log_every_a == 0:
                    logger.info("  Progress %s→%s: %d / %d  (%.0f%%)  fused so far: %d",
                                ca, cb, i_a, len(ca_idx),
                                i_a * 100.0 / len(ca_idx), n_pair_fused)
                    _flush_log()

                if a_matched[i_a]:
                    continue

                t_a = ts_a[i_a]
                # Symmetric window: [t_a - tol/2, t_a + tol/2] catches both
                # earlier and later events from the other camera
                half_tol = tol_ns // 2
                b_lo = int(np.searchsorted(ts_b, t_a - half_tol, side="left"))
                b_hi = int(np.searchsorted(ts_b, t_a + half_tol, side="right"))

                if b_lo >= b_hi:
                    continue

                # Vectorised distances to window
                dx    = cx_b[b_lo:b_hi] - cx_a[i_a]
                dy    = cy_b[b_lo:b_hi] - cy_a[i_a]
                dists = (dx * dx + dy * dy) ** 0.5

                # Mask already-matched B events
                b_window_matched = b_matched[b_lo:b_hi]
                dists[b_window_matched] = np.inf

                if np.all(np.isinf(dists)):
                    continue

                # ── 1. Trajectory-confirmed match (strongest signal) ──────
                partner_tid = tmap.get(int(tid_a[i_a]))
                best_j = -1
                best_dist = np.inf
                if partner_tid is not None:
                    for k_off in range(b_hi - b_lo):
                        if b_matched[b_lo + k_off]:
                            continue
                        if int(tid_b[b_lo + k_off]) == partner_tid:
                            best_j = k_off
                            best_dist = 0.0   # trajectory match overrides distance
                            break

                # ── 2. Spatial fallback (nearest within threshold) ────────
                if best_j < 0:
                    min_j = int(np.argmin(dists))
                    if dists[min_j] <= dyn_thresh:
                        best_j    = min_j
                        best_dist = float(dists[min_j])

                if best_j < 0:
                    continue   # no match found for this ca event

                # ── Record the fused pair ─────────────────────────────────
                b_abs = b_lo + best_j
                a_matched[i_a]   = True
                b_matched[b_abs] = True
                matched_indices.add(int(ca_idx[i_a]))
                matched_indices.add(int(cb_idx[b_abs]))

                gpid = _link_gpid(ca, int(tid_a[i_a]), cb, int(tid_b[b_abs]))

                fused_x, fused_y = self._weighted_fuse_xy(
                    float(cx_a[i_a]), float(cy_a[i_a]), ca,
                    float(cx_b[b_abs]), float(cy_b[b_abs]), cb,
                )

                # Fetch row metadata only for the final record (not in hot loop)
                row_a = combined_df.iloc[int(ca_idx[i_a])]
                row_b = combined_df.iloc[int(cb_idx[b_abs])]
                fused_records.append({
                    "timestamp":        min(row_a["timestamp"], row_b["timestamp"]),
                    "global_person_id": gpid,
                    "track_id":         int(tid_a[i_a]),
                    "class_name":       row_a["class_name"],
                    "edge_id":          _random.choice([row_a["edge_id"],
                                                        row_b["edge_id"]]),
                    "direction":        row_a["direction"],
                    "crossing_x":       round(fused_x, 4),
                    "crossing_y":       round(fused_y, 4),
                    "camera_id":        f"fused:{ca}+{cb}",
                })
                n_pair_fused += 1

            logger.info("  Pair %s↔%s done: %d events fused.", ca, cb, n_pair_fused)
            _flush_log()

        # ── Assign global person IDs to every unmatched event (vectorised) ──
        logger.info("Assigning global person IDs to unmatched events …")
        _flush_log()

        unmatched_mask = np.ones(n, dtype=bool)
        for idx in matched_indices:
            unmatched_mask[idx] = False

        unmatched_df = combined_df[unmatched_mask].copy()
        unmatched_df["global_person_id"] = [
            _get_or_create_gpid(str(cam), int(tid))
            for cam, tid in zip(
                unmatched_df["camera_id"].tolist(),
                unmatched_df["track_id"].tolist(),
            )
        ]
        fused_records.extend(unmatched_df.to_dict("records"))

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
