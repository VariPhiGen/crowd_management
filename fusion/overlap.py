"""
overlap.py — Overlap Zone Management (Shapely point-in-polygon)

Loads overlap zone definitions from config/overlap_zones.json and provides:

  • OverlapZone          — single zone with fast contains_point() test
  • load_overlap_zones() — parse JSON → list[OverlapZone]
  • point_in_any_overlap() — which zones contain (x, y)?
  • get_overlap_zone_for_cameras() — zone shared by two specific cameras

Overlap zones are defined as floor-space polygons in metres.
Shapely is used for all polygon operations; a precomputed axis-aligned
bounding box provides a fast-rejection guard before the full point-in-
polygon test.

Performance note
----------------
Bounding-box rejection eliminates most "clearly outside" queries in O(1).
Full Shapely point-in-polygon runs only when the point is inside the bbox,
which is the slow (but accurate) path.  For typical factory layouts with
a handful of zones this is negligible even at 30 fps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

_DEFAULT_OVERLAP_CFG = Path(__file__).parent.parent / "config" / "overlap_zones.json"
_DEFAULT_CAMERAS_CFG = Path(__file__).parent.parent / "config" / "cameras.json"


# ═══════════════════════════════════════════════════════════════════════════
#  OverlapZone
# ═══════════════════════════════════════════════════════════════════════════

class OverlapZone:
    """
    One overlap region between two or more cameras on the factory floor.

    All spatial quantities are in **metres** on the floor plane.

    Parameters
    ----------
    zone_config : dict
        One entry from the ``"overlap_zones"`` list in ``overlap_zones.json``.

    Attributes
    ----------
    id                  : str
    camera_ids          : list[str]
    polygon             : shapely.Polygon   (exact boundary)
    buffered_polygon    : shapely.Polygon   (expanded by buffer_margin)
    distance_threshold  : float             (m)  max distance to merge detections
    buffer_margin       : float             (m)  soft expansion applied to polygon
    fusion_strategy     : str               e.g. "weighted_average"

    Private
    -------
    _bbox : (minx, miny, maxx, maxy)  precomputed for fast rejection
    _bbuf : (minx, miny, maxx, maxy)  precomputed bbox of buffered polygon
    """

    def __init__(self, zone_config: dict) -> None:
        self.id                 = zone_config["id"]
        self.camera_ids: list[str] = zone_config["cameras"]
        self.polygon            = Polygon(zone_config["floor_polygon"])
        self.distance_threshold = float(zone_config.get("distance_threshold_m", 1.5))
        self.buffer_margin      = float(zone_config.get("buffer_margin_m", 0.5))
        self.fusion_strategy    = zone_config.get("fusion_strategy", "weighted_average")

        # Pre-buffer so contains_point(use_buffer=True) is O(1) polygon check
        self.buffered_polygon   = self.polygon.buffer(self.buffer_margin)

        # Precompute axis-aligned bounding boxes for fast rejection
        #   self._bbox  → tight bbox of the exact polygon
        #   self._bbuf  → tight bbox of the buffered polygon
        self._bbox: tuple[float, float, float, float] = self.polygon.bounds
        self._bbuf: tuple[float, float, float, float] = self.buffered_polygon.bounds

    # ──────────────────────────────────────────────────────────────────────
    #  Spatial tests
    # ──────────────────────────────────────────────────────────────────────

    def contains_point(self, x: float, y: float, use_buffer: bool = True) -> bool:
        """
        Test whether floor position *(x, y)* lies inside this zone.

        Two-stage check for efficiency:

        1. **Bounding-box guard** — axis-aligned rectangle test (O(1)).
           Returns ``False`` immediately if the point is clearly outside.
        2. **Shapely point-in-polygon** — exact test on the selected polygon.

        Parameters
        ----------
        x, y       : float  — floor coordinates in metres
        use_buffer : bool   — if True, test against the polygon expanded by
                              ``buffer_margin``; otherwise use the exact polygon.

        Returns
        -------
        bool
        """
        minx, miny, maxx, maxy = self._bbuf if use_buffer else self._bbox

        # Stage 1: fast bbox rejection
        if x < minx or x > maxx or y < miny or y > maxy:
            return False

        # Stage 2: exact Shapely test
        poly = self.buffered_polygon if use_buffer else self.polygon
        return poly.contains(Point(x, y))

    def area_m2(self, buffered: bool = False) -> float:
        """Return the area of the (optionally buffered) polygon in m²."""
        return self.buffered_polygon.area if buffered else self.polygon.area

    # ──────────────────────────────────────────────────────────────────────
    #  Convenience
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cams = ", ".join(self.camera_ids)
        return (
            f"OverlapZone(id={self.id!r}, cameras=[{cams}], "
            f"area={self.area_m2():.1f} m², "
            f"thresh={self.distance_threshold} m, "
            f"buf={self.buffer_margin} m)"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level functions
# ═══════════════════════════════════════════════════════════════════════════

def load_overlap_zones(
    config_path: str | Path = _DEFAULT_OVERLAP_CFG,
) -> list[OverlapZone]:
    """
    Parse ``overlap_zones.json`` and return one ``OverlapZone`` per entry.

    Parameters
    ----------
    config_path : str | Path
        Path to ``overlap_zones.json``.  Defaults to
        ``config/overlap_zones.json`` relative to the project root.

    Returns
    -------
    list[OverlapZone]
        May be empty if the file contains no zones or does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning("overlap_zones.json not found at %s", path)
        return []

    with open(path) as f:
        data = json.load(f)

    zones: list[OverlapZone] = []
    for zone_cfg in data.get("overlap_zones", []):
        try:
            zones.append(OverlapZone(zone_cfg))
        except Exception as exc:
            logger.error("Failed to parse zone config %s: %s", zone_cfg, exc)

    logger.info(
        "Loaded %d overlap zone(s) from %s",
        len(zones), path,
    )
    return zones


def point_in_any_overlap(
    x: float,
    y: float,
    zones: list[OverlapZone],
    use_buffer: bool = True,
) -> list[str]:
    """
    Return the IDs of every overlap zone that contains floor point *(x, y)*.

    Iterates over *zones* and calls ``zone.contains_point()`` for each one.
    The bounding-box fast-rejection inside ``contains_point`` keeps this
    cheap even for tens of zones.

    Parameters
    ----------
    x, y       : float            — floor coordinates in metres
    zones      : list[OverlapZone]
    use_buffer : bool             — propagated to ``contains_point``

    Returns
    -------
    list[str]
        Zone IDs that contain the point (empty if none).

    Examples
    --------
    >>> zones = load_overlap_zones()
    >>> ids = point_in_any_overlap(18.0, 7.5, zones)
    >>> # ['overlap_cam1_cam2']  if the point is in that zone
    """
    return [
        zone.id
        for zone in zones
        if zone.contains_point(x, y, use_buffer=use_buffer)
    ]


def get_overlap_zone_for_cameras(
    zones: list[OverlapZone],
    cam_a: str,
    cam_b: str,
) -> Optional[OverlapZone]:
    """
    Return the first overlap zone whose camera list includes both *cam_a*
    and *cam_b*.

    Parameters
    ----------
    zones : list[OverlapZone]
    cam_a : str
    cam_b : str

    Returns
    -------
    OverlapZone or None
        ``None`` if no zone covers both cameras.
    """
    for zone in zones:
        if cam_a in zone.camera_ids and cam_b in zone.camera_ids:
            return zone
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  OverlapManager  — higher-level convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════

class OverlapManager:
    """
    Manages all overlap zones for the current deployment.

    Thin wrapper around ``load_overlap_zones`` and the three module-level
    helper functions; provides a single object to pass around in the
    pipeline instead of a raw list.

    Parameters
    ----------
    config_path : str | Path
        Path to ``overlap_zones.json``.
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_OVERLAP_CFG,
    ) -> None:
        self.config_path = Path(config_path)
        self.zones: list[OverlapZone] = load_overlap_zones(self.config_path)

    # ──────────────────────────────────────────────────────────────────────
    #  Query API
    # ──────────────────────────────────────────────────────────────────────

    def zones_for_point(
        self,
        x: float,
        y: float,
        use_buffer: bool = True,
    ) -> list[OverlapZone]:
        """Return all zones that contain floor point *(x, y)*."""
        return [
            zone for zone in self.zones
            if zone.contains_point(x, y, use_buffer=use_buffer)
        ]

    def zone_ids_for_point(
        self,
        x: float,
        y: float,
        use_buffer: bool = True,
    ) -> list[str]:
        """Return zone IDs that contain floor point *(x, y)*."""
        return point_in_any_overlap(x, y, self.zones, use_buffer=use_buffer)

    def is_in_overlap(self, x: float, y: float, use_buffer: bool = True) -> bool:
        """Return True if *(x, y)* is inside any overlap zone."""
        return bool(self.zone_ids_for_point(x, y, use_buffer=use_buffer))

    def zone_for_cameras(
        self,
        cam_a: str,
        cam_b: str,
    ) -> Optional[OverlapZone]:
        """Return the overlap zone shared by *cam_a* and *cam_b*, or None."""
        return get_overlap_zone_for_cameras(self.zones, cam_a, cam_b)

    def zones_for_cameras(
        self,
        cam_a: str,
        cam_b: str,
    ) -> list[OverlapZone]:
        """Return *all* overlap zones shared by *cam_a* and *cam_b*."""
        return [
            z for z in self.zones
            if cam_a in z.camera_ids and cam_b in z.camera_ids
        ]

    def get_zone(self, zone_id: str) -> Optional[OverlapZone]:
        """Look up a single zone by its ID string."""
        return next((z for z in self.zones if z.id == zone_id), None)

    # ──────────────────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a human-readable summary of all loaded zones."""
        print(f"\nOverlap Zones  ({len(self.zones)} total)  [{self.config_path.name}]")
        print("─" * 56)
        for z in self.zones:
            cams   = " ↔ ".join(z.camera_ids)
            coords = list(z.polygon.exterior.coords)
            print(f"  [{z.id}]")
            print(f"    Cameras         : {cams}")
            print(f"    Exact area      : {z.area_m2():.1f} m²")
            print(f"    Buffered area   : {z.area_m2(buffered=True):.1f} m²")
            print(f"    Dist threshold  : {z.distance_threshold} m")
            print(f"    Buffer margin   : {z.buffer_margin} m")
            print(f"    Fusion strategy : {z.fusion_strategy}")
            print(f"    Polygon         : {coords}")
        if not self.zones:
            print("  (none)")
        print()


# ═══════════════════════════════════════════════════════════════════════════
#  Camera coverage utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_camera_polygons(
    cameras_config_path: str | Path = _DEFAULT_CAMERAS_CFG,
) -> dict[str, Polygon]:
    """
    Load each camera's floor-coverage polygon as a Shapely Polygon.

    Returns
    -------
    dict[str, Polygon]
        ``{camera_id: Polygon}``
    """
    path = Path(cameras_config_path)
    with open(path) as f:
        config = json.load(f)

    polys: dict[str, Polygon] = {}
    for cam in config["cameras"]:
        coords = cam.get("floor_coverage_polygon", [])
        if coords:
            polys[cam["id"]] = Polygon(coords)
    return polys


def compute_overlap_area(poly_a: Polygon, poly_b: Polygon) -> float:
    """Return the intersection area (m²) of two camera-coverage polygons."""
    return poly_a.intersection(poly_b).area
