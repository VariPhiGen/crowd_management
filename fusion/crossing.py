"""
fusion/crossing.py
==================
Virtual-edge line-crossing detector for floor-space tracks.

Each *edge* is a named axis-aligned line on the factory floor.  When a
tracked person's smoothed position moves from one side of an edge to the
other between consecutive frames, a **crossing event** is recorded.

Output record columns (comma-separated):

    timestamp,track_id,class_name,edge_id,direction,crossing_x,crossing_y,camera_id

Example row::

    1970-01-01T00:00:00.000,1347,person,x_3.0,+x,3.00,23.11,cam01

Edge definition (from ``config/edges.json``)::

    {
      "edges": [
        {"id": "x_3.0",  "type": "vertical",   "value": 3.0},
        {"id": "y_10.0", "type": "horizontal",  "value": 10.0}
      ]
    }

*vertical*   edges are lines where ``floor_x == value``.  A person moving
in the +X direction (left → right) produces direction ``"+x"``.

*horizontal* edges are lines where ``floor_y == value``.  A person moving
in the +Y direction (bottom → top) produces direction ``"+y"``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Optional

logger = logging.getLogger(__name__)

# Default edge config path (relative to project root, resolved at call site)
_DEFAULT_EDGES_CFG = Path(__file__).parent.parent / "config" / "edges.json"


# ═══════════════════════════════════════════════════════════════════════════
#  Public helpers
# ═══════════════════════════════════════════════════════════════════════════

def generate_edges(
    floor_width_m: float,
    floor_height_m: float,
    step_m: float = 1.0,
    save_path: str | Path | None = None,
) -> list[dict]:
    """
    Dynamically generate a uniform grid of virtual edges.

    Creates one *vertical* edge (``floor_x == n * step_m``) for every integer
    multiple of *step_m* that lies strictly inside [0, floor_width_m], and one
    *horizontal* edge (``floor_y == n * step_m``) for every multiple that lies
    strictly inside [0, floor_height_m].

    Parameters
    ----------
    floor_width_m  : float — X extent of the floor (metres)
    floor_height_m : float — Y extent of the floor (metres)
    step_m         : float — grid spacing (default 1.0 m)
    save_path      : str | Path | None
        When provided the generated edges are written back to this JSON file
        so the user can inspect or manually edit them.

    Returns
    -------
    list[dict]  — same format as :func:`load_edges`
    """
    import math

    edges: list[dict] = []

    # Vertical lines: x = step_m, 2*step_m, …  (stop before the right wall)
    n_x = math.floor(floor_width_m / step_m)
    for i in range(1, n_x + 1):
        val = round(i * step_m, 6)
        if val >= floor_width_m:
            break
        edges.append({
            "id":    f"x_{val:g}",
            "type":  "vertical",
            "value": val,
        })

    # Horizontal lines: y = step_m, 2*step_m, …  (stop before the top wall)
    n_y = math.floor(floor_height_m / step_m)
    for i in range(1, n_y + 1):
        val = round(i * step_m, 6)
        if val >= floor_height_m:
            break
        edges.append({
            "id":    f"y_{val:g}",
            "type":  "horizontal",
            "value": val,
        })

    logger.info(
        "Generated %d virtual edge(s)  (%.1f m × %.1f m floor, %.1f m step)",
        len(edges), floor_width_m, floor_height_m, step_m,
    )

    if save_path is not None:
        _write_edges(edges, Path(save_path), floor_width_m, floor_height_m, step_m)

    return edges


def _write_edges(
    edges: list[dict],
    path: Path,
    floor_width_m: float,
    floor_height_m: float,
    step_m: float,
) -> None:
    """Persist *edges* to a JSON file (overwrites existing content)."""
    payload = {
        "_comment": [
            "Auto-generated at startup from floor_config.json.",
            f"Floor: {floor_width_m} m (X) × {floor_height_m} m (Y),  step: {step_m} m.",
            "Edit values or add custom edges, then remove the _auto flag to pin this file.",
        ],
        "_auto":  True,
        "step_m": step_m,
        "edges":  edges,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Edges config saved → %s  (%d edges)", path, len(edges))
    except Exception as exc:
        logger.warning("Could not save edges config to %s: %s", path, exc)


def load_edges(path: str | Path | None = None) -> list[dict]:
    """
    Load edge definitions from JSON.

    Falls back to ``config/edges.json`` when *path* is ``None``.
    Returns an empty list (no crossings logged) if the file is missing.
    """
    cfg_path = Path(path) if path else _DEFAULT_EDGES_CFG
    if not cfg_path.exists():
        logger.warning("Edges config not found at %s — no crossing lines defined.", cfg_path)
        return []
    try:
        with open(cfg_path) as fh:
            data = json.load(fh)
        edges = data.get("edges", [])
        logger.info("Loaded %d virtual edge(s) from %s", len(edges), cfg_path)
        return edges
    except Exception as exc:
        logger.error("Failed to load edges config %s: %s", cfg_path, exc)
        return []


def load_crossings_csv(filepath: str):
    """
    Reads a line crossings CSV back into a pandas DataFrame, parsing the timestamps.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['timestamp', 'track_id', 'class_name', 'edge_id', 
                                 'direction', 'crossing_x', 'crossing_y', 'camera_id']
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required to use load_crossings_csv. Please install pandas.")
        raise
        
    df = pd.read_csv(filepath)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  LineCrossingDetector
# ═══════════════════════════════════════════════════════════════════════════

class LineCrossingDetector:
    """
    Stateful crossing detector writing to a CSV file.

    Call :meth:`update` once per frame with the latest tracked positions. The method 
    checks for edge crossings and logs them to the CSV.

    Parameters
    ----------
    edges_config_path : str
        Path to the edges.json config file.
    camera_id : str
        Camera source ID for the crossings.
    output_dir : str
        Directory to write the CSV file.
    """

    def __init__(
        self,
        edges_config_path: str,
        camera_id: str,
        output_dir: str,
        append: bool = False,
    ) -> None:
        self._edges: list[dict] = load_edges(edges_config_path)
        self.camera_id = camera_id
        
        # Stores the floor position from the previous frame for each track_id.
        self._prev: dict[int, tuple[float, float]] = {}
        
        # Setup CSV writing: append to existing file (e.g. multiple runs) or create new
        os.makedirs(output_dir, exist_ok=True)
        self.csv_path = os.path.join(output_dir, f"{camera_id}_crossings.csv")
        file_exists = os.path.exists(self.csv_path)
        mode = "a" if (append and file_exists) else "w"
        self._csv_file = open(self.csv_path, mode, newline="", buffering=1)
        self._csv_writer = csv.writer(self._csv_file)
        if mode == "w":
            self._csv_writer.writerow([
                "timestamp", "track_id", "class_name", "edge_id",
                "direction", "crossing_x", "crossing_y", "camera_id"
            ])
        
        logger.info(
            "[%s] Opened crossings CSV log: %s (mode=%s)",
            self.camera_id, self.csv_path, mode,
        )

    def close(self) -> None:
        """
        Flush and close the CSV file.
        """
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()

    def update(
        self,
        track_id: int,
        class_name: str,
        floor_x: float,
        floor_y: float,
        timestamp: datetime,
    ) -> List[dict]:
        """
        Process a single target's position update.

        Parameters
        ----------
        track_id : int
            The unique tracking ID for this object.
        class_name : str
            The class label (e.g., 'person', 'forklift').
        floor_x : float
            Current X coordinate on the floor (metres).
        floor_y : float
            Current Y coordinate on the floor (metres).
        timestamp : datetime
            The OCR-extracted timestamp for this frame.

        Returns
        -------
        list[dict]
            One dict per crossing event triggered by this update.
        """
        events: list[dict] = []

        if track_id < 0:
            return events

        cx, cy = floor_x, floor_y

        if track_id in self._prev:
            px, py = self._prev[track_id]
            for edge in self._edges:
                evt = self._check_crossing(
                    edge, px, py, cx, cy, timestamp, track_id, class_name
                )
                if evt is not None:
                    events.append(evt)
                    logger.debug(
                        "Crossing [%s]: track=%d edge=%s dir=%s @ (%.2f, %.2f) t=%s",
                        self.camera_id, track_id, evt['edge_id'], evt['direction'],
                        evt['crossing_x'], evt['crossing_y'], timestamp.isoformat(),
                    )
                    
                    # Write immediately to line-buffered CSV
                    self._csv_writer.writerow([
                        evt["timestamp"],
                        evt["track_id"],
                        evt["class_name"],
                        evt["edge_id"],
                        evt["direction"],
                        f"{evt['crossing_x']:.2f}",
                        f"{evt['crossing_y']:.2f}",
                        evt["camera_id"]
                    ])

        # Track management logic inside detector needs to be cleaned up periodically
        # using the reset() method or bounded state to avoid memory leaks.
        self._prev[track_id] = (cx, cy)

        return events

    def reset(self) -> None:
        """Clear all previous positions (call on track loss or video reset)."""
        self._prev.clear()

    # ──────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    _TS_FMT = "%Y-%m-%dT%H:%M:%S.%f"

    def _fmt_ts(self, dt: datetime) -> str:
        """Always emit ISO-8601 with millisecond precision (consistent across all rows)."""
        return dt.strftime(self._TS_FMT)[:-3]

    def _check_crossing(
        self,
        edge: dict,
        px: float, py: float,
        cx: float, cy: float,
        timestamp: datetime,
        track_id: int,
        class_name: str,
    ) -> Optional[dict]:
        """
        Test whether the line segment (px, py) → (cx, cy) crosses *edge*.

        Returns a crossing-event dict or ``None``.
        """
        etype = edge.get("type", "")
        val   = float(edge["value"])
        eid   = edge["id"]

        if etype == "vertical":
            # Edge: floor_x == val
            if px == cx:
                return None
            if px < val <= cx:
                direction = "+x"
            elif px > val >= cx:
                direction = "-x"
            else:
                return None
            # Interpolate to find the exact crossing Y point
            t = (val - px) / (cx - px)
            cross_x = val
            cross_y = py + t * (cy - py)

        elif etype == "horizontal":
            # Edge: floor_y == val
            if py == cy:
                return None
            if py < val <= cy:
                direction = "+y"
            elif py > val >= cy:
                direction = "-y"
            else:
                return None
            # Interpolate to find the exact crossing X point
            t = (val - py) / (cy - py)
            cross_y = val
            cross_x = px + t * (cx - px)

        else:
            logger.warning("[%s] Unknown edge type %r for edge %r — skipping.", self.camera_id, etype, eid)
            return None

        return {
            "timestamp":    self._fmt_ts(timestamp),
            "track_id":     track_id,
            "class_name":   class_name,
            "edge_id":      eid,
            "direction":    direction,
            "crossing_x":   round(cross_x, 6),
            "crossing_y":   round(cross_y, 6),
            "camera_id":    self.camera_id,
        }
