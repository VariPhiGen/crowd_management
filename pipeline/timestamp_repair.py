"""
timestamp_repair.py — Pre-fusion timestamp sanitiser

Reads ``expected_date_range`` from fusion_config.json::

    "expected_date_range": {
        "start": "2026-02-23",
        "end":   "2026-03-01"
    }

Before fusion it scans every selected CSV and checks whether any
timestamps fall outside that range.  If they do, it repairs in-place
(year → expected year, month/day checked against range) and prints a
summary table with start/end dates per file.

If all timestamps are already within range → prints
"No repair needed" and skips the rewrite.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_TS_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(T\d{2}:\d{2}:\d{2}(?:\.\d+)?)")


def _load_date_range(config_dir: str):
    """Return (expected_year, start_date, end_date) from fusion_config.json."""
    cfg_path = os.path.join(config_dir, "fusion_config.json")
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        dr = cfg.get("expected_date_range", {})
        start_str = dr.get("start", "")
        end_str = dr.get("end", "")
    except Exception:
        start_str = ""
        end_str = ""

    if not start_str or not end_str:
        return None, None, None

    start_d = datetime.strptime(start_str, "%Y-%m-%d").date()
    end_d = datetime.strptime(end_str, "%Y-%m-%d").date()
    expected_year = str(start_d.year)
    return expected_year, start_d, end_d


def _fix_date(yr: str, mo: str, dy: str, expected_year: str,
              start_d: date, end_d: date) -> tuple[str, str, str, bool]:
    """Fix a single YYYY-MM-DD.  Returns (yr, mo, dy, changed)."""
    changed = False

    if yr != expected_year:
        yr = expected_year
        changed = True

    # Build candidate date and check if it falls within range
    try:
        candidate = date(int(yr), int(mo), int(dy))
    except ValueError:
        # Invalid date (e.g. Feb 30) — clamp month
        mo = f"{start_d.month:02d}"
        changed = True
        return yr, mo, dy, changed

    if start_d <= candidate <= end_d:
        return yr, mo, dy, changed

    # Date is out of range — try cycling the month
    for m_offset in range(1, 12):
        new_month = ((int(mo) - 1 + m_offset) % 12) + 1
        try:
            fixed = date(int(yr), new_month, int(dy))
        except ValueError:
            continue
        if start_d <= fixed <= end_d:
            mo = f"{new_month:02d}"
            changed = True
            return yr, mo, dy, changed

    # Nothing fit — clamp to the nearest bound
    if candidate < start_d:
        mo = f"{start_d.month:02d}"
        dy = f"{start_d.day:02d}"
    else:
        mo = f"{end_d.month:02d}"
        dy = f"{end_d.day:02d}"
    changed = True
    return yr, mo, dy, changed


def _scan_file(path: str, expected_year: str, start_d: date, end_d: date) -> int:
    """Quick scan — count rows outside range. Returns 0 if clean."""
    bad = 0
    with open(path, "r") as f:
        for line in f:
            m = _TS_RE.match(line)
            if not m:
                continue
            try:
                d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                return 1
            if d < start_d or d > end_d:
                return 1
    return bad


def _repair_file(path: str, expected_year: str,
                 start_d: date, end_d: date) -> dict:
    """Repair a single CSV in-place. Returns stats dict."""
    with open(path, "r") as f:
        lines = f.readlines()

    total = 0
    fixed = 0
    first_ts: str | None = None
    last_ts: str | None = None

    for i, line in enumerate(lines):
        m = _TS_RE.match(line)
        if not m:
            continue
        total += 1

        yr, mo, dy, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        yr, mo, dy, did_change = _fix_date(yr, mo, dy, expected_year, start_d, end_d)

        if did_change:
            end_pos = m.end(4)
            lines[i] = f"{yr}-{mo}-{dy}{rest}{line[end_pos:]}"
            fixed += 1

        ts_str = f"{yr}-{mo}-{dy}{rest}"
        if first_ts is None:
            first_ts = ts_str
        last_ts = ts_str

    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(lines)
    os.replace(tmp, path)

    return {
        "total": total,
        "fixed": fixed,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def _date_range_for_file(path: str) -> tuple[str | None, str | None]:
    """Quick first/last timestamp scan (no repair)."""
    first: str | None = None
    last: str | None = None
    with open(path, "r") as f:
        for line in f:
            m = _TS_RE.match(line)
            if m:
                ts = f"{m.group(1)}-{m.group(2)}-{m.group(3)}{m.group(4)}"
                if first is None:
                    first = ts
                last = ts
    return first, last


def repair_before_fusion(
    csv_files: Sequence[str],
    output_dir: str | None = None,
    config_dir: str | None = None,
) -> None:
    """Check and optionally repair crossing + track CSVs before fusion.

    Reads ``expected_date_range`` from fusion_config.json.  If any CSV
    has timestamps outside that range, repairs all selected files
    in-place and prints a summary.  Otherwise prints "No repair needed".
    """
    if not csv_files:
        return

    if output_dir is None:
        output_dir = str(Path(csv_files[0]).parent)
    if config_dir is None:
        config_dir = str(Path(csv_files[0]).parent.parent / "config")

    expected_year, start_d, end_d = _load_date_range(config_dir)
    if expected_year is None:
        print("\n[Timestamp Repair] No expected_date_range in fusion_config.json — skipping.\n")
        return

    # Collect all files (crossings + tracks)
    all_files: list[str] = []
    for cf in csv_files:
        all_files.append(cf)
        track_f = cf.replace("_crossings.csv", "_tracks.csv")
        if os.path.isfile(track_f):
            all_files.append(track_f)

    print(f"\n[Timestamp Repair] Expected range: {start_d} → {end_d}")
    print(f"[Timestamp Repair] Scanning {len(all_files)} files …")

    # Quick scan to decide if repair is needed
    needs_repair = False
    for fpath in all_files:
        if _scan_file(fpath, expected_year, start_d, end_d) > 0:
            needs_repair = True
            break

    if not needs_repair:
        print(f"\n  ✓  All timestamps within expected range — no repair needed.\n")
        print(f"  {'File':<35s}  {'Start':>24s}  {'End':>24s}")
        print(f"  {'-'*85}")
        for fpath in all_files:
            first, last = _date_range_for_file(fpath)
            print(f"  {os.path.basename(fpath):<35s}  {first or 'N/A':>24s}  {last or 'N/A':>24s}")
        print()
        return

    # Repair needed
    print(f"  ⚠  Timestamps outside range detected — repairing …\n")
    logger.info("Timestamp repair: processing %d files", len(all_files))

    results: dict[str, dict] = {}
    for fpath in all_files:
        stats = _repair_file(fpath, expected_year, start_d, end_d)
        results[os.path.basename(fpath)] = stats
        if stats["fixed"]:
            logger.info(
                "  %s: %d fixed (of %d rows)",
                os.path.basename(fpath), stats["fixed"], stats["total"],
            )

    # Summary table
    print(
        f"  {'File':<35s} {'Rows':>10s}  {'Fixed':>10s}  "
        f"{'Start':>24s}  {'End':>24s}"
    )
    print(f"  {'-'*110}")
    total_fixed = 0
    for name, s in sorted(results.items()):
        total_fixed += s["fixed"]
        print(
            f"  {name:<35s} {s['total']:>10,}  {s['fixed']:>10,}  "
            f"{s['first_ts'] or 'N/A':>24s}  {s['last_ts'] or 'N/A':>24s}"
        )

    if total_fixed == 0:
        print(f"\n  ✓  All timestamps already clean — no repairs needed.")
    else:
        print(f"\n  ✓  Repaired {total_fixed:,} timestamps across {len(all_files)} files.")
    print()
