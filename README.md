# factory-floor-viz

Real-time multi-camera factory floor visualisation — detects people across
overlapping camera feeds and renders a unified top-down bird's-eye view.

---

## Architecture

```text
┌───────────────────────────────────────────────────────────────────────┐
│                          factory-floor-viz                            │
│                                                                       │
│  ┌──────────┐   ┌────────────────────────────────────────────────┐    │
│  │  config/ │   │  Calibration (One-time setup)                  │    │
│  │  *.json  │──▶│  lens_correction → homography → ocr_region     │    │
│  └──────────┘   └───────────────────┬────────────────────────────┘    │
│                                     │ (cameras.json, edges.json)      │
│                 ┌───────────────────▼────────────────────────────┐    │
│  Camera Feeds   │  Per-Camera Processing (detection/ & pipeline/)│    │
│  (Video/RTSP) ──▶  YOLOv8 Detection + OCR Timestamp Extractor    │    │
│                 │          ↓ Homography Projection ↓             │    │
│                 │  Line Crossing Detector → Per-Camera CSV       │    │
│                 └───────────────────┬────────────────────────────┘    │
│                                     │ {cam_id}_crossings.csv          │
│                 ┌───────────────────▼────────────────────────────┐    │
│                 │  Multi-Camera Fusion (fusion/multi_camera...)  │    │
│                 │  Hungarian Matching + Overlap Deduplication    │    │
│                 └───────────────────┬────────────────────────────┘    │
│                                     │ fused_crossings.csv             │
│                 ┌───────────────────▼────────────────────────────┐    │
│                 │  Visualisation (visualization/)                │    │
│                 │  floor_renderer.py — OpenCV top-down canvas    │    │
│                 └────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────┘
```

### Data flow (Pipeline)

```text
Camera Feeds (Video/RTSP)
          │
          ▼
   [Lens Undistortion]     ← correct warp using camera intrinsics
          │
          ▼
 [YOLOv8 + OCR Reader]     ← bounding box (foot point) + datetime extraction
          │
          ▼
 [Homography Projector]    ← pixel_x, pixel_y → floor_x, floor_y (metres)
          │
          ▼
 [Line Crossing Detector]  ← cross-product vector matching over edges.json
          │
          ▼
    [Per-Camera CSV]       ← one tracking file per individual camera
          │
          ▼
 [Multi-Camera Fuser]      ← Hungarian temporal/spatial deduplication
          │
          ▼
    [Fused Output CSV]     ← unified multi-camera crossover events
```

---

## Project structure

```
factory-floor-viz/
├── config/
│   ├── floor_config.json       Floor dimensions and grid settings
│   ├── cameras.json            Camera sources, intrinsics, homographies
│   └── overlap_zones.json      Cross-camera overlap region definitions
│
├── calibration/
│   ├── lens_correction.py      Chessboard intrinsic calibration (--intrinsic)
│   ├── homography.py           Perspective homography computation & mapping
│   └── calibrate.py            Interactive floor-point calibration (--calibrate)
│
├── detection/
│   └── detector.py             YOLOv8 wrapper → Detection dataclass
│
├── fusion/
│   ├── overlap.py              OverlapZone geometry (Shapely)
│   └── fuse.py                 DetectionFuser — confidence-weighted merge
│
├── visualization/
│   ├── floor_renderer.py       OpenCV top-down floor canvas
│   └── demo_simulator.py       Synthetic multi-agent demo (no cameras needed)
│
├── main.py                     CLI entry point (argparse)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / navigate to the project

```bash
cd factory-floor-viz
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `opencv-contrib-python` includes the extra calibration modules.
> If you already have `opencv-python` installed you may need to uninstall it
> first to avoid conflicts: `pip uninstall opencv-python`.

### 4. YOLOv8 weights

The first time you run any `--phase 3 / 4 / --run` command, Ultralytics will
automatically download `yolov8n.pt` (~6 MB) to `~/.cache/ultralytics/`.
You can also pre-download it:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

### Step-by-step Execution Pipeline

**Step 1 — Lens Calibration (Optional but recommended)**  
Hold a chessboard pattern in front of your camera and run:
```bash
python main.py --intrinsic cam_1
```
Computes and saves intrinsics allowing 3D flattening without barrel distortion.

**Step 2 — Floor-point Homography Calibration**  
Mark ≥4 points of known metric floor coordinates to transform pixel space to floor space:
```bash
python main.py --calibrate cam_1
```

**Step 3 — Define Floor Coverage**  
Define the area the camera can genuinely see by drawing a polygon:
```bash
python main.py --coverage cam_1
```

**Step 4 — Auto Configure Overlaps**  
Let the system compute the `overlap_zones.json` automatically based on the geometry mapping:
```bash
python main.py --auto-config
```

**Step 5 — Select OCR Timestamp Region**  
Draw an ROI over the digital timestamp baked into your security feeds:
```bash
python main.py --ocr-region cam_1
```

**Step 6 — Verify Configurations**  
Run `--phase 1` to assert config validity. Run `--ocr-test <cam>` to preview extraction on 10 frames:
```bash
python main.py --phase 1
python main.py --ocr-test cam_1
```

**Step 7 — Process Data**  
Process all videos through YOLO detection, floor tracking, line crossing, and spatial fusion:
```bash
python main.py --process
```
*(Or sequentially: `--process-camera cam_1` followed by `--fuse-only`)*

**Step 8 — Visualize and Review Output**  
Inspect `output/fused_crossings.csv` inside your project directory for all unified crowd movement metrics!  

To play back this generated data directly onto the virtual 2D floor map without re-running any models, use the visualization tool:
```bash
python main.py --visualize output/fused_crossings.csv --playback-speed 2.0
```

---

## 🌐 Web UI Dashboard (New)
You can avoid the CLI entirely by using our built-in web dashboard. This allows you to calibrate cameras, configure overlap zones, and playback generated visualization `.mp4` files seamlessly in your browser (compatible with AWS hosting).

```bash
# Start the web interface
python3 web_ui/app.py
```
Then navigate to `http://localhost:5001` and login with `admin` / `password123`.

---

## Output CSV Format
The `--process` mode logs to CSV defining 8 strict columns:

| Column | Example | Description |
|--------|---------|-------------|
| timestamp | `2025-03-14 10:32:15` | UTC or Localized DateTime Extracted By OCR |
| track_id | `12` | The YOLO internal tracker identity |
| class_name | `person` | Detected bounding-box category |
| edge_id | `x_5.0` | The `edges.json` physical floor threshold crossed |
| direction | `forward` | Relative crossover direction (forward/backward) |
| crossing_x | `12.55` | Floor X-axis position (in Metres) |
| crossing_y | `5.00` | Floor Y-axis position (in Metres) |
| camera_id | `fused:cam1+cam2` | Original camera ID, or hybrid when duplicate counts merge |

---

## CLI Reference

### Calibration
- `--intrinsic <cam>`: Lens distortion calibration.
- `--calibrate <cam>`: Homography mapping.
- `--coverage <cam>`: Define floor-viewable polygon.
- `--ocr-region <cam>`: Define OCR timestamp block.
- `--auto-config`: Calculate intersection overlap zones.

### Processing & Review
- `--process`: Master. Runs OCR + YOLO + Crossing algorithms on all cameras, then fuses results.
- `--process-camera <cam>`: Generates crossing CSVs solely for the input camera.
- `--fuse-only`: Deduplicate existing CSV records ignoring YOLO models.
- `--visualize <csv_path>`: Play back generated data mapped to the 2D floor renderer.
- `--run`: Legacy live-tracking operation viewing mode.
- `--demo`: Legacy synthetic data generator.

### Testing & Options
- `--phase [1/2/3/4]`: Pipeline debugging verifications.
- `--ocr-test <cam>`: Sample 10 frames of OCR data parsing.
- `--timestamp-tolerance <s>`: Override overlapping fusion tolerances (defaults to 1.0 seconds).
- `--playback-speed <float>`: Multiplier to speed up the `--visualize` offline viewer (default `1.0`).

---

## Camera configuration

Edit `config/cameras.json` to:
- Change RTSP URLs (`source` field)
- Adjust `floor_coverage_polygon` (floor-space coordinates in metres)
- Change camera display colours (`color` array as `[R, G, B]`)

Edit `config/overlap_zones.json` to:
- Add or remove overlap regions
- Tune `distance_threshold_m` (how close two detections must be to merge)
- Change `fusion_strategy` (`"weighted_average"` is the only built-in strategy)

---

## Key dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` + `opencv-contrib-python` | Video capture, calibration, rendering |
| `ultralytics` | YOLOv8 person detection & tracking |
| `numpy` | Numerical operations, homography math |
| `scipy` | Pairwise distance matrix in fusion |
| `shapely` | Overlap zone polygon geometry |
| `matplotlib` | Auxiliary plots (reprojection error visualisation) |

---

## Troubleshooting

**RTSP stream won't open**  
Verify the camera IP/credentials in VLC first. Use `--source test.mp4` to route physical videos during development.

**Poor OCR Accuracy / Timestamps Return None**  
Ensure your `--ocr-region` correctly encapsulates the digital text overlay. Try to avoid dynamic/cluttered video backgrounds inside the crop zone. Text is parsed using pytesseract + generic adaptive thresholding—adjust params in `TimestampExtractor` if using inverted darker backgrounds. Assure fonts are reasonably scaled.

**Deduplication False-Positives (Or Missed Merges)**  
If cameras are not deduplicating the same person correctly during crossover, adjust your matching sensitivities:
- **Spatial Limit**: Tweak the individual overlap zone's `distance_threshold_m` property within `fusion_config.json` / `overlap_zones.json` (defaults to auto-computed bounds).
- **Temporal Limit**: Use `--timestamp-tolerance <seconds>` or modify `timestamp_tolerance_s` in configurations to allow greater leniency if cameras have heavily unsynchronized local clocks.

**Missing Crossing Output Rows**  
Check `--phase 3` to verify homography projection correctly lands people on the virtual grid. Confirm `edges.json` segment coordinates physically break the tracking path on the map overlay. Verify YOLO is establishing continuous `track_id` values per camera feed.
