# 🏭 Factory Floor Crowd Tracker - 9 Camera Setup Guide

Welcome to the Factory Floor visual tracker! This pipeline is capable of tracking moving persons seamlessly across multiple occluded and non-occluded overlapping camera zones, mapping their real-world floor positions, and logging when they cross designated virtual counting edges.

This documentation explicitly covers how to process directories of recorded videos across up to **9 distinct camera angles**, as well as the theory behind the automatic occlusion handler (`FootEstimator`), and the logic of our weighted multi-camera zone fusion (`DetectionFuser`).

> **New in this release:** Videos can be stored and streamed directly from **AWS S3** — no need to copy files locally before processing. See the [S3 Video Source](#-s3-video-source-support) section for full details.

---

## 📋 Table of Contents

1. [Directory Structure Configuration](#-directory-structure-configuration)
2. [S3 Video Source Support](#-s3-video-source-support) ← **New**
3. [Configuration & Calibration Setup](#-configuration--calibration-setup)
4. [Running the Production Pipeline](#-running-the-production-pipeline)
5. [AWS Production Deployment](#-aws-production-deployment) ← **New**
6. [Theory of Operation](#-theory-of-operation)

---

## 🏗 Directory Structure Configuration

Previously, the pipeline was built to handle a single video per-camera. It has now been upgraded to seamlessly process **entire directories** linearly in batches.

To setup 9 cameras for offline processing:
1. Create a root data directory for your pipeline (e.g. `./videos`).
2. Inside that directory, create an individual folder for each camera using the exact `id` you assign them in `config/cameras.json`.

Example Folder Structure:
```text
videos/
├── cam_1/
│   ├── 0800_0900.mp4
│   ├── 0900_1000.mp4
│   └── 1000_1100.mp4
├── cam_2/
│   ├── cam_2_morning.mp4
│   └── cam_2_afternoon.mp4
...
└── cam_9/
    └── full_day.mp4
```

### Editing `config/cameras.json` (Local Paths)
Inside your configuration, set `"source"` to the **absolute path** of that folder, rather than a single video file.

```json
{
  "cameras": [
    {
      "id": "cam_1",
      "name": "Line 1 Main View",
      "source": "/path/to/my/videos/cam_1",
      "track_point": "bottom",
      ...
    },
    ...
```

---

## ☁️ S3 Video Source Support

The pipeline supports reading video files **directly from AWS S3** using the `S3VideoSource` class in `pipeline/s3_source.py`. This is the **recommended approach** for production deployments, keeping disks lean and your source-of-truth in the cloud.

### How It Works

1. On startup, `PerCameraProcessor` detects if the `"source"` field in `cameras.json` begins with `s3://`.
2. It initialises `S3VideoSource`, which **lists all video keys** under that S3 prefix alphabetically.
3. During processing it **downloads one video at a time**, processes all of its frames, then **immediately deletes the local copy** to minimise disk usage.
4. Temporary files land in `output/tmp_s3/<cam_id>/` and are cleaned up automatically.

### Supported Video Formats in S3

The S3 source recognises these extensions (same as local mode):

| Extension | Format         |
|-----------|----------------|
| `.mp4`    | MPEG-4 (H.264) |
| `.avi`    | AVI            |
| `.mov`    | QuickTime      |
| `.mkv`    | Matroska       |
| `.ts`     | MPEG Transport Stream |

---

### Step 1 — Configure AWS Credentials

The pipeline reads credentials **exclusively** from `config/aws_credentials.json`.

**Create or edit `config/aws_credentials.json`:**

```json
{
    "aws_access_key_id":     "YOUR_ACCESS_KEY_ID",
    "aws_secret_access_key": "YOUR_SECRET_ACCESS_KEY",
    "region_name":           "ap-south-1"
}
```

A template is available at `config/aws_credentials.json.example`.

> **⚠️ SECURITY WARNING:** `config/aws_credentials.json` contains **real AWS keys**. It must be added to `.gitignore` immediately to prevent accidental commits. See [Security Notes](#-security-notes) below.

**Region codes for common deployments:**

| Location       | Region Code        |
|----------------|--------------------|
| India (Mumbai) | `ap-south-1`       |
| Japan (Tokyo)  | `ap-northeast-1`   |
| US East        | `us-east-1`        |
| US West        | `us-west-2`        |

---

### Step 2 — Upload Videos to S3

Organise your S3 bucket so each camera has its own "folder" (prefix):

```
s3://japan-data-2026/
├── cam_1/
│   ├── 0800_0900.mp4
│   ├── 0900_1000.mp4
│   └── 1000_1100.mp4
├── cam_2/
│   ├── cam_2_morning.mp4
│   └── cam_2_afternoon.mp4
...
└── cam_9/
    └── full_day.mp4
```

**Upload using the AWS CLI:**

```bash
# Upload a single file
aws s3 cp /local/path/cam_1/0800_0900.mp4 s3://japan-data-2026/cam_1/

# Upload an entire camera folder (recursive)
aws s3 cp /local/path/cam_1/ s3://japan-data-2026/cam_1/ --recursive

# Sync (only uploads new/changed files)
aws s3 sync /local/path/cam_1/ s3://japan-data-2026/cam_1/
```

**Verify uploads:**

```bash
aws s3 ls s3://japan-data-2026/cam_1/
```

---

### Step 3 — Set S3 URIs in `config/cameras.json`

Change the `"source"` field for each camera to point at the S3 prefix (folder URI). **Trailing slash is optional.**

```json
{
  "cameras": [
    {
      "id": "cam_1",
      "name": "Camera 1 (Left)",
      "track_point": "bottom",
      "source": "s3://japan-data-2026/cam_1",
      ...
    },
    {
      "id": "cam_2",
      "name": "Camera 2 (Right)",
      "track_point": "bottom",
      "source": "s3://japan-data-2026/cam_2",
      ...
    }
  ]
}
```

> **Current live configuration:** Both `cam_1` and `cam_2` are already pointing at `s3://japan-data-2026/cam_1` and `s3://japan-data-2026/cam_2` respectively.

---

### Step 4 — Install Dependencies

Make sure all Python dependencies (including `boto3`) are installed:

```bash
pip install -r requirements.txt
```

`requirements.txt` already includes `boto3`.

---

### Step 5 — Verify S3 Connectivity

Run the connectivity smoke test before a full pipeline run:

```bash
# Quick connectivity check — downloads one file, reads one frame, prints results
python test_s3.py
```

Expected output:
```
Downloading...
Downloaded, size: <some bytes>
Opening cv2...
Opened: True
Read a frame: True
```

If you see `True / True` for "Opened" and "Read a frame", S3 is working correctly.

---

### Step 6 — Run the Pipeline (S3 Mode)

The pipeline commands are **identical** to local mode — the S3 path is transparent:

```bash
# Process a single camera (downloads from S3 automatically)
python main.py --process-camera cam_1
python main.py --process-camera cam_2

# Process all cameras sequentially
python main.py --process
```

During processing you will see log lines like:

```
[cam_1] S3VideoSource: found 3 video(s) at s3://japan-data-2026/cam_1
[cam_1] Downloading s3://japan-data-2026/cam_1/0800_0900.mp4 …
[cam_1] Download complete: 0800_0900.mp4
[cam_1] Processing file: 0800_0900.mp4
[cam_1] Deleted local copy: 0800_0900.mp4   ← freed immediately
[cam_1] Downloading s3://japan-data-2026/cam_1/0900_1000.mp4 …
...
```

Temporary downloads are written to `output/tmp_s3/<cam_id>/` and **deleted automatically** after each video is processed.

---

### S3 Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `FileNotFoundError: Missing AWS credentials file at: config/aws_credentials.json` | Credentials file missing | Create `config/aws_credentials.json` from the `.example` template |
| `NoCredentialsError` or `ClientError: Access Denied` | Wrong keys or insufficient IAM permissions | Verify `aws_access_key_id` / `aws_secret_access_key` in the JSON, and ensure the IAM user has `s3:GetObject` + `s3:ListBucket` permissions |
| `S3VideoSource: found 0 video(s)` | Wrong S3 prefix / no matching files | Check the `"source"` URI in `cameras.json`; verify files exist with `aws s3 ls s3://bucket/prefix/` |
| `botocore.exceptions.EndpointResolutionError` | Wrong region | Set `region_name` in `aws_credentials.json` to match your bucket's region |
| `cv2.VideoCapture` returns `Opened: False` | Corrupt file or unsupported codec | Re-encode with `ffmpeg -i input.mp4 -c:v libx264 output.mp4` |
| High disk usage during processing | Old temp files not cleaned | Check `output/tmp_s3/` — may be a leftover from a crashed run; manually delete |
| `ImportError: No module named 'boto3'` | boto3 not installed | Run `pip install -r requirements.txt` |

#### Required IAM Permissions

The AWS IAM user / role needs at minimum:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:ListBucket",
    "s3:GetObject"
  ],
  "Resource": [
    "arn:aws:s3:::japan-data-2026",
    "arn:aws:s3:::japan-data-2026/*"
  ]
}
```

---

### 🔒 Security Notes

> **CRITICAL:** `config/aws_credentials.json` contains live AWS access keys and **must never be committed to git**.

**Immediately add a `.gitignore`** to the project root (if not already present):

```bash
# Create or append to .gitignore
cat >> .gitignore << 'EOF'
# AWS credentials — NEVER commit real keys
config/aws_credentials.json

# Large model weights
*.pt

# Pipeline outputs
output/
output/tmp_s3/

# Python cache
__pycache__/
*.pyc
.pytest_cache/
EOF
```

**Best practices:**
- Rotate your AWS keys regularly via the IAM console.
- For cloud deployments (EC2/ECS), replace explicit keys with an **IAM Instance Role** — no credentials file needed.
- Restrict the IAM policy to the **minimum permissions** listed above (read-only S3).

---

## 🛠 Configuration & Calibration Setup

Before running the tracking pipeline, each camera must be fully characterized — from lens distortion to real-world floor geometry. **All calibration steps can be performed directly from the Web UI** (no terminal required after initial setup).

---

## 🌐 Recommended: End-to-End Setup via Web UI

This is the easiest way to set up and run the full system without using the command line.

### Step 1 — Start the Web Server
```bash
python3 web_ui/app.py
```
Navigate to `http://localhost:5001` (or your EC2 public IP on port `5001`) and log in:
- **Username:** `admin`
- **Password:** `password123`

---

### Step 2 — Register Your Cameras

1. On the **Dashboard**, click **`+ Add Camera`**.
2. Enter a Camera ID (e.g. `cam_1`), a display name, and your `s3://bucket/cam_1` source URI.
3. Click **Save** — the camera is now registered in `cameras.json`.

> Repeat for all your cameras (up to 9 or more).

---

### Step 3 — Upload Videos to S3

On each **Camera detail page** (`/camera/<id>`), you have two options:

#### Option A: Upload directly from your computer
- Use the **"Upload Video to S3"** form to select a local `.mp4` file and push it straight to the correct S3 folder.

#### Option B: Import from Google Drive 📁 *(New)*
- Paste a **public** Google Drive folder link (or single file link) into the **"Import from Google Drive"** card.
- Click **"Import Videos from Drive → S3"**.
- The server automatically downloads all video files from the folder and uploads them to `s3://bucket/cam_id/`.
- **⚠️ Requirement:** Your Google Drive folder must be shared as **"Anyone with the link"**.

> **Supported video formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.ts`

---

### Step 4 — Lens Distortion Calibration (Intrinsic)

This step removes fisheye/barrel distortion so bounding box feet land accurately on the floor.

**Via Web UI:**
1. Open `/camera/<id>` → scroll to **"Intrinsic Calibration"** card.
2. Select a method (Chessboard / Straight Lines / Circle Grid).
3. Optionally upload a calibration video. **If left blank**, the system automatically extracts the first frame of your S3 footage.
4. Click **"Run Intrinsic Calibration"** — runs in the background, results saved to `cameras.json`.

**Via CLI (alternative):**
```bash
python main.py --intrinsic cam_1 --source path/to/checkerboard.mp4
python main.py --intrinsic cam_1 --method circles --source path/to/grid.mp4
```

---

### Step 5 — OCR Timestamp Region

The system OCR-reads timestamps burned into camera feeds to synchronize all cameras precisely.

**Via Web UI:**
1. On the camera detail page, click **"Set OCR Region"**.
2. A live frame from the camera loads on a canvas.
3. Click and drag a rectangle tightly around the timestamp overlay.
4. Click **"Save Region"** — saved to `cameras.json`.
5. Existing OCR boxes from previous calibrations are shown automatically on load.

**Via CLI (alternative):**
```bash
python main.py --ocr-region cam_1
```

---

### Step 6 — Homography / Floor Projection

This calibrates the pixel-to-meter transformation so detections are mapped accurately to the real-world floor plan.

**Via Web UI:**
1. Click **"Calibrate Homography"** on the camera detail page.
2. A frame from the camera loads on the canvas.
3. Click on a known anchor point in the image, then enter its real-world `X, Y` coordinates in metres in the table.
4. Add ≥4 anchor pairs, then click **"Save Calibration"**.
5. Existing points are shown automatically when reopening.

**Via CLI (alternative):**
```bash
python main.py --calibrate cam_1
```

---

### Step 7 — Auto-Generate Overlap Zones

Once all cameras are calibrated, run the auto-config to compute which cameras overlap each other:
```bash
python main.py --auto-config
```
*(This step is CLI-only as it generates the `overlap_zones.json` from all camera geometries combined.)*

---

### Step 8 — Run the Pipeline & View Results

**Via Web UI (Recommended):**
- On the **Dashboard**, click **"Run Full Pipeline"** to process all cameras, fuse data, and generate the visualization MP4.
- All pipeline output files appear in the **Results** section (`/results`).
- Click the green **▶ Play** button to watch the visualization in a pop-up video player.
- Click **Download** to get the `.csv` crossing data for client delivery.

**Via CLI (alternative):**
```bash
# Process all cameras sequentially
python main.py --process

# Or process one at a time
python main.py --process-camera cam_1
python main.py --process-camera cam_2

# Fuse per-camera CSVs into one unified file
python main.py --fuse-only

# Visualize the result at 2x speed
python main.py --visualize output/fused_crossings.csv --playback-speed 2.0
```

---

## 🚀 AWS Production Deployment

To make the application securely accessible from anywhere for your clients, a production deployment package is provided inside the `deploy/` folder. This skips the built-in Flask development server and uses **Gunicorn** and **Nginx** for robust hosting.

### Deployment Steps:

1. Provision a fresh **Ubuntu 22.04 LTS** EC2 instance on AWS.
2. Ensure its Security Group allows inbound traffic on ports `80` (HTTP) and `22` (SSH).
3. Connect to the instance and git clone/copy this project code to it (e.g., into `/opt/sober_crowd`).
4. Run the automated setup script with absolute paths:
   ```bash
   chmod +x deploy/setup_ec2.sh
   sudo ./deploy/setup_ec2.sh
   ```
5. Follow the on-screen instructions inside the script. It will pause to let you set up your `.env` credentials file (copy from `web_ui/.env.template`).
6. Press enter to let the script finish installing Nginx, configuring the firewall, and setting up Gunicorn as a background `systemd` service (`sober_crowd`).

You can then access the live application securely at your EC2 instance's Public IPv4 DNS or IP address without needing to keep a terminal open!

---

## 🧠 Theory of Operation

### 1. `FootEstimator` (Automatic Occlusion Handling)
In crowded factories, Camera A might only capture the **upper half** of a person because another person (or object) is standing directly in front of them from that camera's perspective. Standard tracking would project the bounding box's *visible* bottom pixel onto the floor, pulling their estimated physical coordinate heavily towards the camera, placing them falsely inside the wrong zone.

The newly built `FootEstimator` intervenes *before* homography:
1. **Dynamic Calibration Grid:** We maintain a median size representation of non-distorted people moving freely through regions of the screen.
2. **Occlusion Detection:** The system detects when a bounding box's height drops disproportionately to its width, and if its bottom edge overlaps cleanly with the mid-to-lower portion of another tracked box.
3. **Reference Extrapolation:** Instead of using the distorted, clipped bottom pixel, we find their unoccluded "reference height" dynamically given their X,Y position in the camera lens, and mathematically push the `foot_y` pixel down safely past the occluding object back to ground level.
4. **Confidence Nerfing:** We slightly decrease the detection's `occlusion_confidence` so the fusion logic knows this position is slightly more theoretical.

### 2. Weighted Zone Fusion (`DetectionFuser`)
After per-camera crossing files are gathered, `python main.py --fuse-only` initiates Hungarian matching to tie points together between `cam_1` ... `cam_9`.

If John is standing in an overlap zone visible from both Camera 1 and Camera 2:
- What happens if Camera 1 correctly tracked his full body, but Camera 2 only saw his shoulder because of the `FootEstimator` occlusion handler?
- Our fusion math inherently favors raw spatial authority. Instead of blindly averaging the two coordinates—putting him squarely in between the true center and the badly occluded center—we calculate a **visibility-weighted center**. 

Camera 1's `occlusion_confidence` remained at `1.0`. Camera 2's `occlusion_confidence` dropped to `0.5` after the estimator had to infer his feet. Therefore, the resulting `(x, y)` coordinate is weighted heavily towards Camera 1's perfect view!
