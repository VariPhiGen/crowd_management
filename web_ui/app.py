from __future__ import annotations

from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify, send_file
import os
import signal
import json
import subprocess
import io
import cv2
from pathlib import Path
from werkzeug.utils import secure_filename
import time
import re
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import signal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- AWS / S3 Setup ---
def get_s3_client():
    creds_path = CONFIG_DIR / "aws_credentials.json"
    if creds_path.exists():
        try:
            with open(creds_path, 'r') as f:
                creds = json.load(f)
                return boto3.client(
                    's3',
                    aws_access_key_id=creds.get('aws_access_key_id'),
                    aws_secret_access_key=creds.get('aws_secret_access_key'),
                    region_name=creds.get('region_name', 'ap-south-1')
                )
        except Exception as e:
            print(f"Failed to load AWS credentials from {creds_path}: {e}")
            
    # Fallback to default environment / IAM Role if file doesn't exist
    return boto3.client('s3')

# Optional default bucket if none specified in camera source
DEFAULT_BUCKET = "crowd-management-pipeline-bucket" # Change as needed

# --- App Initialization ---
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "crimenabi-dev-key-change-in-prod")

# --- Configuration paths ---
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_DIR   = PROJECT_ROOT / "config"
CAMERAS_CFG  = CONFIG_DIR / "cameras.json"
FLOOR_CFG    = CONFIG_DIR / "floor_config.json"
EDGES_CFG    = CONFIG_DIR / "edges.json"
FUSION_CFG   = CONFIG_DIR / "fusion_config.json"
OUTPUT_DIR   = PROJECT_ROOT / "output"

# In-memory store for background job statuses
background_jobs = {}

# Map job keys to their log filenames
_JOB_LOG_MAP = {
    "pipeline_main": "pipeline_process.log",
    "pipeline_fuse": "pipeline_fuse.log",
    "pipeline_visualize": "pipeline_visualize.log",
    "floor_auto_config": "auto_config.log",
    "deep_repair": "deep_repair.log",
}

# Persistent job registry on disk so all gunicorn workers can see/cancel jobs
JOBS_DIR = PROJECT_ROOT / "output" / ".jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

def _save_job_pid(job_id: str, pid: int, name: str = ""):
    """Write a job's PID to disk so any worker can find and kill it."""
    (JOBS_DIR / f"{job_id}.json").write_text(
        json.dumps({"pid": pid, "name": name, "start_time": time.time()})
    )

def _remove_job_pid(job_id: str):
    (JOBS_DIR / f"{job_id}.json").unlink(missing_ok=True)

def _load_job_pid(job_id: str) -> dict | None:
    p = JOBS_DIR / f"{job_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

# ADMIN CREDENTIALS
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "password123")

def get_floor_config() -> dict:
    """Read floor_config.json; return sensible defaults if missing."""
    defaults = {
        "floor_width_m": 30.0, "floor_height_m": 86.5,
        "grid_cell_size_m": 1.0, "major_grid_every_m": 5,
        "origin": "bottom_left", "floor_origin_x_m": 0.0, "floor_origin_y_m": 0.0,
    }
    try:
        with open(FLOOR_CFG) as f:
            data = json.load(f)
        defaults.update(data)
    except Exception:
        pass
    return defaults


def get_edges_info() -> dict:
    """Return basic info about edges.json (count, step, auto flag)."""
    try:
        with open(EDGES_CFG) as f:
            data = json.load(f)
        return {
            "count":   len(data.get("edges", [])),
            "step_m":  data.get("step_m", 1.0),
            "is_auto": data.get("_auto", False),
        }
    except Exception:
        return {"count": 0, "step_m": 1.0, "is_auto": True}


def get_fusion_config() -> dict:
    """Read fusion_config.json; return sensible defaults if missing."""
    defaults = {
        "timestamp_tolerance_s": 1.0,
        "default_distance_threshold_m": 2.0,
        "expected_date_range": {"start": "", "end": ""},
    }
    try:
        with open(FUSION_CFG) as f:
            data = json.load(f)
        defaults.update(data)
    except Exception:
        pass
    return defaults


def save_expected_date_range(start_date: str, end_date: str) -> None:
    """Persist expected_date_range to fusion_config.json."""
    try:
        with open(FUSION_CFG) as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    cfg["expected_date_range"] = {"start": start_date, "end": end_date}
    with open(FUSION_CFG, "w") as f:
        json.dump(cfg, f, indent=4)


def get_cameras():
    """Reads cameras.json to get the list of active cameras."""
    try:
        with open(CAMERAS_CFG, 'r') as f:
            data = json.load(f)
            return data.get('cameras', [])
    except Exception as e:
        print(f"Error reading cameras.json: {e}")
        return []

# --- Authentication Middleware ---
@app.before_request
def require_login():
    # Only allow login and static files if not authenticated
    allowed_routes = ['login', 'static']
    if request.endpoint not in allowed_routes and 'logged_in' not in session:
        return redirect(url_for('login'))

# --- Routes ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] == ADMIN_USERNAME and request.form['password'] == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    cameras      = get_cameras()
    floor_cfg    = get_floor_config()
    edges_info   = get_edges_info()
    fusion_cfg   = get_fusion_config()
    message = request.args.get('floor_msg') or None
    error   = request.args.get('floor_err') or None
    # Default: all cameras selected; remembered across POST so checkboxes persist
    all_cam_ids = [c['id'] for c in cameras]
    selected_cameras = all_cam_ids  # overwritten on POST

    if request.method == 'POST':
        selected_cameras = request.form.getlist('camera_ids') or all_cam_ids
        action = request.form.get('pipeline_action')
        try:
            env = os.environ.copy()
            # Always use the venv Python that runs this Flask app so torch/CUDA is available
            import sys as _sys
            python_exec = _sys.executable

            if action == 'process':
                # Prevent duplicate runs — check if a pipeline job is already active
                for _jf in JOBS_DIR.glob("pipeline_main.json"):
                    _dj = _load_job_pid("pipeline_main")
                    if _dj:
                        try:
                            os.kill(_dj["pid"], 0)
                            error = "A processing job is already running. Cancel it first."
                            return render_template('dashboard.html', cameras=cameras,
                                message=message, error=error, active_jobs={},
                                selected_cameras=selected_cameras, fusion_cfg=fusion_cfg,
                                floor_cfg=floor_cfg, edges_info=edges_info)
                        except (ProcessLookupError, PermissionError):
                            _remove_job_pid("pipeline_main")

                # Read performance / detection settings from form
                workers        = int(request.form.get('workers', 4))
                frame_stride   = int(request.form.get('frame_stride', 2))
                ocr_interval   = int(request.form.get('ocr_interval', 0))
                classes        = request.form.get('classes', 'person,car,motorcycle,truck').strip()
                confidence     = float(request.form.get('confidence', 0.50))
                track_point    = request.form.get('track_point', 'bottom').strip()
                fusion_time    = float(request.form.get('fusion_time_tol', 1.0))
                fusion_dist    = float(request.form.get('fusion_dist_tol', 2.5))
                date_start     = request.form.get('date_range_start', '').strip()
                date_end       = request.form.get('date_range_end', '').strip()
                if date_start and date_end:
                    save_expected_date_range(date_start, date_end)
                # Camera selection — empty list means "all cameras"
                selected_cams  = request.form.getlist('camera_ids')
                cameras_arg    = ','.join(selected_cams) if selected_cams else None

                cmd = [
                    python_exec, "main.py", "--process",
                    "--workers",             str(workers),
                    "--frame-stride",        str(frame_stride),
                    "--ocr-interval",        str(ocr_interval),
                    "--classes",             classes,
                    "--confidence",          str(confidence),
                    "--track-point",         track_point,
                    "--timestamp-tolerance", str(fusion_time),
                    "--fusion-dist-tol",     str(fusion_dist),
                ]
                if cameras_arg:
                    cmd += ["--cameras", cameras_arg]

                OUTPUT_DIR.mkdir(exist_ok=True)
                log_file = OUTPUT_DIR / "pipeline_process.log"
                log_f = open(log_file, 'w')
                proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                        stdout=log_f, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                log_f.close()
                job_name = f'Main Pipeline (workers={workers}, stride={frame_stride}, classes={classes})'
                background_jobs["pipeline_main"] = {
                    'process':    proc,
                    'log_file':   str(log_file),
                    'start_time': time.time(),
                    'status':     'running',
                    'name':       job_name,
                }
                _save_job_pid("pipeline_main", proc.pid, job_name)
                message = (
                    f"Full pipeline started — {workers} cameras in parallel, "
                    f"stride {frame_stride}, OCR {'auto' if ocr_interval == 0 else str(ocr_interval) + 'f'}, "
                    f"classes: {classes}. Check logs below."
                )

            elif action == 'fuse':
                fusion_time   = float(request.form.get('fusion_time_tol', 1.0))
                fusion_dist   = float(request.form.get('fusion_dist_tol', 2.5))
                date_start    = request.form.get('date_range_start', '').strip()
                date_end      = request.form.get('date_range_end', '').strip()
                if date_start and date_end:
                    save_expected_date_range(date_start, date_end)
                selected_cams = request.form.getlist('camera_ids')
                cameras_arg   = ','.join(selected_cams) if selected_cams else None
                cmd = [
                    python_exec, "main.py", "--fuse-only",
                    "--timestamp-tolerance", str(fusion_time),
                    "--fusion-dist-tol",     str(fusion_dist),
                ]
                if cameras_arg:
                    cmd += ["--cameras", cameras_arg]

                log_file = OUTPUT_DIR / "pipeline_fuse.log"
                log_f = open(log_file, 'w')
                proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                        stdout=log_f, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                log_f.close()
                background_jobs["pipeline_fuse"] = {
                    'process':    proc,
                    'log_file':   str(log_file),
                    'start_time': time.time(),
                    'status':     'running',
                    'name':       'Fusion Stage',
                }
                _save_job_pid("pipeline_fuse", proc.pid, "Fusion Stage")
                message = "Fusion stage started in the background."

            elif action == 'deep_repair':
                date_start    = request.form.get('date_range_start', '').strip()
                date_end      = request.form.get('date_range_end', '').strip()
                if not date_start or not date_end:
                    error = "Please set both Start Date and End Date before running Deep Repair."
                else:
                    save_expected_date_range(date_start, date_end)
                    selected_cams = request.form.getlist('camera_ids')
                    cameras_arg   = ','.join(selected_cams) if selected_cams else None
                    cmd = [python_exec, "main.py", "--repair-only"]
                    if cameras_arg:
                        cmd += ["--cameras", cameras_arg]

                    log_file = OUTPUT_DIR / "deep_repair.log"
                    log_f = open(log_file, 'w')
                    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                            stdout=log_f, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                    log_f.close()
                    background_jobs["deep_repair"] = {
                        'process':    proc,
                        'log_file':   str(log_file),
                        'start_time': time.time(),
                        'status':     'running',
                        'name':       'Deep Repair',
                    }
                    _save_job_pid("deep_repair", proc.pid, "Deep Repair")
                    cams_label = cameras_arg or "all cameras"
                    message = f"Deep Repair started for {cams_label} (range: {date_start} → {date_end}). Check logs below."

            elif action == 'visualize':
                csv_path = OUTPUT_DIR / "fused_crossings.csv"
                mp4_path = OUTPUT_DIR / "visualization.mp4"
                if not csv_path.exists():
                    error = "Fused CSV not found. Please run the pipeline or fusion first."
                else:
                    cmd = [python_exec, "main.py", "--visualize", str(csv_path),
                           "--headless-mp4", str(mp4_path)]

                    log_file = OUTPUT_DIR / "pipeline_visualize.log"
                    log_f = open(log_file, 'w')
                    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                            stdout=log_f, stderr=subprocess.STDOUT,
                                            start_new_session=True)
                    log_f.close()
                    background_jobs["pipeline_visualize"] = {
                        'process':    proc,
                        'log_file':   str(log_file),
                        'start_time': time.time(),
                        'status':     'running',
                        'name':       'Visualization Generation',
                    }
                    _save_job_pid("pipeline_visualize", proc.pid, "Visualization Generation")
                    message = "Visualization generation started. MP4 will appear in Results when finished."

        except Exception as e:
            error = f"Failed to start pipeline process: {e}"

    # Prune finished/failed jobs older than 10 minutes to avoid memory growth
    _now = time.time()
    _stale = [k for k, v in background_jobs.items()
              if v['status'] in ('finished', 'failed', 'cancelled')
              and (_now - v.get('start_time', _now)) > 600]
    for k in _stale:
        background_jobs.pop(k, None)

    # Get active global pipeline jobs for dashboard
    active_jobs = {}
    for j_key, j_data in list(background_jobs.items()):
        if j_key.startswith("pipeline_") or j_key == 'floor_auto_config':
            if j_data['status'] == 'running':
                poll = j_data['process'].poll()
                if poll is not None:
                    j_data['status'] = 'finished' if poll == 0 else 'failed'
                    _remove_job_pid(j_key)
            logs = ""
            if os.path.exists(j_data['log_file']):
                try:
                    with open(j_data['log_file'], 'r', encoding='utf-8', errors='replace') as f:
                        logs = f.readlines()[-8:]
                except Exception:
                    logs = []

            active_jobs[j_key] = {
                'name':     j_data['name'],
                'status':   j_data['status'],
                'logs':     "".join(logs),
                'log_file': j_data['log_file'],
                'elapsed':  int(_now - j_data.get('start_time', _now)),
            }

    # Also pick up jobs started by other gunicorn workers (disk registry)
    for jf in JOBS_DIR.glob("*.json"):
        j_key = jf.stem
        if j_key in active_jobs:
            continue
        disk_job = _load_job_pid(j_key)
        if not disk_job:
            continue
        pid = disk_job.get("pid")
        # Check if the process is still alive
        try:
            os.kill(pid, 0)
            status = 'running'
        except (ProcessLookupError, PermissionError):
            status = 'finished'
            _remove_job_pid(j_key)
        log_path = str(OUTPUT_DIR / _JOB_LOG_MAP.get(j_key, f"{j_key}.log"))
        logs = ""
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                    logs = f.readlines()[-8:]
            except Exception:
                logs = []
        active_jobs[j_key] = {
            'name':    disk_job.get("name", j_key),
            'status':  status,
            'logs':    "".join(logs) if isinstance(logs, list) else logs,
            'log_file': log_path,
            'elapsed': int(_now - disk_job.get("start_time", _now)),
        }

    return render_template('dashboard.html', cameras=cameras, message=message, error=error,
                           active_jobs=active_jobs, selected_cameras=selected_cameras,
                           floor_cfg=floor_cfg, edges_info=edges_info, fusion_cfg=fusion_cfg)

@app.route('/api/floor-config', methods=['POST'])
def floor_config_action():
    """Handle floor config save / auto-compute / edge regeneration."""
    import sys as _sys
    action     = request.form.get('floor_action')
    python_exec = _sys.executable

    if action == 'save':
        # Read submitted values and persist to floor_config.json
        try:
            current = get_floor_config()
            current['floor_width_m']    = float(request.form.get('floor_width_m',    current['floor_width_m']))
            current['floor_height_m']   = float(request.form.get('floor_height_m',   current['floor_height_m']))
            current['grid_cell_size_m'] = float(request.form.get('grid_cell_size_m', current['grid_cell_size_m']))
            current['major_grid_every_m'] = int(request.form.get('major_grid_every_m', current['major_grid_every_m']))
            current['floor_origin_x_m'] = float(request.form.get('floor_origin_x_m', current.get('floor_origin_x_m', 0.0)))
            current['floor_origin_y_m'] = float(request.form.get('floor_origin_y_m', current.get('floor_origin_y_m', 0.0)))
            with open(FLOOR_CFG, 'w') as f:
                json.dump(current, f, indent=2)
            return redirect(url_for('dashboard') + '?floor_msg=Floor+config+saved+successfully.')
        except Exception as e:
            return redirect(url_for('dashboard') + f'?floor_err=Save+failed:+{e}')

    elif action == 'auto_compute':
        # Run --auto-config in background
        log_file = OUTPUT_DIR / "auto_config.log"
        OUTPUT_DIR.mkdir(exist_ok=True)
        log_f = open(log_file, 'w')
        env   = os.environ.copy()
        proc  = subprocess.Popen(
            [python_exec, 'main.py', '--auto-config'],
            cwd=str(PROJECT_ROOT), env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_f.close()
        background_jobs['floor_auto_config'] = {
            'process':    proc,
            'log_file':   str(log_file),
            'start_time': time.time(),
            'status':     'running',
            'name':       'Auto-compute Floor Config',
        }
        _save_job_pid('floor_auto_config', proc.pid, 'Auto-compute Floor Config')
        return redirect(url_for('dashboard') + '?floor_msg=Auto-compute+started.+Check+Running+Jobs.')

    elif action == 'regen_edges':
        # Regenerate edges.json synchronously (fast — pure Python)
        try:
            import sys as _sys
            _old_path = list(_sys.path)
            if str(PROJECT_ROOT) not in _sys.path:
                _sys.path.insert(0, str(PROJECT_ROOT))
            from fusion.crossing import generate_edges
            step_m    = float(request.form.get('step_m', 1.0))
            fc        = get_floor_config()
            edges     = generate_edges(
                floor_width_m  = fc['floor_width_m'],
                floor_height_m = fc['floor_height_m'],
                step_m         = step_m,
                save_path      = EDGES_CFG,
            )
            _sys.path = _old_path
            return redirect(url_for('dashboard') + f'?floor_msg=Edges+regenerated:+{len(edges)}+lines+at+{step_m}m+step.')
        except Exception as e:
            return redirect(url_for('dashboard') + f'?floor_err=Edge+regen+failed:+{e}')

    return redirect(url_for('dashboard'))


@app.route('/api/system-health')
def system_health():
    """Return disk space, GPU VRAM, and running job count as JSON."""
    import shutil
    health = {}

    # Disk space for output directory
    try:
        total, used, free = shutil.disk_usage(str(OUTPUT_DIR if OUTPUT_DIR.exists() else PROJECT_ROOT))
        health['disk_free_gb']  = round(free  / 1e9, 1)
        health['disk_total_gb'] = round(total / 1e9, 1)
        health['disk_used_pct'] = round(used / total * 100, 1)
    except Exception:
        health['disk_free_gb'] = None

    # GPU VRAM via nvidia-smi (non-blocking)
    try:
        import subprocess as _sp
        out = _sp.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            timeout=3
        ).decode().strip()
        parts = [p.strip() for p in out.split(',')]
        health['gpu_vram_used_mb']  = int(parts[0])
        health['gpu_vram_total_mb'] = int(parts[1])
        health['gpu_util_pct']      = int(parts[2])
    except Exception:
        health['gpu_vram_used_mb'] = None

    # Running jobs — count from both in-memory and disk registry
    _running = sum(1 for v in background_jobs.values() if v.get('status') == 'running')
    for _jf in JOBS_DIR.glob("*.json"):
        _jkey = _jf.stem
        if _jkey in background_jobs and background_jobs[_jkey].get('status') == 'running':
            continue
        _dj = _load_job_pid(_jkey)
        if _dj:
            try:
                os.kill(_dj["pid"], 0)
                _running += 1
            except (ProcessLookupError, PermissionError):
                _remove_job_pid(_jkey)
    health['running_jobs'] = _running

    return jsonify(health)


@app.route('/api/job/<job_key>/log')
def job_log(job_key):
    """Return full log content for a background job (plain text)."""
    if not session.get('logged_in'):
        return "Unauthorized", 401

    log_path = None
    job = background_jobs.get(job_key)
    if job:
        log_path = job.get('log_file', '')
    if not log_path:
        log_path = str(OUTPUT_DIR / _JOB_LOG_MAP.get(job_key, f"{job_key}.log"))

    if not log_path or not os.path.exists(log_path):
        return "Log file not found", 404
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        return str(e), 500


def parse_s3_uri(uri):
    """Parse s3://bucket/prefix into bucket and prefix."""
    if not uri.startswith("s3://"):
        return None, None
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    # Ensure prefix ends with / for folder listings if it's not empty
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix

@app.route('/api/camera/<cam_id>/frame')
def camera_frame(cam_id):
    """Extracts a single frame from the first video found for this camera."""
    cameras = get_cameras()
    camera = next((c for c in cameras if c['id'] == cam_id), None)
    if not camera:
        return "Camera not found", 404

    source = camera.get('source', '')
    video_url = None

    if source.startswith('s3://'):
        bucket, prefix = parse_s3_uri(source)
        s3 = get_s3_client()
        try:
            # Find the first video file in this prefix
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            for item in response.get('Contents', []):
                if item['Key'].lower().endswith(('.mp4', '.avi', '.mov')):
                    # Generate a presigned URL valid for 1 hour
                    video_url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket, 'Key': item['Key']},
                        ExpiresIn=3600
                    )
                    break
        except Exception as e:
            return f"S3 Error: {e}", 500
    else:
        # Local directory
        source_path = Path(source)
        if source_path.is_dir():
            for ext in ['.mp4', '.avi', '.mov']:
                files = list(source_path.glob(f"*{ext}"))
                if files:
                    video_url = str(files[0])
                    break
        elif source_path.is_file():
            video_url = str(source_path)

    if not video_url:
        return "No video found for this camera to extract a frame from.", 404

    # Use OpenCV to grab the first frame
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        return "Failed to open video stream", 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Failed to read frame", 500

    # Encode to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        return "Failed to encode frame", 500

    io_buf = io.BytesIO(buffer)
    return send_file(io_buf, mimetype='image/jpeg')

@app.route('/camera/add', methods=['POST'])
def add_camera():
    """Register a new camera to cameras.json"""
    cam_id = request.form.get('cam_id')
    name = request.form.get('name')
    source = request.form.get('source')
    s3_output_folder = request.form.get('s3_output_folder')
    
    if cam_id and name and source:
        with open(CAMERAS_CFG, 'r') as f:
            cfg_data = json.load(f)
            
        for c in cfg_data.get('cameras', []):
            if c['id'] == cam_id:
                # If camera ID already exists, redirect back without adding
                return redirect(url_for('dashboard'))
                
        new_cam = {
            "id": cam_id,
            "name": name,
            "source": source,
            "s3_output_folder": s3_output_folder or "pipeline_outputs"
        }
        if 'cameras' not in cfg_data:
            cfg_data['cameras'] = []
        cfg_data['cameras'].append(new_cam)
        
        with open(CAMERAS_CFG, 'w') as f:
            json.dump(cfg_data, f, indent=4)
            
    return redirect(url_for('dashboard'))

@app.route('/camera/<cam_id>', methods=['GET', 'POST'])
def camera_detail(cam_id):
    """View details for a specific camera, list its S3 contents, and handle uploads."""
    cameras = get_cameras()
    camera = next((c for c in cameras if c['id'] == cam_id), None)
    if not camera:
        return "Camera not found", 404

    s3_uri = camera.get('source', '')
    bucket, prefix = parse_s3_uri(s3_uri)
    
    s3_files = []
    error = None
    success = None
    
    s3 = get_s3_client()

    # Handle Uploads & Folder Creation
    if request.method == 'POST':
        if not bucket:
            error = "Camera source is not a valid S3 URI (s3://bucket/path/)"
        else:
            action = request.form.get('action')
            
            if action == 'upload':
                if 'video_file' not in request.files:
                    error = 'No file part'
                else:
                    file = request.files['video_file']
                    if file.filename == '':
                        error = 'No selected file'
                    elif file:
                        filename = secure_filename(file.filename)
                        s3_key = f"{prefix}{filename}"
                        try:
                            s3.upload_fileobj(file, bucket, s3_key)
                            success = f"File {filename} uploaded successfully to s3://{bucket}/{s3_key}"
                        except Exception as e:
                            error = f"Upload failed: {e}"
                            
            elif action == 'create_folder':
                folder_name = request.form.get('folder_name', '').strip()
                if folder_name:
                    # S3 folders are just 0-byte objects ending in /
                    folder_key = f"{prefix}{folder_name}/"
                    try:
                        s3.put_object(Bucket=bucket, Key=folder_key)
                        success = f"Folder {folder_name} created successfully."
                    except Exception as e:
                        error = f"Failed to create folder: {e}"
            elif action == 'delete':
                file_key = request.form.get('file_key')
                if file_key:
                    s3_key = f"{prefix}{file_key}"
                    try:
                        if file_key.endswith('/'):
                            # Recursive delete for "folders"
                            paginator = s3.get_paginator('list_objects_v2')
                            pages = paginator.paginate(Bucket=bucket, Prefix=s3_key)
                            
                            delete_us = []
                            for page in pages:
                                if 'Contents' in page:
                                    for obj in page['Contents']:
                                        delete_us.append({'Key': obj['Key']})
                            
                            if delete_us:
                                # S3 delete_objects can handle up to 1000 keys at once
                                for i in range(0, len(delete_us), 1000):
                                    s3.delete_objects(Bucket=bucket, Delete={'Objects': delete_us[i:i+1000]})
                            
                            success = f"Folder {file_key} and all its contents deleted from S3."
                        else:
                            s3.delete_object(Bucket=bucket, Key=s3_key)
                            success = f"Deleted {file_key} from S3."
                    except Exception as e:
                        error = f"Failed to delete: {e}"
            
            elif action == 'drive_import':
                raw_input = request.form.get('drive_url', '').strip()
                if not raw_input:
                    error = "Please provide at least one Google Drive URL."
                else:
                    try:
                        import gdown
                        import tempfile
                        import shutil

                        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.ts'}
                        uploaded, failed, warnings = [], [], []

                        # Split input into URLs — supports three formats:
                        #   • single folder URL  (one entry, contains "folders/")
                        #   • multiple file URLs, one per line
                        #   • multiple file URLs, comma-separated (or mixed newline+comma)
                        import re as _re
                        urls = [u.strip() for u in _re.split(r'[\n,]+', raw_input) if u.strip()]

                        def _upload_file(local_path: str, fname: str):
                            s3_key = f"{prefix}{fname}"
                            with open(local_path, 'rb') as fh:
                                s3.upload_fileobj(fh, bucket, s3_key)
                            uploaded.append(fname)

                        def _process_tmp(tmp_dir: str):
                            """Walk tmp_dir and upload every video found to S3."""
                            for root, _, files_found in os.walk(tmp_dir):
                                for fname in files_found:
                                    if Path(fname).suffix.lower() in VIDEO_EXTS:
                                        try:
                                            _upload_file(os.path.join(root, fname), fname)
                                        except Exception as up_err:
                                            failed.append(f"{fname}: {up_err}")

                        if len(urls) == 1 and 'folders' in urls[0]:
                            # ── Single folder URL ─────────────────────────────
                            tmp_dir = tempfile.mkdtemp(prefix='gdrive_import_')
                            try:
                                try:
                                    gdown.download_folder(
                                        urls[0], output=tmp_dir,
                                        quiet=False, use_cookies=False,
                                        remaining_ok=False,
                                    )
                                except RuntimeError as folder_err:
                                    msg = str(folder_err)
                                    if 'more than 50 files' in msg or 'remaining_ok' in msg:
                                        # Download the first 50, warn about the rest
                                        gdown.download_folder(
                                            urls[0], output=tmp_dir,
                                            quiet=False, use_cookies=False,
                                            remaining_ok=True,
                                        )
                                        warnings.append(
                                            "Folder has more than 50 files — Google Drive's "
                                            "bulk limit. Only the first 50 were downloaded. "
                                            "To import the rest, paste each file's individual "
                                            "sharing link (one per line) in the box below."
                                        )
                                    else:
                                        raise
                                _process_tmp(tmp_dir)
                            finally:
                                shutil.rmtree(tmp_dir, ignore_errors=True)

                        else:
                            # ── One or more individual file URLs ──────────────
                            # This path has NO 50-file limit — each file is
                            # downloaded independently, so 200+ files work fine.
                            for url in urls:
                                if not url.startswith('http'):
                                    continue
                                tmp_dir = tempfile.mkdtemp(prefix='gdrive_file_')
                                try:
                                    out = gdown.download(
                                        url, output=tmp_dir + '/',
                                        quiet=False, fuzzy=True,
                                    )
                                    if out:
                                        fname = Path(out).name
                                        if Path(fname).suffix.lower() in VIDEO_EXTS:
                                            try:
                                                _upload_file(out, fname)
                                            except Exception as up_err:
                                                failed.append(f"{fname}: {up_err}")
                                        else:
                                            warnings.append(
                                                f"Skipped '{fname}' — not a video file."
                                            )
                                    else:
                                        failed.append(f"{url[:60]}… — download returned nothing")
                                except Exception as dl_err:
                                    failed.append(f"{url[:60]}…: {dl_err}")
                                finally:
                                    shutil.rmtree(tmp_dir, ignore_errors=True)

                        # ── Build result message ──────────────────────────────
                        if uploaded:
                            success = (
                                f"Imported {len(uploaded)} video(s) → "
                                f"s3://{bucket}/{prefix}:  "
                                + ', '.join(uploaded)
                            )
                        elif not failed:
                            error = "No video files were found at the provided URL(s)."

                        if warnings:
                            success = (success or '') + '  ⚠ ' + '  ⚠ '.join(warnings)
                        if failed:
                            error = (error or '') + '  Failed: ' + ';  '.join(failed)

                    except ImportError:
                        error = "gdown is not installed. Run: pip install gdown"
                    except Exception as e:
                        error = f"Drive import failed: {e}"

    # List current S3 contents for this camera's prefix
    if bucket:
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
            
            # Subfolders (CommonPrefixes)
            for cp in response.get('CommonPrefixes', []):
                s3_files.append({
                    'name': cp['Prefix'].replace(prefix, ''),
                    'type': 'Folder',
                    'size': '-'
                })
                
            # Files (Contents)
            for item in response.get('Contents', []):
                if item['Key'] != prefix: # Skip the folder itself
                    is_video = item['Key'].endswith(('.mp4', '.avi', '.mov'))
                    
                    # Generate temporary presigned URL for preview/download
                    presigned_url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket, 'Key': item['Key']},
                        ExpiresIn=3600
                    )
                    
                    s3_files.append({
                        'name': item['Key'].replace(prefix, ''),
                        'type': 'Video File' if is_video else 'File',
                        'size': f"{item['Size'] / (1024*1024):.2f} MB",
                        'url': presigned_url
                    })
        except Exception as e:
            error = f"Error listing S3 bucket: {e}"

    return render_template('camera.html', camera=camera, s3_files=s3_files, error=error, success=success, bucket=bucket, prefix=prefix)

@app.route('/camera/<cam_id>/delete', methods=['POST'])
def delete_camera(cam_id):
    """Remove a camera from cameras.json."""
    try:
        with open(CAMERAS_CFG, 'r') as f:
            cfg_data = json.load(f)
        cameras = cfg_data.get('cameras', [])
        new_cameras = [c for c in cameras if c.get('id') != cam_id]
        if len(new_cameras) == len(cameras):
            return redirect(url_for('dashboard'))
        cfg_data['cameras'] = new_cameras
        with open(CAMERAS_CFG, 'w') as f:
            json.dump(cfg_data, f, indent=4)
    except Exception as e:
        pass  # redirect anyway
    return redirect(url_for('dashboard'))

@app.route('/output/delete', methods=['POST'])
def delete_output():
    """Delete one output file or all files in output/."""
    filename = request.form.get('filename')
    action = request.form.get('action')
    deleted = []
    try:
        if action == 'all':
            if not OUTPUT_DIR.exists():
                return redirect(url_for('results'))
            for f in OUTPUT_DIR.iterdir():
                if f.is_file():
                    f.unlink()
                    deleted.append(f.name)
        elif filename:
            safe = secure_filename(filename)
            if not safe or safe != filename:
                return redirect(url_for('results'))
            path = OUTPUT_DIR / safe
            if path.exists() and path.is_file():
                path.unlink()
                deleted.append(safe)
    except Exception:
        pass
    return redirect(url_for('results'))

@app.route('/camera/<cam_id>/ocr_region', methods=['GET', 'POST'])
def camera_ocr_region(cam_id):
    """Interactive canvas UI for selecting the OCR region for timestamp extraction."""
    cameras = get_cameras()
    camera = next((c for c in cameras if c['id'] == cam_id), None)
    if not camera:
        return "Camera not found", 404

    if request.method == 'POST':
        try:
            data = request.get_json()
            new_region = data.get('ocr_region')
            if not new_region or len(new_region) != 4:
                return jsonify({'success': False, 'error': 'Invalid region data'})
            # Dashboard sends [xmin, ymin, xmax, ymax] normalized; pipeline expects {x, y, w, h}
            xmin, ymin, xmax, ymax = new_region
            saved_region = {
                'x': float(xmin),
                'y': float(ymin),
                'w': float(xmax - xmin),
                'h': float(ymax - ymin),
                'coordinate_format': 'normalized',
            }
            with open(CAMERAS_CFG, 'r') as f:
                cfg_data = json.load(f)
            for c in cfg_data.get('cameras', []):
                if c['id'] == cam_id:
                    c['ocr_region'] = saved_region
                    break
            with open(CAMERAS_CFG, 'w') as f:
                json.dump(cfg_data, f, indent=4)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    return render_template('ocr_region.html', camera=camera)

@app.route('/camera/<cam_id>/coverage', methods=['GET', 'POST'])
def camera_coverage(cam_id):
    """Interactive canvas UI for drawing the floor coverage polygon."""
    cameras = get_cameras()
    camera  = next((c for c in cameras if c['id'] == cam_id), None)
    if not camera:
        return "Camera not found", 404

    if request.method == 'POST':
        try:
            data = request.get_json()
            polygon = data.get('polygon')   # list of [floor_x, floor_y] in metres
            if not polygon or len(polygon) < 3:
                return jsonify({'success': False, 'error': 'Need at least 3 corner points.'})
            with open(CAMERAS_CFG, 'r') as f:
                cfg_data = json.load(f)
            for c in cfg_data.get('cameras', []):
                if c['id'] == cam_id:
                    c['floor_coverage_polygon'] = [[float(p[0]), float(p[1])] for p in polygon]
                    break
            with open(CAMERAS_CFG, 'w') as f:
                json.dump(cfg_data, f, indent=4)
            return jsonify({'success': True, 'count': len(polygon)})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    return render_template('coverage.html', camera=camera)


@app.route('/camera/<cam_id>/calibrate', methods=['GET', 'POST'])
def camera_calibrate(cam_id):
    """Interactive canvas UI for selecting homography calibration points."""
    cameras = get_cameras()
    camera = next((c for c in cameras if c['id'] == cam_id), None)
    if not camera:
        return "Camera not found", 404

    if request.method == 'POST':
        try:
            data = request.get_json()
            image_points = data.get('image_points')
            floor_points = data.get('floor_points')
            frame_size = data.get('frame_size')
            
            if not image_points or not floor_points or len(image_points) < 4 or len(image_points) != len(floor_points):
                return jsonify({'success': False, 'error': 'Need at least 4 matching point pairs.'})
            
            import numpy as np
            
            # Build numpy arrays
            src_pts = np.array(image_points, dtype=np.float64)   # pixel coords
            dst_pts = np.array(floor_points,  dtype=np.float64)  # real-world metres
            
            # Compute homography using RANSAC for robustness
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=8.0)
            
            if H is None:
                return jsonify({'success': False, 'error': 'Homography computation failed — check that points are not collinear.'})
            
            # --- Reprojection error ---
            n = len(image_points)
            per_point_errors = []
            for i in range(n):
                px, py = image_points[i]
                fx, fy = floor_points[i]
                
                # Project pixel → floor via H
                pt = np.array([px, py, 1.0], dtype=np.float64)
                proj = H @ pt
                proj /= proj[2]
                
                err = float(np.sqrt((proj[0] - fx)**2 + (proj[1] - fy)**2))
                per_point_errors.append(round(err, 4))
            
            rmse = float(np.sqrt(np.mean([e**2 for e in per_point_errors])))
            inlier_count = int(mask.sum()) if mask is not None else n
            
            # Quality rating
            if rmse < 0.05:
                quality = "EXCELLENT"
                quality_color = "success"
            elif rmse < 0.15:
                quality = "GOOD"
                quality_color = "success"
            elif rmse < 0.4:
                quality = "ACCEPTABLE"
                quality_color = "warning"
            else:
                quality = "POOR"
                quality_color = "danger"
            
            # --- Save to cameras.json ---
            with open(CAMERAS_CFG, 'r') as f:
                cfg_data = json.load(f)
            
            for c in cfg_data.get('cameras', []):
                if c['id'] == cam_id:
                    if 'calibration_points' not in c:
                        c['calibration_points'] = {}
                    c['calibration_points']['image_points'] = image_points
                    c['calibration_points']['floor_points'] = floor_points
                    c['calibration_points']['coordinate_format'] = "pixel"
                    c['calibration_points']['calibration_frame_size'] = frame_size
                    c['calibration_points']['points_are_undistorted'] = False
                    
                    # Save computed homography matrix
                    c['homography_matrix'] = H.tolist()
                    break
            
            with open(CAMERAS_CFG, 'w') as f:
                json.dump(cfg_data, f, indent=4)
            
            # Delete stale npz so it gets recomputed next pipeline run
            npz = CONFIG_DIR / f"homography_{cam_id}.npz"
            if npz.exists():
                npz.unlink()
            
            return jsonify({
                'success': True,
                'metrics': {
                    'rmse_m':         round(rmse, 4),
                    'rmse_cm':        round(rmse * 100, 2),
                    'per_point_errors_m': per_point_errors,
                    'inliers':        inlier_count,
                    'total_points':   n,
                    'quality':        quality,
                    'quality_color':  quality_color,
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    return render_template('calibrate.html', camera=camera)

@app.route('/camera/<cam_id>/intrinsic', methods=['POST'])
def camera_intrinsic(cam_id):
    """Handles uploading a calibration video and running intrinsic calibration headlessly."""
    file = request.files.get('calibration_video')
    method = request.form.get('method', 'chessboard')
    
    env = os.environ.copy()
    python_exec = "python3"
    
    # Ensure output dir exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    log_file = OUTPUT_DIR / f"intrinsic_{cam_id}.log"
    
    if file and file.filename != '':
        # User uploaded a calibration file – save and use it directly
        filename = secure_filename(file.filename)
        save_path = OUTPUT_DIR / f"calib_upload_{cam_id}_{filename}"
        file.save(str(save_path))
        cmd = [python_exec, "main.py", "--intrinsic", cam_id, "--source", str(save_path), "--method", method, "--headless"]
    else:
        # No file uploaded – check if the camera source is an S3 URI
        cameras = get_cameras()
        cam = next((c for c in cameras if c['id'] == cam_id), None)
        cam_source = cam.get('source', '') if cam else ''
        
        if cam_source.startswith('s3://'):
            # Download a single video frame from S3 to serve as calibration input
            local_frame_path = None
            try:
                bucket, prefix = parse_s3_uri(cam_source)
                s3 = get_s3_client()
                response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
                # Find first video file
                video_key = None
                for item in response.get('Contents', []):
                    if item['Key'].lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        video_key = item['Key']
                        break
                
                if video_key:
                    # Generate presigned URL and extract one frame using OpenCV
                    presigned_url = s3.generate_presigned_url(
                        'get_object',
                        Params={'Bucket': bucket, 'Key': video_key},
                        ExpiresIn=3600
                    )
                    cap = cv2.VideoCapture(presigned_url)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            local_frame_path = str(OUTPUT_DIR / f"calib_frame_{cam_id}.jpg")
                            cv2.imwrite(local_frame_path, frame)
            except Exception as e:
                with open(str(log_file), 'w') as lf:
                    lf.write(f"ERROR: Failed to extract frame from S3 for calibration: {e}\n")
                return redirect(url_for('camera_detail', cam_id=cam_id))
            
            if local_frame_path:
                cmd = [python_exec, "main.py", "--intrinsic", cam_id, "--source", local_frame_path, "--method", method, "--headless"]
            else:
                # No video found in S3 – let main.py handle it (will fail with a clear message)
                cmd = [python_exec, "main.py", "--intrinsic", cam_id, "--method", method, "--headless"]
        else:
            # Local source – pass it directly to main.py
            cmd = [python_exec, "main.py", "--intrinsic", cam_id, "--method", method, "--headless"]
        
    try:
        # Open log file for output
        log_f = open(log_file, 'w')
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT,
                                start_new_session=True)
        log_f.close()
        job_key = f"intrinsic_{cam_id}"
        background_jobs[job_key] = {
            'process': proc,
            'log_file': str(log_file),
            'start_time': time.time(),
            'status': 'running'
        }
        _save_job_pid(job_key, proc.pid, f"Intrinsic calibration ({cam_id})")
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
    return redirect(url_for('camera_detail', cam_id=cam_id))

@app.route('/api/camera/<cam_id>/status')
def camera_status(cam_id):
    """API endpoint to check if any background jobs are running for a camera."""
    status_data = {'jobs': {}}
    
    # Check intrinsic calibration job
    job_key = f"intrinsic_{cam_id}"
    if job_key in background_jobs:
        job = background_jobs[job_key]
        if job['status'] == 'running':
            poll = job['process'].poll()
            if poll is not None:
                job['status'] = 'finished' if poll == 0 else 'failed'
        
        # Read last few lines of log
        logs = ""
        if os.path.exists(job['log_file']):
            try:
                with open(job['log_file'], 'r', encoding='utf-8', errors='replace') as f:
                    logs = f.readlines()[-10:]
            except Exception:
                logs = []
        
        status_data['jobs']['intrinsic'] = {
            'status': job['status'],
            'logs': "".join(logs)
        }
        
    # Check for validation image
    val_image = OUTPUT_DIR / f"intrinsic_check_{cam_id}.jpg"
    if val_image.exists():
        status_data['intrinsic_image'] = f"/output/intrinsic_check_{cam_id}.jpg"

    return jsonify(status_data)

@app.route('/api/cancel_job/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """API endpoint to cancel a running background job.
    
    Works across gunicorn workers by looking up the PID from a shared
    on-disk registry (JOBS_DIR) when the in-memory dict doesn't have it.
    """
    pid = None

    # Try in-memory first (same worker that started the job)
    if job_id in background_jobs:
        job = background_jobs[job_id]
        if job['status'] == 'running':
            pid = job['process'].pid

    # Fallback: disk-based PID registry (works across workers)
    if pid is None:
        disk_job = _load_job_pid(job_id)
        if disk_job:
            pid = disk_job.get("pid")

    if pid is None:
        return jsonify({'success': False, 'error': 'Job not found or not running.'})

    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(3)
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

    # Clean up
    if job_id in background_jobs:
        background_jobs[job_id]['status'] = 'cancelled'
    _remove_job_pid(job_id)
    return jsonify({'success': True, 'message': f'Job {job_id} cancelled.'})

@app.route('/output/<filename>')
def serve_output(filename):
    """Serve files from the output directory (like validation images)."""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/results')
def results():
    """List generated CSV and MP4 files from the output directory."""
    files = []
    if OUTPUT_DIR.exists():
        for file in OUTPUT_DIR.iterdir():
            if file.is_file() and file.suffix in ['.csv', '.mp4']:
                files.append({
                    'name': file.name,
                    'size': f"{file.stat().st_size / (1024*1024):.2f} MB",
                    'type': 'Video' if file.suffix == '.mp4' else 'Data'
                })
    return render_template('results.html', files=files)


# ------------------------------------------------------------------
#  Geo-Transform: add lat/lng to fused crossing CSVs
# ------------------------------------------------------------------

def _list_fused_csvs() -> list[dict]:
    """Return fused_crossings_*.csv files sorted newest-first."""
    out = []
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.glob("fused_crossings*.csv"), reverse=True):
            if "_lat_long" in f.name:
                continue
            out.append({
                "name": f.name,
                "size": f"{f.stat().st_size / (1024*1024):.1f} MB",
            })
    return out


@app.route('/geo-transform', methods=['GET'])
def geo_transform_page():
    fused_files = _list_fused_csvs()
    form_data = {
        "selected_file": "",
        "points": [
            {"floor_x": "", "floor_y": "", "lat": "", "lng": ""},
            {"floor_x": "", "floor_y": "", "lat": "", "lng": ""},
            {"floor_x": "", "floor_y": "", "lat": "", "lng": ""},
        ],
    }
    return render_template(
        'geo_transform.html',
        fused_files=fused_files,
        result=None,
        form_data=form_data,
    )


@app.route('/api/geo-transform', methods=['POST'])
def api_geo_transform():
    """Run geo-transform as a background subprocess, return immediately."""
    import sys as _sys
    data = request.get_json(force=True)
    selected = data.get("fused_file", "")
    points = data.get("points", [])

    if not selected:
        return jsonify({"ok": False, "error": "Please select a fused crossings CSV."})

    valid_pts = []
    for p in points:
        try:
            fx = float(p.get("floor_x", ""))
            fy = float(p.get("floor_y", ""))
            la = float(p.get("lat", ""))
            ln = float(p.get("lng", ""))
            valid_pts.append({"floor_x": fx, "floor_y": fy, "lat": la, "lng": ln})
        except (ValueError, TypeError):
            pass

    if len(valid_pts) < 3:
        return jsonify({"ok": False, "error": f"Need at least 3 valid reference points, got {len(valid_pts)}."})

    input_csv = str(OUTPUT_DIR / selected)
    if not os.path.isfile(input_csv):
        return jsonify({"ok": False, "error": f"File not found: {selected}"})

    stem = selected.replace(".csv", "")
    output_csv = str(OUTPUT_DIR / f"{stem}_lat_long.csv")
    status_file = str(OUTPUT_DIR / ".geo_transform_status.json")

    # Write status "running"
    with open(status_file, "w") as f:
        json.dump({"status": "running", "file": selected}, f)

    # Build a small inline Python script to run in background
    pts_json = json.dumps(valid_pts)
    python_exec = _sys.executable
    script = f"""
import sys, json, os
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from pipeline.geo_transform import GeoRefPoint, run_geo_transform
pts = [GeoRefPoint(**p) for p in json.loads({pts_json!r})]
result = run_geo_transform(pts, {input_csv!r}, {output_csv!r})
if result["ok"]:
    result["output_name"] = os.path.basename({output_csv!r})
with open({status_file!r}, "w") as f:
    json.dump(result, f)
"""
    proc = subprocess.Popen(
        [python_exec, "-u", "-c", script],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return jsonify({"ok": True, "status": "started", "pid": proc.pid})


@app.route('/api/geo-transform/status')
def api_geo_transform_status():
    """Poll the background geo-transform job status."""
    status_file = OUTPUT_DIR / ".geo_transform_status.json"
    if not status_file.exists():
        return jsonify({"status": "idle"})
    try:
        with open(status_file) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception:
        return jsonify({"status": "idle"})

@app.route('/download/<filename>')
def download_file(filename):
    is_video = filename.endswith('.mp4')
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=not is_video)

if __name__ == '__main__':
    # Run the app locally
    app.run(debug=True, host='0.0.0.0', port=5001)
