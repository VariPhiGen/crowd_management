from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, jsonify, send_file
import os
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
CONFIG_DIR = PROJECT_ROOT / "config"
CAMERAS_CFG = CONFIG_DIR / "cameras.json"
OUTPUT_DIR = PROJECT_ROOT / "output"

# In-memory store for background job statuses
background_jobs = {}

# ADMIN CREDENTIALS
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "password123")

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
    cameras = get_cameras()
    message = None
    error = None

    if request.method == 'POST':
        action = request.form.get('pipeline_action')
        try:
            env = os.environ.copy()
            # Always use the venv Python that runs this Flask app so torch/CUDA is available
            import sys as _sys
            python_exec = _sys.executable

            if action == 'process':
                # Read performance settings from form (with safe defaults)
                workers      = int(request.form.get('workers', 4))
                frame_stride = int(request.form.get('frame_stride', 2))
                ocr_interval = int(request.form.get('ocr_interval', 0))

                cmd = [
                    python_exec, "main.py", "--process",
                    "--workers",      str(workers),
                    "--frame-stride", str(frame_stride),
                    "--ocr-interval", str(ocr_interval),
                ]

                OUTPUT_DIR.mkdir(exist_ok=True)
                log_file = OUTPUT_DIR / "pipeline_process.log"
                log_f = open(log_file, 'w')

                proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                        stdout=log_f, stderr=subprocess.STDOUT)
                background_jobs["pipeline_main"] = {
                    'process':    proc,
                    'log_file':   str(log_file),
                    'start_time': time.time(),
                    'status':     'running',
                    'name':       f'Main Pipeline (workers={workers}, stride={frame_stride})',
                }
                message = (
                    f"Full pipeline started — {workers} cameras in parallel, "
                    f"frame stride {frame_stride}, OCR every "
                    f"{'auto (1 s)' if ocr_interval == 0 else str(ocr_interval) + ' frames'}. "
                    "Check logs below."
                )

            elif action == 'fuse':
                cmd = [python_exec, "main.py", "--fuse-only"]

                log_file = OUTPUT_DIR / "pipeline_fuse.log"
                log_f = open(log_file, 'w')

                proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env,
                                        stdout=log_f, stderr=subprocess.STDOUT)
                background_jobs["pipeline_fuse"] = {
                    'process':    proc,
                    'log_file':   str(log_file),
                    'start_time': time.time(),
                    'status':     'running',
                    'name':       'Fusion Stage',
                }
                message = "Fusion stage started in the background."

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
                                            stdout=log_f, stderr=subprocess.STDOUT)
                    background_jobs["pipeline_visualize"] = {
                        'process':    proc,
                        'log_file':   str(log_file),
                        'start_time': time.time(),
                        'status':     'running',
                        'name':       'Visualization Generation',
                    }
                    message = "Visualization generation started. MP4 will appear in Results when finished."

        except Exception as e:
            error = f"Failed to start pipeline process: {e}"

    # Get active global pipeline jobs for dashboard
    active_jobs = {}
    for j_key, j_data in background_jobs.items():
        if j_key.startswith("pipeline_"):
            if j_data['status'] == 'running':
                poll = j_data['process'].poll()
                if poll is not None:
                    j_data['status'] = 'finished' if poll == 0 else 'failed'
            # read tail of log
            logs = ""
            if os.path.exists(j_data['log_file']):
                try:
                    with open(j_data['log_file'], 'r', encoding='utf-8', errors='replace') as f:
                        logs = f.readlines()[-5:]
                except Exception:
                    logs = []
            
            active_jobs[j_key] = {
                'name': j_data['name'],
                'status': j_data['status'],
                'logs': "".join(logs)
            }

    return render_template('dashboard.html', cameras=cameras, message=message, error=error, active_jobs=active_jobs)

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
                drive_url = request.form.get('drive_url', '').strip()
                if not drive_url:
                    error = "Please provide a Google Drive folder URL."
                else:
                    try:
                        import gdown
                        import tempfile
                        import shutil
                        
                        # Create a temp folder for downloading
                        tmp_dir = tempfile.mkdtemp(prefix='gdrive_import_')
                        
                        # Determine if it's a folder or a single file URL
                        if 'drive.google.com/drive' in drive_url and 'folders' in drive_url:
                            # Download all files in the folder
                            gdown.download_folder(drive_url, output=tmp_dir, quiet=True, use_cookies=False)
                        else:
                            # Single file download
                            gdown.download(drive_url, output=tmp_dir + '/', quiet=True, fuzzy=True)
                        
                        # Upload each video file found to S3
                        VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.ts'}
                        uploaded = []
                        failed = []
                        
                        # Recursively find all video files in tmp_dir
                        for root, dirs, files_found in os.walk(tmp_dir):
                            for fname in files_found:
                                if Path(fname).suffix.lower() in VIDEO_EXTS:
                                    local_path = os.path.join(root, fname)
                                    s3_key = f"{prefix}{fname}"
                                    try:
                                        with open(local_path, 'rb') as fh:
                                            s3.upload_fileobj(fh, bucket, s3_key)
                                        uploaded.append(fname)
                                    except Exception as upload_err:
                                        failed.append(f"{fname}: {upload_err}")
                        
                        # Cleanup temp directory
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        
                        if uploaded:
                            success = f"Successfully imported {len(uploaded)} video(s) to S3: {', '.join(uploaded)}"
                        elif not failed:
                            error = "No video files were found in the provided Google Drive folder."
                        
                        if failed:
                            error = (error or '') + f" Failed: {'; '.join(failed)}"
                    
                    except ImportError:
                        error = "gdown is not installed. Run: pip3 install gdown"
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
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log_f, stderr=subprocess.STDOUT)
        background_jobs[f"intrinsic_{cam_id}"] = {
            'process': proc,
            'log_file': str(log_file),
            'start_time': time.time(),
            'status': 'running'
        }
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
    """API endpoint to cancel a running background job."""
    if job_id in background_jobs:
        job = background_jobs[job_id]
        if job['status'] == 'running':
            proc = job['process']
            try:
                # Try terminate first
                proc.terminate()
                job['status'] = 'cancelled'
                return jsonify({'success': True, 'message': f'Job {job_id} cancelled.'})
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    return jsonify({'success': False, 'error': 'Job not found or not running.'})

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

@app.route('/download/<filename>')
def download_file(filename):
    is_video = filename.endswith('.mp4')
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=not is_video)

if __name__ == '__main__':
    # Run the app locally
    app.run(debug=True, host='0.0.0.0', port=5001)
