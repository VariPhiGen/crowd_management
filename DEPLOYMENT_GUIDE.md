# Crimenabi — Deployment & SSH Guide

**Project**: Crimenabi Crowd Analysis  
**Infrastructure**: AWS ap-northeast-1 (Tokyo)  
**Last updated**: 2026-03-17

---

## 1. What You Need First

Before you start, make sure you have these three things:

| Item | Where to get it |
|---|---|
| `crimenabi-deploy-key.pem` | Shared by Nitish (see Section 2) |
| AWS credentials JSON | `credentials/variphi-credentials.json` in the repo |
| Python 3 + boto3 | `pip install boto3` |

---

## 2. SSH Key Setup

The private key `crimenabi-deploy-key.pem` must be on your machine before you can SSH in.

**Step 1 — Save the key file**

Nitish will send you the file `crimenabi-deploy-key.pem`.  
Save it to your SSH directory:

```bash
# macOS / Linux
mv ~/Downloads/crimenabi-deploy-key.pem ~/.ssh/crimenabi-deploy-key.pem

# Fix permissions — SSH will refuse the key if permissions are too open
chmod 400 ~/.ssh/crimenabi-deploy-key.pem
```

**Step 2 — Verify the key works**

```bash
ssh -o StrictHostKeyChecking=no \
    -i ~/.ssh/crimenabi-deploy-key.pem \
    ubuntu@57.180.43.163 \
    "/bin/echo Connection OK"
```

Expected output: `Connection OK`

---

## 3. Instance Details

| Field | Value |
|---|---|
| Instance ID | `i-0356339818eb82519` |
| Public IP | `57.180.43.163` *(changes on stop/start — see note below)* |
| Type | g5.2xlarge (NVIDIA A10G, 8 vCPU, 32 GB RAM) |
| Region | ap-northeast-1 (Tokyo) |
| OS | Ubuntu 22.04 |
| SSH User | `ubuntu` |
| Key pair | `crimenabi-deploy-key` |

> ⚠️ **IP changes every time the instance is started.**  
> Always run `python3 deploy/manage_instance.py status` to get the current IP.

---

## 4. Direct SSH Commands

### Connect to the server

```bash
ssh -i ~/.ssh/crimenabi-deploy-key.pem ubuntu@57.180.43.163
```

### Run a single command without interactive session

```bash
ssh -i ~/.ssh/crimenabi-deploy-key.pem ubuntu@57.180.43.163 \
    "/bin/bash -c 'export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin; YOUR_COMMAND'"
```

### Copy a file to the server (SCP)

```bash
scp -i ~/.ssh/crimenabi-deploy-key.pem \
    ./local_file.txt \
    ubuntu@57.180.43.163:/home/ubuntu/crowd_management/
```

### Copy a file from the server (SCP)

```bash
scp -i ~/.ssh/crimenabi-deploy-key.pem \
    ubuntu@57.180.43.163:/home/ubuntu/crowd_management/output/cam_1_crossings.csv \
    ./
```

---

## 5. Managing the Instance (Start / Stop)

Clone the repo first:

```bash
git clone https://github.com/VariPhiGen/crowd_management.git
cd crowd_management
pip install boto3
```

All instance commands:

```bash
# Check current state and IP
python3 deploy/manage_instance.py status

# Start the stopped instance (~60 sec)
python3 deploy/manage_instance.py start

# Stop when not in use — saves ~$1.51/hour
python3 deploy/manage_instance.py stop

# Open SSH session directly
python3 deploy/manage_instance.py ssh

# Run the full pipeline (start → git pull → YOLO → upload results → stop)
python3 deploy/manage_instance.py run

# Run pipeline for a single camera only
python3 deploy/manage_instance.py run --camera cam_1

# PERMANENT delete (only when project is done)
python3 deploy/manage_instance.py terminate
```

> 💡 After `stop`, compute billing stops immediately.  
> EBS storage (~500 GB) costs ~$48/month even when stopped.

---

## 6. Web Dashboard

The Flask dashboard runs as a persistent service on the instance.  
It starts automatically when the instance boots.

| Field | Value |
|---|---|
| URL | http://57.180.43.163 *(update IP after each start)* |
| Username | `admin` |
| Password | `yxBz0nKZpUJO9HMF` |

> ⚠️ The URL changes each time the instance is stopped and restarted  
> (public IP is re-assigned). Run `status` to get the new IP.

---

## 7. Useful On-Server Commands

Once SSH'd in, the venv is auto-activated and you're in the project directory.

```bash
# Check dashboard service status
sudo systemctl status crimenabi

# Restart dashboard (after a code change)
sudo systemctl restart crimenabi

# View live dashboard logs
tail -f /var/log/crimenabi/dashboard.log

# View Gunicorn error logs
tail -f /var/log/crimenabi/gunicorn-error.log

# Run the pipeline manually
python main.py --process

# Run single camera
python main.py --process-camera cam_1

# Check GPU
nvidia-smi

# Update code
cd /home/ubuntu/crowd_management
git pull origin main
sudo systemctl restart crimenabi
```

---

## 8. Project Structure on Server

```
/home/ubuntu/crowd_management/    ← main code (git repo)
  config/                         ← cameras.json, edges.json, calibration
  output/                         ← generated CSVs (crossings, tracks)
  web_ui/                         ← Flask dashboard
  deploy/                         ← deployment scripts
  main.py                         ← CLI entry point

/opt/crimenabi_venv/              ← Python virtual environment
/var/log/crimenabi/               ← all logs
  dashboard.log
  gunicorn-error.log
  gunicorn-access.log

/var/log/crimenabi_setup.log      ← one-time setup log (CUDA install etc.)
```

---

## 9. S3 Bucket

Input videos and output results are stored in S3.

| Field | Value |
|---|---|
| Bucket | `crimenabi-data-variphi` |
| Region | ap-northeast-1 (Tokyo) |
| Input videos | `s3://crimenabi-data-variphi/cam_1/`, `cam_2/` |
| Pipeline results | `s3://crimenabi-data-variphi/output/YYYYMMDD-HHMM/` |

```bash
# List bucket contents (run from EC2 — no credentials needed, uses IAM role)
aws s3 ls s3://crimenabi-data-variphi/ --region ap-northeast-1

# Download results locally (run from EC2 or with variphi credentials)
aws s3 sync s3://crimenabi-data-variphi/output/ ./output/ --region ap-northeast-1
```

---

## 10. Cost Reminder

| State | Cost |
|---|---|
| Running (g5.2xlarge) | ~$1.51 / hour |
| Stopped (EBS only) | ~$48 / month |
| Terminated | $0 |

**Always stop the instance when not in use:**

```bash
python3 deploy/manage_instance.py stop
```

---

## 11. Troubleshooting

**SSH Permission denied**
```
@ERROR: UNPROTECTED PRIVATE KEY FILE!
```
Fix: `chmod 400 ~/.ssh/crimenabi-deploy-key.pem`

---

**Commands not found (ls, git, etc.) in SSH**
```bash
# Always prefix commands with PATH in non-interactive SSH:
ssh -i ~/.ssh/crimenabi-deploy-key.pem ubuntu@57.180.43.163 \
    "/bin/bash -l -c 'your command'"
```

---

**Dashboard not loading after instance start**
```bash
# SSH in and restart the service
ssh -i ~/.ssh/crimenabi-deploy-key.pem ubuntu@57.180.43.163
sudo systemctl restart crimenabi nginx
```

---

**Instance won't start (InsufficientInstanceCapacity)**
```
No g5.2xlarge capacity in this AZ.
```
Fix: Wait 10–30 min and retry. If persistent, terminate and re-launch:
```bash
python3 deploy/manage_instance.py terminate
python3 deploy/manage_instance.py launch   # tries all 3 AZs
```

---

## 12. Contact

For credentials, key file, or access issues contact:  
**Nitish Mishra** — project owner

For AWS infrastructure issues:  
**Ryushi Shiohama** — IAM / account owner
