"""
deploy/manage_instance.py
==========================
Manages the persistent Crimenabi EC2 g5.2xlarge instance lifecycle.

Commands
--------
  launch      First-time only: launch instance, install CUDA + deps, clone repo.
              Saves the instance ID to deploy/.instance_id for future use.

  start       Start the stopped instance (fast, ~60 sec).

  stop        Stop the running instance to avoid compute charges.
              EBS volume is preserved — next start resumes instantly.

  status      Show current instance state, IP, and uptime.

  run         start → wait → git pull → run pipeline → upload results → stop.
              Fully automated end-to-end processing run.

  ssh         Print the SSH command to connect manually.

  terminate   PERMANENTLY delete the instance and EBS volume.
              Use only when project is completely done.

Usage
-----
  python3 deploy/manage_instance.py launch
  python3 deploy/manage_instance.py start
  python3 deploy/manage_instance.py run
  python3 deploy/manage_instance.py run --camera cam_1
  python3 deploy/manage_instance.py stop
  python3 deploy/manage_instance.py status
  python3 deploy/manage_instance.py ssh
  python3 deploy/manage_instance.py terminate
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# ── Config ────────────────────────────────────────────────────────────────────

REGION         = "ap-northeast-1"
AMI_ID         = "ami-0d52744d6551d851e"   # Ubuntu 22.04
INSTANCE_TYPE  = "g5.2xlarge"
KEY_NAME       = "crimenabi-deploy-key"
KEY_PATH       = Path.home() / ".ssh" / "crimenabi-deploy-key.pem"
SECURITY_GROUP = "sg-0592d60a80c313f1c"
IAM_PROFILE    = "variphi-ec2-analysis-profile"
S3_BUCKET      = "crimenabi-data-variphi"
GITHUB_REPO    = "https://github.com/VariPhiGen/crowd_management.git"
SSH_USER       = "ubuntu"
WORKDIR        = "/home/ubuntu/crowd_management"
VENV           = "/opt/crimenabi_venv"
# Try all AZs in order until one has capacity
SUBNETS = [
    ("subnet-0402a3003ec0073f2", "ap-northeast-1a"),
    ("subnet-08451d20d1f76dd4a", "ap-northeast-1c"),
    ("subnet-0c2230dc6db0d50e8", "ap-northeast-1d"),
]
SUBNET_ID      = SUBNETS[0][0]   # default for first launch

# Instance ID is persisted here so start/stop work without re-specifying
INSTANCE_ID_FILE = Path(__file__).parent / ".instance_id"

REQUIRED_TAGS = [
    {"Key": "Company",     "Value": "Variphi"},
    {"Key": "Project",     "Value": "Crimenabi"},
    {"Key": "Name",        "Value": "crimenabi-pipeline"},
    {"Key": "Environment", "Value": "Production"},
]

# ── Credentials ───────────────────────────────────────────────────────────────

def _creds() -> dict:
    cred_path = Path(__file__).parent.parent / "credentials" / "variphi-credentials.json"
    with open(cred_path) as f:
        raw = json.load(f)
    key = raw.get("AccessKey", raw)
    return {
        "aws_access_key_id":     key["AccessKeyId"],
        "aws_secret_access_key": key["SecretAccessKey"],
        "region_name":           REGION,
    }

def _ec2():
    return boto3.client("ec2", **_creds())


# ── Instance ID helpers ───────────────────────────────────────────────────────

def _save_instance_id(iid: str):
    INSTANCE_ID_FILE.write_text(iid.strip())
    print(f"  Instance ID saved to {INSTANCE_ID_FILE}")

def _load_instance_id() -> str:
    if not INSTANCE_ID_FILE.exists():
        sys.exit(
            "  ❌  No instance ID found.\n"
            "  Run: python3 deploy/manage_instance.py launch"
        )
    return INSTANCE_ID_FILE.read_text().strip()


# ── State helpers ─────────────────────────────────────────────────────────────

def _get_instance_info(ec2, iid: str) -> dict:
    resp = ec2.describe_instances(InstanceIds=[iid])
    return resp["Reservations"][0]["Instances"][0]

def _wait_for_state(ec2, iid: str, target: str, timeout: int = 300):
    print(f"  Waiting for instance to be '{target}'", end="", flush=True)
    start = time.time()
    while True:
        info = _get_instance_info(ec2, iid)
        state = info["State"]["Name"]
        if state == target:
            print(f"  ✅  {target}")
            return info
        if time.time() - start > timeout:
            sys.exit(f"\n  ❌  Timed out waiting for state '{target}' (current: {state})")
        print(".", end="", flush=True)
        time.sleep(8)

def _wait_for_ssh(ip: str, timeout: int = 180):
    print(f"  Waiting for SSH on {ip}", end="", flush=True)
    start = time.time()
    while True:
        ret = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no",
             "-o", "ConnectTimeout=5",
             "-o", "BatchMode=yes",
             "-i", str(KEY_PATH),
             f"{SSH_USER}@{ip}", "echo ok"],
            capture_output=True, text=True
        )
        if ret.returncode == 0:
            print("  ✅  SSH ready")
            return
        if time.time() - start > timeout:
            sys.exit(f"\n  ❌  SSH timed out on {ip}")
        print(".", end="", flush=True)
        time.sleep(8)


# ── Commands ─────────────────────────────────────────────────────────────────

def cmd_launch():
    """First-time launch: provision instance + full environment setup."""
    ec2 = _ec2()

    # Check if instance already exists
    if INSTANCE_ID_FILE.exists():
        iid = _load_instance_id()
        info = _get_instance_info(ec2, iid)
        state = info["State"]["Name"]
        if state not in ("terminated", "shutting-down"):
            sys.exit(
                f"  ❌  Instance {iid} already exists (state: {state}).\n"
                f"  Use 'start' to resume, or 'terminate' first to create a new one."
            )

    # boto3 auto-base64-encodes string UserData — pass raw script directly
    setup_script = _build_setup_user_data()
    encoded_ud   = setup_script

    print("\n" + "=" * 60)
    print("  Launching g5.2xlarge (first-time setup ~15 min)")
    print("=" * 60)

    # Try each AZ until one has capacity
    resp = None
    for subnet, az in SUBNETS:
        try:
            print(f"  Trying {az} ({subnet})...")
            resp = ec2.run_instances(
                ImageId      = AMI_ID,
                InstanceType = INSTANCE_TYPE,
                MinCount     = 1,
                MaxCount     = 1,
                KeyName      = KEY_NAME,
                NetworkInterfaces=[{
                    "AssociatePublicIpAddress": True,
                    "DeviceIndex": 0,
                    "SubnetId": subnet,
                    "Groups": [SECURITY_GROUP],
                }],
                IamInstanceProfile = {"Name": IAM_PROFILE},
                BlockDeviceMappings = [{
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize":          500,
                        "VolumeType":          "gp3",
                        "Iops":                3000,
                        "Throughput":          125,
                        "DeleteOnTermination": True,
                    },
                }],
                TagSpecifications = [
                    {"ResourceType": "instance", "Tags": REQUIRED_TAGS},
                    {"ResourceType": "volume",   "Tags": REQUIRED_TAGS},
                ],
                UserData = encoded_ud,
            )
            print(f"  ✅  Capacity available in {az}")
            break
        except ClientError as e:
            if "InsufficientInstanceCapacity" in str(e):
                print(f"  ⚠️  No capacity in {az}, trying next AZ...")
                continue
            raise

    if resp is None:
        sys.exit("  ❌  No g5.2xlarge capacity in any AZ. Try again later.")

    iid = resp["Instances"][0]["InstanceId"]
    _save_instance_id(iid)

    info = _wait_for_state(ec2, iid, "running")
    ip   = info.get("PublicIpAddress", "—")

    print(f"\n  Instance ID : {iid}")
    print(f"  Public IP   : {ip}")
    print(f"\n  ⏳ Setup running in background (~15 min for CUDA + deps).")
    print(f"  Monitor: ssh -i ~/.ssh/crimenabi-key.pem ubuntu@{ip}")
    print(f"  Log:     tail -f /var/log/crimenabi_setup.log")
    print(f"\n  ⚠️  Stop the instance after setup to avoid idle charges:")
    print(f"  python3 deploy/manage_instance.py stop")
    print("=" * 60)


def cmd_start():
    ec2 = _ec2()
    iid = _load_instance_id()
    info = _get_instance_info(ec2, iid)
    state = info["State"]["Name"]

    if state == "running":
        ip = info.get("PublicIpAddress", "—")
        print(f"  ✅  Already running. IP: {ip}")
        return info

    if state not in ("stopped",):
        sys.exit(f"  ❌  Cannot start instance in state: {state}")

    print(f"  Starting instance {iid}...")
    try:
        ec2.start_instances(InstanceIds=[iid])
    except ClientError as e:
        if "InsufficientInstanceCapacity" in str(e):
            print(f"\n  ❌  No g5.2xlarge capacity available in the current AZ right now.")
            print(f"  Options:")
            print(f"    1. Wait 10–30 min and retry:  python3 deploy/manage_instance.py start")
            print(f"    2. Terminate and relaunch in a different AZ:")
            print(f"       python3 deploy/manage_instance.py terminate")
            print(f"       python3 deploy/manage_instance.py launch")
            sys.exit(1)
        raise
    info = _wait_for_state(ec2, iid, "running")
    ip   = info.get("PublicIpAddress", "—")
    print(f"  Public IP : {ip}")
    return info


def cmd_stop():
    ec2 = _ec2()
    iid = _load_instance_id()
    info = _get_instance_info(ec2, iid)
    state = info["State"]["Name"]

    if state == "stopped":
        print(f"  ✅  Already stopped. No charges running.")
        return

    if state != "running":
        sys.exit(f"  ❌  Cannot stop instance in state: {state}")

    print(f"  Stopping instance {iid}...")
    ec2.stop_instances(InstanceIds=[iid])
    _wait_for_state(ec2, iid, "stopped")
    print(f"  💰 Compute billing stopped. EBS storage ~$0.096/GB/month continues.")


def cmd_status():
    ec2 = _ec2()
    iid = _load_instance_id()
    info = _get_instance_info(ec2, iid)

    state     = info["State"]["Name"]
    ip        = info.get("PublicIpAddress", "—")
    itype     = info.get("InstanceType", "—")
    az        = info["Placement"]["AvailabilityZone"]
    launch_t  = info.get("LaunchTime", "")

    state_icon = {"running": "🟢", "stopped": "🔴", "stopping": "🟡", "pending": "🟡"}.get(state, "⚪")

    print(f"\n  Instance  : {iid}")
    print(f"  State     : {state_icon}  {state}")
    print(f"  Type      : {itype}")
    print(f"  AZ        : {az}")
    print(f"  Public IP : {ip}")
    if launch_t:
        print(f"  Launched  : {launch_t}")

    if state == "running":
        print(f"\n  SSH       : ssh -i ~/.ssh/crimenabi-key.pem ubuntu@{ip}")
        print(f"  Pipeline  : python3 deploy/manage_instance.py run")


def cmd_run(camera_arg: str = ""):
    """Start → SSH → git pull → run pipeline → upload results → stop."""
    print("\n" + "=" * 60)
    print("  Crimenabi Pipeline — Automated Run")
    print("=" * 60)

    # 1. Start
    info = cmd_start()
    ip   = info.get("PublicIpAddress", "—")

    # 2. Wait for SSH
    _wait_for_ssh(ip)

    # 3. Run pipeline via SSH
    pipeline_cmd = f"--process-camera {camera_arg}" if camera_arg else "--process"
    datestamp = datetime.utcnow().strftime("%Y%m%d-%H%M")

    remote_script = f"""
set -e
echo '--- git pull ---'
cd {WORKDIR}
git pull origin main

echo '--- activating venv ---'
source {VENV}/bin/activate

echo '--- running pipeline ---'
python main.py {pipeline_cmd} 2>&1 | tee /tmp/pipeline_run.log

echo '--- uploading results to S3 ---'
aws s3 sync {WORKDIR}/output/ s3://{S3_BUCKET}/output/{datestamp}/ \\
    --exclude 'tmp_s3/*' \\
    --region {REGION}

echo '--- done ---'
"""

    print(f"\n  Running pipeline on {ip} (camera: {camera_arg or 'all'})...")
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-i", str(KEY_PATH),
         f"{SSH_USER}@{ip}",
         remote_script],
        text=True
    )

    if result.returncode != 0:
        print(f"\n  ❌  Pipeline exited with code {result.returncode}")
        print(f"  Instance left RUNNING for inspection. Stop manually when done.")
        print(f"  SSH: ssh -i ~/.ssh/crimenabi-key.pem ubuntu@{ip}")
        sys.exit(1)

    print(f"\n  ✅  Pipeline complete.")
    print(f"  Results: s3://{S3_BUCKET}/output/{datestamp}/")

    # 4. Stop to save costs
    print("\n  Stopping instance to avoid idle charges...")
    cmd_stop()
    print("=" * 60)


def cmd_ssh():
    ec2 = _ec2()
    iid = _load_instance_id()
    info = _get_instance_info(ec2, iid)
    state = info["State"]["Name"]

    if state != "running":
        print(f"  ⚠️  Instance is '{state}'. Start it first:")
        print(f"    python3 deploy/manage_instance.py start")
        return

    ip = info.get("PublicIpAddress", "—")
    cmd = f"ssh -i ~/.ssh/crimenabi-key.pem ubuntu@{ip}"
    print(f"\n  {cmd}\n")
    # Also execute it directly
    os.execlp("ssh", "ssh", "-i", str(KEY_PATH), f"ubuntu@{ip}")


def cmd_deploy():
    """Open port 80, start instance, wait for SSH, run deploy_web.sh."""
    ec2 = _ec2()

    # ── Open port 80 in security group ────────────────────────────────────────
    print("\n[1/4] Opening port 80 in security group...")
    try:
        ec2.authorize_security_group_ingress(
            GroupId    = SECURITY_GROUP,
            IpProtocol = "tcp",
            FromPort   = 80,
            ToPort     = 80,
            CidrIp     = "0.0.0.0/0",
        )
        print("  ✓ Port 80 (HTTP) opened")
    except ClientError as e:
        if "InvalidPermission.Duplicate" in str(e):
            print("  ✓ Port 80 already open")
        else:
            raise

    # ── Start instance ────────────────────────────────────────────────────────
    print("\n[2/4] Starting instance...")
    info = cmd_start()
    ip   = info.get("PublicIpAddress", "—")

    # ── Wait for SSH ──────────────────────────────────────────────────────────
    print("\n[3/4] Waiting for SSH...")
    _wait_for_ssh(ip)

    # ── Run deploy_web.sh on EC2 ──────────────────────────────────────────────
    print("\n[4/4] Running web deployment script on EC2...")
    deploy_script = Path(__file__).parent / "deploy_web.sh"
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no",
         "-i", str(KEY_PATH),
         f"{SSH_USER}@{ip}",
         f"bash -s"],
        input=deploy_script.read_text(),
        text=True,
    )
    if result.returncode != 0:
        print(f"\n  ❌  Deployment failed (exit {result.returncode})")
        print(f"  Instance left running for inspection.")
        print(f"  SSH: ssh -i ~/.ssh/crimenabi-key.pem ubuntu@{ip}")
        sys.exit(1)

    print(f"\n  🌐  Dashboard URL: http://{ip}")
    print(f"  Instance stays RUNNING while dashboard is live.")
    print(f"  Stop with: python3 deploy/manage_instance.py stop")


def cmd_terminate():
    ec2 = _ec2()
    iid = _load_instance_id()
    info = _get_instance_info(ec2, iid)
    state = info["State"]["Name"]

    print(f"\n  ⚠️  WARNING: This will PERMANENTLY delete instance {iid} and its 500 GB EBS volume.")
    print(f"  Current state: {state}")
    confirm = input("  Type 'yes' to confirm termination: ").strip().lower()
    if confirm != "yes":
        print("  Aborted.")
        return

    ec2.terminate_instances(InstanceIds=[iid])
    _wait_for_state(ec2, iid, "terminated", timeout=600)
    INSTANCE_ID_FILE.unlink(missing_ok=True)
    print(f"  ✅  Instance {iid} terminated. All charges stopped.")


# ── User-data for first-time setup ────────────────────────────────────────────

def _build_setup_user_data() -> str:
    return f"""#!/bin/bash
exec > /var/log/crimenabi_setup.log 2>&1
echo "=== Setup started: $(date) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q build-essential git wget curl unzip python3.10 python3.10-venv \
    python3-pip libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 ffmpeg

# NVIDIA driver
apt-get install -y -q ubuntu-drivers-common
ubuntu-drivers autoinstall || true

# CUDA 12.1
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update -q
apt-get install -y -q cuda-toolkit-12-1
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> /etc/environment
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> /etc/environment

# Python venv
python3.10 -m venv {VENV}
source {VENV}/bin/activate
pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
pip install ultralytics easyocr opencv-python-headless boto3 flask pandas numpy scipy -q

# Clone repo
git clone {GITHUB_REPO} {WORKDIR}

# Convenience: activate venv on SSH login for ubuntu user
echo 'source {VENV}/bin/activate' >> /home/ubuntu/.bashrc
echo 'cd {WORKDIR}' >> /home/ubuntu/.bashrc

echo "=== Setup complete: $(date) ==="
"""


# ── CLI ───────────────────────────────────────────────────────────────────────

COMMANDS = {
    "launch":    (cmd_launch,    "First-time: provision g5.2xlarge + install all deps"),
    "start":     (cmd_start,     "Start the stopped instance"),
    "stop":      (cmd_stop,      "Stop the running instance (saves compute cost)"),
    "status":    (cmd_status,    "Show instance state, IP, SSH command"),
    "deploy":    (cmd_deploy,    "Open port 80 + install Nginx/Gunicorn + start dashboard"),
    "run":       (cmd_run,       "start → git pull → pipeline → upload results → stop"),
    "ssh":       (cmd_ssh,       "Print SSH command (or open SSH session)"),
    "terminate": (cmd_terminate, "PERMANENTLY delete instance and EBS (project done)"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crimenabi EC2 instance lifecycle manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:10s}  {v[1]}" for k, v in COMMANDS.items()),
    )
    parser.add_argument("command", choices=list(COMMANDS.keys()))
    parser.add_argument("--camera", metavar="CAM_ID",
                        help="For 'run': process single camera (e.g. cam_1). Default: all.")
    args = parser.parse_args()

    fn, _ = COMMANDS[args.command]

    if args.command == "run":
        fn(camera_arg=args.camera or "")
    else:
        fn()


