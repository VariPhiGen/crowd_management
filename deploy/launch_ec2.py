"""
deploy/launch_ec2.py
====================
Launches a g5.2xlarge EC2 instance to run the crowd_management pipeline
on the Variphi infrastructure.

Usage
-----
# Full pipeline (all cameras):
python deploy/launch_ec2.py

# Single camera:
python deploy/launch_ec2.py --camera cam_1

# Dry run (print config, don't launch):
python deploy/launch_ec2.py --dry-run

Prerequisites
-------------
1. Code + configs uploaded to S3:
     python deploy/upload_code.sh   (or run upload_code.sh directly)
2. Videos uploaded to s3://crimenabi-data-variphi/cam_1/ and cam_2/

Infrastructure
--------------
- IAM User      : variphi-ec2-manager
- Instance type : g5.2xlarge  (8 vCPU, 32 GB RAM, NVIDIA A10G 24 GB)
- AMI           : ami-0d52744d6551d851e  (Ubuntu 22.04)
- Region        : ap-northeast-1 (Tokyo)
- Key pair      : crimenabi-key
- Security group: crimenabi-sg  (sg-0592d60a80c313f1c)  — SSH only
- IAM profile   : variphi-ec2-analysis-profile  (S3 access via role)
- Tags required : Company=Variphi, Project=Crimenabi  (enforced by policy)
"""

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import boto3

# ── Configuration ─────────────────────────────────────────────────────────────

REGION          = "ap-northeast-1"
AMI_ID          = "ami-0d52744d6551d851e"   # Ubuntu 22.04
INSTANCE_TYPE   = "g5.2xlarge"
KEY_NAME        = "crimenabi-key"
SECURITY_GROUP  = "sg-0592d60a80c313f1c"    # crimenabi-sg
IAM_PROFILE     = "variphi-ec2-analysis-profile"
S3_BUCKET       = "crimenabi-data-variphi"
GITHUB_REPO     = "https://github.com/VariPhiGen/crowd_management.git"

# Default VPC subnets (public) — use ap-northeast-1a for best g5 availability
SUBNET_ID       = "subnet-0402a3003ec0073f2"  # vpc-0ef1543c66ccf1702 / 1a

# Required by IAM policy — launch will be denied without these
REQUIRED_TAGS = [
    {"Key": "Company", "Value": "Variphi"},
    {"Key": "Project", "Value": "Crimenabi"},
]

# ── Credentials (from aws_credentials.json) ───────────────────────────────────

def _load_creds() -> dict:
    cred_path = Path(__file__).parent.parent / "credentials" / "variphi-credentials.json"
    with open(cred_path) as f:
        raw = json.load(f)
    key = raw.get("AccessKey", raw)
    return {
        "aws_access_key_id":     key["AccessKeyId"],
        "aws_secret_access_key": key["SecretAccessKey"],
        "region_name":           REGION,
    }


# ── User-data loader ──────────────────────────────────────────────────────────

def _load_user_data(camera_arg: str) -> str:
    ud_path = Path(__file__).parent / "user_data.sh"
    script = ud_path.read_text()
    script = script.replace("__S3_BUCKET__",  S3_BUCKET)
    script = script.replace("__CAMERA_ARG__", camera_arg)
    return base64.b64encode(script.encode()).decode()


# ── Launch ────────────────────────────────────────────────────────────────────

def launch(camera_arg: str = "", dry_run: bool = False):
    creds   = _load_creds()
    ec2     = boto3.client("ec2", **creds)
    date_str = datetime.utcnow().strftime("%Y%m%d-%H%M")

    extra_tags = [
        {"Key": "Date",        "Value": date_str},
        {"Key": "Environment", "Value": "Production"},
        {"Key": "Name",        "Value": f"crimenabi-pipeline-{date_str}"},
    ]
    if camera_arg:
        extra_tags.append({"Key": "Camera", "Value": camera_arg.replace("--camera ", "").strip()})

    all_tags = REQUIRED_TAGS + extra_tags

    user_data = _load_user_data(camera_arg)

    launch_config = dict(
        ImageId      = AMI_ID,
        InstanceType = INSTANCE_TYPE,
        MinCount     = 1,
        MaxCount     = 1,
        KeyName      = KEY_NAME,
        NetworkInterfaces=[{
            "AssociatePublicIpAddress": True,
            "DeviceIndex": 0,
            "SubnetId": SUBNET_ID,
            "Groups": [SECURITY_GROUP],
        }],
        IamInstanceProfile = {"Name": IAM_PROFILE},
        BlockDeviceMappings = [{
            "DeviceName": "/dev/sda1",
            "Ebs": {
                "VolumeSize":           500,
                "VolumeType":           "gp3",
                "Iops":                 3000,
                "Throughput":           125,
                "DeleteOnTermination":  True,
            },
        }],
        TagSpecifications = [
            {"ResourceType": "instance", "Tags": all_tags},
            {"ResourceType": "volume",   "Tags": REQUIRED_TAGS},
        ],
        UserData = user_data,
    )

    print("\n" + "=" * 60)
    print("  Crimenabi Crowd Pipeline — EC2 Launch")
    print("=" * 60)
    print(f"  Instance type : {INSTANCE_TYPE}")
    print(f"  AMI           : {AMI_ID}")
    print(f"  Key pair      : {KEY_NAME}")
    print(f"  Security group: {SECURITY_GROUP}")
    print(f"  IAM profile   : {IAM_PROFILE}")
    print(f"  GitHub repo   : {GITHUB_REPO}")
    print(f"  S3 bucket     : s3://{S3_BUCKET}  (videos in + results out)")
    print(f"  Camera arg    : '{camera_arg}' (empty = all cameras)")
    print(f"  Tags          : {json.dumps(all_tags, indent=14)}")

    if dry_run:
        print("\n[DRY RUN] Would launch instance with above config.")
        print("=" * 60)
        return

    resp = ec2.run_instances(**launch_config)
    instance = resp["Instances"][0]
    instance_id = instance["InstanceId"]

    print(f"\n  ✅ Instance launched: {instance_id}")
    print(f"  State          : {instance['State']['Name']}")
    print(f"  Availability Z : {instance['Placement']['AvailabilityZone']}")
    print(f"\n  The instance will:")
    print(f"    1. Install CUDA + Python deps (~15 min)")
    print(f"    2. git clone {GITHUB_REPO}")
    print(f"    3. Run the pipeline (YOLO + crossing detection)")
    print(f"    4. Upload results to s3://{S3_BUCKET}/output/")
    print(f"    5. Auto-terminate")
    print(f"\n  Monitor via:")
    print(f"    aws ec2 get-console-output --instance-id {instance_id} --region {REGION}")
    print(f"    aws s3 ls s3://{S3_BUCKET}/output/ --region {REGION}")
    print("=" * 60)
    return instance_id


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Crimenabi pipeline on EC2 g5.2xlarge")
    parser.add_argument("--camera",  metavar="CAM_ID", help="Process a single camera (e.g. cam_1). Default: all cameras.")
    parser.add_argument("--dry-run", action="store_true", help="Print launch config without actually launching.")
    args = parser.parse_args()

    camera_arg = f"--process-camera {args.camera}" if args.camera else "--process"
    launch(camera_arg=camera_arg, dry_run=args.dry_run)
