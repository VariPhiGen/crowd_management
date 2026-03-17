"""
deploy/check_permissions.py
============================
Verifies all required AWS permissions for the Variphi deployment.
Run this first before launching any EC2 instances.

Usage:
    python deploy/check_permissions.py
"""

import json
import sys
from pathlib import Path

import boto3

REGION    = "ap-northeast-1"
S3_BUCKET = "crimenabi-data-variphi"
CRED_PATH = Path(__file__).parent.parent / "credentials" / "variphi-credentials.json"


def load_creds():
    with open(CRED_PATH) as f:
        raw = json.load(f)
    key = raw.get("AccessKey", raw)
    return {
        "aws_access_key_id":     key["AccessKeyId"],
        "aws_secret_access_key": key["SecretAccessKey"],
        "region_name":           REGION,
    }


def check(label, fn):
    try:
        result = fn()
        print(f"  ✅  {label}")
        return result
    except Exception as e:
        print(f"  ❌  {label}")
        print(f"       {e}")
        return None


def main():
    creds = load_creds()
    sts   = boto3.client("sts",  **creds)
    s3    = boto3.client("s3",   **creds)
    ec2   = boto3.client("ec2",  **creds)
    iam   = boto3.client("iam",  **creds)

    print("\n" + "=" * 60)
    print("  Variphi Permission Check")
    print("=" * 60)

    # ── Identity ──────────────────────────────────────────────────────────────
    print("\n[Identity]")
    identity = check(
        "sts:GetCallerIdentity",
        lambda: sts.get_caller_identity()
    )
    if identity:
        print(f"       User  : {identity['Arn']}")
        print(f"       Account: {identity['Account']}")

    # ── S3 ────────────────────────────────────────────────────────────────────
    print(f"\n[S3: s3://{S3_BUCKET}]")
    check("s3:ListBucket",
          lambda: s3.list_objects_v2(Bucket=S3_BUCKET, MaxKeys=1))
    check("s3:PutObject (write test)",
          lambda: s3.put_object(Bucket=S3_BUCKET, Key="deploy_test/.keep", Body=b""))
    check("s3:GetObject (read test)",
          lambda: s3.get_object(Bucket=S3_BUCKET, Key="deploy_test/.keep"))
    check("s3:DeleteObject (cleanup test)",
          lambda: s3.delete_object(Bucket=S3_BUCKET, Key="deploy_test/.keep"))

    # ── EC2 ───────────────────────────────────────────────────────────────────
    print("\n[EC2]")
    check("ec2:DescribeInstances",
          lambda: ec2.describe_instances(MaxResults=5))
    check("ec2:DescribeInstanceTypes (g5.2xlarge)",
          lambda: ec2.describe_instance_types(InstanceTypes=["g5.2xlarge"]))
    check("ec2:DescribeImages (AMI ami-0d52744d6551d851e)",
          lambda: ec2.describe_images(ImageIds=["ami-0d52744d6551d851e"]))
    check("ec2:DescribeKeyPairs (crimenabi-key)",
          lambda: ec2.describe_key_pairs(KeyNames=["crimenabi-key"]))
    check("ec2:DescribeSecurityGroups (crimenabi-sg)",
          lambda: ec2.describe_security_groups(GroupIds=["sg-0592d60a80c313f1c"]))
    check("ec2:DescribeSubnets",
          lambda: ec2.describe_subnets(SubnetIds=["subnet-0402a3003ec0073f2"]))

    # ── IAM PassRole ──────────────────────────────────────────────────────────
    print("\n[IAM]")
    check("iam:GetInstanceProfile (variphi-ec2-analysis-profile)",
          lambda: iam.get_instance_profile(InstanceProfileName="variphi-ec2-analysis-profile"))

    # ── Launch check (tag enforcement) ────────────────────────────────────────
    print("\n[EC2 Launch — dry run with DryRun=True]")
    def _dry_run_launch():
        try:
            ec2.run_instances(
                ImageId      = "ami-0d52744d6551d851e",
                InstanceType = "g5.2xlarge",
                MinCount     = 1,
                MaxCount     = 1,
                DryRun       = True,
                IamInstanceProfile = {"Name": "variphi-ec2-analysis-profile"},
                TagSpecifications  = [{
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Company", "Value": "Variphi"},
                        {"Key": "Project", "Value": "Crimenabi"},
                    ],
                }],
            )
        except ec2.exceptions.ClientError as e:
            if "DryRunOperation" in str(e):
                return "DryRunOperation succeeded — launch is permitted"
            raise
    check("ec2:RunInstances (dry run with required tags)", _dry_run_launch)

    print("\n" + "=" * 60)
    print("  Done. Fix any ❌ items before running launch_ec2.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
