# AWS Setup Guide — PSP Fraud Detection PoC

All resources are created in **`eu-west-2` (London)** for FCA data residency
and tagged **`Project=fraud-detection-poc`**. Cloud usage **incurs cost** —
read the README cost warning and complete the billing alerts step.

There are **four human STOP points**. The automated scripts in `scripts/aws/`
must not be run until the STOP points they depend on are complete.

---

## Step 1 — Root account security  (HUMAN REQUIRED — STOP POINT 1)

- Enable **MFA on the AWS root account**.
- Never use the root account for day-to-day work.
- Instructions: aws.amazon.com/console → account menu (top right) →
  **Security credentials** → **Multi-factor authentication (MFA)** → assign a
  device.

➡ **STOP 1: do not proceed until root MFA is enabled.**

## Step 2 — Create IAM user  (HUMAN REQUIRED — STOP POINT 2)

- Go to the **IAM console** → Users → Create user.
- User name: **`fraud-detection-dev`**.
- Attach **only** these policies:
  - `AmazonS3FullAccess`
  - `AmazonEC2FullAccess`
  - `AmazonSageMakerFullAccess`
- Create **access keys** (Application running outside AWS).
- Save them to `~/.aws/credentials` (see Step 4). **Never commit this file** —
  `.aws/` and `aws_credentials` are gitignored.

➡ **STOP 2: do not proceed until the IAM user + access keys exist.**

## Step 3 — Set billing alerts  (HUMAN REQUIRED — STOP POINT 3)

- AWS **Billing** console → **Billing preferences** → enable
  **Receive billing alerts**.
- Create a CloudWatch billing alarm at **£10/month**.
- Create a second alarm at **£25/month**.

➡ **STOP 3: do not run any cost-incurring script until both alerts exist.**

## Step 4 — AWS CLI setup  (AUTONOMOUS after STOP 2)

```bash
pip install awscli
aws configure
# AWS Access Key ID:     <from Step 2>
# AWS Secret Access Key: <from Step 2>
# Default region name:   eu-west-2
# Default output format:  json
```

This writes `~/.aws/credentials` and `~/.aws/config`. These are **never**
committed.

## Step 5 — Create S3 bucket  (AUTONOMOUS)

Run `python scripts/aws/setup_s3.py`. It creates:

- Bucket: **`fraud-detection-poc-<random 6 digits>`**
- Region: **`eu-west-2`** (London — FCA data residency)
- **Block all public access: YES**
- **Versioning: enabled**
- **Encryption: SSE-S3**

---

## STOP POINTS SUMMARY

| # | Action (human) | Blocks |
|---|---|---|
| STOP 1 | Enable MFA on AWS root account | everything |
| STOP 2 | Create IAM user `fraud-detection-dev` + access keys | Steps 4–5, all `scripts/aws/` |
| STOP 3 | Set £10 and £25/month billing alerts | any cost-incurring script (EC2/SageMaker) |
| STOP 4 | GitHub repo + remote (`git remote add origin …`) | final push (A10) — **already done** for this repo |

## Automated scripts (created, run only after the relevant STOP points)

| Script | Purpose | Cost |
|---|---|---|
| `scripts/aws/setup_s3.py` | Create private, versioned, encrypted bucket + folders | £0 create; ~£0.023/GB/mo |
| `scripts/aws/upload_data.py` | Upload `data/raw/*.csv` to S3 (checksum-verified) | ~£0.03/mo storage of ~1.3 GB |
| `scripts/aws/launch_ec2.py` | Launch t3.large (Amazon Linux 2023), auto-bootstrap | **~£0.08/hour running** |
| `scripts/aws/run_training_on_ec2.py` | SSH + run full training pipeline, upload artefacts | EC2 hours |
| `scripts/aws/stop_ec2.py` | Stop the tagged instance (confirm first) | stops the bill |

**Always run `stop_ec2.py` when training finishes — never leave an instance
running overnight.**
