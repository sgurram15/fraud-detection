# Fraud Detection Project — Autonomous Build Mission

## How to use this file
This is a complete task list for Claude Code agent mode.
Work through each phase in order.
At STOP points — pause and wait for the human.
Log every action taken in MISSION_LOG.md as you go.
If you hit an error — try to fix it twice, then log it and move to the next task.

---

## Phase A — GitHub Preparation (FULLY AUTONOMOUS)

### A1 — Audit and fix .gitignore
Ensure the following are in .gitignore and NOT tracked:
- kaggle.json
- .env
- *.pkl (model files)
- data/raw/*.csv
- mlruns/
- mlflow.db
- __pycache__/
- *.pyc
- .ipynb_checkpoints/
- aws_credentials (any file with this name)

Run: git status
Check nothing sensitive is staged or tracked.
If any sensitive file IS tracked, remove it with git rm --cached.

### A2 — Create comprehensive README.md
Replace existing README with a complete version containing:

Project title: PSP Real-Time Fraud Detection — Proof of Concept

Sections:
1. What this is (2 paragraphs — what the system does and why it matters for FCA-regulated PSPs)
2. Architecture overview (reference docs/production_architecture.md)
3. Local setup instructions (step by step, Windows and Mac)
4. AWS setup instructions (reference AWS_SETUP.md which we will create)
5. Running the pipeline (in order: data download, features, training, API, streaming)
6. Running the tests
7. Known limitations and model card reference
8. Cost warning (note which steps incur AWS costs)

### A3 — Create docs/AWS_SETUP.md
A step by step AWS setup guide containing:

Step 1 — Root account security (HUMAN REQUIRED — STOP POINT 1)
- Enable MFA on root account
- Never use root account for day to day work
- Instructions: aws.amazon.com/console → account menu → security credentials → MFA

Step 2 — Create IAM user (HUMAN REQUIRED — STOP POINT 2)
- Go to IAM console
- Create user called: fraud-detection-dev
- Attach these policies only:
  - AmazonS3FullAccess
  - AmazonEC2FullAccess
  - AmazonSageMakerFullAccess
- Generate access keys
- Save to ~/.aws/credentials (never commit this file)

Step 3 — Set billing alert (HUMAN REQUIRED — STOP POINT 3)
- Go to AWS Billing console
- Create a billing alert at £10/month
- Create a second alert at £25/month
- Enable in: Billing preferences → Receive billing alerts

Step 4 — AWS CLI setup (AUTONOMOUS after STOP POINT 2)
- Run: pip install awscli
- Run: aws configure
- Enter: Access Key ID, Secret Access Key, region (eu-west-2), output format (json)

Step 5 — Create S3 bucket (AUTONOMOUS)
- Bucket name: fraud-detection-poc-[random 6 digit number]
- Region: eu-west-2 (London — FCA data residency)
- Block all public access: YES
- Versioning: enabled
- Encryption: SSE-S3

### A4 — Create scripts/aws/setup_s3.py
A script that:
- Creates the S3 bucket with correct settings
- Creates these folders inside it:
  data/raw/
  data/processed/
  models/saved/
  mlruns/
  docs/
- Verifies bucket is private
- Prints confirmation of each step
- Costs: £0 to create, £0.023/GB/month to store

### A5 — Create scripts/aws/upload_data.py
A script that:
- Uploads data/raw/*.csv to s3://[bucket]/data/raw/
- Shows progress bar for large files
- Verifies upload with checksum
- Prints total data uploaded and estimated monthly S3 cost

### A6 — Create scripts/aws/launch_ec2.py
A script that:
- Launches a t3.large EC2 instance (2 vCPU, 8GB RAM)
- Uses Amazon Linux 2023 AMI
- In eu-west-2 region
- With a startup script that automatically:
  - Installs Python 3.11
  - Installs git
  - Clones the GitHub repository
  - Installs all requirements
  - Configures AWS CLI with instance role
  - Prints READY when complete
- Tags the instance: Name=fraud-detection-training
- Prints the instance ID and public IP when launched
- Prints the hourly cost: approximately £0.08/hour
- IMPORTANT: Prints a warning: REMEMBER TO STOP THIS INSTANCE WHEN DONE

### A7 — Create scripts/aws/stop_ec2.py
A script that:
- Lists all running EC2 instances tagged fraud-detection-training
- Asks for confirmation before stopping
- Stops the instance
- Confirms it is stopped
- Prints money saved by stopping

### A8 — Update all data paths
Search all Python files in /src for hardcoded local data paths.
For each one add logic:
- If environment variable USE_S3=true, read from S3
- Otherwise read from local path
- Use a config file at /src/config.py to centralise all paths

Create /src/config.py:
- LOCAL_DATA_PATH = "data/raw/"
- S3_BUCKET = os.getenv("S3_BUCKET", "")
- USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
- MODEL_PATH = "src/models/saved/" if not USE_S3 else f"s3://{S3_BUCKET}/models/saved/"

### A9 — Create EC2 training script
Create scripts/aws/run_training_on_ec2.py that:
- SSHs into the running EC2 instance
- Runs the full training pipeline in order:
  1. python src/features/build_features.py
  2. python src/features/handle_imbalance.py
  3. python src/models/train_baseline.py
  4. python src/models/tune_model.py
  5. python src/models/validate_model.py
- Streams output back to local terminal in real time
- On completion uploads model artifacts to S3
- Prints TRAINING COMPLETE with final metrics summary

### A10 — Final GitHub push
- Run: git add .
- Run: git status (verify nothing sensitive included)
- Run: git commit -m "feat: complete fraud detection POC with AWS deployment scripts"
- Instructions for push will require GitHub remote to be set (STOP POINT 4 if not set)

---

## Phase B — STOP POINTS SUMMARY
These require human action before Claude Code can continue:

STOP 1: Enable MFA on AWS root account
STOP 2: Create IAM user and generate access keys
STOP 3: Set billing alerts
STOP 4: Create GitHub repository and add remote if not already done
         Run: git remote add origin https://github.com/[your-username]/fraud-detection.git

---

## MISSION_LOG.md
Claude Code must create and maintain MISSION_LOG.md.
After every completed task write one line:
[timestamp] [task ID] [DONE/FAILED] [one line description of what happened]

Example:
2026-05-19 14:23 A1 DONE .gitignore updated, 3 sensitive patterns added, no tracked sensitive files found
2026-05-19 14:24 A2 DONE README.md created with 8 sections
2026-05-19 14:31 A3 FAILED boto3 not installed, installed automatically, retried, DONE

---

## IMPORTANT RULES FOR AGENT MODE
1. Never commit a file containing credentials, API keys, or passwords
2. Never make an S3 bucket public
3. Always stop EC2 instances after training — never leave running overnight
4. If unsure about a destructive action — log it and wait for human
5. All AWS resources must be tagged: Project=fraud-detection-poc
6. Stay in eu-west-2 region for all AWS resources (FCA data residency)
