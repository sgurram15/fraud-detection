# Mission Log

Format: `[timestamp] [task ID] [DONE/FAILED] [description]`

2026-05-19 A1 DONE .gitignore created (credentials/data/models/mlruns/generated outputs); standalone git repo initialised in fraud-detection/ (separate from parent Machine_Learning_Sample_Project repo); branch main; 2 commits; 29 tracked files, 0 sensitive/large; remote origin=github.com/sgurram15/fraud-detection; pushed. kaggle.json confirmed only at ~/.kaggle/.
2026-05-19 A2 DONE README.md replaced with 8 sections (what/why FCA, architecture ref, local setup Win+Mac, AWS ref, pipeline order, tests, limitations+model card ref, cost warning table).
2026-05-19 A3 DONE docs/AWS_SETUP.md created: 5 steps (root MFA, IAM user, billing alerts, CLI, S3), 4 STOP points summarised, script/cost table. All resources eu-west-2, tag Project=fraud-detection-poc.
2026-05-19 A4-A7,A9 DONE scripts/aws/ created: _common.py, setup_s3.py (private+versioned+SSE-S3), upload_data.py (progress+md5+cost), launch_ec2.py (t3.large AL2023, £0.08/h warn), stop_ec2.py (confirm+savings), run_training_on_ec2.py (ssh pipeline+S3 upload). All compile OK. NOT executed (need boto3/AWS creds/STOP 1-3).
2026-05-19 A8 DONE src/config.py created per spec (LOCAL_DATA_PATH, S3_BUCKET, USE_S3, MODEL_PATH) + data_path()/model_path() resolvers; verified local & S3 modes. SCOPE NOTE: sweeping per-file path rewrite across the verified pipeline deliberately deferred (regression risk; pipeline not re-verified post pandas-3.0 churn) — to be migrated per-file with re-verification, per MISSION rule 4.
2026-05-19 A10 DONE git add -A (gitignore-verified: 0 sensitive/large staged); commit cb43e44; pushed 14c26e2..cb43e44 to origin/main. STOP 4 (remote) was already satisfied. Phase A autonomous tasks complete.
