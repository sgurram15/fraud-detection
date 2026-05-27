# Fraud Detection POC — Phase C Mission
## Continuing from completed Phase A/B (MISSION.md)

---

## Context

Phase A and B are complete. The following are confirmed working:
- Full feature engineering pipeline (18 features, leakage-free)
- Trained and validated XGBoost model (baseline + tuned, compared on identical test set)
- Production model selected and documented in model_card.md
- MLflow experiment tracking (SQLite backend)
- Feature store (batch + serving mode, <100ms latency)
- Data uploaded to S3
- GitHub repository: https://github.com/sgurram15/fraud-detection

## What remains to complete the POC

Three modules are scaffolding only:
- src/api/          FastAPI scoring endpoint
- src/streaming/    Real-time pipeline simulation
- src/monitoring/   Drift detection

Plus AWS deployment scripts in scripts/aws/ need completing
and running on EC2.

This mission completes all of them in order.

---

## Rules

1. Log every action in MISSION_LOG_C.md as you go
2. At every STOP point — pause and wait for human confirmation
3. Never commit credentials, API keys, or passwords
4. Never use git add . — always add specific paths
5. If unsure about a destructive action — log it and wait
6. If a task fails twice — log it and move to next task
7. All AWS resources tagged: Project=fraud-detection-poc, Region=eu-west-2

---

## MISSION_LOG_C.md format

[timestamp] [task ID] [DONE/FAILED] [one line description]

Example:
2026-05-20 14:23 C1 DONE FastAPI /score endpoint returns 200 with correct schema
2026-05-20 14:31 C3 FAILED missing dependency, installed, retried, DONE

---

## Phase C1 — Complete src/api/main.py

The file already exists as scaffolding. Build it out fully.

### C1.1 — Model loader

Add a model loader at the top of main.py that:
- Loads src/models/saved/tuned_xgboost.pkl on startup
  If USE_S3=true loads from s3://[S3_BUCKET]/models/saved/tuned_xgboost.pkl
- Loads feature store persisted state
- Loads SHAP explainer initialised against the loaded model
- Logs: model version, load time, feature count
- Raises a clear error on startup if any file is missing
- Exposes model metadata via module-level dict:
  MODEL_META = {
    "version": "xgboost-tuned-v1",
    "trained_on": "IEEE-CIS",
    "features": 18,
    "operating_point": "cost-optimal",
    "threshold": [loaded from docs/model_performance/model_comparison.json],
    "loaded_at": [timestamp]
  }

### C1.2 — Input schema

Define a Pydantic model called TransactionRequest:
- transaction_id: str
- card_id: str
- amount: float (must be > 0)
- device_type: str
- hour_of_day: int (0-23)
- day_of_week: int (0-6)
- destination_account_age_days: int (must be >= 0)
- tx_velocity_1h: int (default 0)
- tx_velocity_24h: int (default 0)
- amt_deviation: float (default 1.0)
- is_late_night: bool (default False)
- is_weekend: bool (default False)
- card_age_days: int (default 0)

Define FraudScoreResponse:
- transaction_id: str
- fraud_probability: float
- decision: str  (APPROVE / REVIEW / HOLD)
- threshold_used: float
- reasons: list[str]  (plain English SHAP reasons)
- fca_explanation: dict  (structured for FCA audit log)
- model_version: str
- latency_ms: float

### C1.3 — POST /score endpoint

Implement POST /score that:
- Accepts TransactionRequest
- Validates all fields via Pydantic
- Runs feature store serving mode on the input
- Scores with loaded model
- Applies cost-optimal threshold from MODEL_META
- Routes decision:
  probability < 0.30  → APPROVE
  probability 0.30-0.70 → REVIEW
  probability > 0.70  → HOLD
- Calls explain_prediction() for SHAP reasons
- Calls generate_fca_explanation() for audit dict
- Returns FraudScoreResponse
- Logs request_id, latency_ms, decision to stdout
- Target: under 100ms end-to-end

### C1.4 — POST /score/batch endpoint

Implement POST /score/batch that:
- Accepts list[TransactionRequest] (max 1000)
- Processes using ThreadPoolExecutor
- Returns list[FraudScoreResponse]
- Logs total batch size and avg latency

### C1.5 — GET /health endpoint

Returns:
{
  "status": "healthy",
  "model_version": "xgboost-tuned-v1",
  "model_loaded": true,
  "uptime_seconds": [int],
  "transactions_scored_session": [int]
}

### C1.6 — GET /metrics endpoint

Returns:
{
  "transactions_scored_today": [int],
  "fraud_rate_today": [float],
  "avg_latency_ms": [float],
  "decisions": {
    "APPROVE": [int],
    "REVIEW": [int],
    "HOLD": [int]
  }
}

### C1.7 — Middleware

Add:
- Request ID middleware: UUID per request, added to response headers
- Latency middleware: measures and logs time per request
- Error handler: returns 500 with
  {"error": "internal error", "request_id": [id]}
  Never expose stack traces in responses

### C1.8 — API test script

Create tests/test_api.py that:
- Uses FastAPI TestClient (no running server needed)
- Tests:
  APPROVE case: low-risk transaction, expects decision=APPROVE
  HOLD case: high-risk (amount=5000, new account, late night)
  REVIEW case: medium-risk transaction
  Batch test: 10 transactions, all return valid responses
  Health check: returns status=healthy
  Latency test: 100 sequential requests, avg under 100ms
  Schema test: all required fields present in every response
  Error test: missing required field returns 422 not 500

Print PASS/FAIL per test with reason on failure.

STOP POINT C1:
Run: python tests/test_api.py
Confirm all tests pass before proceeding to C2.
Report back with full test output.

---

## Phase C2 — Complete src/streaming/

Build the full local streaming simulation using Python asyncio
queues. This runs entirely locally — no Kafka or AWS needed.
The same code connects to Amazon MSK in production by swapping
the stream bus for a Kafka client.

### C2.1 — Local stream bus

Create src/streaming/stream_bus.py:
- In-process message bus using asyncio.Queue
- Mimics Kafka topic semantics locally
- Topics:
  transactions_inbound
  transactions_enriched
  transactions_scored
  transactions_flagged
  transactions_approved
  audit_log
- Functions:
  publish(topic, message: dict)
  subscribe(topic) -> AsyncGenerator
  get_stats() -> dict (messages per topic, throughput)
- When USE_S3=true and KAFKA_BOOTSTRAP_SERVERS is set in .env,
  replace queues with confluent-kafka producer/consumer
  connecting to Amazon MSK

### C2.2 — Transaction producer

Create src/streaming/transaction_producer.py:
- Reads test_transaction.csv from data/raw/
  If USE_S3=true reads from S3_BUCKET/data/raw/
- Publishes each row as JSON to transactions_inbound
- Configurable: TRANSACTIONS_PER_SECOND = 10
- Partitions by card_id
- Prints: [timestamp] Published TXN-xxx £amount
- Tracks: total published, errors, elapsed time
- Runs until complete or KeyboardInterrupt

### C2.3 — Feature enrichment consumer

Create src/streaming/feature_enricher.py:
- Subscribes to transactions_inbound
- For each event:
  Runs feature store serving mode
  Appends feature vector to event dict
  Publishes to transactions_enriched
- Prints: [timestamp] Enriched TXN-xxx (Xms)
- Tracks: throughput, avg enrichment latency

### C2.4 — Fraud scorer consumer

Create src/streaming/fraud_scorer.py:
- Subscribes to transactions_enriched
- For each event:
  Calls POST /score on local FastAPI endpoint
  Routes by decision:
    APPROVE → transactions_approved
    REVIEW  → transactions_scored
    HOLD    → transactions_flagged
  Writes audit record to audit_log topic
- Prints: [timestamp] TXN-xxx → HOLD (0.847)
- Tracks: decisions per minute, distribution

### C2.5 — Fraud agent (local Bedrock simulation)

Create src/streaming/fraud_agent.py:
- Subscribes to transactions_flagged
- For each flagged transaction builds a reasoning prompt
- If AWS_BEDROCK_ENABLED=true in .env:
  Calls Amazon Bedrock claude-sonnet-4-20250514
  via boto3 bedrock-runtime client
  Parses JSON response
- If AWS_BEDROCK_ENABLED not set (local mode):
  Uses a rule-based local reasoner that:
  - Score > 0.90 → BLOCK, confidence HIGH
  - Score 0.70-0.90 + new account → HOLD_STEP_UP, confidence HIGH
  - Score 0.70-0.90 + known device → HOLD_STEP_UP, confidence MEDIUM
  - Generates plain English reasoning string
  - Generates fca_narrative string
  Prints: [LOCAL AGENT] Decision made without Bedrock
- In both modes outputs:
  {
    "decision": "HOLD_AND_STEP_UP|BLOCK|MONITOR",
    "confidence": "HIGH|MEDIUM|LOW",
    "reasoning": "plain English paragraph",
    "fca_narrative": "explanation for FCA audit log",
    "alert_fraud_team": true|false,
    "draft_sar": true|false
  }
- Publishes complete audit record to audit_log topic

### C2.6 — Audit log writer

Create src/streaming/audit_writer.py:
- Subscribes to audit_log topic
- Writes each record to:
  Local: data/audit/{date}/{transaction_id}.json
  If USE_S3=true: s3://[S3_BUCKET]/audit/{date}/{transaction_id}.json
- Maintains a local audit_summary.json counting:
  total_decisions, by_decision_type, by_confidence, fraud_rate
- Prints: [timestamp] Audit written TXN-xxx

### C2.7 — Pipeline orchestrator

Create src/streaming/run_pipeline.py:
- Starts all components as concurrent asyncio tasks:
  1. transaction_producer (publishes at 10 TPS)
  2. feature_enricher (consumes inbound, publishes enriched)
  3. fraud_scorer (consumes enriched, routes by score)
  4. fraud_agent (consumes flagged, reasons and decides)
  5. audit_writer (consumes audit_log, writes records)
- Prints a live dashboard updating every 2 seconds:

  ╔══════════════════════════════════════════════════╗
  ║     FRAUD DETECTION PIPELINE — LIVE              ║
  ╠══════════════════════════════════════════════════╣
  ║  Transactions processed:  1,247                  ║
  ║  Throughput:              10.2 TPS               ║
  ║  Avg end-to-end latency:  43ms                   ║
  ╠══════════════════════════════════════════════════╣
  ║  APPROVE:  1,089  (87.3%)                        ║
  ║  REVIEW:     121   (9.7%)                        ║
  ║  HOLD:        37   (3.0%)                        ║
  ╠══════════════════════════════════════════════════╣
  ║  Agent decisions (HOLD transactions):            ║
  ║    HOLD_AND_STEP_UP:  29                         ║
  ║    BLOCK:              8                         ║
  ║    MONITOR:            0                         ║
  ╠══════════════════════════════════════════════════╣
  ║  Est. daily fraud saving:  £24,100               ║
  ║  Est. daily false pos cost: £3,025               ║
  ╚══════════════════════════════════════════════════╝

- Runs until all transactions processed
- Prints final summary report on completion
- Saves final report to docs/pipeline_run_report.json

### C2.8 — Streaming test script

Create tests/test_streaming.py that:
- Runs the full pipeline on 100 test transactions
- Verifies:
  All transactions reach a terminal topic (approved/flagged/scored)
  No transactions lost between topics
  Audit record written for every transaction
  Agent decision present for every HOLD transaction
  End-to-end latency under 500ms per transaction
  No unhandled exceptions during run

Print PASS/FAIL per test.

STOP POINT C2:
Run: python src/streaming/run_pipeline.py
Let it process at least 500 transactions.
Paste the live dashboard output here before proceeding to C3.

---

## Phase C3 — Complete src/monitoring/

### C3.1 — Drift detector

Create src/monitoring/drift_detector.py using Evidently:
- Loads reference dataset: training data feature distributions
  (saved during train_baseline.py run)
- Loads current dataset: last 7 days of scored transactions
  from data/audit/ or S3
- Runs Evidently DataDriftReport comparing:
  Feature distributions (all 18 features)
  Prediction distribution (fraud_probability scores)
  Target drift if confirmed fraud labels available
- Saves HTML report to docs/monitoring/drift_report.html
- Saves JSON summary to docs/monitoring/drift_summary.json
- Prints:
  DRIFT DETECTED: [list of drifted features] — RETRAINING RECOMMENDED
  or
  NO DRIFT DETECTED — model performance stable

### C3.2 — Performance tracker

Create src/monitoring/performance_tracker.py:
- Reads audit records from data/audit/ or S3
- For any records with confirmed fraud outcomes
  (outcome field set by fraud team post-decision):
  Calculates rolling 7-day metrics:
    Recall, Precision, FPR, AUC estimate
    False negative rate trending
    Daily £ loss estimate
- Compares against baseline from docs/model_performance/
- Flags if any metric degrades more than 3% week-on-week
- Saves report to docs/monitoring/performance_report.json

### C3.3 — Monitoring runner

Create src/monitoring/run_monitoring.py:
- Runs drift_detector.py
- Runs performance_tracker.py
- Produces combined monitoring report
- Prints clear HEALTHY / WARNING / RETRAINING_REQUIRED verdict
- In production this runs weekly via AWS EventBridge schedule
  (document this in docs/production_architecture.md)

### C3.4 — Monitoring test

Create tests/test_monitoring.py:
- Generates synthetic audit records (200 normal + 50 drifted)
- Runs drift detector — verifies drift IS detected on drifted set
- Runs on normal set — verifies drift IS NOT detected
- Print PASS/FAIL per test

STOP POINT C3:
Run: python src/monitoring/run_monitoring.py
Paste the verdict output here before proceeding to C4.

---

## Phase C4 — AWS SageMaker Deployment

Deploy the trained model as a SageMaker real-time endpoint.
This replaces the local FastAPI endpoint in production.

### C4.1 — SageMaker scoring script

Create src/api/sagemaker_scoring.py:

def model_fn(model_dir):
  - Loads tuned_xgboost.pkl from model_dir
  - Loads feature store state
  - Loads SHAP explainer
  - Returns model object

def input_fn(request_body, content_type):
  - Parses JSON transaction
  - Returns dict

def predict_fn(input_data, model):
  - Runs feature store serving mode
  - Scores with model
  - Generates SHAP reasons
  - Returns score + reasons

def output_fn(prediction, accept):
  - Formats as FraudScoreResponse JSON
  - Returns response

### C4.2 — SageMaker deployment script

Update scripts/aws/deploy_endpoint.py:
- Reads S3_BUCKET and AWS credentials from .env
- Packages model artifacts:
  tuned_xgboost.pkl
  feature_store state
  sagemaker_scoring.py
- Uploads package to S3: models/sagemaker/model.tar.gz
- Creates SageMaker model pointing to package
- Creates endpoint config:
  Instance type: ml.m5.large
  Initial instance count: 1
- Creates endpoint: fraud-scoring-endpoint
- Waits for endpoint InService state
- Tests endpoint with a sample transaction
- Saves endpoint name to .env as SAGEMAKER_ENDPOINT
- Prints: ENDPOINT LIVE — [endpoint URL]
- Estimated cost: ml.m5.large = £0.13/hour
- Prints: REMEMBER TO DELETE ENDPOINT WHEN DONE

### C4.3 — Endpoint test

Create tests/test_sagemaker_endpoint.py:
- Reads SAGEMAKER_ENDPOINT from .env
- Sends 10 test transactions to live endpoint
- Verifies:
  Response schema matches FraudScoreResponse
  Latency under 200ms per call
  APPROVE/REVIEW/HOLD decisions all appear in 10 samples
  SHAP reasons present in every response
Print PASS/FAIL per test.

### C4.4 — Endpoint teardown script

Create scripts/aws/delete_endpoint.py:
- Reads SAGEMAKER_ENDPOINT from .env
- Shows endpoint state and estimated hourly cost
- Asks: "Delete endpoint [name]? (yes/no)"
- On yes: deletes endpoint, config, and model
- Confirms deleted
- Prints: ENDPOINT DELETED — No further inference charges
- Note: always delete SageMaker endpoints when not in use
  They charge by the hour whether receiving traffic or not

STOP POINT C4:
Human must confirm:
1. AWS credentials are configured (aws configure)
2. EC2 instance is running (or SageMaker has permissions)
3. S3 bucket name is correct in .env
Then tell Claude Code to proceed with C4.2.

---

## Phase C5 — Amazon MSK Streaming (Production)

Connect the local streaming pipeline to Amazon MSK —
the production Kafka service. Your existing streaming code
works unchanged — only the connection config changes.

### C5.1 — MSK cluster setup script

Create scripts/aws/setup_msk.py:
- Reads AWS credentials from .env
- Creates MSK cluster:
  Name: fraud-detection-msk
  Kafka version: 3.5.1
  Broker type: kafka.t3.small (cheapest, sufficient for POC)
  Number of brokers: 2 (minimum for fault tolerance)
  Storage per broker: 20GB
  Region: eu-west-2 (FCA data residency)
  Tag: Project=fraud-detection-poc
- Waits for cluster to be ACTIVE (can take 15-20 minutes)
- Gets bootstrap broker string
- Saves to .env as MSK_BOOTSTRAP_SERVERS
- Prints: MSK CLUSTER ACTIVE — [bootstrap string]
- Estimated cost: kafka.t3.small = £0.07/hour per broker = £0.14/hour total
- Prints: REMEMBER TO DELETE CLUSTER WHEN DONE

### C5.2 — MSK topic creation

Create scripts/aws/setup_msk_topics.py:
- Reads MSK_BOOTSTRAP_SERVERS from .env
- Creates all topics using kafka-python admin client:
  transactions-inbound     (partitions: 4, replication: 2, retention: 7 days)
  transactions-enriched    (partitions: 4, replication: 2, retention: 7 days)
  transactions-approved    (partitions: 4, replication: 2, retention: 1 day)
  transactions-review      (partitions: 4, replication: 2, retention: 7 days)
  transactions-flagged     (partitions: 4, replication: 2, retention: 7 days)
  audit-log                (partitions: 4, replication: 2, retention: 90 days)
- Verifies all topics created
- Prints: ALL TOPICS CREATED on [bootstrap string]

### C5.3 — Update stream bus for MSK

Update src/streaming/stream_bus.py:
- If KAFKA_BOOTSTRAP_SERVERS is set in .env:
  Replace asyncio.Queue with confluent-kafka producer/consumer
  connecting to MSK_BOOTSTRAP_SERVERS
  Use SASL_SSL authentication
- If not set: use local asyncio.Queue (existing behaviour)
- No changes needed to any other streaming file
  They all use stream_bus publish/subscribe — the swap is transparent

### C5.4 — Run pipeline against MSK

Create scripts/aws/run_pipeline_on_ec2.py:
- SSHs into EC2 instance (reads EC2_PUBLIC_IP from .env)
- Sets environment variables on EC2:
  USE_S3=true
  S3_BUCKET=[from .env]
  KAFKA_BOOTSTRAP_SERVERS=[MSK_BOOTSTRAP_SERVERS from .env]
  AWS_BEDROCK_ENABLED=true (if Bedrock access confirmed)
- Runs: python src/streaming/run_pipeline.py
- Streams live dashboard output back to local terminal
- Ends with final summary report

STOP POINT C5:
Human must confirm before creating MSK cluster:
MSK kafka.t3.small = £0.14/hour for 2 brokers
Cluster creation takes 15-20 minutes
Confirm you want to proceed and have budget for ~2 hours = £0.28
Then tell Claude Code to proceed with C5.1.

### C5.5 — MSK teardown script

Create scripts/aws/delete_msk.py:
- Reads MSK cluster ARN from .env
- Shows cluster state and estimated cost incurred
- Asks: "Delete MSK cluster? (yes/no)"
- On yes: deletes cluster
- Confirms deleted
- Prints: MSK CLUSTER DELETED — No further charges

---

## Phase C6 — Amazon Bedrock Agent

Connect the fraud_agent.py to Amazon Bedrock for production
reasoning. Uses Claude Sonnet via boto3.

### C6.1 — Enable Bedrock access

In AWS console:
- Go to Amazon Bedrock → Model access
- Request access to: Anthropic Claude Sonnet
- Takes 1-5 minutes to approve
- No additional cost beyond per-token usage

STOP POINT C6:
Human confirms Bedrock model access approved in AWS console.
Then tell Claude Code to proceed.

### C6.2 — Bedrock agent integration

Update src/streaming/fraud_agent.py Bedrock path:
- Uses boto3 bedrock-runtime client
- Model: claude-sonnet-4-20250514
- Sends fraud reasoning prompt
- Parses JSON response
- Handles throttling with exponential backoff
- Logs token usage per call
- Estimated cost: ~£0.002 per agent invocation
  At 3% fraud rate on 10 TPS = ~26,000 invocations/day = £52/day
  Note: only flagged transactions (score > 0.70) reach the agent
  At realistic 0.5% HOLD rate = ~4,300/day = £8.60/day

### C6.3 — Bedrock test

Create tests/test_bedrock_agent.py:
- Sends 5 hardcoded flagged transactions to Bedrock agent
- Verifies:
  Response parses as valid JSON
  Decision is one of HOLD_AND_STEP_UP / BLOCK / MONITOR
  Reasoning field is non-empty plain English
  fca_narrative field is non-empty
  Token usage logged per call
Print PASS/FAIL per test.

---

## Phase C7 — CloudWatch Monitoring

### C7.1 — CloudWatch metrics publisher

Create src/monitoring/cloudwatch_publisher.py:
- Reads AWS credentials from .env
- Publishes custom metrics to CloudWatch namespace:
  FraudDetection/Pipeline
  Metrics:
    TransactionsProcessed (Count)
    FraudRate (Percent)
    FalsePositiveRate (Percent)
    EndToEndLatencyMs (Milliseconds)
    AgentDecisions_Hold (Count)
    AgentDecisions_Block (Count)
    DailyFraudSaving_GBP (None)
- Publishes every 60 seconds when pipeline is running
- If AWS credentials not set: logs metrics to stdout only

### C7.2 — CloudWatch alarms

Create scripts/aws/setup_cloudwatch.py:
- Creates CloudWatch alarms:
  FPR > 5%: SNS alert to fraud team email
  Latency > 1000ms: SNS alert to on-call
  FraudRate drops to 0% for >1 hour: SNS alert (model may have failed)
  AgentDecisions_Block > 100 in 1 hour: SNS alert (unusually high)
- Creates CloudWatch dashboard: FraudDetectionPOC
  Widgets: transactions/min, fraud rate, latency P50/P95, decision distribution
- Reads ALERT_EMAIL from .env
- Prints: CLOUDWATCH ALARMS CREATED

### C7.3 — CloudWatch test

Create tests/test_cloudwatch.py:
- Publishes 10 test metrics to CloudWatch
- Reads them back and verifies values match
- Verifies dashboard exists
Print PASS/FAIL per test.

---

## Phase C8 — End-to-End Demo Script

### C8.1 — Demo runner

Create scripts/run_demo.py:
A polished demonstration script for showing to a PSP CTO.

Runs through this sequence with clear narration:

Step 1 — System check
  Verifies model loaded, API healthy, pipeline ready
  Prints: SYSTEM READY — Model: xgboost-tuned-v1

Step 2 — Baseline comparison
  Loads docs/model_performance/model_comparison.json
  Prints comparison table: baseline vs tuned

Step 3 — Live scoring demo
  Scores 5 hand-crafted transactions covering:
    1. Clean transaction (low velocity, known device, normal amount)
    2. Suspicious transaction (new account, 10x average, late night)
    3. Definite fraud (new account, 20x average, high velocity, 3am)
    4. Edge case (high amount but long-standing card, trusted device)
    5. Borderline case (medium risk, step-up auth appropriate)
  For each: prints transaction details, score, decision, top 3 SHAP reasons
  Formatted cleanly for screen sharing

Step 4 — Pipeline throughput demo
  Runs pipeline for 60 seconds at 10 TPS
  Shows live dashboard
  Prints final 60-second summary

Step 5 — Financial impact summary
  Calculates and prints:
    Transactions processed in demo
    Fraud caught
    Estimated daily saving at 100k TPS scale
    Infrastructure cost
    ROI: daily saving / daily infrastructure cost

Step 6 — Architecture summary
  Prints one-page ASCII summary of the full AWS architecture
  References docs/production_architecture.md for detail

### C8.2 — Demo test

Create tests/test_demo.py:
- Runs the full demo script
- Verifies it completes without errors
- Verifies all 5 scored transactions produce valid responses
- Verifies financial summary numbers are non-zero and positive
Print PASS/FAIL.

STOP POINT C8:
Run: python scripts/run_demo.py
Paste the full output here.
This is the final POC validation — if this runs cleanly
the POC is complete and ready for a CTO demonstration.

---

## Phase C9 — Documentation and GitHub Push

### C9.1 — Update README.md

Update README.md to reflect completed POC:
- Add Quick Start section:
  git clone https://github.com/sgurram15/fraud-detection
  pip install -r requirements.txt
  python scripts/run_demo.py
- Add Architecture section referencing docs/
- Add Performance section with final model metrics
- Add Cost section with AWS breakdown
- Add FCA Compliance section

### C9.2 — Update production_architecture.md

Add Version 2 AWS production section covering:
- Amazon MSK replacing local stream bus
- SageMaker endpoint replacing local FastAPI
- Amazon Bedrock agent replacing local rule-based agent
- CloudWatch replacing local monitoring
- EventBridge weekly retraining trigger
- All components in eu-west-2 for FCA data residency

### C9.3 — Final git push

Run in order:
git add src/api/main.py
git add src/api/sagemaker_scoring.py
git add src/streaming/
git add src/monitoring/
git add scripts/aws/
git add scripts/run_demo.py
git add tests/test_api.py
git add tests/test_streaming.py
git add tests/test_monitoring.py
git add tests/test_sagemaker_endpoint.py
git add tests/test_bedrock_agent.py
git add tests/test_cloudwatch.py
git add tests/test_demo.py
git add docs/
git add README.md
git status

STOP POINT C9:
Review git status output.
Confirm .env is NOT listed.
Confirm no *.pkl files are listed.
Confirm no *.csv files are listed.
Then: git commit and push.

---

## STOP POINTS SUMMARY

C1: All API tests pass — confirm before building streaming
C2: Pipeline runs 500+ transactions — paste dashboard output
C3: Monitoring verdict printed — confirm before AWS deployment
C4: AWS credentials confirmed — confirm before SageMaker deploy
C5: Budget confirmed for MSK (£0.28 for ~2 hours) — confirm before cluster creation
C6: Bedrock model access approved in AWS console
C8: Full demo runs cleanly — final POC validation
C9: Git status reviewed — confirm before final push

---

## Cost Summary for Phase C

| Resource | Duration | Cost |
|----------|----------|------|
| SageMaker ml.m5.large endpoint | ~2 hours testing | £0.26 |
| MSK kafka.t3.small x2 brokers | ~2 hours testing | £0.28 |
| Bedrock Claude Sonnet (~500 test calls) | One-off | £1.00 |
| CloudWatch metrics | ~1 month | £0.10 |
| EC2 t3.large training run | Already done | £0 |
| S3 storage | Already running | £0 |
| Total Phase C | | ~£1.64 |

Total project cost to date including Phase A/B: under £5.

---

## What You Have When C9 Is Complete

A fully working fraud detection POC that:
- Trains an XGBoost model on real fraud data
- Scores transactions in real time under 100ms
- Explains every decision with SHAP reason codes
- Routes decisions through an AI reasoning agent
- Writes an FCA-compliant immutable audit trail
- Monitors for model drift automatically
- Deploys on AWS SageMaker, MSK, and Bedrock
- Demonstrates live to a PSP CTO in 10 minutes
- Costs £350/month in production
- Pays for itself by catching one fraud event per day

This is the complete proof of concept.