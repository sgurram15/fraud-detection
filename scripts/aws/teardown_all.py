"""Master teardown — destroy ALL billable fraud-detection AWS resources.

Runs every delete script in sequence so nothing is left billing after a run:
  * SageMaker endpoint + config + model   (delete_endpoint.py)
  * CloudWatch alarms + dashboard + SNS    (delete_cloudwatch.py)
  * MSK cluster                            (delete_msk.py)

Each child script is idempotent and tolerant of already-absent resources, so
this is safe to run even if only some resources were created. Bedrock has no
persistent resource to delete (it is per-request).

Confirms once, then runs each child with --yes.

Run:  python scripts/aws/teardown_all.py          (one confirmation)
      python scripts/aws/teardown_all.py --yes    (no prompt)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1]))

from scripts.aws._common import confirm

_SCRIPTS = ["delete_endpoint.py", "delete_cloudwatch.py", "delete_msk.py"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    print("This will DELETE all fraud-detection AWS resources "
          "(SageMaker endpoint, CloudWatch alarms/dashboard/SNS, MSK cluster).")
    if not args.yes and not confirm("Tear everything down?"):
        print("Aborted.")
        return 0

    overall = 0
    for script in _SCRIPTS:
        print(f"\n{'=' * 60}\n  {script}\n{'=' * 60}")
        rc = subprocess.call([sys.executable, str(_HERE / script), "--yes"])
        if rc != 0:
            overall = rc
            print(f"  ({script} exited {rc})")

    print("\nTEARDOWN COMPLETE. Verify in the AWS console that no SageMaker "
          "endpoint or MSK cluster remains (those are the hourly-billed ones).")
    return overall


if __name__ == "__main__":
    sys.exit(main())
