"""C7.4 — CloudWatch publisher test.

All four tests run WITHOUT touching real AWS. boto3 is replaced with a fake so
the no-credentials fallback path and the metric payload are exercised
deterministically regardless of any ambient AWS credentials.

Run: python tests/test_cloudwatch.py
"""

from __future__ import annotations

import io
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.monitoring import cloudwatch_publisher as cw


class _RaisingBoto3:
    """Stands in for the boto3 module; simulates 'no AWS credentials'."""

    def client(self, *a, **k):  # noqa: ANN002, ANN003
        raise RuntimeError("Unable to locate credentials")


class _CapturingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def put_metric_data(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        return {}


class _CapturingBoto3:
    def __init__(self) -> None:
        self.client_obj = _CapturingClient()

    def client(self, *a, **k):  # noqa: ANN002, ANN003
        return self.client_obj


@contextmanager
def _fake_boto3(fake):
    real = sys.modules.get("boto3")
    sys.modules["boto3"] = fake
    try:
        yield fake
    finally:
        if real is not None:
            sys.modules["boto3"] = real
        else:
            sys.modules.pop("boto3", None)


def _result(name: str, passed: bool, reason: str = "") -> bool:
    line = f"[{'PASS' if passed else 'FAIL'}] {name}"
    if not passed and reason:
        line += f"  -- {reason}"
    print(line)
    return passed


def test_publish_no_credentials() -> bool:
    sample = {"transactions_processed": 500, "fraud_rate": 0.012,
              "false_positive_rate": 0.752, "avg_latency_ms": 27.0,
              "decisions_hold": 6, "daily_fraud_saving_gbp": 296445}
    buf = io.StringIO()
    raised = False
    try:
        with _fake_boto3(_RaisingBoto3()), redirect_stdout(buf):
            sent = cw.publish_metrics(sample)
    except Exception:  # noqa: BLE001
        raised = True
        sent = None
    out = buf.getvalue()
    ok = (raised is False and sent is False and "METRICS" in out)
    return _result("publish_metrics returns False + logs locally, no creds",
                   ok, f"sent={sent} raised={raised} out={out!r}")


def test_log_locally_format() -> bool:
    metrics = {"transactions_processed": 500, "fraud_rate": 0.04,
               "false_positive_rate": 0.10, "avg_latency_ms": 27.4,
               "decisions_hold": 6, "daily_fraud_saving_gbp": 296445}
    buf = io.StringIO()
    with redirect_stdout(buf):
        cw._log_locally(metrics)
    out = buf.getvalue()
    ok = ("METRICS" in out and "4.0%" in out          # fraud_rate as percent
          and "27ms" in out                            # latency as integer
          and "£296,445" in out)                       # saving as £ amount
    return _result("_log_locally formats METRICS/percent/int-ms/£", ok,
                   repr(out))


def test_health_fallback() -> bool:
    raised = False
    buf = io.StringIO()
    try:
        with _fake_boto3(_RaisingBoto3()), redirect_stdout(buf):
            sent = cw.publish_pipeline_health("HEALTHY", "model")
    except Exception:  # noqa: BLE001
        raised = True
        sent = None
    ok = (raised is False and sent is False)
    return _result("publish_pipeline_health returns False, no raise, no creds",
                   ok, f"sent={sent} raised={raised}")


def test_missing_keys_default_zero() -> bool:
    raised = False
    with _fake_boto3(_CapturingBoto3()) as fake:
        try:
            sent = cw.publish_metrics({})  # empty dict — every key missing
        except Exception:  # noqa: BLE001
            raised = True
            sent = None
        data = fake.client_obj.calls[0]["MetricData"] if \
            fake.client_obj.calls else []
    all_zero = len(data) == 7 and all(d["Value"] == 0 for d in data)
    ok = (raised is False and sent is True and all_zero)
    return _result("publish_metrics({}) doesn't crash; all 7 values default 0",
                   ok, f"sent={sent} raised={raised} "
                       f"values={[d['Value'] for d in data]}")


def main() -> int:
    print("=" * 64)
    print("CloudWatch publisher test suite (C7.4)")
    print("=" * 64)
    results = [
        test_publish_no_credentials(),
        test_log_locally_format(),
        test_health_fallback(),
        test_missing_keys_default_zero(),
    ]
    print("-" * 64)
    print(f"{sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
