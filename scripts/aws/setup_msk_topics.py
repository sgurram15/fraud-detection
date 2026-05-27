"""C5.2 — Create the MSK Kafka topics.

Creates every pipeline topic on the cluster named by MSK_BOOTSTRAP_SERVERS
(.env) using the kafka-python admin client, then verifies they exist.

Auth: MSK is created with SASL/IAM (setup_msk.py), so the admin client connects
with SASL_SSL + OAUTHBEARER, using an AWS IAM token provider from the
aws-msk-iam-sasl-signer package:
    pip install kafka-python aws-msk-iam-sasl-signer-python

Topic names use hyphens (Kafka convention); stream_bus (C5.3) maps its internal
underscore topic names to these.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.aws._common import REGION, get_env

_DAY_MS = 86_400_000

# (name, partitions, replication_factor, retention_days)
TOPICS = [
    ("transactions-inbound", 4, 2, 7),
    ("transactions-enriched", 4, 2, 7),
    ("transactions-approved", 4, 2, 1),
    ("transactions-review", 4, 2, 7),
    ("transactions-flagged", 4, 2, 7),
    ("audit-log", 4, 2, 90),
]


class _MSKTokenProvider:
    """OAUTHBEARER token provider that signs with the caller's AWS IAM creds."""

    def token(self) -> str:
        from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
        tok, _ = MSKAuthTokenProvider.generate_auth_token(REGION)
        return tok


def _admin_client(bootstrap: str):
    try:
        from kafka.admin import KafkaAdminClient
    except ImportError:
        print("kafka-python not installed: pip install kafka-python",
              file=sys.stderr)
        sys.exit(1)
    return KafkaAdminClient(
        bootstrap_servers=bootstrap.split(","),
        security_protocol="SASL_SSL",
        sasl_mechanism="OAUTHBEARER",
        sasl_oauth_token_provider=_MSKTokenProvider(),
        client_id="fraud-detection-admin",
    )


def main() -> int:
    bootstrap = get_env("MSK_BOOTSTRAP_SERVERS")
    if not bootstrap:
        print("MSK_BOOTSTRAP_SERVERS not set in .env — run setup_msk.py first.",
              file=sys.stderr)
        return 1

    from kafka.admin import NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = _admin_client(bootstrap)
    new_topics = [
        NewTopic(name=name, num_partitions=parts, replication_factor=rf,
                 topic_configs={"retention.ms": str(days * _DAY_MS)})
        for name, parts, rf, days in TOPICS
    ]
    try:
        admin.create_topics(new_topics, validate_only=False)
    except TopicAlreadyExistsError:
        print("Some topics already exist — continuing.")
    except Exception as exc:  # partial creation tolerated; verified below
        print(f"create_topics raised {exc!r}; verifying existing set ...")

    existing = set(admin.list_topics())
    print("Topic verification:")
    all_ok = True
    for name, parts, rf, days in TOPICS:
        ok = name in existing
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISSING'}] {name} "
              f"(p={parts}, rf={rf}, retention={days}d)")
    admin.close()

    if all_ok:
        print(f"\nALL TOPICS CREATED on {bootstrap}")
        return 0
    print("\nSome topics are missing — check broker connectivity / IAM perms.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
