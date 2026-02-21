"""
Transaction consumer: reads streamed transactions, scores with feature store + models, writes results.

Requires trained artifacts in models/artifacts/: feature_store.joblib, isolation_forest.joblib,
autoencoder.pt, ensemble.joblib. Run the training/evaluation pipeline first to create them.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from src.features.feature_store import SimpleFeatureStore
from src.models.autoencoder import FraudAutoencoder
from src.models.ensemble import EnsembleScorer
from src.models.isolation_forest import IsolationForestModel

logger = logging.getLogger(__name__)

DEFAULT_JSONL_INPUT = "data/stream_output/transactions.jsonl"
DEFAULT_SCORED_OUTPUT = "data/stream_output/scored_transactions.jsonl"
DEFAULT_ALERTS_OUTPUT = "data/stream_output/alerts.jsonl"
DEFAULT_ARTIFACTS_DIR = "models/artifacts"


class TransactionConsumer:
    """
    Reads transactions from Kafka or local JSONL, computes features, scores with
    ensemble (Isolation Forest + Autoencoder), writes scored results and optional alerts.
    """

    def __init__(
        self,
        feature_store_path: str | Path | None = None,
        iso_path: str | Path | None = None,
        ae_path: str | Path | None = None,
        ensemble_path: str | Path | None = None,
        scored_path: str | Path = DEFAULT_SCORED_OUTPUT,
        alerts_path: str | Path = DEFAULT_ALERTS_OUTPUT,
    ) -> None:
        """
        Initialize consumer; artifact paths default to models/artifacts/<name>.

        Args:
            feature_store_path: Path to feature_store.joblib.
            iso_path: Path to isolation_forest.joblib.
            ae_path: Path to autoencoder.pt.
            ensemble_path: Path to ensemble.joblib.
            scored_path: Output path for scored_transactions.jsonl.
            alerts_path: Output path for alerts.jsonl (HIGH risk only).
        """
        base = Path(DEFAULT_ARTIFACTS_DIR)
        self._feature_store_path = Path(feature_store_path or base / "feature_store.joblib")
        self._iso_path = Path(iso_path or base / "isolation_forest.joblib")
        self._ae_path = Path(ae_path or base / "autoencoder.pt")
        self._ensemble_path = Path(ensemble_path or base / "ensemble.joblib")
        self.scored_path = Path(scored_path)
        self.alerts_path = Path(alerts_path)
        self._store: SimpleFeatureStore | None = None
        self._iso: IsolationForestModel | None = None
        self._ae: FraudAutoencoder | None = None
        self._ensemble: EnsembleScorer | None = None

    def load_models(self) -> None:
        """Load feature store and all models from artifacts."""
        self._store = SimpleFeatureStore.load(self._feature_store_path)
        self._iso = IsolationForestModel.load(self._iso_path)
        self._ae = FraudAutoencoder.load(self._ae_path)
        self._ensemble = EnsembleScorer.load(self._ensemble_path)
        logger.info("Loaded feature store and models from %s", self._feature_store_path.parent)

    def _score_one(self, transaction: dict) -> tuple[dict, float]:
        """
        Compute features and score one transaction. Returns (output_dict, processing_time_ms).
        """
        t0 = time.perf_counter()
        features = self._store.transform_single(transaction)
        iso_score = self._iso.predict_single(features)
        ae_score = self._ae.predict_single(features)
        result = self._ensemble.score_transaction(iso_score, ae_score)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        transaction_id = str(transaction.get("trans_num", ""))
        timestamp = str(transaction.get("trans_date_trans_time", ""))
        amount = float(transaction.get("amt", 0))

        out = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "amount": amount,
            "fraud_score": result["fraud_score"],
            "risk_tier": result["risk_tier"],
            "iso_score": result["iso_score"],
            "ae_score": result["ae_score"],
            "should_alert": result["should_alert"],
            "processing_time_ms": round(elapsed_ms, 2),
        }
        return out, elapsed_ms

    def run_local(
        self,
        input_path: str | Path = DEFAULT_JSONL_INPUT,
        limit: int | None = None,
    ) -> dict:
        """
        Read from local JSONL, score each transaction, write scored output and alerts.

        Args:
            input_path: Path to transactions.jsonl.
            limit: Max number of transactions to process (None = all).

        Returns:
            Summary dict: total_processed, avg_processing_time_ms, count_low, count_medium, count_high, alerts_triggered.
        """
        if self._store is None or self._iso is None or self._ae is None or self._ensemble is None:
            self.load_models()

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError("Input file not found: %s" % input_path)

        self.scored_path.parent.mkdir(parents=True, exist_ok=True)
        total_time_ms = 0.0
        count = 0
        tier_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        alerts_triggered = 0

        with open(input_path) as f_in, open(self.scored_path, "w") as f_out, open(
            self.alerts_path, "w"
        ) as f_alerts:
            for line in f_in:
                if limit is not None and count >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    txn = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("Skip invalid JSON line: %s", e)
                    continue
                out, elapsed_ms = self._score_one(txn)
                total_time_ms += elapsed_ms
                count += 1
                tier_counts[out["risk_tier"]] += 1
                if out["should_alert"]:
                    alerts_triggered += 1
                f_out.write(json.dumps(out) + "\n")
                if out["risk_tier"] == "HIGH":
                    f_alerts.write(json.dumps(out) + "\n")

        summary = {
            "total_processed": count,
            "avg_processing_time_ms": round(total_time_ms / count, 2) if count else 0,
            "count_low": tier_counts["LOW"],
            "count_medium": tier_counts["MEDIUM"],
            "count_high": tier_counts["HIGH"],
            "alerts_triggered": alerts_triggered,
        }
        logger.info(
            "Processed %d transactions; avg %.2f ms; alerts %d",
            count,
            summary["avg_processing_time_ms"],
            alerts_triggered,
        )
        return summary

    def run_kafka(
        self,
        topic: str = "transactions",
        limit: int | None = None,
    ) -> dict:
        """
        Read from Kafka topic, score each message, write scored output and alerts.

        Args:
            topic: Kafka topic to consume.
            limit: Max number of transactions to process (None = all).

        Returns:
            Summary dict (same as run_local).
        """
        from confluent_kafka import Consumer as KafkaConsumer

        if self._store is None or self._iso is None or self._ae is None or self._ensemble is None:
            self.load_models()

        import yaml
        config_path = Path("configs/kafka_config.yaml")
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}
        bootstrap = cfg.get("bootstrap_servers", "localhost:9092")
        topic_name = cfg.get("topics", {}).get("transactions", topic)
        group_id = cfg.get("consumer", {}).get("group_id", "fraud-detection-group")

        conf = {
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
        }
        consumer = KafkaConsumer(conf)
        consumer.subscribe([topic_name])

        self.scored_path.parent.mkdir(parents=True, exist_ok=True)
        total_time_ms = 0.0
        count = 0
        tier_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        alerts_triggered = 0

        try:
            with open(self.scored_path, "w") as f_out, open(self.alerts_path, "w") as f_alerts:
                while True:
                    if limit is not None and count >= limit:
                        break
                    msg = consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        logger.warning("Consumer error: %s", msg.error())
                        continue
                    try:
                        txn = json.loads(msg.value().decode("utf-8"))
                    except (json.JSONDecodeError, AttributeError) as e:
                        logger.warning("Skip invalid message: %s", e)
                        continue
                    out, elapsed_ms = self._score_one(txn)
                    total_time_ms += elapsed_ms
                    count += 1
                    tier_counts[out["risk_tier"]] += 1
                    if out["should_alert"]:
                        alerts_triggered += 1
                    f_out.write(json.dumps(out) + "\n")
                    if out["risk_tier"] == "HIGH":
                        f_alerts.write(json.dumps(out) + "\n")
        finally:
            consumer.close()

        summary = {
            "total_processed": count,
            "avg_processing_time_ms": round(total_time_ms / count, 2) if count else 0,
            "count_low": tier_counts["LOW"],
            "count_medium": tier_counts["MEDIUM"],
            "count_high": tier_counts["HIGH"],
            "alerts_triggered": alerts_triggered,
        }
        logger.info(
            "Processed %d transactions from Kafka; avg %.2f ms; alerts %d",
            count,
            summary["avg_processing_time_ms"],
            alerts_triggered,
        )
        return summary

    def run(
        self,
        use_kafka: bool = False,
        input_path: str | Path = DEFAULT_JSONL_INPUT,
        topic: str = "transactions",
        limit: int | None = None,
    ) -> dict:
        """
        Run consumer in Kafka or local mode.

        Returns:
            Summary dict.
        """
        if use_kafka:
            return self.run_kafka(topic=topic, limit=limit)
        return self.run_local(input_path=input_path, limit=limit)


def _print_summary(summary: dict) -> None:
    """Print end-of-run summary to stdout."""
    print("\n--- Summary ---")
    print("  Total processed:     %d" % summary["total_processed"])
    print("  Avg processing time: %.2f ms per transaction" % summary["avg_processing_time_ms"])
    print("  Count by risk tier:")
    print("    LOW:    %d" % summary["count_low"])
    print("    MEDIUM: %d" % summary["count_medium"])
    print("    HIGH:   %d" % summary["count_high"])
    print("  Alerts triggered:    %d" % summary["alerts_triggered"])


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume transactions from Kafka or local JSONL, score with ensemble, write results"
    )
    parser.add_argument(
        "--kafka",
        action="store_true",
        help="Read from Kafka topic (default: read from local JSONL)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Read from local JSONL file (default if neither --kafka nor --local)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only N transactions (default: all)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_JSONL_INPUT,
        help="Input JSONL path (local mode)",
    )
    parser.add_argument(
        "--scored",
        type=str,
        default=DEFAULT_SCORED_OUTPUT,
        help="Output path for scored_transactions.jsonl",
    )
    parser.add_argument(
        "--alerts",
        type=str,
        default=DEFAULT_ALERTS_OUTPUT,
        help="Output path for alerts.jsonl",
    )
    args = parser.parse_args()

    use_kafka = args.kafka
    if not args.kafka and not args.local:
        use_kafka = False

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    consumer = TransactionConsumer(scored_path=args.scored, alerts_path=args.alerts)
    try:
        summary = consumer.run(
            use_kafka=use_kafka,
            input_path=args.input,
            limit=args.limit,
        )
        _print_summary(summary)
    except FileNotFoundError as e:
        logger.error("%s", e)
        raise


if __name__ == "__main__":
    _main()
