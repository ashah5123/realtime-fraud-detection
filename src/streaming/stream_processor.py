"""
Stream processor: loads trained models and scores individual transactions with metrics tracking.
"""

import logging
import time
from pathlib import Path

from src.features.feature_store import SimpleFeatureStore
from src.models.autoencoder import FraudAutoencoder
from src.models.ensemble import EnsembleScorer
from src.models.isolation_forest import IsolationForestModel

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_DIR = Path("models/artifacts")
REQUIRED_FIELDS = ("trans_date_trans_time", "cc_num", "amt")


class StreamProcessor:
    """
    Ties the streaming pipeline together: load models once, process transactions one-by-one,
    track metrics (latency, risk tiers, alerts, uptime).
    """

    def __init__(
        self,
        artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    ) -> None:
        """
        Initialize and load trained models from artifacts_dir.

        Args:
            artifacts_dir: Directory containing feature_store.joblib, isolation_forest.joblib,
                autoencoder.pt, ensemble.joblib.
        """
        self.artifacts_dir = Path(artifacts_dir)
        self._store: SimpleFeatureStore | None = None
        self._iso: IsolationForestModel | None = None
        self._ae: FraudAutoencoder | None = None
        self._ensemble: EnsembleScorer | None = None
        self._models_loaded = False

        self._total_processed = 0
        self._latencies_ms: list[float] = []
        self._risk_tier_counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        self._alert_count = 0
        self._start_time: float = time.perf_counter()

        self._load_models()

    def _load_models(self) -> None:
        """Load feature store and all models from artifacts."""
        base = self.artifacts_dir
        self._store = SimpleFeatureStore.load(base / "feature_store.joblib")
        self._iso = IsolationForestModel.load(base / "isolation_forest.joblib")
        self._ae = FraudAutoencoder.load(base / "autoencoder.pt")
        self._ensemble = EnsembleScorer.load(base / "ensemble.joblib")
        self._models_loaded = True
        self._start_time = time.perf_counter()
        logger.info("StreamProcessor: loaded all models from %s", self.artifacts_dir)

    def process_transaction(self, raw_transaction: dict) -> dict:
        """
        Validate input, compute features, score with iso/ae/ensemble, return scored result.

        Args:
            raw_transaction: Single transaction as dict (must include trans_date_trans_time, cc_num, amt).

        Returns:
            Dict with transaction_id, fraud_score, risk_tier, iso_score, ae_score, should_alert, processing_time_ms.

        Raises:
            ValueError: If required fields are missing.
        """
        for field in REQUIRED_FIELDS:
            if field not in raw_transaction:
                raise ValueError("Missing required field: %s" % field)

        t0 = time.perf_counter()
        features = self._store.transform_single(raw_transaction)
        iso_score = self._iso.predict_single(features)
        ae_score = self._ae.predict_single(features)
        result = self._ensemble.score_transaction(iso_score, ae_score)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        self._total_processed += 1
        self._latencies_ms.append(elapsed_ms)
        self._risk_tier_counts[result["risk_tier"]] = self._risk_tier_counts.get(result["risk_tier"], 0) + 1
        if result["should_alert"]:
            self._alert_count += 1

        out = {
            "transaction_id": str(raw_transaction.get("trans_num", "")),
            "fraud_score": result["fraud_score"],
            "risk_tier": result["risk_tier"],
            "iso_score": result["iso_score"],
            "ae_score": result["ae_score"],
            "should_alert": result["should_alert"],
            "processing_time_ms": round(elapsed_ms, 2),
        }
        return out

    def get_metrics(self) -> dict:
        """
        Return current metrics: total_processed, avg_latency_ms, p95_latency_ms,
        risk_tier_counts, alert_count, uptime_seconds.
        """
        n = len(self._latencies_ms)
        if n == 0:
            avg_ms = 0.0
            p95_ms = 0.0
        else:
            avg_ms = sum(self._latencies_ms) / n
            sorted_lat = sorted(self._latencies_ms)
            idx_95 = int(0.95 * n) if n else 0
            idx_95 = min(idx_95, n - 1)
            p95_ms = sorted_lat[idx_95]
        uptime_seconds = time.perf_counter() - self._start_time
        return {
            "total_processed": self._total_processed,
            "avg_latency_ms": round(avg_ms, 2),
            "p95_latency_ms": round(p95_ms, 2),
            "risk_tier_counts": dict(self._risk_tier_counts),
            "alert_count": self._alert_count,
            "uptime_seconds": round(uptime_seconds, 2),
        }

    def health_check(self) -> dict:
        """
        Return health status: status (healthy/unhealthy), models_loaded, avg_latency_ms, total_processed.
        """
        metrics = self.get_metrics()
        status = "healthy" if self._models_loaded else "unhealthy"
        return {
            "status": status,
            "models_loaded": self._models_loaded,
            "avg_latency_ms": metrics["avg_latency_ms"],
            "total_processed": metrics["total_processed"],
        }


def _main() -> None:
    """Initialize processor, process 100 transactions from JSONL, print metrics and health check."""
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    jsonl_path = Path("data/stream_output/transactions.jsonl")
    if not jsonl_path.exists():
        logger.error("Input file not found: %s. Run the producer first.", jsonl_path)
        return

    processor = StreamProcessor()
    count = 0
    with open(jsonl_path) as f:
        for line in f:
            if count >= 100:
                break
            line = line.strip()
            if not line:
                continue
            try:
                txn = json.loads(line)
                result = processor.process_transaction(txn)
                count += 1
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Skip transaction: %s", e)
                continue

    print("\n--- Metrics ---")
    for k, v in processor.get_metrics().items():
        print("  %s: %s" % (k, v))
    print("\n--- Health check ---")
    for k, v in processor.health_check().items():
        print("  %s: %s" % (k, v))


if __name__ == "__main__":
    _main()
