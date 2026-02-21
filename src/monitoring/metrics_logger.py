"""
MetricsLogger: append scored transactions to a JSONL file and compute aggregate metrics on demand.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PREDICTIONS_PATH = Path("data/monitoring/predictions.jsonl")
DEFAULT_MODEL_VERSION = "1.0.0"


def _percentile(sorted_values: list[float], p: float) -> float:
    """Return the p-th percentile (0-100) from a sorted list. Returns 0.0 if empty."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = min(int(p / 100.0 * n), n - 1) if n else 0
    return sorted_values[idx]


def _parse_ts(ts_str: str) -> datetime | None:
    """Parse ISO or 'YYYY-MM-DD HH:MM:SS' timestamp; return None on failure."""
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str.replace("Z", "").split(".")[0], fmt.replace(".%f", "").replace("Z", ""))
        except (ValueError, AttributeError):
            continue
    return None


class MetricsLogger:
    """
    Logs each scored transaction to a JSONL file and computes aggregate metrics by reading that file.
    """

    def __init__(
        self,
        predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> None:
        """
        Args:
            predictions_path: Path to predictions.jsonl (directory is created if needed).
            model_version: Default model_version when not provided to log_transaction.
        """
        self.predictions_path = Path(predictions_path)
        self.model_version = model_version
        self._ensure_dir()  # create data/monitoring/ on first use

    def _ensure_dir(self) -> None:
        """Create parent directory for predictions file if it does not exist."""
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory %s", self.predictions_path.parent)

    def log_transaction(
        self,
        transaction_id: str,
        fraud_score: float,
        risk_tier: str,
        processing_time_ms: float,
        *,
        timestamp: datetime | str | None = None,
        model_version: str | None = None,
    ) -> None:
        """
        Append one scored transaction to predictions.jsonl.

        Args:
            transaction_id: Transaction identifier.
            fraud_score: Ensemble fraud score.
            risk_tier: LOW, MEDIUM, or HIGH.
            processing_time_ms: Latency in milliseconds.
            timestamp: Optional log time (default: now in ISO UTC).
            model_version: Optional override for default model version.
        """
        self._ensure_dir()
        ts = timestamp
        if ts is None:
            ts = datetime.now(timezone.utc).isoformat()
        elif isinstance(ts, datetime):
            ts = ts.isoformat()
        record = {
            "transaction_id": str(transaction_id),
            "timestamp": ts,
            "fraud_score": float(fraud_score),
            "risk_tier": str(risk_tier),
            "processing_time_ms": float(processing_time_ms),
            "model_version": model_version if model_version is not None else self.model_version,
        }
        with open(self.predictions_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        logger.debug("Logged transaction %s to %s", transaction_id, self.predictions_path)

    def log_result(self, result: dict[str, Any], *, model_version: str | None = None) -> None:
        """
        Log a result dict from the stream processor or consumer (must contain
        transaction_id, fraud_score, risk_tier, processing_time_ms).
        """
        self.log_transaction(
            transaction_id=str(result["transaction_id"]),
            fraud_score=float(result["fraud_score"]),
            risk_tier=str(result["risk_tier"]),
            processing_time_ms=float(result["processing_time_ms"]),
            model_version=model_version,
        )

    def _read_predictions(self) -> list[dict[str, Any]]:
        """Read all lines from predictions.jsonl; return list of parsed dicts. Skips invalid lines."""
        records: list[dict[str, Any]] = []
        if not self.predictions_path.exists():
            return records
        with open(self.predictions_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning("Skip invalid JSON line: %s", e)
        return records

    def generate_report(self) -> dict[str, Any]:
        """
        Compute aggregate metrics from predictions.jsonl and return a single dictionary.

        Includes: total_transactions, fraud_rate_pct, avg_fraud_score, score_percentiles (p50, p90, p95, p99),
        volume_per_minute, latency (avg_ms, p50_ms, p95_ms, p99_ms), alert_rate_pct, risk_tier_breakdown.
        """
        records = self._read_predictions()
        total = len(records)
        if total == 0:
            return {
                "total_transactions": 0,
                "fraud_rate_pct": 0.0,
                "avg_fraud_score": 0.0,
                "score_percentiles": {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
                "volume_per_minute": 0.0,
                "latency_ms": {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0},
                "alert_rate_pct": 0.0,
                "risk_tier_breakdown": {"LOW": {"count": 0, "pct": 0.0}, "MEDIUM": {"count": 0, "pct": 0.0}, "HIGH": {"count": 0, "pct": 0.0}},
            }

        scores = [float(r.get("fraud_score", 0)) for r in records]
        latencies = [float(r.get("processing_time_ms", 0)) for r in records]
        risk_tiers = [str(r.get("risk_tier", "LOW")).upper() for r in records]

        high_count = sum(1 for t in risk_tiers if t == "HIGH")
        fraud_rate_pct = (high_count / total) * 100.0
        avg_fraud_score = sum(scores) / total
        sorted_scores = sorted(scores)
        score_percentiles = {
            "p50": round(_percentile(sorted_scores, 50), 4),
            "p90": round(_percentile(sorted_scores, 90), 4),
            "p95": round(_percentile(sorted_scores, 95), 4),
            "p99": round(_percentile(sorted_scores, 99), 4),
        }

        # Volume per minute: total / span in minutes (from timestamps)
        timestamps: list[datetime] = []
        for r in records:
            ts = _parse_ts(str(r.get("timestamp", "")))
            if ts is not None:
                timestamps.append(ts)
        if len(timestamps) >= 2:
            span_seconds = (max(timestamps) - min(timestamps)).total_seconds()
            span_minutes = max(span_seconds / 60.0, 1.0 / 60.0)
            volume_per_minute = total / span_minutes
        else:
            volume_per_minute = float(total) if total else 0.0

        avg_latency = sum(latencies) / total
        sorted_lat = sorted(latencies)
        latency_ms = {
            "avg": round(avg_latency, 2),
            "p50": round(_percentile(sorted_lat, 50), 2),
            "p95": round(_percentile(sorted_lat, 95), 2),
            "p99": round(_percentile(sorted_lat, 99), 2),
        }

        alert_rate_pct = (high_count / total) * 100.0
        tier_counts: dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        for t in risk_tiers:
            tier_counts[t] = tier_counts.get(t, 0) + 1
        risk_tier_breakdown: dict[str, dict[str, Any]] = {}
        for tier in ("LOW", "MEDIUM", "HIGH"):
            count = tier_counts.get(tier, 0)
            risk_tier_breakdown[tier] = {"count": count, "pct": round((count / total) * 100.0, 2)}

        return {
            "total_transactions": total,
            "fraud_rate_pct": round(fraud_rate_pct, 2),
            "avg_fraud_score": round(avg_fraud_score, 4),
            "score_percentiles": score_percentiles,
            "volume_per_minute": round(volume_per_minute, 2),
            "latency_ms": latency_ms,
            "alert_rate_pct": round(alert_rate_pct, 2),
            "risk_tier_breakdown": risk_tier_breakdown,
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    scored_path = Path("data/stream_output/scored_transactions.jsonl")
    if not scored_path.exists():
        logger.error("Scored transactions file not found: %s", scored_path)
        sys.exit(1)

    logger.info("Loading scored transactions from %s", scored_path)
    metrics_logger = MetricsLogger()
    count = 0
    with open(scored_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                metrics_logger.log_result(row)
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skip line: %s", e)

    logger.info("Logged %d transactions to %s", count, metrics_logger.predictions_path)
    report = metrics_logger.generate_report()
    print("\n--- Aggregate report ---")
    print(json.dumps(report, indent=2))
