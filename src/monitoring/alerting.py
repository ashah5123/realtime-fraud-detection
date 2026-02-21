"""
AlertManager: high-risk transaction alerts and drift alerts with configurable channels and rate limiting.
"""

import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ALERT_LOG_PATH = Path("data/monitoring/alerts.log")
HIGH_RISK_THRESHOLD = 0.7
RATE_LIMIT_MAX_PER_MINUTE = 10
ALERT_HISTORY_MAX = 100

# ANSI colors for console
ANSI_RED = "\033[91m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class AlertManager:
    """
    Sends alerts for high-risk transactions (fraud_score > 0.7) and for drift (WARNING/CRITICAL).
    Supports console, log file, and optional webhook. Rate-limited to 10 alerts per minute.
    """

    def __init__(
        self,
        *,
        console: bool = True,
        log_file: str | Path | None = DEFAULT_ALERT_LOG_PATH,
        webhook_url: str | None = None,
        high_risk_threshold: float = HIGH_RISK_THRESHOLD,
        rate_limit_per_minute: int = RATE_LIMIT_MAX_PER_MINUTE,
    ) -> None:
        """
        Args:
            console: If True, print formatted alerts to stdout (color-coded).
            log_file: Path to append alerts (None to disable).
            webhook_url: Optional URL to POST JSON payload (e.g. Slack webhook).
            high_risk_threshold: Alert when fraud_score > this value.
            rate_limit_per_minute: Max transaction alerts per minute.
        """
        self.console = console
        self.log_file = Path(log_file) if log_file else None
        self.webhook_url = webhook_url
        self.high_risk_threshold = high_risk_threshold
        self.rate_limit_per_minute = rate_limit_per_minute
        self._alert_timestamps: list[float] = []
        self._history: deque[dict[str, Any]] = deque(maxlen=ALERT_HISTORY_MAX)

    def _within_rate_limit(self) -> bool:
        """Return True if we can send another transaction alert (under the per-minute cap)."""
        now = time.monotonic()
        cutoff = now - 60.0
        self._alert_timestamps = [t for t in self._alert_timestamps if t > cutoff]
        return len(self._alert_timestamps) < self.rate_limit_per_minute

    def _record_alert_sent(self) -> None:
        self._alert_timestamps.append(time.monotonic())

    def _add_to_history(self, alert_type: str, payload: dict[str, Any]) -> None:
        self._history.append({
            "type": alert_type,
            "payload": payload,
            "at": datetime.now(timezone.utc).isoformat(),
        })

    def _risk_color(self, risk_tier: str) -> str:
        if risk_tier.upper() == "HIGH":
            return ANSI_RED
        if risk_tier.upper() == "MEDIUM":
            return ANSI_YELLOW
        return ANSI_RESET

    def _format_transaction_alert(self, data: dict[str, Any], use_color: bool = True) -> str:
        """Build human-readable alert text for a high-risk transaction."""
        amount = data.get("amount") or data.get("amt") or 0
        fraud_score = data.get("fraud_score", 0)
        iso_score = data.get("iso_score", 0)
        ae_score = data.get("ae_score", 0)
        risk_tier = data.get("risk_tier", "HIGH")
        ts = data.get("timestamp", datetime.now(timezone.utc).isoformat())
        tid = data.get("transaction_id", "")

        color = self._risk_color(risk_tier) if use_color else ""
        reset = ANSI_RESET if use_color else ""

        lines = [
            "%s--- HIGH-RISK TRANSACTION ALERT ---%s" % (color, reset),
            "  transaction_id: %s" % tid,
            "  amount: %.2f" % float(amount),
            "  fraud_score: %.4f" % fraud_score,
            "  iso_score: %.4f  ae_score: %.4f" % (iso_score, ae_score),
            "  risk_tier: %s" % risk_tier,
            "  timestamp: %s" % ts,
        ]
        return "\n".join(lines)

    def _send_console(self, text: str) -> None:
        if self.console:
            print(text)

    def _send_log_file(self, text: str) -> None:
        if self.log_file:
            _ensure_dir(self.log_file)
            with open(self.log_file, "a") as f:
                f.write(text + "\n")

    def _send_webhook(self, payload: dict[str, Any]) -> None:
        if not self.webhook_url:
            return
        try:
            import urllib.request
            import json as _json
            req = urllib.request.Request(
                self.webhook_url,
                data=_json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status >= 400:
                    logger.warning("Webhook returned status %s", resp.status)
        except Exception as e:
            logger.warning("Webhook POST failed: %s", e)

    def alert_transaction(self, transaction: dict[str, Any]) -> bool:
        """
        If fraud_score > high_risk_threshold, send alert via configured channels (subject to rate limit).

        Args:
            transaction: Dict with at least fraud_score, and preferably amount/amt, iso_score, ae_score, risk_tier, timestamp, transaction_id.

        Returns:
            True if alert was sent, False if skipped (below threshold or rate limited).
        """
        fraud_score = float(transaction.get("fraud_score", 0))
        if fraud_score <= self.high_risk_threshold:
            return False

        if not self._within_rate_limit():
            logger.warning("Rate limit exceeded (%d alerts/min); skipping transaction alert", self.rate_limit_per_minute)
            return False

        self._record_alert_sent()
        risk_tier = str(transaction.get("risk_tier", "HIGH"))
        text_console = self._format_transaction_alert(transaction, use_color=True)
        text_plain = self._format_transaction_alert(transaction, use_color=False)

        self._send_console(text_console)
        self._send_log_file(text_plain)
        self._send_webhook({
            "type": "high_risk_transaction",
            "risk_tier": risk_tier,
            "fraud_score": fraud_score,
            "transaction": {k: v for k, v in transaction.items() if k in ("transaction_id", "amount", "amt", "fraud_score", "iso_score", "ae_score", "risk_tier", "timestamp")},
        })

        self._add_to_history("transaction", transaction)
        logger.info("Alert sent for high-risk transaction %s (score=%.2f)", transaction.get("transaction_id"), fraud_score)
        return True

    def alert_drift(self, drift_report: dict[str, Any]) -> bool:
        """
        Send alert if drift overall_status is WARNING or CRITICAL.

        Args:
            drift_report: Dict from DriftDetector.detect() with overall_status, feature_drift, prediction_drift, features_with_drift.

        Returns:
            True if alert was sent, False if status is OK (no alert).
        """
        status = str(drift_report.get("overall_status", "OK")).upper()
        if status not in ("WARNING", "CRITICAL"):
            return False

        # Drift alerts are not rate-limited by transaction limit (separate concern)
        color = ANSI_RED if status == "CRITICAL" else ANSI_YELLOW
        lines = [
            "%s--- DRIFT ALERT ---%s" % (color, ANSI_RESET),
            "  overall_status: %s" % status,
            "  features_with_drift: %s" % (drift_report.get("features_with_drift") or []),
        ]
        pred = drift_report.get("prediction_drift", {})
        if pred:
            lines.append("  prediction_drift PSI: %.4f" % pred.get("psi", 0))
        text = "\n".join(lines)

        self._send_console(text)
        self._send_log_file(text.replace(ANSI_RED, "").replace(ANSI_YELLOW, "").replace(ANSI_RESET, ""))
        self._send_webhook({"type": "drift", "overall_status": status, "drift_report": drift_report})

        self._add_to_history("drift", drift_report)
        logger.info("Drift alert sent: status=%s", status)
        return True

    def get_alert_history(self) -> list[dict[str, Any]]:
        """Return the last up to 100 alerts (each with type, payload, at)."""
        return list(self._history)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    # Console + log file (no webhook)
    manager = AlertManager(console=True, log_file=DEFAULT_ALERT_LOG_PATH)

    # Sample high-risk transactions (fraud_score > 0.7)
    samples = [
        {
            "transaction_id": "txn-001",
            "amount": 500.00,
            "fraud_score": 0.85,
            "iso_score": 0.82,
            "ae_score": 0.88,
            "risk_tier": "HIGH",
            "timestamp": "2024-01-15T14:30:00Z",
        },
        {
            "transaction_id": "txn-002",
            "amt": 1200.50,
            "fraud_score": 0.72,
            "iso_score": 0.70,
            "ae_score": 0.74,
            "risk_tier": "HIGH",
            "timestamp": "2024-01-15T14:31:00Z",
        },
        {
            "transaction_id": "txn-003",
            "amount": 89.99,
            "fraud_score": 0.65,
            "iso_score": 0.60,
            "ae_score": 0.70,
            "risk_tier": "MEDIUM",
            "timestamp": "2024-01-15T14:32:00Z",
        },
    ]

    print("--- Sending sample transaction alerts (console + log file) ---")
    for t in samples:
        sent = manager.alert_transaction(t)
        print("  alert_transaction(%s) -> %s" % (t["transaction_id"], sent))

    print("\n--- Rate limit test: send 15 high-risk alerts rapidly ---")
    sent_count = 0
    for i in range(15):
        t = {
            "transaction_id": "rate-test-%02d" % (i + 1),
            "amount": 100.0 * (i + 1),
            "fraud_score": 0.75 + i * 0.01,
            "iso_score": 0.74,
            "ae_score": 0.76,
            "risk_tier": "HIGH",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if manager.alert_transaction(t):
            sent_count += 1
    print("  Sent %d/15 (max %d/min); rest rate-limited" % (sent_count, manager.rate_limit_per_minute))

    print("\n--- Drift alert test ---")
    drift_warning = {
        "overall_status": "WARNING",
        "features_with_drift": ["amount_zscore", "time_since_last_txn"],
        "prediction_drift": {"psi": 0.15, "status": "MODERATE"},
        "feature_drift": {},
    }
    drift_critical = {
        "overall_status": "CRITICAL",
        "features_with_drift": ["time_since_last_txn"],
        "prediction_drift": {"psi": 0.30, "status": "SIGNIFICANT"},
        "feature_drift": {},
    }
    manager.alert_drift(drift_warning)
    manager.alert_drift(drift_critical)

    print("\n--- Alert history (last %d) ---" % len(manager.get_alert_history()))
    for i, h in enumerate(manager.get_alert_history()[-5:]):
        print("  [%d] %s %s" % (i, h["type"], h["at"]))
    print("Alerts log appended to %s" % manager.log_file)
