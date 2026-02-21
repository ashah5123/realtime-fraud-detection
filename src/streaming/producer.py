"""
Transaction producer: streams transactions from CSV to Kafka or to a local JSONL file.
"""

import argparse
import json
import logging
import os
import signal
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = "data/processed/transactions_8k.csv"
DEFAULT_JSONL_PATH = "data/stream_output/transactions.jsonl"
DEFAULT_KAFKA_CONFIG_PATH = "configs/kafka_config.yaml"
DEFAULT_TOPIC = "transactions"
DEFAULT_SPEED = 1.0  # transactions per second (demo default)
SHUTDOWN = False


def _signal_handler(_signum: int, _frame: object) -> None:
    global SHUTDOWN
    SHUTDOWN = True
    logger.info("Shutdown requested (Ctrl+C)")


def _load_kafka_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


class TransactionProducer:
    """
    Reads transactions from CSV and streams them to Kafka or to a local JSONL file.
    Supports configurable speed and graceful shutdown on SIGINT.
    """

    def __init__(
        self,
        csv_path: str | Path = DEFAULT_CSV_PATH,
        kafka_config_path: str | Path = DEFAULT_KAFKA_CONFIG_PATH,
        topic: str = DEFAULT_TOPIC,
        jsonl_path: str | Path = DEFAULT_JSONL_PATH,
    ) -> None:
        """
        Initialize the producer.

        Args:
            csv_path: Path to transactions CSV.
            kafka_config_path: Path to kafka_config.yaml (for bootstrap_servers, topic, speed).
            topic: Kafka topic name (used when in Kafka mode).
            jsonl_path: Path to output JSONL file (used when in local mode).
        """
        self.csv_path = Path(csv_path)
        self.kafka_config_path = Path(kafka_config_path)
        self.topic = topic
        self.jsonl_path = Path(jsonl_path)
        self._producer = None
        self._df: pd.DataFrame | None = None

    def _load_data(self) -> pd.DataFrame:
        """Load CSV and return DataFrame; parse datetime for JSON serialization."""
        if not self.csv_path.exists():
            raise FileNotFoundError("Transactions CSV not found: %s" % self.csv_path)
        df = pd.read_csv(self.csv_path)
        if "trans_date_trans_time" in df.columns:
            df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"]).astype(str)
        if "dob" in df.columns:
            df["dob"] = pd.to_datetime(df["dob"]).astype(str)
        self._df = df
        logger.info("Loaded %d transactions from %s", len(df), self.csv_path)
        return df

    def _row_to_dict(self, row: pd.Series) -> dict:
        """Convert one row to a JSON-serializable dict (native Python types)."""
        d = row.to_dict()
        out = {}
        for k, v in d.items():
            if pd.isna(v):
                out[k] = None
            elif isinstance(v, (np.integer, np.int64, np.int32)):
                out[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, np.float32)):
                out[k] = float(v)
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    def _get_speed_from_config(self) -> float:
        """Transactions per second from config (speed_multiplier; demo_mode => 1)."""
        cfg = _load_kafka_config(self.kafka_config_path)
        prod = cfg.get("producer", {})
        if prod.get("demo_mode", True):
            return 1.0
        return float(prod.get("speed_multiplier", DEFAULT_SPEED))

    def run_local(
        self,
        limit: int | None = None,
        speed: float | None = None,
    ) -> int:
        """
        Write transactions to local JSONL file, one per line, with simulated timing.

        Args:
            limit: Max number of transactions to send (None = all).
            speed: Transactions per second (None = from config or default).

        Returns:
            Number of transactions written.
        """
        global SHUTDOWN
        df = self._load_data()
        n_total = len(df)
        n_send = min(limit, n_total) if limit is not None else n_total
        if speed is None:
            speed = self._get_speed_from_config()
        interval = 1.0 / speed if speed > 0 else 0

        os.makedirs(self.jsonl_path.parent, exist_ok=True)
        count = 0
        with open(self.jsonl_path, "w") as f:
            for i in range(n_send):
                if SHUTDOWN:
                    logger.info("Stopping after %d transactions (shutdown)", count)
                    break
                row = df.iloc[i]
                obj = self._row_to_dict(row)
                f.write(json.dumps(obj, default=str) + "\n")
                count += 1
                if count % 100 == 0:
                    logger.info("Progress: %d / %d transactions", count, n_send)
                if interval > 0 and i < n_send - 1:
                    time.sleep(interval)
        logger.info("Wrote %d transactions to %s", count, self.jsonl_path)
        return count

    def run_kafka(
        self,
        limit: int | None = None,
        speed: float | None = None,
    ) -> int:
        """
        Send transactions to Kafka topic as JSON, with simulated timing.

        Args:
            limit: Max number of transactions to send (None = all).
            speed: Transactions per second (None = from config or default).

        Returns:
            Number of messages sent.
        """
        from confluent_kafka import Producer as KafkaProducer

        global SHUTDOWN
        cfg = _load_kafka_config(self.kafka_config_path)
        bootstrap = cfg.get("bootstrap_servers", "localhost:9092")
        topic = cfg.get("topics", {}).get("transactions", self.topic)

        df = self._load_data()
        n_total = len(df)
        n_send = min(limit, n_total) if limit is not None else n_total
        if speed is None:
            speed = self._get_speed_from_config()
        interval = 1.0 / speed if speed > 0 else 0

        conf = {"bootstrap.servers": bootstrap}
        producer = KafkaProducer(conf)
        count = 0

        def _on_delivery(err, msg):
            if err:
                logger.error("Delivery failed: %s", err)

        try:
            for i in range(n_send):
                if SHUTDOWN:
                    logger.info("Stopping after %d transactions (shutdown)", count)
                    break
                row = df.iloc[i]
                obj = self._row_to_dict(row)
                payload = json.dumps(obj, default=str)
                producer.produce(topic, value=payload.encode("utf-8"), callback=_on_delivery)
                producer.poll(0)
                count += 1
                if count % 100 == 0:
                    producer.flush()
                    logger.info("Progress: %d / %d transactions", count, n_send)
                if interval > 0 and i < n_send - 1:
                    time.sleep(interval)
            producer.flush()
        finally:
            producer.flush()
        logger.info("Sent %d transactions to Kafka topic %s", count, topic)
        return count

    def run(
        self,
        use_kafka: bool = False,
        limit: int | None = None,
        speed: float | None = None,
    ) -> int:
        """
        Run producer in Kafka or local mode.

        Args:
            use_kafka: If True, send to Kafka; else write to JSONL.
            limit: Max transactions (None = all).
            speed: Transactions per second (None = from config).

        Returns:
            Number of transactions sent/written.
        """
        if use_kafka:
            return self.run_kafka(limit=limit, speed=speed)
        return self.run_local(limit=limit, speed=speed)


def _main() -> None:
    global SHUTDOWN
    signal.signal(signal.SIGINT, _signal_handler)

    parser = argparse.ArgumentParser(description="Stream transactions from CSV to Kafka or local JSONL")
    parser.add_argument(
        "--kafka",
        action="store_true",
        help="Send to Kafka (default: write to local JSONL)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Write to local JSONL file (default if neither --kafka nor --local)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Send only N transactions (default: all)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        metavar="TPS",
        help="Transactions per second (override config)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to transactions CSV",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=DEFAULT_JSONL_PATH,
        help="Output JSONL path (local mode)",
    )
    args = parser.parse_args()

    use_kafka = args.kafka
    if not args.kafka and not args.local:
        use_kafka = False  # default local

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    producer = TransactionProducer(
        csv_path=args.csv,
        jsonl_path=args.jsonl,
    )
    try:
        n = producer.run(use_kafka=use_kafka, limit=args.limit, speed=args.speed)
        print("Done: %d transactions" % n)
    except KeyboardInterrupt:
        print("Interrupted")
    except FileNotFoundError as e:
        logger.error("%s", e)
        raise


if __name__ == "__main__":
    _main()
