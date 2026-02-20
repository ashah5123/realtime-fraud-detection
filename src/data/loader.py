"""
Data loader for the fraud detection transactions dataset.
Loads, parses, validates, and returns a clean DataFrame.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "city",
    "state",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "merch_lat",
    "merch_long",
    "is_fraud",
]


class DataLoader:
    """
    Loads the sampled transactions CSV, parses dates, validates integrity,
    and returns a clean DataFrame with logged statistics.
    """

    def __init__(self, path: str | Path = "data/processed/transactions_8k.csv") -> None:
        """
        Initialize the loader with the path to the transactions CSV.

        Args:
            path: Path to the CSV file (default: data/processed/transactions_8k.csv).
        """
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        """
        Load the dataset, parse datetime, validate integrity, log stats, and return a clean DataFrame.

        Returns:
            Clean pandas DataFrame with trans_date_trans_time as datetime and validated data.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing or validation fails.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")

        df = pd.read_csv(self.path)
        logger.info("Loaded raw dataset from %s", self.path)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
            logger.debug("Dropped column 'Unnamed: 0'")

        df = self._parse_datetimes(df)
        self._validate_integrity(df)
        self._log_statistics(df)

        return df

    def _parse_datetimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse trans_date_trans_time and dob as datetime.

        Args:
            df: Raw DataFrame.

        Returns:
            DataFrame with datetime columns parsed.
        """
        df = df.copy()
        if "trans_date_trans_time" in df.columns:
            df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
            logger.debug("Parsed trans_date_trans_time as datetime")
        if "dob" in df.columns:
            df["dob"] = pd.to_datetime(df["dob"])
            logger.debug("Parsed dob as datetime")
        return df

    def _validate_integrity(self, df: pd.DataFrame) -> None:
        """
        Check for missing columns, nulls, duplicates, and basic data types.
        Logs warnings for issues, raises ValueError on critical failures, and
        deduplicates by trans_num in place if duplicates are found.

        Args:
            df: DataFrame to validate.

        Raises:
            ValueError: If required columns are missing or critical nulls/duplicates found.
        """
        # Required columns
        missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Nulls
        null_counts = df[EXPECTED_COLUMNS].isnull().sum()
        null_cols = null_counts[null_counts > 0]
        if len(null_cols) > 0:
            for col, count in null_cols.items():
                logger.warning("Column %s has %d null values", col, int(count))
            # Optionally raise if key columns have nulls
            key_cols = ["trans_date_trans_time", "cc_num", "amt", "is_fraud"]
            if any(null_counts.get(c, 0) > 0 for c in key_cols):
                raise ValueError("Critical columns contain nulls; cannot proceed.")

        # Duplicates (e.g. on transaction id if we had one; trans_num should be unique)
        if "trans_num" in df.columns:
            n_dup = df["trans_num"].duplicated().sum()
            if n_dup > 0:
                logger.warning("Found %d duplicate trans_num rows; dropping duplicates", int(n_dup))
                df.drop_duplicates(subset=["trans_num"], inplace=True)

        # Data types: numeric columns
        numeric_cols = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "is_fraud"]
        for col in numeric_cols:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning("Column %s is not numeric (dtype=%s); may need conversion", col, df[col].dtype)

        logger.info("Data integrity validation completed")

    def _log_statistics(self, df: pd.DataFrame) -> None:
        """
        Log dataset shape, fraud ratio, memory usage, and column types.

        Args:
            df: DataFrame to summarize.
        """
        n_rows, n_cols = df.shape
        logger.info("Dataset shape: (%d, %d)", n_rows, n_cols)

        if "is_fraud" in df.columns:
            fraud_count = int((df["is_fraud"].astype(int) == 1).sum())
            fraud_ratio = 100 * fraud_count / n_rows if n_rows else 0
            logger.info("Fraud count: %d, fraud ratio: %.2f%%", fraud_count, fraud_ratio)

        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info("Memory usage: %.2f MB", mem_mb)

        for col in df.columns:
            logger.debug("Column %s: %s", col, df[col].dtype)


def load_transactions(path: str | Path = "data/processed/transactions_8k.csv") -> pd.DataFrame:
    """
    Convenience function to load and return the transactions dataset.

    Args:
        path: Path to the CSV file.

    Returns:
        Clean pandas DataFrame.
    """
    loader = DataLoader(path=path)
    return loader.load()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    df = load_transactions()
    # Summary stats for __main__ (user asked to "prints summary stats")
    print("Summary stats:")
    print(f"  Shape: {df.shape}")
    if "is_fraud" in df.columns:
        fraud_count = (df["is_fraud"].astype(int) == 1).sum()
        print(f"  Fraud count: {fraud_count}, ratio: {100 * fraud_count / len(df):.2f}%")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"  Columns: {list(df.columns)}")
