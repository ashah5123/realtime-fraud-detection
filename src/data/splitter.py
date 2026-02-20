"""
Time-aware train/val/test split for fraud detection.
Splits chronologically to avoid future data leaking into past predictions.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.data.preprocessor import DataPreprocessor

DATETIME_COL = "trans_date_trans_time"
TARGET_COL = "is_fraud"
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15
SPLIT_PLOT_PATH = "notebooks/split_visualization.png"


class TimeAwareSplitter:
    """
    Splits data chronologically by transaction time (no random split).
    Train = first 70%, Validation = next 15%, Test = final 15%.
    Ensures no data leakage: max(train date) < min(val date) < min(test date).
    """

    def __init__(
        self,
        train_frac: float = DEFAULT_TRAIN_FRAC,
        val_frac: float = DEFAULT_VAL_FRAC,
        test_frac: float = DEFAULT_TEST_FRAC,
        datetime_col: str = DATETIME_COL,
    ) -> None:
        """
        Initialize the splitter.

        Args:
            train_frac: Fraction of data for training (by time order).
            val_frac: Fraction for validation.
            test_frac: Fraction for test.
            datetime_col: Column name for transaction datetime.
        """
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.datetime_col = datetime_col
        self._train_end_date: pd.Timestamp | None = None
        self._val_end_date: pd.Timestamp | None = None
        self._train_df: pd.DataFrame | None = None
        self._val_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None

    def split(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataframe chronologically into train, validation, and test.

        Args:
            df: Full DataFrame with datetime_col and TARGET_COL.

        Returns:
            (train_df, val_df, test_df) sorted by time.

        Raises:
            ValueError: If datetime_col or required columns are missing, or on leakage.
        """
        if self.datetime_col not in df.columns:
            raise ValueError(f"Missing datetime column: {self.datetime_col}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing target column: {TARGET_COL}")

        data = df.sort_values(self.datetime_col).reset_index(drop=True)
        n = len(data)
        n_train = int(n * self.train_frac)
        n_val = int(n * self.val_frac)
        n_test = n - n_train - n_val

        train_df = data.iloc[:n_train]
        val_df = data.iloc[n_train : n_train + n_val]
        test_df = data.iloc[n_train + n_val :]

        max_train = train_df[self.datetime_col].max()
        min_val = val_df[self.datetime_col].min()
        max_val = val_df[self.datetime_col].max()
        min_test = test_df[self.datetime_col].min()

        assert max_train < min_val, (
            f"Data leakage: max train date {max_train} >= min val date {min_val}"
        )
        assert max_val < min_test or val_df.empty or test_df.empty, (
            f"Data leakage: max val date {max_val} >= min test date {min_test}"
        )
        logger.info("No data leakage: max train < min val, max val < min test verified")

        self._train_end_date = max_train
        self._val_end_date = max_val
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df

        def _fraud_rate(d: pd.DataFrame) -> float:
            return 100 * (d[TARGET_COL].astype(int) == 1).sum() / len(d) if len(d) else 0.0

        logger.info(
            "Split sizes: train=%d, val=%d, test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        logger.info(
            "Train date range: %s to %s",
            train_df[self.datetime_col].min(),
            train_df[self.datetime_col].max(),
        )
        logger.info(
            "Val date range: %s to %s",
            val_df[self.datetime_col].min(),
            val_df[self.datetime_col].max(),
        )
        logger.info(
            "Test date range: %s to %s",
            test_df[self.datetime_col].min(),
            test_df[self.datetime_col].max(),
        )
        logger.info(
            "Fraud rate: train=%.2f%%, val=%.2f%%, test=%.2f%%",
            _fraud_rate(train_df),
            _fraud_rate(val_df),
            _fraud_rate(test_df),
        )

        return train_df, val_df, test_df

    def split_and_preprocess(
        self,
        df: pd.DataFrame,
        preprocessor: "DataPreprocessor",
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """
        Split by time, fit preprocessor on train, transform train/val/test; return X and y.

        Args:
            df: Full DataFrame (with datetime and target).
            preprocessor: DataPreprocessor instance (will be fitted on train).

        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        train_df, val_df, test_df = self.split(df)
        preprocessor.fit(train_df)
        X_train, y_train = preprocessor.transform(train_df)
        X_val, y_val = preprocessor.transform(val_df)
        X_test, y_test = preprocessor.transform(test_df)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def plot_splits(
        self,
        df: pd.DataFrame,
        save_path: str | Path = SPLIT_PLOT_PATH,
    ) -> None:
        """
        Plot fraud rate over time with vertical lines at split boundaries.
        Saves the figure to notebooks/split_visualization.png.

        Args:
            df: Full DataFrame with datetime_col and TARGET_COL (same as passed to split).
            save_path: Where to save the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plot_splits")
            return

        if self._train_end_date is None or self._val_end_date is None:
            raise ValueError("Call split() before plot_splits()")

        data = df.sort_values(self.datetime_col).reset_index(drop=True)
        data = data.copy()
        data["date"] = data[self.datetime_col].dt.date
        daily = (
            data.groupby("date")
            .agg(total=(TARGET_COL, "count"), fraud=(TARGET_COL, "sum"))
            .reset_index()
        )
        daily["fraud_rate"] = 100 * daily["fraud"] / daily["total"]
        daily["date_ts"] = pd.to_datetime(daily["date"])

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily["date_ts"], daily["fraud_rate"], color="steelblue", linewidth=1)
        ax.axvline(
            self._train_end_date,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Train end",
        )
        ax.axvline(
            self._val_end_date,
            color="orange",
            linestyle="--",
            linewidth=2,
            label="Val end",
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Fraud rate (%)")
        ax.set_title("Fraud rate over time with split boundaries")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved split visualization to %s", save_path)


def run_pipeline() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Load data, split by time, fit preprocessor on train, transform all; return X and y splits.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    from src.data.loader import load_transactions
    from src.data.preprocessor import DataPreprocessor

    df = load_transactions()
    splitter = TimeAwareSplitter()
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split_and_preprocess(
        df, preprocessor
    )
    splitter.plot_splits(df, save_path=SPLIT_PLOT_PATH)
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    X_train, X_val, X_test, y_train, y_val, y_test = run_pipeline()
    print("Shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:", X_val.shape, "y_val:", y_val.shape)
    print("  X_test:", X_test.shape, "y_test:", y_test.shape)
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        rate = 100 * (y.astype(int) == 1).sum() / len(y) if len(y) else 0
        print(f"  Fraud rate {name}: {rate:.2f}%")
