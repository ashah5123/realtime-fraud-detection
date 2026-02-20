"""
Velocity and frequency features per card (cc_num).
All features computed without future leakage: only data at or before current transaction time.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATETIME_COL = "trans_date_trans_time"
CC_NUM_COL = "cc_num"
AMT_COL = "amt"
WINDOW_10MIN_SEC = 600
WINDOW_1H_SEC = 3600
WINDOW_24H_SEC = 86400
RAPID_FIRE_COUNT_THRESH = 3
MIN_TXN_FOR_CARD_STD = 3


def _compute_group_velocity(
    times: np.ndarray,
    amounts: np.ndarray,
    global_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For one card's sorted (by time) transactions, compute velocity features row by row.
    No future leakage: for row i only use times/amounts[0:i+1] with time <= times[i].
    """
    n = len(times)
    time_since_last = np.full(n, np.nan, dtype=float)
    rapid_fire = np.zeros(n, dtype=int)
    txn_count_1h = np.zeros(n, dtype=int)
    txn_count_24h = np.zeros(n, dtype=int)
    avg_amount_24h = np.full(n, np.nan, dtype=float)
    amount_zscore = np.full(n, np.nan, dtype=float)

    for i in range(n):
        t_cur = times[i]
        amt_cur = amounts[i]
        times_cur = times[: i + 1]
        amounts_cur = amounts[: i + 1]

        # Time since last transaction (seconds)
        if i > 0:
            delta = (t_cur - times[i - 1])
            try:
                time_since_last[i] = delta / np.timedelta64(1, "s")
            except TypeError:
                time_since_last[i] = getattr(delta, "total_seconds", lambda: float(delta))()

        # Deltas in seconds (for window comparisons)
        diff = t_cur - times_cur
        if hasattr(diff, "astype"):
            deltas_sec = np.asarray(diff).astype("timedelta64[s]").astype(np.float64)
        else:
            deltas_sec = np.array([getattr(d, "total_seconds", lambda: float(d))() for d in diff])
        # Count in last 10 min (including current)
        in_10m = deltas_sec <= WINDOW_10MIN_SEC
        count_10m = in_10m.sum()
        rapid_fire[i] = 1 if count_10m > RAPID_FIRE_COUNT_THRESH else 0

        # Count in last 1h and 24h
        in_1h = deltas_sec <= WINDOW_1H_SEC
        in_24h = deltas_sec <= WINDOW_24H_SEC
        txn_count_1h[i] = in_1h.sum()
        txn_count_24h[i] = in_24h.sum()

        # Avg amount in last 24h
        if in_24h.any():
            avg_amount_24h[i] = amounts_cur[in_24h].mean()

        # Amount zscore: (current - mean_past) / std_past; use global_std if < 3 past txns
        past_amounts = amounts[:i]
        if len(past_amounts) == 0:
            mean_past = amt_cur
            std_used = global_std if global_std > 0 else 1.0
        else:
            mean_past = float(np.mean(past_amounts))
            if len(past_amounts) >= MIN_TXN_FOR_CARD_STD:
                std_past = float(np.std(past_amounts))
                std_used = std_past if std_past > 0 else global_std
            else:
                std_used = global_std if global_std > 0 else 1.0
        amount_zscore[i] = (amt_cur - mean_past) / std_used

    return (
        time_since_last,
        rapid_fire,
        txn_count_1h,
        txn_count_24h,
        avg_amount_24h,
        amount_zscore,
    )


class VelocityFeatureEngineer:
    """
    Computes per-card velocity and frequency features from raw transaction data.
    fit() learns global statistics (e.g. global std for amount zscore fallback).
    transform() computes features in time order without future leakage.
    """

    def __init__(self) -> None:
        """Initialize the velocity feature engineer."""
        self._global_std_amt_: float = 1.0
        self._fitted_ = False

    def fit(self, df: pd.DataFrame) -> "VelocityFeatureEngineer":
        """
        Learn global statistics from training data (e.g. global std of amount).

        Args:
            df: Training DataFrame with trans_date_trans_time, cc_num, amt.

        Returns:
            self (for chaining).
        """
        if DATETIME_COL not in df.columns or CC_NUM_COL not in df.columns or AMT_COL not in df.columns:
            raise ValueError("DataFrame must have trans_date_trans_time, cc_num, amt")
        data = df.copy()
        data[DATETIME_COL] = pd.to_datetime(data[DATETIME_COL])
        amt = data[AMT_COL].astype(float)
        self._global_std_amt_ = float(amt.std())
        if self._global_std_amt_ == 0 or np.isnan(self._global_std_amt_):
            self._global_std_amt_ = 1.0
        logger.info("Fitted VelocityFeatureEngineer: global std(amt) = %.4f", self._global_std_amt_)
        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add velocity/frequency features. Data must be sortable by time.
        Computes per-card features in time order with no future leakage.

        Args:
            df: Raw DataFrame with trans_date_trans_time, cc_num, amt.

        Returns:
            New DataFrame with original columns plus velocity feature columns.
        """
        if DATETIME_COL not in df.columns or CC_NUM_COL not in df.columns or AMT_COL not in df.columns:
            raise ValueError("DataFrame must have trans_date_trans_time, cc_num, amt")
        global_std = self._global_std_amt_ if self._fitted_ else 1.0

        out = df.copy()
        out[DATETIME_COL] = pd.to_datetime(out[DATETIME_COL])
        # Preserve original index for reordering later
        out["_orig_idx"] = np.arange(len(out))
        out_sorted = out.sort_values([CC_NUM_COL, DATETIME_COL]).reset_index(drop=True)

        time_since_last_list: list[float] = []
        rapid_fire_list: list[int] = []
        txn_count_1h_list: list[int] = []
        txn_count_24h_list: list[int] = []
        avg_amount_24h_list: list[float] = []
        amount_zscore_list: list[float] = []

        for _cc, grp in out_sorted.groupby(CC_NUM_COL, sort=False):
            times = grp[DATETIME_COL].values
            amounts = grp[AMT_COL].astype(float).values
            (
                time_since_last,
                rapid_fire,
                txn_count_1h,
                txn_count_24h,
                avg_amount_24h,
                amount_zscore,
            ) = _compute_group_velocity(times, amounts, global_std)
            time_since_last_list.extend(time_since_last.tolist())
            rapid_fire_list.extend(rapid_fire.tolist())
            txn_count_1h_list.extend(txn_count_1h.tolist())
            txn_count_24h_list.extend(txn_count_24h.tolist())
            avg_amount_24h_list.extend(avg_amount_24h.tolist())
            amount_zscore_list.extend(amount_zscore.tolist())

        out_sorted["time_since_last_txn"] = time_since_last_list
        out_sorted["rapid_fire_flag"] = rapid_fire_list
        out_sorted["txn_count_1h"] = txn_count_1h_list
        out_sorted["txn_count_24h"] = txn_count_24h_list
        out_sorted["avg_amount_24h"] = avg_amount_24h_list
        out_sorted["amount_zscore"] = amount_zscore_list
        # Restore original row order
        out = out_sorted.sort_values("_orig_idx").drop(columns=["_orig_idx"]).reset_index(drop=True)

        new_cols = [
            "time_since_last_txn",
            "rapid_fire_flag",
            "txn_count_1h",
            "txn_count_24h",
            "avg_amount_24h",
            "amount_zscore",
        ]
        logger.info("Added velocity feature columns: %s", new_cols)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return transformed df."""
        return self.fit(df).transform(df)


def _main() -> None:
    """Load data, split by time, fit on train, transform train, print new feature names and sample."""
    import logging
    from src.data.loader import load_transactions
    from src.data.splitter import TimeAwareSplitter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    df = load_transactions()
    splitter = TimeAwareSplitter()
    train_df, _val_df, _test_df = splitter.split(df)

    fe = VelocityFeatureEngineer()
    fe.fit(train_df)
    out = fe.transform(train_df)

    new_cols = [
        "time_since_last_txn",
        "rapid_fire_flag",
        "txn_count_1h",
        "txn_count_24h",
        "avg_amount_24h",
        "amount_zscore",
    ]
    print("New feature names:", new_cols)
    print("\nSample (first 5 rows, new features only):")
    print(out[new_cols].head(5).to_string())


if __name__ == "__main__":
    _main()
