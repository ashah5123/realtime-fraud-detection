"""
Transaction feature engineering for fraud detection.
Builds amount, time, geospatial, category, and demographic features from raw data.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATETIME_COL = "trans_date_trans_time"
AMT_COL = "amt"
CC_NUM_COL = "cc_num"
CATEGORY_COL = "category"
TARGET_COL = "is_fraud"
STATE_COL = "state"
MERCH_STATE_COL = "merch_state"
AGE_GROUP_BINS = [17, 30, 45, 60, 150]
AGE_GROUP_LABELS = ["18-30", "31-45", "46-60", "60+"]
TOP_RISK_K = 5
NIGHT_START = 22
NIGHT_END = 6


def _haversine_km(
    lat1: pd.Series | np.ndarray,
    lon1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lon2: pd.Series | np.ndarray,
) -> np.ndarray:
    """Compute great-circle distance in km between (lat1, lon1) and (lat2, lon2)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(lat1)),
        np.radians(np.asarray(lon1)),
        np.radians(np.asarray(lat2)),
        np.radians(np.asarray(lon2)),
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
    return R * c


class TransactionFeatureEngineer:
    """
    Builds transaction features from raw dataframe (before preprocessing).
    fit() learns statistics from training data only; transform() applies to any data.
    """

    def __init__(self) -> None:
        """Initialize the feature engineer."""
        self._mean_amt_by_cc_: pd.Series | None = None
        self._global_mean_amt_: float = 1.0
        self._category_fraud_rate_: pd.Series | None = None
        self._global_fraud_rate_: float = 0.0
        self._high_risk_categories_: set[Any] = set()
        self._fitted_ = False

    def fit(self, df: pd.DataFrame) -> "TransactionFeatureEngineer":
        """
        Learn statistics from training data: mean amount per cc_num,
        category fraud rates, and top high-risk categories.

        Args:
            df: Training DataFrame (raw, with amt, cc_num, category, is_fraud).

        Returns:
            self (for chaining).
        """
        logger.info("Fitting TransactionFeatureEngineer on shape %s", df.shape)
        data = self._ensure_dates(df)

        # Mean amount per cc_num (for amount_to_mean_ratio)
        if AMT_COL in data.columns and CC_NUM_COL in data.columns:
            self._mean_amt_by_cc_ = data.groupby(CC_NUM_COL)[AMT_COL].mean()
            self._global_mean_amt_ = float(data[AMT_COL].mean())
            if self._global_mean_amt_ == 0:
                self._global_mean_amt_ = 1.0
            logger.info("Fitted mean amount per cc_num; global mean amt = %.4f", self._global_mean_amt_)
        else:
            self._global_mean_amt_ = 1.0

        # Category fraud rate and high-risk categories
        if CATEGORY_COL in data.columns and TARGET_COL in data.columns:
            fraud_int = (data[TARGET_COL].astype(int) == 1).astype(int)
            agg = data.assign(_fraud=fraud_int).groupby(CATEGORY_COL).agg(
                total=(TARGET_COL, "count"),
                fraud=("_fraud", "sum"),
            )
            agg["rate"] = agg["fraud"] / agg["total"].replace(0, 1)
            self._category_fraud_rate_ = agg["rate"]
            self._global_fraud_rate_ = float((data[TARGET_COL].astype(int) == 1).sum() / len(data))
            top = self._category_fraud_rate_.nlargest(TOP_RISK_K)
            self._high_risk_categories_ = set(top.index.tolist())
            logger.info(
                "Fitted category fraud rate; global = %.4f; high-risk categories: %s",
                self._global_fraud_rate_,
                self._high_risk_categories_,
            )
        else:
            self._global_fraud_rate_ = 0.0

        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the dataframe. Does not require fit() for
        non-fitted features (e.g. log_amount, time, distance, age); fit() required
        for amount_to_mean_ratio, category_fraud_rate, is_high_risk_category.

        Args:
            df: Raw DataFrame.

        Returns:
            New DataFrame with original columns plus feature columns.
        """
        out = df.copy()
        out = self._ensure_dates(out)

        # ----- Amount features -----
        if AMT_COL in out.columns:
            out["log_amount"] = np.log1p(out[AMT_COL].astype(float))
            out["amount_decimal"] = out[AMT_COL].astype(float) % 1
            out["is_round_amount"] = (out[AMT_COL].astype(float) == np.round(out[AMT_COL].astype(float))).astype(int)
        if self._mean_amt_by_cc_ is not None and AMT_COL in out.columns and CC_NUM_COL in out.columns:
            mean_amt = out[CC_NUM_COL].map(self._mean_amt_by_cc_)
            mean_amt = mean_amt.fillna(self._global_mean_amt_)
            mean_amt = mean_amt.replace(0, self._global_mean_amt_)
            out["amount_to_mean_ratio"] = out[AMT_COL].astype(float) / mean_amt
        elif AMT_COL in out.columns:
            out["amount_to_mean_ratio"] = out[AMT_COL].astype(float) / self._global_mean_amt_

        # ----- Time features -----
        if DATETIME_COL in out.columns:
            dt = pd.to_datetime(out[DATETIME_COL])
            out["hour_of_day"] = dt.dt.hour
            out["day_of_week"] = dt.dt.dayofweek
            out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
            hour = out["hour_of_day"]
            out["is_night"] = ((hour >= NIGHT_START) | (hour < NIGHT_END)).astype(int)

        # ----- Geospatial -----
        if all(c in out.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
            out["distance_to_merchant"] = _haversine_km(
                out["lat"],
                out["long"],
                out["merch_lat"],
                out["merch_long"],
            )
        if STATE_COL in out.columns and MERCH_STATE_COL in out.columns:
            out["is_foreign_state"] = (out[STATE_COL].astype(str) != out[MERCH_STATE_COL].astype(str)).astype(int)
            logger.debug("Added is_foreign_state (cardholder state vs merch_state)")
        # else: skip is_foreign_state

        # ----- Category features (require fit) -----
        if self._category_fraud_rate_ is not None and CATEGORY_COL in out.columns:
            out["category_fraud_rate"] = out[CATEGORY_COL].map(self._category_fraud_rate_).fillna(self._global_fraud_rate_)
            out["is_high_risk_category"] = out[CATEGORY_COL].isin(self._high_risk_categories_).astype(int)
        elif CATEGORY_COL in out.columns:
            out["category_fraud_rate"] = self._global_fraud_rate_
            out["is_high_risk_category"] = 0

        # ----- Demographic -----
        if "dob" in out.columns and DATETIME_COL in out.columns:
            out["age"] = (pd.to_datetime(out[DATETIME_COL]) - pd.to_datetime(out["dob"])).dt.days / 365.25
            out["age_group"] = pd.cut(
                out["age"],
                bins=AGE_GROUP_BINS,
                labels=AGE_GROUP_LABELS,
                include_lowest=True,
            )

        new_cols = [c for c in out.columns if c not in df.columns]
        logger.info("Added %d feature columns: %s", len(new_cols), new_cols)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return transformed df."""
        return self.fit(df).transform(df)

    def _ensure_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse datetime and dob if present; return a copy."""
        d = df.copy()
        if DATETIME_COL in d.columns:
            d[DATETIME_COL] = pd.to_datetime(d[DATETIME_COL])
        if "dob" in d.columns:
            d["dob"] = pd.to_datetime(d["dob"])
        return d


def _main() -> None:
    """Load data, fit on train split, transform, print new feature names and sample."""
    import logging
    from src.data.loader import load_transactions
    from src.data.splitter import TimeAwareSplitter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    df = load_transactions()
    splitter = TimeAwareSplitter()
    train_df, _val_df, _test_df = splitter.split(df)

    fe = TransactionFeatureEngineer()
    fe.fit(train_df)
    out = fe.transform(df)

    new_cols = [c for c in out.columns if c not in df.columns]
    print("New feature names:", new_cols)
    print("\nSample (first 3 rows, new features only):")
    print(out[new_cols].head(3).to_string())


if __name__ == "__main__":
    _main()
