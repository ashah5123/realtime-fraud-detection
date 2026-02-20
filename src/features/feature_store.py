"""
Feature store that combines TransactionFeatureEngineer and VelocityFeatureEngineer.
Provides fit/transform, save/load, and single-transaction scoring.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.transaction_features import TransactionFeatureEngineer
from src.features.velocity_features import VelocityFeatureEngineer

logger = logging.getLogger(__name__)

TARGET_COL = "is_fraud"
NUMERIC_ORIGINALS = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long"]
DEFAULT_ARTIFACT_PATH = "models/artifacts/feature_store.joblib"

# Fixed order for transaction + velocity feature names (excluding numeric originals)
TRANSACTION_FEATURE_NAMES = [
    "log_amount",
    "amount_decimal",
    "is_round_amount",
    "amount_to_mean_ratio",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "distance_to_merchant",
    "category_fraud_rate",
    "is_high_risk_category",
    "age",
    "age_group",
    "is_foreign_state",
]
VELOCITY_FEATURE_NAMES = [
    "time_since_last_txn",
    "rapid_fire_flag",
    "txn_count_1h",
    "txn_count_24h",
    "avg_amount_24h",
    "amount_zscore",
]


class SimpleFeatureStore:
    """
    Ties together TransactionFeatureEngineer and VelocityFeatureEngineer.
    fit(train_df) fits both on training data; transform(df) returns (X, y) feature matrix and target.
    """

    def __init__(self, artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> None:
        """
        Initialize the feature store.

        Args:
            artifact_path: Path to save/load fitted state (joblib).
        """
        self.artifact_path = Path(artifact_path)
        self._txn_engineer = TransactionFeatureEngineer()
        self._velocity_engineer = VelocityFeatureEngineer()
        self._feature_names: list[str] = []
        self._fitted_ = False

    def fit(self, train_df: pd.DataFrame) -> "SimpleFeatureStore":
        """
        Fit both feature engineers on training data and save fitted state.

        Args:
            train_df: Training DataFrame (raw, with required columns).

        Returns:
            self (for chaining).
        """
        logger.info("Fitting SimpleFeatureStore on train shape %s", train_df.shape)
        self._txn_engineer.fit(train_df)
        self._velocity_engineer.fit(train_df)
        # Compute feature names from one row transform
        _sample = self._txn_engineer.transform(train_df.head(1))
        _sample = self._velocity_engineer.transform(_sample)
        self._feature_names = self._resolve_feature_names(_sample)
        self._fitted_ = True
        logger.info("Feature store fitted; %d features", len(self._feature_names))
        return self

    def _resolve_feature_names(self, df: pd.DataFrame) -> list[str]:
        """Build ordered list of feature names: numeric originals + transaction + velocity."""
        out: list[str] = []
        for c in NUMERIC_ORIGINALS:
            if c in df.columns:
                out.append(c)
        for c in TRANSACTION_FEATURE_NAMES:
            if c in df.columns:
                out.append(c)
        for c in VELOCITY_FEATURE_NAMES:
            if c in df.columns:
                out.append(c)
        return out

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply both feature engineers, keep only modeling columns, return (X, y).

        Keeps: numeric originals (amt, lat, long, city_pop, merch_lat, merch_long)
        plus all engineered features. Drops other original columns.

        Args:
            df: Raw DataFrame.

        Returns:
            (X, y): X = feature matrix (DataFrame), y = is_fraud.
        """
        if not self._fitted_:
            raise ValueError("Feature store not fitted. Call fit(train_df) first.")
        combined = self._txn_engineer.transform(df)
        combined = self._velocity_engineer.transform(combined)

        # Extract target
        y = combined[TARGET_COL].copy() if TARGET_COL in combined.columns else pd.Series(dtype=float)

        # Keep only feature columns (use fitted order if available)
        if self._feature_names:
            use_cols = [c for c in self._feature_names if c in combined.columns]
        else:
            use_cols = [c for c in NUMERIC_ORIGINALS if c in combined.columns]
            use_cols += [c for c in TRANSACTION_FEATURE_NAMES + VELOCITY_FEATURE_NAMES if c in combined.columns]
        X = combined[use_cols].copy()

        # Encode age_group to numeric for consistent matrix (0,1,2,3)
        if "age_group" in X.columns:
            if isinstance(X["age_group"].dtype, pd.CategoricalDtype):
                X["age_group"] = X["age_group"].cat.codes
            else:
                mapping = {"18-30": 0, "31-45": 1, "46-60": 2, "60+": 3}
                X["age_group"] = X["age_group"].astype(str).map(mapping).fillna(-1).astype(int)

        # Fill NaN for numeric columns (e.g. time_since_last_txn for first txn)
        X = X.fillna(0)
        logger.info("Transform output X shape %s, y shape %s", X.shape, y.shape)
        return X, y

    def get_feature_names(self) -> list[str]:
        """
        Return the list of feature names in the final output (same order as transform).

        Returns:
            List of column names for X.
        """
        return list(self._feature_names) if self._feature_names else []

    def transform_single(self, transaction: dict) -> np.ndarray:
        """
        For real-time scoring: take one transaction as dict, return feature vector.

        Args:
            transaction: Single row as dict (keys = column names). Should contain
                all raw columns required by the feature engineers.

        Returns:
            1D numpy array of feature values in get_feature_names() order.
            Missing columns are filled with 0.
        """
        if not self._fitted_:
            raise ValueError("Feature store not fitted. Call fit(train_df) first.")
        df = pd.DataFrame([transaction])
        X, _ = self.transform(df)
        names = self.get_feature_names()
        vec = np.zeros(len(names), dtype=np.float64)
        for i, c in enumerate(names):
            if c in X.columns:
                val = X[c].iloc[0]
                vec[i] = float(pd.Series([val]).fillna(0).iloc[0])
        return vec

    def save(self, path: str | Path | None = None) -> None:
        """
        Save fitted feature store (both engineers + feature names) using joblib.

        Args:
            path: Override default artifact path.
        """
        path = Path(path) if path is not None else self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "_txn_engineer": self._txn_engineer,
            "_velocity_engineer": self._velocity_engineer,
            "_feature_names": self._feature_names,
            "_fitted_": self._fitted_,
        }
        joblib.dump(state, path)
        logger.info("Saved feature store to %s", path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> "SimpleFeatureStore":
        """
        Load a fitted feature store from disk.

        Args:
            path: Path to the joblib file.

        Returns:
            Loaded SimpleFeatureStore instance.
        """
        path = Path(path)
        state = joblib.load(path)
        obj = cls(artifact_path=path)
        obj._txn_engineer = state["_txn_engineer"]
        obj._velocity_engineer = state["_velocity_engineer"]
        obj._feature_names = state["_feature_names"]
        obj._fitted_ = state["_fitted_"]
        logger.info("Loaded feature store from %s", path)
        return obj


def _main() -> None:
    """Load data, split, fit store on train, transform train/val/test, print feature names and shapes."""
    import logging
    from src.data.loader import load_transactions
    from src.data.splitter import TimeAwareSplitter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    df = load_transactions()
    splitter = TimeAwareSplitter()
    train_df, val_df, test_df = splitter.split(df)

    store = SimpleFeatureStore()
    store.fit(train_df)

    X_train, y_train = store.transform(train_df)
    X_val, y_val = store.transform(val_df)
    X_test, y_test = store.transform(test_df)

    print("Final feature names:", store.get_feature_names())
    print("\nShapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, "y_val:", y_val.shape)
    print("  X_test: ", X_test.shape, "y_test:", y_test.shape)


if __name__ == "__main__":
    _main()
