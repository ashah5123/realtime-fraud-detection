"""
Data preprocessor for the Sparkov fraud dataset (transactions_8k.csv).
Implements fit/transform (sklearn-style) with imputation, encoding, and scaling.
"""

import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

DROP_COLS = ["first", "last", "trans_num"]
CC_NUM_COL = "cc_num"
DATE_COLS = ["trans_date_trans_time", "dob"]
NUMERIC_IMPUTE = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long"]
SCALE_COLS = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "age"]
LABEL_ENCODE_COLS = ["merchant", "city", "job", "state"]
ONEHOT_COLS = ["category", "gender"]
TARGET_COL = "is_fraud"
CATEGORICAL_FILL = "unknown"
DEFAULT_ARTIFACT_PATH = "models/artifacts/preprocessor.joblib"


class DataPreprocessor:
    """
    Preprocesses the Sparkov fraud dataset: drop cols, parse dates, impute,
    label/one-hot encode, and scale. Fits on train data only; transform applies
    the fitted pipeline.
    """

    def __init__(self, artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> None:
        """
        Initialize the preprocessor.

        Args:
            artifact_path: Path to save/load fitted artifacts (joblib).
        """
        self.artifact_path = Path(artifact_path)
        self.imputer_: SimpleImputer | None = None
        self.ordinal_encoder_: OrdinalEncoder | None = None
        self.onehot_encoder_: OneHotEncoder | None = None
        self.scaler_: RobustScaler | None = None
        self.numeric_cols_: list[str] = []
        self.label_cols_: list[str] = []
        self.onehot_feature_names_: list[str] = []
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fit imputer, encoders, and scaler on the training dataframe.

        Args:
            df: Full training DataFrame (including target).

        Returns:
            self (for chaining).
        """
        logger.info("Fitting preprocessor on shape %s", df.shape)
        data = self._prepare(df)
        shape_after_prepare = data.shape
        logger.info("After drop/parse/age: shape %s", shape_after_prepare)

        # Imputation
        data, self.numeric_cols_ = self._impute_fit(data)
        logger.info("After imputation: shape %s", data.shape)

        # Encoding (fit)
        self._fit_encoders(data)
        data_encoded = self._encode_transform(data, is_fit=False)
        logger.info("After encoding: shape %s", data_encoded.shape)

        # Scale (fit on numeric columns only)
        self.scaler_ = RobustScaler()
        self.scaler_.fit(data_encoded[SCALE_COLS])
        logger.info("Fitted RobustScaler on columns: %s", SCALE_COLS)

        self.fitted_ = True
        return self

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary cols, parse dates, compute age."""
        data = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
        logger.info("Dropped columns: %s", DROP_COLS)

        if "trans_date_trans_time" in data.columns:
            data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
        if "dob" in data.columns:
            data["dob"] = pd.to_datetime(data["dob"])
        data["age"] = (data["trans_date_trans_time"] - data["dob"]).dt.days / 365.25
        logger.info("Parsed dates and computed age")

        return data

    def _impute_fit(self, data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Fit median imputer for numerics, fill categoricals with 'unknown'. Return (data, numeric_cols_used)."""
        numeric_cols = [c for c in NUMERIC_IMPUTE + ["age"] if c in data.columns]
        cat_cols = [c for c in LABEL_ENCODE_COLS + ONEHOT_COLS if c in data.columns]

        null_numeric = data[numeric_cols].isnull().sum()
        cols_with_nulls = null_numeric[null_numeric > 0]
        if len(cols_with_nulls) > 0:
            for col in cols_with_nulls.index:
                logger.info("Numeric nulls in %s: %d (median imputation)", col, int(cols_with_nulls[col]))
        self.imputer_ = SimpleImputer(strategy="median")
        self.imputer_.fit(data[numeric_cols])
        data = data.copy()
        data[numeric_cols] = self.imputer_.transform(data[numeric_cols])

        for col in cat_cols:
            n = data[col].isnull().sum()
            if n > 0:
                logger.info("Categorical nulls in %s: %d (filled with '%s')", col, int(n), CATEGORICAL_FILL)
                data[col] = data[col].fillna(CATEGORICAL_FILL)
        return data, numeric_cols

    def _impute_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted imputer and categorical fill."""
        numeric_cols = [c for c in NUMERIC_IMPUTE + ["age"] if c in data.columns]
        cat_cols = [c for c in LABEL_ENCODE_COLS + ONEHOT_COLS if c in data.columns]
        data = data.copy()
        if self.imputer_ is not None and numeric_cols:
            data[numeric_cols] = self.imputer_.transform(data[numeric_cols])
        for col in cat_cols:
            data[col] = data[col].fillna(CATEGORICAL_FILL)
        return data

    def _fit_encoders(self, data: pd.DataFrame) -> None:
        """Fit ordinal encoder (label) and one-hot encoder on train data."""
        self.ordinal_encoder_ = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.ordinal_encoder_.fit(data[LABEL_ENCODE_COLS])
        logger.info("Fitted OrdinalEncoder on: %s", LABEL_ENCODE_COLS)

        self.onehot_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.onehot_encoder_.fit(data[ONEHOT_COLS])
        self.onehot_feature_names_ = list(self.onehot_encoder_.get_feature_names_out(ONEHOT_COLS))
        logger.info("Fitted OneHotEncoder on: %s -> %d features", ONEHOT_COLS, len(self.onehot_feature_names_))

    def _encode_transform(self, data: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        """Apply label and one-hot encoding; return dataframe with encoded columns."""
        label_arr = self.ordinal_encoder_.transform(data[LABEL_ENCODE_COLS])
        label_df = pd.DataFrame(
            label_arr,
            columns=LABEL_ENCODE_COLS,
            index=data.index,
        )
        onehot_arr = self.onehot_encoder_.transform(data[ONEHOT_COLS])
        onehot_df = pd.DataFrame(
            onehot_arr,
            columns=self.onehot_feature_names_,
            index=data.index,
        )
        numeric_df = data[SCALE_COLS].copy()
        return pd.concat([numeric_df, label_df, onehot_df], axis=1)

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply the fitted pipeline and return (X, y).

        Args:
            df: Full DataFrame (including target).

        Returns:
            (X, y): X = feature matrix (DataFrame), y = is_fraud.

        Raises:
            ValueError: If preprocessor has not been fitted.
        """
        if not self.fitted_:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        logger.info("Transforming data shape %s", df.shape)

        data = self._prepare(df)
        data = self._impute_transform(data)
        # Drop cols we don't use as features before encoding
        to_drop = [CC_NUM_COL, "trans_date_trans_time", "dob"] + [c for c in [TARGET_COL] if c in data.columns]
        y = data[TARGET_COL].copy() if TARGET_COL in data.columns else pd.Series(dtype=float)
        data = data.drop(columns=[c for c in to_drop if c in data.columns], errors="ignore")

        X_encoded = self._encode_transform(data, is_fit=False)
        X_encoded[SCALE_COLS] = self.scaler_.transform(X_encoded[SCALE_COLS])
        logger.info("Scaling applied to: %s", SCALE_COLS)
        logger.info("Output shape: X %s, y %s", X_encoded.shape, y.shape)

        return X_encoded, y

    def save(self, path: str | Path | None = None) -> None:
        """
        Save fitted imputer, encoders, and scaler to disk with joblib.

        Args:
            path: Override default artifact path.
        """
        path = Path(path) if path is not None else self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "imputer_": self.imputer_,
            "ordinal_encoder_": self.ordinal_encoder_,
            "onehot_encoder_": self.onehot_encoder_,
            "scaler_": self.scaler_,
            "numeric_cols_": self.numeric_cols_,
            "onehot_feature_names_": self.onehot_feature_names_,
            "fitted_": self.fitted_,
        }
        joblib.dump(state, path)
        logger.info("Saved preprocessor to %s", path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> "DataPreprocessor":
        """
        Load a fitted preprocessor from disk.

        Args:
            path: Path to the joblib file.

        Returns:
            Loaded DataPreprocessor instance.
        """
        path = Path(path)
        state = joblib.load(path)
        obj = cls(artifact_path=path)
        for key, value in state.items():
            setattr(obj, key, value)
        logger.info("Loaded preprocessor from %s", path)
        return obj


def __main__() -> None:
    """Load data, fit preprocessor, transform, and print result shape and sample."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    from src.data.loader import load_transactions

    df = load_transactions()
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)
    print("Result shape:", X.shape, "X,", y.shape, "y")
    print("Sample (first 3 rows):")
    print(X.head(3))
    preprocessor.save()


if __name__ == "__main__":
    __main__()
