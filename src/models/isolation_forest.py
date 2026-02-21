"""
Isolation Forest model for fraud detection.
Semi-supervised: fit on non-fraud transactions only; score maps to [0,1] (higher = more likely fraud).
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/model_config.yaml"
DEFAULT_ARTIFACT_PATH = "models/artifacts/isolation_forest.joblib"


def _load_iso_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load isolation_forest section from model config YAML."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config not found at %s; using defaults", path)
        return {}
    with open(path) as f:
        full = yaml.safe_load(f) or {}
    return full.get("isolation_forest", {})


class IsolationForestModel:
    """
    Wraps sklearn IsolationForest for fraud detection.
    Fits on non-fraud (y==0) only; scores in [0,1] with higher = more likely fraud.
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    ) -> None:
        """
        Initialize from config (or defaults).

        Args:
            config_path: Path to model_config.yaml.
            artifact_path: Path to save/load model artifact.
        """
        self.artifact_path = Path(artifact_path)
        cfg = _load_iso_config(config_path)
        self._n_estimators = int(cfg.get("n_estimators", 200))
        self._contamination = float(cfg.get("contamination", 0.035))
        self._max_features = float(cfg.get("max_features", 0.8))
        self._max_samples = float(cfg.get("max_samples", 0.8))
        self._random_state = int(cfg.get("random_state", 42))
        self._model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            max_features=self._max_features,
            max_samples=self._max_samples,
            random_state=self._random_state,
        )
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._feature_names: list[str] = []
        self._metadata: dict = {}
        self._fitted_ = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "IsolationForestModel":
        """
        Fit on non-fraud transactions only (semi-supervised).

        Args:
            X: Feature matrix.
            y: Labels (0 = non-fraud, 1 = fraud).

        Returns:
            self (for chaining).
        """
        y = np.asarray(y).ravel()
        mask = y == 0
        n_total = len(y)
        n_normal = int(mask.sum())
        if n_normal == 0:
            raise ValueError("No non-fraud samples (y==0) to fit on.")
        X_train = X[mask] if hasattr(X, "iloc") else np.asarray(X)[mask]
        if hasattr(X_train, "values"):
            X_train = X_train.values
        X_train = np.asarray(X_train, dtype=np.float64)

        t0 = time.perf_counter()
        self._model.fit(X_train)
        fit_time = time.perf_counter() - t0

        # Fit score scaler on -decision_function (so higher raw = more anomalous)
        raw_train = -self._model.decision_function(X_train)
        self._scaler.fit(raw_train.reshape(-1, 1))
        self._feature_names = list(X.columns) if hasattr(X, "columns") else []
        self._metadata = {
            "training_date": datetime.now(timezone.utc).isoformat(),
            "n_samples": n_normal,
            "n_total": n_total,
            "feature_names": self._feature_names,
            "hyperparams": {
                "n_estimators": self._n_estimators,
                "contamination": self._contamination,
                "max_features": self._max_features,
                "max_samples": self._max_samples,
                "random_state": self._random_state,
            },
        }
        self._fitted_ = True
        logger.info(
            "IsolationForest fit on %d non-fraud samples (of %d total) in %.2fs",
            n_normal,
            n_total,
            fit_time,
        )
        return self

    def score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Return anomaly score in [0, 1]; higher = more likely fraud.

        Args:
            X: Feature matrix.

        Returns:
            1D array of scores in [0, 1].
        """
        if not self._fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = -self._model.decision_function(X)
        scaled = self._scaler.transform(raw.reshape(-1, 1)).ravel()
        return np.clip(scaled, 0.0, 1.0)

    def predict_single(self, features: np.ndarray) -> float:
        """
        Score one transaction (real-time).

        Args:
            features: 1D feature vector (same order as get_feature_names()).

        Returns:
            Fraud score in [0, 1].
        """
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return float(self.score(arr)[0])

    def get_feature_names(self) -> list[str]:
        """Return feature names used in training (if available)."""
        return list(self._feature_names)

    def get_metadata(self) -> dict:
        """Return training metadata."""
        return dict(self._metadata)

    def save(self, path: str | Path | None = None) -> None:
        """
        Save model, scaler, and metadata with joblib.

        Args:
            path: Override default artifact path.
        """
        path = Path(path) if path is not None else self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self._model,
            "scaler": self._scaler,
            "feature_names": self._feature_names,
            "metadata": self._metadata,
            "hyperparams": {
                "n_estimators": self._n_estimators,
                "contamination": self._contamination,
                "max_features": self._max_features,
                "max_samples": self._max_samples,
                "random_state": self._random_state,
            },
        }
        joblib.dump(state, path)
        logger.info("Saved IsolationForest to %s", path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> "IsolationForestModel":
        """
        Load a fitted model from disk.

        Args:
            path: Path to the joblib file.

        Returns:
            Loaded IsolationForestModel instance.
        """
        path = Path(path)
        state = joblib.load(path)
        cfg_path = Path(DEFAULT_CONFIG_PATH)
        obj = cls(config_path=cfg_path, artifact_path=path)
        obj._model = state["model"]
        obj._scaler = state["scaler"]
        obj._feature_names = state.get("feature_names", [])
        obj._metadata = state.get("metadata", {})
        hp = state.get("hyperparams", {})
        obj._n_estimators = hp.get("n_estimators", 200)
        obj._contamination = hp.get("contamination", 0.035)
        obj._max_features = hp.get("max_features", 0.8)
        obj._max_samples = hp.get("max_samples", 0.8)
        obj._random_state = hp.get("random_state", 42)
        obj._fitted_ = True
        logger.info("Loaded IsolationForest from %s", path)
        return obj


def _main() -> None:
    """Load data, split, engineer features, train on train, score val, print sample scores."""
    import logging
    from src.data.loader import load_transactions
    from src.features.feature_store import SimpleFeatureStore
    from src.data.splitter import TimeAwareSplitter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    df = load_transactions()
    splitter = TimeAwareSplitter()
    train_df, val_df, test_df = splitter.split(df)
    store = SimpleFeatureStore()
    store.fit(train_df)
    X_train, y_train = store.transform(train_df)
    X_val, y_val = store.transform(val_df)

    model = IsolationForestModel()
    model.fit(X_train, y_train)
    scores_val = model.score(X_val)
    y_val_arr = np.asarray(y_val).ravel()

    fraud_scores = scores_val[y_val_arr == 1]
    non_fraud_scores = scores_val[y_val_arr == 0]
    print("Sample scores (validation set):")
    print("  Non-fraud: mean = %.4f, min = %.4f, max = %.4f" % (non_fraud_scores.mean(), non_fraud_scores.min(), non_fraud_scores.max()))
    print("  Fraud:     mean = %.4f, min = %.4f, max = %.4f" % (fraud_scores.mean(), fraud_scores.min(), fraud_scores.max()))
    print("  (higher score = more likely fraud)")


if __name__ == "__main__":
    _main()
