"""
XGBoost supervised fraud detection model with class imbalance handling and early stopping.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = Path("models/artifacts/xgboost.joblib")


class XGBoostFraudModel:
    """
    Supervised XGBoost classifier for fraud detection.
    Uses scale_pos_weight for class imbalance and early stopping on validation PR-AUC.
    """

    def __init__(
        self,
        *,
        max_depth: int = 5,
        learning_rate: float = 0.01,
        n_estimators: int = 300,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 10,
        random_state: int = 42,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    ) -> None:
        """
        Initialize XGBoost classifier. scale_pos_weight is set at fit time from fraud ratio.

        Args:
            max_depth: Maximum tree depth.
            learning_rate: Learning rate.
            n_estimators: Number of boosting rounds (early stopping may stop earlier).
            subsample: Row subsample ratio.
            colsample_bytree: Column subsample ratio.
            early_stopping_rounds: Patience for early stopping on validation PR-AUC.
            random_state: Random seed.
            artifact_path: Path to save/load model.
        """
        self.artifact_path = Path(artifact_path)
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._n_estimators = n_estimators
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._early_stopping_rounds = early_stopping_rounds
        self._random_state = random_state
        self._feature_names: list[str] = []
        self._scale_pos_weight: float = 1.0
        self._model = XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric="aucpr",
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )
        self._fitted_ = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        *,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        verbose: bool | int = True,
    ) -> "XGBoostFraudModel":
        """
        Supervised training with optional early stopping on validation PR-AUC.

        Args:
            X: Training features.
            y: Training labels (0 = non-fraud, 1 = fraud).
            X_val: Validation features for early stopping.
            y_val: Validation labels.
            verbose: Logging verbosity (True or integer).

        Returns:
            self (for chaining).
        """
        if hasattr(X, "columns"):
            self._feature_names = list(X.columns)
        else:
            self._feature_names = [f"f{i}" for i in range(np.asarray(X).shape[1])]
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel().astype(int)
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0:
            raise ValueError("No positive (fraud) samples in y.")
        self._scale_pos_weight = n_neg / n_pos
        self._model.set_params(scale_pos_weight=self._scale_pos_weight)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val).ravel().astype(int)
            eval_set = [(X_val, y_val)]
            logger.info("Early stopping on validation set (patience=%d, eval_metric=aucpr)", self._early_stopping_rounds)

        logger.info(
            "Training XGBoost: n_train=%d, fraud_ratio=%.2f%%, scale_pos_weight=%.2f",
            len(y),
            100 * n_pos / len(y),
            self._scale_pos_weight,
        )
        self._model.fit(X, y, eval_set=eval_set, verbose=verbose)
        self._fitted_ = True
        if eval_set and hasattr(self._model, "best_iteration"):
            logger.info("Early stopping: best iteration = %s", getattr(self._model, "best_iteration", "?"))
        logger.info("XGBoost training complete")
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict fraud probability [0, 1] for each sample.

        Returns:
            Array of shape (n_samples,) with P(fraud) = P(class=1).
        """
        if not self._fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        proba = self._model.predict_proba(X)
        return np.asarray(proba[:, 1]).ravel()

    def predict_single(self, features: np.ndarray) -> float:
        """
        Score a single transaction (1D or 2D feature vector).

        Args:
            features: Feature vector of shape (n_features,) or (1, n_features).

        Returns:
            Fraud probability in [0, 1].
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return float(self.predict_proba(features)[0])

    def get_feature_importance(self) -> dict[str, float]:
        """
        Return feature name -> importance sorted by importance descending.

        Uses gain-based importance from the booster.
        """
        if not self._fitted_:
            raise ValueError("Model not fitted. Call fit() first.")
        imp = self._model.feature_importances_
        names = self._feature_names if self._feature_names else [f"f{i}" for i in range(len(imp))]
        if len(names) != len(imp):
            names = [f"f{i}" for i in range(len(imp))]
        d = dict(zip(names, map(float, imp)))
        return dict(sorted(d.items(), key=lambda x: -x[1]))

    def save(self, path: str | Path | None = None) -> None:
        """Save model and metadata to joblib."""
        path = Path(path) if path is not None else self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self._model,
            "feature_names": self._feature_names,
            "scale_pos_weight": self._scale_pos_weight,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "n_estimators": self._n_estimators,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "early_stopping_rounds": self._early_stopping_rounds,
            "random_state": self._random_state,
        }
        joblib.dump(state, path)
        logger.info("Saved XGBoost model to %s", path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> "XGBoostFraudModel":
        """Load model from joblib."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError("Model not found: %s" % path)
        state = joblib.load(path)
        obj = cls(
            max_depth=state.get("max_depth", 5),
            learning_rate=state.get("learning_rate", 0.01),
            n_estimators=state.get("n_estimators", 300),
            subsample=state.get("subsample", 0.8),
            colsample_bytree=state.get("colsample_bytree", 0.8),
            early_stopping_rounds=state.get("early_stopping_rounds", 10),
            random_state=state.get("random_state", 42),
            artifact_path=path,
        )
        obj._model = state["model"]
        obj._feature_names = state.get("feature_names", [])
        obj._scale_pos_weight = state.get("scale_pos_weight", 1.0)
        obj._fitted_ = True
        logger.info("Loaded XGBoost model from %s", path)
        return obj


if __name__ == "__main__":
    import logging
    from src.data.loader import load_transactions
    from src.data.splitter import TimeAwareSplitter
    from src.features.feature_store import SimpleFeatureStore

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    data_path = Path("data/processed/transactions_8k.csv")
    if not data_path.exists():
        raise SystemExit("Data not found: %s. Run data pipeline first." % data_path)

    df = load_transactions(data_path)
    splitter = TimeAwareSplitter()
    train_df, val_df, _ = splitter.split(df)

    store = SimpleFeatureStore()
    store.fit(train_df)
    X_train, y_train = store.transform(train_df)
    X_val, y_val = store.transform(val_df)

    model = XGBoostFraudModel()
    model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        verbose=10,
    )

    y_val_arr = np.asarray(y_val).ravel().astype(int)
    scores = model.predict_proba(X_val)
    pred = (scores >= 0.5).astype(int)

    pr_auc = average_precision_score(y_val_arr, scores)
    roc_auc = roc_auc_score(y_val_arr, scores) if len(np.unique(y_val_arr)) > 1 else 0.0
    f1 = f1_score(y_val_arr, pred, zero_division=0)
    prec = precision_score(y_val_arr, pred, zero_division=0)
    rec = recall_score(y_val_arr, pred, zero_division=0)

    print("\n--- Validation metrics ---")
    print("  PR-AUC:    %.4f" % pr_auc)
    print("  ROC-AUC:   %.4f" % roc_auc)
    print("  F1:        %.4f" % f1)
    print("  Precision: %.4f" % prec)
    print("  Recall:    %.4f" % rec)

    importance = model.get_feature_importance()
    print("\n--- Top 10 features ---")
    for i, (name, imp) in enumerate(list(importance.items())[:10], 1):
        print("  %2d. %s  %.4f" % (i, name, imp))

    model.save()
