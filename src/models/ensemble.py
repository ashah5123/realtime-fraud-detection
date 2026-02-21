"""
Ensemble scorer combining Isolation Forest, Autoencoder, and XGBoost with configurable weights and thresholds.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/model_config.yaml"
DEFAULT_ARTIFACT_PATH = "models/artifacts/ensemble.joblib"
DEFAULT_XGB_PATH = Path("models/artifacts/xgboost.joblib")


def _load_ensemble_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load ensemble section from model config YAML."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config not found at %s; using defaults", path)
        return {}
    with open(path) as f:
        full = yaml.safe_load(f) or {}
    return full.get("ensemble", {})


class EnsembleScorer:
    """
    Combines Isolation Forest, Autoencoder, and XGBoost scores via weighted average.
    Supports weight optimization (PR-AUC), threshold optimization (F1 and cost-based), and risk tiers.
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    ) -> None:
        """
        Initialize from config.

        Args:
            config_path: Path to model_config.yaml.
            artifact_path: Path to save/load artifact.
        """
        self.artifact_path = Path(artifact_path)
        cfg = _load_ensemble_config(config_path)
        self._iso_weight = float(cfg.get("iso_weight", 1.0 / 3))
        self._ae_weight = float(cfg.get("ae_weight", 1.0 / 3))
        self._xgb_weight = float(cfg.get("xgb_weight", 1.0 / 3))
        th = cfg.get("thresholds", {})
        self._threshold_low = float(th.get("low", 0.3))
        self._threshold_high = float(th.get("high", 0.7))
        cost = cfg.get("cost", {})
        self._cost_fp = float(cost.get("false_positive", 50))
        self._cost_fn = float(cost.get("false_negative", 500))
        self._optimal_threshold: float = 0.5
        self._xgb_model = None
        self._fitted_ = False

    def _load_xgb(self) -> None:
        """Load XGBoost model from models/artifacts/xgboost.joblib if not already loaded."""
        if self._xgb_model is not None:
            return
        xgb_path = self.artifact_path.parent / "xgboost.joblib"
        if not xgb_path.exists():
            logger.warning("XGBoost artifact not found at %s; xgb_score will require caller to pass it", xgb_path)
            return
        from src.models.xgboost_model import XGBoostFraudModel
        self._xgb_model = XGBoostFraudModel.load(xgb_path)
        logger.info("Loaded XGBoost from %s", xgb_path)

    def _combined_score(
        self,
        iso_scores: np.ndarray,
        ae_scores: np.ndarray,
        xgb_scores: np.ndarray,
    ) -> np.ndarray:
        """Weighted average of three scores; weights should sum to 1."""
        iso = np.asarray(iso_scores, dtype=np.float64).ravel()
        ae = np.asarray(ae_scores, dtype=np.float64).ravel()
        xgb = np.asarray(xgb_scores, dtype=np.float64).ravel()
        return self._iso_weight * iso + self._ae_weight * ae + self._xgb_weight * xgb

    def set_weights(self, iso_weight: float, ae_weight: float, xgb_weight: float) -> None:
        """Set ensemble weights (should sum to 1)."""
        total = iso_weight + ae_weight + xgb_weight
        if abs(total - 1.0) > 1e-6:
            iso_weight = iso_weight / total
            ae_weight = ae_weight / total
            xgb_weight = xgb_weight / total
        self._iso_weight = iso_weight
        self._ae_weight = ae_weight
        self._xgb_weight = xgb_weight
        logger.info("Set weights: iso=%.2f, ae=%.2f, xgb=%.2f", self._iso_weight, self._ae_weight, self._xgb_weight)

    def optimize_weights(
        self,
        iso_scores: np.ndarray,
        ae_scores: np.ndarray,
        xgb_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Find weights that maximize PR-AUC. Grid search over (iso, ae, xgb) in steps of 0.1 that sum to 1.0.

        Args:
            iso_scores: IsolationForest scores per sample.
            ae_scores: Autoencoder scores per sample.
            xgb_scores: XGBoost scores per sample.
            y_true: Binary labels (1 = fraud).

        Returns:
            (best_iso_weight, best_ae_weight, best_xgb_weight).
        """
        iso_scores = np.asarray(iso_scores).ravel()
        ae_scores = np.asarray(ae_scores).ravel()
        xgb_scores = np.asarray(xgb_scores).ravel()
        y_true = np.asarray(y_true).ravel().astype(int)
        best_pr_auc = -1.0
        best_w_iso, best_w_ae, best_w_xgb = self._iso_weight, self._ae_weight, self._xgb_weight
        step = 0.1
        for w_iso in np.arange(0.0, 1.0 + step * 0.5, step):
            for w_ae in np.arange(0.0, 1.0 - w_iso + step * 0.5, step):
                w_xgb = 1.0 - w_iso - w_ae
                if w_xgb < -1e-9:
                    continue
                combined = w_iso * iso_scores + w_ae * ae_scores + w_xgb * xgb_scores
                pr_auc = average_precision_score(y_true, combined)
                if pr_auc > best_pr_auc:
                    best_pr_auc = pr_auc
                    best_w_iso, best_w_ae, best_w_xgb = w_iso, w_ae, w_xgb
        self.set_weights(best_w_iso, best_w_ae, best_w_xgb)
        logger.info(
            "Optimized weights: iso=%.2f, ae=%.2f, xgb=%.2f (PR-AUC=%.4f)",
            best_w_iso, best_w_ae, best_w_xgb, best_pr_auc,
        )
        return best_w_iso, best_w_ae, best_w_xgb

    def _find_threshold_f1(self, y_true: np.ndarray, combined: np.ndarray) -> float:
        """Threshold that maximizes F1."""
        prec, rec, thresh = precision_recall_curve(y_true, combined)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        # thresh has length one less than prec/rec
        best_idx = np.argmax(f1[:-1])
        return float(thresh[best_idx])

    def _find_threshold_cost(
        self,
        y_true: np.ndarray,
        combined: np.ndarray,
    ) -> float:
        """Threshold that minimizes cost = FP*cost_fp + FN*cost_fn."""
        y_true = np.asarray(y_true).ravel().astype(int)
        best_cost = float("inf")
        best_t = 0.5
        for t in np.arange(0.05, 0.96, 0.01):
            pred = (combined >= t).astype(int)
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            cost = fp * self._cost_fp + fn * self._cost_fn
            if cost < best_cost:
                best_cost = cost
                best_t = t
        return best_t

    def optimize_threshold(
        self,
        iso_scores: np.ndarray,
        ae_scores: np.ndarray,
        xgb_scores: np.ndarray,
        y_true: np.ndarray,
        method: str = "cost",
    ) -> float:
        """
        Find optimal decision threshold. Option (a) F1 maximization or (b) cost minimization.

        Args:
            iso_scores: IsolationForest scores.
            ae_scores: Autoencoder scores.
            xgb_scores: XGBoost scores.
            y_true: Binary labels (1 = fraud).
            method: "f1" or "cost".

        Returns:
            Optimal threshold.
        """
        combined = self._combined_score(iso_scores, ae_scores, xgb_scores)
        y_true = np.asarray(y_true).ravel()
        if method == "f1":
            self._optimal_threshold = self._find_threshold_f1(y_true, combined)
        else:
            self._optimal_threshold = self._find_threshold_cost(y_true, combined)
        logger.info("Optimal threshold (%s): %.3f", method, self._optimal_threshold)
        self._fitted_ = True
        return self._optimal_threshold

    def _risk_tier(self, score: float) -> str:
        """Return LOW, MEDIUM, or HIGH from configured thresholds."""
        if score < self._threshold_low:
            return "LOW"
        if score < self._threshold_high:
            return "MEDIUM"
        return "HIGH"

    def score_transaction(
        self,
        iso_score: float,
        ae_score: float,
        xgb_score: float | None = None,
        features: np.ndarray | None = None,
    ) -> dict:
        """
        Score a single transaction from iso, ae, and xgb scores.

        Args:
            iso_score: IsolationForest score in [0, 1].
            ae_score: Autoencoder score in [0, 1].
            xgb_score: XGBoost score in [0, 1]. If None and features is provided, computed from loaded XGBoost.
            features: Optional feature vector; used to compute xgb_score if xgb_score is None and XGBoost is loaded.

        Returns:
            Dict with fraud_score, risk_tier, iso_score, ae_score, xgb_score, should_alert.
        """
        if xgb_score is None and features is not None:
            self._load_xgb()
            if self._xgb_model is not None:
                xgb_score = float(self._xgb_model.predict_single(features))
            else:
                xgb_score = 0.0
        elif xgb_score is None:
            xgb_score = 0.0
        fraud_score = float(
            self._iso_weight * iso_score + self._ae_weight * ae_score + self._xgb_weight * xgb_score
        )
        risk_tier = self._risk_tier(fraud_score)
        should_alert = bool(fraud_score >= self._optimal_threshold)
        return {
            "fraud_score": round(fraud_score, 4),
            "risk_tier": risk_tier,
            "iso_score": round(iso_score, 4),
            "ae_score": round(ae_score, 4),
            "xgb_score": round(xgb_score, 4),
            "should_alert": should_alert,
        }

    def score_batch(
        self,
        iso_scores: np.ndarray,
        ae_scores: np.ndarray,
        xgb_scores: np.ndarray,
    ) -> np.ndarray:
        """Return combined fraud scores for arrays."""
        return self._combined_score(iso_scores, ae_scores, xgb_scores)

    def get_optimal_threshold(self) -> float:
        """Return the stored optimal threshold."""
        return self._optimal_threshold

    def save(self, path: str | Path | None = None) -> None:
        """Save weights, thresholds, and config to joblib."""
        path = Path(path) if path is not None else self.artifact_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "iso_weight": self._iso_weight,
            "ae_weight": self._ae_weight,
            "xgb_weight": self._xgb_weight,
            "threshold_low": self._threshold_low,
            "threshold_high": self._threshold_high,
            "cost_fp": self._cost_fp,
            "cost_fn": self._cost_fn,
            "optimal_threshold": self._optimal_threshold,
            "fitted": self._fitted_,
        }
        joblib.dump(state, path)
        logger.info("Saved ensemble to %s", path)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ARTIFACT_PATH) -> "EnsembleScorer":
        """Load ensemble from joblib. XGBoost is loaded on first use from same dir (xgboost.joblib)."""
        path = Path(path)
        state = joblib.load(path)
        obj = cls(artifact_path=path)
        obj._iso_weight = state["iso_weight"]
        obj._ae_weight = state["ae_weight"]
        obj._xgb_weight = state.get("xgb_weight", 1.0 / 3)
        obj._threshold_low = state["threshold_low"]
        obj._threshold_high = state["threshold_high"]
        obj._cost_fp = state["cost_fp"]
        obj._cost_fn = state["cost_fn"]
        obj._optimal_threshold = state["optimal_threshold"]
        obj._fitted_ = state.get("fitted", True)
        logger.info("Loaded ensemble from %s", path)
        return obj


def _main() -> None:
    """Load data, split, features, train all three models, optimize weights and threshold, print metrics and samples."""
    import logging
    from src.data.loader import load_transactions
    from src.data.splitter import TimeAwareSplitter
    from src.features.feature_store import SimpleFeatureStore
    from src.models.autoencoder import FraudAutoencoder
    from src.models.isolation_forest import IsolationForestModel
    from src.models.xgboost_model import XGBoostFraudModel

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    df = load_transactions()
    splitter = TimeAwareSplitter()
    train_df, val_df, _test_df = splitter.split(df)
    store = SimpleFeatureStore()
    store.fit(train_df)
    X_train, y_train = store.transform(train_df)
    X_val, y_val = store.transform(val_df)

    iso = IsolationForestModel()
    iso.fit(X_train, y_train)
    ae = FraudAutoencoder()
    ae.fit(X_train.values, y_train.values)
    xgb = XGBoostFraudModel()
    xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    xgb.save()

    iso_val = iso.score(X_val)
    ae_val = ae.score(X_val.values)
    xgb_val = xgb.predict_proba(X_val)
    y_val_arr = np.asarray(y_val).ravel().astype(int)

    ensemble = EnsembleScorer()
    ensemble.optimize_weights(iso_val, ae_val, xgb_val, y_val_arr)
    ensemble.optimize_threshold(iso_val, ae_val, xgb_val, y_val_arr, method="cost")

    combined = ensemble.score_batch(iso_val, ae_val, xgb_val)
    th = ensemble.get_optimal_threshold()
    pred = (combined >= th).astype(int)

    pr_auc = average_precision_score(y_val_arr, combined)
    f1 = f1_score(y_val_arr, pred, zero_division=0)
    prec = precision_score(y_val_arr, pred, zero_division=0)
    rec = recall_score(y_val_arr, pred, zero_division=0)

    print("\nEnsemble performance (validation set):")
    print("  PR-AUC:   %.4f" % pr_auc)
    print("  F1:       %.4f" % f1)
    print("  Precision: %.4f" % prec)
    print("  Recall:   %.4f" % rec)
    print("  Optimal threshold: %.3f" % th)

    print("\nSample scored transactions:")
    for i in [0, 1, 2, len(val_df) // 2, -3, -2, -1]:
        idx = max(0, min(i, len(val_df) - 1))
        d = ensemble.score_transaction(
            float(iso_val[idx]), float(ae_val[idx]), xgb_score=float(xgb_val[idx])
        )
        d["is_fraud"] = int(y_val_arr[idx])
        print("  %s" % d)

    ensemble.save()


if __name__ == "__main__":
    _main()
