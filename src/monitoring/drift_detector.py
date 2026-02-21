"""
Drift detection using Population Stability Index (PSI) for feature and prediction drift.
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_PATH = Path("models/artifacts/drift_baseline.joblib")
DEFAULT_N_BINS = 10
PSI_EPSILON = 1e-6

# PSI thresholds
PSI_OK = 0.1
PSI_MODERATE = 0.25


def _psi_status(psi: float) -> str:
    """Map PSI value to status: OK, MODERATE, or SIGNIFICANT."""
    if psi < PSI_OK:
        return "OK"
    if psi <= PSI_MODERATE:
        return "MODERATE"
    return "SIGNIFICANT"


def _bin_edges_from_values(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute bin edges from data using quantiles. Returns 1D array of length n_bins+1."""
    values = np.asarray(values).ravel()
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.linspace(0, 1, n_bins + 1)
    percentiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(values, percentiles)
    # Ensure unique edges so digitize works (avoid constant feature)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([edges[0] - 1e-9, edges[0] + 1e-9])
    return edges


def _bin_counts(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Assign values to bins and return counts per bin. Bins are [edges[i], edges[i+1])."""
    values = np.asarray(values).ravel()
    values = np.clip(values, bin_edges[0], bin_edges[-1] * (1 + 1e-10))
    counts = np.histogram(values, bins=bin_edges)[0]
    return counts.astype(np.float64)


def _proportions(counts: np.ndarray, epsilon: float = PSI_EPSILON) -> np.ndarray:
    """Convert counts to proportions; add epsilon to avoid zeros."""
    total = counts.sum()
    if total <= 0:
        p = np.ones(len(counts)) / len(counts)
    else:
        p = counts / total
    p = np.clip(p, epsilon, 1.0)
    return p / p.sum()


def _compute_psi(baseline_props: np.ndarray, current_props: np.ndarray) -> float:
    """
    Population Stability Index: sum over bins of (current - baseline) * ln(current / baseline).
    """
    # Align lengths (e.g. if baseline has 10 bins and current was binned differently)
    n = min(len(baseline_props), len(current_props))
    if n == 0:
        return 0.0
    p_b = np.asarray(baseline_props[:n], dtype=np.float64)
    p_c = np.asarray(current_props[:n], dtype=np.float64)
    p_b = _proportions(p_b)
    p_c = _proportions(p_c)
    psi = np.sum((p_c - p_b) * np.log(p_c / p_b))
    return float(max(0.0, psi))


class DriftDetector:
    """
    Detects feature and prediction drift using Population Stability Index (PSI).
    Baseline distributions are fitted on training data and saved to disk.
    """

    def __init__(
        self,
        baseline_path: str | Path = DEFAULT_BASELINE_PATH,
        n_bins: int = DEFAULT_N_BINS,
    ) -> None:
        """
        Args:
            baseline_path: Path to save/load drift baseline (feature and score distributions).
            n_bins: Number of bins for PSI histograms.
        """
        self.baseline_path = Path(baseline_path)
        self.n_bins = n_bins
        self._feature_names: list[str] = []
        self._baseline_bin_edges: dict[str, np.ndarray] = {}
        self._baseline_proportions: dict[str, np.ndarray] = {}
        self._score_bin_edges: np.ndarray | None = None
        self._score_baseline_proportions: np.ndarray | None = None
        self._train_avg_score: float = 0.0
        self._fitted_ = False
        self._last_drift_result: dict[str, Any] | None = None

    def _ensure_artifacts_dir(self) -> None:
        """Create parent directory for baseline path if needed."""
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series,
        train_scores: np.ndarray,
    ) -> "DriftDetector":
        """
        Store training feature and score distributions as baseline; save to joblib.

        Args:
            X_train: Training feature matrix (DataFrame or 2D array).
            y_train: Training labels (unused for baseline; kept for API consistency).
            train_scores: Fraud scores from the ensemble on X_train.

        Returns:
            self (for chaining).
        """
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
            X = X_train.values
        else:
            X = np.asarray(X_train)
            self._feature_names = [f"f{i}" for i in range(X.shape[1])]

        train_scores = np.asarray(train_scores).ravel()
        self._train_avg_score = float(np.mean(train_scores))
        logger.info("Fitting drift baseline: %d samples, %d features", X.shape[0], X.shape[1])

        for i, name in enumerate(self._feature_names):
            col = X[:, i]
            edges = _bin_edges_from_values(col, self.n_bins)
            counts = _bin_counts(col, edges)
            self._baseline_bin_edges[name] = edges
            self._baseline_proportions[name] = _proportions(counts)

        self._score_bin_edges = _bin_edges_from_values(train_scores, self.n_bins)
        score_counts = _bin_counts(train_scores, self._score_bin_edges)
        self._score_baseline_proportions = _proportions(score_counts)
        self._fitted_ = True
        self.save()
        logger.info("Drift baseline saved to %s", self.baseline_path)
        return self

    def save(self, path: str | Path | None = None) -> None:
        """Save baseline state to joblib."""
        path = Path(path) if path is not None else self.baseline_path
        self._ensure_artifacts_dir()
        state = {
            "feature_names": self._feature_names,
            "baseline_bin_edges": self._baseline_bin_edges,
            "baseline_proportions": self._baseline_proportions,
            "score_bin_edges": self._score_bin_edges,
            "score_baseline_proportions": self._score_baseline_proportions,
            "train_avg_score": self._train_avg_score,
            "n_bins": self.n_bins,
        }
        joblib.dump(state, path)

    def load(self, path: str | Path | None = None) -> "DriftDetector":
        """Load baseline state from joblib."""
        path = Path(path) if path is not None else self.baseline_path
        if not path.exists():
            raise FileNotFoundError("Drift baseline not found: %s" % path)
        state = joblib.load(path)
        self._feature_names = state["feature_names"]
        self._baseline_bin_edges = state["baseline_bin_edges"]
        self._baseline_proportions = state["baseline_proportions"]
        self._score_bin_edges = state["score_bin_edges"]
        self._score_baseline_proportions = state["score_baseline_proportions"]
        self._train_avg_score = state["train_avg_score"]
        self.n_bins = state.get("n_bins", DEFAULT_N_BINS)
        self._fitted_ = True
        logger.info("Loaded drift baseline from %s", path)
        return self

    def detect(
        self,
        X_current: np.ndarray | pd.DataFrame,
        current_scores: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute feature and prediction drift vs baseline. Call fit() or load() first.

        Returns:
            Dict with feature_drift, prediction_drift, features_with_drift, overall_status.
        """
        if not self._fitted_:
            raise ValueError("DriftDetector not fitted. Call fit() or load() first.")

        if isinstance(X_current, pd.DataFrame):
            X = X_current.values
            # Use columns that match baseline
            cols = [c for c in X_current.columns if c in self._baseline_bin_edges]
            if cols:
                X = X_current[cols].values
                names = cols
            else:
                names = self._feature_names
        else:
            X = np.asarray(X_current)
            names = self._feature_names

        current_scores = np.asarray(current_scores).ravel()
        feature_drift: dict[str, dict[str, Any]] = {}
        features_with_drift: list[str] = []

        for i, name in enumerate(names):
            if name not in self._baseline_bin_edges:
                continue
            edges = self._baseline_bin_edges[name]
            col = X[:, i] if i < X.shape[1] else np.zeros(X.shape[0])
            counts = _bin_counts(col, edges)
            current_props = _proportions(counts)
            baseline_props = self._baseline_proportions[name]
            psi = _compute_psi(baseline_props, current_props)
            status = _psi_status(psi)
            feature_drift[name] = {"psi": round(psi, 4), "status": status}
            if status != "OK":
                features_with_drift.append(name)

        # Prediction drift (score distribution + average score over time)
        current_avg_score = float(np.mean(current_scores))
        if self._score_bin_edges is not None and self._score_baseline_proportions is not None:
            score_counts = _bin_counts(current_scores, self._score_bin_edges)
            score_current_props = _proportions(score_counts)
            score_psi = _compute_psi(self._score_baseline_proportions, score_current_props)
            prediction_status = _psi_status(score_psi)
            prediction_drift = {
                "psi": round(score_psi, 4),
                "status": prediction_status,
                "baseline_avg_score": round(self._train_avg_score, 4),
                "current_avg_score": round(current_avg_score, 4),
            }
        else:
            prediction_drift = {
                "psi": 0.0,
                "status": "OK",
                "baseline_avg_score": round(self._train_avg_score, 4),
                "current_avg_score": round(current_avg_score, 4),
            }

        # Overall status: worst of any
        all_psi = [v["psi"] for v in feature_drift.values()] + [prediction_drift["psi"]]
        max_psi = max(all_psi) if all_psi else 0.0
        if max_psi > PSI_MODERATE:
            overall_status = "CRITICAL"
        elif max_psi >= PSI_OK:
            overall_status = "WARNING"
        else:
            overall_status = "OK"

        result = {
            "feature_drift": feature_drift,
            "prediction_drift": prediction_drift,
            "features_with_drift": features_with_drift,
            "overall_status": overall_status,
        }
        self._last_drift_result = result
        return result

    def generate_drift_report(self) -> str:
        """
        Return a markdown-formatted drift report from the last detect() run.
        If detect() was not run yet, returns a short message.
        """
        r = self._last_drift_result
        if not r:
            return "# Drift Report\n\nNo drift detection run yet. Call detect() first.\n"

        pd = r["prediction_drift"]
        lines = [
            "# Drift Report",
            "",
            "## Overall status: **%s**" % r["overall_status"],
            "",
            "## Prediction drift",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            "| PSI | %.4f |" % pd["psi"],
            "| Status | %s |" % pd["status"],
            "| Baseline avg score | %.4f |" % pd.get("baseline_avg_score", 0),
            "| Current avg score | %.4f |" % pd.get("current_avg_score", 0),
            "",
            "## Feature drift",
            "",
            "| Feature | PSI | Status |",
            "|---------|-----|--------|",
        ]
        for name, info in sorted(r["feature_drift"].items()):
            lines.append("| %s | %.4f | %s |" % (name, info["psi"], info["status"]))
        lines.extend([
            "",
            "## Features with drift (non-OK)",
            "",
        ])
        if r["features_with_drift"]:
            lines.append(", ".join(r["features_with_drift"]))
        else:
            lines.append("None")
        lines.append("")
        lines.append("---")
        lines.append("PSI &lt; 0.1: OK | 0.1–0.25: MODERATE | &gt; 0.25: SIGNIFICANT")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")

    from src.data.loader import load_transactions
    from src.data.splitter import TimeAwareSplitter
    from src.features.feature_store import SimpleFeatureStore
    from src.models.autoencoder import FraudAutoencoder
    from src.models.ensemble import EnsembleScorer
    from src.models.isolation_forest import IsolationForestModel

    data_path = Path("data/processed/transactions_8k.csv")
    if not data_path.exists():
        logger.error("Data not found: %s. Run data pipeline first.", data_path)
        sys.exit(1)

    logger.info("Loading data and splitting")
    df = load_transactions(data_path)
    splitter = TimeAwareSplitter()
    train_df, val_df, test_df = splitter.split(df)

    artifacts_dir = Path("models/artifacts")
    store_path = artifacts_dir / "feature_store.joblib"
    if not store_path.exists():
        logger.info("Fitting feature store on train")
        store = SimpleFeatureStore(artifact_path=store_path)
        store.fit(train_df)
        store.save()
    else:
        store = SimpleFeatureStore.load(store_path)

    X_train, y_train = store.transform(train_df)
    X_test, y_test = store.transform(test_df)

    # Load models and get scores
    iso_path = artifacts_dir / "isolation_forest.joblib"
    ae_path = artifacts_dir / "autoencoder.pt"
    ensemble_path = artifacts_dir / "ensemble.joblib"
    if not iso_path.exists() or not ae_path.exists() or not ensemble_path.exists():
        logger.error("Model artifacts not found. Run training first: python -m src.models.trainer")
        sys.exit(1)

    iso = IsolationForestModel.load(iso_path)
    ae = FraudAutoencoder.load(ae_path)
    ensemble = EnsembleScorer.load(ensemble_path)

    train_iso = iso.score(X_train)
    train_ae = ae.score(X_train.values)
    train_scores = ensemble.score_batch(train_iso, train_ae)

    # Fit drift baseline from training data
    detector = DriftDetector(baseline_path=artifacts_dir / "drift_baseline.joblib")
    detector.fit(X_train, y_train, train_scores)

    # Simulate drift: score test set as "current" distribution
    test_iso = iso.score(X_test)
    test_ae = ae.score(X_test.values)
    current_scores = ensemble.score_batch(test_iso, test_ae)

    result = detector.detect(X_test, current_scores)
    logger.info("Drift detection result: overall_status=%s", result["overall_status"])

    report = detector.generate_drift_report()
    print(report)
