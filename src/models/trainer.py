"""
Full training pipeline: load, split, feature store, train models, evaluate, save all artifacts.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np

from src.data.loader import load_transactions
from src.data.splitter import TimeAwareSplitter
from src.features.feature_store import SimpleFeatureStore
from src.models.autoencoder import FraudAutoencoder
from src.models.ensemble import EnsembleScorer
from src.models.evaluator import run_full_evaluation
from src.models.isolation_forest import IsolationForestModel

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path("models/artifacts")
RESULTS_DIR = Path("results")
CONFIG_PATH = Path("configs/model_config.yaml")


class Trainer:
    """
    Runs the full training pipeline: load data, split, fit feature store,
    train Isolation Forest, Autoencoder, and Ensemble, run evaluation, save all artifacts.
    """

    def __init__(
        self,
        data_path: str | Path = "data/processed/transactions_8k.csv",
        artifacts_dir: str | Path = ARTIFACTS_DIR,
        results_dir: str | Path = RESULTS_DIR,
        config_path: str | Path = CONFIG_PATH,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            data_path: Path to transactions CSV.
            artifacts_dir: Directory for saved models (feature_store, iso, ae, ensemble).
            results_dir: Directory for evaluation report and plots.
            config_path: Path to model_config.yaml.
        """
        self.data_path = Path(data_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.results_dir = Path(results_dir)
        self.config_path = Path(config_path)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self) -> dict:
        """
        Run the full pipeline and save all artifacts.

        Returns:
            Summary dict with keys like train_shape, val_shape, ensemble_pr_auc, etc.
        """
        summary: dict = {}

        # 1. Load data
        t0 = time.perf_counter()
        df = load_transactions(self.data_path)
        summary["load_shape"] = df.shape
        logger.info("Step 1: Loaded data shape %s in %.2fs", df.shape, time.perf_counter() - t0)

        # 2. Split
        t0 = time.perf_counter()
        splitter = TimeAwareSplitter()
        train_df, val_df, test_df = splitter.split(df)
        summary["train_shape"] = train_df.shape
        summary["val_shape"] = val_df.shape
        summary["test_shape"] = test_df.shape
        logger.info(
            "Step 2: Split train=%s val=%s test=%s in %.2fs",
            train_df.shape,
            val_df.shape,
            test_df.shape,
            time.perf_counter() - t0,
        )

        # 3. Feature store: fit on train, transform train and val, save
        t0 = time.perf_counter()
        store = SimpleFeatureStore(artifact_path=self.artifacts_dir / "feature_store.joblib")
        store.fit(train_df)
        X_train, y_train = store.transform(train_df)
        X_val, y_val = store.transform(val_df)
        store.save()
        logger.info(
            "Step 3: Feature store fit and save; X_train=%s X_val=%s in %.2fs",
            X_train.shape,
            X_val.shape,
            time.perf_counter() - t0,
        )

        # 4. Isolation Forest: fit on train, save
        t0 = time.perf_counter()
        iso = IsolationForestModel(artifact_path=self.artifacts_dir / "isolation_forest.joblib")
        iso.fit(X_train, y_train)
        iso.save()
        logger.info("Step 4: Isolation Forest trained and saved in %.2fs", time.perf_counter() - t0)

        # 5. Autoencoder: fit on train, save
        t0 = time.perf_counter()
        ae = FraudAutoencoder(artifact_path=self.artifacts_dir / "autoencoder.pt")
        ae.fit(X_train.values, y_train.values)
        ae.save()
        logger.info("Step 5: Autoencoder trained and saved in %.2fs", time.perf_counter() - t0)

        # 6. Ensemble: optimize weights and threshold on val, save
        t0 = time.perf_counter()
        iso_val = iso.score(X_val)
        ae_val = ae.score(X_val.values)
        y_val_arr = np.asarray(y_val).ravel().astype(int)

        ensemble = EnsembleScorer(artifact_path=self.artifacts_dir / "ensemble.joblib")
        ensemble.optimize_weights(iso_val, ae_val, y_val_arr)
        ensemble.optimize_threshold(iso_val, ae_val, y_val_arr, method="cost")
        ensemble.save()
        ens_val = ensemble.score_batch(iso_val, ae_val)
        logger.info("Step 6: Ensemble optimized and saved in %.2fs", time.perf_counter() - t0)

        # 7. Evaluation: metrics, plots, report
        t0 = time.perf_counter()
        metrics_by_model, cost_by_model = run_full_evaluation(
            y_val_arr,
            iso_val,
            ae_val,
            ens_val,
            results_dir=self.results_dir,
            config_path=self.config_path,
        )
        summary["metrics"] = metrics_by_model
        summary["costs"] = cost_by_model
        m_ens = metrics_by_model.get("Ensemble", {})
        summary["ensemble_pr_auc"] = m_ens.get("pr_auc")
        summary["ensemble_f1"] = m_ens.get("f1")
        logger.info("Step 7: Evaluation done in %.2fs", time.perf_counter() - t0)

        return summary


def _main() -> None:
    """Run the full training pipeline and save all artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    trainer = Trainer()
    summary = trainer.run()
    print("\n--- Pipeline complete ---")
    print("  Artifacts saved to %s" % trainer.artifacts_dir)
    print("  Results and report saved to %s" % trainer.results_dir)
    if summary.get("ensemble_pr_auc") is not None:
        print("  Ensemble PR-AUC: %.4f  F1: %.4f" % (summary["ensemble_pr_auc"], summary.get("ensemble_f1", 0)))


if __name__ == "__main__":
    _main()
