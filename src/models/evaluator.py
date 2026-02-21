"""
Comprehensive model evaluation: metrics, visualizations, comparison, cost analysis, report.
"""

import logging
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/model_config.yaml"
DEFAULT_RESULTS_DIR = "results"


def _load_cost_config(config_path: str | Path) -> tuple[float, float]:
    """Load FP/FN cost from ensemble config."""
    path = Path(config_path)
    if not path.exists():
        return 50.0, 500.0
    with open(path) as f:
        full = yaml.safe_load(f) or {}
    cost = full.get("ensemble", {}).get("cost", {})
    return float(cost.get("false_positive", 50)), float(cost.get("false_negative", 500))


def _threshold_for_f1(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Threshold that maximizes F1."""
    prec, rec, thresh = precision_recall_curve(y_true, y_scores)
    f1 = np.nan_to_num(2 * prec * rec / (prec + rec + 1e-12))
    idx = np.argmax(f1[:-1])
    return float(thresh[idx]) if len(thresh) else 0.5


class ModelEvaluator:
    """
    Comprehensive evaluation: classification metrics, PR/ROC/score plots,
    confusion matrix, threshold tradeoff, model comparison, cost-benefit, markdown report.
    """

    def __init__(
        self,
        results_dir: str | Path = DEFAULT_RESULTS_DIR,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.config_path = Path(config_path)
        os.makedirs(self.results_dir, exist_ok=True)
        self._cost_fp, self._cost_fn = _load_cost_config(self.config_path)
        self._metrics_store: dict = {}
        self._cost_results: dict = {}

    def _metrics_at_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float,
    ) -> dict:
        """Precision, Recall, F1, confusion matrix at given threshold."""
        y_true = np.asarray(y_true).ravel().astype(int)
        y_scores = np.asarray(y_scores).ravel()
        y_pred = (y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
            "threshold": threshold,
        }

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        name: str = "model",
        threshold: float | None = None,
    ) -> dict:
        """
        Compute classification metrics (PR-AUC, ROC-AUC, P/R/F1 at optimal threshold, confusion matrix).

        Args:
            y_true: Binary labels (1 = fraud).
            y_scores: Predicted scores (higher = fraud).
            name: Model name for storage.
            threshold: If None, use F1-optimal threshold.

        Returns:
            Dict of metrics.
        """
        y_true = np.asarray(y_true).ravel().astype(int)
        y_scores = np.asarray(y_scores).ravel()
        if threshold is None:
            threshold = _threshold_for_f1(y_true, y_scores)
        pr_auc = float(average_precision_score(y_true, y_scores))
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            roc_auc = 0.0
        at_thresh = self._metrics_at_threshold(y_true, y_scores, threshold)
        out = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "optimal_threshold": threshold,
            **at_thresh,
        }
        self._metrics_store[name] = out
        logger.info("Computed metrics for %s: PR-AUC=%.4f ROC-AUC=%.4f F1=%.4f", name, pr_auc, roc_auc, out["f1"])
        return out

    def _plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        name: str,
        filepath: Path,
        ax=None,
    ) -> None:
        """Precision-Recall curve with random baseline."""
        import matplotlib.pyplot as plt
        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, label="%s (PR-AUC=%.3f)" % (name, pr_auc))
        if filepath is not None:
            n_pos = (np.asarray(y_true) == 1).sum()
            baseline = n_pos / len(y_true) if len(y_true) else 0
            ax.axhline(baseline, color="gray", linestyle="--", label="Random")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        if filepath is not None:
            ax.set_title("Precision-Recall curve")
        ax.legend(loc="best")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        if filepath is not None:
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Saved PR curve to %s", filepath)

    def _plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray, name: str, filepath: Path) -> None:
        """ROC curve with diagonal baseline."""
        import matplotlib.pyplot as plt
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label="%s (AUC=%.3f)" % (name, roc_auc))
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved ROC curve to %s", filepath)

    def _plot_score_distribution(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        name: str,
        filepath: Path,
    ) -> None:
        """Overlapping histograms: fraud vs non-fraud scores."""
        import matplotlib.pyplot as plt
        y_true = np.asarray(y_true).ravel()
        y_scores = np.asarray(y_scores).ravel()
        scores_fraud = y_scores[y_true == 1]
        scores_non = y_scores[y_true == 0]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(scores_non, bins=40, alpha=0.5, color="green", label="Non-fraud", density=True)
        ax.hist(scores_fraud, bins=40, alpha=0.5, color="red", label="Fraud", density=True)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.set_title("Score distribution: %s" % name)
        ax.legend()
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved score distribution to %s", filepath)

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        name: str,
        filepath: Path,
        threshold: float | None = None,
    ) -> None:
        """Confusion matrix heatmap."""
        import matplotlib.pyplot as plt
        y_true = np.asarray(y_true).ravel().astype(int)
        y_scores = np.asarray(y_scores).ravel()
        if threshold is None:
            threshold = _threshold_for_f1(y_true, y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-fraud", "Fraud"])
        ax.set_yticklabels(["Non-fraud", "Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_title("Confusion matrix: %s (th=%.2f)" % (name, threshold))
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        fig.colorbar(im, ax=ax, label="Count")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved confusion matrix to %s", filepath)

    def _plot_threshold_tradeoff(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        name: str,
        filepath: Path,
    ) -> None:
        """Threshold vs Precision/Recall curve."""
        import matplotlib.pyplot as plt
        prec, rec, thresh = precision_recall_curve(y_true, y_scores)
        # thresh has length len(prec)-1
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(thresh, prec[:-1], label="Precision")
        ax.plot(thresh, rec[:-1], label="Recall")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Threshold vs Precision/Recall: %s" % name)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved threshold tradeoff to %s", filepath)

    def evaluate_single_model(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        name: str,
    ) -> dict:
        """
        Full evaluation for one model: metrics + all 5 plots.

        Returns:
            Metrics dict.
        """
        os.makedirs(self.results_dir, exist_ok=True)
        safe_name = name.replace(" ", "_").lower()
        metrics = self.compute_metrics(y_true, y_scores, name=name)
        th = metrics["optimal_threshold"]
        self._plot_pr_curve(y_true, y_scores, name, self.results_dir / ("pr_curve_%s.png" % safe_name))
        self._plot_roc_curve(y_true, y_scores, name, self.results_dir / ("roc_curve_%s.png" % safe_name))
        self._plot_score_distribution(y_true, y_scores, name, self.results_dir / ("score_dist_%s.png" % safe_name))
        self._plot_confusion_matrix(y_true, y_scores, name, self.results_dir / ("confusion_%s.png" % safe_name), threshold=th)
        self._plot_threshold_tradeoff(y_true, y_scores, name, self.results_dir / ("threshold_tradeoff_%s.png" % safe_name))
        return metrics

    def compare_models(
        self,
        y_true: np.ndarray,
        scores_dict: dict[str, np.ndarray],
    ) -> dict[str, dict]:
        """
        Compare multiple models: table of metrics + overlaid PR curves.

        Args:
            y_true: Binary labels.
            scores_dict: {"Isolation Forest": scores, "Autoencoder": scores, "Ensemble": scores}.

        Returns:
            Dict of model name -> metrics dict.
        """
        import matplotlib.pyplot as plt
        os.makedirs(self.results_dir, exist_ok=True)
        all_metrics: dict[str, dict] = {}
        fig, ax = plt.subplots(figsize=(7, 6))
        y_true_arr = np.asarray(y_true).ravel()
        baseline = (y_true_arr == 1).sum() / len(y_true_arr) if len(y_true_arr) else 0
        ax.axhline(baseline, color="gray", linestyle="--", label="Random")
        for name, y_scores in scores_dict.items():
            m = self.compute_metrics(y_true, y_scores, name=name)
            all_metrics[name] = m
            self._plot_pr_curve(y_true, y_scores, name, None, ax=ax)
        ax.set_title("PR curves: model comparison")
        fig.savefig(self.results_dir / "pr_curves_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved comparison PR curves to %s", self.results_dir / "pr_curves_comparison.png")
        return all_metrics

    def cost_analysis(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        name: str = "model",
        default_threshold: float = 0.5,
    ) -> dict:
        """
        Total cost at different thresholds; find cost-optimal threshold.
        Cost = FP * cost_fp + FN * cost_fn (from config).

        Returns:
            Dict with cost_optimal_threshold, cost_at_optimal, cost_at_default, costs_by_threshold.
        """
        y_true = np.asarray(y_true).ravel().astype(int)
        y_scores = np.asarray(y_scores).ravel()
        best_cost = float("inf")
        best_t = 0.5
        costs_by_t: list[tuple[float, float]] = []
        for t in np.arange(0.05, 0.96, 0.01):
            pred = (y_scores >= t).astype(int)
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            cost = fp * self._cost_fp + fn * self._cost_fn
            costs_by_t.append((float(t), float(cost)))
            if cost < best_cost:
                best_cost = cost
                best_t = t
        pred_default = (y_scores >= default_threshold).astype(int)
        fp_d = ((pred_default == 1) & (y_true == 0)).sum()
        fn_d = ((pred_default == 0) & (y_true == 1)).sum()
        cost_default = float(fp_d * self._cost_fp + fn_d * self._cost_fn)
        out = {
            "cost_optimal_threshold": best_t,
            "cost_at_optimal": best_cost,
            "cost_at_default": cost_default,
            "cost_fp": self._cost_fp,
            "cost_fn": self._cost_fn,
        }
        self._cost_results[name] = out
        logger.info(
            "Cost analysis %s: optimal th=%.2f cost=$%.0f, default th=%.2f cost=$%.0f",
            name,
            best_t,
            best_cost,
            default_threshold,
            cost_default,
        )
        return out

    def generate_report(
        self,
        metrics_by_model: dict[str, dict],
        cost_by_model: dict[str, dict] | None = None,
    ) -> None:
        """
        Write results/evaluation_report.md with metrics tables and plot references.
        """
        cost_by_model = cost_by_model or self._cost_results
        lines = [
            "# Fraud Detection Model Evaluation Report",
            "",
            "## 1. Classification Metrics",
            "",
            "| Model | PR-AUC | ROC-AUC | Optimal Threshold | Precision | Recall | F1 |",
            "|-------|--------|---------|-------------------|-----------|--------|-----|",
        ]
        for name, m in metrics_by_model.items():
            lines.append(
                "| %s | %.4f | %.4f | %.3f | %.4f | %.4f | %.4f |"
                % (name, m["pr_auc"], m["roc_auc"], m["optimal_threshold"], m["precision"], m["recall"], m["f1"])
            )
        lines.extend(["", "## 2. Cost-Benefit Analysis", ""])
        lines.append("Costs: FP = $%.0f, FN = $%.0f" % (self._cost_fp, self._cost_fn))
        lines.extend(["", "| Model | Cost-Optimal Threshold | Cost at Optimal | Cost at Default (0.5) |", "|-------|-------------------------|-----------------|------------------------|"])
        for name, c in cost_by_model.items():
            lines.append(
                "| %s | %.2f | $%.0f | $%.0f |"
                % (name, c["cost_optimal_threshold"], c["cost_at_optimal"], c["cost_at_default"])
            )
        lines.extend([
            "",
            "## 3. Visualizations",
            "",
            "### Model comparison",
            "- ![PR curves comparison](pr_curves_comparison.png)",
            "",
            "### Per-model plots",
            "",
        ])
        for name in metrics_by_model:
            safe = name.replace(" ", "_").lower()
            lines.append("**%s:**" % name)
            lines.append("- PR curve: ![](pr_curve_%s.png)" % safe)
            lines.append("- ROC curve: ![](roc_curve_%s.png)" % safe)
            lines.append("- Score distribution: ![](score_dist_%s.png)" % safe)
            lines.append("- Confusion matrix: ![](confusion_%s.png)" % safe)
            lines.append("- Threshold vs P/R: ![](threshold_tradeoff_%s.png)" % safe)
            lines.append("")
        report_path = self.results_dir / "evaluation_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        logger.info("Saved evaluation report to %s", report_path)


def run_full_evaluation(
    y_true: np.ndarray,
    iso_scores: np.ndarray,
    ae_scores: np.ndarray,
    ensemble_scores: np.ndarray,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> tuple[dict, dict]:
    """
    Run full evaluation: metrics for all 3 models, all plots, comparison, cost analysis, report.

    Returns:
        (metrics_by_model, cost_by_model).
    """
    evaluator = ModelEvaluator(results_dir=results_dir, config_path=config_path)
    scores_dict = {
        "Isolation Forest": iso_scores,
        "Autoencoder": ae_scores,
        "Ensemble": ensemble_scores,
    }
    for name, scores in scores_dict.items():
        evaluator.evaluate_single_model(y_true, scores, name)
    metrics_by_model = evaluator.compare_models(y_true, scores_dict)
    cost_by_model = {}
    for name, scores in scores_dict.items():
        cost_by_model[name] = evaluator.cost_analysis(y_true, scores, name=name, default_threshold=0.5)
    evaluator.generate_report(metrics_by_model, cost_by_model)
    return metrics_by_model, cost_by_model


def _main() -> None:
    """Full pipeline: load, split, features, train all models, evaluate, generate plots and report."""
    import logging
    from src.data.loader import load_transactions
    from src.data.splitter import TimeAwareSplitter
    from src.features.feature_store import SimpleFeatureStore
    from src.models.isolation_forest import IsolationForestModel
    from src.models.autoencoder import FraudAutoencoder
    from src.models.ensemble import EnsembleScorer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
    df = load_transactions()
    splitter = TimeAwareSplitter()
    train_df, val_df, _test_df = splitter.split(df)
    store = SimpleFeatureStore()
    store.fit(train_df)
    X_train, y_train = store.transform(train_df)
    X_val, y_val = store.transform(val_df)
    y_val_arr = np.asarray(y_val).ravel().astype(int)

    from src.models.xgboost_model import XGBoostFraudModel

    iso = IsolationForestModel()
    iso.fit(X_train, y_train)
    ae = FraudAutoencoder()
    ae.fit(X_train.values, y_train.values)
    xgb = XGBoostFraudModel()
    xgb.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    iso_val = iso.score(X_val)
    ae_val = ae.score(X_val.values)
    xgb_val = xgb.predict_proba(X_val)

    ensemble = EnsembleScorer()
    ensemble.optimize_weights(iso_val, ae_val, xgb_val, y_val_arr)
    ensemble.optimize_threshold(iso_val, ae_val, xgb_val, y_val_arr, method="cost")
    ens_val = ensemble.score_batch(iso_val, ae_val, xgb_val)

    metrics, costs = run_full_evaluation(
        y_val_arr,
        iso_val,
        ae_val,
        ens_val,
        results_dir=DEFAULT_RESULTS_DIR,
        config_path=DEFAULT_CONFIG_PATH,
    )
    print("\nEnsemble performance (validation):")
    m = metrics["Ensemble"]
    print("  PR-AUC: %.4f  ROC-AUC: %.4f  F1: %.4f  Precision: %.4f  Recall: %.4f" % (m["pr_auc"], m["roc_auc"], m["f1"], m["precision"], m["recall"]))
    c = costs["Ensemble"]
    print("  Cost-optimal threshold: %.2f  Cost at optimal: $%.0f  Cost at default: $%.0f" % (c["cost_optimal_threshold"], c["cost_at_optimal"], c["cost_at_default"]))
    print("\nReport and plots saved to %s/" % DEFAULT_RESULTS_DIR)


if __name__ == "__main__":
    _main()
