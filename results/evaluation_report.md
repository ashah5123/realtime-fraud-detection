# Fraud Detection Model Evaluation Report

## 1. Classification Metrics

| Model | PR-AUC | ROC-AUC | Optimal Threshold | Precision | Recall | F1 |
|-------|--------|---------|-------------------|-----------|--------|-----|
| Isolation Forest | 0.5381 | 0.9055 | 0.671 | 0.7368 | 0.5385 | 0.6222 |
| Autoencoder | 0.2620 | 0.8505 | 1.000 | 0.3714 | 0.5000 | 0.4262 |
| Ensemble | 0.7271 | 0.9784 | 0.514 | 0.7500 | 0.6923 | 0.7200 |

## 2. Cost-Benefit Analysis

Costs: FP = $50, FN = $500

| Model | Cost-Optimal Threshold | Cost at Optimal | Cost at Default (0.5) |
|-------|-------------------------|-----------------|------------------------|
| Isolation Forest | 0.55 | $5300 | $5700 |
| Autoencoder | 0.55 | $6450 | $6800 |
| Ensemble | 0.51 | $3900 | $4200 |

## 3. Visualizations

### Model comparison
- ![PR curves comparison](pr_curves_comparison.png)

### Per-model plots

**Isolation Forest:**
- PR curve: ![](pr_curve_isolation_forest.png)
- ROC curve: ![](roc_curve_isolation_forest.png)
- Score distribution: ![](score_dist_isolation_forest.png)
- Confusion matrix: ![](confusion_isolation_forest.png)
- Threshold vs P/R: ![](threshold_tradeoff_isolation_forest.png)

**Autoencoder:**
- PR curve: ![](pr_curve_autoencoder.png)
- ROC curve: ![](roc_curve_autoencoder.png)
- Score distribution: ![](score_dist_autoencoder.png)
- Confusion matrix: ![](confusion_autoencoder.png)
- Threshold vs P/R: ![](threshold_tradeoff_autoencoder.png)

**Ensemble:**
- PR curve: ![](pr_curve_ensemble.png)
- ROC curve: ![](roc_curve_ensemble.png)
- Score distribution: ![](score_dist_ensemble.png)
- Confusion matrix: ![](confusion_ensemble.png)
- Threshold vs P/R: ![](threshold_tradeoff_ensemble.png)
