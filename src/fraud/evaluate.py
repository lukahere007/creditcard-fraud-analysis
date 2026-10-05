"""Cross-validated evaluation and cost-based threshold selection."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}


def compare_models(models, X, y, n_splits=5, n_repeats=3, random_state=42) -> pd.DataFrame:
    """Repeated stratified CV. Returns mean and std of PR-AUC and ROC-AUC per model.

    With very few positives a single train/test split is mostly noise, so we average
    over many splits and report the spread alongside the mean.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rows = []
    for name, model in models.items():
        res = cross_validate(clone(model), X, y, cv=cv, scoring=SCORING, n_jobs=1)
        rows.append(
            {
                "model": name,
                "pr_auc_mean": res["test_pr_auc"].mean(),
                "pr_auc_std": res["test_pr_auc"].std(),
                "roc_auc_mean": res["test_roc_auc"].mean(),
                "roc_auc_std": res["test_roc_auc"].std(),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("pr_auc_mean", ascending=False)


def out_of_fold_scores(model, X, y, n_splits=5, random_state=42) -> np.ndarray:
    """Fraud probability for every row, each predicted by a model that never saw that row."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return cross_val_predict(clone(model), X, y, cv=cv, method="predict_proba")[:, 1]


def net_savings(y_true, scores, amounts, threshold: float, review_cost: float) -> dict:
    """Money outcome of flagging every transaction with score >= threshold.

    Each flagged transaction costs `review_cost` to investigate. Each caught fraud
    saves its `Amount`. Missed fraud is the loss we fail to prevent.
    """
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts, dtype=float)
    flagged = np.asarray(scores) >= threshold
    tp = flagged & (y_true == 1)
    fn = ~flagged & (y_true == 1)
    fp = flagged & (y_true == 0)
    saved = amounts[tp].sum()
    review = flagged.sum() * review_cost
    return {
        "threshold": threshold,
        "flagged": int(flagged.sum()),
        "frauds_caught": int(tp.sum()),
        "frauds_missed": int(fn.sum()),
        "false_alarms": int(fp.sum()),
        "fraud_amount_caught": saved,
        "fraud_amount_missed": amounts[fn].sum(),
        "review_cost": review,
        "net_savings": saved - review,
    }


def threshold_table(y_true, scores, amounts, review_cost: float, thresholds=None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.r_[np.round(np.linspace(0.01, 0.99, 99), 2), 0.995, 0.999]
    return pd.DataFrame(
        [net_savings(y_true, scores, amounts, t, review_cost) for t in thresholds]
    ).set_index("threshold")
