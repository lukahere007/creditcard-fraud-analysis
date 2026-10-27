from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud import add_features, compare_models, load_data, make_models, net_savings, split_xy

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def df():
    return load_data(DATA_DIR)


def test_sample_loads_with_expected_columns(df):
    assert {"Time", "Amount", "Class"} <= set(df.columns)
    assert set(df["Class"].unique()) == {0, 1}


def test_add_features_is_rowwise_and_drops_time(df):
    feats = add_features(df)
    assert "Time" not in feats
    assert np.allclose(feats["HourSin"] ** 2 + feats["HourCos"] ** 2, 1)
    # Stateless: transforming a subset equals subsetting the transform.
    pd.testing.assert_frame_equal(add_features(df.iloc[:50]), feats.iloc[:50])


def test_split_xy_excludes_target_and_amount(df):
    X, y = split_xy(add_features(df))
    assert "Class" not in X and "Amount" not in X
    assert len(X) == len(y)


def test_net_savings_counts():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1])
    amounts = np.array([100.0, 50.0, 10.0, 10.0])
    out = net_savings(y, scores, amounts, threshold=0.5, review_cost=5)
    assert out["frauds_caught"] == 1
    assert out["frauds_missed"] == 1
    assert out["false_alarms"] == 1
    assert out["net_savings"] == 100 - 2 * 5


def test_models_beat_baseline_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(scale=0.5, size=n) > 2.2).astype(int))
    models = make_models()
    subset = {k: models[k] for k in ["Baseline (always 'not fraud')", "Logistic regression"]}
    res = compare_models(subset, X, y, n_splits=3, n_repeats=1)
    assert res.loc["Logistic regression", "pr_auc_mean"] > res.loc["Baseline (always 'not fraud')", "pr_auc_mean"] + 0.3
