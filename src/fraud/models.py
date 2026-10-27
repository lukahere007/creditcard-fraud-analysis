"""Candidate models. Every preprocessing step lives inside the pipeline so that
scaling and oversampling are fit on training folds only (no leakage into validation)."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def make_models(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Baseline (always 'not fraud')": Pipeline(
            [("clf", DummyClassifier(strategy="prior"))]
        ),
        "Logistic regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "Random forest": Pipeline(
            [
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "Random forest + SMOTE": Pipeline(
            [
                ("smote", SMOTE(random_state=random_state)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, n_jobs=-1, random_state=random_state
                    ),
                ),
            ]
        ),
        "Gradient boosting": Pipeline(
            [
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
    }
