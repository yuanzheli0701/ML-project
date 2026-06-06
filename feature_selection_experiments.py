from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
TARGET_COL = "bug type"
DROP_COLS = ["ID", TARGET_COL, "species"]
N_SPLITS = 5
N_REPEATS = 10

OUT = {
    "results": Path("feature_selection_results.csv"),
    "selected_features": Path("selected_features.csv"),
    "summary": Path("feature_selection_summary.json"),
    "model": Path("best_model_feature_selected.pkl"),
}


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.n_features_in_ = X_df.shape[1]
        corr = X_df.corr().abs().fillna(0.0)
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > self.threshold)]
        self.keep_indices_ = np.array([i for i in range(X_df.shape[1]) if i not in drop_cols], dtype=int)
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_indices_]

    def get_support(self):
        support = np.zeros(self.n_features_in_, dtype=bool)
        support[self.keep_indices_] = True
        return support

    def fit_transform(self, X, y=None, **fit_params):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self.fit(X, y).transform(X)


def voting_model() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            (
                "et",
                ExtraTreesClassifier(
                    n_estimators=500,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ],
        voting="soft",
    )


def pipeline(selector_name: str, selector) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if selector is not None:
        steps.append(("selector", selector))
    steps.append(("model", voting_model()))
    return Pipeline(steps)


def candidate_pipelines(n_features: int) -> dict[str, Pipeline]:
    candidates = {
        "all_80_features": pipeline("none", None),
        "variance_only": pipeline("variance", VarianceThreshold()),
        "corr_095": pipeline("corr", CorrelationFilter(threshold=0.95)),
        "corr_090": pipeline("corr", CorrelationFilter(threshold=0.90)),
    }

    for k in [15, 20, 25, 30, 35, 40, 50, 60]:
        if k < n_features:
            candidates[f"selectk_f_{k}"] = pipeline(
                "selectk",
                Pipeline(
                    [
                        ("variance", VarianceThreshold()),
                        ("select", SelectKBest(score_func=f_classif, k=k)),
                    ]
                ),
            )

    for k in [20, 30, 40, 50]:
        if k < n_features:
            candidates[f"extratrees_top_{k}"] = pipeline(
                "model_based",
                SelectFromModel(
                    ExtraTreesClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                    threshold=-np.inf,
                    max_features=k,
                ),
            )

    for k in [20, 30, 40]:
        if k < n_features:
            candidates[f"rfe_logreg_{k}"] = pipeline(
                "rfe",
                RFE(
                    estimator=LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                    n_features_to_select=k,
                    step=0.2,
                ),
            )

    return candidates


def selected_feature_names(model: Pipeline, feature_cols: list[str]) -> list[str]:
    if "selector" not in model.named_steps:
        return feature_cols
    selector = model.named_steps["selector"]
    if isinstance(selector, Pipeline):
        current_names = np.array(feature_cols)
        for _, step in selector.steps:
            if hasattr(step, "get_support"):
                support = step.get_support()
                current_names = current_names[support]
        return current_names.tolist()
    if hasattr(selector, "get_support"):
        return np.array(feature_cols)[selector.get_support()].tolist()
    return feature_cols


def evaluate_candidate(name: str, model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    rows = []
    selected_counts = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold_model = pickle.loads(pickle.dumps(model))
        fold_model.fit(X_train, y_train)
        pred = fold_model.predict(X_test)
        rows.append(
            {
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            }
        )
        selected_counts.append(len(selected_feature_names(fold_model, list(X.columns))))

    fold_df = pd.DataFrame(rows)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    holdout_model = pickle.loads(pickle.dumps(model))
    holdout_model.fit(X_train, y_train)
    holdout_pred = holdout_model.predict(X_test)

    return {
        "Selection": name,
        "Selected Feature Count Mean": float(np.mean(selected_counts)),
        "Selected Feature Count Min": int(np.min(selected_counts)),
        "Selected Feature Count Max": int(np.max(selected_counts)),
        "CV Accuracy Mean": fold_df["accuracy"].mean(),
        "CV Accuracy Std": fold_df["accuracy"].std(ddof=0),
        "CV Balanced Acc Mean": fold_df["balanced_accuracy"].mean(),
        "CV Macro F1 Mean": fold_df["macro_f1"].mean(),
        "CV Macro F1 Std": fold_df["macro_f1"].std(ddof=0),
        "CV Weighted F1 Mean": fold_df["weighted_f1"].mean(),
        "Holdout Accuracy": accuracy_score(y_test, holdout_pred),
        "Holdout Balanced Acc": balanced_accuracy_score(y_test, holdout_pred),
        "Holdout Macro F1": f1_score(y_test, holdout_pred, average="macro", zero_division=0),
    }


def main() -> None:
    df = pd.read_csv("features_engineered.csv")
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]

    candidates = candidate_pipelines(len(feature_cols))
    rows = []
    for name, model in candidates.items():
        print(f"Evaluating {name}...")
        rows.append(evaluate_candidate(name, model, X, y))

    results = pd.DataFrame(rows).sort_values(
        ["CV Accuracy Mean", "CV Macro F1 Mean", "Selected Feature Count Mean"],
        ascending=[False, False, True],
    )
    results.to_csv(OUT["results"], index=False)

    accuracy_winner = results.iloc[0]
    compact_pool = results[
        (results["Selected Feature Count Mean"] <= 50)
        & (results["CV Accuracy Mean"] >= accuracy_winner["CV Accuracy Mean"] - 0.005)
    ].copy()
    if compact_pool.empty:
        recommended_row = accuracy_winner
    else:
        recommended_row = compact_pool.sort_values(
            ["CV Macro F1 Mean", "CV Accuracy Mean", "Selected Feature Count Mean"],
            ascending=[False, False, True],
        ).iloc[0]

    recommended_name = str(recommended_row["Selection"])
    best_model = candidates[recommended_name]
    best_model.fit(X, y)
    selected = selected_feature_names(best_model, feature_cols)

    selected_df = pd.DataFrame({"feature": selected})
    selected_df.to_csv(OUT["selected_features"], index=False)

    payload = {
        "model": best_model,
        "model_name": f"Feature-selected Voting ({recommended_name})",
        "selection": recommended_name,
        "feature_cols": feature_cols,
        "selected_features": selected,
        "target_col": TARGET_COL,
        "label_mode": "string_labels",
        "metrics": recommended_row.to_dict(),
        "accuracy_winner_metrics": accuracy_winner.to_dict(),
    }
    with OUT["model"].open("wb") as f:
        pickle.dump(payload, f)

    summary = {
        "accuracy_winner": str(accuracy_winner["Selection"]),
        "accuracy_winner_metrics": accuracy_winner.to_dict(),
        "recommended_compact_selection": recommended_name,
        "recommended_compact_metrics": recommended_row.to_dict(),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "artifacts": {k: str(v) for k, v in OUT.items()},
    }
    OUT["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFeature selection results")
    print(
        results[
            [
                "Selection",
                "Selected Feature Count Mean",
                "CV Accuracy Mean",
                "CV Accuracy Std",
                "CV Macro F1 Mean",
                "Holdout Accuracy",
            ]
        ]
        .head(12)
        .round(4)
        .to_string(index=False)
    )
    print(f"\nAccuracy winner: {accuracy_winner['Selection']}")
    print(f"Recommended compact selection: {recommended_name}")
    print(f"Selected features: {len(selected)}")
    print(", ".join(selected))


if __name__ == "__main__":
    main()
