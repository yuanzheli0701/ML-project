from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from advanced_modeling import fit_with_strategy, find_spec
from improved_modeling import (
    DROP_COLS,
    RANDOM_STATE,
    RARE_CLASSES,
    TARGET_COL,
    add_engineered_features,
    load_data,
)


N_SPLITS = 5
N_REPEATS = 10
MAJORITY_CLASSES = ("Bee", "Bumblebee")

ARTIFACTS = {
    "results": Path("repeated_cv_results.csv"),
    "per_class": Path("repeated_cv_per_class_results.csv"),
    "holdout_reports": Path("repeated_cv_holdout_reports.txt"),
    "summary": Path("repeated_cv_summary.json"),
    "accuracy_model": Path("best_model_repeated_cv_accuracy.pkl"),
    "macro_model": Path("best_model_repeated_cv_macro.pkl"),
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    factory: Callable[[], object]
    fit_strategy: str = "none"
    hierarchical: bool = False


def svm(c: float = 10, class_weight: str | None = None, probability: bool = False) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=c,
                    gamma="scale",
                    class_weight=class_weight,
                    probability=probability,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def lda_shrinkage() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def final_voting() -> VotingClassifier:
    return VotingClassifier(
        estimators=[
            ("svm", svm(c=10, probability=True)),
            ("lda", lda_shrinkage()),
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


class TwoStageBugClassifier(BaseEstimator, ClassifierMixin):
    """Coarse majority-vs-other classifier followed by specialist classifiers."""

    def __init__(
        self,
        coarse_model: object,
        majority_model: object,
        minority_model: object,
        majority_classes: tuple[str, ...] = MAJORITY_CLASSES,
    ):
        self.coarse_model = coarse_model
        self.majority_model = majority_model
        self.minority_model = minority_model
        self.majority_classes = majority_classes

    def fit(self, X, y):
        y_series = pd.Series(y).reset_index(drop=True)
        X_df = pd.DataFrame(X).reset_index(drop=True)
        majority_mask = y_series.isin(self.majority_classes)
        coarse_y = np.where(majority_mask, "majority", "other")

        self.coarse_model_ = clone(self.coarse_model)
        self.majority_model_ = clone(self.majority_model)
        self.minority_model_ = clone(self.minority_model)

        self.coarse_model_.fit(X_df, coarse_y)
        self.majority_model_.fit(X_df.loc[majority_mask], y_series.loc[majority_mask])
        self.minority_model_.fit(X_df.loc[~majority_mask], y_series.loc[~majority_mask])
        self.classes_ = np.array(sorted(y_series.unique()))
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X)
        coarse_pred = np.asarray(self.coarse_model_.predict(X_df))
        pred = np.empty(len(X_df), dtype=object)

        majority_idx = np.where(coarse_pred == "majority")[0]
        other_idx = np.where(coarse_pred != "majority")[0]
        if len(majority_idx):
            pred[majority_idx] = self.majority_model_.predict(X_df.iloc[majority_idx])
        if len(other_idx):
            pred[other_idx] = self.minority_model_.predict(X_df.iloc[other_idx])
        return pred


def candidates() -> list[Candidate]:
    return [
        Candidate("Voting SVM+LDA+ET", "fusion", final_voting, "none"),
        Candidate("Voting SVM+LDA+ET hybrid_50", "fusion_resampled", final_voting, "hybrid_50"),
        Candidate("SVM RBF C=10", "svm", lambda: svm(c=10), "none"),
        Candidate("SVM RBF C=10 balanced", "svm_weighted", lambda: svm(c=10, class_weight="balanced"), "none"),
        Candidate("SVM RBF C=10 hybrid_50", "svm_resampled", lambda: svm(c=10), "hybrid_50"),
        Candidate("LDA shrinkage", "linear", lda_shrinkage, "none"),
        Candidate(
            "Two-stage SVM/SVM/LDA",
            "two_stage",
            lambda: TwoStageBugClassifier(
                coarse_model=svm(c=3, class_weight="balanced"),
                majority_model=svm(c=10),
                minority_model=lda_shrinkage(),
            ),
            "none",
            True,
        ),
        Candidate(
            "Two-stage SVM/SVM/SVM",
            "two_stage",
            lambda: TwoStageBugClassifier(
                coarse_model=svm(c=3, class_weight="balanced"),
                majority_model=svm(c=10),
                minority_model=svm(c=3, class_weight="balanced"),
            ),
            "none",
            True,
        ),
        Candidate(
            "Two-stage LDA/LDA/LDA",
            "two_stage",
            lambda: TwoStageBugClassifier(
                coarse_model=lda_shrinkage(),
                majority_model=lda_shrinkage(),
                minority_model=lda_shrinkage(),
            ),
            "none",
            True,
        ),
    ]


def fit_candidate(candidate: Candidate, X_train: pd.DataFrame, y_train: pd.Series, seed: int):
    model = candidate.factory()
    if candidate.hierarchical:
        model.fit(X_train, y_train)
        return model
    return fit_with_strategy(model, X_train, y_train, candidate.fit_strategy, seed)


def evaluate_repeated_cv(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    labels = sorted(y.unique())
    metric_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []

    for candidate in candidates():
        fold_metrics = []
        per_class_fold_rows = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model = fit_candidate(candidate, X_train, y_train, RANDOM_STATE + fold_idx)
            pred = model.predict(X_test)

            fold_metrics.append(
                {
                    "accuracy": accuracy_score(y_test, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                    "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                    "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
                }
            )
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test,
                pred,
                labels=labels,
                zero_division=0,
            )
            for cls, p, r, f, s in zip(labels, precision, recall, f1, support):
                per_class_fold_rows.append(
                    {
                        "Model": candidate.name,
                        "Family": candidate.family,
                        "Fit Strategy": candidate.fit_strategy,
                        "Class": cls,
                        "Precision": p,
                        "Recall": r,
                        "F1": f,
                        "Support": s,
                    }
                )

        fold_df = pd.DataFrame(fold_metrics)
        metric_rows.append(
            {
                "Model": candidate.name,
                "Family": candidate.family,
                "Fit Strategy": candidate.fit_strategy,
                "Folds": N_SPLITS * N_REPEATS,
                "CV Accuracy Mean": fold_df["accuracy"].mean(),
                "CV Accuracy Std": fold_df["accuracy"].std(ddof=0),
                "CV Balanced Acc Mean": fold_df["balanced_accuracy"].mean(),
                "CV Balanced Acc Std": fold_df["balanced_accuracy"].std(ddof=0),
                "CV Macro F1 Mean": fold_df["macro_f1"].mean(),
                "CV Macro F1 Std": fold_df["macro_f1"].std(ddof=0),
                "CV Weighted F1 Mean": fold_df["weighted_f1"].mean(),
                "CV Weighted F1 Std": fold_df["weighted_f1"].std(ddof=0),
            }
        )
        per_class_rows.extend(per_class_fold_rows)

    results = pd.DataFrame(metric_rows).sort_values(
        ["CV Accuracy Mean", "CV Macro F1 Mean"], ascending=False
    )
    per_class = (
        pd.DataFrame(per_class_rows)
        .groupby(["Model", "Family", "Fit Strategy", "Class"], as_index=False)
        .agg(
            Precision_Mean=("Precision", "mean"),
            Recall_Mean=("Recall", "mean"),
            F1_Mean=("F1", "mean"),
            Support_Total=("Support", "sum"),
        )
    )
    return results, per_class


def write_holdout_reports(
    selected: dict[str, pd.Series],
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    labels = sorted(y.unique())
    sections = []
    by_name = {candidate.name: candidate for candidate in candidates()}

    for title, row in selected.items():
        candidate = by_name[str(row["Model"])]
        model = fit_candidate(candidate, X_train, y_train, RANDOM_STATE + 1000)
        pred = model.predict(X_test)
        sections.append(f"=== {title}: {candidate.name} / {candidate.fit_strategy} ===")
        sections.append(classification_report(y_test, pred, labels=labels, zero_division=0))
        sections.append("Confusion matrix labels: " + ", ".join(labels))
        sections.append(str(confusion_matrix(y_test, pred, labels=labels)))
        sections.append("")

    ARTIFACTS["holdout_reports"].write_text("\n".join(sections), encoding="utf-8")


def train_and_save(row: pd.Series, X: pd.DataFrame, y: pd.Series, output_path: Path) -> None:
    by_name = {candidate.name: candidate for candidate in candidates()}
    candidate = by_name[str(row["Model"])]
    model = fit_candidate(candidate, X, y, RANDOM_STATE + 2000)
    base_df = load_data()
    _, engineered_added = add_engineered_features(base_df)
    payload = {
        "model": model,
        "model_name": candidate.name,
        "family": candidate.family,
        "fit_strategy": candidate.fit_strategy,
        "feature_cols": list(X.columns),
        "engineered_feature_cols": engineered_added,
        "rare_classes_removed": RARE_CLASSES,
        "target_col": TARGET_COL,
        "label_mode": "string_labels",
        "metrics": row.to_dict(),
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    df, _ = add_engineered_features(load_data())
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]

    results, per_class = evaluate_repeated_cv(X, y)
    results.to_csv(ARTIFACTS["results"], index=False)
    per_class.to_csv(ARTIFACTS["per_class"], index=False)

    accuracy_winner = results.sort_values(["CV Accuracy Mean", "CV Macro F1 Mean"], ascending=False).iloc[0]
    macro_winner = results.sort_values(["CV Macro F1 Mean", "CV Accuracy Mean"], ascending=False).iloc[0]

    train_and_save(accuracy_winner, X, y, ARTIFACTS["accuracy_model"])
    train_and_save(macro_winner, X, y, ARTIFACTS["macro_model"])
    write_holdout_reports(
        {"Repeated-CV accuracy winner": accuracy_winner, "Repeated-CV macro-F1 winner": macro_winner},
        X,
        y,
    )

    summary = {
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "folds": N_SPLITS * N_REPEATS,
        "majority_classes": list(MAJORITY_CLASSES),
        "accuracy_winner": accuracy_winner.to_dict(),
        "macro_winner": macro_winner.to_dict(),
        "artifacts": {key: str(path) for key, path in ARTIFACTS.items()},
    }
    ARTIFACTS["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Repeated Stratified CV results")
    print(
        results[
            [
                "Model",
                "Family",
                "Fit Strategy",
                "CV Accuracy Mean",
                "CV Accuracy Std",
                "CV Macro F1 Mean",
                "CV Macro F1 Std",
                "CV Balanced Acc Mean",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\nAccuracy winner:")
    print(
        f"{accuracy_winner['Model']} CV accuracy={accuracy_winner['CV Accuracy Mean']:.4f}, "
        f"macro F1={accuracy_winner['CV Macro F1 Mean']:.4f}"
    )
    print("\nMacro-F1 winner:")
    print(
        f"{macro_winner['Model']} CV accuracy={macro_winner['CV Accuracy Mean']:.4f}, "
        f"macro F1={macro_winner['CV Macro F1 Mean']:.4f}"
    )


if __name__ == "__main__":
    main()
