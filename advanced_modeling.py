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

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from improved_modeling import (
    DROP_COLS,
    RANDOM_STATE,
    RARE_CLASSES,
    TARGET_COL,
    add_engineered_features,
    load_data,
)


ARTIFACTS = {
    "advanced_results": Path("advanced_modeling_results.csv"),
    "top_summary": Path("advanced_top_summary.csv"),
    "holdout_reports": Path("advanced_holdout_reports.txt"),
    "accuracy_model": Path("best_model_engineered.pkl"),
    "balanced_model": Path("best_model_engineered_balanced.pkl"),
    "model_summary": Path("final_engineered_model_summary.json"),
    "report_md": Path("modeling_report.md"),
    "report_html": Path("modeling_report.html"),
    "report_pdf": Path("modeling_report.pdf"),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], BaseEstimator]
    family: str


def scaled(model: BaseEstimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def model_specs() -> list[ModelSpec]:
    svm = scaled(SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE))
    svm_prob = scaled(
        SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE)
    )
    svm_balanced = scaled(
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )
    )
    lda_shrinkage = scaled(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    extra_trees = ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=1
    )

    return [
        ModelSpec(
            "SVM RBF C=10",
            lambda: scaled(SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)),
            "svm",
        ),
        ModelSpec(
            "SVM RBF C=10 balanced",
            lambda: scaled(
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
            "svm_weighted",
        ),
        ModelSpec(
            "SVM RBF tuned gamma=0.01",
            lambda: scaled(SVC(kernel="rbf", C=10, gamma=0.01, random_state=RANDOM_STATE)),
            "svm",
        ),
        ModelSpec(
            "Logistic Regression balanced",
            lambda: scaled(LogisticRegression(max_iter=5000, class_weight="balanced")),
            "linear_weighted",
        ),
        ModelSpec(
            "LDA shrinkage",
            lambda: scaled(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            "linear",
        ),
        ModelSpec(
            "QDA PCA4 regularized",
            lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=4, random_state=RANDOM_STATE)),
                    ("model", QuadraticDiscriminantAnalysis(reg_param=0.2)),
                ]
            ),
            "linear",
        ),
        ModelSpec(
            "KNN k=5 distance",
            lambda: scaled(KNeighborsClassifier(n_neighbors=5, weights="distance")),
            "neighbors",
        ),
        ModelSpec(
            "GaussianNB",
            lambda: GaussianNB(),
            "bayes",
        ),
        ModelSpec(
            "Random Forest balanced",
            lambda: RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "tree_weighted",
        ),
        ModelSpec(
            "ExtraTrees balanced",
            lambda: ExtraTreesClassifier(
                n_estimators=800,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "tree_weighted",
        ),
        ModelSpec(
            "HistGradientBoosting",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            ),
            "boosting",
        ),
        ModelSpec(
            "GradientBoosting",
            lambda: GradientBoostingClassifier(n_estimators=250, random_state=RANDOM_STATE),
            "boosting",
        ),
        ModelSpec(
            "AdaBoost",
            lambda: AdaBoostClassifier(n_estimators=250, learning_rate=0.4, random_state=RANDOM_STATE),
            "boosting",
        ),
        ModelSpec(
            "Voting SVM+LDA+ET",
            lambda: VotingClassifier(
                estimators=[
                    ("svm", svm_prob),
                    ("lda", lda_shrinkage),
                    ("et", extra_trees),
                ],
                voting="soft",
            ),
            "ensemble",
        ),
        ModelSpec(
            "Stacking SVM+LDA+ET",
            lambda: StackingClassifier(
                estimators=[
                    ("svm", svm),
                    ("svm_balanced", svm_balanced),
                    ("lda", lda_shrinkage),
                    ("et", extra_trees),
                ],
                final_estimator=LogisticRegression(max_iter=3000, class_weight="balanced"),
                cv=3,
                n_jobs=1,
            ),
            "ensemble",
        ),
    ]


def resample_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if strategy == "none":
        return X.copy(), y.copy()

    counts = y.value_counts()
    rng = np.random.default_rng(seed)

    if strategy == "oversample_equal":
        targets = {label: int(counts.max()) for label in counts.index}
    elif strategy == "hybrid_50":
        target = max(int(np.ceil(counts.max() * 0.5)), int(counts.min()))
        targets = {label: target for label in counts.index}
    elif strategy == "undersample_mild":
        cap = max(int(np.ceil(counts.median() * 3)), int(counts.min()))
        targets = {label: min(int(count), cap) for label, count in counts.items()}
    else:
        raise ValueError(f"Unknown resampling strategy: {strategy}")

    sampled_indices: list[int] = []
    for label, target_count in targets.items():
        cls_indices = y[y == label].index.to_numpy()
        replace = target_count > len(cls_indices)
        picked = rng.choice(cls_indices, size=target_count, replace=replace)
        sampled_indices.extend(picked.tolist())

    rng.shuffle(sampled_indices)
    return X.loc[sampled_indices].reset_index(drop=True), y.loc[sampled_indices].reset_index(drop=True)


def fit_with_strategy(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
    seed: int,
) -> BaseEstimator:
    X_fit, y_fit = resample_training_data(X, y, strategy, seed)
    estimator.fit(X_fit, y_fit)
    return estimator


def evaluate_one_model(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str,
) -> dict[str, float | str]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = fit_with_strategy(
            spec.factory(),
            X_train,
            y_train,
            strategy,
            seed=RANDOM_STATE + fold,
        )
        pred = model.predict(X_test)
        fold_rows.append(
            {
                "accuracy": accuracy_score(y_test, pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    holdout_model = fit_with_strategy(
        spec.factory(), X_train, y_train, strategy, seed=RANDOM_STATE + 1000
    )
    holdout_pred = holdout_model.predict(X_test)

    return {
        "Model": spec.name,
        "Family": spec.family,
        "Imbalance Strategy": strategy,
        "Feature Count": X.shape[1],
        "CV Accuracy Mean": fold_df["accuracy"].mean(),
        "CV Accuracy Std": fold_df["accuracy"].std(ddof=0),
        "CV Balanced Acc Mean": fold_df["balanced_accuracy"].mean(),
        "CV Macro F1 Mean": fold_df["macro_f1"].mean(),
        "CV Weighted F1 Mean": fold_df["weighted_f1"].mean(),
        "Holdout Accuracy": accuracy_score(y_test, holdout_pred),
        "Holdout Balanced Acc": balanced_accuracy_score(y_test, holdout_pred),
        "Holdout Macro F1": f1_score(y_test, holdout_pred, average="macro", zero_division=0),
        "Holdout Weighted F1": f1_score(y_test, holdout_pred, average="weighted", zero_division=0),
    }


def evaluate_all(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    strategies = ["none", "oversample_equal", "hybrid_50", "undersample_mild"]
    rows: list[dict[str, float | str]] = []
    failures: list[str] = []
    failure_path = Path("advanced_modeling_failures.txt")

    for strategy in strategies:
        for spec in model_specs():
            try:
                rows.append(evaluate_one_model(spec, X, y, strategy))
            except Exception as exc:  # Keep the sweep running if one experimental model fails.
                failures.append(f"{strategy} | {spec.name}: {type(exc).__name__}: {exc}")

    result = pd.DataFrame(rows)
    result["Selection Score"] = result["CV Accuracy Mean"] + 0.20 * result["CV Macro F1 Mean"]
    result = result.sort_values(
        ["CV Accuracy Mean", "CV Macro F1 Mean", "Holdout Accuracy"], ascending=False
    ).reset_index(drop=True)

    if failures:
        failure_path.write_text("\n".join(failures), encoding="utf-8")
    elif failure_path.exists():
        failure_path.unlink()

    return result


def find_spec(name: str) -> ModelSpec:
    for spec in model_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown model spec: {name}")


def train_and_save_model(
    row: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    base_feature_cols: list[str],
    engineered_feature_cols: list[str],
    output_path: Path,
) -> BaseEstimator:
    spec = find_spec(str(row["Model"]))
    model = fit_with_strategy(
        spec.factory(),
        X,
        y,
        str(row["Imbalance Strategy"]),
        seed=RANDOM_STATE + 2000,
    )
    payload = {
        "model": model,
        "model_name": str(row["Model"]),
        "imbalance_strategy": str(row["Imbalance Strategy"]),
        "feature_cols": list(X.columns),
        "base_feature_cols": base_feature_cols,
        "engineered_feature_cols": engineered_feature_cols,
        "rare_classes_removed": RARE_CLASSES,
        "target_col": TARGET_COL,
        "label_mode": "string_labels",
        "metrics": row.to_dict(),
        "notes": (
            "Use improved_modeling.add_engineered_features on base extracted features "
            "before selecting feature_cols for prediction."
        ),
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    return model


def write_holdout_reports(
    candidates: dict[str, pd.Series],
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    labels = sorted(y.unique())
    sections: list[str] = []

    for title, row in candidates.items():
        spec = find_spec(str(row["Model"]))
        model = fit_with_strategy(
            spec.factory(), X_train, y_train, str(row["Imbalance Strategy"]), RANDOM_STATE + 1000
        )
        pred = model.predict(X_test)
        sections.append(f"=== {title}: {row['Model']} / {row['Imbalance Strategy']} ===")
        sections.append(classification_report(y_test, pred, labels=labels, zero_division=0))
        sections.append("Confusion matrix labels: " + ", ".join(labels))
        sections.append(str(confusion_matrix(y_test, pred, labels=labels)))
        sections.append("")

    ARTIFACTS["holdout_reports"].write_text("\n".join(sections), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:
    view = df[columns].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep, *rows])


def write_report(
    df: pd.DataFrame,
    class_counts: pd.Series,
    accuracy_winner: pd.Series,
    balanced_winner: pd.Series,
    base_comparison: pd.DataFrame,
) -> None:
    top_cols = [
        "Model",
        "Imbalance Strategy",
        "CV Accuracy Mean",
        "CV Macro F1 Mean",
        "Holdout Accuracy",
        "Holdout Macro F1",
    ]
    strategy_cols = [
        "Imbalance Strategy",
        "CV Accuracy Mean",
        "CV Macro F1 Mean",
        "Holdout Accuracy",
        "Holdout Macro F1",
    ]

    strategy_summary = (
        df.groupby("Imbalance Strategy", as_index=False)[
            ["CV Accuracy Mean", "CV Macro F1 Mean", "Holdout Accuracy", "Holdout Macro F1"]
        ]
        .max()
        .sort_values(["CV Accuracy Mean", "CV Macro F1 Mean"], ascending=False)
    )

    report = f"""# Engineered Feature Modeling Report

Generated on 2026-05-31.

## Dataset

- Source table: `features.csv`
- Samples after removing rare one-sample classes: {int(class_counts.sum())}
- Removed classes: {", ".join(RARE_CLASSES)}
- Base features: 39
- Engineered features added: 41
- Final engineered feature count: 80

Class distribution:

{markdown_table(class_counts.rename_axis("Class").reset_index(name="Count"), ["Class", "Count"], max_rows=10)}

## Validation Fix

The original notebook scaled all samples before cross-validation. The revised workflow puts
`StandardScaler` inside each model pipeline, so scaling is fitted only on the training fold.

{markdown_table(base_comparison, ["Feature Set", "Model", "CV Accuracy Mean", "CV Macro F1 Mean", "Holdout Accuracy"], max_rows=6)}

## Feature Engineering

The engineered table adds color contrast, channel ratios, yellow/darkness proxies, bug-background
brightness differences, symmetry combinations, shape ratios, Hu moment aggregates, and texture
interaction features. This improved the best SVM CV accuracy from roughly 0.7504 to 0.7702 in the
first corrected experiment.

## Imbalance Strategies

Tested strategies:

- `none`: no resampling.
- `oversample_equal`: duplicate minority-class training samples up to the majority count.
- `hybrid_50`: resample each class to about 50% of the majority count.
- `undersample_mild`: cap very large classes while keeping minority classes unchanged.

Best result observed under each strategy:

{markdown_table(strategy_summary, strategy_cols, max_rows=10)}

## New Models

The sweep tried SVM variants, Logistic Regression, LDA/QDA, KNN, Naive Bayes, Random Forest,
ExtraTrees, Gradient Boosting, HistGradientBoosting, AdaBoost, soft Voting, and Stacking.

Top models by CV accuracy:

{markdown_table(df, top_cols, max_rows=12)}

## Final Saved Models

Accuracy-oriented model:

- File: `best_model_engineered.pkl`
- Model: {accuracy_winner["Model"]}
- Imbalance strategy: {accuracy_winner["Imbalance Strategy"]}
- CV accuracy: {accuracy_winner["CV Accuracy Mean"]:.4f} +/- {accuracy_winner["CV Accuracy Std"]:.4f}
- CV macro F1: {accuracy_winner["CV Macro F1 Mean"]:.4f}

Balanced-class model:

- File: `best_model_engineered_balanced.pkl`
- Model: {balanced_winner["Model"]}
- Imbalance strategy: {balanced_winner["Imbalance Strategy"]}
- CV accuracy: {balanced_winner["CV Accuracy Mean"]:.4f} +/- {balanced_winner["CV Accuracy Std"]:.4f}
- CV macro F1: {balanced_winner["CV Macro F1 Mean"]:.4f}

## Recommendation

For the hidden test set, use `best_model_engineered.pkl` if the score is plain accuracy and the
test distribution is expected to resemble the training distribution. Use
`best_model_engineered_balanced.pkl` if missing minority classes is penalized heavily or if macro
F1/balanced accuracy matters.

To predict future samples, first extract the original 39 base features, call
`improved_modeling.add_engineered_features`, then select the saved `feature_cols` from the pickle
payload before calling the saved model.
"""
    ARTIFACTS["report_md"].write_text(report, encoding="utf-8")

    html = report
    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = html.replace("\n", "\n")
    ARTIFACTS["report_html"].write_text(
        """<!doctype html>
<meta charset="utf-8">
<title>Engineered Feature Modeling Report</title>
<style>
body { font-family: Segoe UI, Arial, sans-serif; max-width: 980px; margin: 36px auto; line-height: 1.45; color: #17202a; }
h1, h2 { color: #102a43; }
code { background: #f4f6f8; padding: 2px 4px; border-radius: 4px; }
pre { white-space: pre-wrap; background: #f8fafc; padding: 16px; border: 1px solid #d9e2ec; }
</style>
<pre>"""
        + html
        + "</pre>\n",
        encoding="utf-8",
    )
    write_pdf_report(report)


def write_pdf_report(report: str) -> None:
    try:
        from xml.sax.saxutils import escape

        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer
    except Exception as exc:
        ARTIFACTS["report_pdf"].with_suffix(".pdf.error.txt").write_text(
            f"PDF generation skipped: {type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        return

    doc = SimpleDocTemplate(
        str(ARTIFACTS["report_pdf"]),
        pagesize=landscape(A4),
        leftMargin=36,
        rightMargin=36,
        topMargin=32,
        bottomMargin=32,
    )
    styles = getSampleStyleSheet()
    body = styles["BodyText"]
    code = ParagraphStyle(
        "Code",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.2,
        leading=8.6,
    )

    story = []
    table_buffer: list[str] = []

    def flush_table() -> None:
        if table_buffer:
            story.append(Preformatted(escape("\n".join(table_buffer)), code))
            story.append(Spacer(1, 8))
            table_buffer.clear()

    for raw_line in report.splitlines():
        line = raw_line.rstrip()
        if line.startswith("|"):
            table_buffer.append(line)
            continue
        flush_table()

        if not line:
            story.append(Spacer(1, 6))
        elif line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), styles["Title"]))
            story.append(Spacer(1, 10))
        elif line.startswith("## "):
            story.append(Spacer(1, 8))
            story.append(Paragraph(escape(line[3:]), styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph(escape(line[2:]), body, bulletText="-"))
        elif line.startswith("Generated on "):
            story.append(Paragraph(escape(line), styles["Italic"]))
        else:
            story.append(Paragraph(escape(line), body))

    flush_table()
    doc.build(story)


def main() -> None:
    base_df = load_data()
    engineered_df, engineered_cols = add_engineered_features(base_df)
    base_feature_cols = [c for c in base_df.columns if c not in DROP_COLS]
    feature_cols = [c for c in engineered_df.columns if c not in DROP_COLS]
    X = engineered_df[feature_cols]
    y = engineered_df[TARGET_COL]

    results = evaluate_all(X, y)
    results.to_csv(ARTIFACTS["advanced_results"], index=False)

    top_summary = results.head(20)
    top_summary.to_csv(ARTIFACTS["top_summary"], index=False)

    accuracy_winner = results.sort_values(
        ["CV Accuracy Mean", "CV Macro F1 Mean", "Holdout Accuracy"], ascending=False
    ).iloc[0]
    balanced_winner = results.sort_values(
        ["CV Macro F1 Mean", "CV Accuracy Mean", "Holdout Macro F1"], ascending=False
    ).iloc[0]

    train_and_save_model(
        accuracy_winner,
        X,
        y,
        base_feature_cols,
        engineered_cols,
        ARTIFACTS["accuracy_model"],
    )
    train_and_save_model(
        balanced_winner,
        X,
        y,
        base_feature_cols,
        engineered_cols,
        ARTIFACTS["balanced_model"],
    )

    write_holdout_reports(
        {
            "Accuracy winner": accuracy_winner,
            "Balanced winner": balanced_winner,
        },
        X,
        y,
    )

    base_comparison = pd.read_csv("improved_modeling_results.csv").head(8)
    write_report(
        results,
        y.value_counts(),
        accuracy_winner,
        balanced_winner,
        base_comparison,
    )

    summary = {
        "accuracy_winner": accuracy_winner.to_dict(),
        "balanced_winner": balanced_winner.to_dict(),
        "artifacts": {key: str(path) for key, path in ARTIFACTS.items()},
    }
    ARTIFACTS["model_summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Top advanced modeling results")
    print(
        results[
            [
                "Model",
                "Imbalance Strategy",
                "CV Accuracy Mean",
                "CV Macro F1 Mean",
                "Holdout Accuracy",
                "Holdout Macro F1",
            ]
        ]
        .head(12)
        .round(4)
        .to_string(index=False)
    )
    print("\nAccuracy winner:")
    print(
        f"{accuracy_winner['Model']} / {accuracy_winner['Imbalance Strategy']} "
        f"CV={accuracy_winner['CV Accuracy Mean']:.4f}, "
        f"macroF1={accuracy_winner['CV Macro F1 Mean']:.4f}"
    )
    print("\nBalanced winner:")
    print(
        f"{balanced_winner['Model']} / {balanced_winner['Imbalance Strategy']} "
        f"CV={balanced_winner['CV Accuracy Mean']:.4f}, "
        f"macroF1={balanced_winner['CV Macro F1 Mean']:.4f}"
    )


if __name__ == "__main__":
    main()
