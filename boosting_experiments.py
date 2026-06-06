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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from advanced_modeling import resample_training_data
from improved_modeling import (
    DROP_COLS,
    RANDOM_STATE,
    RARE_CLASSES,
    TARGET_COL,
    add_engineered_features,
    load_data,
)


ARTIFACTS = {
    "results": Path("boosting_model_results.csv"),
    "top_summary": Path("boosting_top_summary.csv"),
    "holdout_reports": Path("boosting_holdout_reports.txt"),
    "accuracy_model": Path("best_model_boosting.pkl"),
    "macro_model": Path("best_model_boosting_macro.pkl"),
    "summary": Path("boosting_model_summary.json"),
    "report_md": Path("boosting_report.md"),
}


@dataclass(frozen=True)
class BoostSpec:
    name: str
    family: str
    factory: Callable[[int], object]


def model_specs() -> list[BoostSpec]:
    return [
        BoostSpec(
            "XGBoost conservative",
            "xgboost",
            lambda n_classes: XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                n_estimators=220,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=3.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            ),
        ),
        BoostSpec(
            "XGBoost shallow",
            "xgboost",
            lambda n_classes: XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                n_estimators=320,
                max_depth=2,
                learning_rate=0.035,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.0,
                reg_lambda=4.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            ),
        ),
        BoostSpec(
            "XGBoost deeper",
            "xgboost",
            lambda n_classes: XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                n_estimators=180,
                max_depth=4,
                learning_rate=0.045,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
                verbosity=0,
            ),
        ),
        BoostSpec(
            "LightGBM compact",
            "lightgbm",
            lambda n_classes: LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                n_estimators=220,
                learning_rate=0.05,
                num_leaves=15,
                max_depth=4,
                min_child_samples=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.05,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
                force_col_wise=True,
            ),
        ),
        BoostSpec(
            "LightGBM shallow",
            "lightgbm",
            lambda n_classes: LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                n_estimators=320,
                learning_rate=0.035,
                num_leaves=7,
                max_depth=3,
                min_child_samples=5,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=3.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
                force_col_wise=True,
            ),
        ),
        BoostSpec(
            "LightGBM regularized",
            "lightgbm",
            lambda n_classes: LGBMClassifier(
                objective="multiclass",
                num_class=n_classes,
                n_estimators=180,
                learning_rate=0.045,
                num_leaves=11,
                max_depth=4,
                min_child_samples=10,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.2,
                reg_lambda=5.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbosity=-1,
                force_col_wise=True,
            ),
        ),
        BoostSpec(
            "CatBoost compact",
            "catboost",
            lambda n_classes: CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="Accuracy",
                iterations=220,
                depth=4,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
        ),
        BoostSpec(
            "CatBoost shallow",
            "catboost",
            lambda n_classes: CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="Accuracy",
                iterations=320,
                depth=3,
                learning_rate=0.035,
                l2_leaf_reg=5.0,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
        ),
        BoostSpec(
            "CatBoost regularized",
            "catboost",
            lambda n_classes: CatBoostClassifier(
                loss_function="MultiClass",
                eval_metric="Accuracy",
                iterations=180,
                depth=5,
                learning_rate=0.045,
                l2_leaf_reg=8.0,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            ),
        ),
    ]


def prediction_1d(pred: object) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.astype(int).reshape(-1)


def fit_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    seed: int,
) -> object:
    sample_weight = None
    if strategy == "balanced_weight":
        X_fit = X_train
        y_fit = y_train
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    else:
        X_fit, y_fit = resample_training_data(X_train, y_train, strategy, seed)

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_fit, y_fit, **fit_kwargs)
    return model


def evaluate_one(
    spec: BoostSpec,
    X: pd.DataFrame,
    y: pd.Series,
    n_classes: int,
    strategy: str,
) -> dict[str, float | str | int]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_rows: list[dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = fit_model(
            spec.factory(n_classes),
            X_train,
            y_train,
            strategy,
            RANDOM_STATE + fold,
        )
        pred = prediction_1d(model.predict(X_test))
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
    holdout_model = fit_model(
        spec.factory(n_classes),
        X_train,
        y_train,
        strategy,
        RANDOM_STATE + 1000,
    )
    holdout_pred = prediction_1d(holdout_model.predict(X_test))

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


def evaluate_all(X: pd.DataFrame, y: pd.Series, n_classes: int) -> pd.DataFrame:
    strategies = ["none", "balanced_weight", "oversample_equal", "hybrid_50"]
    rows: list[dict[str, float | str | int]] = []
    failures: list[str] = []

    for strategy in strategies:
        for spec in model_specs():
            try:
                rows.append(evaluate_one(spec, X, y, n_classes, strategy))
            except Exception as exc:
                failures.append(f"{strategy} | {spec.name}: {type(exc).__name__}: {exc}")

    failure_path = Path("boosting_failures.txt")
    if failures:
        failure_path.write_text("\n".join(failures), encoding="utf-8")
    elif failure_path.exists():
        failure_path.unlink()

    result = pd.DataFrame(rows)
    result["Selection Score"] = result["CV Accuracy Mean"] + 0.20 * result["CV Macro F1 Mean"]
    return result.sort_values(
        ["CV Accuracy Mean", "CV Macro F1 Mean", "Holdout Accuracy"], ascending=False
    ).reset_index(drop=True)


def find_spec(name: str) -> BoostSpec:
    for spec in model_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown model: {name}")


def train_and_save(
    row: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    label_encoder: LabelEncoder,
    base_feature_cols: list[str],
    engineered_feature_cols: list[str],
    output_path: Path,
) -> object:
    spec = find_spec(str(row["Model"]))
    model = fit_model(
        spec.factory(len(label_encoder.classes_)),
        X,
        y,
        str(row["Imbalance Strategy"]),
        RANDOM_STATE + 2000,
    )
    payload = {
        "model": model,
        "model_name": str(row["Model"]),
        "family": str(row["Family"]),
        "imbalance_strategy": str(row["Imbalance Strategy"]),
        "feature_cols": list(X.columns),
        "base_feature_cols": base_feature_cols,
        "engineered_feature_cols": engineered_feature_cols,
        "label_encoder": label_encoder,
        "rare_classes_removed": RARE_CLASSES,
        "target_col": TARGET_COL,
        "label_mode": "encoded_labels",
        "metrics": row.to_dict(),
        "notes": (
            "Use improved_modeling.add_engineered_features on base extracted features, "
            "select feature_cols, predict encoded labels, then inverse_transform with label_encoder."
        ),
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    return model


def write_holdout_reports(
    candidates: dict[str, pd.Series],
    X: pd.DataFrame,
    y: pd.Series,
    label_encoder: LabelEncoder,
) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    sections: list[str] = []

    for title, row in candidates.items():
        spec = find_spec(str(row["Model"]))
        model = fit_model(
            spec.factory(len(label_encoder.classes_)),
            X_train,
            y_train,
            str(row["Imbalance Strategy"]),
            RANDOM_STATE + 1000,
        )
        pred = prediction_1d(model.predict(X_test))
        sections.append(f"=== {title}: {row['Model']} / {row['Imbalance Strategy']} ===")
        sections.append(
            classification_report(
                y_test,
                pred,
                labels=list(range(len(label_encoder.classes_))),
                target_names=list(label_encoder.classes_),
                zero_division=0,
            )
        )
        sections.append("Confusion matrix labels: " + ", ".join(label_encoder.classes_))
        sections.append(
            str(confusion_matrix(y_test, pred, labels=list(range(len(label_encoder.classes_)))))
        )
        sections.append("")

    ARTIFACTS["holdout_reports"].write_text("\n".join(sections), encoding="utf-8")


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 12) -> str:
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
    results: pd.DataFrame,
    accuracy_winner: pd.Series,
    macro_winner: pd.Series,
    previous_best: pd.Series | None,
) -> None:
    cols = [
        "Model",
        "Family",
        "Imbalance Strategy",
        "CV Accuracy Mean",
        "CV Macro F1 Mean",
        "Holdout Accuracy",
        "Holdout Macro F1",
    ]
    lines = [
        "# XGBoost / LightGBM / CatBoost Experiments",
        "",
        "Feature set: 80 engineered features from `features_engineered.csv`.",
        "",
        "Top boosting-family results:",
        "",
        markdown_table(results, cols, max_rows=15),
        "",
        "Best boosting model by CV accuracy:",
        "",
        f"- File: `{ARTIFACTS['accuracy_model']}`",
        f"- Model: {accuracy_winner['Model']}",
        f"- Strategy: {accuracy_winner['Imbalance Strategy']}",
        f"- CV accuracy: {accuracy_winner['CV Accuracy Mean']:.4f} +/- {accuracy_winner['CV Accuracy Std']:.4f}",
        f"- CV macro F1: {accuracy_winner['CV Macro F1 Mean']:.4f}",
        f"- Holdout accuracy: {accuracy_winner['Holdout Accuracy']:.4f}",
        "",
        "Best boosting model by CV macro F1:",
        "",
        f"- File: `{ARTIFACTS['macro_model']}`",
        f"- Model: {macro_winner['Model']}",
        f"- Strategy: {macro_winner['Imbalance Strategy']}",
        f"- CV accuracy: {macro_winner['CV Accuracy Mean']:.4f}",
        f"- CV macro F1: {macro_winner['CV Macro F1 Mean']:.4f}",
        f"- Holdout macro F1: {macro_winner['Holdout Macro F1']:.4f}",
    ]
    if previous_best is not None:
        lines.extend(
            [
                "",
                "Previous best non-boosting/fusion baseline:",
                "",
                f"- Model: {previous_best['Model']} / {previous_best['Imbalance Strategy']}",
                f"- CV accuracy: {previous_best['CV Accuracy Mean']:.4f}",
                f"- CV macro F1: {previous_best['CV Macro F1 Mean']:.4f}",
                f"- Holdout accuracy: {previous_best['Holdout Accuracy']:.4f}",
            ]
        )
    ARTIFACTS["report_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    base_df = load_data()
    engineered_df, engineered_cols = add_engineered_features(base_df)
    base_feature_cols = [c for c in base_df.columns if c not in DROP_COLS]
    feature_cols = [c for c in engineered_df.columns if c not in DROP_COLS]
    X = engineered_df[feature_cols].astype(np.float32)

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(engineered_df[TARGET_COL]), index=engineered_df.index)

    results = evaluate_all(X, y, len(label_encoder.classes_))
    results.to_csv(ARTIFACTS["results"], index=False)
    results.head(20).to_csv(ARTIFACTS["top_summary"], index=False)

    accuracy_winner = results.sort_values(
        ["CV Accuracy Mean", "CV Macro F1 Mean", "Holdout Accuracy"], ascending=False
    ).iloc[0]
    macro_winner = results.sort_values(
        ["CV Macro F1 Mean", "CV Accuracy Mean", "Holdout Macro F1"], ascending=False
    ).iloc[0]

    train_and_save(
        accuracy_winner,
        X,
        y,
        label_encoder,
        base_feature_cols,
        engineered_cols,
        ARTIFACTS["accuracy_model"],
    )
    train_and_save(
        macro_winner,
        X,
        y,
        label_encoder,
        base_feature_cols,
        engineered_cols,
        ARTIFACTS["macro_model"],
    )
    write_holdout_reports(
        {"Accuracy winner": accuracy_winner, "Macro-F1 winner": macro_winner},
        X,
        y,
        label_encoder,
    )

    previous_best = None
    if Path("advanced_modeling_results.csv").exists():
        previous_best = pd.read_csv("advanced_modeling_results.csv").iloc[0]

    write_report(results, accuracy_winner, macro_winner, previous_best)

    summary = {
        "classes": list(label_encoder.classes_),
        "accuracy_winner": accuracy_winner.to_dict(),
        "macro_winner": macro_winner.to_dict(),
        "previous_best": None if previous_best is None else previous_best.to_dict(),
        "artifacts": {key: str(path) for key, path in ARTIFACTS.items()},
    }
    ARTIFACTS["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Top boosting results")
    print(
        results[
            [
                "Model",
                "Family",
                "Imbalance Strategy",
                "CV Accuracy Mean",
                "CV Macro F1 Mean",
                "Holdout Accuracy",
                "Holdout Macro F1",
            ]
        ]
        .head(15)
        .round(4)
        .to_string(index=False)
    )
    print("\nBest boosting model by CV accuracy:")
    print(
        f"{accuracy_winner['Model']} / {accuracy_winner['Imbalance Strategy']} "
        f"CV={accuracy_winner['CV Accuracy Mean']:.4f}, "
        f"macroF1={accuracy_winner['CV Macro F1 Mean']:.4f}, "
        f"holdout={accuracy_winner['Holdout Accuracy']:.4f}"
    )
    print("\nBest boosting model by CV macro F1:")
    print(
        f"{macro_winner['Model']} / {macro_winner['Imbalance Strategy']} "
        f"CV={macro_winner['CV Accuracy Mean']:.4f}, "
        f"macroF1={macro_winner['CV Macro F1 Mean']:.4f}, "
        f"holdout_macroF1={macro_winner['Holdout Macro F1']:.4f}"
    )


if __name__ == "__main__":
    main()
