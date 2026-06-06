from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
TARGET_COL = "bug type"
DROP_COLS = ["ID", TARGET_COL, "species"]
RARE_CLASSES = ["Bee & Bumblebee", "Dragonfly"]


def ratio(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b + eps)


def load_data(path: Path = Path("features.csv")) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[~df[TARGET_COL].isin(RARE_CLASSES)].copy()


def add_engineered_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added: list[str] = []

    def add(name: str, value: pd.Series | np.ndarray) -> None:
        out[name] = value
        added.append(name)

    rgb_means = out[["r_mean", "g_mean", "b_mean"]]
    rgb_medians = out[["r_median", "g_median", "b_median"]]
    rgb_stds = out[["r_std", "g_std", "b_std"]]
    rgb_maxs = out[["r_max", "g_max", "b_max"]]
    rgb_mins = out[["r_min", "g_min", "b_min"]]

    add("r_range", out["r_max"] - out["r_min"])
    add("g_range", out["g_max"] - out["g_min"])
    add("b_range", out["b_max"] - out["b_min"])
    add("mean_channel_spread", rgb_means.max(axis=1) - rgb_means.min(axis=1))
    add("median_channel_spread", rgb_medians.max(axis=1) - rgb_medians.min(axis=1))
    add("range_channel_spread", rgb_maxs.max(axis=1) - rgb_mins.min(axis=1))
    add("mean_color_std", rgb_stds.mean(axis=1))
    add("max_color_std", rgb_stds.max(axis=1))

    add("rg_mean_diff", out["r_mean"] - out["g_mean"])
    add("rb_mean_diff", out["r_mean"] - out["b_mean"])
    add("gb_mean_diff", out["g_mean"] - out["b_mean"])
    add("rg_mean_ratio", ratio(out["r_mean"], out["g_mean"]))
    add("rb_mean_ratio", ratio(out["r_mean"], out["b_mean"]))
    add("gb_mean_ratio", ratio(out["g_mean"], out["b_mean"]))
    add("red_cv", ratio(out["r_std"], out["r_mean"]))
    add("green_cv", ratio(out["g_std"], out["g_mean"]))
    add("blue_cv", ratio(out["b_std"], out["b_mean"]))

    # Yellow/black contrast proxies are useful for bee, bumblebee, and wasp classes.
    add("yellow_score", (out["r_mean"] + out["g_mean"]) / 2.0 - out["b_mean"])
    add("warmth_score", out["r_mean"] - out["b_mean"])
    add("darkness_score", 255.0 - out["brightness_mean"])
    add("saturation_brightness_ratio", ratio(out["saturation_mean"], out["brightness_mean"]))
    add("bug_bg_brightness_diff", out["brightness_mean"] - out["bg_brightness"])
    add("bug_bg_brightness_absdiff", (out["brightness_mean"] - out["bg_brightness"]).abs())

    add("symmetry_mean", (out["sym_h"] + out["sym_v"]) / 2.0)
    add("symmetry_product", out["sym_h"] * out["sym_v"])
    add("symmetry_absdiff", (out["sym_h"] - out["sym_v"]).abs())
    add("aspect_log", np.log1p(out["sym_aspect"].clip(lower=0)))
    add("area_sqrt", np.sqrt(out["area"].clip(lower=0)))
    add("perimeter_area_ratio", ratio(out["perimeter"], np.sqrt(out["area"].clip(lower=0))))
    add("convex_area_ratio", ratio(out["convex_area"], out["area"]))
    add("solidity_extent_product", out["solidity"] * out["extent"])
    add("eccentricity_aspect_product", out["eccentricity"] * out["sym_aspect"])

    hu_cols = [f"hu{i}" for i in range(1, 8)]
    hu_abs = out[hu_cols].abs()
    add("hu_abs_sum", hu_abs.sum(axis=1))
    add("hu_abs_mean", hu_abs.mean(axis=1))
    add("hu_abs_std", hu_abs.std(axis=1))
    add("hu_energy", (out[hu_cols] ** 2).sum(axis=1))

    add("edge_texture_product", out["edge_density"] * out["texture_contrast"])
    add("edge_pixel_product", out["edge_density"] * out["pixel_ratio"])
    add("texture_brightness_ratio", ratio(out["texture_contrast"], out["brightness_mean"]))
    add("texture_saturation_ratio", ratio(out["texture_contrast"], out["saturation_mean"]))
    add("color_texture_product", out["color_dominance"] * out["texture_contrast"])

    numeric_cols = [c for c in out.columns if c not in DROP_COLS]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out, added


def make_models(n_features: int) -> dict[str, object]:
    k_best = min(35, n_features)
    selector = SelectKBest(score_func=f_classif, k=k_best)

    return {
        "Logistic Regression balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
            ]
        ),
        "KNN k=5 distance": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5, weights="distance")),
            ]
        ),
        "SVM RBF C=10": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)),
            ]
        ),
        "SVM RBF C=10 balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=10,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM RBF C=10 SelectKBest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("variance", VarianceThreshold()),
                ("select", selector),
                ("model", SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest notebook": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=2,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees balanced": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "LDA": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis()),
            ]
        ),
        "LDA shrinkage": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        ),
    }


def evaluate_feature_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
    }

    rows: list[dict[str, float | str | int]] = []
    for name, model in make_models(len(feature_cols)).items():
        cv_result = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rows.append(
            {
                "Feature Set": feature_set,
                "Feature Count": len(feature_cols),
                "Model": name,
                "CV Accuracy Mean": cv_result["test_accuracy"].mean(),
                "CV Accuracy Std": cv_result["test_accuracy"].std(),
                "CV Balanced Acc Mean": cv_result["test_balanced_accuracy"].mean(),
                "CV Macro F1 Mean": cv_result["test_macro_f1"].mean(),
                "CV Weighted F1 Mean": cv_result["test_weighted_f1"].mean(),
                "Holdout Accuracy": accuracy_score(y_test, y_pred),
                "Holdout Balanced Acc": balanced_accuracy_score(y_test, y_pred),
                "Holdout Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "Holdout Weighted F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["CV Accuracy Mean", "CV Macro F1 Mean"], ascending=False
    )


def print_top(title: str, results: pd.DataFrame, n: int = 6) -> None:
    cols = [
        "Model",
        "Feature Count",
        "CV Accuracy Mean",
        "CV Accuracy Std",
        "CV Macro F1 Mean",
        "Holdout Accuracy",
        "Holdout Macro F1",
    ]
    print(f"\n{title}")
    print("=" * len(title))
    print(results[cols].head(n).round(4).to_string(index=False))


def main() -> None:
    base_df = load_data()
    engineered_df, added = add_engineered_features(base_df)

    base_results = evaluate_feature_set(base_df, "base_pipeline")
    engineered_results = evaluate_feature_set(engineered_df, "engineered_pipeline")
    all_results = pd.concat([base_results, engineered_results], ignore_index=True)
    all_results = all_results.sort_values(["CV Accuracy Mean", "CV Macro F1 Mean"], ascending=False)

    base_results.to_csv("pipeline_validation_results.csv", index=False)
    engineered_results.to_csv("feature_engineering_results.csv", index=False)
    all_results.to_csv("improved_modeling_results.csv", index=False)
    engineered_df.to_csv("features_engineered.csv", index=False)

    Path("engineered_feature_columns.txt").write_text("\n".join(added) + "\n", encoding="utf-8")

    print(f"Samples after rare-class filtering: {len(base_df)}")
    print(f"Base feature count: {len([c for c in base_df.columns if c not in DROP_COLS])}")
    print(f"Engineered feature count: {len([c for c in engineered_df.columns if c not in DROP_COLS])}")
    print(f"Added engineered features: {len(added)}")

    print_top("Corrected Pipeline Results - Base Features", base_results)
    print_top("Feature Engineering Results - Engineered Features", engineered_results)
    print_top("Overall Top Results", all_results, n=10)

    best = all_results.iloc[0]
    print("\nBest by CV accuracy")
    print(
        f"{best['Model']} on {best['Feature Set']}: "
        f"CV accuracy={best['CV Accuracy Mean']:.4f}+/-{best['CV Accuracy Std']:.4f}, "
        f"CV macro F1={best['CV Macro F1 Mean']:.4f}, "
        f"holdout accuracy={best['Holdout Accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
