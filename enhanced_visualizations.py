from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from advanced_modeling import RANDOM_STATE, find_spec, fit_with_strategy
from improved_modeling import DROP_COLS, TARGET_COL, add_engineered_features, load_data

try:
    import umap
except Exception:
    umap = None


OUT = {
    "class_balance": Path("fig_class_balance_enhanced.png"),
    "embedding_panel": Path("fig_embedding_panel_enhanced.png"),
    "final_confusion": Path("fig_final_model_confusion_enhanced.png"),
    "model_comparison": Path("fig_model_comparison_enhanced.png"),
    "feature_importance": Path("fig_feature_importance_enhanced.png"),
    "feature_boxplots": Path("fig_feature_boxplots_enhanced.png"),
}


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def class_balance_plot(df_raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.15]})
    bug_counts = df_raw[TARGET_COL].value_counts()
    bug_pct = bug_counts / bug_counts.sum() * 100
    palette = sns.color_palette("Set2", len(bug_counts))

    sns.barplot(x=bug_counts.index, y=bug_counts.values, ax=axes[0], palette=palette, hue=bug_counts.index, legend=False)
    axes[0].set_title("Bug Type Distribution", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Samples")
    axes[0].tick_params(axis="x", rotation=28)
    for i, (count, pct) in enumerate(zip(bug_counts.values, bug_pct.values)):
        axes[0].text(i, count + 2, f"{count}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    species_counts = df_raw["species"].value_counts().head(12).sort_values()
    sns.barplot(x=species_counts.values, y=species_counts.index, ax=axes[1], color="#4C78A8")
    axes[1].set_title("Top Species Labels", fontweight="bold")
    axes[1].set_xlabel("Samples")
    axes[1].set_ylabel("")
    for y_pos, count in enumerate(species_counts.values):
        axes[1].text(count + 0.8, y_pos, str(count), va="center", fontsize=9)

    fig.suptitle("Dataset Imbalance: Majority Classes Dominate the Training Set", fontweight="bold", y=1.03)
    savefig(OUT["class_balance"])


def embedding_panel(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].values.astype(float)
    y = df[TARGET_COL].values
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_xy = pca.fit_transform(X_scaled)
    tsne_xy = TSNE(
        n_components=2,
        random_state=RANDOM_STATE,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    ).fit_transform(X_scaled)
    if umap is not None:
        umap_xy = umap.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    else:
        umap_xy = np.full_like(pca_xy, np.nan)

    embeddings = [
        ("PCA", pca_xy, f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}"),
        ("t-SNE", tsne_xy, "Nonlinear neighborhood view"),
        ("UMAP", umap_xy, "Nonlinear manifold view"),
    ]
    classes = sorted(pd.unique(y))
    palette = dict(zip(classes, sns.color_palette("Set2", len(classes))))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (title, coords, subtitle) in zip(axes, embeddings):
        if np.isnan(coords).all():
            ax.text(0.5, 0.5, "UMAP not available", ha="center", va="center")
            ax.set_axis_off()
            continue
        for cls in classes:
            mask = y == cls
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=42,
                alpha=0.78,
                label=cls,
                color=palette[cls],
                edgecolors="white",
                linewidths=0.4,
            )
            centroid = coords[mask].mean(axis=0)
            ax.scatter(centroid[0], centroid[1], s=115, marker="X", color=palette[cls], edgecolors="black", linewidths=0.6)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.text(0.02, 0.98, subtitle, transform=ax.transAxes, va="top", fontsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(classes), frameon=False)
    fig.suptitle("Projection Comparison on 80 Engineered Features", fontweight="bold", y=1.03)
    plt.subplots_adjust(bottom=0.2)
    savefig(OUT["embedding_panel"])


def final_confusion_plot(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL]
    labels = sorted(y.unique())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    row = pd.read_csv("advanced_modeling_results.csv").iloc[0]
    spec = find_spec(str(row["Model"]))
    model = fit_with_strategy(spec.factory(), X_train, y_train, str(row["Imbalance Strategy"]), RANDOM_STATE + 1000)
    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred, labels=labels)
    cm_norm = confusion_matrix(y_test, pred, labels=labels, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=axes[0], cmap="Blues", colorbar=False, values_format="d")
    axes[0].set_title("Final Voting Model - Counts", fontweight="bold")
    ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(ax=axes[1], cmap="Greens", colorbar=False, values_format=".2f")
    axes[1].set_title("Final Voting Model - Recall by Class", fontweight="bold")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Holdout Confusion Matrix: Accuracy-Oriented Final Model", fontweight="bold", y=1.03)
    savefig(OUT["final_confusion"])


def model_comparison_plot() -> None:
    advanced = pd.read_csv("advanced_modeling_results.csv")
    boosting = pd.read_csv("boosting_model_results.csv")
    rows = []
    for _, row in advanced.head(8).iterrows():
        rows.append(
            {
                "Model": f"{row['Model']}\n({row['Imbalance Strategy']})",
                "Group": "Sklearn / Fusion",
                "CV Accuracy": row["CV Accuracy Mean"],
                "CV Macro F1": row["CV Macro F1 Mean"],
            }
        )
    for _, row in boosting.head(5).iterrows():
        rows.append(
            {
                "Model": f"{row['Model']}\n({row['Imbalance Strategy']})",
                "Group": "Boosted Trees",
                "CV Accuracy": row["CV Accuracy Mean"],
                "CV Macro F1": row["CV Macro F1 Mean"],
            }
        )
    plot_df = pd.DataFrame(rows).sort_values("CV Accuracy", ascending=True)
    colors = plot_df["Group"].map({"Sklearn / Fusion": "#4C78A8", "Boosted Trees": "#F58518"})

    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    ax.barh(plot_df["Model"], plot_df["CV Accuracy"], color=colors)
    ax.scatter(plot_df["CV Macro F1"], plot_df["Model"], color="#222222", s=42, label="CV Macro F1")
    ax.set_xlim(0.45, 0.84)
    ax.set_xlabel("Score")
    ax.set_title("Model Comparison: Accuracy vs Macro F1", fontweight="bold")
    ax.axvline(0.8024, color="#4C78A8", linestyle="--", linewidth=1, alpha=0.8)
    ax.text(0.804, len(plot_df) - 0.7, "Voting best", fontsize=9, color="#4C78A8")
    ax.legend(loc="lower right")
    savefig(OUT["model_comparison"])


def feature_importance_plot() -> None:
    with open("best_model_engineered.pkl", "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    feature_cols = payload["feature_cols"]
    et = model.named_estimators_.get("et")
    if et is None or not hasattr(et, "feature_importances_"):
        return
    importances = pd.Series(et.feature_importances_, index=feature_cols).sort_values(ascending=False).head(22).sort_values()

    fig, ax = plt.subplots(figsize=(9.5, 7))
    sns.barplot(x=importances.values, y=importances.index, ax=ax, color="#59A14F")
    ax.set_title("Top Feature Importances from ExtraTrees Component", fontweight="bold")
    ax.set_xlabel("Gini importance")
    ax.set_ylabel("")
    savefig(OUT["feature_importance"])


def feature_boxplots(df: pd.DataFrame) -> None:
    candidate_features = [
        "yellow_score",
        "bug_bg_brightness_absdiff",
        "sym_aspect",
        "edge_texture_product",
        "pixel_ratio",
        "texture_contrast",
    ]
    plot_features = [c for c in candidate_features if c in df.columns]
    long_df = df[[TARGET_COL, *plot_features]].melt(id_vars=TARGET_COL, var_name="Feature", value_name="Value")

    g = sns.catplot(
        data=long_df,
        x=TARGET_COL,
        y="Value",
        col="Feature",
        col_wrap=3,
        kind="box",
        sharey=False,
        height=3.3,
        aspect=1.15,
        color="#9ecae1",
        fliersize=2,
    )
    g.set_xticklabels(rotation=35)
    g.set_axis_labels("", "Value")
    g.fig.suptitle("Class Separation Signals in Engineered Features", fontweight="bold", y=1.03)
    g.savefig(OUT["feature_boxplots"], dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(g.fig)


def main() -> None:
    setup_style()
    raw = pd.read_csv("features.csv")
    df, _ = add_engineered_features(load_data())

    class_balance_plot(raw)
    embedding_panel(df)
    final_confusion_plot(df)
    model_comparison_plot()
    feature_importance_plot()
    feature_boxplots(df)

    print("Enhanced visualization files:")
    for path in OUT.values():
        if path.exists():
            print(f"  {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
