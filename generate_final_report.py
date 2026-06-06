from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_MD = Path("final_project_report.md")
REPORT_PDF = Path("final_project_report.pdf")


FIGURES = {
    "distribution": Path("fig_distribution.png"),
    "pca": Path("fig_pca.png"),
    "tsne": Path("fig_tsne.png"),
    "umap": Path("fig_umap.png"),
    "correlation": Path("fig_correlation.png"),
    "confusion": Path("fig_confusion_matrix.png"),
    "per_class": Path("fig_per_class.png"),
    "class_balance_enhanced": Path("fig_class_balance_enhanced.png"),
    "embedding_panel_enhanced": Path("fig_embedding_panel_enhanced.png"),
    "final_confusion_enhanced": Path("fig_final_model_confusion_enhanced.png"),
    "model_comparison_enhanced": Path("fig_model_comparison_enhanced.png"),
    "feature_importance_enhanced": Path("fig_feature_importance_enhanced.png"),
    "feature_boxplots_enhanced": Path("fig_feature_boxplots_enhanced.png"),
}


def fmt(x: float) -> str:
    return f"{x:.4f}"


def read_tables() -> dict[str, pd.DataFrame]:
    features = pd.read_csv("features.csv")
    advanced = pd.read_csv("advanced_modeling_results.csv")
    if Path("model_results.csv").exists():
        model_results = pd.read_csv("model_results.csv")
    else:
        model_results = pd.DataFrame(
            {
                "Model": [
                    "Logistic Regression",
                    "KNN (k=5)",
                    "SVM (RBF)",
                    "Random Forest",
                    "KMeans",
                    "DBSCAN",
                    "LDA",
                ],
                "Score": [
                    "0.7094+/-0.0504",
                    "0.7220+/-0.0449",
                    "0.7178+/-0.0415",
                    "0.7253+/-0.0654",
                    "Sil=0.1876|ARI=0.0808",
                    "Sil=0.6111|ARI=0.0814",
                    "0.7420+/-0.0391",
                ],
            }
        )
    if Path("final_results.csv").exists():
        final_results = pd.read_csv("final_results.csv")
    else:
        final_results = pd.DataFrame(
            {
                "Model": [
                    "Logistic Regression (balanced)",
                    "KNN (best)",
                    "SVM (best)",
                    "Random Forest (best)",
                    "Gradient Boosting",
                    "LDA",
                ],
                "CV Mean": [0.640653, 0.738204, 0.762286, 0.705306, 0.684980, 0.741959],
                "CV Std": [0.072506, 0.052419, 0.043599, 0.059384, 0.068274, 0.039077],
                "Test Acc": [0.68, 0.60, 0.70, 0.70, 0.64, 0.70],
            }
        )
    tables = {
        "features": features,
        "model_results": model_results,
        "final_results": final_results,
        "advanced": advanced,
    }
    if Path("boosting_model_results.csv").exists():
        tables["boosting"] = pd.read_csv("boosting_model_results.csv")
    return tables


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None, n: int | None = None) -> str:
    view = df.copy()
    if columns:
        view = view[columns]
    if n:
        view = view.head(n)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(fmt)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    rows = ["| " + " | ".join(cell(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, sep, *rows])


def build_markdown(tables: dict[str, pd.DataFrame]) -> str:
    features = tables["features"]
    model_results = tables["model_results"]
    final_results = tables["final_results"].copy()
    advanced = tables["advanced"].copy()
    boosting = tables.get("boosting")

    raw_counts = features["bug type"].value_counts().rename_axis("bug type").reset_index(name="count")
    filtered_counts = (
        features[~features["bug type"].isin(["Dragonfly", "Bee & Bumblebee"])]["bug type"]
        .value_counts()
        .rename_axis("bug type")
        .reset_index(name="count")
    )
    species_counts = features["species"].value_counts().head(10).rename_axis("species").reset_index(name="count")

    final_results["CV Mean"] = final_results["CV Mean"].astype(float)
    final_results["CV Std"] = final_results["CV Std"].astype(float)
    final_results["Test Acc"] = final_results["Test Acc"].astype(float)

    top_advanced = advanced[
        [
            "Model",
            "Imbalance Strategy",
            "CV Accuracy Mean",
            "CV Macro F1 Mean",
            "Holdout Accuracy",
            "Holdout Macro F1",
        ]
    ].head(10)

    best = advanced.iloc[0]
    balanced = advanced.sort_values(["CV Macro F1 Mean", "CV Accuracy Mean"], ascending=False).iloc[0]
    boosting_section = ""
    if boosting is not None:
        boosting = boosting.copy()
        top_boosting = boosting[
            [
                "Model",
                "Family",
                "Imbalance Strategy",
                "CV Accuracy Mean",
                "CV Macro F1 Mean",
                "Holdout Accuracy",
            ]
        ].head(8)
        boost_best = boosting.iloc[0]
        boost_macro = boosting.sort_values(["CV Macro F1 Mean", "CV Accuracy Mean"], ascending=False).iloc[0]
        boosting_section = f"""

        ## XGBoost, LightGBM, and CatBoost

        Additional boosted-tree libraries were tested on the same 80 engineered features. The sweep
        included XGBoost, LightGBM, and CatBoost with no resampling, balanced sample weights,
        equal oversampling, and hybrid_50 resampling.

        {markdown_table(top_boosting)}

        The best boosted-tree result by CV accuracy was `{boost_best["Model"]}` with
        `{boost_best["Imbalance Strategy"]}`, reaching {fmt(boost_best["CV Accuracy Mean"])} CV accuracy.
        This did not exceed the soft voting ensemble at {fmt(best["CV Accuracy Mean"])}. The best
        boosted-tree macro-F1 model was `{boost_macro["Model"]}`, but its holdout macro F1 was only
        {fmt(boost_macro["Holdout Macro F1"])}, so it was not selected as the final model.
        """

    figure_keys = [
        "class_balance_enhanced",
        "embedding_panel_enhanced",
        "correlation",
        "final_confusion_enhanced",
        "model_comparison_enhanced",
        "feature_importance_enhanced",
        "feature_boxplots_enhanced",
    ]
    figure_md = "\n".join(
        f"![{name}]({FIGURES[name].as_posix()})" for name in figure_keys if FIGURES[name].exists()
    )

    report = dedent(
        f"""
        # To Bee or Not To Bee - Machine Learning Report

        Course: IG.2412  
        Date: 2026-06-06  
        Task: classify pollinator insects from masked high-resolution images.

        ## Executive Summary

        This project extracts shape, color, texture, and symmetry features from pollinator insect
        images and trains machine learning models to predict the `bug type` label. The original
        feature set contains 39 numeric features. A corrected validation workflow and 41 engineered
        features were added, producing an 80-feature table.

        The best accuracy-oriented model is a soft voting ensemble combining SVM, shrinkage LDA,
        and ExtraTrees. It achieved a 5-fold cross-validation accuracy of {fmt(best["CV Accuracy Mean"])}
        +/- {fmt(best["CV Accuracy Std"])} and a holdout accuracy of {fmt(best["Holdout Accuracy"])}.
        A second balanced-class SVM model was also saved for cases where minority-class recall is more
        important than plain accuracy.

        ## Dataset

        The available training set contains 250 labeled samples. Images 1-250 have segmentation
        masks and labels; images 251-347 are the hidden test set described in the project brief.
        The current workspace does not contain the hidden `test/` images, so this report focuses on
        training/validation and model selection.

        Raw class distribution:

        {markdown_table(raw_counts)}

        For cross-validation, `Dragonfly` and `Bee & Bumblebee` were removed because each has only
        one sample, which is not enough for stratified folds. The evaluated five-class setup is:

        {markdown_table(filtered_counts)}

        Top species labels:

        {markdown_table(species_counts)}

        ## Feature Extraction

        Base features were computed from each image and mask:

        - Symmetry: horizontal and vertical mask overlap plus aspect ratio.
        - Shape: Hu moments, solidity, eccentricity, extent, perimeter, area, convex area, compactness.
        - Color: RGB min/max/mean/median/std inside the bug mask.
        - Size/background: bug pixel ratio and background brightness.
        - Texture/appearance: color dominance, edge density, texture contrast, brightness mean, saturation mean.

        Feature engineering then added 41 derived features, including RGB ranges, channel ratios,
        yellow/darkness proxies, bug-background brightness differences, symmetry interactions,
        shape ratios, Hu moment aggregates, and edge-texture products.

        ## Visual Exploration

        The enhanced distribution plot shows a strong class imbalance: Bee and Bumblebee dominate
        the dataset, while Butterfly, Hover fly, and Wasp have far fewer examples. The combined
        PCA/t-SNE/UMAP panel uses the 80 engineered features and shows partial class separation,
        but Bee/Bumblebee overlap remains substantial. The final-model confusion matrix and feature
        importance plot make the selected model easier to interpret.

        {figure_md}

        ## Modeling Protocol

        The original notebook scaled all samples before cross-validation. This was corrected by
        placing `StandardScaler` inside sklearn `Pipeline` objects, so scaling is fitted only on
        training folds. Evaluation used stratified 5-fold cross-validation and a fixed 80/20 stratified
        holdout split for additional sanity checks.

        Metrics:

        - Accuracy: main likely leaderboard metric.
        - Balanced accuracy and macro F1: useful because minority classes are small.
        - Per-class precision/recall/F1: used to inspect failure modes.

        ## Initial Models and Clustering

        Initial notebook results:

        {markdown_table(model_results)}

        KMeans and DBSCAN were included as unsupervised clustering baselines. Their adjusted Rand
        index scores are low, showing that unsupervised clusters do not naturally recover the target
        classes from the hand-crafted feature space.

        ## Tuned Supervised Models

        The first tuned supervised comparison used Logistic Regression, KNN, SVM, Random Forest,
        Gradient Boosting, and LDA:

        {markdown_table(final_results, ["Model", "CV Mean", "CV Std", "Test Acc"])}

        SVM had the strongest cross-validation accuracy among these classic models, while SVM,
        Random Forest, and LDA tied on the holdout split.

        ## Corrected Validation and Engineered Features

        After correcting the scaler placement, the original SVM score was lower than in the notebook,
        indicating mild data leakage in the earlier validation workflow. Adding engineered features
        recovered and improved performance:

        - Corrected original-feature SVM CV accuracy: 0.7504
        - Engineered-feature SVM CV accuracy: 0.7702
        - Engineered-feature voting ensemble CV accuracy: {fmt(best["CV Accuracy Mean"])}

        ## Class Imbalance Experiments

        The dataset is imbalanced, so the advanced sweep tested:

        - no resampling
        - equal oversampling of minority classes
        - `hybrid_50`, resampling each class toward roughly half the majority size
        - mild undersampling of dominant classes

        Top advanced modeling results:

        {markdown_table(top_advanced)}

        {boosting_section}

        The accuracy winner used no resampling. The best macro-F1-oriented model was
        `{balanced["Model"]}` with `{balanced["Imbalance Strategy"]}`. This trade-off matters:
        resampling improves minority-class attention but can reduce plain accuracy.

        ## Final Model Selection

        Accuracy-oriented final model:

        - File: `best_model_engineered.pkl`
        - Model: {best["Model"]}
        - Imbalance strategy: {best["Imbalance Strategy"]}
        - CV accuracy: {fmt(best["CV Accuracy Mean"])} +/- {fmt(best["CV Accuracy Std"])}
        - CV macro F1: {fmt(best["CV Macro F1 Mean"])}
        - Holdout accuracy: {fmt(best["Holdout Accuracy"])}

        Balanced-class alternative:

        - File: `best_model_engineered_balanced.pkl`
        - Model: {balanced["Model"]}
        - Imbalance strategy: {balanced["Imbalance Strategy"]}
        - CV accuracy: {fmt(balanced["CV Accuracy Mean"])} +/- {fmt(balanced["CV Accuracy Std"])}
        - CV macro F1: {fmt(balanced["CV Macro F1 Mean"])}
        - Holdout macro F1: {fmt(balanced["Holdout Macro F1"])}

        ## Prediction Pipeline

        To predict images 251-347 when the test images become available:

        1. Extract the same 39 base features from each test image and mask.
        2. Apply `add_engineered_features` to create the 80-feature representation.
        3. Select the saved `feature_cols` from `best_model_engineered.pkl`.
        4. Call the saved voting ensemble model and write a CSV with columns `ID` and `bug type`.

        ## Limitations

        The rare one-sample classes were excluded from cross-validation and the saved final models
        predict five classes. If the hidden test set contains `Dragonfly` or `Bee & Bumblebee`,
        a separate fallback strategy would be needed. The small number of Hover fly and Wasp samples
        also makes their validation scores volatile.

        ## Conclusion

        The final recommended system uses engineered hand-crafted features and soft voting model
        fusion. Compared with the corrected single SVM baseline, the ensemble improves cross-validation
        accuracy from 0.7504 to {fmt(best["CV Accuracy Mean"])} and improves holdout accuracy to
        {fmt(best["Holdout Accuracy"])}. This is the strongest current candidate for the hidden test
        predictions, assuming the final score is based primarily on plain classification accuracy.
        """
    ).strip() + "\n"
    return "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines()) + "\n"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "")
    )
    return Paragraph(safe, style)


def pdf_table(df: pd.DataFrame, columns: list[str] | None, style: ParagraphStyle, max_rows: int | None = None) -> Table:
    view = df.copy()
    if columns:
        view = view[columns]
    if max_rows:
        view = view.head(max_rows)
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(fmt)
    data = [[para(str(col), style) for col in view.columns]]
    data.extend([[para(str(value), style) for value in row] for row in view.itertuples(index=False, name=None)])
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d2dc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def add_figure(story: list, path: Path, caption: str, styles: dict[str, ParagraphStyle]) -> None:
    if not path.exists():
        return
    story.append(KeepTogether([scaled_image(path, 6.8 * inch, 4.5 * inch), para(caption, styles["caption"])]))
    story.append(Spacer(1, 10))


def build_pdf(tables: dict[str, pd.DataFrame]) -> None:
    features = tables["features"]
    model_results = tables["model_results"]
    final_results = tables["final_results"].copy()
    advanced = tables["advanced"].copy()
    boosting = tables.get("boosting")
    best = advanced.iloc[0]
    balanced = advanced.sort_values(["CV Macro F1 Mean", "CV Accuracy Mean"], ascending=False).iloc[0]
    top_boosting = None
    boost_best = None
    boost_macro = None
    if boosting is not None:
        boosting = boosting.copy()
        top_boosting = boosting[
            [
                "Model",
                "Family",
                "Imbalance Strategy",
                "CV Accuracy Mean",
                "CV Macro F1 Mean",
                "Holdout Accuracy",
            ]
        ].head(8)
        boost_best = boosting.iloc[0]
        boost_macro = boosting.sort_values(["CV Macro F1 Mean", "CV Accuracy Mean"], ascending=False).iloc[0]

    raw_counts = features["bug type"].value_counts().rename_axis("bug type").reset_index(name="count")
    filtered_counts = (
        features[~features["bug type"].isin(["Dragonfly", "Bee & Bumblebee"])]["bug type"]
        .value_counts()
        .rename_axis("bug type")
        .reset_index(name="count")
    )
    species_counts = features["species"].value_counts().head(10).rename_axis("species").reset_index(name="count")
    top_advanced = advanced[
        [
            "Model",
            "Imbalance Strategy",
            "CV Accuracy Mean",
            "CV Macro F1 Mean",
            "Holdout Accuracy",
            "Holdout Macro F1",
        ]
    ].head(10)

    final_results["CV Mean"] = final_results["CV Mean"].astype(float)
    final_results["CV Std"] = final_results["CV Std"].astype(float)
    final_results["Test Acc"] = final_results["Test Acc"].astype(float)

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=34,
        bottomMargin=34,
    )
    base_styles = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("Title", parent=base_styles["Title"], fontSize=22, leading=26, alignment=TA_CENTER),
        "h1": ParagraphStyle("Heading1", parent=base_styles["Heading1"], fontSize=15, leading=18, spaceBefore=12),
        "h2": ParagraphStyle("Heading2", parent=base_styles["Heading2"], fontSize=12, leading=15, spaceBefore=8),
        "body": ParagraphStyle("Body", parent=base_styles["BodyText"], fontSize=9.4, leading=12),
        "small": ParagraphStyle("Small", parent=base_styles["BodyText"], fontSize=7.5, leading=9),
        "caption": ParagraphStyle(
            "Caption",
            parent=base_styles["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4f5b66"),
        ),
    }

    story: list = []
    story.append(para("To Bee or Not To Bee", styles["title"]))
    story.append(para("Machine Learning Report - IG.2412", styles["h2"]))
    story.append(para("Date: 2026-06-06", styles["body"]))
    story.append(Spacer(1, 8))
    story.append(
        para(
            "This report summarizes the feature extraction, visualization, modeling, class imbalance experiments, "
            "and final model selection for pollinator insect classification.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 12))

    story.append(para("Executive Summary", styles["h1"]))
    story.append(
        para(
            f"The best accuracy-oriented model is a soft voting ensemble combining SVM, shrinkage LDA, "
            f"and ExtraTrees. It achieved 5-fold CV accuracy {fmt(best['CV Accuracy Mean'])} +/- "
            f"{fmt(best['CV Accuracy Std'])} and holdout accuracy {fmt(best['Holdout Accuracy'])}. "
            f"A balanced-class alternative was also saved for minority-class-sensitive scoring.",
            styles["body"],
        )
    )

    story.append(para("Dataset", styles["h1"]))
    story.append(
        para(
            "The training set contains 250 labeled samples with masks. Images 251-347 are the hidden test set "
            "described in the project brief, but the current workspace does not contain those test images.",
            styles["body"],
        )
    )
    story.append(para("Raw class distribution", styles["h2"]))
    story.append(pdf_table(raw_counts, None, styles["small"]))
    story.append(Spacer(1, 8))
    story.append(
        para(
            "Dragonfly and Bee & Bumblebee each have one sample. They were excluded from stratified validation, "
            "leaving the five evaluated classes below.",
            styles["body"],
        )
    )
    story.append(pdf_table(filtered_counts, None, styles["small"]))
    story.append(Spacer(1, 8))
    story.append(para("Top species labels", styles["h2"]))
    story.append(pdf_table(species_counts, None, styles["small"]))

    story.append(PageBreak())
    story.append(para("Feature Extraction", styles["h1"]))
    feature_texts = [
        "Symmetry: horizontal and vertical mask overlap plus mask aspect ratio.",
        "Shape: Hu moments, solidity, eccentricity, extent, perimeter, area, convex area, and compactness.",
        "Color: RGB min, max, mean, median, and standard deviation inside the insect mask.",
        "Size/background: bug pixel ratio and background brightness.",
        "Texture/appearance: color dominance, edge density, texture contrast, brightness mean, and saturation mean.",
        "Engineered additions: RGB ranges, channel ratios, yellow/darkness proxies, bug-background brightness differences, symmetry interactions, shape ratios, Hu aggregates, and edge-texture products.",
    ]
    for text in feature_texts:
        story.append(para(text, styles["body"]))
    story.append(Spacer(1, 10))
    add_figure(
        story,
        FIGURES["class_balance_enhanced"] if FIGURES["class_balance_enhanced"].exists() else FIGURES["distribution"],
        "Figure 1. Annotated bug type imbalance and top species labels.",
        styles,
    )

    story.append(para("Visual Exploration", styles["h1"]))
    story.append(
        para(
            "PCA, t-SNE, and UMAP show partial separation, but the dominant Bee and Bumblebee classes overlap. "
            "This overlap explains why model fusion and richer color/shape interactions help more than a single model.",
            styles["body"],
        )
    )
    add_figure(
        story,
        FIGURES["embedding_panel_enhanced"] if FIGURES["embedding_panel_enhanced"].exists() else FIGURES["pca"],
        "Figure 2. PCA, t-SNE, and UMAP comparison on engineered features.",
        styles,
    )
    add_figure(story, FIGURES["correlation"], "Figure 3. Feature correlation heatmap.", styles)

    story.append(PageBreak())
    story.append(para("Modeling Protocol", styles["h1"]))
    story.append(
        para(
            "The corrected workflow places StandardScaler inside each sklearn Pipeline, preventing test folds "
            "from influencing scaling parameters. Evaluation uses stratified 5-fold cross-validation and a fixed "
            "80/20 holdout split.",
            styles["body"],
        )
    )
    story.append(para("Initial notebook models and clustering", styles["h2"]))
    story.append(pdf_table(model_results, None, styles["small"]))
    story.append(Spacer(1, 10))
    story.append(
        para(
            "KMeans and DBSCAN produced low adjusted Rand index values, so unsupervised clusters do not directly "
            "recover the target classes in the hand-crafted feature space.",
            styles["body"],
        )
    )
    story.append(para("Tuned supervised comparison", styles["h2"]))
    story.append(pdf_table(final_results, ["Model", "CV Mean", "CV Std", "Test Acc"], styles["small"]))
    story.append(Spacer(1, 10))
    add_figure(story, FIGURES["confusion"], "Figure 4. Cross-validated confusion matrix for the tuned SVM baseline.", styles)
    add_figure(story, FIGURES["per_class"], "Figure 5. Per-class performance for the tuned baseline.", styles)

    story.append(para("Corrected Validation and Feature Engineering", styles["h1"]))
    story.append(
        para(
            "After fixing scaler placement, the original-feature SVM CV accuracy was 0.7504. With engineered "
            "features, SVM reached 0.7702. The final voting ensemble reached "
            f"{fmt(best['CV Accuracy Mean'])}, showing that feature engineering and model fusion both helped.",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(para("Class Imbalance and Advanced Models", styles["h1"]))
    story.append(
        para(
            "The sweep tested no resampling, equal oversampling, hybrid_50 resampling, and mild undersampling. "
            "It also compared SVM variants, Logistic Regression, LDA/QDA, KNN, Naive Bayes, Random Forest, "
            "ExtraTrees, Gradient Boosting, HistGradientBoosting, AdaBoost, Voting, and Stacking.",
            styles["body"],
        )
    )
    story.append(pdf_table(top_advanced, None, styles["small"], max_rows=10))
    if top_boosting is not None and boost_best is not None and boost_macro is not None:
        story.append(Spacer(1, 10))
        story.append(para("XGBoost, LightGBM, and CatBoost", styles["h1"]))
        story.append(
            para(
                "Additional boosted-tree libraries were tested on the same 80 engineered features. "
                "The best boosted-tree model did not exceed the soft voting ensemble, so the final "
                "accuracy-oriented recommendation remains unchanged.",
                styles["body"],
            )
        )
        story.append(pdf_table(top_boosting, None, styles["small"], max_rows=8))
        story.append(
            para(
                f"Best boosted-tree CV accuracy: {boost_best['Model']} / {boost_best['Imbalance Strategy']} "
                f"at {fmt(boost_best['CV Accuracy Mean'])}. Best boosted-tree macro F1: "
                f"{boost_macro['Model']} / {boost_macro['Imbalance Strategy']} at "
                f"{fmt(boost_macro['CV Macro F1 Mean'])}, but its holdout macro F1 was "
                f"{fmt(boost_macro['Holdout Macro F1'])}.",
                styles["body"],
            )
        )
    add_figure(
        story,
        FIGURES["model_comparison_enhanced"],
        "Figure 6. Final model comparison across fusion and boosted-tree approaches.",
        styles,
    )
    add_figure(
        story,
        FIGURES["final_confusion_enhanced"],
        "Figure 7. Holdout confusion matrix for the final voting model.",
        styles,
    )
    add_figure(
        story,
        FIGURES["feature_importance_enhanced"],
        "Figure 8. Top feature importances from the ExtraTrees component.",
        styles,
    )
    add_figure(
        story,
        FIGURES["feature_boxplots_enhanced"],
        "Figure 9. Interpretable engineered features by class.",
        styles,
    )

    story.append(para("Final Model Selection", styles["h1"]))
    story.append(
        para(
            f"Accuracy-oriented model: best_model_engineered.pkl. Model = {best['Model']}; strategy = "
            f"{best['Imbalance Strategy']}; CV accuracy = {fmt(best['CV Accuracy Mean'])} +/- "
            f"{fmt(best['CV Accuracy Std'])}; CV macro F1 = {fmt(best['CV Macro F1 Mean'])}; "
            f"holdout accuracy = {fmt(best['Holdout Accuracy'])}.",
            styles["body"],
        )
    )
    story.append(
        para(
            f"Balanced-class alternative: best_model_engineered_balanced.pkl. Model = {balanced['Model']}; "
            f"strategy = {balanced['Imbalance Strategy']}; CV accuracy = {fmt(balanced['CV Accuracy Mean'])} +/- "
            f"{fmt(balanced['CV Accuracy Std'])}; CV macro F1 = {fmt(balanced['CV Macro F1 Mean'])}; "
            f"holdout macro F1 = {fmt(balanced['Holdout Macro F1'])}.",
            styles["body"],
        )
    )

    story.append(para("Prediction Pipeline", styles["h1"]))
    for text in [
        "Extract the same 39 base features from each test image and mask.",
        "Apply add_engineered_features to create the 80-feature representation.",
        "Select the saved feature_cols from the pickle payload.",
        "Call the saved model and write a CSV with columns ID and bug type.",
    ]:
        story.append(para(text, styles["body"]))

    story.append(para("Limitations and Conclusion", styles["h1"]))
    story.append(
        para(
            "The saved final models predict five classes because two original labels have only one sample each. "
            "If the hidden test set contains those rare labels, an additional fallback rule is needed. Within the "
            "evaluated five-class setup, the engineered-feature soft voting ensemble is the strongest candidate.",
            styles["body"],
        )
    )

    doc.build(story)


def main() -> None:
    tables = read_tables()
    REPORT_MD.write_text(build_markdown(tables), encoding="utf-8")
    build_pdf(tables)
    print(f"Wrote {REPORT_MD}")
    print(f"Wrote {REPORT_PDF}")


if __name__ == "__main__":
    main()
