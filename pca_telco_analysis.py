"""Run a reproducible PCA analysis of a Telco Customer Churn data file.

The script reads an Excel or CSV export, preprocesses numeric and categorical
columns, creates PCA diagnostic plots, and compares logistic regression models
with all encoded features and with the first three principal components.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# A non-interactive backend makes the script work on servers and in notebooks
# without a display while still writing PNG files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


DEFAULT_FILENAMES = (
    "CST-570-RS-WAFn-UseC-Telco-Customer-Churn.xlsx",
    "Telco-Customer-Churn.xlsx",
    "Telco-Customer-Churn.csv",
)


def find_input_file(requested: str | None) -> Path:
    """Return the requested file or find a conventional filename."""
    if requested:
        path = Path(requested).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"Data file not found: {path}")

    for filename in DEFAULT_FILENAMES:
        path = Path(filename)
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No data file found. Pass the Excel or CSV path with --data."
    )


def read_data(path: Path) -> pd.DataFrame:
    """Read Excel or CSV input and normalize column names."""
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("The input file must have an Excel or CSV extension.")

    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ValueError("The input file does not contain any rows.")
    return frame


def locate_column(frame: pd.DataFrame, name: str) -> str:
    """Find a column without depending on capitalization or whitespace."""
    normalized = {column.casefold().replace(" ", ""): column for column in frame}
    key = name.casefold().replace(" ", "")
    if key not in normalized:
        raise ValueError(f"Required column '{name}' was not found.")
    return normalized[key]


def prepare_features(
    frame: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Encode predictors and return encoded data, labels, and label text."""
    data = frame.copy()
    target = data.pop(target_column).astype("string").str.strip()

    # CustomerID is an identifier, not a behavior measure.  Other columns
    # with one value per row are also excluded to avoid meaningless predictors.
    identifier_columns = [
        column
        for column in data.columns
        if column.casefold().replace(" ", "") in {"customerid", "customer_id"}
        or data[column].nunique(dropna=False) == len(data)
    ]
    data = data.drop(columns=identifier_columns)

    # TotalCharges is commonly stored as text because blank charges occur for
    # new customers; converting it lets it be treated as a numeric measure.
    for column in data.columns:
        if column.casefold().replace(" ", "") == "totalcharges":
            data[column] = pd.to_numeric(data[column], errors="coerce")

    numeric = data.select_dtypes(include=["number", "bool"]).copy()
    categorical = data.drop(columns=numeric.columns, errors="ignore").copy()

    if not numeric.empty:
        numeric = numeric.fillna(numeric.median(numeric_only=True))
    if not categorical.empty:
        categorical = categorical.fillna("Missing").astype(str)
        categorical = pd.get_dummies(categorical, drop_first=False, dtype=float)

    features = pd.concat([numeric, categorical], axis=1).astype(float)
    if features.empty:
        raise ValueError("No usable predictor columns remain after preprocessing.")

    # PCA and the model need a numeric churn label.  Retain the original text
    # labels for plot legends and support Yes/No as well as 1/0 inputs.
    lowered = target.str.casefold()
    if set(lowered.dropna().unique()) <= {"yes", "no"}:
        labels = lowered.map({"no": 0, "yes": 1})
    elif set(lowered.dropna().unique()) <= {"true", "false"}:
        labels = lowered.map({"false": 0, "true": 1})
    else:
        labels = pd.to_numeric(target, errors="coerce")
    if labels.isna().any() or labels.nunique() != 2:
        raise ValueError(
            "The churn column must contain exactly two non-missing classes "
            "(for example, Yes/No)."
        )
    return features, labels.astype(int), target


def save_variance_plot(pca: PCA, output_dir: Path) -> None:
    """Plot variance explained by each component and cumulatively."""
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(components, pca.explained_variance_ratio_, alpha=0.7, label="Individual")
    axis.plot(components, cumulative, marker="o", color="darkred", label="Cumulative")
    axis.axhline(0.90, color="gray", linestyle="--", label="90% target")
    axis.set(
        title="PCA explained variance",
        xlabel="Principal component",
        ylabel="Variance ratio",
    )
    axis.set_ylim(0, 1.05)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "explained_variance.png", dpi=160)
    plt.close(figure)


def save_projection_plot(
    scores: np.ndarray, label_text: pd.Series, output_dir: Path
) -> None:
    """Plot customers in the first two PCA dimensions, colored by churn."""
    figure, axis = plt.subplots(figsize=(8, 6))
    unique_labels = list(pd.unique(label_text))
    colors = {label: color for label, color in zip(unique_labels, ("#2b6cb0", "#c53030"))}
    for label in unique_labels:
        selected = label_text == label
        axis.scatter(
            scores[selected, 0],
            scores[selected, 1],
            s=18,
            alpha=0.55,
            color=colors[label],
            label=f"{label} (n={selected.sum()})",
        )
    axis.set(
        title="Customers projected onto the first two principal components",
        xlabel="PC1",
        ylabel="PC2",
    )
    axis.legend(title="Churn")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "pca_projection.png", dpi=160)
    plt.close(figure)


def save_loading_plot(pca: PCA, feature_names: list[str], output_dir: Path) -> None:
    """Plot the ten largest absolute PC1 loadings."""
    loadings = pd.Series(pca.components_[0], index=feature_names)
    selected = loadings.reindex(loadings.abs().sort_values(ascending=False).head(10).index)
    figure, axis = plt.subplots(figsize=(9, 6))
    selected.sort_values().plot.barh(ax=axis, color=np.where(selected.sort_values() >= 0, "#2b6cb0", "#c53030"))
    axis.set(
        title="Largest absolute loadings for PC1",
        xlabel="Loading (signed contribution)",
        ylabel="Encoded feature",
    )
    axis.axvline(0, color="black", linewidth=0.8)
    figure.tight_layout()
    figure.savefig(output_dir / "pc1_loadings.png", dpi=160)
    plt.close(figure)


def evaluate_models(
    features: pd.DataFrame, labels: pd.Series, n_components: int
) -> pd.DataFrame:
    """Compare all standardized features with a compact PCA representation."""
    train_indices, test_indices = train_test_split(
        np.arange(len(features)),
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(features.iloc[train_indices])
    test_scaled = scaler.transform(features.iloc[test_indices])
    pca = PCA(n_components=min(n_components, train_scaled.shape[1]))
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)

    results = []
    for name, train_x, test_x in (
        ("All encoded features", train_scaled, test_scaled),
        (f"First {pca.n_components_} principal components", train_scores, test_scores),
    ):
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(train_x, labels.iloc[train_indices])
        predictions = model.predict(test_x)
        probabilities = model.predict_proba(test_x)[:, 1]
        results.append(
            {
                "representation": name,
                "accuracy": accuracy_score(labels.iloc[test_indices], predictions),
                "balanced_accuracy": balanced_accuracy_score(
                    labels.iloc[test_indices], predictions
                ),
                "f1": f1_score(labels.iloc[test_indices], predictions),
                "roc_auc": roc_auc_score(labels.iloc[test_indices], probabilities),
            }
        )
    return pd.DataFrame(results)


def run_analysis(data_path: Path, output_dir: Path) -> None:
    """Run preprocessing, PCA, plots, and the model comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_data(data_path)
    target_column = locate_column(frame, "Churn")
    features, labels, label_text = prepare_features(frame, target_column)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pca = PCA()
    scores = pca.fit_transform(scaled_features)

    save_variance_plot(pca, output_dir)
    if scores.shape[1] >= 2:
        save_projection_plot(scores, label_text, output_dir)
    save_loading_plot(pca, list(features.columns), output_dir)

    print(f"Rows: {len(frame):,}; encoded features: {features.shape[1]:,}")
    print("\nExplained variance:")
    for component, variance in enumerate(pca.explained_variance_ratio_[:5], start=1):
        print(f"  PC{component}: {variance:.2%}")
    print(f"  First 3 components: {pca.explained_variance_ratio_[:3].sum():.2%}")
    components_for_90 = np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1
    print(f"  Components needed for 90%: {components_for_90}")
    print("\nLargest absolute PC1 loadings:")
    loadings = pd.Series(pca.components_[0], index=features.columns)
    print(loadings.reindex(loadings.abs().sort_values(ascending=False).head(10).index).to_string())

    metrics = evaluate_models(features, labels, n_components=3)
    print("\nHold-out logistic regression comparison (random_state=42):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nPlots saved to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    """Define the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        help="Path to the Telco Excel or CSV file. If omitted, a default filename is searched.",
    )
    parser.add_argument(
        "--output-dir",
        default="pca_outputs",
        help="Directory for generated plots (default: pca_outputs).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_analysis(find_input_file(arguments.data), Path(arguments.output_dir))
