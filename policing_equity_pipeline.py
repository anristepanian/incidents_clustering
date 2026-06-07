"""
Policing Equity case-study pipeline

This script cleans the San Francisco incident-report data, engineers features,
performs sparse preprocessing, applies dimensionality reduction, clusters
incidents, and saves report-ready tables and figures.

Expected input file:
    cpe-data/Dept_49-00081/49-00081_Incident-Reports_2012_to_May_2015.csv

Run from the project root, or change DATA_PATH below.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_PATH = Path("cpe-data/Dept_49-00081/49-00081_Incident-Reports_2012_to_May_2015.csv")
OUTPUT_DIR = Path("outputs")
FIGURE_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42
SVD_COMPONENTS = 50
BEST_K = 5                    # Change this after reviewing cluster_evaluation.csv and plots.
K_RANGE = range(2, 11)
EVAL_SAMPLE_SIZE = 30_000     # Used for faster k evaluation.
SILHOUETTE_SAMPLE_SIZE = 5_000
PLOT_SAMPLE_SIZE = 100_000    # Used for scatter plots only.

REQUIRED_COLUMNS = [
    "INCIDENT_UNIQUE_IDENTIFIER",
    "INCIDENT_REASON",
    "INCIDENT_REASON_DESCRIPTION",
    "INCIDENT_DAY",
    "INCIDENT_DATE",
    "INCIDENT_TIME",
    "LOCATION_DISTRICT",
    "LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION",
    "DISPOSITION",
    "LOCATION_LONGITUDE",
    "LOCATION_LATITUDE",
]

MODEL_FEATURES = [
    # Incident type
    "incident_reason_simple",
    "INCIDENT_REASON_DESCRIPTION",

    # Time
    "incident_year",
    "incident_month",
    "incident_dayofweek",
    "incident_time_minutes",
    "is_weekend",
    "time_period",

    # Geography
    "LOCATION_DISTRICT",
    "LOCATION_LONGITUDE",
    "LOCATION_LATITUDE",
    "lat_bin",
    "lon_bin",

    # Outcome
    "disposition_simple",
]

NUMERIC_FEATURES = [
    "incident_year",
    "incident_month",
    "incident_dayofweek",
    "incident_time_minutes",
    "is_weekend",
    "LOCATION_LONGITUDE",
    "LOCATION_LATITUDE",
    "lat_bin",
    "lon_bin",
]

CATEGORICAL_FEATURES = [
    "incident_reason_simple",
    "INCIDENT_REASON_DESCRIPTION",
    "time_period",
    "LOCATION_DISTRICT",
    "disposition_simple",
]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.show()


def sample_dataframe(df: pd.DataFrame, max_rows: int, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()


def sample_array(X: np.ndarray, max_rows: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=max_rows, replace=False)
    return X[idx]


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .replace(["nan", "none", "unknown", "unk", ""], pd.NA)
    )


def minutes_to_time(minutes: float) -> str:
    if pd.isna(minutes):
        return "missing"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours:02d}:{mins:02d}"


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a sparse OneHotEncoder compatible with older/newer scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def top_values(series: pd.Series, n: int = 3) -> str:
    shares = series.fillna("missing").value_counts(normalize=True).head(n)
    return "; ".join([f"{idx}: {val:.1%}" for idx, val in shares.items()])


def plot_stacked_distribution(
    data: pd.DataFrame,
    row_col: str,
    category_col: str,
    title: str,
    filename: str,
    top_n: int | None = 10,
    normalize: bool = True,
    legend_title: str | None = None,
) -> pd.DataFrame:
    """Create a stacked bar chart with the legend outside the plot area."""
    plot_df = data[[row_col, category_col]].copy()

    if top_n is not None:
        top_categories = plot_df[category_col].value_counts().head(top_n).index
        plot_df[category_col] = np.where(
            plot_df[category_col].isin(top_categories),
            plot_df[category_col],
            "other",
        )

    table = pd.crosstab(plot_df[row_col], plot_df[category_col], normalize="index" if normalize else False)
    table = table.round(3)

    ax = table.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.title(title)
    plt.xlabel(row_col.replace("_", " ").title())
    plt.ylabel("Share within cluster" if normalize else "Number of incidents")

    ax.legend(
        title=legend_title or category_col,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    save_current_figure(filename)
    return table


# -----------------------------------------------------------------------------
# 1. Load and inspect data
# -----------------------------------------------------------------------------

def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_COLUMNS)
    return df


def create_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "INCIDENT_UNIQUE_IDENTIFIER": "id_like",
        "INCIDENT_REASON": "incident_type",
        "INCIDENT_REASON_DESCRIPTION": "incident_type",
        "INCIDENT_DAY": "date_time",
        "INCIDENT_DATE": "date_time",
        "INCIDENT_TIME": "date_time",
        "LOCATION_DISTRICT": "geographic",
        "LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION": "geographic",
        "LOCATION_LONGITUDE": "geographic",
        "LOCATION_LATITUDE": "geographic",
        "DISPOSITION": "outcome",
    }

    summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_share": df.isna().mean().values,
        "unique_values": df.nunique(dropna=True).values,
    })
    summary["column_group"] = summary["column"].map(groups).fillna("other")
    return summary.sort_values(["column_group", "column"])


# -----------------------------------------------------------------------------
# 2. Clean and engineer features
# -----------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df_prepared = df.copy()

    # Some Kaggle files include repeated header rows. Remove them if present.
    header_like = (
        df_prepared["INCIDENT_UNIQUE_IDENTIFIER"].astype(str).str.upper().eq("INCIDENT_UNIQUE_IDENTIFIER")
        | df_prepared["INCIDENT_REASON"].astype(str).str.upper().isin(["INCIDENT_REASON", "CATEGORY"])
    )
    df_prepared = df_prepared.loc[~header_like].reset_index(drop=True)

    # Clean relevant text columns.
    text_columns = [
        "INCIDENT_REASON",
        "INCIDENT_REASON_DESCRIPTION",
        "INCIDENT_DAY",
        "LOCATION_DISTRICT",
        "DISPOSITION",
    ]
    for col in text_columns:
        df_prepared[col] = clean_text(df_prepared[col])

    # Convert coordinates to numeric. They must not be one-hot encoded.
    df_prepared["LOCATION_LONGITUDE"] = pd.to_numeric(df_prepared["LOCATION_LONGITUDE"], errors="coerce")
    df_prepared["LOCATION_LATITUDE"] = pd.to_numeric(df_prepared["LOCATION_LATITUDE"], errors="coerce")

    # Decompose date into interpretable features.
    incident_date = pd.to_datetime(df_prepared["INCIDENT_DATE"], errors="coerce")
    df_prepared["incident_year"] = incident_date.dt.year
    df_prepared["incident_month"] = incident_date.dt.month
    df_prepared["incident_dayofweek"] = incident_date.dt.dayofweek

    # Convert time to minutes after midnight.
    incident_time = pd.to_datetime(df_prepared["INCIDENT_TIME"], format="%H:%M", errors="coerce")
    df_prepared["incident_time_minutes"] = incident_time.dt.hour * 60 + incident_time.dt.minute

    # Additional engineered features.
    df_prepared["is_weekend"] = df_prepared["incident_dayofweek"].isin([5, 6]).astype(int)

    df_prepared["time_period"] = pd.cut(
        df_prepared["incident_time_minutes"],
        bins=[-1, 359, 719, 1079, 1439],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype("string").fillna("missing")

    df_prepared["lat_bin"] = df_prepared["LOCATION_LATITUDE"].round(2)
    df_prepared["lon_bin"] = df_prepared["LOCATION_LONGITUDE"].round(2)
    df_prepared["incident_reason_simple"] = clean_text(df_prepared["INCIDENT_REASON"])
    df_prepared["disposition_simple"] = clean_text(df_prepared["DISPOSITION"])

    return df_prepared


def create_feature_summary(X_raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": X_raw.columns,
        "dtype": X_raw.dtypes.astype(str).values,
        "missing_values": X_raw.isna().sum().values,
        "missing_share": X_raw.isna().mean().values,
        "unique_values": X_raw.nunique(dropna=True).values,
    }).sort_values("unique_values", ascending=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


# -----------------------------------------------------------------------------
# 3. Dimensionality reduction
# -----------------------------------------------------------------------------

def reduce_dimensions(X_processed) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    n_components = min(SVD_COMPONENTS, X_processed.shape[1] - 1)
    if n_components < 2:
        raise ValueError("Not enough processed features for dimensionality reduction.")

    svd_50 = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_svd = svd_50.fit_transform(X_processed)

    svd_df = pd.DataFrame({
        "SVD1": X_svd[:, 0],
        "SVD2": X_svd[:, 1],
    })

    variance_df = pd.DataFrame({
        "component": np.arange(1, n_components + 1),
        "explained_variance_ratio": svd_50.explained_variance_ratio_,
        "cumulative_explained_variance": svd_50.explained_variance_ratio_.cumsum(),
    })

    return svd_df, X_svd, variance_df


# -----------------------------------------------------------------------------
# 4. Clustering and interpretation
# -----------------------------------------------------------------------------

def evaluate_cluster_numbers(X_svd: np.ndarray) -> pd.DataFrame:
    X_eval = sample_array(X_svd, EVAL_SAMPLE_SIZE, RANDOM_STATE)
    rows = []

    for k in K_RANGE:
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            batch_size=4096,
            n_init=10,
        )
        labels = model.fit_predict(X_eval)

        sil_sample = min(SILHOUETTE_SAMPLE_SIZE, X_eval.shape[0])
        rows.append({
            "k": k,
            "silhouette_score": silhouette_score(
                X_eval,
                labels,
                sample_size=sil_sample,
                random_state=RANDOM_STATE,
            ),
            "davies_bouldin_score": davies_bouldin_score(X_eval, labels),
            "inertia": model.inertia_,
        })

    return pd.DataFrame(rows)


def fit_final_clusters(X_svd: np.ndarray, k: int = BEST_K) -> np.ndarray:
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=10,
    )
    return model.fit_predict(X_svd)


def create_cluster_summary(df_clustered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cluster_counts = (
        df_clustered["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_counts.columns = ["cluster", "number_of_incidents"]
    cluster_counts["share_of_total"] = (cluster_counts["number_of_incidents"] / len(df_clustered)).round(4)

    numeric_cols = [
        "incident_year",
        "incident_month",
        "incident_dayofweek",
        "incident_time_minutes",
        "LOCATION_LONGITUDE",
        "LOCATION_LATITUDE",
    ]
    numeric_summary = df_clustered.groupby("cluster")[numeric_cols].agg(["mean", "median", "min", "max"]).round(2)

    interpretation_rows = []
    for cluster_id, group in df_clustered.groupby("cluster"):
        interpretation_rows.append({
            "cluster": cluster_id,
            "number_of_incidents": len(group),
            "share_of_total": round(len(group) / len(df_clustered) * 100, 2),
            "top_incident_reasons": top_values(group["incident_reason_simple"]),
            "top_reason_descriptions": top_values(group["INCIDENT_REASON_DESCRIPTION"]),
            "top_days": top_values(group["INCIDENT_DAY"]),
            "top_time_periods": top_values(group["time_period"]),
            "top_districts": top_values(group["LOCATION_DISTRICT"]),
            "top_dispositions": top_values(group["disposition_simple"]),
            "avg_incident_time": minutes_to_time(group["incident_time_minutes"].mean()),
            "avg_latitude": round(group["LOCATION_LATITUDE"].mean(), 5),
            "avg_longitude": round(group["LOCATION_LONGITUDE"].mean(), 5),
        })

    interpretation = pd.DataFrame(interpretation_rows)
    return cluster_counts, numeric_summary, interpretation


# -----------------------------------------------------------------------------
# 5. Visualizations
# -----------------------------------------------------------------------------

def create_plots(
    df_clustered: pd.DataFrame,
    svd_df: pd.DataFrame,
    cluster_counts: pd.DataFrame,
    cluster_eval: pd.DataFrame,
    variance_df: pd.DataFrame,
) -> None:
    # Cluster evaluation plots.
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_eval["k"], cluster_eval["silhouette_score"], marker="o")
    plt.title("Silhouette Score by Number of Clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    save_current_figure("silhouette_by_k.png")

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_eval["k"], cluster_eval["davies_bouldin_score"], marker="o")
    plt.title("Davies-Bouldin Score by Number of Clusters")
    plt.xlabel("Number of clusters")
    plt.ylabel("Davies-Bouldin score")
    save_current_figure("davies_bouldin_by_k.png")

    plt.figure(figsize=(8, 5))
    plt.plot(cluster_eval["k"], cluster_eval["inertia"], marker="o")
    plt.title("Elbow Plot")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    save_current_figure("elbow_plot.png")

    plt.figure(figsize=(8, 5))
    plt.plot(variance_df["component"], variance_df["cumulative_explained_variance"], marker="o")
    plt.title("Cumulative Explained Variance by SVD Components")
    plt.xlabel("Number of SVD components")
    plt.ylabel("Cumulative explained variance")
    save_current_figure("svd_cumulative_explained_variance.png")

    # Cluster sizes.
    plt.figure(figsize=(8, 5))
    plt.bar(cluster_counts["cluster"].astype(str), cluster_counts["number_of_incidents"])
    plt.title("Number of Incidents per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of incidents")
    save_current_figure("incidents_per_cluster.png")

    # 2D SVD cluster scatter. Use sample for readability.
    svd_plot_df = sample_dataframe(svd_df, PLOT_SAMPLE_SIZE, RANDOM_STATE)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        svd_plot_df["SVD1"],
        svd_plot_df["SVD2"],
        c=svd_plot_df["cluster"],
        s=2,
        alpha=0.4,
    )
    plt.title("Incident Clusters in Reduced 2D Space")
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    plt.colorbar(scatter, label="Cluster")
    save_current_figure("svd_clusters_2d.png")

    # Geographic scatter. Use sample for readability.
    geo_plot_df = df_clustered.dropna(subset=["LOCATION_LONGITUDE", "LOCATION_LATITUDE"])
    geo_plot_df = geo_plot_df[
        geo_plot_df["LOCATION_LONGITUDE"].between(-180, 180)
        & geo_plot_df["LOCATION_LATITUDE"].between(-90, 90)
    ]
    geo_plot_df = sample_dataframe(geo_plot_df, PLOT_SAMPLE_SIZE, RANDOM_STATE)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        geo_plot_df["LOCATION_LONGITUDE"],
        geo_plot_df["LOCATION_LATITUDE"],
        c=geo_plot_df["cluster"],
        s=2,
        alpha=0.4,
    )
    plt.title("Geographic Distribution of Incident Clusters")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(scatter, label="Cluster")
    save_current_figure("geographic_cluster_distribution.png")

    # Stacked distributions with external legends.
    plot_stacked_distribution(
        df_clustered,
        row_col="cluster",
        category_col="incident_reason_simple",
        title="Top Incident Reason Distribution by Cluster",
        filename="incident_reason_by_cluster.png",
        top_n=10,
        legend_title="Incident reason",
    )

    plot_stacked_distribution(
        df_clustered,
        row_col="cluster",
        category_col="disposition_simple",
        title="Disposition Distribution by Cluster",
        filename="disposition_by_cluster.png",
        top_n=None,
        legend_title="Disposition",
    )

    plot_stacked_distribution(
        df_clustered,
        row_col="cluster",
        category_col="time_period",
        title="Time Period Distribution by Cluster",
        filename="time_period_by_cluster.png",
        top_n=None,
        legend_title="Time period",
    )

    plot_stacked_distribution(
        df_clustered,
        row_col="LOCATION_DISTRICT",
        category_col="cluster",
        title="Incident Clusters by District",
        filename="clusters_by_district.png",
        top_n=None,
        normalize=False,
        legend_title="Cluster",
    )


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    df = load_data(DATA_PATH)
    column_summary = create_column_summary(df)
    column_summary.to_csv(OUTPUT_DIR / "column_summary.csv", index=False)

    df_prepared = prepare_features(df)
    X_raw = df_prepared[MODEL_FEATURES].copy()

    feature_summary = create_feature_summary(X_raw)
    feature_summary.to_csv(OUTPUT_DIR / "feature_summary.csv", index=False)
    X_raw.to_csv(OUTPUT_DIR / "policing_equity_features.csv", index=False)

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X_raw)

    svd_df, X_svd, variance_df = reduce_dimensions(X_processed)
    variance_df.to_csv(OUTPUT_DIR / "svd_explained_variance.csv", index=False)

    svd_components_df = pd.DataFrame(X_svd, columns=[f"SVD_{i + 1}" for i in range(X_svd.shape[1])])
    svd_components_df.to_csv(OUTPUT_DIR / "svd_components.csv", index=False)

    cluster_eval = evaluate_cluster_numbers(X_svd)
    cluster_eval.to_csv(OUTPUT_DIR / "cluster_evaluation.csv", index=False)

    cluster_labels = fit_final_clusters(X_svd, BEST_K)
    df_clustered = df_prepared.copy()
    df_clustered["cluster"] = cluster_labels
    svd_df["cluster"] = cluster_labels

    cluster_counts, numeric_summary, interpretation = create_cluster_summary(df_clustered)
    cluster_counts.to_csv(OUTPUT_DIR / "cluster_counts.csv", index=False)
    numeric_summary.to_csv(OUTPUT_DIR / "numeric_cluster_summary.csv")
    interpretation.to_csv(OUTPUT_DIR / "final_cluster_summary_for_report.csv", index=False)
    df_clustered.to_csv(OUTPUT_DIR / "policing_equity_clustered.csv", index=False)
    svd_df.to_csv(OUTPUT_DIR / "svd_2d_coordinates.csv", index=False)

    create_plots(df_clustered, svd_df, cluster_counts, cluster_eval, variance_df)

    print("Pipeline finished.")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")
    print(f"Figures saved to: {FIGURE_DIR.resolve()}")


if __name__ == "__main__":
    main()
