from __future__ import annotations

from pipeline import *

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constants
RANDOM_STATE = 42
SVD_COMPONENTS = 50
BEST_K = 4

REQUIRED_COLUMNS: list[str] = [
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

MODEL_FEATURES: list[str] = [
    "incident_reason_simple",
    "INCIDENT_REASON_DESCRIPTION",
    "incident_year",
    "incident_month",
    "incident_dayofweek",
    "incident_time_minutes",
    "is_weekend",
    "time_period",
    "LOCATION_DISTRICT",
    "LOCATION_LONGITUDE",
    "LOCATION_LATITUDE",
    "lat_bin",
    "lon_bin",
    "disposition_simple",
]

NUMERIC_FEATURES: list[str] = [
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

CATEGORICAL_FEATURES: list[str] = [
    "incident_reason_simple",
    "INCIDENT_REASON_DESCRIPTION",
    "time_period",
    "LOCATION_DISTRICT",
    "disposition_simple",
]

# Helper functions

def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise ValueError if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_text(series: pd.Series) -> pd.Series:
    """Lowercase, strip whitespace, replace placeholder strings with NaN."""
    cleaned = series.astype("object").where(series.notna(), np.nan)
    cleaned = cleaned.astype(str).str.strip().str.lower()
    cleaned = cleaned.replace(["nan", "none", "<na>", "unknown", "unk", ""], np.nan)
    return cleaned.astype("object")


def minutes_to_time(minutes: float) -> str:
    """Convert fractional minutes-past-midnight to HH:MM string."""
    if pd.isna(minutes):
        return "missing"
    h, m = int(minutes // 60), int(minutes % 60)
    return f"{h:02d}:{m:02d}"


def make_ohe() -> OneHotEncoder:
    """Create a OneHotEncoder compatible with both old and new scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def top_values(series: pd.Series, n: int = 3) -> str:
    """Return the top-n most frequent values as 'value: share%' separated by '; '."""
    shares = series.fillna("missing").value_counts(normalize=True).head(n)
    return "; ".join(f"{idx}: {val:.1%}" for idx, val in shares.items())


def sample_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return a random sample of n rows (or a full copy if df is smaller)."""
    return df.copy() if len(df) <= n else df.sample(n=n, random_state=RANDOM_STATE).copy()


def sample_arr(X: np.ndarray, n: int) -> np.ndarray:
    """Return a random sample of n rows from a numpy array."""
    if X.shape[0] <= n:
        return X
    idx = np.random.default_rng(RANDOM_STATE).choice(X.shape[0], size=n, replace=False)
    return X[idx]


# Feature engineering

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning and feature engineering steps to the raw DataFrame.
    Returns df_clean with all derived columns added.
    """
    df_clean = df.copy()

    # Remove repeated header rows (CSV export artefact)
    header_like = (
        df_clean["INCIDENT_UNIQUE_IDENTIFIER"].astype(str).str.upper()
            .eq("INCIDENT_UNIQUE_IDENTIFIER")
        | df_clean["INCIDENT_REASON"].astype(str).str.upper()
            .isin(["INCIDENT_REASON", "CATEGORY"])
    )
    df_clean = df_clean.loc[~header_like].reset_index(drop=True)

    # Normalise text columns
    for col in ["INCIDENT_REASON", "INCIDENT_REASON_DESCRIPTION",
                "INCIDENT_DAY", "LOCATION_DISTRICT", "DISPOSITION"]:
        df_clean[col] = clean_text(df_clean[col])

    # Coordinates as float
    df_clean["LOCATION_LONGITUDE"] = pd.to_numeric(df_clean["LOCATION_LONGITUDE"], errors="coerce")
    df_clean["LOCATION_LATITUDE"]  = pd.to_numeric(df_clean["LOCATION_LATITUDE"],  errors="coerce")

    # Date-derived features
    incident_date = pd.to_datetime(df_clean["INCIDENT_DATE"], errors="coerce")
    df_clean["incident_year"]      = incident_date.dt.year
    df_clean["incident_month"]     = incident_date.dt.month
    df_clean["incident_dayofweek"] = incident_date.dt.dayofweek

    # Time as minutes past midnight
    incident_time = pd.to_datetime(df_clean["INCIDENT_TIME"], format="%H:%M", errors="coerce")
    df_clean["incident_time_minutes"] = incident_time.dt.hour * 60 + incident_time.dt.minute

    # Weekend flag
    df_clean["is_weekend"] = (
        df_clean["incident_dayofweek"].fillna(-1).isin([5, 6]).astype(int)
    )

    # Four-period time-of-day label
    time_period = pd.cut(
        df_clean["incident_time_minutes"],
        bins=[-1, 359, 719, 1079, 1439],
        labels=["night", "morning", "afternoon", "evening"],
    )
    df_clean["time_period"] = (
        time_period.astype("object").where(time_period.notna(), "missing")
    )

    # Geographic bins (0.01° ≈ 1 km)
    df_clean["lat_bin"] = df_clean["LOCATION_LATITUDE"].round(2)
    df_clean["lon_bin"] = df_clean["LOCATION_LONGITUDE"].round(2)

    # Simplified text features
    df_clean["incident_reason_simple"] = clean_text(df_clean["INCIDENT_REASON"])
    df_clean["disposition_simple"]     = clean_text(df_clean["DISPOSITION"])

    return df_clean


# Preprocessing pipeline factory

def build_preprocessor() -> ColumnTransformer:
    """Build and return the sklearn ColumnTransformer (unfitted)."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_ohe()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      NUMERIC_FEATURES),
            ("cat", categorical_pipeline,  CATEGORICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


# SVD

def run_svd(X_processed, n_components: int = SVD_COMPONENTS):
    """Fit TruncatedSVD and return (X_svd, svd_model, variance_df)."""
    n_comp = min(n_components, X_processed.shape[1] - 1)
    svd_model = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    X_svd = svd_model.fit_transform(X_processed)

    variance_df = pd.DataFrame({
        "component":            np.arange(1, n_comp + 1),
        "explained_var_ratio":  svd_model.explained_variance_ratio_,
        "cumulative_explained_var": svd_model.explained_variance_ratio_.cumsum(),
    })
    return X_svd, svd_model, variance_df


# Clustering

def run_clustering(X_svd: np.ndarray, k: int = BEST_K) -> tuple[np.ndarray, MiniBatchKMeans]:
    """Fit MiniBatch K-Means and return (labels, model)."""
    model = MiniBatchKMeans(
        n_clusters=k, random_state=RANDOM_STATE, batch_size=4096, n_init=10
    )
    labels = model.fit_predict(X_svd)
    return labels, model
