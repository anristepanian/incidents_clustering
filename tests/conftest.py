"""
Shared pytest fixtures.

Two data tiers:
  - synthetic_raw_df  : always available, no CSV needed, used for unit & pipeline tests
  - real_df           : loads the actual CSV; tests using this are auto-skipped if the
                        file is absent (set DATA_PATH in pytest.ini or env var)
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed during tests

import numpy as np
import pandas as pd
import pytest

from pipeline import (
    BEST_K,
    MODEL_FEATURES,
    SVD_COMPONENTS,
    build_preprocessor,
    feature_engineer,
    run_clustering,
    run_svd,
)

# Path to the real dataset (override with DATA_PATH env variable)
DATA_PATH = Path(os.environ.get("DATA_PATH",
                 "49-00081_Incident-Reports_2012_to_May_2015.csv"))


# Synthetic data

@pytest.fixture(scope="session")
def synthetic_raw_df() -> pd.DataFrame:
    """
    500-row synthetic DataFrame that matches the CPE schema.
    Used for all fast unit and pipeline tests — no real CSV required.
    """
    rng = np.random.default_rng(42)
    n = 500

    reasons      = ["traffic stop", "suspicious person", "domestic disturbance",
                    "welfare check", "assault", "UNKNOWN", ""]
    descriptions = ["vehicle code violation", "suspicious activity",
                    "disturbance call", "nan"]
    dispositions = ["citation issued", "arrest made", "unfounded",
                    "scene dismissal", "medical handoff", "UNKNOWN"]
    districts    = ["SOUTHERN", "MISSION", "BAYVIEW", "NORTHERN", "TENDERLOIN",
                    "CENTRAL", "INGLESIDE"]
    days         = ["monday", "tuesday", "wednesday", "thursday",
                    "friday", "saturday", "sunday"]

    dates = pd.date_range("2012-01-01", "2015-05-31", periods=n)
    hours = rng.integers(0, 24, n)
    mins  = rng.integers(0, 60, n)
    times = np.array([f"{h:02d}:{m:02d}" for h, m in zip(hours, mins)], dtype=object)

    # Realistic SF bounding box
    lats = rng.uniform(37.70, 37.81, n)
    lons = rng.uniform(-122.52, -122.35, n)

    # Introduce ~5% missing values in time and coords
    lats[rng.choice(n, 25, replace=False)] = np.nan
    lons[rng.choice(n, 25, replace=False)] = np.nan
    times[rng.choice(n, 30, replace=False)] = None

    # One bogus header row (like a real CSV export artefact)
    df = pd.DataFrame({
        "INCIDENT_UNIQUE_IDENTIFIER": np.concatenate([["INCIDENT_UNIQUE_IDENTIFIER"],
                                                       rng.integers(1000, 9999, n - 1).astype(str)]),
        "INCIDENT_REASON":            np.concatenate([["INCIDENT_REASON"],
                                                       rng.choice(reasons, n - 1)]),
        "INCIDENT_REASON_DESCRIPTION": np.concatenate([["INCIDENT_REASON_DESCRIPTION"],
                                                        rng.choice(descriptions, n - 1)]),
        "INCIDENT_DAY":               np.concatenate([["INCIDENT_DAY"],
                                                       rng.choice(days, n - 1)]),
        "INCIDENT_DATE":              np.concatenate([["INCIDENT_DATE"],
                                                       dates[1:].strftime("%Y-%m-%d")]),
        "INCIDENT_TIME":              np.concatenate([["INCIDENT_TIME"], times[1:]]),
        "LOCATION_DISTRICT":          np.concatenate([["LOCATION_DISTRICT"],
                                                       rng.choice(districts, n - 1)]),
        "LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION":
                                      [f"{i} Main St" for i in range(n)],
        "DISPOSITION":                np.concatenate([["DISPOSITION"],
                                                       rng.choice(dispositions, n - 1)]),
        "LOCATION_LONGITUDE":         np.concatenate([[np.nan], lons[1:]]),
        "LOCATION_LATITUDE":          np.concatenate([[np.nan], lats[1:]]),
    })
    return df


@pytest.fixture(scope="session")
def synthetic_clean_df(synthetic_raw_df) -> pd.DataFrame:
    """Feature-engineered version of the synthetic dataset."""
    return feature_engineer(synthetic_raw_df)


@pytest.fixture(scope="session")
def synthetic_X_raw(synthetic_clean_df) -> pd.DataFrame:
    """Model-feature subset of the cleaned synthetic data."""
    return synthetic_clean_df[MODEL_FEATURES].copy()


@pytest.fixture(scope="session")
def synthetic_X_processed(synthetic_X_raw):
    """Preprocessed sparse matrix from synthetic data."""
    pre = build_preprocessor()
    return pre.fit_transform(synthetic_X_raw)


@pytest.fixture(scope="session")
def synthetic_X_svd(synthetic_X_processed):
    """SVD-reduced array from synthetic data (fewer components to keep tests fast)."""
    X_svd, _, _ = run_svd(synthetic_X_processed, n_components=20)
    return X_svd


@pytest.fixture(scope="session")
def synthetic_variance_df(synthetic_X_processed):
    """Variance DataFrame from SVD run on synthetic data."""
    _, _, variance_df = run_svd(synthetic_X_processed, n_components=20)
    return variance_df


@pytest.fixture(scope="session")
def synthetic_cluster_labels(synthetic_X_svd):
    """Cluster labels produced by K-Means on synthetic data."""
    labels, _ = run_clustering(synthetic_X_svd, k=BEST_K)
    return labels


# Real data (skipped automatically when CSV is absent)

real_data_available = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason=f"Real dataset not found at '{DATA_PATH}'. "
           f"Set DATA_PATH env var or place the CSV in the project root.",
)


@pytest.fixture(scope="session")
def real_raw_df():
    """Load a 10,000-row sample of the real CSV (session-scoped for speed)."""
    if not DATA_PATH.exists():
        pytest.skip(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False, nrows=10_000)
    return df


@pytest.fixture(scope="session")
def real_clean_df(real_raw_df):
    return feature_engineer(real_raw_df)


@pytest.fixture(scope="session")
def real_X_processed(real_clean_df):
    X_raw = real_clean_df[MODEL_FEATURES].copy()
    pre = build_preprocessor()
    return pre.fit_transform(X_raw)


@pytest.fixture(scope="session")
def real_X_svd(real_X_processed):
    X_svd, _, _ = run_svd(real_X_processed, n_components=SVD_COMPONENTS)
    return X_svd


@pytest.fixture(scope="session")
def real_cluster_labels(real_X_svd):
    labels, _ = run_clustering(real_X_svd, k=BEST_K)
    return labels


@pytest.fixture(scope="session")
def real_df_clustered(real_clean_df, real_cluster_labels):
    df = real_clean_df.copy()
    df["cluster"] = real_cluster_labels
    return df
