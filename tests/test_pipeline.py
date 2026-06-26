"""
Integration tests for the full data-transformation pipeline:
  feature engineering → preprocessing → SVD

All tests use synthetic data and run in <5 s.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from pipeline import (
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    REQUIRED_COLUMNS,
    validate_columns,
)


# Schema validation

class TestSchemaValidation:

    def test_synthetic_df_has_all_required_columns(self, synthetic_raw_df):
        """The synthetic fixture itself must satisfy the schema check."""
        validate_columns(synthetic_raw_df, REQUIRED_COLUMNS)

    def test_validate_raises_on_dropped_column(self, synthetic_raw_df):
        df_dropped = synthetic_raw_df.drop(columns=["INCIDENT_REASON"])
        with pytest.raises(ValueError, match="INCIDENT_REASON"):
            validate_columns(df_dropped, REQUIRED_COLUMNS)


# Feature engineering — header row removal

class TestHeaderRowRemoval:

    def test_header_rows_are_removed(self, synthetic_raw_df, synthetic_clean_df):
        """The synthetic raw DF contains 1 bogus header row; it must be gone."""
        raw_ids = synthetic_raw_df["INCIDENT_UNIQUE_IDENTIFIER"].astype(str).str.upper()
        assert "INCIDENT_UNIQUE_IDENTIFIER" in raw_ids.values, \
            "Test precondition: raw DF must contain a header row"

        clean_ids = synthetic_clean_df["INCIDENT_UNIQUE_IDENTIFIER"].astype(str).str.upper()
        assert "INCIDENT_UNIQUE_IDENTIFIER" not in clean_ids.values, \
            "Header row must be removed by feature_engineer()"

    def test_clean_df_is_smaller_than_raw(self, synthetic_raw_df, synthetic_clean_df):
        assert len(synthetic_clean_df) < len(synthetic_raw_df)


# Feature engineering — temporal features

class TestTemporalFeatures:

    def test_incident_time_minutes_in_valid_range(self, synthetic_clean_df):
        col = synthetic_clean_df["incident_time_minutes"].dropna()
        assert (col >= 0).all(), "incident_time_minutes must be >= 0"
        assert (col <= 1439).all(), "incident_time_minutes must be <= 1439 (23:59)"

    def test_is_weekend_is_binary(self, synthetic_clean_df):
        col = synthetic_clean_df["is_weekend"]
        assert col.isin([0, 1]).all(), \
            f"is_weekend must be 0 or 1 only; found: {col.unique()}"

    def test_weekend_matches_dayofweek(self, synthetic_clean_df):
        df = synthetic_clean_df.dropna(subset=["incident_dayofweek"])
        weekend_mask = df["incident_dayofweek"].isin([5, 6])
        assert (df.loc[weekend_mask, "is_weekend"] == 1).all(), \
            "is_weekend must be 1 for Saturday (5) and Sunday (6)"
        assert (df.loc[~weekend_mask, "is_weekend"] == 0).all(), \
            "is_weekend must be 0 for Monday–Friday"

    def test_time_period_contains_only_valid_labels(self, synthetic_clean_df):
        valid = {"night", "morning", "afternoon", "evening", "missing"}
        found = set(synthetic_clean_df["time_period"].dropna().unique())
        assert found.issubset(valid), \
            f"Unexpected time_period values: {found - valid}"

    def test_time_period_night_for_early_hours(self, synthetic_clean_df):
        """Incidents before 06:00 must be labelled 'night'."""
        early = synthetic_clean_df[
            synthetic_clean_df["incident_time_minutes"].between(0, 359)
        ]
        if len(early) > 0:
            assert (early["time_period"] == "night").all()

    def test_incident_year_within_dataset_range(self, synthetic_clean_df):
        years = synthetic_clean_df["incident_year"].dropna()
        assert years.between(2012, 2015).all(), \
            f"Years out of expected range: {years[~years.between(2012, 2015)].unique()}"

    def test_incident_month_between_1_and_12(self, synthetic_clean_df):
        months = synthetic_clean_df["incident_month"].dropna()
        assert months.between(1, 12).all()

    def test_incident_dayofweek_between_0_and_6(self, synthetic_clean_df):
        dow = synthetic_clean_df["incident_dayofweek"].dropna()
        assert dow.between(0, 6).all()


# Feature engineering — geographic features

class TestGeographicFeatures:

    def test_lat_bin_is_rounded_to_2dp(self, synthetic_clean_df):
        lat = synthetic_clean_df["LOCATION_LATITUDE"].dropna()
        lat_bin = synthetic_clean_df.loc[lat.index, "lat_bin"].dropna()
        # bin should equal lat rounded to 2dp
        expected = lat.loc[lat_bin.index].round(2)
        pd.testing.assert_series_equal(lat_bin, expected, check_names=False)

    def test_lon_bin_is_rounded_to_2dp(self, synthetic_clean_df):
        lon = synthetic_clean_df["LOCATION_LONGITUDE"].dropna()
        lon_bin = synthetic_clean_df.loc[lon.index, "lon_bin"].dropna()
        expected = lon.loc[lon_bin.index].round(2)
        pd.testing.assert_series_equal(lon_bin, expected, check_names=False)

    def test_coordinates_within_sf_bounds(self, synthetic_clean_df):
        lat = synthetic_clean_df["LOCATION_LATITUDE"].dropna()
        lon = synthetic_clean_df["LOCATION_LONGITUDE"].dropna()
        assert lat.between(37.60, 37.90).all(), \
            "Latitudes must fall within San Francisco bounds"
        assert lon.between(-122.55, -122.30).all(), \
            "Longitudes must fall within San Francisco bounds"


# Feature engineering — text normalization

class TestTextFeatures:

    def test_incident_reason_simple_is_lowercase(self, synthetic_clean_df):
        vals = synthetic_clean_df["incident_reason_simple"].dropna()
        assert (vals == vals.str.lower()).all(), \
            "incident_reason_simple must be fully lowercase"

    def test_disposition_simple_is_lowercase(self, synthetic_clean_df):
        vals = synthetic_clean_df["disposition_simple"].dropna()
        assert (vals == vals.str.lower()).all()

    def test_unknown_not_in_cleaned_reason(self, synthetic_clean_df):
        vals = synthetic_clean_df["incident_reason_simple"].dropna()
        assert "unknown" not in vals.values, \
            "'unknown' should be NaN after cleaning, not a category"


# Feature engineering — model feature set

class TestModelFeatureSet:

    def test_all_model_features_present_after_engineering(self, synthetic_clean_df):
        missing = [f for f in MODEL_FEATURES if f not in synthetic_clean_df.columns]
        assert not missing, f"Missing model features after engineering: {missing}"

    def test_x_raw_has_correct_shape(self, synthetic_X_raw):
        assert synthetic_X_raw.shape[1] == len(MODEL_FEATURES)


# Preprocessing

class TestPreprocessing:

    def test_output_is_sparse_matrix(self, synthetic_X_processed):
        assert scipy.sparse.issparse(synthetic_X_processed), \
            "Preprocessed output must be a sparse matrix (memory-efficient for OHE)"

    def test_no_nan_in_processed_output(self, synthetic_X_processed):
        dense = synthetic_X_processed.toarray()
        assert not np.isnan(dense).any(), \
            "Preprocessed matrix must contain no NaN values (imputation should fix all)"

    def test_no_inf_in_processed_output(self, synthetic_X_processed):
        dense = synthetic_X_processed.toarray()
        assert np.isfinite(dense).all(), \
            "Preprocessed matrix must contain no infinite values"

    def test_row_count_preserved(self, synthetic_X_raw, synthetic_X_processed):
        assert synthetic_X_processed.shape[0] == len(synthetic_X_raw), \
            "Preprocessing must not add or drop rows"

    def test_column_count_expands_due_to_ohe(self, synthetic_X_raw, synthetic_X_processed):
        assert synthetic_X_processed.shape[1] > len(MODEL_FEATURES), \
            "OHE should expand the column count beyond the raw feature count"

    def test_numeric_features_are_scaled(self, synthetic_X_processed):
        """Numeric columns (first N) should have mean ≈ 0 after StandardScaler."""
        dense = synthetic_X_processed.toarray()
        n_numeric = len(NUMERIC_FEATURES)
        numeric_block = dense[:, :n_numeric]
        col_means = numeric_block.mean(axis=0)
        assert np.allclose(col_means, 0, atol=1e-6), \
            f"Scaled numeric columns should have mean ≈ 0; got max |mean| = {np.abs(col_means).max():.6f}"


# SVD

class TestSVD:

    def test_output_shape(self, synthetic_X_processed, synthetic_X_svd):
        expected_rows = synthetic_X_processed.shape[0]
        assert synthetic_X_svd.shape[0] == expected_rows, \
            "SVD must preserve row count"

    def test_output_is_dense(self, synthetic_X_svd):
        assert isinstance(synthetic_X_svd, np.ndarray), \
            "SVD output must be a dense numpy array"

    def test_all_values_finite(self, synthetic_X_svd):
        assert np.isfinite(synthetic_X_svd).all(), \
            "SVD output must contain no NaN or infinite values"

    def test_explained_variance_ratios_positive(self, synthetic_variance_df):
        assert (synthetic_variance_df["explained_var_ratio"] > 0).all(), \
            "Every SVD component must explain a strictly positive amount of variance"

    def test_explained_variance_ratios_sum_leq_one(self, synthetic_variance_df):
        total = synthetic_variance_df["explained_var_ratio"].sum()
        assert total <= 1.0 + 1e-9, \
            f"Explained variance ratios cannot sum to more than 1.0; got {total:.4f}"

    def test_cumulative_variance_is_monotonically_increasing(self, synthetic_variance_df):
        cumvar = synthetic_variance_df["cumulative_explained_var"].values
        diffs = np.diff(cumvar)
        assert (diffs >= 0).all(), \
            "Cumulative explained variance must be non-decreasing"

    def test_cumulative_variance_ends_at_total(self, synthetic_variance_df):
        total_ratio = synthetic_variance_df["explained_var_ratio"].sum()
        cumulative_end = synthetic_variance_df["cumulative_explained_var"].iloc[-1]
        assert abs(cumulative_end - total_ratio) < 1e-9
