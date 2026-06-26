"""
Pure unit tests for every helper function in pipeline.py.
These tests are fast (<1 s total), need no CSV, and no ML libraries.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline import (
    clean_text,
    minutes_to_time,
    top_values,
    sample_df,
    sample_arr,
    validate_columns,
    make_ohe,
)


# clean_text

class TestCleanText:

    def test_lowercases_all_values(self):
        s = pd.Series(["TRAFFIC STOP", "Assault", "WELFARE CHECK"])
        result = clean_text(s)
        assert all(v == v.lower() for v in result.dropna()), \
            "clean_text must lowercase everything"

    def test_strips_leading_trailing_whitespace(self):
        s = pd.Series(["  traffic stop  ", "\twelfare check\n"])
        result = clean_text(s)
        assert result.iloc[0] == "traffic stop"
        assert result.iloc[1] == "welfare check"

    def test_replaces_unknown_with_nan(self):
        s = pd.Series(["UNKNOWN", "unknown", "unk"])
        result = clean_text(s)
        assert result.isna().all(), \
            "Placeholder values ('unknown', 'unk') must become NaN"

    def test_replaces_none_string_with_nan(self):
        s = pd.Series(["none", "None", "NONE"])
        result = clean_text(s)
        assert result.isna().all()

    def test_replaces_nan_string_with_nan(self):
        s = pd.Series(["nan", "NaN"])
        result = clean_text(s)
        assert result.isna().all()

    def test_replaces_empty_string_with_nan(self):
        s = pd.Series([""])
        result = clean_text(s)
        assert result.isna().all()

    def test_preserves_actual_null(self):
        s = pd.Series([None, np.nan])
        result = clean_text(s)
        assert result.isna().all()

    def test_passes_through_valid_text(self):
        s = pd.Series(["traffic stop", "assault"])
        result = clean_text(s)
        assert result.tolist() == ["traffic stop", "assault"]

    def test_mixed_series(self):
        s = pd.Series(["TRAFFIC STOP", "UNKNOWN", None, "  assault  "])
        result = clean_text(s)
        assert result.iloc[0] == "traffic stop"
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == "assault"


# minutes_to_time

class TestMinutesToTime:

    def test_midnight(self):
        assert minutes_to_time(0) == "00:00"

    def test_noon(self):
        assert minutes_to_time(720) == "12:00"

    def test_one_thirty(self):
        assert minutes_to_time(90) == "01:30"

    def test_end_of_day(self):
        assert minutes_to_time(1439) == "23:59"

    def test_nan_returns_missing(self):
        assert minutes_to_time(float("nan")) == "missing"

    def test_none_returns_missing(self):
        assert minutes_to_time(None) == "missing"

    def test_zero_padding_hours(self):
        result = minutes_to_time(65)   # 1h 5m
        assert result == "01:05", f"Expected '01:05', got '{result}'"

    def test_zero_padding_minutes(self):
        result = minutes_to_time(600)  # 10h 0m
        assert result == "10:00"

    def test_float_input_truncates(self):
        result = minutes_to_time(90.9)
        assert result == "01:30", "Should truncate fractional minutes"


# top_values

class TestTopValues:

    def test_returns_string(self):
        s = pd.Series(["a", "a", "b", "c"])
        assert isinstance(top_values(s), str)

    def test_most_frequent_first(self):
        s = pd.Series(["a"] * 10 + ["b"] * 5 + ["c"] * 1)
        result = top_values(s, n=1)
        assert result.startswith("a:"), \
            f"Most frequent value should be first, got: {result}"

    def test_format_contains_percentage(self):
        s = pd.Series(["a", "a", "b"])
        result = top_values(s)
        assert "%" in result, "Result must contain percentage signs"

    def test_separator_is_semicolon(self):
        s = pd.Series(["a", "b", "c"])
        result = top_values(s, n=3)
        assert "; " in result, "Values must be separated by '; '"

    def test_missing_counted_as_missing_label(self):
        s = pd.Series([None, None, "a"])
        result = top_values(s, n=1)
        assert "missing" in result, \
            "NaN values should appear as 'missing' in the result"

    def test_n_limits_output(self):
        s = pd.Series(list("abcdefghij"))
        result = top_values(s, n=2)
        parts = result.split("; ")
        assert len(parts) == 2, f"Expected 2 entries, got {len(parts)}"

    def test_percentages_sum_close_to_100_for_full_n(self):
        s = pd.Series(["a"] * 60 + ["b"] * 40)
        result = top_values(s, n=2)
        # Extract numbers
        import re
        numbers = [float(x.strip("%")) for x in re.findall(r"[\d.]+%", result)]
        assert abs(sum(numbers) - 100.0) < 1.0


# sample_df

class TestSampleDf:

    def test_returns_copy_when_df_smaller_than_n(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = sample_df(df, n=100)
        assert len(result) == len(df)
        # Must be a copy, not a view
        result["a"].iloc[0] = 999
        assert df["a"].iloc[0] != 999

    def test_returns_exactly_n_rows_when_df_larger(self):
        df = pd.DataFrame({"a": range(1000)})
        result = sample_df(df, n=50)
        assert len(result) == 50

    def test_is_reproducible(self):
        df = pd.DataFrame({"a": range(1000)})
        r1 = sample_df(df, n=50)
        r2 = sample_df(df, n=50)
        assert r1["a"].tolist() == r2["a"].tolist(), \
            "sample_df must be deterministic (fixed RANDOM_STATE)"

    def test_returns_subset_of_original(self):
        df = pd.DataFrame({"a": range(1000)})
        result = sample_df(df, n=50)
        assert set(result["a"]).issubset(set(df["a"]))


# sample_arr

class TestSampleArr:

    def test_returns_same_when_smaller_than_n(self):
        X = np.arange(30).reshape(10, 3)
        result = sample_arr(X, n=100)
        np.testing.assert_array_equal(result, X)

    def test_returns_n_rows(self):
        X = np.random.rand(1000, 10)
        result = sample_arr(X, n=50)
        assert result.shape == (50, 10)

    def test_preserves_column_count(self):
        X = np.random.rand(1000, 7)
        result = sample_arr(X, n=100)
        assert result.shape[1] == 7

    def test_is_reproducible(self):
        X = np.random.rand(1000, 5)
        r1 = sample_arr(X, n=50)
        r2 = sample_arr(X, n=50)
        np.testing.assert_array_equal(r1, r2)


# validate_columns

class TestValidateColumns:

    def test_passes_when_all_columns_present(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        # Should not raise
        validate_columns(df, ["a", "b"])

    def test_raises_on_single_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_columns(df, ["a", "z"])

    def test_raises_on_multiple_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError) as exc_info:
            validate_columns(df, ["x", "y", "z"])
        assert "x" in str(exc_info.value)
        assert "y" in str(exc_info.value)

    def test_error_message_names_missing_columns(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="MISSING_COL"):
            validate_columns(df, ["a", "MISSING_COL"])


# make_ohe

class TestMakeOhe:

    def test_returns_onehotencoder(self):
        from sklearn.preprocessing import OneHotEncoder
        ohe = make_ohe()
        assert isinstance(ohe, OneHotEncoder)

    def test_handle_unknown_is_ignore(self):
        ohe = make_ohe()
        assert ohe.handle_unknown == "ignore", \
            "OHE must silently ignore unseen categories at inference time"

    def test_dtype_is_float32(self):
        ohe = make_ohe()
        assert ohe.dtype == np.float32
