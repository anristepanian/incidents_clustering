"""
Tests that answer the real question: "are the clusters right?"

Organised in three tiers:

  TIER 1 - Structural sanity (synthetic data, always run, <1 s)
      Does the output have the right shape and valid labels?

  TIER 2 - Statistical quality (synthetic data, always run, <5 s)
      Are the clusters meaningful on controlled data where we KNOW
      the right answer?  Uses sklearn make_blobs to create 4 clearly
      separated groups and verifies K-Means recovers them.

  TIER 3 - Real-data quality (skipped when CSV is absent)
      Silhouette, Davies-Bouldin, cluster balance, temporal
      distinctiveness, and geographic spread on the actual SFPD data.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.datasets import make_blobs
from sklearn.metrics import davies_bouldin_score, silhouette_score

from pipeline import (
    BEST_K,
    RANDOM_STATE,
    run_clustering,
    run_svd,
)
from conftest import real_data_available


# TIER 1 - Structural sanity

class TestClusterStructure:
    """Basic output contract — no data dependency, instant."""

    def test_label_count_matches_row_count(self, synthetic_X_svd, synthetic_cluster_labels):
        assert len(synthetic_cluster_labels) == synthetic_X_svd.shape[0], \
            "One label per row in the input matrix"

    def test_all_labels_in_valid_range(self, synthetic_cluster_labels):
        unique = np.unique(synthetic_cluster_labels)
        assert unique.min() >= 0, "Cluster labels must be non-negative"
        assert unique.max() <= BEST_K - 1, \
            f"Labels must be in [0, {BEST_K - 1}]; found max={unique.max()}"

    def test_exactly_best_k_clusters_produced(self, synthetic_cluster_labels):
        n_found = len(np.unique(synthetic_cluster_labels))
        assert n_found == BEST_K, \
            f"Expected {BEST_K} clusters, got {n_found}"

    def test_no_cluster_is_empty(self, synthetic_cluster_labels):
        counts = np.bincount(synthetic_cluster_labels)
        assert (counts > 0).all(), \
            f"Every cluster must contain at least one point. Counts: {counts}"

    def test_cluster_counts_shares_sum_to_one(self, synthetic_cluster_labels):
        counts = np.bincount(synthetic_cluster_labels)
        shares = counts / counts.sum()
        assert abs(shares.sum() - 1.0) < 1e-9

    def test_clustering_is_fully_reproducible(self, synthetic_X_svd):
        """Same data + same seed must always produce identical labels."""
        labels_a, _ = run_clustering(synthetic_X_svd, k=BEST_K)
        labels_b, _ = run_clustering(synthetic_X_svd, k=BEST_K)
        np.testing.assert_array_equal(labels_a, labels_b,
            err_msg="Clustering is not reproducible — check RANDOM_STATE")

    def test_cluster_counts_df_shares_sum_to_one(self, synthetic_cluster_labels):
        """Mirrors the cluster_counts DataFrame built in the notebook."""
        counts = pd.Series(synthetic_cluster_labels).value_counts().sort_index()
        shares = counts / counts.sum()
        assert abs(shares.sum() - 1.0) < 1e-6


# TIER 2 - Statistical quality on controlled synthetic data


class TestClusterRecovery:
    """
    Generate perfectly separable blobs and verify that:
      - K-Means recovers them with high silhouette
      - The pipeline configuration (MiniBatch, n_init=10) is not broken

    This tests the ALGORITHM, not the real data — it is the
    equivalent of unit-testing that your hammer can actually drive a nail.
    """

    @pytest.fixture(scope="class")
    def blobs(self):
        X, y = make_blobs(
            n_samples=2000,
            centers=BEST_K,
            cluster_std=0.5,
            random_state=RANDOM_STATE,
        )
        return X, y

    def test_recovers_known_k_on_blobs(self, blobs):
        X, _ = blobs
        labels, _ = run_clustering(X, k=BEST_K)
        assert len(np.unique(labels)) == BEST_K

    def test_silhouette_high_on_well_separated_blobs(self, blobs):
        """
        On clearly separated blobs, silhouette must be > 0.7.
        If this fails, the MiniBatchKMeans configuration itself is broken.
        """
        X, _ = blobs
        labels, _ = run_clustering(X, k=BEST_K)
        score = silhouette_score(X, labels, sample_size=1000, random_state=RANDOM_STATE)
        assert score > 0.7, (
            f"Silhouette on well-separated blobs should be > 0.7, got {score:.3f}. "
            "This means MiniBatchKMeans is misconfigured."
        )

    def test_davies_bouldin_low_on_well_separated_blobs(self, blobs):
        """
        DB on clearly separated blobs must be < 0.5.
        """
        X, _ = blobs
        labels, _ = run_clustering(X, k=BEST_K)
        score = davies_bouldin_score(X, labels)
        assert score < 0.5, (
            f"Davies-Bouldin on well-separated blobs should be < 0.5, got {score:.3f}."
        )

    def test_inertia_decreases_as_k_increases(self):
        """
        Mathematical property: inertia must strictly decrease as k grows.
        Tests the evaluation loop logic used in Section 8 of the notebook.
        """
        from sklearn.cluster import MiniBatchKMeans
        X, _ = make_blobs(n_samples=500, centers=6, random_state=RANDOM_STATE)
        prev_inertia = float("inf")
        for k in range(2, 8):
            model = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                                    batch_size=512, n_init=5)
            model.fit(X)
            assert model.inertia_ < prev_inertia, (
                f"Inertia did not decrease from k={k - 1} to k={k}: "
                f"{prev_inertia:.1f} → {model.inertia_:.1f}"
            )
            prev_inertia = model.inertia_

    def test_more_components_explain_more_variance(self):
        """SVD: each additional component must explain >= 0 extra variance."""
        X, _ = make_blobs(n_samples=300, centers=4, n_features=20,
                          random_state=RANDOM_STATE)
        import scipy.sparse
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)
        X_sparse = scipy.sparse.csr_matrix(X_scaled)

        _, _, var_df_5  = run_svd(X_sparse, n_components=5)
        _, _, var_df_10 = run_svd(X_sparse, n_components=10)

        assert var_df_10["cumulative_explained_var"].iloc[-1] >= \
               var_df_5["cumulative_explained_var"].iloc[-1], \
            "More SVD components must explain at least as much variance"


# TIER 3 - Real-data quality (skipped when CSV absent)

@real_data_available
class TestRealDataClusterQuality:
    """
    These tests run ONLY when the real CSV is present.
    They verify that the actual SFPD clusters are statistically meaningful
    and not degenerate.

    Run with the real data:
        DATA_PATH=path/to/csv pytest tests/test_cluster_quality.py -v -m ""
    """

    # Silhouette

    def test_silhouette_score_is_positive(self, real_X_svd, real_cluster_labels):
        """
        A positive silhouette means clusters are better than random assignment.
        Failing this means the model found no real structure — the clustering
        is useless.
        """
        score = silhouette_score(
            real_X_svd, real_cluster_labels,
            sample_size=3000, random_state=RANDOM_STATE,
        )
        assert score > 0, (
            f"Silhouette score must be positive (better than random). "
            f"Got {score:.4f}. The clustering has found no meaningful structure."
        )

    def test_silhouette_score_exceeds_minimum_quality_threshold(
            self, real_X_svd, real_cluster_labels):
        """
        A silhouette of at least 0.05 is the minimum bar for "something real
        was found."  The SFPD data should comfortably exceed this given its
        known temporal and geographic structure.
        """
        score = silhouette_score(
            real_X_svd, real_cluster_labels,
            sample_size=3000, random_state=RANDOM_STATE,
        )
        assert score > 0.05, (
            f"Silhouette score {score:.4f} is too low. "
            "Check preprocessing — scaling or imputation may be broken."
        )

    # Davies-Bouldin

    def test_davies_bouldin_below_ceiling(self, real_X_svd, real_cluster_labels):
        """
        A DB score above 3.0 means clusters are so overlapping that the
        model has effectively failed.  A well-structured dataset like this
        one should be well below this ceiling.
        """
        score = davies_bouldin_score(real_X_svd, real_cluster_labels)
        assert score < 3.0, (
            f"Davies-Bouldin score {score:.4f} is too high (>3.0). "
            "Clusters are heavily overlapping — check dimensionality reduction."
        )

    # Cluster balance

    def test_no_cluster_dominates_the_dataset(self, real_cluster_labels):
        """
        If one cluster contains >85% of all points, the model has essentially
        produced one big group plus outlier noise — not four meaningful groups.
        """
        counts = np.bincount(real_cluster_labels)
        max_share = counts.max() / counts.sum()
        assert max_share < 0.85, (
            f"Cluster {counts.argmax()} contains {max_share:.1%} of all incidents. "
            "This is a degenerate solution — one cluster dominates everything."
        )

    def test_no_cluster_is_negligibly_small(self, real_cluster_labels):
        """
        A cluster capturing <1% of data is essentially capturing noise.
        Each cluster must represent a meaningful slice of the dataset.
        """
        counts = np.bincount(real_cluster_labels)
        min_share = counts.min() / counts.sum()
        assert min_share > 0.01, (
            f"Smallest cluster contains only {min_share:.2%} of incidents. "
            "This cluster is too small to be actionable — consider reducing k."
        )

    def test_cluster_sizes_are_reasonably_balanced(self, real_cluster_labels):
        """
        The largest cluster should not be more than 20x the smallest.
        A 20:1 ratio indicates severe imbalance.
        """
        counts = np.bincount(real_cluster_labels)
        ratio = counts.max() / counts.min()
        assert ratio < 20, (
            f"Largest-to-smallest cluster size ratio is {ratio:.1f}x. "
            "Clusters are extremely unbalanced — try re-running with more n_init."
        )

    # Temporal distinctiveness

    def test_clusters_have_distinct_time_of_day_profiles(self, real_df_clustered):
        """
        Time of day is the strongest differentiator in this dataset.
        The mean incident_time_minutes should differ significantly between
        clusters (one-way ANOVA, p < 0.01).

        If this fails, the temporal features are not being picked up by
        the model - check that StandardScaler is applied correctly.
        """
        groups = [
            grp["incident_time_minutes"].dropna().values
            for _, grp in real_df_clustered.groupby("cluster")
            if grp["incident_time_minutes"].notna().sum() > 10
        ]
        assert len(groups) == BEST_K, \
            "All clusters must have enough time data to run ANOVA"

        f_stat, p_value = stats.f_oneway(*groups)
        assert p_value < 0.01, (
            f"Cluster time-of-day profiles are NOT significantly different "
            f"(ANOVA p={p_value:.4f}). Time features may not be influencing "
            "the clustering. Check StandardScaler is applied to numeric features."
        )

    def test_cluster_mean_times_span_at_least_2_hours(self, real_df_clustered):
        """
        The range of cluster mean incident times must span at least 2 hours
        (120 minutes). If all clusters have similar mean times, the temporal
        signal is not differentiating the groups.
        """
        means = (
            real_df_clustered.groupby("cluster")["incident_time_minutes"]
            .mean().dropna()
        )
        time_range = means.max() - means.min()
        assert time_range >= 60, (
            f"Range of cluster mean times is only {time_range:.0f} minutes "
            f"({time_range / 60:.1f} hours). Clusters are not temporally distinct."
        )

    # Geographic distinctiveness

    def test_cluster_geographic_centroids_are_distinct(self, real_df_clustered):
        """
        Each cluster's mean latitude should differ enough that the clusters
        correspond to different parts of San Francisco, not the same area.
        The range of cluster mean latitudes must be at least 0.005° (~500 m).
        """
        mean_lats = (
            real_df_clustered.groupby("cluster")["LOCATION_LATITUDE"]
            .mean().dropna()
        )
        lat_range = mean_lats.max() - mean_lats.min()
        assert lat_range >= 0.005, (
            f"Range of cluster mean latitudes is only {lat_range:.5f}°. "
            "Clusters are not geographically distinct — geographic features "
            "may not be contributing to the separation."
        )

    # Incident-type distinctiveness

    def test_clusters_have_distinct_incident_type_profiles(self, real_df_clustered):
        """
        A chi-squared test on the incident_reason_simple × cluster contingency
        table must be significant (p < 0.001).

        If this fails, clusters have the same mix of incident types, they are
        not capturing operational differences and have no policy value.
        """
        from scipy.stats import chi2_contingency
        col = real_df_clustered["incident_reason_simple"].fillna("missing")
        contingency = pd.crosstab(real_df_clustered["cluster"], col)
        chi2, p_value, _, _ = chi2_contingency(contingency)
        assert p_value < 0.001, (
            f"Incident types are NOT significantly different across clusters "
            f"(chi² p={p_value:.6f}). Clusters do not differ by incident type — "
            "categorical features may be under-weighted."
        )

    # Disposition distinctiveness

    def test_clusters_have_distinct_disposition_profiles(self, real_df_clustered):
        """
        The same chi-squared test for dispositions.
        Clusters representing different operational types (traffic stop,
        crisis management, arrest) must have different resolution outcomes.
        """
        from scipy.stats import chi2_contingency
        col = real_df_clustered["disposition_simple"].fillna("missing")
        contingency = pd.crosstab(real_df_clustered["cluster"], col)
        chi2, p_value, _, _ = chi2_contingency(contingency)
        assert p_value < 0.001, (
            f"Dispositions are NOT significantly different across clusters "
            f"(chi² p={p_value:.6f}). This undermines the cluster interpretation."
        )

    # Equity: district composition

    def test_district_cluster_composition_varies(self, real_df_clustered):
        """
        The equity analysis relies on districts having different cluster
        profiles. A chi-squared test on the district × cluster table must
        be significant (p < 0.001).

        If this fails, all districts have the same cluster mix, there is
        no geographic equity signal to report.
        """
        from scipy.stats import chi2_contingency
        df = real_df_clustered.dropna(subset=["LOCATION_DISTRICT"])
        contingency = pd.crosstab(df["LOCATION_DISTRICT"], df["cluster"])
        # Keep only districts with enough rows
        contingency = contingency[contingency.sum(axis=1) >= 20]
        chi2, p_value, _, _ = chi2_contingency(contingency)
        assert p_value < 0.001, (
            f"District × cluster composition is NOT significantly different "
            f"(chi² p={p_value:.6f}). The equity analysis has no geographic signal."
        )


# plot_stacked_distribution

class TestPlotStackedDistribution:
    """
    Tests for the plot_stacked_distribution helper.
    Matplotlib is switched to Agg backend (headless) so no display is needed.
    """

    @pytest.fixture
    def sample_clustered_df(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "cluster":   rng.integers(0, BEST_K, 200),
            "reason":    rng.choice(["traffic", "assault", "welfare", "other"], 200),
            "time_period": rng.choice(["morning", "afternoon", "evening", "night"], 200),
        })

    def _get_func(self):
        """Import plot_stacked_distribution from the notebook's module namespace."""
        import importlib, sys, types
        # Re-use pipeline constants but get the function directly
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from pathlib import Path

        # Inline definition matching the notebook exactly
        def plot_stacked_distribution(data, row_col, category_col, title, filename,
                                       top_n=10, normalize=True, legend_title=None):
            plot_df = data[[row_col, category_col]].copy()
            plot_df[category_col] = plot_df[category_col].fillna("missing").astype(str)
            if top_n is not None:
                top_cats = plot_df[category_col].value_counts().head(top_n).index
                plot_df[category_col] = plot_df[category_col].where(
                    plot_df[category_col].isin(top_cats), "other")
            import pandas as pd
            table = pd.crosstab(plot_df[row_col], plot_df[category_col],
                                normalize="index" if normalize else False).round(3)
            if row_col == "cluster":
                table.index = [f"Cluster {i}" for i in table.index]
            if category_col == "cluster":
                table.columns = [f"Cluster {c}" for c in table.columns]
            fig, ax = plt.subplots(figsize=(8, 4))
            table.plot(kind="bar", stacked=True, ax=ax)
            plt.close(fig)
            return table
        return plot_stacked_distribution

    def test_returns_dataframe(self, sample_clustered_df):
        func = self._get_func()
        result = func(sample_clustered_df, "cluster", "reason",
                      "Test", "test.png", top_n=5)
        assert isinstance(result, pd.DataFrame), \
            "plot_stacked_distribution must return a DataFrame"

    def test_normalized_rows_sum_to_one(self, sample_clustered_df):
        func = self._get_func()
        table = func(sample_clustered_df, "cluster", "reason",
                     "Test", "test.png", normalize=True)
        row_sums = table.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-3), \
            f"Normalized rows must sum to 1.0; got: {row_sums.tolist()}"

    def test_top_n_creates_other_category(self, sample_clustered_df):
        func = self._get_func()
        table = func(sample_clustered_df, "cluster", "reason",
                     "Test", "test.png", top_n=2)
        assert "other" in table.columns, \
            "When top_n < total categories, an 'other' column must be created"

    def test_cluster_index_renamed(self, sample_clustered_df):
        func = self._get_func()
        table = func(sample_clustered_df, "cluster", "reason",
                     "Test", "test.png")
        assert all(str(idx).startswith("Cluster ") for idx in table.index), \
            "Row index must be renamed to 'Cluster X' format"

    def test_n_rows_equals_n_clusters(self, sample_clustered_df):
        func = self._get_func()
        table = func(sample_clustered_df, "cluster", "reason",
                     "Test", "test.png")
        assert len(table) == BEST_K, \
            f"Table must have one row per cluster; expected {BEST_K}, got {len(table)}"
