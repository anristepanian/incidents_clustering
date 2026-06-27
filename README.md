# Police Incidents Clustering

An unsupervised machine learning pipeline that isolates, compresses, and clusters
high-dimensional public safety logs from the Center for Policing Equity (CPE).
By combining **Truncated SVD** and **MiniBatch K-Means**, the pipeline identifies
four distinct operational signatures within ~315k incident records to help
municipal leaders understand policing patterns, optimize resource allocation, and
audit public safety equity across districts.

## Legend & Goals

 There has been a lot of concerns about policing equity in the local community. Driven by the political discourse, the
 local municipal administration has decided to quantitatively investigate this issue. For a couple of years, there has
 been a standardized data collection process with respect to policing activities in place. Until the present day, the
 dataset has grown to a considerable size and overlooking patterns in the data has become an intricate task.<br><br>
 In the first step, the goal was to investigate the provided dataset towards homogeneous categories of similar policing
 incidents. It was found that the geographic information is contained in the dataset and might be useful for the
 investigation of patterns within the data. The goal was to get a better overview over policing activities in the
 community which had to, ultimately, lift the political discussion to a more informed level.

## Dataset

Data Science for Good (Kaggle):

https://www.kaggle.com/datasets/center-for-policing-equity/data-science-for-good

The analysis targets the high-fidelity incident report file under `49-00081_Incident-Reports_2012_to_May_2015.csv`,
selected for its completeness and temporal coverage (2012 - mid-2015).

## Pipeline

In this use case, the most important was to reduce the complexity of the dataset which is, at first, hard to overlook and to provide visualizations which capture the main characteristics of the whole dataset.
For this purpose, different techniques were considered for dimensionality reduction. Also, there were provided insights into preferably homogeneous clusters of policing activities and visualizations, which allowed to interpret these clusters.
Finally, descriptive statistics about each group along with the number of incidents per cluster were provided.

### 1. Exploratory Data Analysis (`main.ipynb`)

Before any modeling, the raw data is explored to understand structure, assess
data quality, and form hypotheses about patterns clustering might reveal:

- Missing-value rates per column
- Top incident reasons and disposition distributions
- Temporal patterns (hour of day, day of week, monthly volume trend)
- Geographic scatter and per-district incident counts

### 2. Data Engineering & Feature Processing (`main.ipynb`)

- Dropped uninformative identifiers and high-cardinality street addresses to
  eliminate noise that would produce near-unique one-hot encodings.
- Preserved exact coordinate vectors (`LOCATION_LONGITUDE`, `LOCATION_LATITUDE`)
  plus coarsened geographic bins (0.01° ≈ 1 km) for neighborhood-level signal.
- Extracted temporal features: year, month, day of week, continuous time in
  minutes, binary weekend flag, and a four-period time-of-day label
  (night / morning / afternoon / evening).
- Encoded categorical attributes (`INCIDENT_REASON`, `DISPOSITION`, `LOCATION_DISTRICT`)
  into memory-efficient sparse one-hot matrices.
- Applied median imputation for numerics and most-frequent imputation for
  categoricals; StandardScaler applied to numeric features to prevent coordinate
  magnitudes from dominating Euclidean distance calculations.

### 3. Dimensionality Reduction (`main.ipynb`)

Two techniques are evaluated, they serve different purposes and are not
interchangeable:

| Technique | Role | Why |
|-----------|------|-----|
| **TruncatedSVD** | Clustering pipeline | Operates on sparse matrices directly; produces a stable dense representation that K-Means can use; can transform new data |
| **t-SNE** | Visualisation only | Non-linear; reveals local cluster separation better than SVD's first two components; cannot transform unseen data; applied to a 5,000-record sample only |

TruncatedSVD projects the high-dimensional sparse feature space down to 50
continuous components, preserving the majority of structural variance while
bypassing the curse of dimensionality.

### 4. Clustering & Hyperparameter Selection (`main.ipynb`)

An iterative evaluation loop (k = 2 to 10) is run using three complementary
metrics:

- **Silhouette score:** how well-separated clusters are globally (higher = better)
- **Davies-Bouldin index:** ratio of within-cluster scatter to between-cluster
  distance (lower = better)
- **Inertia / Elbow:** total within-cluster sum of squares (look for the elbow)

**Why k = 4:**  
The silhouette score peaks at k = 3, but the Davies-Bouldin index reaches its
minimum at k = 4. When these two metrics disagree, it typically means that one
of the k = 3 clusters is internally heterogeneous, it contains two genuinely
different subgroups that k = 4 correctly separates. The Davies-Bouldin index
detects this internal scatter; silhouette does not, because it only compares a
point to its nearest *other* cluster rather than examining within-cluster
compactness. The elbow plot confirms that the inertia drop from k = 3 to k = 4
is still meaningful, while gains beyond k = 4 diminish sharply. k = 4 therefore
produces clusters that are both statistically compact and practically
interpretable.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

1. Clone the repository.
2. Download the CPE dataset from Kaggle and place the target CSV at the path
   configured in `main.ipynb` (`DATA_PATH`).
3. Optionally run `db_explore.ipynb` to review the source-file filtering logic
   and choose a different department file.
4. Run the full pipeline:

```bash
jupyter nbconvert --to notebook --execute main.ipynb --output main_executed.ipynb
```

Or open interactively:

```bash
jupyter notebook main.ipynb
```

> [!NOTE]
> `policing_equity_clustered.csv` and the raw dataset are not pushed to this repository due to file size. Both are generated locally when you run the notebook.

## How to test

The test suite lives in `tests/` and is built with **pytest**. All functions from `main.ipynb` are extracted into 
`pipeline.py`, which both the notebook and the tests import from.

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

### Running the tests

```bash
# Run everything (fast tests only, ~2 s)
pytest tests/

# Run with the real dataset to unlock cluster quality tests
DATA_PATH=49-00081_Incident-Reports_2012_to_May_2015.csv pytest tests/

# Run a single file
pytest tests/test_helpers.py
pytest tests/test_cluster_quality.py

# Stop on first failure
pytest tests/ -x

# See coverage
pytest tests/ --cov=pipeline --cov-report=term-missing
```

### What is tested

The suite contains **102 tests** across three files:

| File | What it covers                                                                                                                                                                                                    | Needs CSV? |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| `test_helpers.py` | Every pure helper function (`clean_text`, `minutes_to_time`, `top_values`, `sample_df`, etc.), edge cases, null handling, output format                                                                           | No |
| `test_pipeline.py` | Full transformation chain: header row removal, temporal feature ranges, weekend flag logic, geographic bounds, text normalisation, sparse matrix output, StandardScaler zeroing, SVD shape and monotonic variance | No |
| `test_cluster_quality.py` | Whether the clusters are actually meaningful - split into three tiers (see below)                                                                                                                                 | Tier 3 only |

### Cluster quality tiers

**Tier 1: Structural sanity** *(always run, instant)*  
Labels in valid range, exactly k clusters produced, no empty cluster, full reproducibility with the same random seed.

**Tier 2: Algorithm correctness** *(always run, synthetic data)*  
Creates `make_blobs` with 4 clearly separated groups and asserts the pipeline recovers them: silhouette > 0.7, Davies-Bouldin < 0.5, inertia strictly decreasing as k grows. If this fails, the algorithm itself is broken, not just the data.

**Tier 3: Real data quality** *(skipped without CSV)*  
Runs on 10,000 records from the actual dataset and asserts:
- Silhouette > 0 (clusters are better than random)
- Davies-Bouldin < 3.0 (clusters are not heavily overlapping)
- No cluster captures > 85% of incidents (not a degenerate solution)
- No cluster captures < 1% of incidents (not capturing noise)
- ANOVA on time of day across clusters: p < 0.01
- Geographic centroid spread ≥ 500 m across clusters
- Chi-squared on incident type, disposition, and district: p < 0.001

## Outputs

The pipeline writes all results to an `outputs/` folder:

```text
outputs/
├── column_summary.csv                    # Per-column dtype, missing rate, cardinality
├── feature_summary.csv                   # Same for model features after engineering
├── svd_explained_variance.csv            # Explained variance per SVD component
├── svd_components.csv                    # Full SVD-transformed dataset (50 dims)
├── svd_2d_coordinates.csv                # First 2 SVD dims + cluster label
├── tsne_2d_coordinates.csv               # t-SNE 2D embedding (sample) (not pushed)
├── cluster_evaluation.csv                # Silhouette, Davies-Bouldin, inertia for k=2…10
├── cluster_counts.csv                    # Incident count and share per cluster
├── numeric_cluster_summary.csv           # Mean/median/std of numeric features per cluster
├── final_cluster_summary_for_report.csv  # Full cluster profiles (top values, averages)
├── policing_equity_clustered.csv         # Full dataset with cluster column (not pushed)
└── figures/
```

## Key Visualizations

### Cluster selection

**`cluster_evaluation_combined.png:`** All three metrics in one panel with the
chosen k marked. The silhouette score peaks at k = 3; the Davies-Bouldin index
reaches its minimum at k = 4. The elbow confirms k = 4 as the point of
diminishing inertia returns.

![Cluster Evaluation](outputs/figures/cluster_evaluation_combined.png)

---

### Dimensionality reduction comparison

**`dim_reduction_clusters_comparison.png:`** SVD and t-SNE projections of the
same incidents, colored by cluster. The t-SNE panel better reveals whether
clusters are genuinely separated; the SVD panel shows the linear axes used by
K-Means.

![Dimensionality Reduction Comparison](outputs/figures/dim_reduction_clusters_comparison.png)

---

### Cluster profiles

**`cluster_profile_heatmap.png:`** Absolute mean values and z-score-normalized
relative differences across clusters for all numeric features. Time of day is
the strongest differentiator; geographic coordinates reveal spatial concentration.

![Cluster Profile Heatmap](outputs/figures/cluster_profile_heatmap.png)

---

### Geographic distribution

**`geographic_cluster_distribution.png:`** Spatial polarization of clusters:
traffic enforcement concentrated along highways vs. crisis management clustered
in dense urban centres.

![Geographic Cluster Distribution](outputs/figures/geographic_cluster_distribution.png)

---

### Equity analysis

**`equity_cluster_share_by_district.png:`** For each district, the share of
incidents belonging to each cluster. Divergent rows indicate districts that
experience a qualitatively different policing profile from the rest of the
community.

![Equity Analysis](outputs/figures/equity_cluster_share_by_district.png)

## Cluster Conclusions

| Cluster | Label | Description |
|---------|-------|-------------|
| **0** | Administrative & Low-Friction Controls | Routine premise checks and minor incidents resolved quickly via scene dismissals or verbal warnings |
| **1** | High-Severity Interventions | Serious criminal violations and warrant executions resulting primarily in formal arrests and physical bookings |
| **2** | Routine Regulatory & Traffic Enforcement | Highly transactional, vehicle-code-driven stops resolved almost exclusively via field citations or summonses |
| **3** | Crisis & Medical Case Management | Public health incidents (mental health crises, overdoses, welfare checks) requiring psychiatric holds and medical handoffs rather than criminal processing |

**Equity note:** The cluster composition varies across districts (see
`equity_cluster_share_by_district.png`). Districts where Cluster 1 or Cluster 3
is disproportionately concentrated relative to the city-wide average may warrant
further investigation. These findings are descriptive and causal conclusions
regarding racial or socioeconomic disparities require demographic data not
present in this dataset.

## License

[MIT License](LICENSE)

## Authors

[Anri :cowboy_hat_face:](https://github.com/anristepanian)
