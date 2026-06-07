# Policing Equity — Unsupervised Learning Case Study

This repository contains a cleaned analysis pipeline for the IU Machine Learning — Unsupervised Learning and Feature Engineering case study, Task 2: Policing Equity.

## Goal

The goal is to reduce the complexity of a large policing-incident dataset, identify homogeneous groups of incidents, and create visualizations and descriptive cluster summaries that can support interpretation.

## Dataset

The analysis uses the Center for Policing Equity Kaggle dataset:

```text
cpe-data/Dept_49-00081/49-00081_Incident-Reports_2012_to_May_2015.csv
```

Place the Kaggle data folder in the project root so the path above exists, or change `DATA_PATH` in the notebook/script.

## Files

```text
policing_equity_clean.ipynb      Main cleaned notebook
policing_equity_pipeline.py      Script version of the same pipeline
requirements.txt                 Python dependencies
README_policing_equity.md        This file
```

## Method summary

1. Load and inspect the selected incident-report file.
2. Group variables into date/time, geographic, incident-type, outcome, and ID-like fields.
3. Clean text fields and convert latitude/longitude to numeric values.
4. Transform date/time fields into model-friendly features.
5. Engineer weekend, time-period, and rounded geographic-bin features.
6. Exclude record identifiers and full street addresses from clustering.
7. Use sparse one-hot encoding for categorical variables and scaling for numeric variables.
8. Apply TruncatedSVD for dimensionality reduction.
9. Use MiniBatch K-Means for clustering.
10. Export cluster summaries and report-ready figures.

## Why some columns are excluded

- `INCIDENT_UNIQUE_IDENTIFIER` is excluded from modelling because it only identifies records and does not describe incident similarity.
- `LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION` is excluded from modelling because it has very high cardinality and can dominate the encoded feature space. Geographic information is still represented through district, longitude, latitude, and rounded coordinate bins.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook policing_equity_clean.ipynb
```

Or run the script:

```bash
python policing_equity_pipeline.py
```

## Outputs

The pipeline creates an `outputs/` folder containing:

```text
column_summary.csv
feature_summary.csv
policing_equity_features.csv
svd_explained_variance.csv
svd_components.csv
cluster_evaluation.csv
cluster_counts.csv
numeric_cluster_summary.csv
final_cluster_summary_for_report.csv
policing_equity_clustered.csv
svd_2d_coordinates.csv
figures/
```

Recommended report figures:

```text
outputs/figures/svd_clusters_2d.png
outputs/figures/geographic_cluster_distribution.png
outputs/figures/incidents_per_cluster.png
outputs/figures/incident_reason_by_cluster.png
outputs/figures/disposition_by_cluster.png
```

## Important limitation

The clusters show recurring patterns in the incident data. They do not by themselves prove causality, discrimination, or individual-level bias. Any equity claim should be made cautiously and would require additional statistical analysis and contextual data.
