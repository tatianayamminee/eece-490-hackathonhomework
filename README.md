# Unknown Dataset Hackathon – automatic tabular ML

One program that takes an unfamiliar table, reads the task from its metadata
(classification, regression, or clustering with a given number of groups),
inspects the data, chooses its own preprocessing and model within a
60-second CPU budget, and writes predictions plus a structured data profile.

**Team:** Tatiana Yammine, Celena Saadeh, Thea Jelwan, Mohamad al habib ghazal. 

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.10+ (tested on 3.13)
pip install -r requirements.txt                      # pinned versions

# Standard entry point: a dataset directory holding train.csv, test.csv, metadata.json
python run.py path/to/dataset --out output/dataset_name

# Or explicit files / overrides
python run.py --train train.csv --test test.csv --meta metadata.json --out out/
python run.py --train data.csv --task clustering --n-clusters 4 --out out/
```

Outputs written to `--out`:

| file | content |
|---|---|
| `predictions.csv` | one row per test row (per train row when there is no test file): optional id column + `prediction` (class label, number, or cluster id 0..k-1) |
| `profile.json` | structured data profile (see below) |
| `run_report.json` | what was tried, cross-validation scores, chosen model, elapsed time |

### Reproduce our local evaluation

```bash
python tools/make_synthetic.py data/synthetic      # 8 messy synthetic datasets with hidden labels
python tools/check.py data/synthetic               # runs run.py on each and scores the outputs
python -m pytest tests -q                           # fast unit + end-to-end tests (~1 min)
```

## Input contract and assumptions

The official contract is released at kickoff. Everything contract-specific is
isolated in `automl/config.py` (file names, metadata keys, output file and
column names, time limit, internal selection metric), so adapting to the real
contract is a few constant changes, not code changes. Current assumptions:

* A dataset is a directory with a training table (`train.csv`, also
  `.parquet`/`.tsv`), an optional `test.csv`, and `metadata.json`.
* Metadata keys accepted (first match wins): task `task | task_type |
  problem_type`; target `target | target_column | label`; clusters
  `n_clusters | num_clusters | k`; id column `id | id_column`; feature types
  `feature_types | column_types` as `{col: type}`, `[{name, type}]` or
  `{type: [cols]}`. Task names are matched loosely (`binary`, `multiclass`,
  `clf`, `reg`, `cluster` ...).
* If the target is not named, it is the column present in train but not in
  test, else a column named `target/label/y/class`, else the last column.
* If the task is not named: clustering when there is no target, else
  classification when the target is non-numeric or has at most 20 distinct
  values, else regression.
* Declared feature types override our inference; the id column is passed
  through to the output and never used as a feature.
* Clustering: labels are produced for the test table if one exists, otherwise
  for the training table. If `n_clusters` is missing we choose k by silhouette.
* Metrics used for internal model selection: macro-F1 (classification, robust
  to imbalance) and RMSE (regression). These are only used to choose between
  candidate models; change `CLASSIFICATION_METRIC` / `REGRESSION_METRIC` in
  `config.py` if the published scoring uses something else (accuracy, ROC-AUC,
  log loss, MAE, R2 and RMSLE are implemented).
* A valid `predictions.csv` (majority class / median) and `profile.json` are
  written in the first seconds, so a timeout or a modelling crash never leaves
  the dataset without output. All writes are atomic.

## Approach

### 1. Inspect (`automl/profile.py`, `automl/preprocess.py`)

For every column we infer `numeric | categorical | datetime | text`, parsing
numbers stored as strings (`"1,234"`, `"$5"`, `"12%"`), recognising common
missing tokens (`?`, `NA`, `-`, `null`, ...), and detecting dates. We then flag

* constant columns (dropped),
* duplicate columns, exact or equal up to relabelling of values (dropped),
* id-like columns: unique integer sequences or uniquely-valued columns with an
  id-style name (dropped),
* missingness per column, scale spread between numeric columns, skewness,
  outlier fraction, high-cardinality categoricals, strongly correlated pairs,
  duplicate rows,
* for supervised tasks: class counts and an imbalance flag, target skew, and a
  mutual-information relevance score per feature, from which likely irrelevant
  features are listed.

All of this is written to `profile.json` (`summary`, per-column `columns`,
`target_info`, `feature_relevance`).

### 2. Prepare

Two preprocessing flavours are fitted on training data only and applied
identically to test data:

* **tree mode** (for gradient boosting / forests): numeric columns with median
  imputation plus missing indicators, categoricals as integer codes (rare
  levels and unseen levels share one code) plus frequency encoding, datetime
  columns expanded to year/month/day/weekday/hour/ordinal.
* **linear mode** (for linear models, SVM, kNN, clustering): the same, but
  categoricals one-hot encoded (up to 20 levels, otherwise frequency encoded),
  heavily skewed numerics log-transformed, values clipped at the 0.1/99.9
  percentiles, then standardised.

### 3. Model, under a time budget (`automl/supervised.py`)

Candidates are tried in order of expected value per second: LightGBM,
logistic regression / ridge, class-weighted variants when the target is
imbalanced, extra-trees, random forest, a slower LightGBM, a log-target
variant for skewed regression targets, and SVM / kNN on small data. Each is
scored with stratified 5-fold cross-validation (3-fold above 5k rows; model
selection uses a 20k-row subsample on larger data) producing out-of-fold
predictions. Before each candidate we project its cost from the first fold and
skip it when it would not fit in the remaining budget, always reserving time
for the final refit of the best model on all rows.

The out-of-fold predictions feed a greedy (Caruana-style) ensemble; it
replaces the best single model only when it improves the cross-validated
score. For classification under a macro-F1 style metric, per-class
probability scaling factors are tuned on the out-of-fold probabilities to
correct for imbalance. The chosen model(s) are refit on all training data.

### 4. Clustering (`automl/clustering.py`)

Irrelevant features are what usually breaks clustering, so on top of the
standard linear preprocessing we iteratively select features that actually
separate the clusters (between/within variance ratio of each feature under
the current k-means partition, re-clustering on the surviving features), and
down-weight one-hot blocks so a binary flag cannot dominate the distances.
Candidate partitions come from k-means on all features, on the selected
features and on PCA components, a Gaussian mixture (handles elongated,
unequal clusters) and Ward linkage. The final partition maximises a rank
average of silhouette on the selected features and agreement (adjusted Rand
index) with the other candidates. Cluster ids are renumbered by size.

### 5. Speed

Typical runtime on 4 CPU cores: 2 s for clustering, 20–35 s for supervised
datasets of a few thousand rows, ~45 s for 40–60k rows (the budget adapts to
the machine: slower CPUs simply try fewer candidates). The 60 s limit and a
10 s safety margin are in `config.py`.

## Results on our synthetic datasets

Measured on 4 CPU cores. Generated by `tools/make_synthetic.py` with missing values, mixed scales,
string-encoded numbers, categoricals with rare levels, constant / duplicate /
noise columns, id columns, class imbalance, nonlinear and skewed targets.

| dataset | rows (train / test) | task | result | time |
|---|---|---|---|---|
| clf_imbalanced (8 % positives) | 2 800 / 1 200 | classification | accuracy 0.937, macro-F1 0.781 | 23 s |
| clf_multiclass (4 classes, nonlinear) | 2 100 / 900 | classification | accuracy 0.866, macro-F1 0.866 | 34 s |
| clf_large (3 classes) | 42 000 / 18 000 | classification | accuracy 0.878, macro-F1 0.878 | 43 s |
| reg_nonlinear_skewed (Friedman, exp target, dates) | 2 450 / 1 050 | regression | R² 0.823 | 27 s |
| reg_linear_missing (up to 40 % missing, city effects) | 3 500 / 1 500 | regression | R² 0.696 (oracle LightGBM on true features: 0.668) | 23 s |
| reg_large (Friedman) | 35 000 / 15 000 | regression | R² 0.907 | 44 s |
| clu_blobs_noise (4 blobs, 6 noise dims) | 2 500 | clustering | ARI 1.00 | 2.5 s |
| clu_aniso (3 elongated, unequal groups) | 1 800 | clustering | ARI 1.00 | 2.4 s |

## Repository layout

```
run.py                 standard entry point (CLI)
automl/config.py       everything contract-specific: file names, keys, limits, metrics
automl/io.py           dataset loading, metadata parsing, atomic output writing
automl/profile.py      column type inference, duplicate/id detection, profile.json
automl/preprocess.py   tree / linear preprocessing
automl/supervised.py   time-budgeted candidate search, ensembling, class scaling
automl/clustering.py   feature selection, candidate partitions, selection
automl/pipeline.py     orchestration for one dataset
tools/make_synthetic.py  messy synthetic datasets with hidden labels
tools/check.py           local checker: runs and scores every dataset in a folder
tests/                   pytest suite
```

## Dependencies

Pinned in `requirements.txt`: numpy 2.2.6, pandas 2.2.3, scipy 1.15.3,
scikit-learn 1.6.1, lightgbm 4.6.0. CPU only; no network access needed at
run time.
