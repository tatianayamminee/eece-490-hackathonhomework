"""Everything that depends on the official input/output contract lives here.

The kickoff materials define the exact file names, metadata keys and output
format. Until then these are best guesses; when the real contract arrives,
change the values below (not the modelling code).
"""

# ---------------------------------------------------------------- input files
# Candidate names tried, in order, inside a dataset directory.
TRAIN_FILES = ["train.csv", "train.parquet", "data.csv", "X_train.csv"]
TEST_FILES = ["test.csv", "test.parquet", "X_test.csv"]
META_FILES = ["metadata.json", "meta.json", "task.json", "info.json", "dataset.json"]

# Metadata keys (first one present wins).
META_KEYS = {
    "task": ["task", "task_type", "problem_type", "type"],
    "target": ["target", "target_column", "label", "label_column", "y"],
    "n_clusters": ["n_clusters", "num_clusters", "k", "n_groups", "num_groups"],
    "id": ["id", "id_column", "index_column", "row_id"],
    "feature_types": ["feature_types", "column_types", "features", "dtypes"],
}

# Values that map to each canonical task name.
TASK_ALIASES = {
    "classification": {"classification", "classify", "binary", "multiclass",
                       "binary_classification", "multiclass_classification", "clf"},
    "regression": {"regression", "regress", "reg"},
    "clustering": {"clustering", "cluster", "unsupervised"},
}

# Feature-type labels that mean "treat as categorical" / "numeric" / "datetime".
CATEGORICAL_TYPES = {"categorical", "category", "cat", "nominal", "ordinal", "string",
                     "object", "bool", "boolean", "text"}
NUMERIC_TYPES = {"numeric", "numerical", "number", "num", "continuous", "float",
                 "int", "integer", "real"}
DATETIME_TYPES = {"datetime", "date", "time", "timestamp"}

# --------------------------------------------------------------- output files
PREDICTIONS_FILE = "predictions.csv"
PROFILE_FILE = "profile.json"
PREDICTION_COLUMN = "prediction"   # name of the prediction column
WRITE_ID_COLUMN = True             # include the id column when one exists

# ---------------------------------------------------------------- time budget
TIME_LIMIT_SECONDS = 60.0
SAFETY_MARGIN_SECONDS = 10.0       # reserved for loading, profiling and writing

# -------------------------------------------------------------- metric guesses
# Used for internal model selection only. Swap once the scoring rules are known.
CLASSIFICATION_METRIC = "f1_macro"   # robust to class imbalance
REGRESSION_METRIC = "rmse"           # alternatives: "r2", "mae", "rmsle"
RANDOM_STATE = 42
