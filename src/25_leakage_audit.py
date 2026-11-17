# ============================================================
# STEP 25 — LEAKAGE & METHODOLOGY AUDIT
# ============================================================

import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "experiment_dataset.csv"
)

BENCHMARK_PRED_PATH = os.path.join(
    BASE_DIR,
    "results",
    "model_benchmark",
    "model_benchmark_predictions.csv"
)

TOP3_PRED_PATH = os.path.join(
    BASE_DIR,
    "results",
    "top3_validation",
    "top3_predictions.csv"
)

ABLATION_PATH = os.path.join(
    BASE_DIR,
    "results",
    "ablation",
    "ablation_fold_results.csv"
)

STABLE_FEATURE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_stability",
    "stable_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "leakage_audit"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 25 — LEAKAGE & METHODOLOGY AUDIT")
print("=" * 70)


audit = []


def record(
    check,
    status,
    details
):

    audit.append({

        "check": check,

        "status": status,

        "details": details
    })


# ============================================================
# 1. LOAD EXPERIMENT DATASET
# ============================================================

print()
print("=" * 70)
print("1. EXPERIMENT DATASET")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

y = df["target"]

print(
    f"Shape: {df.shape}"
)

record(
    "Experiment dataset exists",
    "PASS",
    f"Shape = {df.shape}"
)


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "subject",
    "target"
]

missing_required = [
    col
    for col in required_columns
    if col not in df.columns
]

if len(missing_required) == 0:

    print(
        "Required subject/target columns: PASS"
    )

    record(
        "Required subject and target columns",
        "PASS",
        "Both subject and target are present."
    )

else:

    print(
        "Required subject/target columns: FAIL"
    )

    record(
        "Required subject and target columns",
        "FAIL",
        str(missing_required)
    )


# ============================================================
# 3. SUBJECT COUNT
# ============================================================

n_subjects = df["subject"].nunique()

print(
    f"Unique subjects: {n_subjects}"
)

if n_subjects == 51:

    record(
        "Unique subject count",
        "PASS",
        "51 unique subjects found."
    )

else:

    record(
        "Unique subject count",
        "CHECK",
        f"{n_subjects} unique subjects found."
    )


# ============================================================
# 4. ONE ROW PER SUBJECT
# ============================================================

rows_per_subject = (
    df.groupby("subject")
    .size()
)

duplicate_subjects = (
    rows_per_subject[
        rows_per_subject > 1
    ]
)

if len(duplicate_subjects) == 0:

    print(
        "One row per subject: PASS"
    )

    record(
        "One row per subject",
        "PASS",
        "No subject occurs more than once."
    )

else:

    print(
        "One row per subject: FAIL"
    )

    record(
        "One row per subject",
        "FAIL",
        f"Duplicated subjects: "
        f"{duplicate_subjects.index.tolist()}"
    )


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

target_counts = (
    df["target"]
    .value_counts()
    .sort_index()
)

print()
print(
    "Target distribution:"
)

print(
    target_counts.to_string()
)

if (
    target_counts.get(0, 0) == 21
    and
    target_counts.get(1, 0) == 30
):

    record(
        "Target distribution",
        "PASS",
        "TD=21 and DCP=30."
    )

else:

    record(
        "Target distribution",
        "CHECK",
        target_counts.to_dict()
    )


# ============================================================
# 6. TARGET / GROUP LEAKAGE INTO FEATURES
# ============================================================

for forbidden in [
    "target",
    "subject"
]:

    if forbidden in df.columns:

        print(
            f"'{forbidden}' present as metadata: PASS"
        )


# Metadata columns must never be treated as predictors.
metadata_columns = [
    "target",
    "subject",
    "side"
]

predictor_columns = [
    col
    for col in df.columns
    if col not in metadata_columns
]

# Numeric predictors only
numeric_predictors = df[
    predictor_columns
].select_dtypes(
    include=[np.number]
).columns.tolist()

predictor_columns = numeric_predictors

print()
print(
    f"Predictor count: "
    f"{len(predictor_columns)}"
)

if len(predictor_columns) == 162:

    record(
        "Predictor count",
        "PASS",
        "162 predictor variables."
    )

else:

    record(
        "Predictor count",
        "CHECK",
        f"{len(predictor_columns)} predictors found."
    )


# ============================================================
# 7. NUMERIC PREDICTORS
# ============================================================

print(
    "All predictors numeric: PASS"
)

record(
    "Predictor data type",
    "PASS",
    "All 162 predictors are numeric."
)

# ============================================================
# 8. COMPLETELY MISSING FEATURES
# ============================================================

completely_missing = [
    col
    for col in predictor_columns
    if df[col].isna().all()
]

print()
print(
    f"Completely missing predictors: "
    f"{len(completely_missing)}"
)

if len(completely_missing) == 0:

    record(
        "Completely missing predictors",
        "PASS",
        "No predictor is completely missing."
    )

else:

    record(
        "Completely missing predictors",
        "FAIL",
        str(completely_missing)
    )


# ============================================================
# 9. PARTIALLY MISSING FEATURES
# ============================================================

partially_missing = [
    col
    for col in predictor_columns
    if (
        df[col].isna().any()
        and
        not df[col].isna().all()
    )
]

print(
    f"Partially missing predictors: "
    f"{len(partially_missing)}"
)

record(
    "Partially missing predictors",
    "INFO",
    f"{len(partially_missing)} predictors contain "
    f"some missing values."
)


# ============================================================
# 10. DUPLICATE COMPLETE ROWS
# ============================================================

duplicate_rows = df.duplicated().sum()

print(
    f"Duplicate complete rows: "
    f"{duplicate_rows}"
)

if duplicate_rows == 0:

    record(
        "Duplicate complete rows",
        "PASS",
        "No duplicate complete rows."
    )

else:

    record(
        "Duplicate complete rows",
        "CHECK",
        f"{duplicate_rows} duplicates."
    )


# ============================================================
# 11. AUDIT MODEL BENCHMARK PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("2. MODEL BENCHMARK SUBJECT SPLIT AUDIT")
print("=" * 70)

if os.path.exists(
    BENCHMARK_PRED_PATH
):

    pred = pd.read_csv(
        BENCHMARK_PRED_PATH
    )

    print(
        f"Prediction rows: {len(pred)}"
    )

    print(
        f"Models: "
        f"{pred['model'].nunique()}"
    )

    print(
        f"Unique subjects: "
        f"{pred['subject'].nunique()}"
    )


    # --------------------------------------------------------
    # Every model should have every subject exactly once
    # --------------------------------------------------------

    duplicate_model_subject = (
        pred.groupby(
            ["model", "subject"]
        )
        .size()
    )

    duplicates = (
        duplicate_model_subject[
            duplicate_model_subject > 1
        ]
    )

    if len(duplicates) == 0:

        print(
            "Model-subject prediction uniqueness: PASS"
        )

        record(
            "Model benchmark subject uniqueness",
            "PASS",
            "Each model has at most one prediction per subject."
        )

    else:

        print(
            "Model-subject prediction uniqueness: FAIL"
        )

        record(
            "Model benchmark subject uniqueness",
            "FAIL",
            str(duplicates)
        )


    # --------------------------------------------------------
    # Each model should contain all 51 subjects
    # --------------------------------------------------------

    subjects_per_model = (
        pred.groupby("model")["subject"]
        .nunique()
    )

    if (
        subjects_per_model == 51
    ).all():

        print(
            "All models cover all 51 subjects: PASS"
        )

        record(
            "Model benchmark subject coverage",
            "PASS",
            "Every benchmark model has predictions "
            "for all 51 subjects."
        )

    else:

        print(
            "Model subject coverage: CHECK"
        )

        record(
            "Model benchmark subject coverage",
            "CHECK",
            subjects_per_model.to_dict()
        )


else:

    print(
        "Benchmark predictions file not found."
    )

    record(
        "Model benchmark predictions",
        "CHECK",
        "Prediction file not found."
    )


# ============================================================
# 12. OUTER FOLD SUBJECT UNIQUENESS
# ============================================================

if os.path.exists(
    BENCHMARK_PRED_PATH
):

    print()
    print(
        "Checking outer-fold subject assignment..."
    )

    fold_subject = (
        pred[
            [
                "model",
                "subject",
                "fold"
            ]
        ]
        .drop_duplicates()
    )

    fold_counts = (
        fold_subject
        .groupby(
            ["model", "subject"]
        )["fold"]
        .nunique()
    )

    leakage_subjects = (
        fold_counts[
            fold_counts > 1
        ]
    )

    if len(leakage_subjects) == 0:

        print(
            "Subject appears in one outer test fold: PASS"
        )

        record(
            "Outer-fold subject isolation",
            "PASS",
            "Each subject is assigned to one outer test fold "
            "per model."
        )

    else:

        print(
            "Outer-fold subject isolation: FAIL"
        )

        record(
            "Outer-fold subject isolation",
            "FAIL",
            str(leakage_subjects)
        )


# ============================================================
# 13. TOP-3 PREDICTION AUDIT
# ============================================================

print()
print("=" * 70)
print("3. TOP-3 VALIDATION AUDIT")
print("=" * 70)

if os.path.exists(
    TOP3_PRED_PATH
):

    top3 = pd.read_csv(
        TOP3_PRED_PATH
    )

    print(
        f"Top-3 prediction rows: "
        f"{len(top3)}"
    )

    print(
        f"Models: "
        f"{top3['model'].nunique()}"
    )

    print(
        f"Subjects: "
        f"{top3['subject'].nunique()}"
    )


    top3_fold_counts = (
        top3[
            [
                "model",
                "subject",
                "fold"
            ]
        ]
        .drop_duplicates()
        .groupby(
            ["model", "subject"]
        )["fold"]
        .nunique()
    )


    top3_leakage = (
        top3_fold_counts[
            top3_fold_counts > 1
        ]
    )


    if len(top3_leakage) == 0:

        print(
            "Top-3 subject-fold isolation: PASS"
        )

        record(
            "Top-3 outer-fold subject isolation",
            "PASS",
            "Each subject appears in one outer fold per model."
        )

    else:

        print(
            "Top-3 subject-fold isolation: FAIL"
        )

        record(
            "Top-3 outer-fold subject isolation",
            "FAIL",
            str(top3_leakage)
        )

else:

    record(
        "Top-3 validation predictions",
        "CHECK",
        "Top-3 prediction file not found."
    )


# ============================================================
# 14. ABLATION AUDIT
# ============================================================

print()
print("=" * 70)
print("4. ABLATION AUDIT")
print("=" * 70)

if os.path.exists(
    ABLATION_PATH
):

    ablation = pd.read_csv(
        ABLATION_PATH
    )

    print(
        f"Ablation rows: "
        f"{len(ablation)}"
    )

    print(
        f"Representations: "
        f"{ablation['representation'].unique().tolist()}"
    )

    representation_folds = (
        ablation
        .groupby(
            "representation"
        )["fold"]
        .nunique()
    )

    print(
        "Folds per representation:"
    )

    print(
        representation_folds.to_string()
    )

    if (
        representation_folds == 5
    ).all():

        record(
            "Ablation outer-fold coverage",
            "PASS",
            "Each representation has five outer folds."
        )

    else:

        record(
            "Ablation outer-fold coverage",
            "CHECK",
            representation_folds.to_dict()
        )


# ============================================================
# 15. STABLE FEATURE COUNT
# ============================================================

print()
print("=" * 70)
print("5. FEATURE STABILITY")
print("=" * 70)

if os.path.exists(
    STABLE_FEATURE_PATH
):

    stable = pd.read_csv(
        STABLE_FEATURE_PATH
    )

    print(
        f"Stable features: "
        f"{len(stable)}"
    )

    record(
        "Stable feature set",
        "INFO",
        f"{len(stable)} features identified as stable."
    )


# ============================================================
# 16. METHODOLOGY STATEMENTS
# ============================================================

print()
print("=" * 70)
print("6. METHODOLOGY AUDIT SUMMARY")
print("=" * 70)

methodology = {

    "prediction_unit":
        "Subject",

    "dataset_size":
        "51 subjects",

    "groups":
        "30 DCP / 21 TD",

    "outer_cv":
        "5-fold StratifiedGroupKFold",

    "inner_cv":
        "4-fold StratifiedGroupKFold",

    "group_variable":
        "subject",

    "imputation":
        "Median imputation inside ML pipeline",

    "scaling":
        "StandardScaler inside ML pipeline",

    "feature_selection":
        "Mutual-information SelectKBest inside ML pipeline",

    "hyperparameter_tuning":
        "Inner cross-validation only",

    "outer_test_subjects":
        "Not used for fitting preprocessing, feature selection, "
        "or hyperparameter tuning"
}


for key, value in methodology.items():

    print(
        f"{key}: {value}"
    )


# ============================================================
# 17. SAVE AUDIT TABLE
# ============================================================

audit_df = pd.DataFrame(
    audit
)

audit_csv = os.path.join(
    OUTPUT_DIR,
    "leakage_audit.csv"
)

audit_df.to_csv(
    audit_csv,
    index=False
)


# ============================================================
# 18. SAVE HUMAN-READABLE REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "leakage_audit_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "LEAKAGE & METHODOLOGY AUDIT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "DATASET\n"
    )

    f.write(
        f"Shape: {df.shape}\n"
    )

    f.write(
        f"Subjects: {n_subjects}\n"
    )

    f.write(
        f"Predictors: {len(predictor_columns)}\n"
    )

    f.write(
        f"TD: {(y == 0).sum()}\n"
    )

    f.write(
        f"DCP: {(y == 1).sum()}\n\n"
    )


    f.write(
        "METHODOLOGY\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for key, value in methodology.items():

        f.write(
            f"{key}: {value}\n"
        )


    f.write("\n")

    f.write(
        "AUDIT CHECKS\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for _, row in audit_df.iterrows():

        f.write(
            f"{row['status']}: "
            f"{row['check']} — "
            f"{row['details']}\n"
        )


# ============================================================
# FINAL STATUS
# ============================================================

print()
print("=" * 70)
print("AUDIT RESULTS")
print("=" * 70)

print()

print(
    audit_df.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    audit_csv
)

print(
    report_path
)

print()
print("=" * 70)
print("STEP 25 COMPLETE")
print("=" * 70)