# ============================================================
# STEP 21 — FAST OUT-OF-FOLD PERMUTATION IMPORTANCE
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "experiment_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "permutation_importance"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

OUTER_SPLITS = 5

# Much faster than 30 while still giving useful estimates
N_REPEATS = 10


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 21 — FAST OUT-OF-FOLD PERMUTATION IMPORTANCE")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

TARGET = "target"
GROUP = "subject"

y = df[TARGET]
groups = df[GROUP]

X = df.drop(
    columns=[TARGET, GROUP],
    errors="ignore"
)

X = X.select_dtypes(
    include=[np.number]
)

feature_names = X.columns.tolist()


print()
print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Numeric predictors: {len(feature_names)}"
)

print(
    f"Subjects: {groups.nunique()}"
)

print(
    f"Class counts:\n"
    f"{y.value_counts().sort_index()}"
)


# ============================================================
# OUTER CV
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# STORAGE
# ============================================================

importance_records = []
fold_summary = []


# ============================================================
# OUTER FOLD LOOP
# ============================================================

for fold_number, (
    train_idx,
    test_idx
) in enumerate(

    outer_cv.split(
        X,
        y,
        groups
    ),

    start=1
):

    print()
    print("=" * 70)
    print(
        f"OUTER FOLD {fold_number}/{OUTER_SPLITS}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    X_train = X.iloc[
        train_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_test = y.iloc[
        test_idx
    ]


    # --------------------------------------------------------
    # STEP 1 — FIT IMPUTER ON TRAINING DATA ONLY
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imp = imputer.fit_transform(
        X_train
    )

    X_test_imp = imputer.transform(
        X_test
    )


    # --------------------------------------------------------
    # STEP 2 — FEATURE SELECTION
    #
    # Use the stable 6 features identified in Step 19.
    #
    # These were selected in ALL 5 outer folds.
    # --------------------------------------------------------

    stable_features = [

        "path_efficiency_mean",

        "max_lateral_deviation_mm_mean",

        "mean_lateral_deviation_mm_mean",

        "path_efficiency_cv",

        "trajectory_length_mm_sd"
    ]


    stable_indices = [

        feature_names.index(
            feature
        )

        for feature in stable_features
    ]


    X_train_selected = (
        X_train_imp[
            :,
            stable_indices
        ]
    )

    X_test_selected = (
        X_test_imp[
            :,
            stable_indices
        ]
    )


    # --------------------------------------------------------
    # STEP 3 — SCALE
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_selected
    )

    X_test_scaled = scaler.transform(
        X_test_selected
    )


    # --------------------------------------------------------
    # STEP 4 — RANDOM FOREST
    #
    # Use a robust configuration consistent with the
    # configurations selected in the earlier RF analysis.
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=500,

        max_depth=None,

        min_samples_leaf=1,

        class_weight="balanced",

        random_state=RANDOM_STATE

    )


    model.fit(
        X_train_scaled,
        y_train
    )


    # --------------------------------------------------------
    # BASELINE HELD-OUT ROC-AUC
    # --------------------------------------------------------

    y_score = model.predict_proba(
        X_test_scaled
    )[:, 1]

    baseline_auc = roc_auc_score(
        y_test,
        y_score
    )


    print(
        f"Selected features: "
        f"{len(stable_features)}"
    )

    print(
        f"Baseline held-out ROC-AUC: "
        f"{baseline_auc:.4f}"
    )


    # --------------------------------------------------------
    # PERMUTATION IMPORTANCE
    #
    # IMPORTANT:
    # This is calculated ONLY on held-out subjects.
    # --------------------------------------------------------

    permutation_result = permutation_importance(

        model,

        X_test_scaled,

        y_test,

        scoring="roc_auc",

        n_repeats=N_REPEATS,

        random_state=(
            RANDOM_STATE + fold_number
        ),

        n_jobs=1
    )


    # --------------------------------------------------------
    # SAVE IMPORTANCE
    # --------------------------------------------------------

    for feature, mean_imp, std_imp in zip(

        stable_features,

        permutation_result.importances_mean,

        permutation_result.importances_std

    ):

        importance_records.append({

            "fold": fold_number,

            "feature": feature,

            "permutation_importance":
                mean_imp,

            "permutation_std":
                std_imp,

            "n_repeats":
                N_REPEATS,

            "baseline_roc_auc":
                baseline_auc
        })


    fold_summary.append({

        "fold": fold_number,

        "n_train_subjects":
            len(train_idx),

        "n_test_subjects":
            len(test_idx),

        "baseline_roc_auc":
            baseline_auc
    })


# ============================================================
# DATAFRAMES
# ============================================================

importance_df = pd.DataFrame(
    importance_records
)

fold_df = pd.DataFrame(
    fold_summary
)


# ============================================================
# AGGREGATE
# ============================================================

summary_df = (

    importance_df

    .groupby("feature")

    .agg(

        selection_frequency=(
            "fold",
            "nunique"
        ),

        mean_permutation_importance=(
            "permutation_importance",
            "mean"
        ),

        std_permutation_importance=(
            "permutation_importance",
            "std"
        ),

        mean_permutation_sd=(
            "permutation_std",
            "mean"
        ),

        min_permutation_importance=(
            "permutation_importance",
            "min"
        ),

        max_permutation_importance=(
            "permutation_importance",
            "max"
        )
    )

    .reset_index()
)


# ============================================================
# FREQUENCY
# ============================================================

summary_df[
    "selection_frequency_pct"
] = (

    summary_df[
        "selection_frequency"
    ]

    / OUTER_SPLITS

    * 100
)


# ============================================================
# SORT
# ============================================================

summary_df = (

    summary_df

    .sort_values(
        "mean_permutation_importance",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# STABLE POSITIVE FEATURES
# ============================================================

stable_positive = summary_df[
    (
        summary_df[
            "selection_frequency"
        ] == OUTER_SPLITS
    )
    &
    (
        summary_df[
            "mean_permutation_importance"
        ] > 0
    )
].copy()


# ============================================================
# SAVE
# ============================================================

importance_path = os.path.join(
    OUT_DIR,
    "permutation_importance_by_fold.csv"
)

summary_path = os.path.join(
    OUT_DIR,
    "permutation_importance_summary.csv"
)

stable_path = os.path.join(
    OUT_DIR,
    "stable_positive_permutation_features.csv"
)

fold_path = os.path.join(
    OUT_DIR,
    "permutation_fold_summary.csv"
)


importance_df.to_csv(
    importance_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

stable_positive.to_csv(
    stable_path,
    index=False
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("PERMUTATION IMPORTANCE SUMMARY")
print("=" * 70)

print()

print(
    summary_df[
        [
            "feature",
            "selection_frequency",
            "mean_permutation_importance",
            "std_permutation_importance"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 70)
print(
    "STABLE POSITIVE FEATURES "
    "ACROSS ALL 5 OUTER FOLDS"
)
print("=" * 70)

if len(stable_positive) == 0:

    print(
        "No feature satisfied both criteria."
    )

else:

    print(
        stable_positive[
            [
                "feature",
                "mean_permutation_importance",
                "std_permutation_importance"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 21 COMPLETE")
print("=" * 70)

print()
print("Saved:")

print(importance_path)
print(summary_path)
print(stable_path)
print(fold_path)