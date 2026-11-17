# ============================================================
# STEP 19 — RANDOM FOREST FEATURE IMPORTANCE STABILITY
# ============================================================

import os
import ast
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV
)


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
    "feature_importance_stability"
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
INNER_SPLITS = 4


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 19 — RANDOM FOREST FEATURE IMPORTANCE STABILITY")
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
# CROSS-VALIDATION
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

inner_cv = StratifiedGroupKFold(
    n_splits=INNER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "selector",
        SelectKBest(
            score_func=mutual_info_classif
        )
    ),

    (
        "model",
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    )
])


# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "selector__k": [
        10,
        20,
        40,
        "all"
    ],

    "model__n_estimators": [
        200,
        500
    ],

    "model__max_depth": [
        None,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]
}


# ============================================================
# STORAGE
# ============================================================

importance_records = []

selected_records = []

fold_summary = []


# ============================================================
# OUTER CV
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
    # TRAIN / TEST SPLIT
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

    groups_train = groups.iloc[
        train_idx
    ]

    groups_test = groups.iloc[
        test_idx
    ]


    # --------------------------------------------------------
    # INNER HYPERPARAMETER SEARCH
    # --------------------------------------------------------

    search = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        scoring="f1",

        cv=inner_cv,

        n_jobs=1,

        refit=True
    )


    search.fit(
        X_train,
        y_train,
        groups=groups_train
    )


    best_pipeline = (
        search.best_estimator_
    )


    # --------------------------------------------------------
    # EXTRACT SELECTOR
    # --------------------------------------------------------

    selector = (
        best_pipeline
        .named_steps["selector"]
    )

    model = (
        best_pipeline
        .named_steps["model"]
    )


    support = selector.get_support()

    selected_features = [
        feature_names[i]

        for i, selected in enumerate(
            support
        )

        if selected
    ]


    # --------------------------------------------------------
    # FEATURE IMPORTANCES
    # --------------------------------------------------------

    importances = (
        model.feature_importances_
    )


    # Sanity check
    if len(importances) != len(
        selected_features
    ):

        raise ValueError(
            "Feature importance length does not "
            "match selected feature count."
        )


    # --------------------------------------------------------
    # NORMALIZE IMPORTANCES
    # --------------------------------------------------------

    importance_sum = np.sum(
        importances
    )

    if importance_sum > 0:

        normalized_importances = (
            importances / importance_sum
        )

    else:

        normalized_importances = (
            importances
        )


    # --------------------------------------------------------
    # RECORD FOLD INFORMATION
    # --------------------------------------------------------

    fold_summary.append({

        "fold": fold_number,

        "n_train_subjects":
            groups_train.nunique(),

        "n_test_subjects":
            groups_test.nunique(),

        "n_selected_features":
            len(selected_features),

        "inner_best_f1":
            search.best_score_,

        "best_params":
            str(search.best_params_)
    })


    print(
        f"Selected features: "
        f"{len(selected_features)}"
    )

    print(
        f"Best inner F1: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"Best parameters: "
        f"{search.best_params_}"
    )


    # --------------------------------------------------------
    # SAVE SELECTED FEATURES
    # --------------------------------------------------------

    for rank, (
        feature,
        importance
    ) in enumerate(

        sorted(
            zip(
                selected_features,
                normalized_importances
            ),

            key=lambda x: x[1],

            reverse=True
        ),

        start=1
    ):

        selected_records.append({

            "fold": fold_number,

            "rank": rank,

            "feature": feature,

            "importance": importance
        })


    # --------------------------------------------------------
    # SAVE ALL SELECTED FEATURE IMPORTANCES
    # --------------------------------------------------------

    for feature, importance in zip(

        selected_features,

        normalized_importances

    ):

        importance_records.append({

            "fold": fold_number,

            "feature": feature,

            "importance": importance
        })


# ============================================================
# DATAFRAMES
# ============================================================

importance_df = pd.DataFrame(
    importance_records
)

selected_df = pd.DataFrame(
    selected_records
)

fold_df = pd.DataFrame(
    fold_summary
)


# ============================================================
# FEATURE STABILITY SUMMARY
# ============================================================

feature_summary = (

    importance_df

    .groupby("feature")

    .agg(

        selection_frequency=(
            "fold",
            "nunique"
        ),

        mean_importance=(
            "importance",
            "mean"
        ),

        std_importance=(
            "importance",
            "std"
        ),

        min_importance=(
            "importance",
            "min"
        ),

        max_importance=(
            "importance",
            "max"
        )
    )

    .reset_index()
)


# ============================================================
# SELECTION FREQUENCY
# ============================================================

feature_summary[
    "selection_frequency_pct"
] = (

    feature_summary[
        "selection_frequency"
    ]

    / OUTER_SPLITS

    * 100
)


# ============================================================
# MEAN RANK
# ============================================================

rank_summary = (

    selected_df

    .groupby("feature")

    .agg(

        mean_rank=(
            "rank",
            "mean"
        ),

        median_rank=(
            "rank",
            "median"
        )
    )

    .reset_index()
)


feature_summary = feature_summary.merge(

    rank_summary,

    on="feature",

    how="left"
)


# ============================================================
# SORT
# ============================================================

feature_summary = (

    feature_summary

    .sort_values(

        [
            "selection_frequency",
            "mean_importance"
        ],

        ascending=[
            False,
            False
        ]
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# STABLE FEATURES
# ============================================================

stable_features = feature_summary[
    feature_summary[
        "selection_frequency"
    ] == OUTER_SPLITS
].copy()


# ============================================================
# SAVE RESULTS
# ============================================================

importance_path = os.path.join(
    OUT_DIR,
    "rf_feature_importance_by_fold.csv"
)

selected_path = os.path.join(
    OUT_DIR,
    "rf_selected_features_by_fold.csv"
)

summary_path = os.path.join(
    OUT_DIR,
    "rf_feature_importance_stability.csv"
)

stable_path = os.path.join(
    OUT_DIR,
    "rf_stable_important_features.csv"
)

fold_path = os.path.join(
    OUT_DIR,
    "rf_fold_summary.csv"
)


importance_df.to_csv(
    importance_path,
    index=False
)

selected_df.to_csv(
    selected_path,
    index=False
)

feature_summary.to_csv(
    summary_path,
    index=False
)

stable_features.to_csv(
    stable_path,
    index=False
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE STABILITY")
print("=" * 70)

print()

print(
    feature_summary[
        [
            "feature",
            "selection_frequency",
            "selection_frequency_pct",
            "mean_importance",
            "std_importance",
            "mean_rank"
        ]
    ]

    .head(20)

    .to_string(
        index=False
    )
)


# ============================================================
# PRINT STABLE FEATURES
# ============================================================

print()
print("=" * 70)
print(
    f"FEATURES SELECTED IN ALL "
    f"{OUTER_SPLITS} OUTER FOLDS"
)
print("=" * 70)

if len(stable_features) == 0:

    print(
        "No feature was selected in "
        "all outer folds."
    )

else:

    print(
        stable_features[
            [
                "feature",
                "mean_importance",
                "std_importance",
                "mean_rank"
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
print("STEP 19 COMPLETE")
print("=" * 70)

print()
print("Saved:")

print(importance_path)
print(selected_path)
print(summary_path)
print(stable_path)
print(fold_path)