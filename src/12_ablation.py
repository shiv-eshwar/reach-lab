# ============================================================
# STEP 12 — ABLATION STUDY
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


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

STABLE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_stability",
    "stable_features.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ablation"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 12 — ABLATION STUDY")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

TARGET = "target"
GROUP = "subject"

y = df[TARGET]
groups = df[GROUP]


# ============================================================
# LOAD GLOBAL STABLE FEATURES
# ============================================================

# This file is printed for reference only.
# It is NOT used for evaluation.
# Using it directly would leak information from outer test subjects.

if os.path.exists(STABLE_PATH):

    stable_df = pd.read_csv(
        STABLE_PATH
    )

    global_stable_features = (
        stable_df["feature"].tolist()
    )

else:

    global_stable_features = []


print()
print(
    f"Dataset shape: {df.shape}"
)

print(
    f"Subjects: {groups.nunique()}"
)

print(
    f"Global stable features "
    f"(reference only): "
    f"{len(global_stable_features)}"
)

if len(global_stable_features) > 0:

    print()
    print("Global stable features:")
    
    for feature in global_stable_features:
        print(
            f"  {feature}"
        )


# ============================================================
# DEFINE REPRESENTATIONS
# ============================================================

# All numeric predictors
all_numeric = (
    df
    .drop(
        columns=[
            TARGET,
            GROUP
        ],
        errors="ignore"
    )
    .select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)


# ------------------------------------------------------------
# Representation 1 — Static
# ------------------------------------------------------------

# Static representation contains only mean descriptors.
# These represent average behavior across repetitions.

static_features = [
    c
    for c in all_numeric
    if c.endswith("_mean")
]


# ------------------------------------------------------------
# Representation 2 — Repetition-Aware
# ------------------------------------------------------------

# Full descriptor representation.
# Includes:
#   mean
#   sd
#   cv
#   slope
#   early_late_change
#   monotonicity

repetition_aware_features = (
    all_numeric.copy()
)


# ------------------------------------------------------------
# Representation 3 — Stable Features
# ------------------------------------------------------------

# Candidate pool is the complete numeric feature space.
#
# Stable features will be selected independently inside
# each outer training fold.
#
# Therefore the global stable_features.csv is NOT used
# for model evaluation.

stable_feature_candidates = (
    all_numeric.copy()
)


representations = {

    "Static":
        static_features,

    "Repetition_Aware":
        repetition_aware_features,

    "Stable_Features":
        stable_feature_candidates
}


# ============================================================
# PRINT REPRESENTATION SIZES
# ============================================================

print()
print("=" * 70)
print("REPRESENTATION SIZES")
print("=" * 70)

for name, features in representations.items():

    print(
        f"{name}: "
        f"{len(features)} features"
    )


# ============================================================
# CV SETTINGS
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# RESULTS STORAGE
# ============================================================

all_results = []


# ============================================================
# RUN ABLATION
# ============================================================

for representation_name, feature_list in representations.items():

    print()
    print("=" * 70)
    print(
        f"REPRESENTATION: "
        f"{representation_name}"
    )
    print("=" * 70)

    X = df[
        feature_list
    ]

    fold_metrics = []


    # ========================================================
    # OUTER CV
    # ========================================================

    for fold, (
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
        print(
            f"Outer fold "
            f"{fold}/5"
        )


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


        # ====================================================
        # STABLE FEATURES
        # FULLY NESTED BRANCH
        # ====================================================

        if representation_name == "Stable_Features":

            # ------------------------------------------------
            # Inner CV for model configuration
            # ------------------------------------------------

            stable_inner_cv = (
                StratifiedGroupKFold(
                    n_splits=4,
                    shuffle=True,
                    random_state=42
                )
            )


            model_configs = [

                {
                    "max_features": "sqrt",
                    "min_samples_leaf": 1
                },

                {
                    "max_features": "sqrt",
                    "min_samples_leaf": 2
                },

                {
                    "max_features": "sqrt",
                    "min_samples_leaf": 4
                },

                {
                    "max_features": "log2",
                    "min_samples_leaf": 1
                },

                {
                    "max_features": "log2",
                    "min_samples_leaf": 2
                },

                {
                    "max_features": "log2",
                    "min_samples_leaf": 4
                }

            ]


            inner_scores = []


            # ------------------------------------------------
            # Evaluate each RF configuration
            # ------------------------------------------------

            for config in model_configs:

                config_scores = []


                for (
                    inner_train_idx,
                    inner_val_idx
                ) in stable_inner_cv.split(
                    X_train,
                    y_train,
                    groups_train
                ):

                    X_inner_train = (
                        X_train.iloc[
                            inner_train_idx
                        ]
                    )

                    X_inner_val = (
                        X_train.iloc[
                            inner_val_idx
                        ]
                    )

                    y_inner_train = (
                        y_train.iloc[
                            inner_train_idx
                        ]
                    )

                    y_inner_val = (
                        y_train.iloc[
                            inner_val_idx
                        ]
                    )

                    groups_inner_train = (
                        groups_train.iloc[
                            inner_train_idx
                        ]
                    )


                    # ========================================
                    # STABILITY SELECTION
                    # ONLY INNER TRAINING SUBJECTS
                    # ========================================

                    stability_counts = {}


                    stability_cv = (
                        StratifiedGroupKFold(
                            n_splits=4,
                            shuffle=True,
                            random_state=42
                        )
                    )


                    for (
                        stab_train_idx,
                        stab_val_idx
                    ) in stability_cv.split(
                        X_inner_train,
                        y_inner_train,
                        groups_inner_train
                    ):

                        X_stab_train = (
                            X_inner_train.iloc[
                                stab_train_idx
                            ]
                        )

                        y_stab_train = (
                            y_inner_train.iloc[
                                stab_train_idx
                            ]
                        )


                        imputer_stab = (
                            SimpleImputer(
                                strategy="median"
                            )
                        )


                        X_stab_imp = (
                            imputer_stab.fit_transform(
                                X_stab_train
                            )
                        )


                        k_stab = min(
                            10,
                            X_stab_imp.shape[1]
                        )


                        selector_stab = (
                            SelectKBest(
                                score_func=
                                    mutual_info_classif,
                                k=k_stab
                            )
                        )


                        selector_stab.fit(
                            X_stab_imp,
                            y_stab_train
                        )


                        selected_features = (
                            X_inner_train
                            .columns[
                                selector_stab
                                .get_support()
                            ]
                        )


                        for feature in (
                            selected_features
                        ):

                            stability_counts[
                                feature
                            ] = (
                                stability_counts
                                .get(
                                    feature,
                                    0
                                )
                                + 1
                            )


                    # ========================================
                    # FEATURES SELECTED IN >= 3/4 FOLDS
                    # ========================================

                    stable_inner_features = [

                        feature

                        for feature, count
                        in stability_counts.items()

                        if count >= 3

                    ]


                    # Fallback if none survive
                    if (
                        len(
                            stable_inner_features
                        ) == 0
                    ):

                        stable_inner_features = (
                            list(
                                X_inner_train.columns
                            )
                        )


                    # ========================================
                    # FIT INNER MODEL
                    # ========================================

                    inner_pipeline = Pipeline([

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
                            "model",
                            RandomForestClassifier(

                                n_estimators=200,

                                max_depth=None,

                                max_features=
                                    config[
                                        "max_features"
                                    ],

                                min_samples_leaf=
                                    config[
                                        "min_samples_leaf"
                                    ],

                                random_state=42,

                                class_weight=
                                    "balanced"
                            )
                        )

                    ])


                    inner_pipeline.fit(

                        X_inner_train[
                            stable_inner_features
                        ],

                        y_inner_train

                    )


                    inner_prob = (
                        inner_pipeline
                        .predict_proba(

                            X_inner_val[
                                stable_inner_features
                            ]

                        )[:, 1]
                    )


                    config_scores.append(

                        roc_auc_score(
                            y_inner_val,
                            inner_prob
                        )

                    )


                # End inner folds


                inner_scores.append({

                    "config":
                        config,

                    "mean_auc":
                        np.mean(
                            config_scores
                        )

                })


            # End configurations


            # =================================================
            # SELECT BEST CONFIGURATION
            # =================================================

            best_config = max(

                inner_scores,

                key=lambda x:
                    x["mean_auc"]

            )["config"]


            # =================================================
            # FINAL STABILITY SELECTION
            # USING COMPLETE OUTER TRAINING SET
            # =================================================

            final_stability_counts = {}


            final_stability_cv = (
                StratifiedGroupKFold(
                    n_splits=4,
                    shuffle=True,
                    random_state=42
                )
            )


            for (
                stab_train_idx,
                stab_val_idx
            ) in final_stability_cv.split(

                X_train,
                y_train,
                groups_train

            ):

                X_stab_train = (
                    X_train.iloc[
                        stab_train_idx
                    ]
                )

                y_stab_train = (
                    y_train.iloc[
                        stab_train_idx
                    ]
                )


                imputer_stab = (
                    SimpleImputer(
                        strategy="median"
                    )
                )


                X_stab_imp = (
                    imputer_stab.fit_transform(
                        X_stab_train
                    )
                )


                k_stab = min(
                    10,
                    X_stab_imp.shape[1]
                )


                selector_stab = (
                    SelectKBest(
                        score_func=
                            mutual_info_classif,
                        k=k_stab
                    )
                )


                selector_stab.fit(
                    X_stab_imp,
                    y_stab_train
                )


                selected_features = (
                    X_train
                    .columns[
                        selector_stab
                        .get_support()
                    ]
                )


                for feature in (
                    selected_features
                ):

                    final_stability_counts[
                        feature
                    ] = (
                        final_stability_counts
                        .get(
                            feature,
                            0
                        )
                        + 1
                    )


            # ------------------------------------------------
            # Keep features selected in >= 3/4 folds
            # ------------------------------------------------

            fold_stable_features = [

                feature

                for feature, count
                in final_stability_counts.items()

                if count >= 3

            ]


            # Fallback
            if (
                len(
                    fold_stable_features
                ) == 0
            ):

                fold_stable_features = (
                    list(
                        X_train.columns
                    )
                )


            print(
                "Fold-local stable features: "
                f"{len(fold_stable_features)}"
            )


            # =================================================
            # FINAL OUTER MODEL
            # =================================================

            best_model = Pipeline([

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
                    "model",
                    RandomForestClassifier(

                        n_estimators=200,

                        max_depth=None,

                        max_features=
                            best_config[
                                "max_features"
                            ],

                        min_samples_leaf=
                            best_config[
                                "min_samples_leaf"
                            ],

                        random_state=42,

                        class_weight="balanced"

                    )
                )

            ])


            best_model.fit(

                X_train[
                    fold_stable_features
                ],

                y_train

            )


            # =================================================
            # OUTER TEST PREDICTION
            # =================================================

            y_pred = (
                best_model.predict(
                    X_test[
                        fold_stable_features
                    ]
                )
            )


            y_prob = (
                best_model
                .predict_proba(
                    X_test[
                        fold_stable_features
                    ]
                )[:, 1]
            )


            # =================================================
            # METRICS
            # =================================================

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            auc = roc_auc_score(
                y_test,
                y_prob
            )


            print(
                f"Accuracy: "
                f"{accuracy:.3f}"
            )

            print(
                f"F1: "
                f"{f1:.3f}"
            )

            print(
                f"ROC-AUC: "
                f"{auc:.3f}"
            )

            print(
                "Best params:",
                {
                    "model__max_depth":
                        None,

                    "model__max_features":
                        best_config[
                            "max_features"
                        ],

                    "model__min_samples_leaf":
                        best_config[
                            "min_samples_leaf"
                        ],

                    "model__n_estimators":
                        200
                }
            )


        # ====================================================
        # STATIC / REPETITION-AWARE
        # ====================================================

        else:

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
                        score_func=
                            mutual_info_classif
                    )
                ),

                (
                    "model",
                    RandomForestClassifier(
                        random_state=42,
                        class_weight="balanced"
                    )
                )

            ])


            # ------------------------------------------------
            # Inner CV
            # ------------------------------------------------

            inner_cv = (
                StratifiedGroupKFold(
                    n_splits=4,
                    shuffle=True,
                    random_state=42
                )
            )


            max_k = X_train.shape[1]


            if max_k < 10:

                k_values = [
                    max_k
                ]

            else:

                k_values = sorted(

                    set(

                        k

                        for k in [
                            10,
                            20,
                            40
                        ]

                        if k <= max_k

                    )

                )


            param_grid = {

                "selector__k":
                    k_values,

                "model__n_estimators":
                    [200],

                "model__max_depth":
                    [None],

                "model__max_features":
                    [
                        "sqrt",
                        "log2"
                    ],

                "model__min_samples_leaf":
                    [
                        1,
                        2,
                        4
                    ]

            }


            search = GridSearchCV(

                pipeline,

                param_grid,

                scoring="roc_auc",

                cv=inner_cv,

                n_jobs=1,

                refit=True

            )


            search.fit(

                X_train,

                y_train,

                groups=groups_train

            )


            best_model = (
                search.best_estimator_
            )


            # ------------------------------------------------
            # Test prediction
            # ------------------------------------------------

            y_pred = (
                best_model.predict(
                    X_test
                )
            )


            if hasattr(
                best_model,
                "predict_proba"
            ):

                y_prob = (
                    best_model
                    .predict_proba(
                        X_test
                    )[:, 1]
                )

            else:

                y_prob = (
                    best_model
                    .decision_function(
                        X_test
                    )
                )


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            auc = roc_auc_score(
                y_test,
                y_prob
            )


            print(
                f"Accuracy: "
                f"{accuracy:.3f}"
            )

            print(
                f"F1: "
                f"{f1:.3f}"
            )

            print(
                f"ROC-AUC: "
                f"{auc:.3f}"
            )

            print(
                "Best params:",
                search.best_params_
            )


        # ====================================================
        # STORE OUTER-FOLD RESULTS
        # ====================================================

        fold_metrics.append({

            "representation":
                representation_name,

            "fold":
                fold,

            "accuracy":
                accuracy,

            "f1":
                f1,

            "roc_auc":
                auc

        })


    # ========================================================
    # SUMMARY FOR REPRESENTATION
    # ========================================================

    fold_df = pd.DataFrame(
        fold_metrics
    )


    print()
    print(
        f"{representation_name} "
        f"SUMMARY"
    )


    print(
        f"Accuracy: "
        f"{fold_df['accuracy'].mean():.3f} ± "
        f"{fold_df['accuracy'].std():.3f}"
    )


    print(
        f"F1: "
        f"{fold_df['f1'].mean():.3f} ± "
        f"{fold_df['f1'].std():.3f}"
    )


    print(
        f"ROC-AUC: "
        f"{fold_df['roc_auc'].mean():.3f} ± "
        f"{fold_df['roc_auc'].std():.3f}"
    )


    all_results.extend(
        fold_metrics
    )


# ============================================================
# SAVE FOLD RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)


fold_results_path = os.path.join(
    OUT_DIR,
    "ablation_fold_results.csv"
)


results_df.to_csv(
    fold_results_path,
    index=False
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_df = (

    results_df

    .groupby(
        "representation"
    )

    .agg(

        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_std=(
            "accuracy",
            "std"
        ),

        f1_mean=(
            "f1",
            "mean"
        ),

        f1_std=(
            "f1",
            "std"
        ),

        roc_auc_mean=(
            "roc_auc",
            "mean"
        ),

        roc_auc_std=(
            "roc_auc",
            "std"
        )

    )

    .reset_index()

)


summary_path = os.path.join(
    OUT_DIR,
    "ablation_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("ABLATION SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("STEP 12 COMPLETE")
print("=" * 70)

print()
print("Results saved to:")

print(
    fold_results_path
)

print(
    summary_path
)