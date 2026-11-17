# ============================================================
# STEP 20 — OUT-OF-FOLD SHAP EXPLAINABILITY
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

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
    "shap_oof"
)

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

OUTER_SPLITS = 5
INNER_SPLITS = 4

TOP_FEATURES = 15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 20 — OUT-OF-FOLD SHAP EXPLAINABILITY")
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

shap_records = []

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
    # TRAIN / TEST DATA
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

    groups_test = groups.iloc[
        test_idx
    ]

    groups_train = groups.iloc[
        train_idx
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
    # EXTRACT FITTED STEPS
    # --------------------------------------------------------

    imputer = (
        best_pipeline
        .named_steps["imputer"]
    )

    scaler = (
        best_pipeline
        .named_steps["scaler"]
    )

    selector = (
        best_pipeline
        .named_steps["selector"]
    )

    model = (
        best_pipeline
        .named_steps["model"]
    )


    # --------------------------------------------------------
    # TRANSFORM HELD-OUT DATA
    # --------------------------------------------------------

    X_test_imputed = (
        imputer.transform(
            X_test
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_imputed
        )
    )

    X_test_selected = (
        selector.transform(
            X_test_scaled
        )
    )


    # --------------------------------------------------------
    # SELECTED FEATURE NAMES
    # --------------------------------------------------------

    support = selector.get_support()

    selected_features = [

        feature_names[i]

        for i, selected in enumerate(
            support
        )

        if selected
    ]


    # --------------------------------------------------------
    # SHAP EXPLAINER
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )


    shap_values = explainer.shap_values(
        X_test_selected
    )


    # --------------------------------------------------------
    # HANDLE SHAP OUTPUT FORMAT
    # --------------------------------------------------------

    # Different SHAP versions can return:
    #
    #   list[class_0_values, class_1_values]
    #
    # or:
    #
    #   ndarray(n_samples, n_features)
    #
    # or:
    #
    #   ndarray(n_samples, n_features, n_classes)
    #
    # We need the positive-class explanation.

    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) == 2:

            shap_positive = (
                shap_values[1]
            )

        else:

            shap_positive = (
                shap_values[-1]
            )

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 2:

            shap_positive = (
                shap_array
            )

        elif shap_array.ndim == 3:

            # Usually:
            # samples × features × classes

            shap_positive = (
                shap_array[:, :, 1]
            )

        else:

            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{shap_array.shape}"
            )


    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    if shap_positive.shape != (
        X_test_selected.shape
    ):

        raise ValueError(
            "SHAP shape mismatch.\n"
            f"SHAP: {shap_positive.shape}\n"
            f"Data: {X_test_selected.shape}"
        )


    # --------------------------------------------------------
    # SAVE OOF SHAP VALUES
    # --------------------------------------------------------

    for row_idx in range(
        len(X_test_selected)
    ):

        subject = groups_test.iloc[
            row_idx
        ]

        actual = y_test.iloc[
            row_idx
        ]


        for feature_idx, feature in enumerate(
            selected_features
        ):

            shap_records.append({

                "fold": fold_number,

                "subject": subject,

                "true_target": actual,

                "feature": feature,

                "feature_value": (
                    X_test_selected[
                        row_idx,
                        feature_idx
                    ]
                ),

                "shap_value": (
                    shap_positive[
                        row_idx,
                        feature_idx
                    ]
                ),

                "abs_shap_value": abs(
                    shap_positive[
                        row_idx,
                        feature_idx
                    ]
                )
            })


    # --------------------------------------------------------
    # FOLD SUMMARY
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
        f"Held-out subjects: "
        f"{len(X_test)}"
    )

    print(
        f"Best inner F1: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"Best parameters: "
        f"{search.best_params_}"
    )


# ============================================================
# DATAFRAME
# ============================================================

shap_df = pd.DataFrame(
    shap_records
)

fold_df = pd.DataFrame(
    fold_summary
)


# ============================================================
# AGGREGATE SHAP IMPORTANCE
# ============================================================

shap_summary = (

    shap_df

    .groupby("feature")

    .agg(

        n_subjects=(
            "subject",
            "nunique"
        ),

        selection_frequency=(
            "fold",
            "nunique"
        ),

        mean_abs_shap=(
            "abs_shap_value",
            "mean"
        ),

        median_abs_shap=(
            "abs_shap_value",
            "median"
        ),

        std_abs_shap=(
            "abs_shap_value",
            "std"
        ),

        mean_shap=(
            "shap_value",
            "mean"
        ),

        median_shap=(
            "shap_value",
            "median"
        )
    )

    .reset_index()
)


# ============================================================
# SHAP SELECTION FREQUENCY
# ============================================================

shap_summary[
    "selection_frequency_pct"
] = (

    shap_summary[
        "selection_frequency"
    ]

    / OUTER_SPLITS

    * 100
)


# ============================================================
# SORT BY SHAP IMPORTANCE
# ============================================================

shap_summary = (

    shap_summary

    .sort_values(
        "mean_abs_shap",
        ascending=False
    )

    .reset_index(
        drop=True
    )
)


# ============================================================
# SAVE
# ============================================================

shap_values_path = os.path.join(
    OUT_DIR,
    "oof_shap_values.csv"
)

shap_summary_path = os.path.join(
    OUT_DIR,
    "oof_shap_summary.csv"
)

fold_path = os.path.join(
    OUT_DIR,
    "shap_fold_summary.csv"
)

shap_df.to_csv(
    shap_values_path,
    index=False
)

shap_summary.to_csv(
    shap_summary_path,
    index=False
)

fold_df.to_csv(
    fold_path,
    index=False
)


# ============================================================
# TOP FEATURES
# ============================================================

top_features = shap_summary.head(
    TOP_FEATURES
)

print()
print("=" * 70)
print("TOP SHAP FEATURES")
print("=" * 70)

print(
    top_features[
        [
            "feature",
            "selection_frequency",
            "mean_abs_shap",
            "std_abs_shap",
            "mean_shap"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FIGURE 8 — SHAP IMPORTANCE
# ============================================================

plot_df = (
    shap_summary
    .head(TOP_FEATURES)
    .sort_values(
        "mean_abs_shap"
    )
)

plt.figure(
    figsize=(9, 7)
)

plt.barh(
    plot_df["feature"],
    plot_df["mean_abs_shap"]
)

plt.xlabel(
    "Mean |SHAP value|"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Out-of-Fold SHAP Feature Importance — Random Forest"
)

plt.tight_layout()

importance_figure_path = os.path.join(
    FIGURE_DIR,
    "Figure_8_SHAP_importance.png"
)

plt.savefig(
    importance_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 9 — SHAP DISTRIBUTION / BEESWARM-LIKE PLOT
# ============================================================

top_feature_names = (
    top_features[
        "feature"
    ].tolist()
)

beeswarm_df = shap_df[
    shap_df["feature"].isin(
        top_feature_names
    )
].copy()


# Order features by mean absolute SHAP
feature_order = (
    top_features
    .sort_values(
        "mean_abs_shap"
    )["feature"]
    .tolist()
)

feature_to_y = {

    feature: idx

    for idx, feature in enumerate(
        feature_order
    )
}


plt.figure(
    figsize=(10, 8)
)

rng = np.random.default_rng(
    RANDOM_STATE
)

for feature in feature_order:

    feature_data = beeswarm_df[
        beeswarm_df["feature"] == feature
    ]

    y_base = feature_to_y[
        feature
    ]

    jitter = rng.normal(
        0,
        0.08,
        size=len(feature_data)
    )

    plt.scatter(

        feature_data[
            "shap_value"
        ],

        np.full(
            len(feature_data),
            y_base
        ) + jitter,

        s=25,

        alpha=0.65
    )


plt.axvline(
    0,
    linestyle="--",
    linewidth=1
)

plt.yticks(
    range(
        len(feature_order)
    ),
    feature_order
)

plt.xlabel(
    "SHAP value"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Out-of-Fold SHAP Contribution Distribution — Random Forest"
)

plt.tight_layout()

beeswarm_figure_path = os.path.join(
    FIGURE_DIR,
    "Figure_9_SHAP_beeswarm.png"
)

plt.savefig(
    beeswarm_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 20 COMPLETE")
print("=" * 70)

print()
print("Saved SHAP data:")

print(
    shap_values_path
)

print(
    shap_summary_path
)

print(
    fold_path
)

print()
print("Saved figures:")

print(
    importance_figure_path
)

print(
    beeswarm_figure_path
)