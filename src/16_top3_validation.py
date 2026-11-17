# ============================================================
# STEP 16 — TOP-3 MODEL VALIDATION
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix
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
    "top3_validation"
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
print("STEP 16 — TOP-3 MODEL VALIDATION")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)

print()
print(
    f"Dataset shape: {df.shape}"
)


TARGET = "target"
GROUP = "subject"


y = df[TARGET]
groups = df[GROUP]

X = df.drop(
    columns=[TARGET, GROUP],
    errors="ignore"
)

# Keep numeric predictors only
X = X.select_dtypes(
    include=[np.number]
)

feature_names = X.columns.tolist()

print(
    f"Predictor count: {len(feature_names)}"
)

print(
    f"Subjects: {groups.nunique()}"
)

print(
    f"Class counts:\n{y.value_counts().sort_index()}"
)


# ============================================================
# OUTER / INNER CV
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
# MODEL DEFINITIONS
# ============================================================

models = {

    "Random_Forest": {

        "model": RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "params": {

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
    },


    "Extra_Trees": {

        "model": ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "params": {

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
    },


    "Linear_SVM": {

        "model": LinearSVC(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "params": {

            "selector__k": [
                10,
                20,
                40,
                "all"
            ],

            "model__C": [
                0.01,
                0.1,
                1,
                10
            ]
        }
    }
}


# ============================================================
# STORAGE
# ============================================================

all_results = []
all_predictions = []
all_params = []


# ============================================================
# RUN MODELS
# ============================================================

for model_name, config in models.items():

    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    model = config["model"]

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
            model
        )
    ])

    fold_number = 0

    for train_idx, test_idx in outer_cv.split(
        X,
        y,
        groups
    ):

        fold_number += 1

        print(
            f"\nOuter fold {fold_number}/{OUTER_SPLITS}"
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

        groups_test = groups.iloc[
            test_idx
        ]


        # ----------------------------------------------------
        # INNER CV HYPERPARAMETER SEARCH
        # ----------------------------------------------------

        search = GridSearchCV(

            estimator=pipeline,

            param_grid=config["params"],

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


        best_model = search.best_estimator_


        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        y_pred = best_model.predict(
            X_test
        )


        # LinearSVC has decision_function
        if hasattr(
            best_model,
            "predict_proba"
        ):

            y_score = best_model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_score = best_model.decision_function(
                X_test
            )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            y_score
        )


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        tn, fp, fn, tp = cm.ravel()


        specificity = (

            tn / (tn + fp)

            if (tn + fp) > 0

            else np.nan
        )


        # ----------------------------------------------------
        # SAVE FOLD RESULT
        # ----------------------------------------------------

        all_results.append({

            "model": model_name,

            "fold": fold_number,

            "n_train_subjects":
                len(np.unique(groups_train)),

            "n_test_subjects":
                len(np.unique(groups_test)),

            "accuracy": accuracy,

            "f1": f1,

            "precision": precision,

            "recall_sensitivity": recall,

            "specificity": specificity,

            "roc_auc": roc_auc
        })


        # ----------------------------------------------------
        # SAVE PREDICTIONS
        # ----------------------------------------------------

        for subject, actual, predicted, score in zip(

            groups_test,

            y_test,

            y_pred,

            y_score

        ):

            all_predictions.append({

                "model": model_name,

                "fold": fold_number,

                "subject": subject,

                "true_target": actual,

                "predicted_target": predicted,

                "predicted_score": score
            })


        # ----------------------------------------------------
        # SAVE BEST PARAMETERS
        # ----------------------------------------------------

        all_params.append({

            "model": model_name,

            "fold": fold_number,

            "best_params":
                str(search.best_params_),

            "best_inner_f1":
                search.best_score_
        })


        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"F1: {f1:.4f}"
        )

        print(
            f"ROC-AUC: {roc_auc:.4f}"
        )

        print(
            f"Specificity: {specificity:.4f}"
        )


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    all_results
)

predictions_df = pd.DataFrame(
    all_predictions
)

params_df = pd.DataFrame(
    all_params
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = (

    results_df

    .groupby("model")

    .agg({

        "accuracy": [
            "mean",
            "std"
        ],

        "f1": [
            "mean",
            "std"
        ],

        "precision": [
            "mean",
            "std"
        ],

        "recall_sensitivity": [
            "mean",
            "std"
        ],

        "specificity": [
            "mean",
            "std"
        ],

        "roc_auc": [
            "mean",
            "std"
        ]
    })

)

summary_df.columns = [

    "_".join(col)

    for col in summary_df.columns
]

summary_df = summary_df.reset_index()


# ============================================================
# SAVE
# ============================================================

results_path = os.path.join(
    OUT_DIR,
    "top3_fold_results.csv"
)

summary_path = os.path.join(
    OUT_DIR,
    "top3_summary.csv"
)

predictions_path = os.path.join(
    OUT_DIR,
    "top3_predictions.csv"
)

params_path = os.path.join(
    OUT_DIR,
    "top3_best_parameters.csv"
)


results_df.to_csv(
    results_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

params_df.to_csv(
    params_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)
print("TOP-3 MODEL SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("STEP 16 COMPLETE")
print("=" * 70)

print()
print("Saved:")

print(results_path)
print(summary_path)
print(predictions_path)
print(params_path)