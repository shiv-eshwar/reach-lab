from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experiment_dataset.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "nested_cv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

OUTER_SPLITS = 5
INNER_SPLITS = 4

TOP_K_FEATURES = 20


# ============================================================
# FEATURE SELECTION SCORE
# ============================================================

def mutual_information_score(X, y):

    return mutual_info_classif(
        X,
        y,
        random_state=RANDOM_STATE
    )


# ============================================================
# MODELS
# ============================================================

MODELS = {

    "Logistic_L1": (
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        {
            "selector__k": [10, 20, 40],
            "model__C": [0.01, 0.1, 1, 10],
        }
    ),

    "Logistic_L2": (
        LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE
        ),
        {
            "selector__k": [10, 20, 40],
            "model__C": [0.01, 0.1, 1, 10],
        }
    ),

    "Linear_SVM": (
        SVC(
            kernel="linear",
            probability=True,
            random_state=RANDOM_STATE
        ),
        {
            "selector__k": [10, 20, 40],
            "model__C": [0.01, 0.1, 1, 10],
        }
    ),

    "RBF_SVM": (
        SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE
        ),
        {
            "selector__k": [10, 20, 40],
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": ["scale", 0.01, 0.1],
        }
    ),

    "Polynomial_SVM": (
        SVC(
            kernel="poly",
            probability=True,
            random_state=RANDOM_STATE
        ),
        {
            "selector__k": [10, 20, 40],
            "model__C": [0.1, 1, 10],
            "model__degree": [2, 3],
            "model__gamma": ["scale"],
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "selector__k": [10, 20, 40],
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        }
    ),

    "LDA": (
        LinearDiscriminantAnalysis(),
        {
            "selector__k": [10, 20, 40],
            "model__solver": ["lsqr"],
            "model__shrinkage": ["auto", None],
        }
    ),


    "Random_Forest": (
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        {
            "selector__k": [10, 20, 40],
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 3, 5],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        }
    ),

    "Extra_Trees": (
        ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        {
            "selector__k": [10, 20, 40],
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 3, 5],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        }
    ),

    "HistGradientBoosting": (
        HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        {
            "selector__k": [10, 20, 40],
            "model__max_iter": [100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_leaf_nodes": [7, 15],
            "model__l2_regularization": [0.0, 1.0],
        }
    ),
}


# ============================================================
# PIPELINE
# ============================================================

def make_pipeline(model):

    return Pipeline(
        steps=[
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
                    score_func=mutual_information_score,
                    k=TOP_K_FEATURES
                )
            ),

            (
                "model",
                model
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_prob
        )
    else:
        metrics["roc_auc"] = np.nan

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 10 — LEAKAGE-SAFE NESTED SUBJECT-LEVEL CV")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"subject": str}
    )

    print(
        f"\nDataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # X / y / groups
    # --------------------------------------------------------

    metadata = [
        "subject",
        "side",
        "group",
        "target",
    ]

    feature_columns = [
        col
        for col in df.columns
        if col not in metadata
    ]

    X = df[feature_columns]

    y = df["target"]

    groups = df["subject"]

    print(
        f"Subjects: {groups.nunique()}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Outer CV
    # --------------------------------------------------------

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_results = []
    all_predictions = []

    # --------------------------------------------------------
    # Model loop
    # --------------------------------------------------------

    for model_name, (
        model,
        param_grid
    ) in MODELS.items():

        print("\n" + "=" * 70)
        print(
            f"MODEL: {model_name}"
        )
        print("=" * 70)

        fold_metrics = []

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

            print(
                f"\nOuter fold {fold}/{OUTER_SPLITS}"
            )

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            groups_train = groups.iloc[
                train_idx
            ]

            groups_test = groups.iloc[
                test_idx
            ]

            print(
                f"Train subjects: "
                f"{groups_train.nunique()}"
            )

            print(
                f"Test subjects: "
                f"{groups_test.nunique()}"
            )

            # ------------------------------------------------
            # Inner grouped CV
            # ------------------------------------------------

            inner_cv = StratifiedGroupKFold(
                n_splits=INNER_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE
            )

            pipeline = make_pipeline(
                clone(model)
            )

            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
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

            best_model = search.best_estimator_

            y_pred = best_model.predict(
                X_test
            )

            if hasattr(
                best_model,
                "predict_proba"
            ):
                y_prob = best_model.predict_proba(
                    X_test
                )[:, 1]

            else:
                y_prob = None

            metrics = calculate_metrics(
                y_test,
                y_pred,
                y_prob
            )

            metrics["model"] = model_name
            metrics["fold"] = fold

            fold_metrics.append(
                metrics
            )

            # ----------------------------------------------
            # Save predictions
            # ----------------------------------------------

            for i, idx in enumerate(
                test_idx
            ):

                prediction_row = {
                    "model": model_name,
                    "fold": fold,
                    "subject": df.iloc[idx][
                        "subject"
                    ],
                    "true_group": df.iloc[idx][
                        "group"
                    ],
                    "true_target": int(
                        y.iloc[idx]
                    ),
                    "predicted_target": int(
                        y_pred[i]
                    ),
                }

                if y_prob is not None:
                    prediction_row[
                        "predicted_probability"
                    ] = float(
                        y_prob[i]
                    )

                all_predictions.append(
                    prediction_row
                )

            print(
                f"Accuracy: {metrics['accuracy']:.3f}"
            )

            print(
                f"F1: {metrics['f1']:.3f}"
            )

            print(
                f"ROC-AUC: {metrics['roc_auc']:.3f}"
            )

            print(
                f"Best params: "
                f"{search.best_params__str__ if False else search.best_params_}"
            )

        # ----------------------------------------------------
        # Aggregate model results
        # ----------------------------------------------------

        fold_df = pd.DataFrame(
            fold_metrics
        )

        summary = {
            "model": model_name,

            "accuracy_mean":
                fold_df["accuracy"].mean(),

            "accuracy_std":
                fold_df["accuracy"].std(
                    ddof=1
                ),

            "precision_mean":
                fold_df["precision"].mean(),

            "precision_std":
                fold_df["precision"].std(
                    ddof=1
                ),

            "recall_mean":
                fold_df["recall"].mean(),

            "recall_std":
                fold_df["recall"].std(
                    ddof=1
                ),

            "f1_mean":
                fold_df["f1"].mean(),

            "f1_std":
                fold_df["f1"].std(
                    ddof=1
                ),

            "roc_auc_mean":
                fold_df["roc_auc"].mean(),

            "roc_auc_std":
                fold_df["roc_auc"].std(
                    ddof=1
                ),
        }

        all_results.append(
            summary
        )

        print(
            f"\n{model_name} summary:"
        )

        print(
            f"Accuracy: "
            f"{summary['accuracy_mean']:.3f} "
            f"± "
            f"{summary['accuracy_std']:.3f}"
        )

        print(
            f"F1: "
            f"{summary['f1_mean']:.3f} "
            f"± "
            f"{summary['f1_std']:.3f}"
        )

        print(
            f"ROC-AUC: "
            f"{summary['roc_auc_mean']:.3f} "
            f"± "
            f"{summary['roc_auc_std']:.3f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    results_df.to_csv(
        RESULTS_DIR
        / "nested_cv_results.csv",
        index=False
    )

    predictions_df.to_csv(
        RESULTS_DIR
        / "nested_cv_predictions.csv",
        index=False
    )

    with open(
        RESULTS_DIR
        / "feature_columns.json",
        "w"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("NESTED CV COMPLETE")
    print("=" * 70)

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
        / "nested_cv_results.csv"
    )

    print(
        RESULTS_DIR
        / "nested_cv_predictions.csv"
    )

    print("\nFinal model summary:")

    print(
        results_df[
            [
                "model",
                "accuracy_mean",
                "f1_mean",
                "roc_auc_mean",
            ]
        ]
        .sort_values(
            "roc_auc_mean",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()