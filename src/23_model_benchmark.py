# ============================================================
# STEP 23 — FAST COMPLETE MODEL BENCHMARK
# ============================================================

import os
from functools import partial
import warnings
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.model_selection import (
    StratifiedGroupKFold,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "experiment_dataset.csv"
)

OUTPUT_DIR = os.environ.get("REACH_BENCHMARK_DIR") or os.path.join(
    BASE_DIR,
    "results",
    "model_benchmark"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 23 — FAST COMPLETE MODEL BENCHMARK")
print("=" * 70)

df = pd.read_csv(INPUT_PATH)

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

print()
print(f"Dataset shape : {df.shape}")
print(f"Subjects      : {groups.nunique()}")
print(f"Features      : {X.shape[1]}")
print(f"TD            : {(y == 0).sum()}")
print(f"DCP           : {(y == 1).sum()}")


# ============================================================
# SUBJECT-LEVEL CV
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

inner_cv = StratifiedGroupKFold(
    n_splits=4,
    shuffle=True,
    random_state=42
)


# ============================================================
# COMMON PIPELINE
# ============================================================

def make_pipeline(model):

    return Pipeline([
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
                score_func=partial(mutual_info_classif, random_state=42)
            )
        ),

        (
            "model",
            model
        )
    ])


# ============================================================
# 16 MODELS
# Small grids intentionally used for speed.
# ============================================================

models = {

    # 1
    "Logistic_L1": (
        LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=3000,
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__C": [0.1, 1]
        }
    ),

    # 2
    "Logistic_L2": (
        LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=3000,
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__C": [0.1, 1]
        }
    ),

    # 3
    "Linear_SVM": (
        SVC(
            kernel="linear",
            probability=True,
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__C": [0.1, 1]
        }
    ),

    # 4
    "RBF_SVM": (
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__C": [1, 10],
            "model__gamma": ["scale"]
        }
    ),

    # 5
    "Polynomial_SVM": (
        SVC(
            kernel="poly",
            probability=True,
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__C": [0.1, 1],
            "model__degree": [2]
        }
    ),

    # 6
    "KNN": (
        KNeighborsClassifier(),
        {
            "selector__k": [20, 40],
            "model__n_neighbors": [3, 5],
            "model__weights": ["distance"]
        }
    ),

    # 7
    "LDA": (
        LinearDiscriminantAnalysis(),
        {
            "selector__k": [20, 40]
        }
    ),

    # 8
    "Random_Forest": (
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=1
        ),
        {
            "selector__k": [20, 40],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1]
        }
    ),

    # 9
    "Extra_Trees": (
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=1
        ),
        {
            "selector__k": [20, 40],
            "model__max_depth": [None, 10],
            "model__min_samples_leaf": [1]
        }
    ),

    # 10
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": [100]
        }
    ),

    # 11
    "Gradient_Boosting": (
        GradientBoostingClassifier(
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__n_estimators": [100],
            "model__learning_rate": [0.05, 0.1]
        }
    ),

    # 12
    "AdaBoost": (
        AdaBoostClassifier(
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__n_estimators": [50, 100],
            "model__learning_rate": [0.5]
        }
    ),

    # 13
    "Decision_Tree": (
        DecisionTreeClassifier(
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__max_depth": [3, 5, None]
        }
    ),

    # 14
    "Gaussian_NB": (
        GaussianNB(),
        {
            "selector__k": [20, 40],
            "model__var_smoothing": [1e-9, 1e-7]
        }
    ),

    # 15
    "MLP": (
        MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=1000,
            early_stopping=True,
            random_state=42
        ),
        {
            "selector__k": [20, 40],
            "model__alpha": [0.0001, 0.01]
        }
    ),

    # 16
    "XGBoost": None
}


# ============================================================
# XGBOOST
# ============================================================

try:

    from xgboost import XGBClassifier

    models["XGBoost"] = (
        XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        ),
        {
            "selector__k": [20, 40],
            "model__max_depth": [2, 3]
        }
    )

    print()
    print("XGBoost available.")

except ImportError:

    del models["XGBoost"]

    print()
    print("XGBoost not installed — skipped.")


# ============================================================
# MODEL LIST
# ============================================================

print()
print("=" * 70)
print(
    f"MODELS TO BENCHMARK: {len(models)}"
)
print("=" * 70)

for i, name in enumerate(
    models.keys(),
    start=1
):
    print(
        f"{i:02d}. {name}"
    )


# ============================================================
# STORAGE
# ============================================================

summary_results = []
fold_results = []
prediction_results = []


# ============================================================
# BENCHMARK
# ============================================================

for model_number, (
    model_name,
    config
) in enumerate(
    models.items(),
    start=1
):

    print()
    print("#" * 70)
    print(
        f"MODEL {model_number}/{len(models)}: "
        f"{model_name}"
    )
    print("#" * 70)

    model, param_grid = config

    model_fold_results = []

    # --------------------------------------------------------
    # OUTER 5-FOLD
    # --------------------------------------------------------

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


        # ----------------------------------------------------
        # Pipeline
        # ----------------------------------------------------

        pipeline = make_pipeline(
            clone(model)
        )


        # ----------------------------------------------------
        # Inner CV
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        y_pred = best_model.predict(
            X_test
        )

        if hasattr(
            best_model,
            "predict_proba"
        ):

            y_score = (
                best_model
                .predict_proba(X_test)[:, 1]
            )

        else:

            y_score = (
                best_model
                .decision_function(X_test)
            )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        roc_auc = roc_auc_score(
            y_test,
            y_score
        )


        result = {

            "model":
                model_name,

            "fold":
                fold,

            "accuracy":
                accuracy,

            "f1":
                f1,

            "roc_auc":
                roc_auc,

            "best_parameters":
                str(
                    search.best_params_
                )
        }


        model_fold_results.append(
            result
        )

        fold_results.append(
            result
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        for subject, true_y, pred_y, score in zip(

            groups.iloc[test_idx],

            y_test,

            y_pred,

            y_score

        ):

            prediction_results.append({

                "model":
                    model_name,

                "fold":
                    fold,

                "subject":
                    subject,

                "true_target":
                    int(true_y),

                "predicted_target":
                    int(pred_y),

                "predicted_score":
                    float(score)

            })


        print(
            f"  Fold {fold}: "
            f"Acc={accuracy:.4f} | "
            f"F1={f1:.4f} | "
            f"AUC={roc_auc:.4f}"
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    model_df = pd.DataFrame(
        model_fold_results
    )

    summary_results.append({

        "model":
            model_name,

        "accuracy_mean":
            model_df["accuracy"].mean(),

        "accuracy_std":
            model_df["accuracy"].std(),

        "f1_mean":
            model_df["f1"].mean(),

        "f1_std":
            model_df["f1"].std(),

        "roc_auc_mean":
            model_df["roc_auc"].mean(),

        "roc_auc_std":
            model_df["roc_auc"].std()
    })


    print()
    print(
        f"{model_name}: "
        f"Accuracy="
        f"{model_df['accuracy'].mean():.4f} "
        f"+/- "
        f"{model_df['accuracy'].std():.4f}"
    )

    print(
        f"             F1="
        f"{model_df['f1'].mean():.4f} "
        f"+/- "
        f"{model_df['f1'].std():.4f}"
    )

    print(
        f"             AUC="
        f"{model_df['roc_auc'].mean():.4f} "
        f"+/- "
        f"{model_df['roc_auc'].std():.4f}"
    )


# ============================================================
# SAVE
# ============================================================

summary_df = pd.DataFrame(
    summary_results
)

fold_df = pd.DataFrame(
    fold_results
)

prediction_df = pd.DataFrame(
    prediction_results
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "model_benchmark_summary.csv"
)

fold_path = os.path.join(
    OUTPUT_DIR,
    "model_benchmark_fold_results.csv"
)

prediction_path = os.path.join(
    OUTPUT_DIR,
    "model_benchmark_predictions.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)

fold_df.to_csv(
    fold_path,
    index=False
)

prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 70)
print("COMPLETE MODEL BENCHMARK")
print("=" * 70)

print()

print(
    summary_df
    .sort_values(
        "roc_auc_mean",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("STEP 23 COMPLETE")
print("=" * 70)

print()
print(
    f"Saved to: {OUTPUT_DIR}"
)