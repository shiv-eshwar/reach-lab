"""
STEP 5 — ML + EXPLAINABLE AI

Objectives 4, 5 & 7:
- Feature fusion
- Leakage-safe feature selection
- Machine-learning classification: DCP vs TD
- Cross-validation
- SHAP explainability
- LIME explainability

Dataset:
- 51 subjects
- 30 DCP
- 21 TD

Important:
All model evaluation is performed using leakage-safe 5-fold
stratified cross-validation.

The Random Forest model is used specifically as the explainability
model for SHAP/LIME because of its compact tree-based structure.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

import xgboost as xgb
import lightgbm as lgb
import shap
from catboost import CatBoostClassifier


# ============================================================
# WARNINGS
# ============================================================

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "subject_biomarkers.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# BIOMARKERS
# ============================================================

BIOMARKER_COLS = [
    "Movement_Smoothness_Index",
    "Trajectory_Stability_Index",
    "Reach_Efficiency_Index",
    "Joint_Coordination_Index",
    "Motion_Variability_Index",
    "Functional_Consistency_Index",
    "Motor_Adaptability_Index",
    "Movement_Stability_Index",
]


# ============================================================
# CROSS-VALIDATION
# ============================================================

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# MODELS
# ============================================================

def make_models():

    return {

        "RandomForest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced",
            ),

        "XGBoost":
            xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            ),

        "LightGBM":
            lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                min_child_samples=5,
                verbose=-1,
                random_state=42,
            ),

        "CatBoost":
            CatBoostClassifier(
                iterations=150,
                depth=3,
                learning_rate=0.05,
                l2_leaf_reg=5,
                loss_function="Logloss",
                verbose=False,
                random_seed=42,
                allow_writing_files=False,
            ),

        "SVM_RBF":
            SVC(
                kernel="rbf",
                probability=True,
                C=1.0,
                random_state=42,
            ),
    }


# ============================================================
# MUTUAL INFORMATION SELECTOR
# ============================================================

class MutualInformationSelector(
    BaseEstimator,
    TransformerMixin
):
    """
    Sklearn-compatible transformer.

    Selects the top-k features using mutual information.
    Fitted independently inside every CV fold.
    """

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):

        X = np.asarray(X)

        scores = mutual_info_classif(
            X,
            y,
            random_state=42,
        )

        self.scores_ = scores

        k = min(
            self.k,
            X.shape[1],
        )

        self.indices_ = np.argsort(
            scores
        )[-k:]

        return self

    def transform(self, X):

        X = np.asarray(X)

        return X[
            :,
            self.indices_,
        ]

    def fit_transform(self, X, y):

        self.fit(X, y)

        return self.transform(X)

    def get_support(self):

        mask = np.zeros(
            len(self.scores_),
            dtype=bool,
        )

        mask[
            self.indices_
        ] = True

        return mask


# ============================================================
# LEAKAGE-SAFE PIPELINE
# ============================================================

def make_pipeline(model):

    return Pipeline(
        steps=[

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "feature_selection",

                MutualInformationSelector(
                    k=5
                ),
            ),

            (
                "model",

                model,
            ),
        ]
    )


# ============================================================
# FEATURE RANKING
# ============================================================

def feature_ranking(X, y):

    print("\n" + "=" * 60)
    print("FEATURE RANKING")
    print("=" * 60)

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imp = imputer.fit_transform(X)

    scores = mutual_info_classif(
        X_imp,
        y,
        random_state=42,
    )

    ranking = (
        pd.DataFrame(
            {
                "biomarker": BIOMARKER_COLS,
                "mutual_information": scores,
            }
        )
        .sort_values(
            "mutual_information",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranking.to_csv(
        RESULTS_DIR / "feature_ranking.csv",
        index=False,
    )

    print(ranking)

    return ranking


# ============================================================
# RFE FEATURE SELECTION
# ============================================================

def rfe_ranking(X, y):

    print("\n" + "=" * 60)
    print("RFE FEATURE SELECTION")
    print("=" * 60)

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imp = imputer.fit_transform(X)

    estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    selector = RFE(
        estimator=estimator,
        n_features_to_select=5,
        step=1,
    )

    selector.fit(
        X_imp,
        y,
    )

    ranking = (
        pd.DataFrame(
            {
                "biomarker": BIOMARKER_COLS,
                "rfe_rank": selector.ranking_,
                "rfe_selected": selector.support_,
            }
        )
        .sort_values(
            ["rfe_rank", "biomarker"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    ranking.to_csv(
        RESULTS_DIR / "rfe_feature_ranking.csv",
        index=False,
    )

    print(ranking)

    selected = ranking.loc[
        ranking["rfe_selected"],
        "biomarker"
    ].tolist()

    print("\nRFE selected top 5:")

    for feature in selected:
        print(
            f"  ✓ {feature}"
        )

    with open(
        RESULTS_DIR / "rfe_selected_features.json",
        "w"
    ) as file:

        json.dump(
            selected,
            file,
            indent=2,
        )

    return ranking


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models(X, y):

    print("\n" + "=" * 60)
    print("LEAKAGE-SAFE MODEL EVALUATION")
    print("=" * 60)

    models = make_models()

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = {}

    for name, model in models.items():

        pipeline = make_pipeline(
            model
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=CV,
            scoring=scoring,
            return_train_score=False,
        )

        results[name] = {}

        for metric in scoring:

            values = scores[
                f"test_{metric}"
            ]

            results[name][
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            results[name][
                f"{metric}_std"
            ] = float(
                np.std(values)
            )

        print(
            f"\n{name}"
        )

        print(
            f"Accuracy : "
            f"{results[name]['accuracy_mean']:.3f}"
            f" ± "
            f"{results[name]['accuracy_std']:.3f}"
        )

        print(
            f"Precision: "
            f"{results[name]['precision_mean']:.3f}"
            f" ± "
            f"{results[name]['precision_std']:.3f}"
        )

        print(
            f"Recall   : "
            f"{results[name]['recall_mean']:.3f}"
            f" ± "
            f"{results[name]['recall_std']:.3f}"
        )

        print(
            f"F1       : "
            f"{results[name]['f1_mean']:.3f}"
            f" ± "
            f"{results[name]['f1_std']:.3f}"
        )

        print(
            f"ROC-AUC  : "
            f"{results[name]['roc_auc_mean']:.3f}"
            f" ± "
            f"{results[name]['roc_auc_std']:.3f}"
        )

    with open(
        RESULTS_DIR / "model_results.json",
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    return results


# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================

def generate_oof_predictions(
    X,
    y,
    df,
):

    print("\n" + "=" * 60)
    print("OUT-OF-FOLD EVALUATION")
    print("=" * 60)

    models = make_models()

    all_oof = {}

    for name, model in models.items():

        probabilities = np.zeros(
            len(y)
        )

        predictions = np.zeros(
            len(y),
            dtype=int,
        )

        for fold, (
            train_idx,
            test_idx
        ) in enumerate(
            CV.split(X, y),
            start=1,
        ):

            pipeline = make_pipeline(
                model
            )

            pipeline.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            probabilities[
                test_idx
            ] = pipeline.predict_proba(
                X.iloc[test_idx]
            )[:, 1]

            predictions[
                test_idx
            ] = (
                probabilities[test_idx]
                >= 0.5
            ).astype(int)

        accuracy = accuracy_score(
            y,
            predictions,
        )

        precision = precision_score(
            y,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0,
        )

        auc = roc_auc_score(
            y,
            probabilities,
        )

        print(
            f"\n{name}"
        )

        print(
            f"OOF Accuracy : {accuracy:.3f}"
        )

        print(
            f"OOF Precision: {precision:.3f}"
        )

        print(
            f"OOF Recall   : {recall:.3f}"
        )

        print(
            f"OOF F1       : {f1:.3f}"
        )

        print(
            f"OOF ROC-AUC  : {auc:.3f}"
        )

        all_oof[name] = {
            "probability": probabilities,
            "prediction": predictions,
        }

    # --------------------------------------------------------
    # Save OOF predictions for every model
    # --------------------------------------------------------

    oof_df = df[
        [
            "subject",
            "group",
            "side",
            "Functional_Resilience_Index",
        ]
    ].copy()

    for name, values in all_oof.items():

        safe_name = name.lower()

        oof_df[
            f"{safe_name}_probability"
        ] = values["probability"]

        oof_df[
            f"{safe_name}_prediction"
        ] = np.where(
            values["prediction"] == 1,
            "DCP",
            "TD",
        )

    oof_df.to_csv(
        RESULTS_DIR / "oof_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # OOF confusion matrix for each model
    # --------------------------------------------------------

    cm_rows = []

    for name, values in all_oof.items():

        cm = confusion_matrix(
            y,
            values["prediction"],
        )

        tn, fp, fn, tp = cm.ravel()

        cm_rows.append(
            {
                "model": name,
                "true_TD_pred_TD": int(tn),
                "true_TD_pred_DCP": int(fp),
                "true_DCP_pred_TD": int(fn),
                "true_DCP_pred_DCP": int(tp),
            }
        )

    cm_df = pd.DataFrame(
        cm_rows
    )

    cm_df.to_csv(
        RESULTS_DIR / "oof_confusion_matrices.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Random Forest OOF confusion matrix
    # --------------------------------------------------------

    rf_cm = confusion_matrix(
        y,
        all_oof["RandomForest"]["prediction"],
    )

    np.savetxt(
        RESULTS_DIR / "confusion_matrix.csv",
        rf_cm,
        delimiter=",",
        fmt="%d",
    )

    print(
        "\nRandom Forest OOF confusion matrix:"
    )

    print(rf_cm)

    return all_oof


# ============================================================
# RANDOM FOREST EXPLAINABILITY MODEL
# ============================================================

def train_explainability_model(
    X,
    y,
):

    print("\n" + "=" * 60)
    print("EXPLAINABILITY MODEL")
    print("=" * 60)

    print(
        "Random Forest selected specifically "
        "for SHAP/LIME interpretation."
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imp = imputer.fit_transform(
        X
    )

    selector = MutualInformationSelector(
        k=5
    )

    X_selected = selector.fit_transform(
        X_imp,
        y,
    )

    selected_mask = selector.get_support()

    selected_features = [
        feature
        for feature, selected
        in zip(
            BIOMARKER_COLS,
            selected_mask,
        )
        if selected
    ]

    print(
        "\nFeatures selected for explainability:"
    )

    for feature in selected_features:

        print(
            f"  ✓ {feature}"
        )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(
        X_selected,
        y,
    )

    return (
        imputer,
        selector,
        model,
        X_selected,
        selected_features,
    )


# ============================================================
# SHAP
# ============================================================

def run_shap(
    model,
    X_selected,
    selected_features,
):

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY")
    print("=" * 60)

    explainer = shap.TreeExplainer(
        model
    )

    explanation = explainer(
        X_selected
    )

    values = explanation.values

    if values.ndim == 3:

        shap_values = values[
            :,
            :,
            1
        ]

    else:

        shap_values = values

    mean_abs = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    shap_importance = (
        pd.DataFrame(
            {
                "biomarker":
                    selected_features,

                "mean_abs_shap":
                    mean_abs,
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\nMean absolute SHAP importance:"
    )

    print(
        shap_importance
    )

    shap_importance.to_csv(
        RESULTS_DIR
        / "shap_feature_importance.csv",
        index=False,
    )

    plt.figure(
        figsize=(8, 6)
    )

    shap.summary_plot(
        shap_values,
        X_selected,
        feature_names=selected_features,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "shap_summary.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    np.save(
        RESULTS_DIR
        / "shap_values.npy",
        shap_values,
    )

    return shap_importance


# ============================================================
# LIME
# ============================================================

def run_lime(
    model,
    X_selected,
    selected_features,
):

    print("\n" + "=" * 60)
    print("LIME EXPLAINABILITY")
    print("=" * 60)

    try:

        from lime.lime_tabular import (
            LimeTabularExplainer
        )

    except ImportError:

        print(
            "\nLIME is not installed."
        )

        print(
            "Install it using:"
        )

        print(
            "pip install lime"
        )

        return

    explainer = LimeTabularExplainer(
        X_selected,
        feature_names=selected_features,
        class_names=[
            "TD",
            "DCP",
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    explanation = explainer.explain_instance(
        X_selected[0],
        model.predict_proba,
        num_features=len(
            selected_features
        ),
    )

    lime_rows = []

    for feature, weight in (
        explanation.as_list()
    ):

        lime_rows.append(
            {
                "feature": feature,
                "weight": weight,
            }
        )

    lime_df = pd.DataFrame(
        lime_rows
    )

    lime_df.to_csv(
        RESULTS_DIR
        / "lime_explanation.csv",
        index=False,
    )

    print(
        "\nLIME explanation:"
    )

    print(
        lime_df
    )

    try:

        explanation.save_to_file(
            str(
                RESULTS_DIR
                / "lime_explanation.html"
            )
        )

    except Exception:

        pass


# ============================================================
# ROC CURVES
# ============================================================

def generate_roc_data(
    X,
    y,
):

    print("\n" + "=" * 60)
    print("ROC CURVES")
    print("=" * 60)

    models = make_models()

    roc_results = {}

    plt.figure(
        figsize=(8, 6)
    )

    for name, model in models.items():

        probabilities = np.zeros(
            len(y)
        )

        for train_idx, test_idx in CV.split(
            X,
            y,
        ):

            pipeline = make_pipeline(
                model
            )

            pipeline.fit(
                X.iloc[train_idx],
                y[train_idx],
            )

            probabilities[
                test_idx
            ] = pipeline.predict_proba(
                X.iloc[test_idx]
            )[:, 1]

        fpr, tpr, thresholds = roc_curve(
            y,
            probabilities,
        )

        auc = roc_auc_score(
            y,
            probabilities,
        )

        roc_results[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(auc),
        }

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves — DCP vs TD"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "roc_curves.png",
        dpi=200,
    )

    plt.close()

    with open(
        RESULTS_DIR
        / "roc_data.json",
        "w"
    ) as file:

        json.dump(
            roc_results,
            file,
            indent=2,
        )

    return roc_results


# ============================================================
# SUBJECT SCORES
# ============================================================

def save_subject_scores(
    df,
    imputer,
    selector,
    model,
):

    X_original = df[
        BIOMARKER_COLS
    ]

    X_imp = imputer.transform(
        X_original
    )

    X_sel = selector.transform(
        X_imp
    )

    probabilities = (
        model.predict_proba(
            X_sel
        )[:, 1]
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    scored = df[
        [
            "subject",
            "group",
            "side",
            "Functional_Resilience_Index",
        ]
    ].copy()

    scored[
        "dcp_probability"
    ] = probabilities

    scored[
        "predicted_group"
    ] = np.where(
        predictions == 1,
        "DCP",
        "TD",
    )

    scored[
        "prediction_correct"
    ] = (
        scored["predicted_group"]
        == scored["group"]
    )

    scored.to_csv(
        RESULTS_DIR
        / "subject_biomarkers_scored.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 5 — ML + EXPLAINABLE AI")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run 04_biomarkers.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "subject": str,
        },
    )

    print(
        f"\nSubjects: "
        f"{len(df)}"
    )

    print(
        f"Groups:\n"
        f"{df['group'].value_counts()}"
    )

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------

    X = df[
        BIOMARKER_COLS
    ].copy()

    y = (
        df["group"] == "DCP"
    ).astype(int).to_numpy()

    # --------------------------------------------------------
    # Feature ranking
    # --------------------------------------------------------

    feature_ranking(
        X,
        y,
    )

    # --------------------------------------------------------
    # RFE
    # --------------------------------------------------------

    rfe_ranking(
        X,
        y,
    )

    # --------------------------------------------------------
    # ML
    # --------------------------------------------------------

    evaluate_models(
        X,
        y,
    )

    # --------------------------------------------------------
    # OOF predictions + confusion matrices
    # --------------------------------------------------------

    generate_oof_predictions(
        X,
        y,
        df,
    )

    # --------------------------------------------------------
    # Explainability model
    # --------------------------------------------------------

    (
        imputer,
        selector,
        model,
        X_selected,
        selected_features,
    ) = train_explainability_model(
        X,
        y,
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    run_shap(
        model,
        X_selected,
        selected_features,
    )

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

    run_lime(
        model,
        X_selected,
        selected_features,
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    generate_roc_data(
        X,
        y,
    )

    # --------------------------------------------------------
    # Subject scores
    # --------------------------------------------------------

    save_subject_scores(
        df,
        imputer,
        selector,
        model,
    )

    # --------------------------------------------------------
    # Selected features
    # --------------------------------------------------------

    with open(
        RESULTS_DIR
        / "selected_features.json",
        "w"
    ) as file:

        json.dump(
            selected_features,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STEP 5 COMPLETE")
    print("=" * 60)

    print(
        "\nResults saved in:"
    )

    print(
        RESULTS_DIR
    )

    print(
        "\nGenerated:"
    )

    print(
        "  ✓ model_results.json"
    )

    print(
        "  ✓ feature_ranking.csv"
    )

    print(
        "  ✓ rfe_feature_ranking.csv"
    )

    print(
        "  ✓ rfe_selected_features.json"
    )

    print(
        "  ✓ oof_predictions.csv"
    )

    print(
        "  ✓ oof_confusion_matrices.csv"
    )

    print(
        "  ✓ confusion_matrix.csv"
    )

    print(
        "  ✓ shap_feature_importance.csv"
    )

    print(
        "  ✓ shap_summary.png"
    )

    print(
        "  ✓ shap_values.npy"
    )

    print(
        "  ✓ lime_explanation.csv"
    )

    print(
        "  ✓ lime_explanation.html"
    )

    print(
        "  ✓ roc_curves.png"
    )

    print(
        "  ✓ roc_data.json"
    )

    print(
        "  ✓ subject_biomarkers_scored.csv"
    )

    print(
        "  ✓ selected_features.json"
    )


if __name__ == "__main__":
    main()