# ============================================================
# STEP 17 — CONFUSION MATRICES + CLASSIFICATION METRICS
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    brier_score_loss
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "top3_validation",
    "top3_predictions.csv"
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
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("STEP 17 — CONFUSION MATRICES + CLASSIFICATION METRICS")
print("=" * 70)

df = pd.read_csv(
    INPUT_PATH
)

print()
print(
    f"Prediction rows: {len(df)}"
)

print(
    f"Models: {df['model'].unique().tolist()}"
)

print(
    f"Subjects: {df['subject'].nunique()}"
)


# ============================================================
# STORAGE
# ============================================================

metrics = []
confusion_rows = []


# ============================================================
# ANALYZE EACH MODEL
# ============================================================

for model_name in sorted(
    df["model"].unique()
):

    print()
    print("=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    model_df = df[
        df["model"] == model_name
    ].copy()

    y_true = model_df[
        "true_target"
    ].to_numpy()

    y_pred = model_df[
        "predicted_target"
    ].to_numpy()

    y_score = model_df[
        "predicted_score"
    ].to_numpy()


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()


    print()
    print("Confusion Matrix")
    print(
        "                 Predicted"
    )
    print(
        "                 TD    DCP"
    )
    print(
        f"Actual TD       {tn:3d}   {fp:3d}"
    )
    print(
        f"Actual DCP      {fn:3d}   {tp:3d}"
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else np.nan
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_true,
        y_score
    )


    # --------------------------------------------------------
    # PPV / NPV
    # --------------------------------------------------------

    ppv = (

        tp / (tp + fp)

        if (tp + fp) > 0

        else np.nan
    )

    npv = (

        tn / (tn + fn)

        if (tn + fn) > 0

        else np.nan
    )


    # --------------------------------------------------------
    # BRIER SCORE
    # --------------------------------------------------------

    # IMPORTANT:
    # Linear SVM and other decision-function models do not
    # produce calibrated probabilities.
    #
    # Therefore Brier score is calculated only if scores are
    # already within [0, 1].

    if (
        np.min(y_score) >= 0
        and np.max(y_score) <= 1
    ):

        brier = brier_score_loss(
            y_true,
            y_score
        )

    else:

        brier = np.nan


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        f"Accuracy:            {accuracy:.4f}"
    )

    print(
        f"Sensitivity:         {sensitivity:.4f}"
    )

    print(
        f"Specificity:         {specificity:.4f}"
    )

    print(
        f"Precision (PPV):     {ppv:.4f}"
    )

    print(
        f"NPV:                 {npv:.4f}"
    )

    print(
        f"F1:                  {f1:.4f}"
    )

    print(
        f"Balanced Accuracy:   {balanced_accuracy:.4f}"
    )

    print(
        f"MCC:                 {mcc:.4f}"
    )

    print(
        f"ROC-AUC:             {roc_auc:.4f}"
    )

    if np.isnan(brier):

        print(
            "Brier Score:         N/A "
            "(decision scores)"
        )

    else:

        print(
            f"Brier Score:         {brier:.4f}"
        )


    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics.append({

        "model": model_name,

        "n_subjects": len(model_df),

        "true_TD": int(tn + fp),

        "true_DCP": int(fn + tp),

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp),

        "accuracy": accuracy,

        "sensitivity": sensitivity,

        "specificity": specificity,

        "precision_PPV": ppv,

        "NPV": npv,

        "F1": f1,

        "balanced_accuracy": balanced_accuracy,

        "MCC": mcc,

        "ROC_AUC": roc_auc,

        "Brier_score": brier
    })


    # --------------------------------------------------------
    # SAVE CONFUSION MATRIX
    # --------------------------------------------------------

    confusion_rows.extend([

        {
            "model": model_name,
            "actual": "TD",
            "predicted": "TD",
            "count": int(tn)
        },

        {
            "model": model_name,
            "actual": "TD",
            "predicted": "DCP",
            "count": int(fp)
        },

        {
            "model": model_name,
            "actual": "DCP",
            "predicted": "TD",
            "count": int(fn)
        },

        {
            "model": model_name,
            "actual": "DCP",
            "predicted": "DCP",
            "count": int(tp)
        }
    ])


# ============================================================
# DATAFRAMES
# ============================================================

metrics_df = pd.DataFrame(
    metrics
)

confusion_df = pd.DataFrame(
    confusion_rows
)


# ============================================================
# SAVE
# ============================================================

metrics_path = os.path.join(
    OUT_DIR,
    "top3_classification_metrics.csv"
)

confusion_path = os.path.join(
    OUT_DIR,
    "top3_confusion_matrices.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

confusion_df.to_csv(
    confusion_path,
    index=False
)


# ============================================================
# SUMMARY TABLE
# ============================================================

print()
print("=" * 70)
print("FINAL TOP-3 CLASSIFICATION METRICS")
print("=" * 70)

display_columns = [
    "model",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision_PPV",
    "NPV",
    "F1",
    "balanced_accuracy",
    "MCC",
    "ROC_AUC"
]

print(
    metrics_df[
        display_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 17 COMPLETE")
print("=" * 70)

print()
print("Saved:")

print(
    metrics_path
)

print(
    confusion_path
)