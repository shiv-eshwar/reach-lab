# ============================================================
# STEP 18 — ROC + PRECISION-RECALL CURVES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
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

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "results",
    "top3_validation"
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)

os.makedirs(
    DATA_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 18 — ROC + PRECISION-RECALL CURVES")
print("=" * 70)

df = pd.read_csv(
    INPUT_PATH
)

print()
print(
    f"Prediction rows: {len(df)}"
)

print(
    f"Subjects: {df['subject'].nunique()}"
)

print(
    f"Models: {df['model'].unique().tolist()}"
)


# ============================================================
# MODEL ORDER
# ============================================================

model_order = [
    "Random_Forest",
    "Extra_Trees",
    "Linear_SVM"
]


# ============================================================
# STORAGE
# ============================================================

roc_data = []
pr_data = []
summary = []


# ============================================================
# PROCESS EACH MODEL
# ============================================================

for model_name in model_order:

    model_df = df[
        df["model"] == model_name
    ].copy()

    y_true = model_df[
        "true_target"
    ].to_numpy()

    y_score = model_df[
        "predicted_score"
    ].to_numpy()


    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, roc_thresholds = roc_curve(
        y_true,
        y_score
    )

    roc_auc = roc_auc_score(
        y_true,
        y_score
    )


    for f, t, threshold in zip(
        fpr,
        tpr,
        roc_thresholds
    ):

        roc_data.append({

            "model": model_name,

            "false_positive_rate": f,

            "true_positive_rate": t,

            "threshold": threshold
        })


    # --------------------------------------------------------
    # PRECISION-RECALL
    # --------------------------------------------------------

    precision, recall, pr_thresholds = (
        precision_recall_curve(
            y_true,
            y_score
        )
    )

    average_precision = (
        average_precision_score(
            y_true,
            y_score
        )
    )


    # PR thresholds has one fewer element
    # than precision and recall.

    for i in range(
        len(precision)
    ):

        threshold = (

            pr_thresholds[i]

            if i < len(pr_thresholds)

            else np.nan
        )

        pr_data.append({

            "model": model_name,

            "precision": precision[i],

            "recall": recall[i],

            "threshold": threshold
        })


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    positive_prevalence = (
        np.mean(y_true)
    )

    summary.append({

        "model": model_name,

        "n_subjects": len(y_true),

        "positive_prevalence": (
            positive_prevalence
        ),

        "roc_auc": roc_auc,

        "average_precision": (
            average_precision
        )
    })


    print()
    print(
        f"{model_name}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"Average Precision: "
        f"{average_precision:.4f}"
    )


# ============================================================
# SAVE CURVE DATA
# ============================================================

roc_df = pd.DataFrame(
    roc_data
)

pr_df = pd.DataFrame(
    pr_data
)

summary_df = pd.DataFrame(
    summary
)


roc_path = os.path.join(
    DATA_DIR,
    "top3_roc_curve_data.csv"
)

pr_path = os.path.join(
    DATA_DIR,
    "top3_pr_curve_data.csv"
)

summary_path = os.path.join(
    DATA_DIR,
    "top3_curve_summary.csv"
)


roc_df.to_csv(
    roc_path,
    index=False
)

pr_df.to_csv(
    pr_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FIGURE 6 — ROC CURVES
# ============================================================

plt.figure(
    figsize=(8, 7)
)

for model_name in model_order:

    model_roc = roc_df[
        roc_df["model"] == model_name
    ]

    auc_value = summary_df.loc[
        summary_df["model"] == model_name,
        "roc_auc"
    ].iloc[0]

    plt.plot(
        model_roc[
            "false_positive_rate"
        ],
        model_roc[
            "true_positive_rate"
        ],
        linewidth=2,
        label=(
            f"{model_name.replace('_', ' ')} "
            f"(AUC = {auc_value:.3f})"
        )
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Chance"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves — Top 3 Models"
)

plt.legend(
    loc="lower right"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


roc_figure_path = os.path.join(
    FIGURE_DIR,
    "Figure_6_ROC_curves.png"
)

plt.savefig(
    roc_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FIGURE 7 — PRECISION-RECALL CURVES
# ============================================================

plt.figure(
    figsize=(8, 7)
)

for model_name in model_order:

    model_pr = pr_df[
        pr_df["model"] == model_name
    ]

    ap_value = summary_df.loc[
        summary_df["model"] == model_name,
        "average_precision"
    ].iloc[0]

    plt.plot(
        model_pr["recall"],
        model_pr["precision"],
        linewidth=2,
        label=(
            f"{model_name.replace('_', ' ')} "
            f"(AP = {ap_value:.3f})"
        )
    )


# ------------------------------------------------------------
# Positive-class prevalence
# ------------------------------------------------------------

prevalence = summary_df[
    "positive_prevalence"
].iloc[0]

plt.axhline(
    prevalence,
    linestyle="--",
    linewidth=1.5,
    label=(
        f"Prevalence = {prevalence:.3f}"
    )
)


plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision–Recall Curves — Top 3 Models"
)

plt.legend(
    loc="lower left"
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()


pr_figure_path = os.path.join(
    FIGURE_DIR,
    "Figure_7_Precision_Recall_curves.png"
)

plt.savefig(
    pr_figure_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("CURVE SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("=" * 70)
print("STEP 18 COMPLETE")
print("=" * 70)

print()
print("Figures saved:")

print(
    roc_figure_path
)

print(
    pr_figure_path
)

print()
print("Curve data saved:")

print(
    roc_path
)

print(
    pr_path
)

print(
    summary_path
)