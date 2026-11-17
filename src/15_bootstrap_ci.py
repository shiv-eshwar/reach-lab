# ============================================================
# STEP 15 — BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PREDICTIONS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "nested_cv",
    "nested_cv_predictions.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "bootstrap_ci"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

N_BOOTSTRAPS = 10000
RANDOM_STATE = 42

rng = np.random.default_rng(
    RANDOM_STATE
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("STEP 15 — BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)

pred = pd.read_csv(
    PREDICTIONS_PATH
)

print()
print(f"Prediction rows: {len(pred)}")

print(
    f"Columns: {pred.columns.tolist()}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "model",
    "subject",
    "true_target",
    "predicted_target",
    "predicted_probability"
]

for col in required_columns:

    if col not in pred.columns:

        raise ValueError(
            f"Required column '{col}' "
            f"not found in predictions file."
        )


# ============================================================
# RANDOM FOREST PREDICTIONS
# ============================================================

rf_pred = pred[
    pred["model"] == "Random_Forest"
].copy()

print()
print(
    f"Random Forest prediction rows: "
    f"{len(rf_pred)}"
)


if len(rf_pred) == 0:

    raise ValueError(
        "No Random_Forest predictions found."
    )


# ============================================================
# CHECK SUBJECT COUNT
# ============================================================

print(
    f"Unique subjects: "
    f"{rf_pred['subject'].nunique()}"
)

print(
    f"Class counts:"
)

print(
    rf_pred["true_target"].value_counts().sort_index()
)


# ============================================================
# EXTRACT VALUES
# ============================================================

y_true = rf_pred[
    "true_target"
].to_numpy()

y_pred = rf_pred[
    "predicted_target"
].to_numpy()

y_prob = rf_pred[
    "predicted_probability"
].to_numpy()


# ============================================================
# POINT ESTIMATES
# ============================================================

accuracy_point = accuracy_score(
    y_true,
    y_pred
)

f1_point = f1_score(
    y_true,
    y_pred
)

roc_auc_point = roc_auc_score(
    y_true,
    y_prob
)


print()
print("=" * 70)
print("POINT ESTIMATES")
print("=" * 70)

print(
    f"Accuracy: {accuracy_point:.4f}"
)

print(
    f"F1: {f1_point:.4f}"
)

print(
    f"ROC-AUC: {roc_auc_point:.4f}"
)


# ============================================================
# STRATIFIED BOOTSTRAP
# ============================================================

print()
print(
    f"Running {N_BOOTSTRAPS:,} stratified "
    f"bootstrap samples..."
)


# Separate the two classes
class_0_indices = np.where(
    y_true == 0
)[0]

class_1_indices = np.where(
    y_true == 1
)[0]


n_class_0 = len(
    class_0_indices
)

n_class_1 = len(
    class_1_indices
)


accuracy_boot = np.empty(
    N_BOOTSTRAPS
)

f1_boot = np.empty(
    N_BOOTSTRAPS
)

auc_boot = np.empty(
    N_BOOTSTRAPS
)


for i in range(
    N_BOOTSTRAPS
):

    # --------------------------------------------------------
    # Resample within each class
    # --------------------------------------------------------

    sample_0 = rng.choice(
        class_0_indices,
        size=n_class_0,
        replace=True
    )

    sample_1 = rng.choice(
        class_1_indices,
        size=n_class_1,
        replace=True
    )

    indices = np.concatenate(
        [
            sample_0,
            sample_1
        ]
    )

    # --------------------------------------------------------
    # Bootstrap sample
    # --------------------------------------------------------

    y_true_b = y_true[
        indices
    ]

    y_pred_b = y_pred[
        indices
    ]

    y_prob_b = y_prob[
        indices
    ]

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy_boot[i] = accuracy_score(
        y_true_b,
        y_pred_b
    )

    f1_boot[i] = f1_score(
        y_true_b,
        y_pred_b,
        zero_division=0
    )

    auc_boot[i] = roc_auc_score(
        y_true_b,
        y_prob_b
    )


# ============================================================
# CONFIDENCE INTERVAL FUNCTION
# ============================================================

def percentile_ci(values):

    lower = np.percentile(
        values,
        2.5
    )

    upper = np.percentile(
        values,
        97.5
    )

    return lower, upper


accuracy_lower, accuracy_upper = (
    percentile_ci(
        accuracy_boot
    )
)

f1_lower, f1_upper = (
    percentile_ci(
        f1_boot
    )
)

auc_lower, auc_upper = (
    percentile_ci(
        auc_boot
    )
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame({

    "model": [
        "Random_Forest",
        "Random_Forest",
        "Random_Forest"
    ],

    "metric": [
        "Accuracy",
        "F1",
        "ROC-AUC"
    ],

    "point_estimate": [
        accuracy_point,
        f1_point,
        roc_auc_point
    ],

    "ci_lower_95": [
        accuracy_lower,
        f1_lower,
        auc_lower
    ],

    "ci_upper_95": [
        accuracy_upper,
        f1_upper,
        auc_upper
    ],

    "n_bootstrap": [
        N_BOOTSTRAPS,
        N_BOOTSTRAPS,
        N_BOOTSTRAPS
    ]
})


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    OUT_DIR,
    "random_forest_bootstrap_ci.csv"
)

results.to_csv(
    output_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("95% BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)

for _, row in results.iterrows():

    print(
        f"{row['metric']}: "
        f"{row['point_estimate']:.4f} "
        f"["
        f"{row['ci_lower_95']:.4f}, "
        f"{row['ci_upper_95']:.4f}"
        f"]"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 15 COMPLETE")
print("=" * 70)

print()
print("Results saved to:")

print(
    output_path
)