# ============================================================
# STEP 22 — ABLATION STATISTICAL COMPARISON
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    BASE_DIR,
    "results",
    "ablation",
    "ablation_fold_results.csv"
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
print("STEP 22 — ABLATION STATISTICAL COMPARISON")
print("=" * 70)

df = pd.read_csv(
    INPUT_PATH
)

print()
print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {df.columns.tolist()}"
)

print(
    f"Representations: "
    f"{df['representation'].unique().tolist()}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "representation",
    "fold"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"not found."
        )


# ============================================================
# IDENTIFY METRIC COLUMNS
# ============================================================

possible_metrics = [
    "accuracy",
    "f1",
    "roc_auc"
]

metrics = [
    metric
    for metric in possible_metrics
    if metric in df.columns
]

if len(metrics) == 0:

    raise ValueError(
        "No expected metric columns found."
    )


print()
print(
    f"Metrics found: {metrics}"
)


# ============================================================
# REPRESENTATIONS
# ============================================================

STATIC = "Static"
REPETITION_AWARE = "Repetition_Aware"


if STATIC not in df["representation"].values:

    raise ValueError(
        "Static representation not found."
    )


if REPETITION_AWARE not in df[
    "representation"
].values:

    raise ValueError(
        "Repetition_Aware representation "
        "not found."
    )


# ============================================================
# PAIRED COMPARISON
# ============================================================

results = []


for metric in metrics:

    print()
    print("=" * 70)
    print(
        f"METRIC: {metric.upper()}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Static
    # --------------------------------------------------------

    static_df = (

        df[
            df["representation"]
            == STATIC
        ]

        [["fold", metric]]

        .rename(
            columns={
                metric:
                    "static"
            }
        )
    )


    # --------------------------------------------------------
    # Repetition-aware
    # --------------------------------------------------------

    rep_df = (

        df[
            df["representation"]
            == REPETITION_AWARE
        ]

        [["fold", metric]]

        .rename(
            columns={
                metric:
                    "repetition_aware"
            }
        )
    )


    # --------------------------------------------------------
    # Match folds
    # --------------------------------------------------------

    paired = static_df.merge(
        rep_df,
        on="fold",
        how="inner"
    )


    if len(paired) < 2:

        raise ValueError(
            f"Not enough paired folds for "
            f"{metric}."
        )


    static_values = paired[
        "static"
    ].to_numpy()

    rep_values = paired[
        "repetition_aware"
    ].to_numpy()


    # --------------------------------------------------------
    # Differences
    # --------------------------------------------------------

    differences = (
        rep_values
        - static_values
    )

    mean_difference = np.mean(
        differences
    )

    median_difference = np.median(
        differences
    )


    # --------------------------------------------------------
    # Wilcoxon signed-rank test
    # --------------------------------------------------------

    if np.allclose(
        differences,
        0
    ):

        statistic = 0.0
        p_value = 1.0

    else:

        test = wilcoxon(
            rep_values,
            static_values,
            alternative="two-sided"
        )

        statistic = test.statistic
        p_value = test.pvalue


    # --------------------------------------------------------
    # Rank-biserial effect size
    # --------------------------------------------------------

    positive = np.sum(
        differences > 0
    )

    negative = np.sum(
        differences < 0
    )

    nonzero = positive + negative

    if nonzero > 0:

        rank_biserial = (
            (positive - negative)
            / nonzero
        )

    else:

        rank_biserial = 0.0


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print(
        "Static fold values:"
    )

    print(
        np.round(
            static_values,
            4
        )
    )

    print(
        "Repetition-aware fold values:"
    )

    print(
        np.round(
            rep_values,
            4
        )
    )

    print(
        "Fold differences:"
    )

    print(
        np.round(
            differences,
            4
        )
    )

    print(
        f"Mean difference: "
        f"{mean_difference:.6f}"
    )

    print(
        f"Median difference: "
        f"{median_difference:.6f}"
    )

    print(
        f"Wilcoxon statistic: "
        f"{statistic:.4f}"
    )

    print(
        f"Wilcoxon p-value: "
        f"{p_value:.6f}"
    )

    print(
        f"Rank-biserial effect: "
        f"{rank_biserial:.4f}"
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.append({

        "metric": metric,

        "n_paired_folds":
            len(paired),

        "static_mean":
            np.mean(static_values),

        "static_std":
            np.std(
                static_values,
                ddof=1
            ),

        "repetition_aware_mean":
            np.mean(rep_values),

        "repetition_aware_std":
            np.std(
                rep_values,
                ddof=1
            ),

        "mean_difference":
            mean_difference,

        "median_difference":
            median_difference,

        "wilcoxon_statistic":
            statistic,

        "wilcoxon_p_value":
            p_value,

        "rank_biserial_effect":
            rank_biserial
    })


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# MULTIPLE-COMPARISON CORRECTION
# ============================================================

# Only three primary metrics are tested.
# Apply Benjamini-Hochberg FDR correction.

p_values = results_df[
    "wilcoxon_p_value"
].to_numpy()

order = np.argsort(
    p_values
)

sorted_p = p_values[
    order
]

m = len(
    sorted_p
)

adjusted = np.empty(
    m
)

for i in range(
    m
):

    adjusted[i] = (
        sorted_p[i]
        * m
        / (i + 1)
    )


# Enforce monotonicity
for i in range(
    m - 2,
    -1,
    -1
):

    adjusted[i] = min(
        adjusted[i],
        adjusted[i + 1]
    )

adjusted = np.clip(
    adjusted,
    0,
    1
)

p_adjusted = np.empty(
    m
)

p_adjusted[
    order
] = adjusted


results_df[
    "wilcoxon_p_fdr"
] = p_adjusted


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH = os.path.join(
    OUT_DIR,
    "ablation_statistical_comparison.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL ABLATION STATISTICAL COMPARISON")
print("=" * 70)

print()

print(
    results_df.to_string(
        index=False
    )
)


print()
print("=" * 70)
print("STEP 22 COMPLETE")
print("=" * 70)

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_PATH
)