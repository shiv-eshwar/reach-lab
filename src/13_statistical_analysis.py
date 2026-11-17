# ============================================================
# STEP 13 — STATISTICAL EFFECT-SIZE ANALYSIS
# ============================================================

import os
import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu


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

STABLE_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_stability",
    "stable_features.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "statistical_analysis"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 13 — STATISTICAL EFFECT-SIZE ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

stable_df = pd.read_csv(STABLE_PATH)

stable_features = stable_df[
    "feature"
].tolist()

print()
print(f"Dataset shape: {df.shape}")
print(f"Subjects: {df['subject'].nunique()}")
print(f"Stable features: {len(stable_features)}")


# ============================================================
# GROUPS
# ============================================================

td = df[df["target"] == 0]
dcp = df[df["target"] == 1]

print()
print(f"TD subjects: {len(td)}")
print(f"DCP subjects: {len(dcp)}")


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

results = []


for feature in stable_features:

    td_values = pd.to_numeric(
        td[feature],
        errors="coerce"
    ).dropna()

    dcp_values = pd.to_numeric(
        dcp[feature],
        errors="coerce"
    ).dropna()

    if len(td_values) < 3 or len(dcp_values) < 3:
        continue

    # --------------------------------------------------------
    # Mann-Whitney U
    # --------------------------------------------------------

    try:

        u_stat, p_value = mannwhitneyu(
            td_values,
            dcp_values,
            alternative="two-sided"
        )

    except Exception:

        u_stat = np.nan
        p_value = np.nan

    # --------------------------------------------------------
    # Rank-biserial correlation
    # --------------------------------------------------------

    n1 = len(td_values)
    n2 = len(dcp_values)

    if n1 > 0 and n2 > 0 and not np.isnan(u_stat):

        rank_biserial = (
            1 -
            (2 * u_stat) /
            (n1 * n2)
        )

    else:

        rank_biserial = np.nan

    # --------------------------------------------------------
    # Cohen's d
    # --------------------------------------------------------

    mean_td = td_values.mean()
    mean_dcp = dcp_values.mean()

    std_td = td_values.std(ddof=1)
    std_dcp = dcp_values.std(ddof=1)

    pooled_std = np.sqrt(
        (
            (n1 - 1) * std_td ** 2
            +
            (n2 - 1) * std_dcp ** 2
        )
        /
        (n1 + n2 - 2)
    )

    if pooled_std > 0:

        cohens_d = (
            mean_dcp - mean_td
        ) / pooled_std

    else:

        cohens_d = np.nan

    # --------------------------------------------------------
    # Median difference
    # --------------------------------------------------------

    median_td = td_values.median()
    median_dcp = dcp_values.median()

    results.append({

        "feature": feature,

        "TD_n": n1,
        "DCP_n": n2,

        "TD_mean": mean_td,
        "DCP_mean": mean_dcp,

        "TD_median": median_td,
        "DCP_median": median_dcp,

        "TD_std": std_td,
        "DCP_std": std_dcp,

        "median_difference":
            median_dcp - median_td,

        "cohens_d":
            cohens_d,

        "rank_biserial":
            rank_biserial,

        "mann_whitney_U":
            u_stat,

        "p_value":
            p_value
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

if not results_df.empty:

    # Multiple-testing correction
    results_df["p_adjusted"] = np.nan

    valid = results_df["p_value"].notna()

    pvals = results_df.loc[
        valid,
        "p_value"
    ].values

    # Benjamini-Hochberg FDR
    order = np.argsort(pvals)

    ranked = pvals[order]

    m = len(ranked)

    adjusted = np.empty(m)

    for i in range(m):

        rank = i + 1

        adjusted[i] = (
            ranked[i] * m / rank
        )

    # enforce monotonicity
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    corrected = np.empty(m)

    corrected[order] = np.minimum(
        adjusted,
        1.0
    )

    results_df.loc[
        valid,
        "p_adjusted"
    ] = corrected


# ============================================================
# EFFECT-SIZE CATEGORY
# ============================================================

def effect_category(d):

    if pd.isna(d):
        return "NA"

    a = abs(d)

    if a < 0.2:
        return "negligible"

    elif a < 0.5:
        return "small"

    elif a < 0.8:
        return "medium"

    else:
        return "large"


results_df[
    "effect_size_category"
] = results_df[
    "cohens_d"
].apply(effect_category)


# ============================================================
# SORT
# ============================================================

results_df = results_df.sort_values(
    by="p_adjusted",
    na_position="last"
).reset_index(
    drop=True
)


# ============================================================
# SAVE
# ============================================================

output_path = os.path.join(
    OUT_DIR,
    "stable_feature_statistics.csv"
)

results_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("STATISTICAL RESULTS")
print("=" * 70)

if results_df.empty:

    print("No features could be analysed.")

else:

    display_columns = [
        "feature",
        "TD_mean",
        "DCP_mean",
        "TD_median",
        "DCP_median",
        "cohens_d",
        "rank_biserial",
        "p_value",
        "p_adjusted",
        "effect_size_category"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# SIGNIFICANT FEATURES
# ============================================================

significant = results_df[
    results_df["p_adjusted"] < 0.05
].copy()

significant_path = os.path.join(
    OUT_DIR,
    "significant_features.csv"
)

significant.to_csv(
    significant_path,
    index=False
)


print()
print("=" * 70)
print("FDR-SIGNIFICANT FEATURES")
print("=" * 70)

if significant.empty:

    print(
        "No features remained significant after "
        "Benjamini-Hochberg FDR correction."
    )

else:

    print(
        significant[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 13 COMPLETE")
print("=" * 70)

print()
print("Results saved to:")
print(output_path)
print(significant_path)