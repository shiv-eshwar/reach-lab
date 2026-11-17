# ============================================================
# STEP 14 — GENERATE RESEARCH FIGURES
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ABLATION_PATH = os.path.join(
    BASE_DIR,
    "results",
    "ablation",
    "ablation_summary.csv"
)

STABILITY_PATH = os.path.join(
    BASE_DIR,
    "results",
    "feature_stability",
    "feature_stability.csv"
)

STATS_PATH = os.path.join(
    BASE_DIR,
    "results",
    "statistical_analysis",
    "stable_feature_statistics.csv"
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
    "figures"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# GENERAL PLOT SETTINGS
# ============================================================

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 14 — GENERATE RESEARCH FIGURES")
print("=" * 70)

ablation = pd.read_csv(
    ABLATION_PATH
)

stability = pd.read_csv(
    STABILITY_PATH
)

stats = pd.read_csv(
    STATS_PATH
)

df = pd.read_csv(
    DATA_PATH
)

print()
print(f"Ablation rows: {len(ablation)}")
print(f"Stability rows: {len(stability)}")
print(f"Statistical rows: {len(stats)}")


# ============================================================
# FIGURE 1 — ABLATION PERFORMANCE
# ============================================================

print()
print("Generating Figure 1...")

representations = ablation[
    "representation"
].tolist()

accuracy = ablation[
    "accuracy_mean"
].values

f1 = ablation[
    "f1_mean"
].values

roc_auc = ablation[
    "roc_auc_mean"
].values

x = np.arange(
    len(representations)
)

width = 0.25

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.bar(
    x - width,
    accuracy,
    width,
    label="Accuracy"
)

ax.bar(
    x,
    f1,
    width,
    label="F1"
)

ax.bar(
    x + width,
    roc_auc,
    width,
    label="ROC-AUC"
)

ax.set_ylabel(
    "Mean score"
)

ax.set_title(
    "Ablation Study: Representation Performance"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    representations,
    rotation=15
)

ax.set_ylim(
    0,
    1.1
)

ax.legend()

ax.grid(
    axis="y",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "Figure_1_ablation_performance.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 2 — FEATURE STABILITY
# ============================================================

print("Generating Figure 2...")

stable_plot = stability[
    stability["selection_count"] > 0
].copy()

stable_plot = stable_plot.sort_values(
    "selection_count",
    ascending=True
).tail(15)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

ax.barh(
    stable_plot["feature"],
    stable_plot["selection_frequency"]
)

ax.set_xlabel(
    "Selection frequency across outer folds"
)

ax.set_title(
    "Feature Selection Stability"
)

ax.set_xlim(
    0,
    1.05
)

ax.grid(
    axis="x",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "Figure_2_feature_stability.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 3 — EFFECT SIZE
# ============================================================

print("Generating Figure 3...")

effect_plot = stats.copy()

effect_plot = effect_plot.sort_values(
    "cohens_d"
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

ax.barh(
    effect_plot["feature"],
    effect_plot["cohens_d"]
)

ax.axvline(
    0,
    linewidth=1
)

ax.set_xlabel(
    "Cohen's d"
)

ax.set_title(
    "Effect Sizes of Stable Features"
)

ax.grid(
    axis="x",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "Figure_3_effect_sizes.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# FIGURE 4 — GROUP DISTRIBUTIONS
# ============================================================

print("Generating Figure 4...")

features_for_distribution = stats[
    "feature"
].tolist()

# Create one figure per feature.
# This avoids compressed multi-panel plots.

for i, feature in enumerate(
    features_for_distribution,
    start=1
):

    td_values = df.loc[
        df["target"] == 0,
        feature
    ].dropna()

    dcp_values = df.loc[
        df["target"] == 1,
        feature
    ].dropna()

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        [
            td_values,
            dcp_values
        ],
        labels=[
            "TD",
            "DCP"
        ]
    )

    ax.set_ylabel(
        feature
    )

    ax.set_title(
        f"TD vs DCP: {feature}"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    filename = (
        f"Figure_4_{i:02d}_"
        f"{feature}.png"
    )

    fig.savefig(
        os.path.join(
            OUT_DIR,
            filename
        ),
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FIGURE 5 — FEATURE IMPORTANCE
# ============================================================

print("Generating Figure 5...")

importance_plot = stability[
    stability["selection_count"] > 0
].copy()

importance_plot = importance_plot.sort_values(
    "mean_importance",
    ascending=True
).tail(15)

fig, ax = plt.subplots(
    figsize=(9, 7)
)

ax.barh(
    importance_plot["feature"],
    importance_plot["mean_importance"]
)

ax.set_xlabel(
    "Mean Random Forest importance"
)

ax.set_title(
    "Stable Feature Importance"
)

ax.grid(
    axis="x",
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "Figure_5_feature_importance.png"
    ),
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# SAVE FIGURE INDEX
# ============================================================

figure_files = sorted(
    [
        f
        for f in os.listdir(OUT_DIR)
        if f.endswith(".png")
    ]
)

index_path = os.path.join(
    OUT_DIR,
    "figure_index.txt"
)

with open(
    index_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 14 — FIGURE INDEX\n"
    )

    f.write(
        "=======================\n\n"
    )

    for filename in figure_files:

        f.write(
            filename + "\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 14 COMPLETE")
print("=" * 70)

print()
print("Figures saved to:")
print(OUT_DIR)

print()
print("Generated figures:")

for filename in figure_files:

    print(
        f"  {filename}"
    )