# ============================================================
# STEP 24 — FINAL RESULTS COMPARISON & CONSOLIDATION
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

ABLATION_DIR = os.path.join(
    BASE_DIR,
    "results",
    "ablation"
)

BENCHMARK_DIR = os.path.join(
    BASE_DIR,
    "results",
    "model_benchmark"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "final_comparison"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("STEP 24 — FINAL RESULTS COMPARISON")
print("=" * 70)


# ============================================================
# 1. ABLATION RESULTS
# ============================================================

ablation_path = os.path.join(
    ABLATION_DIR,
    "ablation_summary.csv"
)

ablation = pd.read_csv(
    ablation_path
)

print()
print("ABLATION RESULTS")
print("-" * 70)

print(
    ablation.to_string(
        index=False
    )
)


# ============================================================
# 2. MODEL BENCHMARK
# ============================================================

benchmark_path = os.path.join(
    BENCHMARK_DIR,
    "model_benchmark_summary.csv"
)

benchmark = pd.read_csv(
    benchmark_path
)

print()
print("MODEL BENCHMARK")
print("-" * 70)

print(
    benchmark.to_string(
        index=False
    )
)


# ============================================================
# 3. MODEL RANKING TABLE
# ============================================================

ranking = benchmark.copy()

# Rank independently by each metric
ranking["accuracy_rank"] = (
    ranking["accuracy_mean"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

ranking["f1_rank"] = (
    ranking["f1_mean"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

ranking["roc_auc_rank"] = (
    ranking["roc_auc_mean"]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


ranking = ranking.sort_values(
    "roc_auc_mean",
    ascending=False
)


ranking_path = os.path.join(
    OUTPUT_DIR,
    "model_benchmark_with_metric_ranks.csv"
)

ranking.to_csv(
    ranking_path,
    index=False
)


# ============================================================
# 4. TOP MODELS BY EACH METRIC
# ============================================================

top_accuracy = (
    benchmark
    .sort_values(
        "accuracy_mean",
        ascending=False
    )
    .head(5)
)

top_f1 = (
    benchmark
    .sort_values(
        "f1_mean",
        ascending=False
    )
    .head(5)
)

top_auc = (
    benchmark
    .sort_values(
        "roc_auc_mean",
        ascending=False
    )
    .head(5)
)


print()
print("=" * 70)
print("TOP 5 BY ACCURACY")
print("=" * 70)

print(
    top_accuracy[
        [
            "model",
            "accuracy_mean",
            "accuracy_std"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 70)
print("TOP 5 BY F1")
print("=" * 70)

print(
    top_f1[
        [
            "model",
            "f1_mean",
            "f1_std"
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 70)
print("TOP 5 BY ROC-AUC")
print("=" * 70)

print(
    top_auc[
        [
            "model",
            "roc_auc_mean",
            "roc_auc_std"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 5. REPRESENTATION COMPARISON
# ============================================================

representation_table = ablation.copy()

representation_path = os.path.join(
    OUTPUT_DIR,
    "representation_comparison.csv"
)

representation_table.to_csv(
    representation_path,
    index=False
)


# ============================================================
# 6. MERGE MAIN RESULTS INTO ONE SUMMARY
# ============================================================

# Keep the most important ablation columns
ablation_clean = ablation[
    [
        "representation",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std"
    ]
].copy()


# Rename for clarity
ablation_clean = ablation_clean.rename(
    columns={
        "accuracy_mean":
            "Accuracy_Mean",

        "accuracy_std":
            "Accuracy_SD",

        "f1_mean":
            "F1_Mean",

        "f1_std":
            "F1_SD",

        "roc_auc_mean":
            "ROC_AUC_Mean",

        "roc_auc_std":
            "ROC_AUC_SD"
    }
)


benchmark_clean = benchmark[
    [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std"
    ]
].copy()


benchmark_clean = benchmark_clean.rename(
    columns={
        "accuracy_mean":
            "Accuracy_Mean",

        "accuracy_std":
            "Accuracy_SD",

        "f1_mean":
            "F1_Mean",

        "f1_std":
            "F1_SD",

        "roc_auc_mean":
            "ROC_AUC_Mean",

        "roc_auc_std":
            "ROC_AUC_SD"
    }
)


# ============================================================
# 7. SAVE MASTER TABLE
# ============================================================

master_path = os.path.join(
    OUTPUT_DIR,
    "final_results_master.csv"
)

# Save as two sections in separate sheets is not possible
# in CSV, so save the model benchmark as the primary table
# and representation comparison separately.

benchmark_clean.to_csv(
    master_path,
    index=False
)


# ============================================================
# 8. RESEARCH SUMMARY TEXT
# ============================================================

summary_lines = []

summary_lines.append(
    "FINAL PROJECT RESULTS SUMMARY"
)

summary_lines.append(
    "=" * 70
)

summary_lines.append("")

summary_lines.append(
    "Dataset:"
)

summary_lines.append(
    "51 subjects; 30 DCP and 21 TD."
)

summary_lines.append(
    "One subject is treated as one prediction unit."
)

summary_lines.append("")

summary_lines.append(
    "Representation comparison:"
)

for _, row in ablation_clean.iterrows():

    summary_lines.append(
        f"{row['representation']}: "
        f"Accuracy = "
        f"{row['Accuracy_Mean']:.4f} "
        f"+/- "
        f"{row['Accuracy_SD']:.4f}; "
        f"F1 = "
        f"{row['F1_Mean']:.4f} "
        f"+/- "
        f"{row['F1_SD']:.4f}; "
        f"ROC-AUC = "
        f"{row['ROC_AUC_Mean']:.4f} "
        f"+/- "
        f"{row['ROC_AUC_SD']:.4f}"
    )


summary_lines.append("")

summary_lines.append(
    "Model benchmark:"
)

for _, row in (
    benchmark
    .sort_values(
        "roc_auc_mean",
        ascending=False
    )
    .iterrows()
):

    summary_lines.append(
        f"{row['model']}: "
        f"Accuracy = "
        f"{row['accuracy_mean']:.4f} "
        f"+/- "
        f"{row['accuracy_std']:.4f}; "
        f"F1 = "
        f"{row['f1_mean']:.4f} "
        f"+/- "
        f"{row['f1_std']:.4f}; "
        f"ROC-AUC = "
        f"{row['roc_auc_mean']:.4f} "
        f"+/- "
        f"{row['roc_auc_std']:.4f}"
    )


summary_lines.append("")

summary_lines.append(
    "Important interpretation:"
)

summary_lines.append(
    "Model performance differs by evaluation metric; "
    "therefore no single model is designated as an "
    "overall winner from the benchmark alone."
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "final_results_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            summary_lines
        )
    )


# ============================================================
# 9. PRINT OUTPUT FILES
# ============================================================

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    ranking_path
)

print(
    representation_path
)

print(
    master_path
)

print(
    summary_path
)


print()
print("=" * 70)
print("STEP 24 COMPLETE")
print("=" * 70)