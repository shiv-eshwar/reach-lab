# ============================================================
# STEP 11 — FEATURE STABILITY ANALYSIS
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "experiment_dataset.csv"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "feature_stability"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 11 — FEATURE STABILITY ANALYSIS")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

TARGET = "target"
GROUP = "subject"

y = df[TARGET]
groups = df[GROUP]

# Keep only numeric predictors.
# Metadata columns such as side/group are excluded.
X = df.drop(columns=[TARGET, GROUP], errors="ignore")
X = X.select_dtypes(include=[np.number])

feature_names = X.columns.tolist()

print()
print(f"Dataset shape: {df.shape}")
print(f"Subjects: {df[GROUP].nunique()}")
print(f"Features: {len(feature_names)}")


# ============================================================
# OUTER SUBJECT-LEVEL CV
# ============================================================

outer_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# RANDOM FOREST PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "selector",
        SelectKBest(
            score_func=mutual_info_classif
        )
    ),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )
    )
])


# ============================================================
# INNER-CV HYPERPARAMETER SEARCH
# ============================================================

param_grid = {
    "selector__k": [10, 20, 40],

    "model__n_estimators": [200],

    "model__max_depth": [None],

    "model__max_features": ["sqrt", "log2"],

    "model__min_samples_leaf": [1, 2, 4]
}


# ============================================================
# STORAGE
# ============================================================

selection_counts = {
    feature: 0
    for feature in feature_names
}

importance_records = []

fold_feature_records = []


# ============================================================
# OUTER CV
# ============================================================

for fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups),
    start=1
):

    print()
    print("=" * 70)
    print(f"OUTER FOLD {fold}/5")
    print("=" * 70)

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    groups_train = groups.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    print(f"Train subjects: {groups_train.nunique()}")
    print(f"Test subjects: {groups.iloc[test_idx].nunique()}")

    # --------------------------------------------------------
    # Inner subject-level CV
    # --------------------------------------------------------

    inner_cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
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

    print("Best parameters:")
    print(search.best_params_)

    # --------------------------------------------------------
    # Selected features
    # --------------------------------------------------------

    selector = best_model.named_steps["selector"]

    selected_mask = selector.get_support()

    selected_features = [
        feature_names[i]
        for i, selected in enumerate(selected_mask)
        if selected
    ]

    print()
    print(f"Selected features: {len(selected_features)}")

    for feature in selected_features:
        print(f"  {feature}")

        selection_counts[feature] += 1

    fold_feature_records.append({
        "fold": fold,
        "selected_features": selected_features
    })

    # --------------------------------------------------------
    # Random Forest feature importance
    # --------------------------------------------------------

    rf = best_model.named_steps["model"]

    importances = rf.feature_importances_

    for feature, importance in zip(
        selected_features,
        importances
    ):
        importance_records.append({
            "fold": fold,
            "feature": feature,
            "importance": importance
        })


# ============================================================
# FEATURE SELECTION STABILITY
# ============================================================

print()
print("=" * 70)
print("FEATURE SELECTION STABILITY")
print("=" * 70)

stability_df = pd.DataFrame([
    {
        "feature": feature,
        "selection_count": count,
        "selection_frequency": count / 5
    }
    for feature, count in selection_counts.items()
])

stability_df = stability_df.sort_values(
    by=["selection_count", "feature"],
    ascending=[False, True]
).reset_index(drop=True)


# ============================================================
# MEAN FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(importance_records)

if not importance_df.empty:

    mean_importance = (
        importance_df
        .groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={
            "mean": "mean_importance",
            "std": "importance_std"
        })
    )

    stability_df = stability_df.merge(
        mean_importance,
        on="feature",
        how="left"
    )

else:

    stability_df["mean_importance"] = np.nan
    stability_df["importance_std"] = np.nan


# ============================================================
# SAVE RESULTS
# ============================================================

stability_path = os.path.join(
    OUT_DIR,
    "feature_stability.csv"
)

importance_path = os.path.join(
    OUT_DIR,
    "feature_importance_by_fold.csv"
)

fold_features_path = os.path.join(
    OUT_DIR,
    "selected_features_by_fold.json"
)

stability_df.to_csv(
    stability_path,
    index=False
)

importance_df.to_csv(
    importance_path,
    index=False
)

with open(
    fold_features_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        fold_feature_records,
        f,
        indent=4
    )


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print()

print(
    stability_df[
        stability_df["selection_count"] > 0
    ].head(30).to_string(index=False)
)


# ============================================================
# STABLE FEATURES
# ============================================================

stable_features = stability_df[
    stability_df["selection_count"] >= 4
].copy()

stable_path = os.path.join(
    OUT_DIR,
    "stable_features.csv"
)

stable_features.to_csv(
    stable_path,
    index=False
)

print()
print("=" * 70)
print("STABLE FEATURES (SELECTED IN >= 4/5 OUTER FOLDS)")
print("=" * 70)

if len(stable_features) == 0:

    print("No feature was selected in >= 4/5 folds.")

else:

    print(
        stable_features.to_string(index=False)
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("STEP 11 COMPLETE")
print("=" * 70)

print()
print("Results saved to:")
print(stability_path)
print(importance_path)
print(fold_features_path)
print(stable_path)