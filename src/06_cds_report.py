"""
06_cds_report.py

Objective 7:
Final Clinical Decision Support output.

Uses the SAME explainability setup as Step 5:
- 5 biomarkers selected using RFE
- Mutual information retained for biomarker ranking
- Random Forest: 500 trees, max_depth=4, min_samples_leaf=2
- LIME local explanations

Produces:
- CDS JSON report
- Human-readable TXT report
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

from lime.lime_tabular import LimeTabularExplainer


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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

JSON_OUTPUT = (
    RESULTS_DIR
    / "cds_reports.json"
)

TXT_OUTPUT = (
    RESULTS_DIR
    / "cds_report.txt"
)


# ============================================================
# ALL BIOMARKERS
# ============================================================

ALL_BIOMARKER_COLS = [
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
# QUALITATIVE LABELS
# ============================================================

LABELS = {

    "Movement_Smoothness_Index": (
        "Movement Smoothness",
        ["High", "Moderate", "Reduced", "Low"],
    ),

    "Trajectory_Stability_Index": (
        "Trajectory Stability",
        ["High", "Moderate", "Reduced", "Poor"],
    ),

    "Reach_Efficiency_Index": (
        "Reach Efficiency",
        ["High", "Moderate", "Reduced", "Low"],
    ),

    "Joint_Coordination_Index": (
        "Joint Coordination",
        ["Strong", "Moderate", "Reduced", "Poor"],
    ),

    "Motion_Variability_Index": (
        "Movement Variability",
        ["High", "Moderate", "Reduced", "Low"],
    ),

    "Functional_Consistency_Index": (
        "Functional Consistency",
        ["High", "Moderate", "Reduced", "Low"],
    ),

    "Motor_Adaptability_Index": (
        "Motor Adaptability",
        ["High", "Moderate", "Reduced", "Low"],
    ),

    "Movement_Stability_Index": (
        "Movement Stability",
        ["High", "Moderate", "Reduced", "Low"],
    ),
}


# ============================================================
# QUALITATIVE INTERPRETATION
# ============================================================

def qualitative(z_value, bands):

    if pd.isna(z_value):
        return "Unavailable"

    if z_value > 0.75:
        return bands[0]

    if z_value > 0.10:
        return bands[1]

    if z_value > -0.75:
        return bands[2]

    return bands[3]


# ============================================================
# FRI INTERPRETATION
# ============================================================

def fri_band(fri):

    if pd.isna(fri):
        return "Unavailable"

    if fri >= 60:
        return "High"

    if fri >= 45:
        return "Moderate"

    if fri >= 30:
        return "Reduced"

    return "Low"


# ============================================================
# STEP 5-CONSISTENT RFE FEATURE SELECTION
# ============================================================

def select_features(X, y):

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # ========================================================
    # USE THE FINAL RFE SELECTION FROM STEP 5
    # ========================================================

    selected_features = [
        "Functional_Consistency_Index",
        "Motion_Variability_Index",
        "Movement_Smoothness_Index",
        "Movement_Stability_Index",
        "Trajectory_Stability_Index",
    ]

    selected_indices = [
        ALL_BIOMARKER_COLS.index(feature)
        for feature in selected_features
    ]

    X_selected = X_imp[:, selected_indices]

    # MI ranking retained for reporting only
    mi_scores = mutual_info_classif(
        X_imp,
        y,
        random_state=42
    )

    ranking = (
        pd.DataFrame({
            "biomarker": ALL_BIOMARKER_COLS,
            "mutual_information": mi_scores,
            "selected_for_cds": [
                feature in selected_features
                for feature in ALL_BIOMARKER_COLS
            ]
        })
        .sort_values(
            "mutual_information",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return (
        imputer,
        selected_features,
        X_selected,
        ranking,
    )


# ============================================================
# TRAIN STEP-5-CONSISTENT MODEL
# ============================================================

def train_model(
    X_selected,
    y,
):

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

    return model


# ============================================================
# LIME EXPLAINER
# ============================================================

def create_lime_explainer(
    X_selected,
    selected_features,
):

    return LimeTabularExplainer(
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


# ============================================================
# SINGLE SUBJECT REPORT
# ============================================================

def create_subject_report(
    row,
    row_position,
    X_selected,
    selected_features,
    model,
    explainer,
):

    instance = X_selected[
        row_position
    ]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = float(
        model.predict_proba(
            instance.reshape(1, -1)
        )[0, 1]
    )

    prediction = (
        "DCP"
        if probability >= 0.5
        else "TD"
    )

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

    explanation = (
        explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=len(
                selected_features
            ),
        )
    )

    lime_factors = []

    for feature, weight in (
        explanation.as_list()
    ):

        lime_factors.append(
            {
                "feature": feature,
                "weight": round(
                    float(weight),
                    4,
                ),
            }
        )

    # --------------------------------------------------------
    # Biomarker assessment
    # --------------------------------------------------------

    assessment = {}

    for column, (
        display_name,
        bands,
    ) in LABELS.items():

        value = row[column]

        assessment[
            display_name
        ] = {
            "value": (
                round(
                    float(value),
                    4,
                )
                if pd.notna(value)
                else None
            ),

            "interpretation":
                qualitative(
                    value,
                    bands,
                ),
        }

    # --------------------------------------------------------
    # Functional Resilience
    # --------------------------------------------------------

    fri = row[
        "Functional_Resilience_Index"
    ]

    fri_value = (
        round(
            float(fri),
            2,
        )
        if pd.notna(fri)
        else None
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    return {

        "subject":
            str(row["subject"]),

        "true_group":
            str(row["group"]),

        "performing_side":
            str(row["side"]),

        "predicted_group":
            prediction,

        "dcp_probability":
            round(
                probability,
                4,
            ),

        "functional_resilience_index":
            fri_value,

        "functional_resilience_band":
            fri_band(fri),

        "assessment":
            assessment,

        "lime_top_factors":
            lime_factors,

    }


# ============================================================
# TEXT REPORT
# ============================================================

def write_text_report(
    reports
):

    with open(
        TXT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "CLINICAL DECISION SUPPORT REPORT\n"
        )

        file.write(
            "=" * 70
            + "\n\n"
        )

        file.write(
            "Markerless Upper-Limb Performance Assessment\n"
        )

        file.write(
            "Digital Biomarker + Functional Resilience Analysis\n\n"
        )

        for label, report in (
            reports.items()
        ):

            file.write(
                "-" * 70
                + "\n"
            )

            file.write(
                f"Case: {label}\n"
            )

            file.write(
                f"Subject: "
                f"{report['subject']}\n"
            )

            file.write(
                f"True group: "
                f"{report['true_group']}\n"
            )

            file.write(
                f"Performing side: "
                f"{report['performing_side']}\n"
            )

            file.write(
                f"Predicted group: "
                f"{report['predicted_group']}\n"
            )

            file.write(
                f"DCP probability: "
                f"{report['dcp_probability']:.3f}\n"
            )

            file.write(
                f"Functional Resilience Index: "
                f"{report['functional_resilience_index']}\n"
            )

            file.write(
                f"FRI band: "
                f"{report['functional_resilience_band']}\n\n"
            )

            file.write(
                "BIOMARKER ASSESSMENT\n"
            )

            for name, details in (
                report[
                    "assessment"
                ].items()
            ):

                value = details[
                    "value"
                ]

                if value is None:
                    value_text = "Unavailable"
                else:
                    value_text = (
                        f"{value:.4f}"
                    )

                file.write(
                    f"  {name:30s} "
                    f"{value_text:>12s}   "
                    f"{details['interpretation']}\n"
                )

            file.write(
                "\nLIME TOP FACTORS\n"
            )

            for factor in (
                report[
                    "lime_top_factors"
                ]
            ):

                file.write(
                    f"  {factor['feature']}"
                    f"  weight="
                    f"{factor['weight']:+.4f}\n"
                )

            file.write(
                "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 6 — CLINICAL DECISION SUPPORT")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run 04_biomarkers.py first."
        )

    # --------------------------------------------------------
    # Load subject biomarkers
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE,
        dtype={
            "subject": str,
        },
    )

    print(
        f"\nSubjects: {len(df)}"
    )

    print(
        "\nGroups:"
    )

    print(
        df["group"].value_counts()
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X = df[
        ALL_BIOMARKER_COLS
    ].copy()

    y = (
        df["group"] == "DCP"
    ).astype(int).to_numpy()

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    (
        imputer,
        selected_features,
        X_selected,
        ranking,
    ) = select_features(
        X,
        y,
    )

    print(
        "\nFeatures selected for CDS:"
    )

    for feature in selected_features:
        print(
            f"  ✓ {feature}"
        )

    # --------------------------------------------------------
    # Save selected-feature ranking
    # --------------------------------------------------------

    ranking.to_csv(
        RESULTS_DIR
        / "cds_feature_ranking.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Train same RF configuration as Step 5
    # --------------------------------------------------------

    model = train_model(
        X_selected,
        y,
    )

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

    explainer = (
        create_lime_explainer(
            X_selected,
            selected_features,
        )
    )

    # --------------------------------------------------------
    # Model scores for every subject
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_selected
        )[:, 1]
    )

    predictions = np.where(
        probabilities >= 0.5,
        "DCP",
        "TD",
    )

    # --------------------------------------------------------
    # Representative cases
    # --------------------------------------------------------

    dcp = df[
        df["group"] == "DCP"
    ].copy()

    td = df[
        df["group"] == "TD"
    ].copy()

    if dcp.empty or td.empty:

        raise ValueError(
            "Both DCP and TD groups are required."
        )

    # Lowest-FRI DCP
    dcp_row = (
        dcp
        .sort_values(
            "Functional_Resilience_Index"
        )
        .iloc[0]
    )

    # Highest-FRI TD
    td_row = (
        td
        .sort_values(
            "Functional_Resilience_Index",
            ascending=False,
        )
        .iloc[0]
    )

    # Convert dataframe indices into positions
    dcp_position = (
        df.index.get_loc(
            dcp_row.name
        )
    )

    td_position = (
        df.index.get_loc(
            td_row.name
        )
    )

    # --------------------------------------------------------
    # Generate reports
    # --------------------------------------------------------

    reports = {}

    reports[
        "DCP_example"
    ] = create_subject_report(
        dcp_row,
        dcp_position,
        X_selected,
        selected_features,
        model,
        explainer,
    )

    reports[
        "TD_example"
    ] = create_subject_report(
        td_row,
        td_position,
        X_selected,
        selected_features,
        model,
        explainer,
    )

    # --------------------------------------------------------
    # Save complete CDS scores
    # --------------------------------------------------------

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
    ] = predictions

    scored[
        "prediction_correct"
    ] = (
        scored["predicted_group"]
        == scored["group"]
    )

    scored.to_csv(
        RESULTS_DIR
        / "cds_subject_scores.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            reports,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save TXT
    # --------------------------------------------------------

    write_text_report(
        reports
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CLINICAL DECISION SUPPORT RESULTS")
    print("=" * 60)

    for label, report in (
        reports.items()
    ):

        print(
            f"\n===== {label} ====="
        )

        print(
            f"Subject: "
            f"{report['subject']}"
        )

        print(
            f"True group: "
            f"{report['true_group']}"
        )

        print(
            f"Predicted group: "
            f"{report['predicted_group']}"
        )

        print(
            f"DCP probability: "
            f"{report['dcp_probability']:.3f}"
        )

        print(
            f"Functional Resilience Index: "
            f"{report['functional_resilience_index']}"
        )

        print(
            f"FRI band: "
            f"{report['functional_resilience_band']}"
        )

        print(
            "\nAssessment:"
        )

        for name, details in (
            report[
                "assessment"
            ].items()
        ):

            print(
                f"  {name:30s}: "
                f"{details['interpretation']}"
            )

        print(
            "\nTop LIME factors:"
        )

        for factor in (
            report[
                "lime_top_factors"
            ][:5]
        ):

            print(
                f"  {factor['feature']:45s}"
                f" weight={factor['weight']:+.3f}"
            )

    print("\n" + "=" * 60)
    print("STEP 6 COMPLETE")
    print("=" * 60)

    print(
        f"\nJSON output:\n"
        f"{JSON_OUTPUT}"
    )

    print(
        f"\nText output:\n"
        f"{TXT_OUTPUT}"
    )

    print(
        "\nAdditional output:"
    )

    print(
        RESULTS_DIR
        / "cds_subject_scores.csv"
    )

    print(
        RESULTS_DIR
        / "cds_feature_ranking.csv"
    )


if __name__ == "__main__":
    main()