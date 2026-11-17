"""
04_biomarkers.py

Objectives 3 & 6:
- Engineer explainable kinematic digital biomarkers
- Model functional consistency across rep1-rep8
- Model motor adaptability
- Compute Functional Resilience Index (FRI)

Output:
One row per subject.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REP_FEATURES_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rep_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "subject_biomarkers.csv"
)


# ============================================================
# HELPERS
# ============================================================

def zscore(series):
    """Cohort-standardized score."""

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    std = series.std(
        ddof=0
    )

    if pd.isna(std) or std == 0:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        series - series.mean()
    ) / std


def mean_zscore(
    dataframe,
    columns,
    directions,
):
    """
    Combine standardized features.

    directions:
        +1 -> higher is better
        -1 -> lower is better
    """

    components = []

    for column, direction in zip(
        columns,
        directions,
    ):

        components.append(
            direction
            * zscore(
                dataframe[column]
            )
        )

    if not components:
        return pd.Series(
            np.nan,
            index=dataframe.index,
        )

    return pd.concat(
        components,
        axis=1,
    ).mean(axis=1)


# ============================================================
# REPETITION-LEVEL ANALYSIS
# ============================================================

def calculate_consistency(sub):
    """
    Functional consistency across eight repetitions.

    Lower normalized variability indicates
    greater consistency.
    """

    cvs = []

    features = [
        "trajectory_length_mm",
        "movement_duration_s",
        "peak_velocity_mm_s",
        "path_efficiency",
    ]

    for column in features:

        values = pd.to_numeric(
            sub[column],
            errors="coerce",
        ).dropna()

        if len(values) >= 3:

            mean_value = values.mean()

            if abs(mean_value) > 1e-12:

                cv = (
                    values.std(ddof=0)
                    / abs(mean_value)
                )

                cvs.append(cv)

    if not cvs:
        return np.nan

    return 1.0 - np.clip(
        np.mean(cvs),
        0.0,
        1.0,
    )


def calculate_stability(sub):
    """
    Movement stability based on repetition-to-repetition
    variability of joint ROM.
    """

    cvs = []

    features = [
        "shoulder_rom_deg",
        "elbow_rom_deg",
        "wrist_rom_deg",
    ]

    for column in features:

        values = pd.to_numeric(
            sub[column],
            errors="coerce",
        ).dropna()

        if len(values) >= 3:

            mean_value = values.mean()

            if abs(mean_value) > 1e-12:

                cv = (
                    values.std(ddof=0)
                    / abs(mean_value)
                )

                cvs.append(cv)

    if not cvs:
        return np.nan

    return 1.0 - np.clip(
        np.mean(cvs),
        0.0,
        1.0,
    )


def calculate_adaptability(sub):
    """
    Compare movement quality during early repetitions
    (rep1-rep4) with later repetitions (rep5-rep8).

    Positive value = improvement.
    """

    early = sub[
        sub["rep"].between(1, 4)
    ]

    late = sub[
        sub["rep"].between(5, 8)
    ]

    if (
        early.empty
        or late.empty
    ):
        return np.nan

    components = []

    # Higher is better
    higher_is_better = [
        "path_efficiency",
    ]

    for column in higher_is_better:

        early_value = early[column].mean()
        late_value = late[column].mean()

        if (
            pd.notna(early_value)
            and pd.notna(late_value)
            and abs(early_value) > 1e-12
        ):

            change = (
                late_value
                - early_value
            ) / abs(early_value)

            components.append(
                change
            )

    # Lower is better
    lower_is_better = [
        "smoothness_norm_jerk",
        "variability_speed_cv",
        "stability_lateral_deviation_mm",
        "movement_duration_s",
    ]

    for column in lower_is_better:

        early_value = early[column].mean()
        late_value = late[column].mean()

        if (
            pd.notna(early_value)
            and pd.notna(late_value)
            and abs(early_value) > 1e-12
        ):

            change = (
                early_value
                - late_value
            ) / abs(early_value)

            components.append(
                change
            )

    if not components:
        return np.nan

    return np.mean(
        components
    )


# ============================================================
# SUBJECT-LEVEL REPETITION FEATURES
# ============================================================

def build_subject_level_features(rep_df):

    rows = []

    for subject, sub in rep_df.groupby(
        "subject",
        sort=True,
    ):

        sub = (
            sub
            .sort_values("rep")
            .reset_index(drop=True)
        )

        row = {
            "subject": subject,
            "group": sub["group"].iloc[0],
            "side": sub["side"].iloc[0],
        }

        # ----------------------------------------------------
        # Functional consistency
        # ----------------------------------------------------

        row[
            "consistency_score"
        ] = calculate_consistency(
            sub
        )

        # ----------------------------------------------------
        # Movement stability
        # ----------------------------------------------------

        row[
            "movement_stability_score"
        ] = calculate_stability(
            sub
        )

        # ----------------------------------------------------
        # Motor adaptability
        # ----------------------------------------------------

        row[
            "adaptive_motor_control_index"
        ] = calculate_adaptability(
            sub
        )

        # ----------------------------------------------------
        # Repetition-level summary statistics
        # ----------------------------------------------------

        summary_features = [
            "trajectory_length_mm",
            "path_efficiency",
            "reach_distance_mm",
            "movement_duration_s",
            "peak_velocity_mm_s",
            "mean_velocity_mm_s",
            "mean_acceleration_mm_s2",
            "peak_acceleration_mm_s2",
            "shoulder_rom_deg",
            "elbow_rom_deg",
            "wrist_rom_deg",
            "smoothness_norm_jerk",
            "smoothness_sparc",
            "coordination_elbow_wrist",
            "coordination_shoulder_elbow",
            "variability_speed_cv",
            "shoulder_angle_sd",
            "elbow_angle_sd",
            "wrist_angle_sd",
            "stability_lateral_deviation_mm",
        ]

        for feature in summary_features:

            values = pd.to_numeric(
                sub[feature],
                errors="coerce",
            )

            row[
                f"mean_{feature}"
            ] = values.mean()

            row[
                f"sd_{feature}"
            ] = values.std(
                ddof=0
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# BIOMARKER ENGINEERING
# ============================================================

def engineer_biomarkers(subject_df):

    df = subject_df.copy()

    # --------------------------------------------------------
    # 1. Movement Smoothness Index
    # --------------------------------------------------------
    # Lower normalized jerk = smoother movement.

    df[
        "Movement_Smoothness_Index"
    ] = mean_zscore(
        df,
        [
            "mean_smoothness_norm_jerk",
            "mean_smoothness_sparc",
        ],
        [
            -1,
            1,
        ],
    )

    # --------------------------------------------------------
    # 2. Trajectory Stability Index
    # --------------------------------------------------------
    # Lower deviation = more stable.
    #
    # Convex hull and overshoot are retained as published
    # validation features when available in the dataset.
    #
    # Our newly extracted lateral deviation is also included.

    stability_columns = [
        "mean_stability_lateral_deviation_mm",
    ]

    stability_directions = [
        -1,
    ]

    if "mean_trajectory_length_mm" in df.columns:
        stability_columns.append(
            "sd_trajectory_length_mm"
        )
        stability_directions.append(-1)

    df[
        "Trajectory_Stability_Index"
    ] = mean_zscore(
        df,
        stability_columns,
        stability_directions,
    )

    # --------------------------------------------------------
    # 3. Reach Efficiency Index
    # --------------------------------------------------------
    # Higher path efficiency = better.

    df[
        "Reach_Efficiency_Index"
    ] = mean_zscore(
        df,
        [
            "mean_path_efficiency",
            "mean_reach_distance_mm",
        ],
        [
            1,
            1,
        ],
    )

    # --------------------------------------------------------
    # 4. Joint Coordination Index
    # --------------------------------------------------------

    df[
        "Joint_Coordination_Index"
    ] = mean_zscore(
        df,
        [
            "mean_coordination_elbow_wrist",
            "mean_coordination_shoulder_elbow",
        ],
        [
            1,
            1,
        ],
    )

    # --------------------------------------------------------
    # 5. Motion Variability Index
    # --------------------------------------------------------
    # Higher value = greater variability.

    df[
        "Motion_Variability_Index"
    ] = mean_zscore(
        df,
        [
            "mean_variability_speed_cv",
            "mean_shoulder_angle_sd",
            "mean_elbow_angle_sd",
            "mean_wrist_angle_sd",
        ],
        [
            1,
            1,
            1,
            1,
        ],
    )

    # --------------------------------------------------------
    # 6. Functional Consistency Index
    # --------------------------------------------------------

    df[
        "Functional_Consistency_Index"
    ] = zscore(
        df[
            "consistency_score"
        ]
    )

    # --------------------------------------------------------
    # 7. Motor Adaptability Index
    # --------------------------------------------------------

    df[
        "Motor_Adaptability_Index"
    ] = zscore(
        df[
            "adaptive_motor_control_index"
        ]
    )

    # --------------------------------------------------------
    # 8. Movement Stability Index
    # --------------------------------------------------------

    df[
        "Movement_Stability_Index"
    ] = zscore(
        df[
            "movement_stability_score"
        ]
    )

    # --------------------------------------------------------
    # Functional Resilience Index
    # --------------------------------------------------------
    #
    # Components:
    #   Functional Consistency
    #   Movement Stability
    #   Motor Adaptability
    #   Motion Variability
    #
    # Higher FRI = better functional resilience.

    df[
        "Functional_Resilience_Raw"
    ] = (
        df[
            "Functional_Consistency_Index"
        ]
        + df[
            "Movement_Stability_Index"
        ]
        + df[
            "Motor_Adaptability_Index"
        ]
        - df[
            "Motion_Variability_Index"
        ]
    ) / 4.0

    # --------------------------------------------------------
    # Standardized readable FRI
    # --------------------------------------------------------
    #
    # Convert the raw score into a centered 0-100 score.
    #
    # Unlike the previous implementation, this does NOT use
    # dataset minimum and maximum values.

    raw = df[
        "Functional_Resilience_Raw"
    ]

    mean = raw.mean()
    std = raw.std(ddof=0)

    if (
        pd.isna(std)
        or std == 0
    ):

        df[
            "Functional_Resilience_Index"
        ] = 50.0

    else:

        standardized = (
            (raw - mean)
            / std
        )

        df[
            "Functional_Resilience_Index"
        ] = (
            50
            + 15 * standardized
        ).clip(
            0,
            100,
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 4 — DIGITAL BIOMARKER ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------------
    # Input check
    # --------------------------------------------------------

    if not REP_FEATURES_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{REP_FEATURES_FILE}\n\n"
            "Run 03_features.py first."
        )

    # --------------------------------------------------------
    # Load repetition-level features
    # --------------------------------------------------------

    rep_df = pd.read_csv(
        REP_FEATURES_FILE,
        dtype={
            "subject": str,
        },
    )

    print(
        f"\nInput records: {len(rep_df)}"
    )

    print(
        f"Subjects: "
        f"{rep_df['subject'].nunique()}"
    )

    # --------------------------------------------------------
    # Verify eight repetitions
    # --------------------------------------------------------

    # Preserve the established biomarker schema for older reports.
    if "mean_lateral_deviation_mm" in rep_df:
        rep_df["stability_lateral_deviation_mm"] = rep_df["mean_lateral_deviation_mm"]

    repetition_counts = (
        rep_df
        .groupby("subject")["rep"]
        .nunique()
    )

    complete_subjects = (
        repetition_counts == 8
    ).sum()

    print(
        f"Subjects with all 8 repetitions: "
        f"{complete_subjects}/"
        f"{len(repetition_counts)}"
    )

    # --------------------------------------------------------
    # Build subject-level data
    # --------------------------------------------------------

    subject_df = (
        build_subject_level_features(
            rep_df
        )
    )

    print(
        f"\nSubject-level records: "
        f"{len(subject_df)}"
    )

    # --------------------------------------------------------
    # Engineer biomarkers
    # --------------------------------------------------------

    biomarkers = engineer_biomarkers(
        subject_df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    biomarkers.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    biomarker_columns = [
        "Movement_Smoothness_Index",
        "Trajectory_Stability_Index",
        "Reach_Efficiency_Index",
        "Joint_Coordination_Index",
        "Motion_Variability_Index",
        "Functional_Consistency_Index",
        "Motor_Adaptability_Index",
        "Movement_Stability_Index",
        "Functional_Resilience_Index",
    ]

    print("\n" + "=" * 60)
    print("BIOMARKER ENGINEERING COMPLETE")
    print("=" * 60)

    print(
        f"\nOutput shape: "
        f"{biomarkers.shape}"
    )

    print(
        f"Output file:\n"
        f"{OUTPUT_FILE}"
    )

    print("\nBiomarkers generated:")

    for column in biomarker_columns:
        print(
            f"  ✓ {column}"
        )

    print("\nBiomarker summary:")

    print(
        biomarkers[
            biomarker_columns
        ].describe().T[
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
    )

    print("\nGroup-wise FRI:")

    print(
        biomarkers
        .groupby("group")[
            "Functional_Resilience_Index"
        ]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print("\nFirst 5 subjects:")

    print(
        biomarkers[
            [
                "subject",
                "group",
                "side",
            ]
            + biomarker_columns
        ].head()
    )


if __name__ == "__main__":
    main()