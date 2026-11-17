from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rep_features.csv"
)

OUTPUT_LONG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "repetition_trajectories_long.csv"
)

OUTPUT_WIDE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "repetition_trajectories_wide.csv"
)


# ============================================================
# REPETITION-LEVEL FEATURES
# ============================================================

TRAJECTORY_FEATURES = [
    "trajectory_length_mm",
    "path_efficiency",
    "reach_distance_mm",
    "position_error_mm",
    "mean_lateral_deviation_mm",
    "max_lateral_deviation_mm",

    "movement_duration_s",
    "peak_velocity_mm_s",
    "mean_velocity_mm_s",
    "mean_acceleration_mm_s2",
    "peak_acceleration_mm_s2",

    "shoulder_rom_deg",
    "elbow_rom_deg",
    "wrist_rom_deg",

    "shoulder_angvel_mean",
    "elbow_angvel_mean",
    "wrist_angvel_mean",

    "shoulder_angacc_mean",
    "elbow_angacc_mean",
    "wrist_angacc_mean",

    "smoothness_norm_jerk",
    "smoothness_sparc",

    "coordination_elbow_wrist",
    "coordination_shoulder_elbow",

    "variability_speed_cv",

    "shoulder_angle_sd",
    "elbow_angle_sd",
    "wrist_angle_sd",

]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 7 — REPETITION-AWARE REPRESENTATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 03_features.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"subject": str}
    )

    print(f"\nInput shape: {df.shape}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required_columns = [
        "subject",
        "side",
        "group",
        "rep",
    ] + TRAJECTORY_FEATURES

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    print(f"Subjects: {df['subject'].nunique()}")
    print(f"Groups:\n{df.groupby('group')['subject'].nunique()}")
    print(
        "\nRepetitions per subject:\n"
        f"{df.groupby('subject')['rep'].nunique().value_counts().sort_index()}"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        ["subject", "rep"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    long_df = df[
        [
            "subject",
            "side",
            "group",
            "rep",
        ] + TRAJECTORY_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Convert numerical features
    # --------------------------------------------------------

    for column in TRAJECTORY_FEATURES:
        long_df[column] = pd.to_numeric(
            long_df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Validate unique subject-repetition records
    # --------------------------------------------------------

    duplicates = long_df.duplicated(
        ["subject", "rep"]
    ).sum()

    if duplicates != 0:
        raise ValueError(
            f"Found {duplicates} duplicate subject-repetition records."
        )

    # --------------------------------------------------------
    # Create wide repetition representation
    #
    # Example:
    #
    # subject
    # trajectory_length_mm_rep1
    # trajectory_length_mm_rep2
    # ...
    # trajectory_length_mm_rep8
    # --------------------------------------------------------

    metadata = (
        long_df[
            ["subject", "side", "group"]
        ]
        .drop_duplicates("subject")
        .set_index("subject")
    )

    wide = metadata.copy()

    for feature in TRAJECTORY_FEATURES:

        pivot = long_df.pivot(
            index="subject",
            columns="rep",
            values=feature
        )

        pivot = pivot.reindex(
            columns=range(1, 9)
        )

        pivot.columns = [
            f"{feature}_rep{rep}"
            for rep in pivot.columns
        ]

        wide = wide.join(pivot)

    wide = wide.reset_index()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_LONG.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    long_df.to_csv(
        OUTPUT_LONG,
        index=False
    )

    wide.to_csv(
        OUTPUT_WIDE,
        index=False
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_rows = long_df["subject"].nunique() * 8

    print("\n" + "=" * 70)
    print("REPETITION-AWARE REPRESENTATION COMPLETE")
    print("=" * 70)

    print(f"\nLong representation shape: {long_df.shape}")
    print(f"Wide representation shape: {wide.shape}")

    print(f"\nExpected long rows: {expected_rows}")
    print(f"Actual long rows:   {len(long_df)}")

    print(
        f"\nUnique subjects: {wide['subject'].nunique()}"
    )

    print(
        f"Wide feature columns: "
        f"{len(wide.columns) - 3}"
    )

    print("\nOutput files:")
    print(f"Long: {OUTPUT_LONG}")
    print(f"Wide: {OUTPUT_WIDE}")

    print("\nFirst 5 wide records:")
    print(
        wide.head().to_string(index=False)
    )


if __name__ == "__main__":
    main()