from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "repetition_trajectories_long.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "repetition_descriptors.csv"
)


# ============================================================
# FEATURES
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
# DESCRIPTOR FUNCTION
# ============================================================

def calculate_descriptors(values):

    values = np.asarray(values, dtype=float)

    valid = np.isfinite(values)

    if valid.sum() == 0:
        return {
            "mean": np.nan,
            "sd": np.nan,
            "cv": np.nan,
            "slope": np.nan,
            "early_late_change": np.nan,
            "monotonicity": np.nan,
        }

    x = np.arange(1, len(values) + 1)
    xv = x[valid]
    yv = values[valid]

    mean_value = np.mean(yv)

    if len(yv) >= 2:
        slope = np.polyfit(xv, yv, 1)[0]

        differences = np.diff(yv)

        if len(differences) > 0:
            monotonicity = (
                np.sum(np.sign(differences))
                / len(differences)
            )
        else:
            monotonicity = np.nan
    else:
        slope = np.nan
        monotonicity = np.nan

    sd_value = np.std(yv, ddof=0)

    if abs(mean_value) > 1e-12:
        cv_value = sd_value / abs(mean_value)
    else:
        cv_value = np.nan

    early = values[:4]
    late = values[4:]

    early_valid = early[np.isfinite(early)]
    late_valid = late[np.isfinite(late)]

    if (
        len(early_valid) > 0
        and len(late_valid) > 0
    ):
        early_mean = np.mean(early_valid)
        late_mean = np.mean(late_valid)

        if abs(early_mean) > 1e-12:
            early_late_change = (
                late_mean - early_mean
            ) / abs(early_mean)
        else:
            early_late_change = np.nan
    else:
        early_late_change = np.nan

    return {
        "mean": mean_value,
        "sd": sd_value,
        "cv": cv_value,
        "slope": slope,
        "early_late_change": early_late_change,
        "monotonicity": monotonicity,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 8 — REPETITION DESCRIPTOR ENGINEERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 07_repetition_aware.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"subject": str}
    )

    print(f"\nInput shape: {df.shape}")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    required = [
        "subject",
        "side",
        "group",
        "rep",
    ] + TRAJECTORY_FEATURES

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing)
        )

    df = df.sort_values(
        ["subject", "rep"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Build subject-level descriptors
    # --------------------------------------------------------

    rows = []

    for subject, sub in df.groupby(
        "subject",
        sort=True
    ):

        sub = sub.sort_values("rep")

        row = {
            "subject": subject,
            "side": sub["side"].iloc[0],
            "group": sub["group"].iloc[0],
        }

        for feature in TRAJECTORY_FEATURES:

            values = pd.to_numeric(
                sub[feature],
                errors="coerce"
            ).to_numpy()

            descriptors = calculate_descriptors(
                values
            )

            for name, value in descriptors.items():

                row[
                    f"{feature}_{name}"
                ] = value

        rows.append(row)

    output = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    descriptor_count = (
        len(output.columns) - 3
    )

    expected_descriptors = (
        len(TRAJECTORY_FEATURES) * 6
    )

    print("\n" + "=" * 70)
    print("REPETITION DESCRIPTOR ENGINEERING COMPLETE")
    print("=" * 70)

    print(f"\nOutput shape: {output.shape}")

    print(
        f"Subjects: {output['subject'].nunique()}"
    )

    print(
        f"Descriptor columns: {descriptor_count}"
    )

    print(
        f"Expected descriptor columns: "
        f"{expected_descriptors}"
    )

    print(
        f"\nGroup counts:\n"
        f"{output.groupby('group')['subject'].nunique()}"
    )

    print(
        "\nMissing-value summary:"
    )

    print(
        output.isna().sum()
        .sort_values(ascending=False)
        .head(15)
    )

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    print("\nFirst 5 rows:")
    print(
        output.head().to_string(index=False)
    )


if __name__ == "__main__":
    main()