"""
03_features.py

Objective 2 & Objective 3:
Extract spatial, temporal, angular and movement-quality
kinematic features from each subject × repetition.

Output:
One row per subject × repetition.
"""

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
    / "clean_frames.pkl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rep_features.csv"
)


# ============================================================
# DATASET PARAMETERS
# ============================================================

FPS = 25.0
DT = 1.0 / FPS


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def safe_array(values):
    """Convert values to a numeric NumPy array."""

    return pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    ).to_numpy(dtype=float)


def valid_values(values):
    """Return finite values only."""

    values = np.asarray(
        values,
        dtype=float,
    )

    return values[
        np.isfinite(values)
    ]


def normalized_jerk(position, dt):
    """
    Dimensionless normalized jerk.

    Lower values indicate smoother movement.
    """

    position = np.asarray(
        position,
        dtype=float,
    )

    valid = np.isfinite(position).all(axis=1)

    position = position[valid]

    if len(position) < 5:
        return np.nan

    velocity = np.gradient(
        position,
        dt,
        axis=0,
    )

    acceleration = np.gradient(
        velocity,
        dt,
        axis=0,
    )

    jerk = np.gradient(
        acceleration,
        dt,
        axis=0,
    )

    speed = np.linalg.norm(
        velocity,
        axis=1,
    )

    jerk_squared = np.sum(
        jerk ** 2,
        axis=1,
    )

    duration = dt * (
        len(position) - 1
    )

    distance = (
        np.sum(speed) * dt
    )

    if (
        duration <= 0
        or distance <= 0
    ):
        return np.nan

    integral = np.trapezoid(
        jerk_squared,
        dx=dt,
    )

    value = (
        0.5
        * integral
        * duration ** 5
        / distance ** 2
    )

    return np.sqrt(
        max(value, 0)
    )


def spectral_arc_length(speed, dt):
    """
    SPARC-like spectral arc-length smoothness measure.

    More negative values indicate greater spectral complexity.
    """

    speed = valid_values(speed)

    if len(speed) < 5:
        return np.nan

    maximum = np.max(
        np.abs(speed)
    )

    if maximum <= 0:
        return np.nan

    speed = speed / maximum

    frequency = np.fft.rfftfreq(
        len(speed),
        d=dt,
    )

    magnitude = np.abs(
        np.fft.rfft(speed)
    )

    magnitude = (
        magnitude
        / (magnitude.max() + 1e-12)
    )

    # Movement frequencies up to 10 Hz
    mask = frequency <= 10

    frequency = frequency[mask]
    magnitude = magnitude[mask]

    if len(frequency) < 3:
        return np.nan

    return -np.sum(
        np.sqrt(
            np.diff(frequency) ** 2
            + np.diff(magnitude) ** 2
        )
    )


def range_of_motion(angle):
    """Calculate angular range of motion."""

    values = valid_values(angle)

    if len(values) == 0:
        return np.nan

    return (
        np.max(values)
        - np.min(values)
    )


def angular_velocity_stats(angle, dt):
    """Calculate mean and peak absolute angular velocity."""

    values = valid_values(angle)

    if len(values) < 2:
        return np.nan, np.nan

    velocity = (
        np.diff(values) / dt
    )

    return (
        np.mean(np.abs(velocity)),
        np.max(np.abs(velocity)),
    )


def angular_acceleration_stats(angle, dt):
    """Calculate mean and peak absolute angular acceleration."""

    values = valid_values(angle)

    if len(values) < 3:
        return np.nan, np.nan

    velocity = np.diff(values) / dt

    if len(velocity) < 2:
        return np.nan, np.nan

    acceleration = np.diff(velocity) / dt

    return (
        np.mean(np.abs(acceleration)),
        np.max(np.abs(acceleration)),
    )


def correlation(x, y):
    """Safe Pearson correlation."""

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if valid.sum() < 4:
        return np.nan

    x = x[valid]
    y = y[valid]

    if (
        np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan

    return np.corrcoef(
        x,
        y,
    )[0, 1]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_rep_features(sub):
    """
    Extract features from one subject × repetition.
    """

    sub = (
        sub
        .sort_values("frame")
        .reset_index(drop=True)
    )

    n = len(sub)

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    wrist = sub[
        [
            "wrist_x_mm",
            "wrist_y_mm",
        ]
    ].to_numpy(
        dtype=float
    )

    elbow = sub[
        [
            "elbow_x_mm",
            "elbow_y_mm",
        ]
    ].to_numpy(
        dtype=float
    )

    shoulder = sub[
        [
            "shoulder_x_mm",
            "shoulder_y_mm",
        ]
    ].to_numpy(
        dtype=float
    )

    mcp3 = sub[
        [
            "mcp3_x_mm",
            "mcp3_y_mm",
        ]
    ].to_numpy(
        dtype=float
    )

    target = sub[
        [
            "object_x_mm",
            "object_y_mm",
        ]
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Joint angles
    # --------------------------------------------------------

    shoulder_angle = safe_array(
        sub["shoulder_angle"]
    )

    elbow_angle = safe_array(
        sub["elbow_angle"]
    )

    wrist_angle = safe_array(
        sub["wrist_angle"]
    )

    # ========================================================
    # SPATIAL FEATURES
    # ========================================================

    valid_wrist = np.isfinite(
        wrist
    ).all(axis=1)

    valid_wrist_points = wrist[
        valid_wrist
    ]

    if len(valid_wrist_points) >= 2:

        wrist_diffs = np.diff(
            valid_wrist_points,
            axis=0,
        )

        step_lengths = np.linalg.norm(
            wrist_diffs,
            axis=1,
        )

        trajectory_length = np.sum(
            step_lengths
        )

        reach_distance = np.linalg.norm(
            valid_wrist_points[-1]
            - valid_wrist_points[0]
        )

    else:

        step_lengths = np.array(
            [],
            dtype=float,
        )

        trajectory_length = np.nan
        reach_distance = np.nan

    if (
        np.isfinite(trajectory_length)
        and trajectory_length > 0
    ):

        path_efficiency = (
            reach_distance
            / trajectory_length
        )

    else:

        path_efficiency = np.nan

    # --------------------------------------------------------
    # Position error
    # --------------------------------------------------------

    paired = np.isfinite(wrist).all(axis=1) & np.isfinite(target).all(axis=1)
    if paired.any():
        position_error = np.linalg.norm((wrist[paired] - target[paired])[-1])
    else:
        position_error = np.nan

    # ========================================================
    # TRAJECTORY DEVIATION
    # ========================================================

    if (
        len(valid_wrist_points) >= 2
        and np.isfinite(
            reach_distance
        )
        and reach_distance > 0
    ):

        start = (
            valid_wrist_points[0]
        )

        end = (
            valid_wrist_points[-1]
        )

        direction = (
            end - start
        )

        direction = (
            direction
            / (
                np.linalg.norm(direction)
                + 1e-12
            )
        )

        relative = (
            valid_wrist_points
            - start
        )

        projection = (
            relative @ direction
        )

        projected = np.outer(
            projection,
            direction,
        )

        lateral = (
            relative
            - projected
        )

        lateral_deviation = np.linalg.norm(
            lateral,
            axis=1,
        )

        mean_lateral_deviation = np.mean(
            lateral_deviation
        )

        max_lateral_deviation = np.max(
            lateral_deviation
        )

    else:

        mean_lateral_deviation = np.nan
        max_lateral_deviation = np.nan

    # ========================================================
    # TEMPORAL FEATURES
    # ========================================================

    movement_duration = (
        DT * (n - 1)
    )

    if len(step_lengths) > 0:

        speed = (
            step_lengths / DT
        )

        peak_velocity = np.max(
            speed
        )

        mean_velocity = np.mean(
            speed
        )

    else:

        speed = np.array(
            [],
            dtype=float,
        )

        peak_velocity = np.nan
        mean_velocity = np.nan

    if len(speed) > 1:

        acceleration = (
            np.diff(speed)
            / DT
        )

        mean_acceleration = np.mean(
            np.abs(acceleration)
        )

        peak_acceleration = np.max(
            np.abs(acceleration)
        )

    else:

        mean_acceleration = np.nan
        peak_acceleration = np.nan

    # ========================================================
    # ANGULAR FEATURES
    # ========================================================

    shoulder_rom = range_of_motion(
        shoulder_angle
    )

    elbow_rom = range_of_motion(
        elbow_angle
    )

    wrist_rom = range_of_motion(
        wrist_angle
    )

    (
        shoulder_angvel_mean,
        shoulder_angvel_peak,
    ) = angular_velocity_stats(
        shoulder_angle,
        DT,
    )

    (
        elbow_angvel_mean,
        elbow_angvel_peak,
    ) = angular_velocity_stats(
        elbow_angle,
        DT,
    )

    (
        wrist_angvel_mean,
        wrist_angvel_peak,
    ) = angular_velocity_stats(
        wrist_angle,
        DT,
    )

    (
        shoulder_angacc_mean,
        shoulder_angacc_peak,
    ) = angular_acceleration_stats(
        shoulder_angle,
        DT,
    )

    (
        elbow_angacc_mean,
        elbow_angacc_peak,
    ) = angular_acceleration_stats(
        elbow_angle,
        DT,
    )

    (
        wrist_angacc_mean,
        wrist_angacc_peak,
    ) = angular_acceleration_stats(
        wrist_angle,
        DT,
    )

    # ========================================================
    # MOVEMENT QUALITY
    # ========================================================

    normalized_jerk_value = normalized_jerk(
        wrist,
        DT,
    )

    sparc_value = spectral_arc_length(
        speed,
        DT,
    )

    # --------------------------------------------------------
    # Joint coordination
    # --------------------------------------------------------

    elbow_wrist_coordination = correlation(
        elbow_angle,
        wrist_angle,
    )

    shoulder_elbow_coordination = correlation(
        shoulder_angle,
        elbow_angle,
    )

    # --------------------------------------------------------
    # Speed variability
    # --------------------------------------------------------

    if (
        len(speed) > 1
        and np.isfinite(mean_velocity)
        and mean_velocity > 0
    ):

        speed_cv = (
            np.std(speed)
            / mean_velocity
        )

    else:

        speed_cv = np.nan

    # --------------------------------------------------------
    # Angular variability
    # --------------------------------------------------------

    shoulder_angle_sd = (
        np.nanstd(
            shoulder_angle
        )
        if np.isfinite(
            shoulder_angle
        ).any()
        else np.nan
    )

    elbow_angle_sd = (
        np.nanstd(
            elbow_angle
        )
        if np.isfinite(
            elbow_angle
        ).any()
        else np.nan
    )

    wrist_angle_sd = (
        np.nanstd(
            wrist_angle
        )
        if np.isfinite(
            wrist_angle
        ).any()
        else np.nan
    )

    # ========================================================
    # RETURN
    # ========================================================

    return pd.Series({

        # Spatial
        "trajectory_length_mm":
            trajectory_length,

        "path_efficiency":
            path_efficiency,

        "reach_distance_mm":
            reach_distance,

        "position_error_mm":
            position_error,

        "mean_lateral_deviation_mm":
            mean_lateral_deviation,

        "max_lateral_deviation_mm":
            max_lateral_deviation,

        # Temporal
        "movement_duration_s":
            movement_duration,

        "peak_velocity_mm_s":
            peak_velocity,

        "mean_velocity_mm_s":
            mean_velocity,

        "mean_acceleration_mm_s2":
            mean_acceleration,

        "peak_acceleration_mm_s2":
            peak_acceleration,

        # Angular
        "shoulder_rom_deg":
            shoulder_rom,

        "elbow_rom_deg":
            elbow_rom,

        "wrist_rom_deg":
            wrist_rom,

        "shoulder_angvel_mean":
            shoulder_angvel_mean,

        "shoulder_angvel_peak":
            shoulder_angvel_peak,

        "elbow_angvel_mean":
            elbow_angvel_mean,

        "elbow_angvel_peak":
            elbow_angvel_peak,

        "wrist_angvel_mean":
            wrist_angvel_mean,

        "wrist_angvel_peak":
            wrist_angvel_peak,

        "shoulder_angacc_mean":
            shoulder_angacc_mean,

        "shoulder_angacc_peak":
            shoulder_angacc_peak,

        "elbow_angacc_mean":
            elbow_angacc_mean,

        "elbow_angacc_peak":
            elbow_angacc_peak,

        "wrist_angacc_mean":
            wrist_angacc_mean,

        "wrist_angacc_peak":
            wrist_angacc_peak,

        # Movement quality
        "smoothness_norm_jerk":
            normalized_jerk_value,

        "smoothness_sparc":
            sparc_value,

        "coordination_elbow_wrist":
            elbow_wrist_coordination,

        "coordination_shoulder_elbow":
            shoulder_elbow_coordination,

        "variability_speed_cv":
            speed_cv,

        "shoulder_angle_sd":
            shoulder_angle_sd,

        "elbow_angle_sd":
            elbow_angle_sd,

        "wrist_angle_sd":
            wrist_angle_sd,

        # Metadata
        "n_frames":
            n,
    })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 3 — KINEMATIC FEATURE EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Input check
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 02_preprocess.py first."
        )

    df = pd.read_pickle(
        INPUT_FILE
    )

    print(
        f"\nInput rows: {len(df):,}"
    )

    print(
        f"Subjects: {df['subject'].nunique()}"
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    rows = []

    groups = df.groupby(
        [
            "subject",
            "side",
            "group",
            "rep",
        ],
        sort=False,
    )

    total = len(groups)

    print(
        f"Repetitions to process: {total}"
    )

    for number, (keys, sub) in enumerate(
        groups,
        start=1,
    ):

        features = extract_rep_features(
            sub
        )

        subject, side, group, rep = keys

        row = {
            "subject": subject,
            "side": side,
            "group": group,
            "rep": rep,
        }

        row.update(
            features.to_dict()
        )

        rows.append(row)

        if (
            number % 50 == 0
            or number == total
        ):

            print(
                f"Processed "
                f"{number}/{total}"
            )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values(
            [
                "subject",
                "rep",
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        f"\nOutput shape: {result.shape}"
    )

    print(
        f"Subjects: {result['subject'].nunique()}"
    )

    print(
        f"Repetition records: {len(result)}"
    )

    print(
        f"Feature count: "
        f"{len(result.columns) - 4}"
    )

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    print("\nNaN rates:")

    nan_rates = (
        result
        .isna()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        nan_rates[
            nan_rates > 0
        ].head(15)
    )

    print("\nFirst 5 records:")

    print(
        result.head()
    )


if __name__ == "__main__":
    main()