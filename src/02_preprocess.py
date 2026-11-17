"""
02_preprocess.py

Objective 1:
- Missing value handling
- Noise filtering
- Temporal alignment
- Coordinate normalization
- Time normalization
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "raw_frames.pkl"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "clean_frames.pkl"


# ============================================================
# DATASET PARAMETERS
# ============================================================

PX_TO_MM = 2.0
FPS = 25.0


COORD_COLS = [
    "wrist_x",
    "wrist_y",
    "elbow_x",
    "elbow_y",
    "shoulder_x",
    "shoulder_y",
    "mcp3_x",
    "mcp3_y",
    "object_x",
    "object_y",
]

ANGLE_COLS = [
    "shoulder_angle",
    "elbow_angle",
    "wrist_angle",
]


# ============================================================
# SAVITZKY-GOLAY FILTER
# ============================================================

def smooth_series(series):
    """
    Apply Savitzky-Golay smoothing while safely handling
    short trajectories.
    """

    values = series.to_numpy(dtype=float)
    n = len(values)

    if n < 5:
        return series

    # Largest valid odd window <= 7
    window = min(7, n)

    if window % 2 == 0:
        window -= 1

    if window < 5:
        return series

    try:

        smoothed = savgol_filter(
            values,
            window_length=window,
            polyorder=2,
        )

        return pd.Series(
            smoothed,
            index=series.index,
        )

    except Exception:

        return series


# ============================================================
# CLEAN ONE REPETITION
# ============================================================

def clean_trajectory(group):
    """
    Preprocess one subject/repetition trajectory.
    """

    group = (
        group
        .sort_values("frame")
        .reset_index(drop=True)
        .copy()
    )

    # --------------------------------------------------------
    # Missing values + filtering
    # --------------------------------------------------------

    for column in COORD_COLS + ANGLE_COLS:

        series = group[column]

        # Completely missing feature
        if series.isna().all():
            continue

        # Linear interpolation
        series = (
            series
            .interpolate(
                method="linear",
                limit_direction="both",
            )
        )

        # Savitzky-Golay smoothing
        series = smooth_series(series)

        group[column] = series

    # --------------------------------------------------------
    # Pixel → millimetre conversion
    # --------------------------------------------------------

    for column in COORD_COLS:

        group[f"{column}_mm"] = (
            group[column] * PX_TO_MM
        )

    # --------------------------------------------------------
    # Time axis
    # --------------------------------------------------------

    start_frame = group["frame"].iloc[0]

    group["t_sec"] = (
        group["frame"] - start_frame
    ) / FPS

    # --------------------------------------------------------
    # Normalized time
    # 0 → start
    # 1 → PTA/end
    # --------------------------------------------------------

    if len(group) > 1:

        group["t_norm"] = (
            group["t_sec"] /
            group["t_sec"].iloc[-1]
        )

    else:

        group["t_norm"] = 0.0

    return group


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("STEP 2 — PREPROCESSING")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 01_load_raw.py first."
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_pickle(INPUT_FILE)

    print(
        f"\nInput rows: {len(df):,}"
    )

    print(
        f"Subjects: {df['subject'].nunique()}"
    )

    print(
        f"Repetitions: {len(df.groupby(['subject', 'rep']))}"
    )

    # --------------------------------------------------------
    # Missing values BEFORE preprocessing
    # --------------------------------------------------------

    print("\nMissing values BEFORE preprocessing:")

    before_missing = (
        df[COORD_COLS + ANGLE_COLS]
        .isna()
        .sum()
    )

    print(
        before_missing[
            before_missing > 0
        ]
    )

    # --------------------------------------------------------
    # Process each subject / repetition
    # --------------------------------------------------------

    processed = []

    groups = df.groupby(
        [
            "subject",
            "side",
            "group",
            "rep",
        ],
        sort=False,
    )

    total_groups = len(groups)

    print(
        f"\nProcessing {total_groups} "
        "repetitions..."
    )

    for number, (_, trajectory) in enumerate(
        groups,
        start=1,
    ):

        processed.append(
            clean_trajectory(
                trajectory
            )
        )

        if (
            number % 50 == 0
            or number == total_groups
        ):

            print(
                f"Processed "
                f"{number}/{total_groups}"
            )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    cleaned = pd.concat(
        processed,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Missing values AFTER preprocessing
    # --------------------------------------------------------

    print("\nMissing values AFTER preprocessing:")

    after_missing = (
        cleaned[
            COORD_COLS + ANGLE_COLS
        ]
        .isna()
        .sum()
    )

    remaining = after_missing[
        after_missing > 0
    ]

    if len(remaining) == 0:

        print("No missing values.")

    else:

        print(remaining)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_pickle(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"\nInput shape : {df.shape}"
    )

    print(
        f"Output shape: {cleaned.shape}"
    )

    print(
        f"Subjects    : "
        f"{cleaned['subject'].nunique()}"
    )

    print(
        f"Groups      : "
        f"{cleaned.groupby('group')['subject'].nunique().to_dict()}"
    )

    print(
        f"Output file : {OUTPUT_FILE}"
    )

    print("\nSample:")
    print(
        cleaned[
            [
                "subject",
                "rep",
                "frame",
                "t_sec",
                "t_norm",
                "wrist_x_mm",
                "wrist_y_mm",
                "elbow_angle",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()