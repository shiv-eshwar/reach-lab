from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "repetition_descriptors.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "experiment_dataset.csv"
)


def main():

    print("=" * 70)
    print("STEP 9 — EXPERIMENT DATASET CONSTRUCTION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run 08_repetition_descriptors.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        dtype={"subject": str}
    )

    print(f"\nInput shape: {df.shape}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if df["subject"].duplicated().any():
        raise ValueError(
            "Duplicate subject records detected."
        )

    if df["group"].isna().any():
        raise ValueError(
            "Missing group labels detected."
        )

    # --------------------------------------------------------
    # Encode target
    # DCP = 1
    # TD  = 0
    # --------------------------------------------------------

    valid_groups = {"DCP", "TD"}

    if not set(df["group"].unique()).issubset(
        valid_groups
    ):
        raise ValueError(
            f"Unexpected group labels: "
            f"{df['group'].unique()}"
        )

    df["target"] = (
        df["group"] == "DCP"
    ).astype(int)

    # --------------------------------------------------------
    # Remove metadata that must not be used as predictors
    # --------------------------------------------------------

    metadata_columns = [
        "subject",
        "side",
        "group",
        "target",
    ]

    feature_columns = [
        col
        for col in df.columns
        if col not in metadata_columns
    ]

    # --------------------------------------------------------
    # Remove completely missing features
    #
    # IMPORTANT:
    # This is only structural filtering.
    # No imputation or scaling is performed here.
    # --------------------------------------------------------

    complete_missing = [
        col
        for col in feature_columns
        if df[col].isna().all()
    ]

    if complete_missing:
        print(
            "\nRemoving completely missing features:"
        )

        for col in complete_missing:
            print(f"  - {col}")

        feature_columns = [
            col
            for col in feature_columns
            if col not in complete_missing
        ]

    # --------------------------------------------------------
    # Construct experiment dataset
    # --------------------------------------------------------

    experiment_df = df[
        [
            "subject",
            "side",
            "group",
            "target",
        ] + feature_columns
    ].copy()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    experiment_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT DATASET COMPLETE")
    print("=" * 70)

    print(
        f"\nOutput shape: "
        f"{experiment_df.shape}"
    )

    print(
        f"Subjects: "
        f"{experiment_df['subject'].nunique()}"
    )

    print(
        f"Predictor features: "
        f"{len(feature_columns)}"
    )

    print("\nTarget distribution:")

    print(
        experiment_df["target"]
        .value_counts()
        .sort_index()
    )

    print("\nGroup distribution:")

    print(
        experiment_df["group"]
        .value_counts()
    )

    print(
        "\nCompletely missing predictors remaining:",
        experiment_df[
            feature_columns
        ].isna().all().sum()
    )

    print(
        "\nPredictors containing some missing values:",
        experiment_df[
            feature_columns
        ].isna().any().sum()
    )

    print(
        f"\nOutput file:\n{OUTPUT_FILE}"
    )

    print("\nFirst 5 metadata rows:")
    print(
        experiment_df[
            ["subject", "side", "group", "target"]
        ].head().to_string(index=False)
    )


if __name__ == "__main__":
    main()