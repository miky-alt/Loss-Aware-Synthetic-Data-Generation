"""
Data loaders for all datasets used in the project.

Each loader returns a unified DatasetBundle with:
- real: pd.DataFrame (cleaned, with categorical values preserved for SDV)
- target_col: str (column name to use for downstream classification)
- name: str (human-readable dataset name)
- domain: str (e.g. "medical", "socioeconomic")
"""

from dataclasses import dataclass

import pandas as pd
from sdv.metadata import Metadata
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


@dataclass
class DatasetBundle:
    real: pd.DataFrame
    target_col: str
    name: str
    domain: str


def build_sdv_metadata(data: pd.DataFrame) -> Metadata:
    """Detect SDV metadata while preserving pandas boolean semantics."""
    metadata = Metadata.detect_from_dataframe(data)
    boolean_columns = data.select_dtypes(include="bool").columns
    metadata.update_columns_metadata(
        {column: {"sdtype": "boolean"} for column in boolean_columns}
    )
    return metadata


def load_uci_adult() -> DatasetBundle:
    """
    UCI Adult (Census Income) dataset.
    ~48k rows, 14 features.
    Target: income >50K (binary).
    Domain: socioeconomic.
    """
    dataset = fetch_ucirepo(id=2)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = pd.concat([X, y], axis=1)
    df.columns = [c.strip() for c in df.columns]

    # Normalize target to a boolean so SDV detects its semantic type correctly.
    target_col = "income"
    df[target_col] = df[target_col].astype(str).str.strip().str.replace(".", "", regex=False)
    df[target_col] = df[target_col].str.contains(">50K")
    df["sex"] = df["sex"].astype(str).str.strip().str.lower().eq("male")

    df = df.dropna().reset_index(drop=True)

    return DatasetBundle(
        real=df,
        target_col=target_col,
        name="UCI Adult (Census Income)",
        domain="socioeconomic",
    )


def load_diabetes_130() -> DatasetBundle:
    """
    Diabetes 130-US Hospitals dataset.
    ~100k rows, 50 features.
    Target: readmitted within 30 days (binary).
    Domain: medical.
    """
    dataset = fetch_ucirepo(id=296)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = pd.concat([X, y], axis=1)
    df.columns = [c.strip() for c in df.columns]

    # Normalize target to a boolean: early readmission or not.
    target_col = "readmitted"
    df[target_col] = df[target_col].astype(str).str.strip() == "<30"

    # Normalize other binary columns to booleans so SDV detects their semantic type correctly.
    df["gender"] = df["gender"].astype(str).str.strip().str.lower().eq("male")
    df["change"] = df["change"].astype(str).str.strip().eq("Ch")
    df["diabetesMed"] = df["diabetesMed"].astype(str).str.strip().eq("Yes")

    # Drop columns with too many missing values
    df = df.replace("?", pd.NA)
    missing_threshold = 0.4
    df = df.dropna(thresh=int(len(df) * (1 - missing_threshold)), axis=1)
    df = df.dropna().reset_index(drop=True)

    # ID columns are codes for categories, not continuous measurements.
    for column in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        if column in df:
            df[column] = df[column].astype(str).astype("category")

    return DatasetBundle(
        real=df,
        target_col=target_col,
        name="Diabetes 130-US Hospitals",
        domain="medical",
    )


def load_heart_disease() -> DatasetBundle:
    """
    Heart Disease (Cleveland) dataset.
    ~300 rows, 13 features.
    Target: presence of heart disease (binary).
    Domain: medical.
    """
    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    df = pd.concat([X, y], axis=1)
    df.columns = [c.strip() for c in df.columns]

    # Normalize target to a boolean: disease absent or present.
    target_col = "num"
    df[target_col] = df[target_col] > 0

    df = df.dropna().reset_index(drop=True)

    # Preserve the semantic types of encoded categorical features for SDV.
    for column in ("sex", "fbs", "exang"):
        df[column] = df[column].astype(bool)
    for column in ("cp", "restecg", "slope", "ca", "thal"):
        df[column] = df[column].astype(str).astype("category")

    return DatasetBundle(
        real=df,
        target_col=target_col,
        name="Heart Disease (Cleveland)",
        domain="medical",
    )


LOADERS = {
    "adult": load_uci_adult,
    "diabetes": load_diabetes_130,
    "heart": load_heart_disease,
}


def load_dataset(name: str) -> DatasetBundle:
    """
    Load a dataset by short name.
    Available: 'adult', 'diabetes', 'heart'
    """
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(LOADERS.keys())}")
    return LOADERS[name]()


def split_dataset(
    bundle: DatasetBundle,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[DatasetBundle, DatasetBundle]:
    """
    Split a DatasetBundle into train/test, stratified on target_col.

    The train split is what the generator should fit() on; the test split is
    held out and never seen during training, so evaluation measures
    generalization rather than memorization of the training data.
    """
    train_df, test_df = train_test_split(
        bundle.real,
        test_size=test_size,
        random_state=seed,
        stratify=bundle.real[bundle.target_col],
    )
    train_bundle = DatasetBundle(
        real=train_df.reset_index(drop=True),
        target_col=bundle.target_col,
        name=bundle.name,
        domain=bundle.domain,
    )
    test_bundle = DatasetBundle(
        real=test_df.reset_index(drop=True),
        target_col=bundle.target_col,
        name=bundle.name,
        domain=bundle.domain,
    )
    return train_bundle, test_bundle
