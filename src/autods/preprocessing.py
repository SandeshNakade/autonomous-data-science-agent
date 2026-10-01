"""Allowlisted preprocessing and data splitting tools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schemas import MLPlan


class PreparationError(ValueError):
    """A dataset cannot safely be prepared for one of the fixed recipes."""


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: tuple[str, ...]


@dataclass
class PreparedData:
    """Validation data for the feedback loop plus a holdout used exactly once."""

    validation_split: DataSplit
    holdout_split: DataSplit


def split_dataset(data: pd.DataFrame, plan: MLPlan) -> PreparedData:
    """Create fit/validation/holdout partitions before any transformer is fitted."""

    if plan.target_column not in data.columns:
        raise PreparationError(f"Target '{plan.target_column}' is missing from the CSV")
    filtered = data.dropna(subset=[plan.target_column]).copy()
    if plan.task_type.is_classification:
        y = filtered[plan.target_column].astype(str)
        counts = y.value_counts()
        if len(counts) < 2:
            raise PreparationError("Classification requires at least two observed classes")
        if counts.min() < 4:
            raise PreparationError(
                "Each classification class needs at least four rows for validation and holdout splits"
            )
        stratify: pd.Series | None = y
    else:
        y = pd.to_numeric(filtered[plan.target_column], errors="coerce")
        valid_target = y.notna()
        filtered = filtered.loc[valid_target].copy()
        y = y.loc[valid_target]
        if len(y) < 10 or y.nunique() < 2:
            raise PreparationError("Regression target needs at least ten valid, non-constant values")
        stratify = None

    excluded = {plan.target_column, *plan.preprocessing.excluded_columns}
    feature_columns = tuple(column for column in filtered.columns if column not in excluded)
    if not feature_columns:
        raise PreparationError("No usable features remain after excluding target/identifier columns")
    X = filtered.loc[:, feature_columns].copy()
    if len(X) < 20:
        raise PreparationError("At least twenty rows are required for validation and holdout splits")
    try:
        X_train_validation, X_holdout, y_train_validation, y_holdout = train_test_split(
            X,
            y,
            test_size=plan.test_size,
            random_state=plan.random_state,
            stratify=stratify,
        )
        validation_size = min(0.30, max(0.10, plan.test_size / (1 - plan.test_size)))
        validation_stratify = y_train_validation if plan.task_type.is_classification else None
        X_train, X_validation, y_train, y_validation = train_test_split(
            X_train_validation,
            y_train_validation,
            test_size=validation_size,
            random_state=plan.random_state + 1,
            stratify=validation_stratify,
        )
    except ValueError as exc:
        raise PreparationError(f"Could not create validation/holdout splits: {exc}") from exc
    return PreparedData(
        validation_split=DataSplit(
            X_train=X_train,
            X_test=X_validation,
            y_train=y_train,
            y_test=y_validation,
            feature_columns=feature_columns,
        ),
        holdout_split=DataSplit(
            X_train=pd.concat([X_train, X_validation]),
            X_test=X_holdout,
            y_train=pd.concat([y_train, y_validation]),
            y_test=y_holdout,
            feature_columns=feature_columns,
        ),
    )


def build_preprocessor(X_train: pd.DataFrame, plan: MLPlan) -> ColumnTransformer:
    """Return a fixed, inspectable transformation recipe for tabular features."""

    numeric_columns = list(X_train.select_dtypes(include=[np.number, "bool"]).columns)
    categorical_columns = [column for column in X_train.columns if column not in numeric_columns]
    if not numeric_columns and not categorical_columns:
        raise PreparationError("No compatible numeric or categorical feature columns found")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, object]] = [
            ("impute", SimpleImputer(strategy=plan.preprocessing.numeric_imputation))
        ]
        if plan.preprocessing.scale_numeric:
            numeric_steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy=plan.preprocessing.categorical_imputation,
                        fill_value="__missing__",
                    ),
                ),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    return ColumnTransformer(transformers=transformers, remainder="drop")
