"""Deterministic model factory, training, and evaluation tools."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .preprocessing import DataSplit, build_preprocessor
from .schemas import ExperimentResult, MLPlan, TaskType

try:  # Available in newer scikit-learn releases; keep the stated >=1.2 support.
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - exercised only on older supported sklearn
    root_mean_squared_error = None


class TrainingError(RuntimeError):
    """An allowlisted estimator could not be trained or evaluated."""


@dataclass
class ModelBundle:
    """The only serialised serving artifact produced by a successful candidate."""

    pipeline: Any
    target_encoder: LabelEncoder | None
    feature_columns: tuple[str, ...]
    task_type: str
    metric: str

    def _aligned_frame(self, records: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
        return frame.reindex(columns=list(self.feature_columns))

    def predict(self, records: list[dict[str, Any]] | pd.DataFrame) -> list[Any]:
        predictions = self.pipeline.predict(self._aligned_frame(records))
        if self.target_encoder is not None:
            return self.target_encoder.inverse_transform(np.asarray(predictions, dtype=int)).tolist()
        return np.asarray(predictions).tolist()

    def predict_proba(self, records: list[dict[str, Any]] | pd.DataFrame) -> list[list[float]] | None:
        if not hasattr(self.pipeline, "predict_proba"):
            return None
        probabilities = self.pipeline.predict_proba(self._aligned_frame(records))
        return np.asarray(probabilities).tolist()


def _build_estimator(model_name: str, task: TaskType, random_state: int) -> Any:
    if model_name == "logistic_regression" and task.is_classification:
        return LogisticRegression(max_iter=1_000, random_state=random_state)
    if model_name == "linear_regression" and task == TaskType.REGRESSION:
        return LinearRegression()
    if model_name == "ridge" and task == TaskType.REGRESSION:
        return Ridge(alpha=1.0, random_state=random_state)
    if model_name == "random_forest":
        if task.is_classification:
            return RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            )
        return RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise TrainingError("XGBoost is optional; install the 'boosting' extra to use it") from exc
        common = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": random_state,
            "n_jobs": 1,
        }
        if task.is_classification:
            return XGBClassifier(eval_metric="logloss", **common)
        return XGBRegressor(objective="reg:squarederror", **common)
    raise TrainingError(f"Model '{model_name}' is not a valid recipe for {task.value}")


def _evaluate(
    pipeline: Pipeline,
    split: DataSplit,
    plan: MLPlan,
) -> tuple[float, dict[str, float]]:
    predictions = pipeline.predict(split.X_test)
    if plan.task_type.is_classification:
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(split.y_test, predictions)),
            "f1_macro": float(f1_score(split.y_test, predictions, average="macro", zero_division=0)),
        }
        if plan.task_type == TaskType.BINARY_CLASSIFICATION:
            metrics["f1"] = float(f1_score(split.y_test, predictions, zero_division=0))
        if hasattr(pipeline, "predict_proba"):
            probabilities = pipeline.predict_proba(split.X_test)
            try:
                if plan.task_type == TaskType.BINARY_CLASSIFICATION:
                    metrics["roc_auc"] = float(roc_auc_score(split.y_test, probabilities[:, 1]))
                else:
                    metrics["roc_auc_ovr"] = float(
                        roc_auc_score(split.y_test, probabilities, multi_class="ovr", average="macro")
                    )
            except ValueError:
                # A tiny test fold can contain one class. Other valid metrics remain.
                pass
    else:
        if root_mean_squared_error is not None:
            rmse = float(root_mean_squared_error(split.y_test, predictions))
        else:
            rmse = float(mean_squared_error(split.y_test, predictions, squared=False))
        metrics = {"r2": float(r2_score(split.y_test, predictions)), "rmse": rmse}
    if plan.metric not in metrics:
        raise TrainingError(f"Metric '{plan.metric}' was unavailable for this evaluation split")
    return metrics[plan.metric], metrics


def train_and_evaluate(
    split: DataSplit,
    plan: MLPlan,
    model_name: str,
    experiment_id: str,
    strategy: str,
) -> ExperimentResult:
    """Fit one fixed recipe and return structured measurements plus its artifact."""

    started = perf_counter()
    try:
        y_train: pd.Series | np.ndarray = split.y_train
        y_test: pd.Series | np.ndarray = split.y_test
        encoder: LabelEncoder | None = None
        if plan.task_type.is_classification:
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(split.y_train)
            y_test = encoder.transform(split.y_test)
            split = DataSplit(
                X_train=split.X_train,
                X_test=split.X_test,
                y_train=pd.Series(y_train, index=split.X_train.index),
                y_test=pd.Series(y_test, index=split.X_test.index),
                feature_columns=split.feature_columns,
            )
        estimator = _build_estimator(model_name, plan.task_type, plan.random_state)
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(split.X_train, plan)),
                ("model", estimator),
            ]
        )
        pipeline.fit(split.X_train, y_train)
        score, metrics = _evaluate(pipeline, split, plan)
        bundle = ModelBundle(
            pipeline=pipeline,
            target_encoder=encoder,
            feature_columns=split.feature_columns,
            task_type=plan.task_type.value,
            metric=plan.metric,
        )
        return ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            metric=plan.metric,
            score=score,
            metrics=metrics,
            status="completed",
            elapsed_seconds=round(perf_counter() - started, 4),
            strategy=strategy,
            model_bundle=bundle,
        )
    except Exception as exc:
        return ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            metric=plan.metric,
            score=None,
            metrics={},
            status="failed",
            elapsed_seconds=round(perf_counter() - started, 4),
            strategy=strategy,
            error=f"{type(exc).__name__}: {exc}",
        )
