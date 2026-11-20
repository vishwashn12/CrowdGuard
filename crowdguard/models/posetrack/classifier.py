"""Loader and batch predictor for the trained anomaly classifier.

The pickle produced by the training pipeline is a dict::

    {
        "classifier":   fitted estimator (CatBoost by default),
        "scaler":       fitted StandardScaler, or None,
        "feature_cols": ordered feature names,
        "label_names":  ["normal", "sitting", "opposite"],
        "model_name":   e.g. "CatBoost",
    }

The feature order in the pickle is verified against
:data:`crowdguard.models.posetrack.features.FEATURE_NAMES` at load time. A
mismatch is fatal: feeding columns in the wrong order produces confident
nonsense rather than an error, so it must be caught at startup.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from crowdguard.models.contracts import AnomalyLabel
from crowdguard.models.posetrack.features import FEATURE_COUNT, FEATURE_NAMES
from crowdguard.utils.exceptions import CheckpointError

logger = logging.getLogger(__name__)


class AnomalyClassifier:
    """Wraps the pickled estimator and its scaler."""

    def __init__(
        self,
        classifier: Any,
        scaler: Any | None,
        feature_cols: list[str],
        label_names: list[str],
        model_name: str,
        source_path: Path | None = None,
    ) -> None:
        self._classifier = classifier
        self._scaler = scaler
        self.feature_cols = list(feature_cols)
        self.label_names = list(label_names)
        self.model_name = model_name
        self.source_path = source_path

        self.prediction_count = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, strict_feature_order: bool = True) -> "AnomalyClassifier":
        """Load a classifier from a pickle produced by the training pipeline.

        Raises:
            CheckpointError: If the file is missing, malformed, or its feature
                order does not match this codebase.
        """
        model_path = Path(path)
        if not model_path.exists():
            raise CheckpointError(
                f"Anomaly classifier not found: {model_path}\n"
                f"Train one with the project's train_model.py, or copy an existing "
                f"best_model.pkl into assets/."
            )

        try:
            with model_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            raise CheckpointError(f"Could not unpickle classifier {model_path}: {exc}") from exc

        if not isinstance(payload, dict) or "classifier" not in payload:
            raise CheckpointError(
                f"{model_path} is not a recognised classifier bundle: expected a dict "
                f"with a 'classifier' key, got {type(payload).__name__}"
            )

        classifier = payload["classifier"]
        feature_cols = payload.get("feature_cols") or list(FEATURE_NAMES)
        label_names = payload.get("label_names") or ["normal", "sitting", "opposite"]
        model_name = payload.get("model_name", type(classifier).__name__)

        if strict_feature_order and tuple(feature_cols) != FEATURE_NAMES:
            raise CheckpointError(
                "Classifier feature order does not match this codebase.\n"
                f"  pickle expects : {list(feature_cols)}\n"
                f"  this code emits: {list(FEATURE_NAMES)}\n"
                "Predictions would be silently wrong. Retrain the model or align "
                "crowdguard/models/posetrack/features.py with the training script."
            )

        if len(feature_cols) != FEATURE_COUNT:
            raise CheckpointError(
                f"Classifier expects {len(feature_cols)} features but this codebase "
                f"produces {FEATURE_COUNT}."
            )

        # Tree ensembles default to multi-threaded predict, which fights the
        # worker threads for cores. One job per call is faster in aggregate here.
        if hasattr(classifier, "n_jobs"):
            try:
                classifier.n_jobs = 1
            except Exception:
                pass

        logger.info(
            "Loaded anomaly classifier %s from %s (scaler: %s)",
            model_name, model_path.name,
            "yes" if payload.get("scaler") is not None else "no",
        )

        return cls(
            classifier=classifier,
            scaler=payload.get("scaler"),
            feature_cols=feature_cols,
            label_names=label_names,
            model_name=model_name,
            source_path=model_path,
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray) -> list[AnomalyLabel]:
        """Classify a batch of feature vectors.

        Args:
            features: ``(n_samples, 10)`` array in :data:`FEATURE_NAMES` order.

        Returns:
            One :class:`AnomalyLabel` per row.

        Raises:
            ValueError: On a column-count mismatch.
        """
        if features.size == 0:
            return []

        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != FEATURE_COUNT:
            raise ValueError(
                f"Expected {FEATURE_COUNT} features per sample, got {matrix.shape[1]}"
            )

        # Guard against NaN/inf reaching the estimator: a missing keypoint or a
        # zero-height box upstream would otherwise abort the whole batch.
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if self._scaler is not None:
            matrix = self._scaler.transform(matrix)

        raw = self._classifier.predict(matrix)
        predictions = np.asarray(raw).reshape(-1).astype(int)

        self.prediction_count += len(predictions)
        return [self._to_label(value) for value in predictions]

    def predict_proba(self, features: np.ndarray) -> np.ndarray | None:
        """Class probabilities, when the estimator supports them."""
        if not hasattr(self._classifier, "predict_proba") or features.size == 0:
            return None

        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        if self._scaler is not None:
            matrix = self._scaler.transform(matrix)

        try:
            return np.asarray(self._classifier.predict_proba(matrix))
        except Exception:
            return None

    @staticmethod
    def _to_label(value: int) -> AnomalyLabel:
        try:
            return AnomalyLabel(int(value))
        except ValueError:
            return AnomalyLabel.NORMAL

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "source": str(self.source_path) if self.source_path else None,
            "has_scaler": self._scaler is not None,
            "feature_cols": self.feature_cols,
            "label_names": self.label_names,
            "predictions": self.prediction_count,
        }
