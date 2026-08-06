"""ML Sentinel detectors — the fleet of silent-problem finders.

Each detector is a small, focused class that takes a :class:`DataHubClient`
and a target entity, and returns zero or more :class:`Finding` objects.
Detectors are intentionally independent so they can run in parallel and so
new detectors can be added without touching existing ones.

Currently shipped detectors:

* :class:`ModelDriftDetector`       — prediction / feature distribution drift
* :class:`BrokenLineageDetector`    — missing or stale upstream edges
* :class:`StaleFeatureDetector`     — feature tables past their freshness SLA
* :class:`SchemaMismatchDetector`   — serving schema diverged from training schema
* :class:`TrainingServingSkewDetector` — feature distribution skew between train/serve
"""

from __future__ import annotations

from .broken_lineage import BrokenLineageDetector
from .model_drift import ModelDriftDetector
from .schema_mismatch import SchemaMismatchDetector
from .stale_feature import StaleFeatureDetector
from .training_serving_skew import TrainingServingSkewDetector

__all__ = [
    "BrokenLineageDetector",
    "ModelDriftDetector",
    "SchemaMismatchDetector",
    "StaleFeatureDetector",
    "TrainingServingSkewDetector",
]
