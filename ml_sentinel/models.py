"""Core data models for ML Sentinel.

These dataclasses mirror the subset of DataHub's metadata model that ML Sentinel
cares about: ML entities (datasets, feature tables, models, deployments), the
lineage edges that connect them, and the findings the agent produces.

We keep them framework-agnostic (plain dataclasses + enums) so they work equally
well against the live DataHub MCP server and against the bundled mock dataset.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Entities — the nodes in DataHub's lineage graph that we monitor.
# --------------------------------------------------------------------------- #


class EntityType(str, enum.Enum):
    """The DataHub entity types ML Sentinel understands."""

    DATASET = "dataset"
    ML_FEATURE_TABLE = "mlFeatureTable"
    ML_FEATURE = "mlFeature"
    ML_MODEL = "mlModel"
    ML_MODEL_DEPLOYMENT = "mlModelDeployment"
    DATA_PROCESS = "dataProcess"  # Airflow / Spark job


@dataclass
class SchemaField:
    """A single column / field in a dataset's schema."""

    name: str
    type: str
    nullable: bool = True
    description: str = ""

    def matches(self, other: "SchemaField") -> bool:
        """Two fields 'match' if name and type agree (nullable is advisory)."""
        return self.name == other.name and self.type == other.type


@dataclass
class Entity:
    """A node in DataHub's lineage graph.

    ``urn`` is a DataHub URN, e.g. ``urn:li:dataset:(urn:li:dataPlatform:snowflake,...)``.
    """

    urn: str
    name: str
    type: EntityType
    platform: str = ""
    schema_fields: list[SchemaField] = field(default_factory=list)
    # ISO-8601 timestamps of the last metadata update / data refresh.
    last_modified: Optional[datetime] = None
    # Free-form properties bag — mirrors DataHub's structured properties.
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def short_urn(self) -> str:
        """A human-friendly short label derived from the URN."""
        # urn:li:dataset:(urn:li:dataPlatform:snowflake,db.schema.table,PROD) -> db.schema.table
        if ":" in self.urn and "(" in self.urn:
            try:
                inner = self.urn.split("(", 1)[1].rstrip(")")
                parts = inner.split(",")
                return parts[1] if len(parts) > 1 else self.name
            except Exception:
                pass
        return self.name


@dataclass
class LineageEdge:
    """A directed edge in DataHub's lineage graph: upstream -> downstream."""

    upstream_urn: str
    downstream_urn: str
    # When the lineage edge was last observed / confirmed in DataHub.
    created_at: Optional[datetime] = None
    # Optional: the SQL or operation that produced this edge.
    transformation: str = ""

    @property
    def is_stale(self) -> bool:
        """An edge is 'stale' if it hasn't been confirmed in > 30 days."""
        if self.created_at is None:
            return True
        age = (datetime.now(timezone.utc) - self.created_at).days
        return age > 30


# --------------------------------------------------------------------------- #
# Distributions — used by the drift / skew detectors.
# --------------------------------------------------------------------------- #


@dataclass
class FeatureDistribution:
    """A statistical snapshot of a feature's values.

    Only the fields needed by the detectors are populated; the rest are ``None``.
    """

    feature_name: str
    mean: Optional[float] = None
    std: Optional[float] = None
    null_rate: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p95: Optional[float] = None
    n_samples: Optional[int] = None
    # For categorical features: {category: proportion}
    categories: Optional[dict[str, float]] = None

    def psi_against(self, other: "FeatureDistribution") -> Optional[float]:
        """Population Stability Index (PSI) between two distributions.

        PSI < 0.1 = stable, 0.1-0.25 = minor shift, > 0.25 = significant drift.
        Returns ``None`` if there isn't enough data to compute it.
        """
        if self.categories and other.categories:
            return _psi_categorical(self.categories, other.categories)
        if (
            self.mean is not None
            and other.mean is not None
            and self.std is not None
            and other.std is not None
        ):
            return _psi_numeric(self, other)
        return None


@dataclass
class ModelMetrics:
    """Operational metrics for a deployed model."""

    model_urn: str
    accuracy: Optional[float] = None
    prediction_drift_score: Optional[float] = None  # PSI of predictions
    data_quality_score: Optional[float] = None
    last_serve_time: Optional[datetime] = None
    # Distributions of features at training time.
    training_distributions: list[FeatureDistribution] = field(default_factory=list)
    # Distributions of features at serving time.
    serving_distributions: list[FeatureDistribution] = field(default_factory=list)
    # Distribution of the model's predictions (serving).
    prediction_distribution: Optional[FeatureDistribution] = None


# --------------------------------------------------------------------------- #
# Findings — what the detectors emit and the reporter writes back to DataHub.
# --------------------------------------------------------------------------- #


class Severity(str, enum.Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class FindingType(str, enum.Enum):
    MODEL_DRIFT = "model_drift"
    BROKEN_LINEAGE = "broken_lineage"
    STALE_FEATURE = "stale_feature"
    SCHEMA_MISMATCH = "schema_mismatch"
    TRAINING_SERVING_SKEW = "training_serving_skew"


@dataclass
class Finding:
    """A single silent problem the agent detected."""

    type: FindingType
    severity: Severity
    title: str
    description: str
    # The URN of the entity the finding is attached to.
    entity_urn: str
    # Evidence — concrete numbers that a human (or another agent) can verify.
    evidence: dict[str, Any] = field(default_factory=dict)
    # A suggested remediation, expressed as DataHub operations where possible.
    remediation: str = ""
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def tag_name(self) -> str:
        """The DataHub tag we apply to flag the entity, e.g. ``ml-sentinel:drift``."""
        return f"ml-sentinel:{self.type.value.replace('_', '-')}"

    @property
    def assertion_urn(self) -> str:
        """A deterministic assertion URN for this finding.

        DataHub assertions are first-class entities; we synthesise a stable URN
        so re-running the agent updates the assertion instead of duplicating it.
        """
        import hashlib

        digest = hashlib.sha1(
            f"{self.entity_urn}:{self.type.value}".encode()
        ).hexdigest()[:12]
        return f"urn:li:assertion:mlSentinel:{self.type.value}:{digest}"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (for JSON output / DataHub documents)."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "entity_urn": self.entity_urn,
            "entity_name": self.evidence.get("entity_name", ""),
            "evidence": _jsonable(self.evidence),
            "remediation": self.remediation,
            "detected_at": self.detected_at.isoformat(),
            "assertion_urn": self.assertion_urn,
            "tag_name": self.tag_name,
        }


# --------------------------------------------------------------------------- #
# PSI helpers.
# --------------------------------------------------------------------------- #


def _psi_numeric(a: FeatureDistribution, b: FeatureDistribution) -> float:
    """PSI for numeric features using a simple 10-bucket histogram.

    We approximate buckets from the combined min/max, then compare the share
    of samples that fall in each bucket between the two distributions.
    """
    import math

    lo = min(a.min or 0, b.min or 0)
    hi = max(a.max or 1, b.max or 1)
    if hi <= lo:
        return 0.0
    n_buckets = 10
    edges = [lo + i * (hi - lo) / n_buckets for i in range(n_buckets + 1)]

    def bucket_share(dist: FeatureDistribution) -> list[float]:
        if dist.mean is None or dist.std is None:
            # Use a uniform fallback when we only have summary stats.
            return [1.0 / n_buckets] * n_buckets
        # Normal approximation: P(bucket) = CDF(edge_{i+1}) - CDF(edge_i)
        from math import erf, sqrt

        def cdf(x: float) -> float:
            return 0.5 * (1 + erf((x - dist.mean) / (dist.std * sqrt(2) or 1)))

        return [max(cdf(edges[i + 1]) - cdf(edges[i]), 1e-6) for i in range(n_buckets)]

    a_share = bucket_share(a)
    b_share = bucket_share(b)
    return _psi_from_shares(a_share, b_share)


def _psi_categorical(
    a: dict[str, float], b: dict[str, float]
) -> float:
    """PSI for categorical features."""
    cats = sorted(set(a) | set(b))
    a_share = [max(a.get(c, 1e-6), 1e-6) for c in cats]
    b_share = [max(b.get(c, 1e-6), 1e-6) for c in cats]
    # Normalise
    sa, sb = sum(a_share), sum(b_share)
    a_share = [v / sa for v in a_share]
    b_share = [v / sb for v in b_share]
    return _psi_from_shares(a_share, b_share)


def _psi_from_shares(expected: list[float], actual: list[float]) -> float:
    """Core PSI formula: sum( (actual - expected) * ln(actual / expected) )."""
    import math

    psi = 0.0
    for e, a in zip(expected, actual):
        e = max(e, 1e-6)
        a = max(a, 1e-6)
        psi += (a - e) * math.log(a / e)
    return psi


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of nested dataclasses / datetimes to JSON types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, enum.Enum):
        return obj.value
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj
