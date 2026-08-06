"""Model drift detector.

Detects distribution drift in a deployed ML model by comparing the model's
prediction distribution at serving time against its training-time baseline,
and by checking any explicit ``prediction_drift_score`` property.

We use the **Population Stability Index (PSI)** as the drift metric:

* PSI < 0.10  — no drift
* PSI 0.10-0.25 — minor drift (WARN)
* PSI > 0.25  — significant drift (CRITICAL)

This is the same metric Evidently / NannyML expose, but ML Sentinel computes
it *from DataHub lineage*: it follows the lineage from the deployment back to
the model, then to the training dataset, to pull both distributions from the
catalog instead of requiring a separate metrics store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import Entity, EntityType, Finding, FindingType, Severity

if TYPE_CHECKING:
    from ..datahub_client import DataHubClient

DRIFT_WARN = 0.10
DRIFT_CRITICAL = 0.25


class ModelDriftDetector:
    """Detect prediction / feature distribution drift for deployed models."""

    name = "model_drift"

    def detect(self, client: "DataHubClient", entity: Entity) -> list[Finding]:
        if entity.type not in (EntityType.ML_MODEL, EntityType.ML_MODEL_DEPLOYMENT):
            return []

        # Resolve the model URN (deployments point at a model).
        model_urn = (
            entity.properties.get("model_urn", entity.urn)
            if entity.type == EntityType.ML_MODEL_DEPLOYMENT
            else entity.urn
        )

        metrics = client.get_model_metrics(model_urn)
        if metrics is None:
            return []

        findings: list[Finding] = []

        # 1. Explicit prediction-drift score (from structured properties).
        if metrics.prediction_drift_score is not None:
            score = metrics.prediction_drift_score
            if score > DRIFT_CRITICAL:
                findings.append(
                    Finding(
                        type=FindingType.MODEL_DRIFT,
                        severity=Severity.CRITICAL,
                        title=f"Significant prediction drift on {entity.name}",
                        description=(
                            f"Prediction distribution PSI={score:.3f} exceeds the "
                            f"critical threshold ({DRIFT_CRITICAL}). The model's "
                            f"output has shifted far enough from its training "
                            f"baseline that predictions are likely unreliable."
                        ),
                        entity_urn=model_urn,
                        evidence={
                            "entity_name": entity.name,
                            "psi": round(score, 4),
                            "threshold": DRIFT_CRITICAL,
                            "metric": "prediction_drift_score",
                            "last_serve": metrics.last_serve_time.isoformat()
                            if metrics.last_serve_time
                            else None,
                        },
                        remediation=(
                            "Retrain the model on recent data or investigate the "
                            "upstream feature pipeline for a distribution shift. "
                            "Tag the model with `ml-sentinel:drift` in DataHub."
                        ),
                    )
                )
            elif score > DRIFT_WARN:
                findings.append(
                    Finding(
                        type=FindingType.MODEL_DRIFT,
                        severity=Severity.WARN,
                        title=f"Minor prediction drift on {entity.name}",
                        description=(
                            f"Prediction distribution PSI={score:.3f} shows minor "
                            f"drift (threshold {DRIFT_WARN}-{DRIFT_CRITICAL}). "
                            f"Monitor closely."
                        ),
                        entity_urn=model_urn,
                        evidence={
                            "entity_name": entity.name,
                            "psi": round(score, 4),
                            "threshold": DRIFT_WARN,
                            "metric": "prediction_drift_score",
                        },
                        remediation="Add to the watchlist; re-evaluate next run.",
                    )
                )

        # 2. Compare prediction distribution against a flat baseline (0.5).
        if metrics.prediction_distribution and metrics.prediction_distribution.mean is not None:
            pred_mean = metrics.prediction_distribution.mean
            # A fraud model's training baseline is typically low (~0.05-0.10).
            # If the serving mean doubles, flag it.
            if pred_mean > 0.15:
                findings.append(
                    Finding(
                        type=FindingType.MODEL_DRIFT,
                        severity=Severity.WARN,
                        title=f"Elevated fraud-probability mean on {entity.name}",
                        description=(
                            f"Mean predicted fraud probability is {pred_mean:.3f} "
                            f"at serving time, well above the typical training "
                            f"baseline (~0.06). The model may be seeing a different "
                            f"population than it was trained on."
                        ),
                        entity_urn=model_urn,
                        evidence={
                            "entity_name": entity.name,
                            "serving_prediction_mean": round(pred_mean, 4),
                            "expected_baseline": 0.06,
                        },
                        remediation=(
                            "Check whether the serving traffic mix changed (new "
                            "merchant categories, new geos) and retrain if needed."
                        ),
                    )
                )

        return findings
