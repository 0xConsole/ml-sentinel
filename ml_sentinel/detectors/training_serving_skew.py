"""Training-serving skew detector.

Detects distribution skew between the features the model was trained on and
the features it sees at serving time. Unlike *drift* (which compares the
model's predictions over time), *skew* compares the input features directly
between the two environments.

Common causes:
* The serving pipeline applies a different preprocessing / normalization.
* The training data was sampled / filtered differently from production traffic.
* A feature's definition changed between training and serving.

We compute the **Population Stability Index (PSI)** for each feature that has
both a training and a serving distribution recorded in DataHub:

* PSI < 0.10  — no skew
* PSI 0.10-0.25 — minor skew (WARN)
* PSI > 0.25  — significant skew (CRITICAL)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import Entity, EntityType, Finding, FindingType, Severity

if TYPE_CHECKING:
    from ..datahub_client import DataHubClient

SKEW_WARN = 0.10
SKEW_CRITICAL = 0.25


class TrainingServingSkewDetector:
    """Flag features whose train/serve distributions diverge (PSI)."""

    name = "training_serving_skew"

    def detect(self, client: "DataHubClient", entity: Entity) -> list[Finding]:
        if entity.type not in (EntityType.ML_MODEL, EntityType.ML_MODEL_DEPLOYMENT):
            return []

        model_urn = (
            entity.properties.get("model_urn", entity.urn)
            if entity.type == EntityType.ML_MODEL_DEPLOYMENT
            else entity.urn
        )

        metrics = client.get_model_metrics(model_urn)
        if not metrics or not metrics.training_distributions or not metrics.serving_distributions:
            return []

        # Index distributions by feature name.
        train = {d.feature_name: d for d in metrics.training_distributions}
        serve = {d.feature_name: d for d in metrics.serving_distributions}

        findings: list[Finding] = []
        for fname, train_dist in train.items():
            serve_dist = serve.get(fname)
            if serve_dist is None:
                continue  # handled by the schema-mismatch detector

            psi = train_dist.psi_against(serve_dist)
            if psi is None:
                continue

            if psi > SKEW_CRITICAL:
                findings.append(
                    Finding(
                        type=FindingType.TRAINING_SERVING_SKEW,
                        severity=Severity.CRITICAL,
                        title=f"Training-serving skew on feature '{fname}' ({entity.name})",
                        description=(
                            f"Feature '{fname}' has a training-serving PSI of "
                            f"{psi:.3f} (critical threshold {SKEW_CRITICAL}). "
                            f"The serving distribution has shifted far enough "
                            f"from training that the model's learned weights for "
                            f"this feature are likely miscalibrated."
                        ),
                        entity_urn=model_urn,
                        evidence={
                            "entity_name": entity.name,
                            "feature": fname,
                            "psi": round(psi, 4),
                            "threshold": SKEW_CRITICAL,
                            "training_stats": _dist_summary(train_dist),
                            "serving_stats": _dist_summary(serve_dist),
                        },
                        remediation=(
                            f"Compare the preprocessing for '{fname}' between "
                            f"the training and serving pipelines. If the serving "
                            f"pipeline is correct, retrain on recent serving data."
                        ),
                    )
                )
            elif psi > SKEW_WARN:
                findings.append(
                    Finding(
                        type=FindingType.TRAINING_SERVING_SKEW,
                        severity=Severity.WARN,
                        title=f"Minor training-serving skew on '{fname}' ({entity.name})",
                        description=(
                            f"Feature '{fname}' PSI={psi:.3f} (minor skew, "
                            f"threshold {SKEW_WARN}). Monitor."
                        ),
                        entity_urn=model_urn,
                        evidence={
                            "entity_name": entity.name,
                            "feature": fname,
                            "psi": round(psi, 4),
                            "threshold": SKEW_WARN,
                            "training_stats": _dist_summary(train_dist),
                            "serving_stats": _dist_summary(serve_dist),
                        },
                        remediation="Add to the watchlist; re-evaluate next run.",
                    )
                )
        return findings


def _dist_summary(d) -> dict:
    """Compact stats summary for evidence."""
    out = {}
    if d.mean is not None:
        out["mean"] = round(d.mean, 3)
    if d.std is not None:
        out["std"] = round(d.std, 3)
    if d.null_rate is not None:
        out["null_rate"] = round(d.null_rate, 4)
    if d.n_samples is not None:
        out["n_samples"] = d.n_samples
    if d.categories:
        out["categories"] = {k: round(v, 3) for k, v in d.categories.items()}
    return out
