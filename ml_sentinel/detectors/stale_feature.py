"""Stale feature detector.

Detects feature tables whose data hasn't been refreshed within their freshness
SLA. Each feature table can declare a ``freshness_sla_hours`` structured
property in DataHub; if the table's ``last_modified`` is older than the SLA,
the feature is stale.

Stale features are one of the most common silent killers in production ML: the
model keeps serving predictions based on features computed days or weeks ago,
and nobody notices until the predictions degrade — by which point the damage
is done.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..models import Entity, EntityType, Finding, FindingType, Severity

if TYPE_CHECKING:
    from ..datahub_client import DataHubClient


class StaleFeatureDetector:
    """Flag feature tables past their freshness SLA."""

    name = "stale_feature"

    def detect(self, client: "DataHubClient", entity: Entity) -> list[Finding]:
        if entity.type != EntityType.ML_FEATURE_TABLE:
            return []

        sla_hours = entity.properties.get("freshness_sla_hours")
        if not sla_hours or entity.last_modified is None:
            return []

        sla_hours = float(sla_hours)
        now = datetime.now(timezone.utc)
        age_hours = (now - entity.last_modified).total_seconds() / 3600.0

        if age_hours <= sla_hours:
            return []

        # Severity scales with how far past the SLA we are.
        ratio = age_hours / sla_hours
        if ratio >= 3:
            severity = Severity.CRITICAL
        elif ratio >= 1.5:
            severity = Severity.WARN
        else:
            severity = Severity.WARN

        findings = [
            Finding(
                type=FindingType.STALE_FEATURE,
                severity=severity,
                title=f"Stale feature table: {entity.name}",
                description=(
                    f"Feature table '{entity.name}' was last updated "
                    f"{age_hours:.1f} hours ago, exceeding its freshness SLA of "
                    f"{sla_hours:.0f}h by {ratio:.1f}x. Models consuming this "
                    f"table are making predictions on stale features."
                ),
                entity_urn=entity.urn,
                evidence={
                    "entity_name": entity.name,
                    "last_modified": entity.last_modified.isoformat(),
                    "age_hours": round(age_hours, 1),
                    "sla_hours": sla_hours,
                    "sla_breach_ratio": round(ratio, 2),
                },
                remediation=(
                    "Check the feature-compute pipeline (Airflow / Spark job) "
                    f"that writes '{entity.name}'. If the job failed, restart it. "
                    "If the SLA is unrealistic, update the "
                    "freshness_sla_hours property in DataHub."
                ),
            )
        ]
        return findings
