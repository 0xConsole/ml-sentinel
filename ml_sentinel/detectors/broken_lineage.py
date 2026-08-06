"""Broken lineage detector.

Detects entities whose lineage graph is incomplete or stale. Specifically:

1. **Missing upstream** — an entity (e.g. a feature table) that *should* have
   an upstream data source but has no upstream edges in DataHub. This usually
   means the ingestion job that populates the lineage stopped running or lost
   permissions.

2. **Stale edge** — a lineage edge that hasn't been confirmed in > 30 days,
   meaning DataHub hasn't seen the pipeline run in a month.

The detector walks the downstream lineage from the entity and checks that
every expected upstream is present. For ML models it verifies that both a
training dataset and a deployment exist in the lineage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import Entity, EntityType, Finding, FindingType, Severity

if TYPE_CHECKING:
    from ..datahub_client import DataHubClient

EDGE_STALE_DAYS = 30


class BrokenLineageDetector:
    """Find missing or stale lineage edges in the ML pipeline."""

    name = "broken_lineage"

    def detect(self, client: "DataHubClient", entity: Entity) -> list[Finding]:
        findings: list[Finding] = []

        if entity.type == EntityType.ML_FEATURE_TABLE:
            findings.extend(self._check_feature_table_lineage(client, entity))
        elif entity.type == EntityType.ML_MODEL:
            findings.extend(self._check_model_lineage(client, entity))

        return findings

    def _check_feature_table_lineage(
        self, client: "DataHubClient", entity: Entity
    ) -> list[Finding]:
        """A feature table should trace back to at least one raw dataset.

        We distinguish between *process* upstreams (the job that computes the
        features) and *dataset* upstreams (the raw data the features are built
        from). A feature table with only a process upstream but no dataset
        upstream has broken lineage — the raw-data provenance is lost.
        """
        findings = []
        upstreams = client.get_lineage(entity.urn, direction="upstream")

        has_dataset_upstream = any(
            "dataset" in e.upstream_urn.lower() for e in upstreams
        )
        has_process_upstream = any(
            "dataProcess" in e.upstream_urn for e in upstreams
        )

        if not upstreams:
            findings.append(
                Finding(
                    type=FindingType.BROKEN_LINEAGE,
                    severity=Severity.CRITICAL,
                    title=f"Feature table {entity.name} has no upstream lineage",
                    description=(
                        f"The feature table '{entity.name}' has no upstream "
                        f"entities in DataHub at all. Its source dataset and "
                        f"compute-job lineage are both missing — likely because "
                        f"the ingestion job stopped running or lost access."
                    ),
                    entity_urn=entity.urn,
                    evidence={
                        "entity_name": entity.name,
                        "upstream_count": 0,
                        "expected": ">=1 dataset + >=1 process upstream",
                    },
                    remediation=(
                        "Restart the feature-compute ingestion job and verify "
                        "its DataHub source connector still has source access."
                    ),
                )
            )
        elif not has_dataset_upstream and has_process_upstream:
            # Has a process upstream but no dataset upstream — the raw-data
            # provenance link is broken.
            findings.append(
                Finding(
                    type=FindingType.BROKEN_LINEAGE,
                    severity=Severity.CRITICAL,
                    title=f"Feature table {entity.name} missing raw-dataset lineage",
                    description=(
                        f"The feature table '{entity.name}' has a compute-job "
                        f"upstream but no raw-dataset upstream in DataHub. The "
                        f"provenance link from the source data to this feature "
                        f"table is broken — DataHub cannot trace which raw "
                        f"dataset the features were built from, so impact "
                        f"analysis and training-data audits are blocked."
                    ),
                    entity_urn=entity.urn,
                    evidence={
                        "entity_name": entity.name,
                        "upstream_count": len(upstreams),
                        "has_dataset_upstream": False,
                        "has_process_upstream": True,
                        "upstream_urns": [e.upstream_urn for e in upstreams],
                    },
                    remediation=(
                        "Re-emit the lineage edge from the raw source dataset "
                        f"to '{entity.name}' (or to its compute job) so the "
                        f"full training-data provenance is preserved."
                    ),
                )
            )
        else:
            # Check for stale edges.
            for edge in upstreams:
                if edge.is_stale:
                    findings.append(
                        Finding(
                            type=FindingType.BROKEN_LINEAGE,
                            severity=Severity.WARN,
                            title=f"Stale lineage edge upstream of {entity.name}",
                            description=(
                                f"The lineage edge from {edge.upstream_urn} to "
                                f"{entity.name} hasn't been confirmed in DataHub "
                                f"for over {EDGE_STALE_DAYS} days. The pipeline "
                                f"may have stopped emitting lineage events."
                            ),
                            entity_urn=entity.urn,
                            evidence={
                                "entity_name": entity.name,
                                "upstream_urn": edge.upstream_urn,
                                "edge_age_days": (
                                    datetime_now() - edge.created_at
                                ).days
                                if edge.created_at
                                else None,
                            },
                            remediation=(
                                "Verify the upstream job is still running and "
                                "emitting lineage to DataHub."
                            ),
                        )
                    )
        return findings

    def _check_model_lineage(
        self, client: "DataHubClient", entity: Entity
    ) -> list[Finding]:
        """A deployed model should have both a training dataset and a deployment downstream."""
        findings = []
        upstreams = client.get_lineage(entity.urn, direction="upstream")
        downstreams = client.get_lineage(entity.urn, direction="downstream")

        has_training_data = any(
            self._is_dataset(u.upstream_urn) for u in upstreams
        )
        has_deployment = any(
            self._is_deployment(d.downstream_urn) for d in downstreams
        )

        if not has_training_data:
            findings.append(
                Finding(
                    type=FindingType.BROKEN_LINEAGE,
                    severity=Severity.CRITICAL,
                    title=f"Model {entity.name} has no training dataset in lineage",
                    description=(
                        f"The model '{entity.name}' has no training dataset "
                        f"linked in its upstream lineage. Without this, the "
                        f"model's training provenance is untraceable and "
                        f"training-serving skew cannot be assessed."
                    ),
                    entity_urn=entity.urn,
                    evidence={
                        "entity_name": entity.name,
                        "upstream_count": len(upstreams),
                        "has_training_data": False,
                    },
                    remediation=(
                        "Link the training dataset as an upstream of this model "
                        "in DataHub so the full lineage path is preserved."
                    ),
                )
            )

        if not has_deployment:
            findings.append(
                Finding(
                    type=FindingType.BROKEN_LINEAGE,
                    severity=Severity.WARN,
                    title=f"Model {entity.name} is not deployed",
                    description=(
                        f"The model '{entity.name}' has no downstream deployment "
                        f"in DataHub. It may be undeployed or the deployment "
                        f"ingestion is missing."
                    ),
                    entity_urn=entity.urn,
                    evidence={
                        "entity_name": entity.name,
                        "downstream_count": len(downstreams),
                        "has_deployment": False,
                    },
                    remediation="Register the deployment in DataHub if the model is live.",
                )
            )
        return findings

    @staticmethod
    def _is_dataset(urn: str) -> bool:
        return "dataset" in urn.lower() or "mlFeatureTable" in urn

    @staticmethod
    def _is_deployment(urn: str) -> bool:
        return "mlModelDeployment" in urn or "deployment" in urn.lower()


def datetime_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
