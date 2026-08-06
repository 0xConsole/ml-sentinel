"""ML Sentinel agent — the orchestrator.

The agent is the brain that ties everything together:

1. It asks DataHub for every ML entity in the estate (models, feature tables,
   deployments).
2. For each entity it runs the full detector fleet.
3. It collects all findings and hands them to the :class:`Reporter`, which
   writes them back to DataHub as assertions / tags / documents.

The agent is transport-agnostic — point it at a mock, MCP, or GraphQL client
and it behaves identically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .datahub_client import DataHubClient
from .detectors import (
    BrokenLineageDetector,
    ModelDriftDetector,
    SchemaMismatchDetector,
    StaleFeatureDetector,
    TrainingServingSkewDetector,
)
from .models import Entity, EntityType, Finding

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """The result of a full agent scan."""

    entities_scanned: int = 0
    findings: list[Finding] = field(default_factory=list)
    assertions_written: int = 0
    tags_written: int = 0
    documents_written: int = 0
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity.value == "CRITICAL")

    @property
    def warn_count(self) -> int:
        return sum(1 for f in self.findings if f.severity.value == "WARN")

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity.value == "INFO")

    def by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.type.value] = counts.get(f.type.value, 0) + 1
        return counts


class MLSentinelAgent:
    """The production-ML silent-problem hunter."""

    def __init__(self, client: DataHubClient, write_back: bool = True):
        self.client = client
        self.write_back = write_back
        # The detector fleet — one per silent-problem class.
        self.detectors = [
            ModelDriftDetector(),
            BrokenLineageDetector(),
            StaleFeatureDetector(),
            SchemaMismatchDetector(),
            TrainingServingSkewDetector(),
        ]

    def scan(
        self,
        entity_types: Optional[list[EntityType]] = None,
        query: str = "*",
    ) -> ScanResult:
        """Run the full detector fleet against the ML estate.

        Args:
            entity_types: Restrict the scan to these entity types. Defaults to
                models, feature tables, and deployments.
            query: DataHub search query to scope the estate.
        """
        import time

        start = time.monotonic()
        result = ScanResult()

        if entity_types is None:
            entity_types = [
                EntityType.ML_MODEL,
                EntityType.ML_FEATURE_TABLE,
                EntityType.ML_MODEL_DEPLOYMENT,
            ]

        # Gather the target entities from DataHub.
        all_entities: list[Entity] = []
        for etype in entity_types:
            entities = self.client.search_ml_entities(query=query, entity_type=etype)
            all_entities.extend(entities)
            logger.info("Found %d %s entities", len(entities), etype.value)

        result.entities_scanned = len(all_entities)
        seen_urns: set[str] = set()

        # Run every detector against every entity.
        for entity in all_entities:
            logger.debug("Scanning %s (%s)", entity.name, entity.type.value)
            for detector in self.detectors:
                try:
                    findings = detector.detect(self.client, entity)
                    for finding in findings:
                        # Deduplicate by assertion URN — the same finding can be
                        # produced for both a model and its deployment (they
                        # resolve to the same model_urn). Keep the first.
                        if finding.assertion_urn in seen_urns:
                            continue
                        seen_urns.add(finding.assertion_urn)
                        result.findings.append(finding)
                    if findings:
                        logger.info(
                            "Detector '%s' found %d issue(s) on %s",
                            detector.name,
                            len(findings),
                            entity.name,
                        )
                except Exception as exc:
                    logger.error(
                        "Detector '%s' crashed on %s: %s",
                        detector.name,
                        entity.name,
                        exc,
                    )

        # Write findings back to DataHub.
        if self.write_back and result.findings:
            from .reporter import Reporter

            reporter = Reporter(self.client)
            stats = reporter.report(result.findings)
            result.assertions_written = stats["assertions"]
            result.tags_written = stats["tags"]
            result.documents_written = stats["documents"]

        result.duration_seconds = time.monotonic() - start
        return result

    def scan_single(self, urn: str) -> ScanResult:
        """Run the fleet against a single entity by URN (used by the MCP tool)."""
        entity = self.client.get_entity(urn)
        result = ScanResult(entities_scanned=1)
        if entity is None:
            return result
        for detector in self.detectors:
            try:
                result.findings.extend(detector.detect(self.client, entity))
            except Exception as exc:
                logger.error("Detector '%s' crashed on %s: %s", detector.name, urn, exc)
        if self.write_back and result.findings:
            from .reporter import Reporter

            reporter = Reporter(self.client)
            stats = reporter.report(result.findings)
            result.assertions_written = stats["assertions"]
            result.tags_written = stats["tags"]
            result.documents_written = stats["documents"]
        return result
