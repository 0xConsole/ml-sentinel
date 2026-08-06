"""Reporter — writes findings back to DataHub.

This is the **write-back** half of ML Sentinel and the part that makes the
agent's knowledge *inheritable*: instead of surfacing problems in a separate
dashboard that nobody looks at, ML Sentinel writes every finding back into the
DataHub catalog itself, in three complementary ways:

1. **Assertions** — a real DataHub assertion entity (with a failing run event)
   is created for each finding, so it shows up in DataHub's data-quality /
   observability views alongside other assertions. The next person or agent
   that queries the entity sees the failure.

2. **Tags** — the affected entity is tagged with ``ml-sentinel:<type>`` (e.g.
   ``ml-sentinel:model-drift``) so it's immediately visible in search and in
   the DataHub UI.

3. **Documents** — a summary document is saved to DataHub's knowledge base so
   a human or another agent can read the full narrative (what happened, the
   evidence, and the suggested remediation).

The reporter uses whatever write tools the active transport exposes. The
MCP transport uses ``add_tags`` + ``save_document`` (+ the GraphQL API for
assertions when available). The mock transport records everything in memory
so the demo can show what *would* be written.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .models import Finding, Severity

if TYPE_CHECKING:
    from .datahub_client import DataHubClient

logger = logging.getLogger(__name__)


class Reporter:
    """Writes :class:`Finding` objects back to DataHub."""

    def __init__(self, client: "DataHubClient"):
        self.client = client

    def report(self, findings: list[Finding]) -> dict[str, int]:
        """Write all findings back and return a small stats dict."""
        assertions = 0
        tags = 0
        documents = 0

        # Group findings by entity for the document summary.
        by_entity: dict[str, list[Finding]] = {}
        for finding in findings:
            by_entity.setdefault(finding.entity_urn, []).append(finding)

        for finding in findings:
            # 1. Assertion (real entity in DataHub when the transport supports it).
            try:
                if self.client.add_assertion(finding):
                    assertions += 1
            except Exception as exc:
                logger.error("Failed to write assertion for %s: %s", finding.entity_urn, exc)

            # 2. Tag.
            try:
                if self.client.add_tag(finding.entity_urn, finding.tag_name):
                    tags += 1
            except Exception as exc:
                logger.error("Failed to tag %s: %s", finding.entity_urn, exc)

        # 3. One summary document per entity (not per finding, to avoid noise).
        for urn, entity_findings in by_entity.items():
            title = f"ML Sentinel report — {entity_findings[0].evidence.get('entity_name', urn)}"
            content = self._render_document(urn, entity_findings)
            try:
                if self.client.save_document(title, content):
                    documents += 1
            except Exception as exc:
                logger.error("Failed to save document for %s: %s", urn, exc)

        logger.info(
            "Wrote %d assertions, %d tags, %d documents for %d findings",
            assertions,
            tags,
            documents,
            len(findings),
        )
        return {
            "assertions": assertions,
            "tags": tags,
            "documents": documents,
        }

    @staticmethod
    def _render_document(urn: str, findings: list[Finding]) -> str:
        """Render a markdown summary document for an entity's findings."""
        lines = [
            f"# ML Sentinel Report",
            "",
            f"**Entity:** `{urn}`",
            f"**Findings:** {len(findings)}",
            f"**Generated:** {findings[0].detected_at.isoformat()}",
            "",
            "## Findings",
            "",
        ]
        for i, f in enumerate(findings, 1):
            emoji = {"CRITICAL": "🔴", "WARN": "🟡", "INFO": "🔵"}.get(
                f.severity.value, "⚪"
            )
            lines.extend(
                [
                    f"### {emoji} {i}. {f.title}",
                    "",
                    f"- **Type:** `{f.type.value}`",
                    f"- **Severity:** `{f.severity.value}`",
                    f"- **Assertion URN:** `{f.assertion_urn}`",
                    f"- **Tag:** `{f.tag_name}`",
                    "",
                    f"**Description**",
                    "",
                    f.description,
                    "",
                    f"**Evidence**",
                    "",
                ]
            )
            for key, value in f.evidence.items():
                lines.append(f"- `{key}`: `{value}`")
            lines.extend(
                [
                    "",
                    f"**Remediation**",
                    "",
                    f.remediation,
                    "",
                    "---",
                    "",
                ]
            )
        lines.append(
            "\n_Generated by [ML Sentinel](https://github.com/0xConsole/ml-sentinel) "
            "— a Production ML agent for DataHub._"
        )
        return "\n".join(lines)
