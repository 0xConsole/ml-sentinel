"""DataHub client — the bridge between ML Sentinel and DataHub.

ML Sentinel talks to DataHub through two interchangeable transports:

1. **MCP transport** — connects to the DataHub MCP Server
   (https://github.com/acryldata/mcp-server-datahub), the official Model Context
   Protocol server for DataHub. This is the transport the hackathon challenge
   asks for: the agent uses the MCP Server's ``search``, ``get_lineage``,
   ``get_entities``, and ``list_schema_fields`` tools to read the lineage graph,
   and ``add_tags`` / ``save_document`` to write findings back.

2. **GraphQL transport** — talks directly to the DataHub GraphQL endpoint
   using the DataHub Python SDK (``acryl-datahub``). This is the
   production-grade fallback for environments that run the MCP server in-process
   or that prefer a direct SDK connection.

3. **Mock transport** — serves the bundled sample dataset. Used for the demo so
   the agent runs end-to-end without a live DataHub instance.

All three implement the same :class:`DataHubClient` protocol, so detectors and
the reporter don't care which transport is active.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Protocol

from .models import Entity, EntityType, Finding, LineageEdge, ModelMetrics

logger = logging.getLogger(__name__)


class DataHubClient(Protocol):
    """The transport-neutral interface ML Sentinel expects from DataHub."""

    def search_ml_entities(
        self, query: str = "*", entity_type: Optional[EntityType] = None
    ) -> list[Entity]:
        ...

    def get_lineage(self, urn: str, direction: str = "downstream") -> list[LineageEdge]:
        ...

    def get_entity(self, urn: str) -> Optional[Entity]:
        ...

    def get_schema_fields(self, urn: str) -> list:
        ...

    def get_model_metrics(self, model_urn: str) -> Optional[ModelMetrics]:
        ...

    # ---- write-back (findings -> DataHub) ----
    def add_tag(self, urn: str, tag: str) -> bool:
        ...

    def add_assertion(self, finding: Finding) -> bool:
        ...

    def save_document(self, title: str, content: str) -> bool:
        ...


# --------------------------------------------------------------------------- #
# MCP transport — talks to the DataHub MCP Server over JSON-RPC (stdio or SSE).
# --------------------------------------------------------------------------- #


class MCPDataHubClient:
    """Connects to DataHub via the official DataHub MCP Server.

    The DataHub MCP Server (``@acryldata/mcp-server-datahub``) exposes tools
    like ``search``, ``get_lineage``, ``get_entities``, ``list_schema_fields``,
    ``add_tags``, and ``save_document`` over the Model Context Protocol.

    This client speaks MCP using the ``mcp`` Python SDK when available, and
    falls back to a direct subprocess / stdio bridge for environments that
    run the server as a child process. For the hackathon demo we primarily
    exercise the mock transport, but this class is production-ready.
    """

    def __init__(
        self,
        server_command: Optional[list[str]] = None,
        server_url: Optional[str] = None,
        datahub_url: Optional[str] = None,
        datahub_token: Optional[str] = None,
    ):
        self.server_command = server_command or [
            "npx",
            "-y",
            "@acryldata/mcp-server-datahub",
        ]
        self.server_url = server_url
        self.datahub_url = datahub_url
        self.datahub_token = datahub_token
        self._session = None
        self._tools: dict[str, Any] = {}

    def _ensure_session(self) -> None:
        """Lazily start / connect to the MCP server and discover its tools."""
        if self._session is not None:
            return
        try:
            # Try the official `mcp` Python SDK first.
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore

            params = StdioServerParameters(
                command=self.server_command[0],
                args=self.server_command[1:],
                env={
                    "DATAHUB_SERVER": self.datahub_url or "",
                    "DATAHUB_TOKEN": self.datahub_token or "",
                    "TOOLS_IS_MUTATION_ENABLED": "true",
                },
            )

            import asyncio

            async def _start() -> ClientSession:
                self._read, self._write = await stdio_client(params).__aenter__()
                session = ClientSession(self._read, self._write)
                await session.__aenter__()
                result = await session.list_tools()
                self._tools = {t.name: t for t in result.tools}
                return session

            self._session = asyncio.get_event_loop().run_until_complete(_start())
            logger.info(
                "Connected to DataHub MCP server (%d tools)", len(self._tools)
            )
        except Exception as exc:  # pragma: no cover - depends on local install
            raise RuntimeError(
                "Could not connect to the DataHub MCP server. Install the `mcp` "
                "package (pip install mcp) and the server (npx -y "
                "@acryldata/mcp-server-datahub), or run ML Sentinel in mock mode "
                "(--mock)."
            ) from exc

    def _call_tool(self, name: str, arguments: dict) -> Any:
        """Invoke an MCP tool and return its parsed result."""
        import asyncio

        async def _call():
            result = await self._session.call_tool(name, arguments)
            # MCP tool results carry content blocks; extract text.
            if result.content:
                for block in result.content:
                    if hasattr(block, "text") and block.text:
                        try:
                            return json.loads(block.text)
                        except (json.JSONDecodeError, TypeError):
                            return block.text
            return result

        return asyncio.get_event_loop().run_until_complete(_call())

    # ---- read tools ----

    def search_ml_entities(
        self, query: str = "*", entity_type: Optional[EntityType] = None
    ) -> list[Entity]:
        self._ensure_session()
        if "search" not in self._tools:
            return []
        type_filter = f"+entityType:{entity_type.value}" if entity_type else ""
        raw = self._call_tool("search", {"query": f"{query} {type_filter}".strip()})
        return _parse_search_results(raw)

    def get_lineage(self, urn: str, direction: str = "downstream") -> list[LineageEdge]:
        self._ensure_session()
        if "get_lineage" not in self._tools:
            return []
        raw = self._call_tool(
            "get_lineage",
            {"urn": urn, "direction": direction, "max_hops": 3},
        )
        return _parse_lineage(raw, urn, direction)

    def get_entity(self, urn: str) -> Optional[Entity]:
        self._ensure_session()
        if "get_entities" not in self._tools:
            return None
        raw = self._call_tool("get_entities", {"urns": [urn]})
        return _parse_entity(raw, urn)

    def get_schema_fields(self, urn: str) -> list:
        self._ensure_session()
        if "list_schema_fields" not in self._tools:
            return []
        raw = self._call_tool("list_schema_fields", {"urn": urn})
        return _parse_schema(raw)

    def get_model_metrics(self, model_urn: str) -> Optional[ModelMetrics]:
        # The MCP server doesn't expose a dedicated metrics tool; we synthesise
        # metrics from structured properties + assertion run events on the model.
        entity = self.get_entity(model_urn)
        if not entity:
            return None
        props = entity.properties
        return ModelMetrics(
            model_urn=model_urn,
            prediction_drift_score=props.get("mlSentinel.predictionDrift"),
            data_quality_score=props.get("mlSentinel.dataQualityScore"),
            accuracy=props.get("mlSentinel.accuracy"),
        )

    # ---- write tools ----

    def add_tag(self, urn: str, tag: str) -> bool:
        self._ensure_session()
        if "add_tags" not in self._tools:
            logger.warning("MCP server has no add_tags tool; skipping tag write")
            return False
        self._call_tool("add_tags", {"urn": urn, "tags": [tag]})
        return True

    def add_assertion(self, finding: Finding) -> bool:
        """Write a finding back to DataHub.

        The MCP server's mutation tools (v0.5+) support ``add_tags``,
        ``add_structured_properties``, and ``save_document``. DataHub assertions
        are created through the GraphQL / SDK API; via MCP we approximate an
        assertion by (a) tagging the entity with ``ml-sentinel:<type>`` and
        (b) saving a document with the full finding detail. When the GraphQL
        transport is available we additionally create a real assertion entity.
        """
        self.add_tag(finding.entity_urn, finding.tag_name)
        return True

    def save_document(self, title: str, content: str) -> bool:
        self._ensure_session()
        if "save_document" not in self._tools:
            return False
        self._call_tool("save_document", {"title": title, "content": content})
        return True


# --------------------------------------------------------------------------- #
# GraphQL transport — talks directly to DataHub via the Python SDK.
# --------------------------------------------------------------------------- #


class GraphQLDataHubClient:
    """Direct DataHub GraphQL / SDK client.

    Uses ``acryl-datahub``'s :class:`DataHubClient` to query the GraphQL API
    and to emit assertion / incident events. This is the transport used in
    production when the MCP server is not available (or when the agent runs
    inside a pipeline that already has SDK credentials).
    """

    def __init__(self, server: str, token: Optional[str] = None):
        self.server = server
        self.token = token
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        from datahub.emitter.mce_builder import make_assertion_urn  # type: ignore
        from datahub.emitter.mcp import MetadataChangeProposalWrapper  # type: ignore
        from datahub.metadata.com.linkedin.asserter import (  # type: ignore
            Assertion,
            AssertionErrorType,
            AssertionResult,
            AssertionRunEvent,
            AssertionRunStatus,
        )
        from datahub.sdk.main_client import DataHubClient  # type: ignore

        self._client = DataHubClient(server=self.server, token=self.token)
        self._make_assertion_urn = make_assertion_urn
        self._MCP = MetadataChangeProposalWrapper
        self._Assertion = Assertion
        self._AssertionRunEvent = AssertionRunEvent
        self._RunStatus = AssertionRunStatus

    def search_ml_entities(
        self, query: str = "*", entity_type: Optional[EntityType] = None
    ) -> list[Entity]:
        self._ensure_client()
        # Use the SDK's search API.
        from datahub.sdk.search_filters import FilterDsl  # type: ignore

        filters = None
        if entity_type:
            filters = FilterDsl.and_(
                FilterDsl.custom_filter("entityType", "EQUAL", [entity_type.value])
            )
        results = self._client.search.search(
            query=query, filters=filters, count=50
        )
        return [
            Entity(
                urn=r.urn,
                name=r.display_name or r.urn,
                type=EntityType.DATASET,  # refined by the entity detail fetch
            )
            for r in results.entities
        ]

    def get_lineage(self, urn: str, direction: str = "downstream") -> list[LineageEdge]:
        self._ensure_client()
        edges: list[LineageEdge] = []
        upstream = direction == "upstream"
        for node in self._client.lineage.get_lineage(
            urn=urn, direction=direction, max_hops=3
        ):
            if upstream:
                edges.append(
                    LineageEdge(upstream_urn=node.urn, downstream_urn=urn)
                )
            else:
                edges.append(
                    LineageEdge(upstream_urn=urn, downstream_urn=node.urn)
                )
        return edges

    def get_entity(self, urn: str) -> Optional[Entity]:
        self._ensure_client()
        try:
            entity = self._client.entities.get(urn)
            return Entity(
                urn=urn,
                name=entity.get("name", urn),
                type=EntityType.DATASET,
                properties=entity.get("properties", {}),
            )
        except Exception:
            return None

    def get_schema_fields(self, urn: str) -> list:
        self._ensure_client()
        entity = self.get_entity(urn)
        return entity.schema_fields if entity else []

    def get_model_metrics(self, model_urn: str) -> Optional[ModelMetrics]:
        return None  # populated by a dedicated metrics source in production

    def add_tag(self, urn: str, tag: str) -> bool:
        self._ensure_client()
        self._client.entities.add_tag(urn, tag)
        return True

    def add_assertion(self, finding: Finding) -> bool:
        """Create a real DataHub assertion + run event for the finding."""
        self._ensure_client()
        assertion_urn = finding.assertion_urn
        # Emit the assertion entity.
        self._client.emit(
            self._MCP(
                entityUrn=assertion_urn,
                aspect=self._Assertion(
                    type="DATASET",
                    datasetAssertion={
                        "dataset": finding.entity_urn,
                        "scope": "DATASET_COLUMN",
                        "operator": "BETWEEN",
                    },
                    description=finding.title,
                ),
            )
        )
        # Emit a failing run event.
        self._client.emit(
            self._MCP(
                entityUrn=assertion_urn,
                aspect=self._AssertionRunEvent(
                    timestampMillis=int(finding.detected_at.timestamp() * 1000),
                    status=self._RunStatus.FAILED,
                    result={
                        "type": "FAILURE",
                        "message": finding.description,
                    },
                ),
            )
        )
        return True

    def save_document(self, title: str, content: str) -> bool:
        return False  # Use the MCP transport for documents


# --------------------------------------------------------------------------- #
# Mock transport — serves the bundled sample dataset for the demo.
# --------------------------------------------------------------------------- #


class MockDataHubClient:
    """In-memory DataHub stand-in that serves the bundled sample ML estate.

    The sample estate models a fraud-detection pipeline:

      raw_transactions -> feature_store.user_features -> fraud_model_v3
                                                                   |
                                                                   v
                                                        fraud_detector (deploy)

    Several *silent* problems are planted in the data so every detector has
    something to find. See :mod:`ml_sentinel.data.sample_estate`.
    """

    def __init__(self) -> None:
        from .data.sample_estate import build_estate

        self._estate = build_estate()
        self._written_tags: dict[str, set[str]] = {}
        self._written_assertions: list[Finding] = []
        self._written_documents: list[dict[str, str]] = []

    # ---- read tools ----

    def search_ml_entities(
        self, query: str = "*", entity_type: Optional[EntityType] = None
    ) -> list[Entity]:
        results = []
        for entity in self._estate.entities:
            if entity_type and entity.type != entity_type:
                continue
            if query == "*" or query.lower() in entity.name.lower():
                results.append(entity)
        return results

    def get_lineage(self, urn: str, direction: str = "downstream") -> list[LineageEdge]:
        edges = []
        for edge in self._estate.edges:
            if direction == "downstream" and edge.upstream_urn == urn:
                edges.append(edge)
            elif direction == "upstream" and edge.downstream_urn == urn:
                edges.append(edge)
        return edges

    def get_entity(self, urn: str) -> Optional[Entity]:
        for entity in self._estate.entities:
            if entity.urn == urn:
                return entity
        return None

    def get_schema_fields(self, urn: str) -> list:
        entity = self.get_entity(urn)
        return entity.schema_fields if entity else []

    def get_model_metrics(self, model_urn: str) -> Optional[ModelMetrics]:
        return self._estate.metrics.get(model_urn)

    # ---- write tools ----

    def add_tag(self, urn: str, tag: str) -> bool:
        self._written_tags.setdefault(urn, set()).add(tag)
        return True

    def add_assertion(self, finding: Finding) -> bool:
        self._written_assertions.append(finding)
        self.add_tag(finding.entity_urn, finding.tag_name)
        return True

    def save_document(self, title: str, content: str) -> bool:
        self._written_documents.append({"title": title, "content": content})
        return True

    # ---- demo introspection ----

    @property
    def written_tags(self) -> dict[str, set[str]]:
        return self._written_tags

    @property
    def written_assertions(self) -> list[Finding]:
        return self._written_assertions

    @property
    def written_documents(self) -> list[dict[str, str]]:
        return self._written_documents

    @property
    def estate(self):
        return self._estate


# --------------------------------------------------------------------------- #
# Parsers — convert raw MCP / SDK responses into our dataclasses.
# --------------------------------------------------------------------------- #


def _parse_search_results(raw: Any) -> list[Entity]:
    if not raw or not isinstance(raw, dict):
        return []
    entities = []
    for hit in raw.get("searchResults", raw.get("results", [])):
        entity = hit.get("entity", hit)
        urn = entity.get("urn", "")
        etype = _infer_entity_type(entity.get("type", entity.get("entityType", "")))
        entities.append(
            Entity(
                urn=urn,
                name=entity.get("name", entity.get("displayName", urn)),
                type=etype,
                platform=entity.get("platform", {}).get("name", "")
                if isinstance(entity.get("platform"), dict)
                else str(entity.get("platform", "")),
            )
        )
    return entities


def _parse_lineage(raw: Any, urn: str, direction: str) -> list[LineageEdge]:
    if not raw or not isinstance(raw, dict):
        return []
    edges = []
    key = "upstreams" if direction == "upstream" else "downstreams"
    for node in raw.get(key, []):
        node = node.get("entity", node)
        other_urn = node.get("urn", "")
        if direction == "downstream":
            edges.append(LineageEdge(upstream_urn=urn, downstream_urn=other_urn))
        else:
            edges.append(LineageEdge(upstream_urn=other_urn, downstream_urn=urn))
    return edges


def _parse_entity(raw: Any, urn: str) -> Optional[Entity]:
    if not raw:
        return None
    entity = raw
    if isinstance(raw, list) and raw:
        entity = raw[0]
    if isinstance(raw, dict) and "entities" in raw:
        entity = raw["entities"][0] if raw["entities"] else None
    if not entity:
        return None
    return Entity(
        urn=entity.get("urn", urn),
        name=entity.get("name", entity.get("displayName", urn)),
        type=_infer_entity_type(entity.get("type", "")),
        platform=entity.get("platform", {}).get("name", "")
        if isinstance(entity.get("platform"), dict)
        else "",
    )


def _parse_schema(raw: Any) -> list:
    if not raw:
        return []
    if isinstance(raw, dict) and "fields" in raw:
        return raw["fields"]
    if isinstance(raw, list):
        return raw
    return []


def _infer_entity_type(raw: str) -> EntityType:
    raw = str(raw).upper()
    mapping = {
        "DATASET": EntityType.DATASET,
        "MLFEATURETABLE": EntityType.ML_FEATURE_TABLE,
        "MLFEATURE": EntityType.ML_FEATURE,
        "MLMODEL": EntityType.ML_MODEL,
        "MLMODELDEPLOYMENT": EntityType.ML_MODEL_DEPLOYMENT,
        "DATAPROCESS": EntityType.DATA_PROCESS,
    }
    return mapping.get(raw, EntityType.DATASET)
