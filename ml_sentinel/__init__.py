"""ML Sentinel — a Production ML agent that uses DataHub's lineage graph to
detect silent failures before they cost money.

ML Sentinel connects to a DataHub deployment (via the DataHub MCP Server or
the DataHub Python SDK / GraphQL API), traverses the end-to-end ML lineage graph
(training data -> features -> models -> deployments), and runs a fleet of
detectors that surface silent problems:

  * Model drift — prediction distribution shifted vs. the training baseline
  * Broken lineage links — upstream/downstream edges are missing or stale
  * Stale features — feature tables haven't been updated within their SLA
  * Schema mismatches — the serving schema drifted from the training schema
  * Training-serving skew — feature distributions differ between train/serve

Findings are written back to DataHub as assertions / incidents and as
structured properties on the affected entities, so the *next* person or agent
that queries the catalog inherits the knowledge.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
