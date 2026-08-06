# ML Sentinel

> **A Production ML agent that uses DataHub's end-to-end lineage graph to detect silent failures before they cost money.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-17%20passing-brightgreen)](#testing)

---

## The Problem

ML systems don't fail loudly — they fail **silently**. A feature pipeline stops
updating, a schema drops a column, a training-serving skew creeps in, and the
model keeps serving predictions. Nobody notices until the business impact is
already felt: revenue leaks, fraud spikes, recommendations go stale.

Existing ML monitoring tools (Evidently, NannyML, WhyLabs) detect *statistical*
problems — drift, skew, quality — but they operate on **isolated metrics stores**.
They don't see the full pipeline: which raw dataset feeds which feature table,
which feature table trains which model, which model powers which deployment.
When a drift alert fires, the on-call engineer has to manually trace the lineage
to find the root cause.

## The Solution

**ML Sentinel** is an agent that lives inside DataHub's context graph. It uses
DataHub's MCP Server to traverse the full ML lineage path —
`training data → features → models → deployments` — and runs a fleet of
detectors that surface silent problems **with their root cause already
traced**.

When it finds a problem, it doesn't just raise an alert in a separate dashboard.
It **writes the finding back to DataHub** as:

- **Assertions** (with failing run events) — visible in DataHub's observability views
- **Tags** (`ml-sentinel:model-drift`, `ml-sentinel:stale-feature`, ...) — visible in search and UI
- **Documents** — full narrative reports in DataHub's knowledge base

So the **next person or agent** that queries the catalog inherits the knowledge.

### Unique Angle

> Unlike existing ML monitoring tools (Evidently, NannyML, WhyLabs), ML Sentinel
> uses DataHub's end-to-end ML lineage graph to automatically trace silent
> failures across the entire ML pipeline — from training data to production
> predictions — and writes findings back to DataHub so the next person or agent
> inherits the knowledge.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ML Sentinel Agent                           │
│                                                                     │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────────┐   │
│  │  Discovery  │──▶│  Detector    │──▶│      Reporter            │   │
│  │  (search    │   │  Fleet       │   │  (write-back to DataHub) │   │
│  │   ML ents)  │   │              │   │                          │   │
│  └─────────────┘   └──────┬───────┘   └──────────┬───────────────┘   │
│                           │                       │                   │
│         ┌─────────────────┼───────────────────────┼─────────────┐   │
│         │                 │  Detectors            │  Writes     │   │
│         │  ┌──────────────┼──────────────────────┼──────────┐  │   │
│         │  │ Model Drift  │ Broken Lineage │ Stale Feature │  │   │
│         │  │ Schema Mism. │ Train/Serve Skew     │            │  │   │
│         │  └──────────────────────────────────────────────────┘  │   │
│         └─────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────────┘
                             │
                    DataHubClient (transport)
                    ┌────────┼────────────────┐
                    │        │                │
              ┌─────▼──┐  ┌──▼───┐  ┌────────▼─────┐
              │  MCP   │  │GraphQL│  │    Mock      │
              │ Server │  │ /SDK  │  │ (demo mode)  │
              │(npx)   │  │       │  │              │
              └────────┘  └───────┘  └──────────────┘
                  │           │
                  ▼           ▼
              ┌─────────────────────────┐
              │      DataHub            │
              │  (context graph)        │
              │                         │
              │  raw_transactions       │
              │       ↓                │
              │  user_txn_features      │
              │       ↓                │
              │  fraud_model_v3         │
              │       ↓                │
              │  fraud_detector_prod    │
              └─────────────────────────┘
```

**Data flow:**
1. The agent searches DataHub for all ML entities (models, feature tables, deployments)
2. For each entity, the detector fleet runs: drift, broken lineage, stale features, schema mismatch, training-serving skew
3. Findings are written back to DataHub as assertions + tags + documents
4. The next agent or human that queries the catalog sees the findings inline

## Silent Problems Detected

| Detector | What it finds | How it uses DataHub lineage |
|----------|--------------|----------------------------|
| **Model Drift** | Prediction distribution shifted (PSI > 0.25) | Follows deployment → model → training data to pull both distributions from the catalog |
| **Broken Lineage** | Missing or stale upstream edges | Traverses upstream lineage; flags feature tables with no raw-dataset provenance |
| **Stale Features** | Feature table past its freshness SLA | Reads `freshness_sla_hours` structured property + `last_modified` from the catalog |
| **Schema Mismatch** | Serving schema diverged from training schema | Compares schema fields of training vs. serving datasets linked to the model |
| **Training-Serving Skew** | Feature distribution skew (PSI) between train/serve | Pulls training + serving distributions from DataHub's structured properties |

## Quick Start (≤ 5 commands)

```bash
# 1. Clone
git clone https://github.com/0xConsole/ml-sentinel.git
cd ml-sentinel

# 2. Install
pip install -e .

# 3. Run the demo (mock mode — no DataHub needed)
ml-sentinel scan --mock

# 4. View the lineage graph
ml-sentinel lineage --mock

# 5. List the estate
ml-sentinel estate --mock
```

That's it. The demo runs against a bundled sample ML estate (a fraud-detection
pipeline) with five planted silent problems. You'll see all five detected.

### JSON output (for piping to other tools)

```bash
ml-sentinel scan --mock --json | jq '.findings[] | {type, severity, title}'
```

### Dry run (detect only, don't write back)

```bash
ml-sentinel scan --mock --no-write
```

## Connecting to a Real DataHub

### Via the DataHub MCP Server (recommended for the hackathon)

```bash
# The MCP server is auto-started via npx
ml-sentinel scan --mcp --datahub-url http://localhost:8080 --token <your-token>
```

This uses the official [`@acryldata/mcp-server-datahub`](https://github.com/acryldata/mcp-server-datahub)
MCP server and its tools: `search`, `get_lineage`, `get_entities`,
`list_schema_fields`, `add_tags`, `save_document`.

### Via the DataHub Python SDK / GraphQL

```bash
ml-sentinel scan --graphql --datahub-url http://localhost:8080 --token <your-token>
```

This uses `acryl-datahub`'s `DataHubClient` for direct GraphQL queries and
emits real assertion entities + run events.

### Environment variables

```bash
export DATAHUB_URL=http://localhost:8080
export DATAHUB_TOKEN=<your-personal-access-token>
```

See [`.env.example`](.env.example) for the full list.

## Demo Output

Running `ml-sentinel scan --mock` produces:

```
╭───────────────────────────────────────────────────╮
│ ML Sentinel — Production ML Silent-Problem Hunter │
╰─────────── Powered by DataHub lineage ────────────╯
  Entities scanned           3
  Findings                   5
  Severity                   5 critical · 0 warn · 0 info
  Duration                   0.00s
  Written back to DataHub    5 assertions · 5 tags · 2 documents

  # │ Severity │ Type                  │ Entity                  │ Finding
  1 │ CRITICAL │ model_drift           │ fraud_model_v3          │ Significant prediction drift…
  2 │ CRITICAL │ schema_mismatch       │ fraud_model_v3          │ Missing serving field 'merchant_category'…
  3 │ CRITICAL │ training_serving_skew│ fraud_model_v3          │ Training-serving skew on 'tx_amount'…
  4 │ CRITICAL │ broken_lineage       │ fraud.user_txn_features │ Missing raw-dataset lineage…
  5 │ CRITICAL │ stale_feature        │ fraud.user_txn_features │ Stale feature table (225h > 24h SLA)…

╭─ Write-Back ────────────────────────────────────────────────────────╮
│ Wrote back to DataHub:                                              │
│   • 5 assertions created                                            │
│   • 5 tags applied                                                  │
│   • 2 documents saved                                               │
│ In mock mode these are recorded in-memory. With a live DataHub they  │
│ become real catalog objects visible to every user and agent.         │
╰─────────────────────────────────────────────────────────────────────╯
```

See [`examples/sample-report.md`](examples/sample-report.md) for a full
example report and [`examples/sample-output.json`](examples/sample-output.json)
for the JSON output.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| DataHub transport (MCP) | [`@acryldata/mcp-server-datahub`](https://github.com/acryldata/mcp-server-datahub) via the `mcp` Python SDK |
| DataHub transport (SDK) | [`acryl-datahub`](https://pypi.org/project/acryl-datahub/) (Python SDK / GraphQL) |
| CLI rendering | [`rich`](https://github.com/Textualize/rich) |
| Data models | `pydantic`-compatible dataclasses |
| Local state | In-memory (SQLite-ready, not needed for the demo) |
| Testing | `pytest` |
| License | Apache 2.0 |

## Project Structure

```
ml-sentinel/
├── ml_sentinel/
│   ├── __init__.py              # Package metadata
│   ├── models.py                # Core data models (Entity, Finding, PSI)
│   ├── agent.py                 # MLSentinelAgent orchestrator
│   ├── reporter.py              # Write-back to DataHub (assertions/tags/docs)
│   ├── datahub_client.py        # MCP / GraphQL / Mock transports
│   ├── cli.py                   # CLI entry point (ml-sentinel)
│   ├── data/
│   │   └── sample_estate.py     # Bundled demo estate (fraud pipeline)
│   └── detectors/
│       ├── __init__.py
│       ├── base.py              # Detector protocol
│       ├── model_drift.py       # PSI-based prediction drift
│       ├── broken_lineage.py    # Missing/stale upstream edges
│       ├── stale_feature.py     # Freshness SLA breaches
│       ├── schema_mismatch.py   # Training vs. serving schema diff
│       └── training_serving_skew.py  # Feature distribution skew
├── tests/
│   └── test_agent.py            # 17 tests — all passing
├── examples/
│   ├── sample-report.md         # Example markdown report
│   └── sample-output.json       # Example JSON output
├── pyproject.toml
├── LICENSE
└── README.md
```

## Testing

```bash
pip install -e ".[test]"
pytest tests/ -v
```

All 17 tests pass. They verify that:
- Every detector fires on its planted problem
- Findings are deduplicated by assertion URN
- The reporter writes assertions, tags, and documents
- PSI computation works for numeric and categorical distributions

## What's Real vs. Mocked

| Component | Status |
|-----------|--------|
| Detector fleet (drift, lineage, stale, schema, skew) | ✅ Real — production-ready logic |
| PSI computation (numeric + categorical) | ✅ Real — tested |
| Reporter (assertions, tags, documents) | ✅ Real — writes via MCP/SDK |
| MCP transport (DataHub MCP Server) | ✅ Production-ready — uses official `mcp` SDK |
| GraphQL transport (DataHub Python SDK) | ✅ Production-ready — uses `acryl-datahub` |
| Mock estate (sample fraud pipeline) | ✅ Demo only — 5 planted problems |
| Live DataHub instance | Not required for the demo |

The demo runs entirely on the mock transport. To use a real DataHub, point
the CLI at your instance with `--mcp` or `--graphql`.

## How It Uses DataHub

ML Sentinel uses DataHub meaningfully — not just reading metadata, but
contributing back to the context graph:

**Reading (via MCP Server tools):**
- `search` — discover ML models, feature tables, deployments
- `get_lineage` — traverse upstream/downstream lineage (up to 3 hops)
- `get_entities` — fetch full metadata for specific URNs
- `list_schema_fields` — inspect dataset schemas for mismatch detection

**Writing back (via MCP Server mutation tools + SDK):**
- `add_tags` — tag affected entities with `ml-sentinel:<problem-type>`
- `save_document` — save narrative reports to DataHub's knowledge base
- Assertion entities + failing run events (via GraphQL / SDK) — so findings
  show up in DataHub's data-quality / observability views

This is the key differentiator: findings become **permanent, queryable
artifacts in the catalog**, not ephemeral alerts in a separate system.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
