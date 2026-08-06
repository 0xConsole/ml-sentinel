# Devpost Submission Text — ML Sentinel

## Project Title

ML Sentinel — A Production ML Agent that Detects Silent Failures Using DataHub's Lineage Graph

## Short Description (140 chars)

ML Sentinel uses DataHub's MCP Server to traverse end-to-end ML lineage and catch silent problems — drift, broken lineage, stale features, schema mismatches, training-serving skew — before they cost money. It writes findings back to DataHub so the next agent inherits the knowledge.

## Full Description

### The Problem

ML systems don't fail loudly — they fail **silently**. A feature pipeline stops updating, a serving schema drops a column, a training-serving skew creeps in, and the model keeps serving predictions. Nobody notices until the business impact is already felt: revenue leaks, fraud spikes, recommendations go stale.

Existing ML monitoring tools (Evidently, NannyML, WhyLabs) detect statistical problems — drift, skew, quality — but they operate on isolated metrics stores. They don't see the full pipeline: which raw dataset feeds which feature table, which feature table trains which model, which model powers which deployment. When a drift alert fires, the on-call engineer has to manually trace the lineage to find the root cause.

### The Solution

**ML Sentinel** is a Production ML agent that lives inside DataHub's context graph. It uses DataHub's MCP Server to traverse the full ML lineage path — `training data → features → models → deployments` — and runs a fleet of five detectors that surface silent problems **with their root cause already traced**:

1. **Model Drift** — detects prediction distribution shifts using Population Stability Index (PSI), following lineage from deployment → model → training data to pull both distributions from the catalog
2. **Broken Lineage** — finds feature tables with missing raw-dataset provenance (the link from source data to features is broken, blocking impact analysis)
3. **Stale Features** — flags feature tables past their freshness SLA (reads `freshness_sla_hours` from DataHub structured properties)
4. **Schema Mismatch** — compares training vs. serving schemas to find dropped or re-typed features that silently degrade model quality
5. **Training-Serving Skew** — computes PSI between training and serving feature distributions to catch preprocessing divergences

### The Unique Angle

> Unlike existing ML monitoring tools (Evidently, NannyML, WhyLabs), ML Sentinel uses DataHub's end-to-end ML lineage graph to automatically trace silent failures across the entire ML pipeline — from training data to production predictions — and writes findings back to DataHub so the next person or agent inherits the knowledge.

The critical differentiator is **write-back**: when ML Sentinel finds a problem, it doesn't just raise an alert in a separate dashboard that nobody looks at. It writes the finding back to DataHub as:

- **Assertions** (with failing run events) — visible in DataHub's data-quality / observability views
- **Tags** (`ml-sentinel:model-drift`, `ml-sentinel:stale-feature`, etc.) — visible in search and the DataHub UI
- **Documents** — full narrative reports (what happened, the evidence, the remediation) saved to DataHub's knowledge base

This means the **next person or agent** that queries the DataHub catalog sees the findings inline. The knowledge is inherited, not lost in a separate monitoring silo.

### How It Uses DataHub

**Reading the context graph (via MCP Server):**
- `search` — discover ML models, feature tables, and deployments
- `get_lineage` — traverse upstream/downstream lineage (up to 3 hops) to trace root causes
- `get_entities` — fetch full metadata for specific URNs (schemas, properties, metrics)
- `list_schema_fields` — inspect dataset schemas for mismatch detection

**Writing back to the context graph (via MCP Server mutation tools + SDK):**
- `add_tags` — tag affected entities with `ml-sentinel:<problem-type>`
- `save_document` — save narrative reports to DataHub's knowledge base
- Assertion entities + failing run events (via GraphQL / SDK) — so findings appear in DataHub's observability views alongside other assertions

This goes beyond reading metadata — ML Sentinel **contributes back to the graph**, which is exactly what the challenge asks for.

### Architecture

The agent has three layers:
1. **Discovery** — searches DataHub for all ML entities
2. **Detector Fleet** — five independent detectors, each focused on one silent-problem class, each using DataHub lineage to trace root causes
3. **Reporter** — writes findings back to DataHub as assertions, tags, and documents

The agent is transport-agnostic: it works with the DataHub MCP Server (recommended), the DataHub Python SDK / GraphQL API, or a bundled mock estate for the demo.

### Demo

The demo runs against a bundled sample ML estate — a fraud-detection pipeline (`raw_transactions → user_txn_features → fraud_model_v3 → fraud_detector_prod`) with five planted silent problems. Running `ml-sentinel scan --mock` detects all five in under a second and shows the write-back to DataHub.

### Tech Stack
- Python 3.9+
- DataHub MCP Server (`@acryldata/mcp-server-datahub`) via the `mcp` Python SDK
- DataHub Python SDK (`acryl-datahub`) for GraphQL / assertion emission
- `rich` for CLI rendering
- `pytest` for testing (17 tests, all passing)
- Apache 2.0 license

### Setup (under 5 commands)
```bash
git clone https://github.com/0xConsole/ml-sentinel.git && cd ml-sentinel
pip install -e .
ml-sentinel scan --mock          # detect all 5 silent problems
ml-sentinel lineage --mock        # view the lineage graph
ml-sentinel estate --mock         # list the ML estate
```

### Real-World Usefulness

ML teams at companies running production ML (fraud detection, recommendation systems, ad targeting) face these exact silent-failure scenarios daily. Feature pipelines break, schemas drift, training-serving skew creeps in — and the model silently degrades. ML Sentinel catches these problems early, with root causes already traced through DataHub's lineage, and makes the findings permanent in the catalog so the whole team benefits.

A data/ML platform team would deploy ML Sentinel as a scheduled job (cron, Airflow, or a DataHub Action) that scans the estate periodically and writes findings back. The next on-call engineer, or the next AI agent that queries DataHub, inherits the knowledge.

### Links
- **GitHub:** https://github.com/0xConsole/ml-sentinel
- **Demo video:** [linked on Devpost]
- **Live demo:** Run locally with `pip install -e . && ml-sentinel scan --mock` (no DataHub instance needed)
