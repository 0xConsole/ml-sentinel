# Demo Video Script — ML Sentinel (3 minutes)

## Overview
- **Duration:** ~2:45 (under 3:00 limit)
- **Format:** Screen recording with voiceover
- **Tool:** `asciinema` or terminal screen recording + voiceover

---

## Script

### [0:00 — 0:15] Intro: The Problem

**On screen:** Title slide — "ML Sentinel — Production ML Silent-Problem Hunter for DataHub"

**Voiceover:**
> "ML systems don't fail loudly — they fail silently. A feature pipeline stops
> updating, a schema drops a column, a training-serving skew creeps in, and the
> model keeps serving predictions. Nobody notices until revenue leaks or fraud
> spikes."

### [0:15 — 0:35] The Unique Angle

**On screen:** Comparison slide — "Existing tools (Evidently, NannyML, WhyLabs) detect drift. ML Sentinel traces the root cause through DataHub's lineage graph."

**Voiceover:**
> "Existing ML monitoring tools detect statistical problems, but they operate
> on isolated metrics stores. They don't see the full pipeline. ML Sentinel
> lives inside DataHub's context graph, uses the MCP Server to traverse
> end-to-end ML lineage, and writes findings back to DataHub so the next agent
> inherits the knowledge."

### [0:35 — 0:55] Show the ML Estate

**On screen:** Terminal — run `ml-sentinel estate --mock`

```bash
$ ml-sentinel estate --mock
```

**Voiceover:**
> "Here's our sample ML estate — a fraud-detection pipeline. We have a raw
> transactions dataset, a feature table, a model, a deployment, and the
> feature-compute job. Five silent problems are planted in this estate."

### [0:55 — 1:15] Show the Lineage Graph

**On screen:** Terminal — run `ml-sentinel lineage --mock`

```bash
$ ml-sentinel lineage --mock
```

**Voiceover:**
> "Here's the lineage graph. Notice the path from raw transactions through the
> feature job to the feature table, then to the model and deployment. But
> notice something — the direct edge from raw_transactions to the feature table
> is missing. That's a broken lineage link, and one of the problems we'll
> detect."

### [1:15 — 1:55] Run the Scan

**On screen:** Terminal — run `ml-sentinel scan --mock`

```bash
$ ml-sentinel scan --mock
```

**Voiceover:**
> "Now let's run the agent. ML Sentinel searches DataHub for all ML entities,
> runs five detectors against each one, and writes findings back to the
> catalog. In under a second, it found five critical problems."

**On screen:** The output table showing all 5 findings:

```
  Entities scanned           3
  Findings                   5
  Severity                   5 critical · 0 warn · 0 info

  # │ Severity │ Type                  │ Entity                  │ Finding
  1 │ CRITICAL │ model_drift           │ fraud_model_v3          │ Significant prediction drift (PSI=0.31)
  2 │ CRITICAL │ schema_mismatch       │ fraud_model_v3          │ Missing serving field 'merchant_category'
  3 │ CRITICAL │ training_serving_skew │ fraud_model_v3          │ Training-serving skew on 'tx_amount' (PSI=0.98)
  4 │ CRITICAL │ broken_lineage        │ fraud.user_txn_features │ Missing raw-dataset lineage
  5 │ CRITICAL │ stale_feature         │ fraud.user_txn_features │ Stale (225h > 24h SLA, 9.4x breach)
```

**Voiceover:**
> "Let's walk through what it found. Model drift — the prediction distribution
> shifted with a PSI of 0.31, past the critical threshold. Schema mismatch —
> the serving schema dropped the merchant_category feature that the model was
> trained on. Training-serving skew — the tx_amount feature shifted from a
> mean of $84 to $142. Broken lineage — the feature table lost its raw-dataset
> provenance. And stale features — the feature table hasn't been updated in
> 225 hours, 9 times its SLA."

### [1:55 — 2:25] Show the Write-Back

**On screen:** Scroll down to the write-back panel:

```
╭─ Write-Back ────────────────────────────────────────────────────────╮
│ Wrote back to DataHub:                                              │
│   • 5 assertions created                                            │
│   • 5 tags applied                                                  │
│   • 2 documents saved                                               │
│ In mock mode these are recorded in-memory. With a live DataHub they  │
│ become real catalog objects visible to every user and agent.         │
╰─────────────────────────────────────────────────────────────────────╯
```

**Voiceover:**
> "Critically, ML Sentinel doesn't just raise alerts — it writes the findings
> back to DataHub. Five assertions with failing run events, five tags like
> 'ml-sentinel:stale-feature' applied to the affected entities, and two
> narrative documents saved to the knowledge base. The next person or agent
> that queries the catalog sees these inline."

### [2:25 — 2:45] JSON Output + Closing

**On screen:** Terminal — run `ml-sentinel scan --mock --json | jq '.findings[] | {type, severity, title}'`

```bash
$ ml-sentinel scan --mock --json | jq '.findings[] | {type, severity, title}'
```

**Voiceover:**
> "The agent also outputs JSON for piping to other tools or agents. Every
> finding includes the entity URN, evidence, remediation steps, and a stable
> assertion URN so re-runs update existing assertions instead of duplicating
> them."

**On screen:** Final slide — "ML Sentinel — github.com/0xConsole/ml-sentinel — Apache 2.0"

**Voiceover:**
> "ML Sentinel — open source, Apache 2.0, on GitHub. Catch silent ML failures
> before they cost money. Thanks for watching."

---

## Recording Tips

1. Use a dark terminal theme for readability
2. Increase font size (18pt+) for screen recording
3. Record at 1080p
4. Practice the voiceover 2-3 times before recording
5. Keep each section tight — the script is ~2:45, leaving 15s buffer

## Alternative: Asciinema GIF

If a video is hard to produce, a GIF of the terminal output works:

```bash
# Install asciinema
pip install asciinema

# Record
asciinema rec demo.cast
# (run the commands above, then exit)

# Convert to GIF
pip install asciinema-to-gif
asciinema-to-gif demo.cast demo.gif
```
