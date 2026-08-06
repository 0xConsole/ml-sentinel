"""Schema mismatch detector.

Detects cases where the **serving schema** has diverged from the **training
schema** for a model. This is a classic source of silent failures: the model
was trained on a set of features, but the serving pipeline silently dropped,
renamed, or re-typed one of them. The model still produces predictions (often
because a default value is imputed), but the predictions are wrong.

The detector:

1. Finds the model's training dataset and serving dataset from its properties
   (``training_data_urn`` / ``serving_data_urn``) or from lineage.
2. Compares the schema fields of the two datasets.
3. Flags any field that is present in training but missing at serving, or
   whose type changed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import Entity, EntityType, Finding, FindingType, SchemaField, Severity

if TYPE_CHECKING:
    from ..datahub_client import DataHubClient


class SchemaMismatchDetector:
    """Flag training/serving schema divergences."""

    name = "schema_mismatch"

    def detect(self, client: "DataHubClient", entity: Entity) -> list[Finding]:
        if entity.type != EntityType.ML_MODEL:
            return []

        training_urn = entity.properties.get("training_data_urn")
        serving_urn = entity.properties.get("serving_data_urn")

        if not training_urn or not serving_urn:
            return []

        training_entity = client.get_entity(training_urn)
        serving_entity = client.get_entity(serving_urn)
        if not training_entity or not serving_entity:
            return []

        train_fields = {f.name: f for f in training_entity.schema_fields}
        serve_fields = {f.name: f for f in serving_entity.schema_fields}

        findings: list[Finding] = []

        # Fields present in training but missing at serving.
        missing = sorted(set(train_fields) - set(serve_fields))
        for fname in missing:
            train_field = train_fields[fname]
            findings.append(
                Finding(
                    type=FindingType.SCHEMA_MISMATCH,
                    severity=Severity.CRITICAL,
                    title=f"Missing serving field '{fname}' for model {entity.name}",
                    description=(
                        f"Feature '{fname}' (type {train_field.type}) is present "
                        f"in the training schema of '{entity.name}' but is MISSING "
                        f"from the serving schema. The model is receiving a "
                        f"default/null value for this feature at serving time, "
                        f"which silently degrades prediction quality."
                    ),
                    entity_urn=entity.urn,
                    evidence={
                        "entity_name": entity.name,
                        "field": fname,
                        "training_type": train_field.type,
                        "serving_type": None,
                        "training_dataset": training_urn,
                        "serving_dataset": serving_urn,
                        "issue": "missing_in_serving",
                    },
                    remediation=(
                        f"Restore the '{fname}' feature in the serving pipeline, "
                        f"or retrain the model without it if it's been "
                        f"intentionally dropped."
                    ),
                )
            )

        # Fields present in both but with a different type.
        common = set(train_fields) & set(serve_fields)
        for fname in sorted(common):
            tf = train_fields[fname]
            sf = serve_fields[fname]
            if tf.type != sf.type:
                findings.append(
                    Finding(
                        type=FindingType.SCHEMA_MISMATCH,
                        severity=Severity.CRITICAL,
                        title=f"Type changed for field '{fname}' on {entity.name}",
                        description=(
                            f"Field '{fname}' changed type from {tf.type} "
                            f"(training) to {sf.type} (serving). The model may "
                            f"produce incorrect predictions due to implicit "
                            f"casting."
                        ),
                        entity_urn=entity.urn,
                        evidence={
                            "entity_name": entity.name,
                            "field": fname,
                            "training_type": tf.type,
                            "serving_type": sf.type,
                            "issue": "type_mismatch",
                        },
                        remediation=(
                            f"Align the serving schema for '{fname}' with the "
                            f"training type ({tf.type})."
                        ),
                    )
                )

        # Extra fields at serving (not in training) — lower severity.
        extra = sorted(set(serve_fields) - set(train_fields))
        for fname in extra:
            findings.append(
                Finding(
                    type=FindingType.SCHEMA_MISMATCH,
                    severity=Severity.INFO,
                    title=f"Unexpected serving field '{fname}' for {entity.name}",
                    description=(
                        f"Field '{fname}' is present in the serving schema but "
                        f"was not used during training of '{entity.name}'. "
                        f"It's likely harmless but indicates schema drift."
                    ),
                    entity_urn=entity.urn,
                    evidence={
                        "entity_name": entity.name,
                        "field": fname,
                        "serving_type": serve_fields[fname].type,
                        "issue": "extra_in_serving",
                    },
                    remediation="No action needed unless this field should be a model input.",
                )
            )

        return findings
