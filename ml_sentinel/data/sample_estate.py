"""Bundled sample ML estate for the demo.

This module builds an in-memory DataHub-like estate for a **fraud-detection
pipeline** so ML Sentinel can run end-to-end without a live DataHub instance.

The estate contains the full lineage path the challenge asks for:

    raw_transactions  (Snowflake dataset)
         |
         v
    user_txn_features  (feature table)
         |
         v
    fraud_model_v3  (ML model)
         |
         v
    fraud_detector_prod  (ML model deployment)

We deliberately plant FIVE silent problems — one per detector — so the demo
shows the full fleet firing:

1. **Model drift**        — fraud_model_v3's prediction distribution shifted
                            (PSI = 0.31 > 0.25 threshold).
2. **Broken lineage**     — the edge raw_transactions -> user_txn_features is
                            missing (the feature job stopped writing upstream).
3. **Stale feature**      — user_txn_features.last_modified is 9 days ago but
                            its SLA is 24h.
4. **Schema mismatch**    — the serving schema dropped the ``merchant_category``
                            column that the training schema expects.
5. **Training-serving skew** — the ``tx_amount`` feature has a training mean of
                            $84 but a serving mean of $142 (PSI 0.27).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from ..models import (
    Entity,
    EntityType,
    FeatureDistribution,
    LineageEdge,
    ModelMetrics,
    SchemaField,
)

NOW = datetime(2026, 8, 6, 12, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# URNs (DataHub-style)
# --------------------------------------------------------------------------- #

URN_RAW_TXNS = "urn:li:dataset:(urn:li:dataPlatform:snowflake,fraud.raw_transactions,PROD)"
URN_FEATURE_TABLE = "urn:li:mlFeatureTable:(urn:li:dataPlatform:feast,fraud.user_txn_features,PROD)"
URN_FEATURE_AMOUNT = "urn:li:mlFeature:(urn:li:dataPlatform:feast,fraud.user_txn_features.tx_amount,PROD)"
URN_FEATURE_MERCHANT = "urn:li:mlFeature:(urn:li:dataPlatform:feast,fraud.user_txn_features.merchant_category,PROD)"
URN_MODEL = "urn:li:mlModel:(urn:li:dataPlatform:feast,fraud_model_v3,PROD)"
URN_DEPLOYMENT = "urn:li:mlModelDeployment:(urn:li:dataPlatform:sagemaker,fraud_detector_prod,PROD)"
URN_FEATURE_JOB = "urn:li:dataProcess:(urn:li:dataPlatform:airflow,compute_user_features,PROD)"


@dataclass
class Estate:
    """The full in-memory estate: entities + lineage + metrics."""

    entities: list[Entity] = field(default_factory=list)
    edges: list[LineageEdge] = field(default_factory=dict)  # type: ignore[assignment]
    metrics: dict[str, ModelMetrics] = field(default_factory=dict)


def build_estate() -> Estate:
    """Build the sample estate with the five planted silent problems."""
    estate = Estate()

    # ---- Entities -------------------------------------------------------- #

    # 1. Raw transactions dataset (upstream of everything)
    raw_txns = Entity(
        urn=URN_RAW_TXNS,
        name="fraud.raw_transactions",
        type=EntityType.DATASET,
        platform="snowflake",
        last_modified=NOW - timedelta(hours=2),
        schema_fields=[
            SchemaField("transaction_id", "STRING", False),
            SchemaField("user_id", "STRING", False),
            SchemaField("amount", "DOUBLE", False),
            SchemaField("merchant_category", "STRING", True),
            SchemaField("txn_timestamp", "TIMESTAMP", False),
            SchemaField("is_fraud", "BOOLEAN", True),
        ],
        properties={"freshness_sla_hours": 6},
    )

    # 2. Feature table — STALE (last_modified 9 days ago, SLA is 24h)  [Problem 3]
    feature_table = Entity(
        urn=URN_FEATURE_TABLE,
        name="fraud.user_txn_features",
        type=EntityType.ML_FEATURE_TABLE,
        platform="feast",
        last_modified=NOW - timedelta(days=9),  # <-- stale: 9 days >> 24h SLA
        schema_fields=[
            # Training schema (what the model was trained on)
            SchemaField("user_id", "STRING", False),
            SchemaField("tx_amount", "DOUBLE", False),
            SchemaField("txn_count_24h", "INTEGER", True),
            SchemaField("merchant_category", "STRING", True),  # <-- dropped at serve
            SchemaField("avg_txn_amount_7d", "DOUBLE", True),
        ],
        properties={"freshness_sla_hours": 24},
    )

    # 3. Serving dataset — SCHEMA MISMATCH (missing merchant_category)  [Problem 4]
    serving_table = Entity(
        urn="urn:li:dataset:(urn:li:dataPlatform:snowflake,fraud.serving_features,PROD)",
        name="fraud.serving_features",
        type=EntityType.DATASET,
        platform="snowflake",
        last_modified=NOW - timedelta(minutes=30),
        schema_fields=[
            SchemaField("user_id", "STRING", False),
            SchemaField("tx_amount", "DOUBLE", False),
            SchemaField("txn_count_24h", "INTEGER", True),
            # merchant_category is MISSING here  <--
            SchemaField("avg_txn_amount_7d", "DOUBLE", True),
        ],
    )

    # 4. ML model
    model = Entity(
        urn=URN_MODEL,
        name="fraud_model_v3",
        type=EntityType.ML_MODEL,
        platform="feast",
        last_modified=NOW - timedelta(days=40),
        schema_fields=[
            SchemaField("tx_amount", "DOUBLE", False),
            SchemaField("txn_count_24h", "INTEGER", True),
            SchemaField("merchant_category", "STRING", True),
            SchemaField("avg_txn_amount_7d", "DOUBLE", True),
        ],
        properties={
            "training_data_urn": URN_FEATURE_TABLE,
            "serving_data_urn": "urn:li:dataset:(urn:li:dataPlatform:snowflake,fraud.serving_features,PROD)",
            "version": "v3",
            "accuracy": 0.94,
        },
    )

    # 5. Deployment
    deployment = Entity(
        urn=URN_DEPLOYMENT,
        name="fraud_detector_prod",
        type=EntityType.ML_MODEL_DEPLOYMENT,
        platform="sagemaker",
        last_modified=NOW - timedelta(days=40),
        properties={
            "model_urn": URN_MODEL,
            "endpoint": "fraud-detector-prod",
            "status": "HEALTHY",  # <-- looks healthy, but it isn't
        },
    )

    # 6. The feature-compute Airflow job
    feature_job = Entity(
        urn=URN_FEATURE_JOB,
        name="compute_user_features",
        type=EntityType.DATA_PROCESS,
        platform="airflow",
        last_modified=NOW - timedelta(days=9),
        properties={"schedule": "0 * * * *", "status": "FAILED"},
    )

    estate.entities = [
        raw_txns,
        feature_table,
        serving_table,
        model,
        deployment,
        feature_job,
    ]

    # ---- Lineage --------------------------------------------------------- #
    # Full path: raw -> feature_table -> model -> deployment
    #   PLUS feature_table -> serving_table (the serving copy)
    #
    # PROBLEM 2 (broken lineage): the edge raw_txns -> feature_table is
    # MISSING on purpose. The detector should flag that the feature table
    # claims an upstream it can no longer trace.
    estate.edges = [
        # raw_txns -> feature_job  (the job reads raw transactions)
        LineageEdge(
            URN_RAW_TXNS,
            URN_FEATURE_JOB,
            created_at=NOW - timedelta(days=9),
            transformation="SELECT * FROM fraud.raw_transactions",
        ),
        # feature_job -> feature_table  (the job writes the feature table)
        LineageEdge(
            URN_FEATURE_JOB,
            URN_FEATURE_TABLE,
            created_at=NOW - timedelta(days=9),
            transformation="INSERT INTO fraud.user_txn_features ...",
        ),
        # NOTE: raw_txns -> feature_table edge is intentionally absent
        # (this is the "broken lineage" the detector must find).
        # feature_table -> model
        LineageEdge(
            URN_FEATURE_TABLE,
            URN_MODEL,
            created_at=NOW - timedelta(days=40),
            transformation="training_data",
        ),
        # serving_table -> model  (serving features feed the deployed model)
        LineageEdge(
            "urn:li:dataset:(urn:li:dataPlatform:snowflake,fraud.serving_features,PROD)",
            URN_MODEL,
            created_at=NOW - timedelta(days=40),
            transformation="serving_data",
        ),
        # model -> deployment
        LineageEdge(
            URN_MODEL,
            URN_DEPLOYMENT,
            created_at=NOW - timedelta(days=40),
            transformation="deploy",
        ),
    ]

    # ---- Metrics (with planted drift + skew) ----------------------------- #

    # Problem 1: MODEL DRIFT — prediction distribution shifted
    # Problem 5: TRAINING-SERVING SKEW — tx_amount mean shifted $84 -> $142
    estate.metrics = {
        URN_MODEL: ModelMetrics(
            model_urn=URN_MODEL,
            accuracy=0.94,
            prediction_drift_score=0.31,  # PSI > 0.25 = significant
            data_quality_score=0.72,
            last_serve_time=NOW - timedelta(minutes=5),
            training_distributions=[
                # Training: tx_amount mean ~$84
                FeatureDistribution(
                    feature_name="tx_amount",
                    mean=84.2,
                    std=45.0,
                    null_rate=0.0,
                    min=1.0,
                    max=990.0,
                    p95=180.0,
                    n_samples=1_200_000,
                ),
                FeatureDistribution(
                    feature_name="txn_count_24h",
                    mean=3.1,
                    std=2.0,
                    null_rate=0.0,
                    min=0,
                    max=25,
                    p95=8,
                    n_samples=1_200_000,
                ),
                FeatureDistribution(
                    feature_name="merchant_category",
                    categories={
                        "retail": 0.42,
                        "food": 0.23,
                        "travel": 0.15,
                        "digital": 0.12,
                        "other": 0.08,
                    },
                    null_rate=0.0,
                    n_samples=1_200_000,
                ),
            ],
            serving_distributions=[
                # Serving: tx_amount mean ~$142  <-- SKEW (PSI ~0.27)
                FeatureDistribution(
                    feature_name="tx_amount",
                    mean=142.7,  # <-- shifted up
                    std=68.0,
                    null_rate=0.0,
                    min=1.0,
                    max=1500.0,
                    p95=310.0,
                    n_samples=85_000,
                ),
                FeatureDistribution(
                    feature_name="txn_count_24h",
                    mean=3.4,
                    std=2.1,
                    null_rate=0.0,
                    min=0,
                    max=28,
                    p95=9,
                    n_samples=85_000,
                ),
                # merchant_category is missing at serve — handled by schema detector
                FeatureDistribution(
                    feature_name="merchant_category",
                    categories={
                        "retail": 0.45,
                        "food": 0.20,
                        "travel": 0.09,   # <-- travel dropped
                        "digital": 0.19,   # <-- digital rose
                        "other": 0.07,
                    },
                    null_rate=0.0,
                    n_samples=85_000,
                ),
            ],
            prediction_distribution=FeatureDistribution(
                feature_name="fraud_probability",
                mean=0.18,         # <-- up from training baseline ~0.06
                std=0.21,
                null_rate=0.0,
                min=0.0,
                max=1.0,
                p95=0.62,
                n_samples=85_000,
            ),
        ),
    }

    return estate
