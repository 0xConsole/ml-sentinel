"""Unit tests for ML Sentinel detectors and agent.

These tests exercise the detector fleet against the bundled mock estate and
verify that every planted silent problem is detected.
"""

import pytest

from ml_sentinel.agent import MLSentinelAgent
from ml_sentinel.data.sample_estate import build_estate
from ml_sentinel.datahub_client import MockDataHubClient
from ml_sentinel.models import FindingType, Severity


@pytest.fixture
def mock_client():
    return MockDataHubClient()


@pytest.fixture
def agent(mock_client):
    return MLSentinelAgent(mock_client, write_back=True)


@pytest.fixture
def agent_no_write(mock_client):
    return MLSentinelAgent(mock_client, write_back=False)


class TestAgentScan:
    def test_scan_finds_all_five_problem_types(self, agent_no_write):
        result = agent_no_write.scan()
        types_found = {f.type for f in result.findings}
        assert FindingType.MODEL_DRIFT in types_found
        assert FindingType.BROKEN_LINEAGE in types_found
        assert FindingType.STALE_FEATURE in types_found
        assert FindingType.SCHEMA_MISMATCH in types_found
        assert FindingType.TRAINING_SERVING_SKEW in types_found

    def test_scan_finds_exactly_five_findings(self, agent_no_write):
        """One finding per planted problem (deduped by assertion URN)."""
        result = agent_no_write.scan()
        assert len(result.findings) == 5

    def test_all_findings_are_critical(self, agent_no_write):
        """All five planted problems are critical-severity."""
        result = agent_no_write.scan()
        assert result.critical_count == 5
        assert result.warn_count == 0

    def test_scan_scans_three_entities(self, agent_no_write):
        """Model, feature table, and deployment = 3 entities."""
        result = agent_no_write.scan()
        assert result.entities_scanned == 3

    def test_scan_writes_back_to_datahub(self, agent):
        result = agent.scan()
        assert result.assertions_written == 5
        assert result.tags_written == 5
        assert result.documents_written >= 1

    def test_scan_duration_is_positive(self, agent_no_write):
        result = agent_no_write.scan()
        assert result.duration_seconds >= 0


class TestModelDriftDetector:
    def test_detects_prediction_drift(self, agent_no_write):
        result = agent_no_write.scan()
        drift = [f for f in result.findings if f.type == FindingType.MODEL_DRIFT]
        assert len(drift) >= 1
        assert drift[0].severity == Severity.CRITICAL
        assert "PSI" in drift[0].description


class TestBrokenLineageDetector:
    def test_detects_missing_dataset_upstream(self, agent_no_write):
        result = agent_no_write.scan()
        broken = [f for f in result.findings if f.type == FindingType.BROKEN_LINEAGE]
        assert len(broken) == 1
        assert "raw-dataset lineage" in broken[0].title
        assert broken[0].evidence["has_dataset_upstream"] is False
        assert broken[0].evidence["has_process_upstream"] is True


class TestStaleFeatureDetector:
    def test_detects_stale_feature_table(self, agent_no_write):
        result = agent_no_write.scan()
        stale = [f for f in result.findings if f.type == FindingType.STALE_FEATURE]
        assert len(stale) == 1
        assert "225" in stale[0].description  # ~225 hours stale
        assert stale[0].evidence["sla_breach_ratio"] > 3  # critical


class TestSchemaMismatchDetector:
    def test_detects_missing_merchant_category(self, agent_no_write):
        result = agent_no_write.scan()
        mismatch = [f for f in result.findings if f.type == FindingType.SCHEMA_MISMATCH]
        assert len(mismatch) == 1
        assert "merchant_category" in mismatch[0].title
        assert mismatch[0].evidence["issue"] == "missing_in_serving"


class TestTrainingServingSkewDetector:
    def test_detects_tx_amount_skew(self, agent_no_write):
        result = agent_no_write.scan()
        skew = [f for f in result.findings if f.type == FindingType.TRAINING_SERVING_SKEW]
        assert len(skew) == 1
        assert "tx_amount" in skew[0].title
        assert skew[0].evidence["psi"] > 0.25  # critical threshold


class TestReporter:
    def test_reporter_creates_assertions(self, mock_client):
        from ml_sentinel.reporter import Reporter

        agent = MLSentinelAgent(mock_client, write_back=False)
        result = agent.scan()
        reporter = Reporter(mock_client)
        stats = reporter.report(result.findings)
        assert stats["assertions"] == 5
        assert stats["tags"] == 5
        assert stats["documents"] >= 1

    def test_reporter_writes_markdown_documents(self, mock_client):
        from ml_sentinel.reporter import Reporter

        agent = MLSentinelAgent(mock_client, write_back=False)
        result = agent.scan()
        reporter = Reporter(mock_client)
        reporter.report(result.findings)
        docs = mock_client.written_documents
        assert len(docs) >= 1
        content = docs[0]["content"]
        assert "ML Sentinel Report" in content
        assert "Remediation" in content

    def test_tags_are_namespaced(self, mock_client):
        from ml_sentinel.reporter import Reporter

        agent = MLSentinelAgent(mock_client, write_back=False)
        result = agent.scan()
        reporter = Reporter(mock_client)
        reporter.report(result.findings)
        for urn, tags in mock_client.written_tags.items():
            for tag in tags:
                assert tag.startswith("ml-sentinel:")


class TestPSI:
    def test_psi_stable_distributions(self):
        from ml_sentinel.models import FeatureDistribution

        a = FeatureDistribution("x", mean=100, std=10, min=50, max=150, n_samples=1000)
        b = FeatureDistribution("x", mean=100, std=10, min=50, max=150, n_samples=1000)
        psi = a.psi_against(b)
        assert psi is not None
        assert psi < 0.1  # stable

    def test_psi_shifted_distributions(self):
        from ml_sentinel.models import FeatureDistribution

        a = FeatureDistribution("x", mean=84, std=45, min=1, max=990, n_samples=1000)
        b = FeatureDistribution("x", mean=142, std=68, min=1, max=1500, n_samples=1000)
        psi = a.psi_against(b)
        assert psi is not None
        assert psi > 0.1  # drifted

    def test_psi_categorical(self):
        from ml_sentinel.models import FeatureDistribution

        a = FeatureDistribution("cat", categories={"a": 0.5, "b": 0.5})
        b = FeatureDistribution("cat", categories={"a": 0.3, "b": 0.7})
        psi = a.psi_against(b)
        assert psi is not None
        assert psi > 0  # some shift


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
