"""Tests for FastAPI serving app."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture
def mock_stream_processor():
    """Mock StreamProcessor so app startup and endpoints work without model artifacts."""
    mock = MagicMock()
    mock.health_check.return_value = {
        "status": "healthy",
        "models_loaded": True,
        "avg_latency_ms": 0.0,
        "total_processed": 0,
    }
    mock.get_metrics.return_value = {
        "total_processed": 0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "risk_tier_counts": {"LOW": 0, "MEDIUM": 0, "HIGH": 0},
        "alert_count": 0,
        "uptime_seconds": 0.0,
    }
    mock.process_transaction.return_value = {
        "transaction_id": "test-123",
        "fraud_score": 0.25,
        "risk_tier": "LOW",
        "iso_score": 0.2,
        "ae_score": 0.3,
        "xgb_score": 0.2,
        "should_alert": False,
        "processing_time_ms": 5.0,
    }
    return mock


@pytest.fixture
def client(mock_stream_processor):
    """TestClient with StreamProcessor mocked so startup succeeds."""
    with patch("src.serving.app.StreamProcessor", return_value=mock_stream_processor):
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    """GET /health returns 200 and status=healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_endpoint(client, mock_stream_processor):
    """POST /predict with valid transaction returns 200 and has fraud_score."""
    body = {
        "trans_date_trans_time": "2020-06-21 12:14:25",
        "cc_num": 2291163933867244,
        "amt": 50.0,
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    mock_stream_processor.process_transaction.assert_called_once()


def test_predict_invalid_input(client):
    """POST /predict with empty body returns 422."""
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_metrics_endpoint(client):
    """GET /metrics returns 200."""
    response = client.get("/metrics")
    assert response.status_code == 200
