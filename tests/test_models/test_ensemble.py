"""Tests for EnsembleScorer."""

import pytest

from src.models.ensemble import EnsembleScorer


@pytest.fixture
def ensemble():
    """EnsembleScorer with default thresholds (low=0.3, high=0.7)."""
    e = EnsembleScorer()
    e._threshold_low = 0.3
    e._threshold_high = 0.7
    e._optimal_threshold = 0.5
    e._iso_weight = 0.4
    e._ae_weight = 0.6
    e._fitted_ = True
    return e


def test_score_in_range(ensemble):
    """fraud_score should be between 0 and 1."""
    result = ensemble.score_transaction(0.0, 0.0)
    assert 0 <= result["fraud_score"] <= 1
    result = ensemble.score_transaction(1.0, 1.0)
    assert 0 <= result["fraud_score"] <= 1
    result = ensemble.score_transaction(0.5, 0.5)
    assert 0 <= result["fraud_score"] <= 1


def test_risk_tier_low(ensemble):
    """score < 0.3 should be LOW."""
    result = ensemble.score_transaction(0.1, 0.1)
    assert result["risk_tier"] == "LOW"
    result = ensemble.score_transaction(0.0, 0.0)
    assert result["risk_tier"] == "LOW"


def test_risk_tier_medium(ensemble):
    """score 0.3-0.7 should be MEDIUM."""
    result = ensemble.score_transaction(0.4, 0.4)
    assert result["risk_tier"] == "MEDIUM"
    result = ensemble.score_transaction(0.5, 0.5)
    assert result["risk_tier"] == "MEDIUM"


def test_risk_tier_high(ensemble):
    """score > 0.7 should be HIGH."""
    result = ensemble.score_transaction(0.9, 0.9)
    assert result["risk_tier"] == "HIGH"
    result = ensemble.score_transaction(1.0, 0.8)
    assert result["risk_tier"] == "HIGH"


def test_should_alert_high_risk(ensemble):
    """HIGH tier should have should_alert=True (score >= optimal_threshold)."""
    result = ensemble.score_transaction(0.9, 0.9)
    assert result["risk_tier"] == "HIGH"
    assert result["should_alert"] is True


def test_should_not_alert_low_risk(ensemble):
    """LOW tier should have should_alert=False."""
    result = ensemble.score_transaction(0.1, 0.1)
    assert result["risk_tier"] == "LOW"
    assert result["should_alert"] is False
