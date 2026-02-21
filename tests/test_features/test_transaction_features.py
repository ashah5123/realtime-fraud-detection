"""Tests for TransactionFeatureEngineer."""

import pytest
import pandas as pd

from src.features.transaction_features import TransactionFeatureEngineer


def test_log_amount_positive(sample_dataframe):
    """log_amount should be > 0 for positive amt."""
    eng = TransactionFeatureEngineer()
    eng.fit(sample_dataframe)
    out = eng.transform(sample_dataframe)
    assert "log_amount" in out.columns
    assert (out["log_amount"] > 0).all()


def test_amount_decimal_extraction(sample_dataframe):
    """amount_decimal of 49.95 should be 0.95 (within float tolerance)."""
    df = sample_dataframe.copy()
    df["amt"] = 49.95
    eng = TransactionFeatureEngineer()
    eng.fit(df)
    out = eng.transform(df)
    assert "amount_decimal" in out.columns
    assert out["amount_decimal"].iloc[0] == pytest.approx(0.95, abs=0.01)


def test_is_round_amount(sample_dataframe):
    """50.00 should flag as round, 49.95 should not."""
    df = sample_dataframe.copy()
    df.loc[0, "amt"] = 50.0
    df.loc[1, "amt"] = 49.95
    eng = TransactionFeatureEngineer()
    eng.fit(df)
    out = eng.transform(df)
    assert "is_round_amount" in out.columns
    assert out["is_round_amount"].iloc[0] == 1
    assert out["is_round_amount"].iloc[1] == 0


def test_hour_extraction(sample_dataframe):
    """hour should be 0-23."""
    eng = TransactionFeatureEngineer()
    eng.fit(sample_dataframe)
    out = eng.transform(sample_dataframe)
    assert "hour_of_day" in out.columns
    assert out["hour_of_day"].min() >= 0
    assert out["hour_of_day"].max() <= 23


def test_is_weekend(sample_dataframe):
    """Saturday/Sunday should be 1, weekday should be 0."""
    # sample_dataframe has "2020-06-21 12:14:25" -> 2020-06-21 is Sunday (dayofweek=6)
    df = sample_dataframe.copy()
    df["trans_date_trans_time"] = "2020-06-21 12:14:25"  # Sunday
    eng = TransactionFeatureEngineer()
    eng.fit(df)
    out = eng.transform(df)
    assert "is_weekend" in out.columns
    assert (out["is_weekend"] == 1).all()
    # Weekday
    df2 = df.copy()
    df2["trans_date_trans_time"] = "2020-06-22 12:14:25"  # Monday
    out2 = eng.transform(df2)
    assert (out2["is_weekend"] == 0).all()


def test_distance_to_merchant(sample_dataframe):
    """distance should be >= 0."""
    eng = TransactionFeatureEngineer()
    eng.fit(sample_dataframe)
    out = eng.transform(sample_dataframe)
    assert "distance_to_merchant" in out.columns
    assert (out["distance_to_merchant"] >= 0).all()


def test_age_calculation(sample_dataframe):
    """age should be positive and reasonable (0-120)."""
    eng = TransactionFeatureEngineer()
    eng.fit(sample_dataframe)
    out = eng.transform(sample_dataframe)
    assert "age" in out.columns
    assert (out["age"] >= 0).all()
    assert (out["age"] <= 120).all()
