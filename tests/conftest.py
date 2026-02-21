import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_transaction():
    """Single raw transaction dict matching the Sparkov dataset schema."""
    return {
        "trans_date_trans_time": "2020-06-21 12:14:25",
        "cc_num": 2291163933867244,
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": 2.86,
        "gender": "M",
        "city": "Moravian Falls",
        "state": "NC",
        "lat": 36.0788,
        "long": -81.1781,
        "city_pop": 3495,
        "job": "Psychologist",
        "dob": "1988-03-09",
        "trans_num": "abc123",
        "merch_lat": 36.011293,
        "merch_long": -82.048315,
        "is_fraud": 0
    }


@pytest.fixture
def sample_dataframe(sample_transaction):
    """DataFrame with 5 sample transactions."""
    rows = []
    for i in range(5):
        row = sample_transaction.copy()
        row["amt"] = float(np.random.uniform(1, 500))
        row["trans_num"] = f"txn_{i}"
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def sample_features():
    """Numpy array of 25 features (matching feature store output)."""
    return np.random.randn(1, 25)
