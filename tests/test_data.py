import pytest

# Sample test data for testing the trading assistant
sample_data = {
    "prices": [100, 101, 102, 103, 104],
    "volumes": [10, 15, 10, 20, 25],
    "timestamps": [
        "2023-01-01T00:00:00Z",
        "2023-01-01T01:00:00Z",
        "2023-01-01T02:00:00Z",
        "2023-01-01T03:00:00Z",
        "2023-01-01T04:00:00Z"
    ]
}

def test_sample_data_structure():
    assert isinstance(sample_data, dict)
    assert "prices" in sample_data
    assert "volumes" in sample_data
    assert "timestamps" in sample_data

def test_prices_length():
    assert len(sample_data["prices"]) == 5

def test_volumes_length():
    assert len(sample_data["volumes"]) == 5

def test_timestamps_length():
    assert len(sample_data["timestamps"]) == 5