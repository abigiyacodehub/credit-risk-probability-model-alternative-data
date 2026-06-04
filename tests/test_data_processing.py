import pandas as pd

from src.data_processing import (
    TARGET_COL,
    aggregate_customer_features,
    calculate_woe_iv,
    create_processed_dataset,
    extract_datetime_features,
)


def sample_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionId": ["t1", "t2", "t3", "t4", "t5", "t6"],
            "CustomerId": ["c1", "c1", "c2", "c2", "c3", "c3"],
            "ProductCategory": [
                "airtime", "data", "airtime", "utility", "data", "data"
            ],
            "ChannelId": ["web", "app", "app", "web", "app", "ussd"],
            "ProviderId": ["p1", "p1", "p2", "p2", "p3", "p3"],
            "PricingStrategy": [1, 1, 2, 2, 4, 4],
            "ProductId": ["a", "b", "a", "c", "b", "b"],
            "Amount": [100.0, 200.0, 500.0, 700.0, -50.0, 80.0],
            "Value": [100, 200, 500, 700, 50, 80],
            "FraudResult": [0, 0, 0, 0, 1, 0],
            "TransactionStartTime": [
                "2024-01-01T08:00:00Z",
                "2024-01-02T09:00:00Z",
                "2024-02-01T10:00:00Z",
                "2024-02-03T11:00:00Z",
                "2024-03-01T12:00:00Z",
                "2024-03-05T13:00:00Z",
            ],
        }
    )


def test_datetime_features_are_extracted():
    enriched = extract_datetime_features(sample_transactions())
    assert {
        "transaction_hour",
        "transaction_day",
        "transaction_month",
        "transaction_year",
    }.issubset(enriched.columns)
    assert enriched.loc[0, "transaction_hour"] == 8


def test_customer_aggregation_contains_required_features():
    features = aggregate_customer_features(sample_transactions())
    assert {
        "total_transaction_amount",
        "average_transaction_amount",
        "transaction_count",
        "std_transaction_amount",
    }.issubset(features.columns)
    assert len(features) == 3


def test_rfm_clustering_and_proxy_target_are_created():
    processed = create_processed_dataset(sample_transactions())
    assert TARGET_COL in processed.columns
    assert "rfm_cluster" in processed.columns
    assert set(processed[TARGET_COL].unique()).issubset({0, 1})


def test_woe_iv_returns_table_and_value():
    processed = create_processed_dataset(sample_transactions())
    table, iv = calculate_woe_iv(processed, "most_common_channel", TARGET_COL)
    assert {"woe", "iv"}.issubset(table.columns)
    assert isinstance(iv, float)
