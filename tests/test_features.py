import pandas as pd

from src.features import TARGET_COL, add_proxy_risk_target, build_customer_features, feature_columns


def test_customer_feature_engineering_and_target():
    tx = pd.DataFrame(
        {
            "TransactionId": ["t1", "t2", "t3", "t4"],
            "CustomerId": ["c1", "c1", "c2", "c2"],
            "ProductId": ["p1", "p2", "p1", "p1"],
            "ProductCategory": ["airtime", "data", "airtime", "airtime"],
            "ChannelId": ["web", "app", "app", "app"],
            "ProviderId": ["a", "b", "a", "a"],
            "Amount": [100.0, -20.0, 500.0, 700.0],
            "Value": [100, 20, 500, 700],
            "PricingStrategy": [1, 1, 2, 2],
            "FraudResult": [0, 1, 0, 0],
            "TransactionStartTime": pd.to_datetime(
                ["2024-01-01", "2024-01-03", "2024-02-01", "2024-02-02"], utc=True
            ),
        }
    )
    features = build_customer_features(tx)
    labeled, metadata = add_proxy_risk_target(features)

    assert len(labeled) == 2
    assert TARGET_COL in labeled.columns
    assert labeled.loc[labeled["CustomerId"] == "c1", TARGET_COL].iloc[0] == 1
    assert metadata["target_rate"] > 0
    assert "fraud_txn_count" not in feature_columns(labeled)
