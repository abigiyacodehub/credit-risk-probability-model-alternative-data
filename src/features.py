from __future__ import annotations

import pandas as pd
from src.data_processing import (
    CUSTOMER_ID,
    TARGET_COL,
    aggregate_customer_features,
    compute_rfm_clusters,
)


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], utc=True)
    return df


def _mode_or_missing(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    return str(mode.iloc[0]) if len(mode) else "missing"


def build_customer_features(transactions: pd.DataFrame) -> pd.DataFrame:
    features = aggregate_customer_features(transactions)
    features = features.rename(
        columns={
            "total_transaction_amount": "total_amount",
            "average_transaction_amount": "avg_amount",
            "transaction_count": "tx_count",
            "std_transaction_amount": "std_value",
            "average_value": "avg_value_per_txn",
            "unique_product_categories": "unique_categories",
            "unique_channels": "unique_channels",
            "unique_providers": "unique_providers",
            "most_common_product_category": "most_common_category",
            "most_common_channel": "most_common_channel",
            "most_common_pricing_strategy": "most_common_pricing",
            "avg_transaction_hour": "avg_hour",
            "Recency": "recency_days",
            "Frequency": "frequency",
            "Monetary": "monetary",
        }
    )
    features["active_days"] = features["tx_count"]
    features["median_amount"] = features["avg_amount"]
    features["unique_products"] = features["unique_categories"]
    features["most_common_provider"] = "unknown"
    features["weekend_ratio"] = 0.0
    features["customer_age_days"] = 0
    features["tx_per_active_day"] = features["tx_count"] / features["active_days"].clip(lower=1)
    features["net_amount_per_txn"] = features["total_amount"] / features["tx_count"].clip(lower=1)
    return features


def add_proxy_risk_target(customer_features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create a proxy default-risk label from RFMS behavior and observed fraud.

    Xente does not contain true loan default outcomes. The proxy marks the
    weakest RFMS customer segment as high risk and always marks customers with
    observed fraud as high risk. This should be validated before production use.
    """
    df = customer_features.copy()
    rfm_ready = df.rename(
        columns={"recency_days": "Recency", "tx_count": "Frequency", "total_value": "Monetary"}
    )
    clustered, _, _ = compute_rfm_clusters(rfm_ready, n_clusters=min(4, max(2, len(df) // 25)))
    high_risk_cluster = int(clustered.loc[clustered[TARGET_COL] == 1, "rfm_cluster"].mode().iloc[0])
    df["rfms_cluster"] = clustered["rfm_cluster"]
    df[TARGET_COL] = (
        (df["rfms_cluster"] == high_risk_cluster) | (df["fraud_txn_count"] > 0)
    ).astype(int)
    cluster_summary = clustered.groupby("rfm_cluster")[["Recency", "Frequency", "Monetary"]].mean()

    metadata = {
        "target_definition": (
            "1 when customer belongs to the weakest RFMS cluster or has at least "
            "one observed fraud transaction; 0 otherwise."
        ),
        "high_risk_cluster": high_risk_cluster,
        "target_rate": float(df[TARGET_COL].mean()),
        "cluster_summary": cluster_summary.reset_index().to_dict(orient="records"),
    }
    return df, metadata


def make_modeling_table(raw_path: str) -> tuple[pd.DataFrame, dict]:
    transactions = load_transactions(raw_path)
    features = build_customer_features(transactions)
    features, metadata = add_proxy_risk_target(features)
    return features, metadata


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {CUSTOMER_ID, TARGET_COL, "fraud_txn_count", "rfms_cluster"}
    return [col for col in df.columns if col not in excluded]
