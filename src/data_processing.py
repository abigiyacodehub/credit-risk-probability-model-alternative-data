from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CUSTOMER_ID = "CustomerId"
TARGET_COL = "is_high_risk"

NUMERIC_AGG_COLUMNS = [
    "total_transaction_amount",
    "average_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
]


def extract_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["TransactionStartTime"] = pd.to_datetime(
        output["TransactionStartTime"], utc=True, errors="coerce"
    )
    output["transaction_hour"] = output["TransactionStartTime"].dt.hour
    output["transaction_day"] = output["TransactionStartTime"].dt.day
    output["transaction_month"] = output["TransactionStartTime"].dt.month
    output["transaction_year"] = output["TransactionStartTime"].dt.year
    return output


def aggregate_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    output = extract_datetime_features(df)
    output["is_credit"] = (output["Amount"] < 0).astype(int)
    output["is_debit"] = (output["Amount"] > 0).astype(int)
    snapshot_date = output["TransactionStartTime"].max() + pd.Timedelta(days=1)

    grouped = output.groupby(CUSTOMER_ID)
    features = grouped.agg(
        total_transaction_amount=("Amount", "sum"),
        average_transaction_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        std_transaction_amount=("Amount", "std"),
        total_value=("Value", "sum"),
        average_value=("Value", "mean"),
        max_value=("Value", "max"),
        credit_txn_ratio=("is_credit", "mean"),
        debit_txn_ratio=("is_debit", "mean"),
        unique_product_categories=("ProductCategory", "nunique"),
        unique_channels=("ChannelId", "nunique"),
        unique_providers=("ProviderId", "nunique"),
        fraud_txn_count=("FraudResult", "sum"),
        most_common_product_category=("ProductCategory", lambda s: s.mode().iloc[0]),
        most_common_channel=("ChannelId", lambda s: s.mode().iloc[0]),
        most_common_pricing_strategy=("PricingStrategy", lambda s: s.mode().iloc[0]),
        avg_transaction_hour=("transaction_hour", "mean"),
        avg_transaction_day=("transaction_day", "mean"),
        avg_transaction_month=("transaction_month", "mean"),
        avg_transaction_year=("transaction_year", "mean"),
        last_transaction=("TransactionStartTime", "max"),
    ).reset_index()
    features["std_transaction_amount"] = features["std_transaction_amount"].fillna(0)
    features["Recency"] = (snapshot_date - features["last_transaction"]).dt.days
    features["Frequency"] = features["transaction_count"]
    features["Monetary"] = features["total_value"]
    return features.drop(columns=["last_transaction"])


def compute_rfm_clusters(
    customer_features: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    output = customer_features.copy()
    n_clusters = max(1, min(n_clusters, len(output)))
    rfm = output[["Recency", "Frequency", "Monetary"]].fillna(0)
    rfm_for_clustering = rfm.copy()
    rfm_for_clustering["Monetary"] = np.log1p(rfm_for_clustering["Monetary"].clip(lower=0))
    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(rfm_for_clustering)
    clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    output["rfm_cluster"] = clusterer.fit_predict(scaled_rfm)

    cluster_profile = output.groupby("rfm_cluster")[["Recency", "Frequency", "Monetary"]].mean()
    risk_score = (
        cluster_profile["Recency"].rank(pct=True)
        + (1 - cluster_profile["Frequency"].rank(pct=True))
        + (1 - cluster_profile["Monetary"].rank(pct=True))
    )
    highest_risk_cluster = int(risk_score.idxmax())
    output[TARGET_COL] = (output["rfm_cluster"] == highest_risk_cluster).astype(int)
    return output, clusterer, scaler


def calculate_woe_iv(
    df: pd.DataFrame, feature: str, target: str = TARGET_COL, epsilon: float = 1e-6
) -> tuple[pd.DataFrame, float]:
    grouped = df.groupby(feature, observed=False)[target].agg(["count", "sum"]).reset_index()
    grouped.columns = [feature, "total", "events"]
    grouped["non_events"] = grouped["total"] - grouped["events"]
    total_events = grouped["events"].sum()
    total_non_events = grouped["non_events"].sum()
    grouped["event_rate"] = (grouped["events"] + epsilon) / (total_events + epsilon)
    grouped["non_event_rate"] = (grouped["non_events"] + epsilon) / (
        total_non_events + epsilon
    )
    grouped["woe"] = np.log(grouped["event_rate"] / grouped["non_event_rate"])
    grouped["iv"] = (grouped["event_rate"] - grouped["non_event_rate"]) * grouped["woe"]
    return grouped, float(grouped["iv"].sum())


class WoEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Iterable[str] | None = None):
        self.columns = list(columns or [])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        train = X.copy()
        train[TARGET_COL] = y
        self.woe_maps_ = {}
        self.iv_values_ = {}
        for col in self.columns:
            woe_table, iv = calculate_woe_iv(train[[col, TARGET_COL]], col)
            self.woe_maps_[col] = dict(zip(woe_table[col], woe_table["woe"]))
            self.iv_values_[col] = iv
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        output = X.copy()
        for col, mapping in self.woe_maps_.items():
            output[f"{col}_woe"] = output[col].map(mapping).fillna(0.0)
        return output


def build_preprocessing_pipeline(
    numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])


def create_processed_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    customer_features = aggregate_customer_features(raw_df)
    processed, _, _ = compute_rfm_clusters(customer_features)
    processed[TARGET_COL] = (
        (processed[TARGET_COL] == 1) | (processed["fraud_txn_count"] > 0)
    ).astype(int)
    return processed
