from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src.config import MODEL_PATH


def probability_to_risk_tier(probability: float) -> str:
    if probability < 0.25:
        return "LOW"
    if probability < 0.5:
        return "MEDIUM"
    if probability < 0.75:
        return "HIGH"
    return "VERY_HIGH"


class CreditRiskPredictor:
    def __init__(self, model_path: str = str(MODEL_PATH)):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.feature_columns = artifact["feature_columns"]
        self.metrics = artifact.get("metrics", {})
        self.target_metadata = artifact.get("target_metadata", {})

    def predict(self, records: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> list[dict]:
        if isinstance(records, dict):
            df = pd.DataFrame([records])
        elif isinstance(records, list):
            df = pd.DataFrame(records)
        else:
            df = records.copy()

        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = None
        X = df[self.feature_columns]
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)
        return [
            {
                "probability_of_default": round(float(prob), 4),
                "risk_tier": probability_to_risk_tier(float(prob)),
                "prediction": int(pred),
                "recommendation": "REVIEW_OR_DECLINE" if pred else "APPROVE",
            }
            for prob, pred in zip(probabilities, predictions)
        ]


if __name__ == "__main__":
    sample = {
        "tx_count": 5,
        "active_days": 2,
        "total_amount": 2000,
        "total_value": 2000,
        "avg_amount": 400,
        "median_amount": 300,
        "max_value": 1000,
        "std_value": 250,
        "credit_txn_ratio": 0.0,
        "debit_txn_ratio": 1.0,
        "unique_products": 2,
        "unique_categories": 1,
        "unique_channels": 1,
        "unique_providers": 1,
        "most_common_category": "airtime",
        "most_common_channel": "ChannelId_3",
        "most_common_provider": "ProviderId_6",
        "most_common_pricing": "2",
        "avg_hour": 11.5,
        "weekend_ratio": 0.2,
        "customer_age_days": 10,
        "recency_days": 30,
        "tx_per_active_day": 2.5,
        "avg_value_per_txn": 400,
        "net_amount_per_txn": 400,
    }
    print(CreditRiskPredictor().predict(sample)[0])
