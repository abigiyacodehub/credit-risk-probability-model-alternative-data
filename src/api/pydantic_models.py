from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_tier: str
    prediction: int
    recommendation: str


class BatchPredictionRequest(BaseModel):
    records: list[dict[str, Any]]


class BatchPredictionResponse(BaseModel):
    total: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
