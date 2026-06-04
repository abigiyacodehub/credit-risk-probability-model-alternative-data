from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.api.pydantic_models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.predict import CreditRiskPredictor


app = FastAPI(
    title="Alternative Data Credit Risk PD API",
    description="Scores customer probability of default from Xente-style transaction features.",
    version="1.0.0",
)

predictor: CreditRiskPredictor | None = None


@app.on_event("startup")
def load_model() -> None:
    global predictor
    try:
        predictor = CreditRiskPredictor()
    except FileNotFoundError:
        predictor = None


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if predictor else "degraded", model_loaded=predictor is not None
    )


@app.get("/model/info")
def model_info() -> dict:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model has not been trained yet.")
    return {
        "feature_count": len(predictor.feature_columns),
        "best_model": predictor.metrics.get("best_model"),
        "target_definition": predictor.target_metadata.get("target_definition"),
    }


@app.post("/predict")
def predict_one(request: PredictionRequest) -> PredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model has not been trained yet.")
    return PredictionResponse(**predictor.predict(request.model_dump())[0])


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model has not been trained yet.")
    predictions = predictor.predict(request.records)
    return BatchPredictionResponse(
        total=len(predictions),
        predictions=[PredictionResponse(**prediction) for prediction in predictions],
    )
