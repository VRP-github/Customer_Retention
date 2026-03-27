from __future__ import annotations
 
import os
import logging
from contextlib import asynccontextmanager
from typing import Any
 
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("churn-api")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "Best_Churn_Predictor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "latest")


model_store: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Connecting to MLflow at %s", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
 
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    logger.info("Loading model from %s …", model_uri)
    try:
        model_store["model"] = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        model_store["model"] = None
        model_store["load_error"] = str(exc)
 
    yield  
 
    logger.info("Shutting down — releasing model.")
    model_store.clear()
 

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predicts the probability that a customer will churn, "
        "based on their RFM (Recency, Frequency, Monetary) profile."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)
 
class CustomerFeatures(BaseModel):
 
    Frequency: int = Field(..., ge=1, description="Number of unique invoices")
    Monetary: float = Field(..., gt=0, description="Total revenue (£)")
    F_Score: int = Field(..., ge=1, le=5, description="Frequency quintile score 1–5")
    M_Score: int = Field(..., ge=1, le=5, description="Monetary quintile score 1–5")
 
    @field_validator("Monetary")
    @classmethod
    def monetary_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Monetary must be > 0")
        return round(v, 4)
 
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Frequency": 12,
                    "Monetary": 3250.75,
                    "F_Score": 4,
                    "M_Score": 5,
                }
            ]
        }
    }
 
 
class BatchRequest(BaseModel):
 
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=500)
 
 
class PredictionResult(BaseModel):
    churn_probability: float = Field(..., description="P(churn) in [0, 1]")
    churn_label: bool = Field(..., description="True if churn_probability >= 0.5")
    risk_tier: str = Field(..., description="Low / Medium / High / Critical")
 
 
class SingleResponse(BaseModel):
    prediction: PredictionResult
    model_name: str
    model_stage: str
    api_version: str
 
 
class BatchResponse(BaseModel):
    predictions: list[PredictionResult]
    total: int
    model_name: str
    model_stage: str
    api_version: str
 
FEATURE_ORDER = ["Frequency", "Monetary", "F_Score", "M_Score"]
 
 
def _risk_tier(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    if prob < 0.55:
        return "Medium"
    if prob < 0.75:
        return "High"
    return "Critical"
 
 
def _predict(features_df: pd.DataFrame) -> list[PredictionResult]:
    model = model_store.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model unavailable: {model_store.get('load_error', 'unknown error')}",
        )
    probabilities: np.ndarray = model.predict(features_df)

    try:
        proba = model._model_impl.predict_proba(features_df)[:, 1]
    except AttributeError:
        proba = np.asarray(probabilities, dtype=float)
 
    results = []
    for p in proba:
        prob = float(np.clip(p, 0.0, 1.0))
        results.append(
            PredictionResult(
                churn_probability=round(prob, 4),
                churn_label=prob >= 0.5,
                risk_tier=_risk_tier(prob),
            )
        )
    return results
 
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Customer Churn API — see /docs for usage."}
 
 
@app.get("/health", tags=["ops"])
async def health():
    model_ok = model_store.get("model") is not None
    return {
        "status": "healthy" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
    }
 
 
@app.post("/predict", response_model=SingleResponse, tags=["inference"])
async def predict_single(customer: CustomerFeatures):
    df = pd.DataFrame([customer.model_dump()])[FEATURE_ORDER]
    result = _predict(df)[0]
    return SingleResponse(
        prediction=result,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
        api_version=app.version,
    )
 
 
@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
async def predict_batch(batch: BatchRequest):
    df = pd.DataFrame([c.model_dump() for c in batch.customers])[FEATURE_ORDER]
    results = _predict(df)
    return BatchResponse(
        predictions=results,
        total=len(results),
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
        api_version=app.version,
    )

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)