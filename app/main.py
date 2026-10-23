from contextlib import asynccontextmanager
from pathlib import Path

import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError("Model artifact not found. Run 'python -m src.train' first.")

    artifact = joblib.load(MODEL_PATH)

    app.state.model = artifact["model"]
    app.state.feature_names = artifact["feature_names"]
    app.state.class_names = artifact["class_names"]
    app.state.model_accuracy = artifact["accuracy"]

    yield


app = FastAPI(
    title="ModelOps Inference API",
    description="A model-serving API for the ModelOps platform.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    sepal_length: float = Field(gt=0, le=10)
    sepal_width: float = Field(gt=0, le=10)
    petal_length: float = Field(gt=0, le=10)
    petal_width: float = Field(gt=0, le=10)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "modelops",
    }


@app.get("/model/info")
def model_info():
    return {
        "model_type": "LogisticRegression",
        "framework": "scikit-learn",
        "version": "0.1.0",
        "evaluation_accuracy": app.state.model_accuracy,
        "features": app.state.feature_names,
        "classes": app.state.class_names,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    model = app.state.model

    features = [
        [
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]
    ]

    try:
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]

        return {
            "prediction": app.state.class_names[prediction],
            "class_id": prediction,
            "confidence": float(max(probabilities)),
            "model_version": "0.1.0",
        }

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Invalid prediction input.",
        ) from exc
