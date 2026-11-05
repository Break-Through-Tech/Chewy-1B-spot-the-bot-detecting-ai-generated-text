"""FastAPI application for Chewy Spot the Bot."""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .inference import SpotTheBotPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = SpotTheBotPredictor()  # Load exactly once per worker.
    yield


app = FastAPI(title="Spot the Bot API", version="0.1.0", lifespan=lifespan)

# Localhost is allowed for development. Set FRONTEND_ORIGIN to deployed Vercel URL.
allowed_origins = ["http://localhost:3000"]

if os.getenv("FRONTEND_ORIGIN"):
    allowed_origins.append(os.environ["FRONTEND_ORIGIN"].rstrip("/"))

app.add_middleware(CORSMiddleware,
                   allow_origins=allowed_origins,
                   allow_methods=["POST", "GET"],
                   allow_headers=["Content-Type"]
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class PredictResponse(BaseModel):
    prediction: str
    ai_score: float
    threshold: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Text must not be blank")
    return app.state.predictor.predict(request.text)
