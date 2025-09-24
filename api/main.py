"""
FastAPI mínima con:
- POST /predict  -> predicción + probabilidad
- POST /llm/ask  -> respuesta LLM + metadatos
Incluye middleware de latencia y request_id.
"""
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.predict import predict_single
from src.llm import ask_llm

app = FastAPI(title="ML+LLM API", version="0.1.0")

# Middleware de latencia + request_id
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()
    response = await call_next(request)
    process_time = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = str(process_time)
    return response

# Schemas inline (simple)
class PredictIn(BaseModel):
    features: Dict[str, Any] = Field(..., description="Diccionario de features para el modelo")

class PredictOut(BaseModel):
    prediction: int
    probability: float
    model_version: Optional[str] = "pipeline.joblib"

class LLMAskIn(BaseModel):
    question: str
    mode: Optional[str] = "prompting"

class LLMAskOut(BaseModel):
    answer: str
    citations: list = []
    latency_ms: int
    token_usage: Dict[str, int]
    mode: str
    confidence: Optional[float] = None
    guardrail: Optional[str] = None

@app.post("/predict", response_model=PredictOut)
def post_predict(payload: PredictIn):
    res = predict_single(payload.features)
    return PredictOut(**res)

@app.post("/llm/ask", response_model=LLMAskOut)
def post_llm(payload: LLMAskIn):
    res = ask_llm(payload.question, mode=payload.mode)
    return LLMAskOut(**res)
