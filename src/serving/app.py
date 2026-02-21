"""
FastAPI application for fraud detection: single and batch prediction, health, metrics.
"""

import logging
import time
import uuid
from contextvars import ContextVar
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.streaming.stream_processor import StreamProcessor

logger = logging.getLogger(__name__)

# Transaction input: required fields + allow extra for full payload
class TransactionInput(BaseModel):
    trans_date_trans_time: str
    cc_num: int | str
    amt: float

    model_config = {"extra": "allow"}

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json", exclude_none=False)
        extra = getattr(self, "__pydantic_extra__", None)
        if extra:
            d.update(extra)
        return d


class PredictionOutput(BaseModel):
    transaction_id: str
    fraud_score: float
    risk_tier: str
    iso_score: float
    ae_score: float
    xgb_score: float
    should_alert: bool
    processing_time_ms: float


class BatchInput(BaseModel):
    transactions: list[TransactionInput] = Field(..., max_length=100)


class BatchOutput(BaseModel):
    predictions: list[PredictionOutput]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    avg_latency_ms: float
    total_processed: int


# Global processor and start time (set on startup)
stream_processor: StreamProcessor | None = None
app_start_time: float = 0.0
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for transactions using Isolation Forest, Autoencoder, and ensemble.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_processing_time_and_log(request: Request, call_next):
    correlation_id = str(uuid.uuid4())
    correlation_id_ctx.set(correlation_id)
    start = time.perf_counter()
    logger.info("Request %s %s correlation_id=%s", request.method, request.url.path, correlation_id)
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Processing-Time"] = "%.2f" % elapsed_ms
    response.headers["X-Correlation-ID"] = correlation_id
    logger.info("Response %s %s status=%s correlation_id=%s time_ms=%.2f", request.method, request.url.path, response.status_code, correlation_id, elapsed_ms)
    return response


@app.on_event("startup")
async def startup_event():
    global stream_processor, app_start_time
    app_start_time = time.perf_counter()
    try:
        stream_processor = StreamProcessor()
        logger.info("StreamProcessor loaded; API ready")
    except Exception as e:
        logger.error("Startup failed: %s", e)
        raise


def get_processor() -> StreamProcessor:
    if stream_processor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return stream_processor


@app.post("/predict", response_model=PredictionOutput)
async def predict(body: TransactionInput):
    """Score a single transaction."""
    proc = get_processor()
    try:
        result = proc.process_transaction(body.to_dict())
        return PredictionOutput(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Predict failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(body: BatchInput):
    """Score up to 100 transactions."""
    proc = get_processor()
    predictions: list[PredictionOutput] = []
    for txn in body.transactions:
        try:
            result = proc.process_transaction(txn.to_dict())
            predictions.append(PredictionOutput(**result))
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid transaction: %s" % e)
        except Exception as e:
            logger.exception("Batch item failed: %s", e)
            raise HTTPException(status_code=500, detail="Batch prediction failed")
    return BatchOutput(predictions=predictions)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check: status, models loaded, latency, total processed."""
    proc = get_processor()
    return HealthResponse(**proc.health_check())


@app.get("/metrics")
async def metrics():
    """Return processing metrics: total_processed, latencies, risk tier counts, alerts, uptime."""
    proc = get_processor()
    return proc.get_metrics()


@app.get("/")
async def root():
    """Basic info and uptime."""
    uptime = time.perf_counter() - app_start_time if app_start_time else 0
    return {"service": "fraud-detection-api", "uptime_seconds": round(uptime, 2)}
