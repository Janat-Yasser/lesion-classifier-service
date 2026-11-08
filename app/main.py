# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""HTTP API for the HAM10000 lesion classifier.

  GET  /health       liveness + whether the model loaded
  GET  /version      the model card: version, classes, test metrics, limits
  POST /v1/predict   one dermatoscopic image (JPEG or PNG, up to MAX_UPLOAD_MB)
  GET  /v1/drift     how the recent prediction mix compares with the test set
  GET  /metrics      Prometheus metrics

Configuration (environment): MODEL_DIR (default ./model), MAX_UPLOAD_MB (5),
DRIFT_WINDOW (500). Every response carries an X-Request-ID; every request is
logged as one JSON line.
"""
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .model import LesionModel
from .monitoring import BRIGHTNESS, CONFIDENCE, LATENCY, PREDICTIONS, REQUESTS, DriftMonitor

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
MAX_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "5")) * 2**20)
DRIFT_WINDOW = int(os.environ.get("DRIFT_WINDOW", "500"))
ALLOWED = {"JPEG", "PNG"}
MAX_PIXELS = 40_000_000   # refuse decompression bombs before decoding
DISCLAIMER = "Research demo trained on HAM10000. Not a medical device; not for diagnosis."

log = logging.getLogger("lesion")
logging.basicConfig(level=logging.INFO, format="%(message)s")
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = LesionModel(MODEL_DIR)
    state["model"] = model
    state["drift"] = DriftMonitor(model.classes, model.card["reference_prediction_share"], window=DRIFT_WINDOW)
    log.info(json.dumps({"event": "model_loaded", "version": model.version}))
    yield
    state.clear()


app = FastAPI(title="Lesion classifier", version="1.0.0", lifespan=lifespan,
              description="Serves the HAM10000 EfficientNet-B0. " + DISCLAIMER)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    t0 = time.perf_counter()
    endpoint = request.url.path
    try:
        response = await call_next(request)
        status = response.status_code
    except Exception:
        status = 500
        response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        log.exception(json.dumps({"event": "error", "request_id": rid, "path": endpoint}))
    seconds = time.perf_counter() - t0
    if endpoint != "/metrics":
        REQUESTS.labels(endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(seconds)
        log.info(json.dumps({"event": "request", "request_id": rid, "method": request.method, "path": endpoint,
                             "status": status, "ms": round(seconds * 1000, 1)}))
    response.headers["X-Request-ID"] = rid
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/version")
def version():
    card = state["model"].card
    keys = ["version", "architecture", "classes", "class_names", "input_size", "temperature", "melanoma_threshold",
            "test_metrics", "trained_on", "data_license", "intended_use", "limitations", "source_run"]
    return {k: card[k] for k in keys if k in card}


def read_image(raw: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(raw))
        if img.format not in ALLOWED:
            raise HTTPException(415, f"Unsupported image format {img.format}; send JPEG or PNG.")
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "Image has too many pixels.")
        img.load()
    except UnidentifiedImageError:
        raise HTTPException(415, "Not an image; send JPEG or PNG.") from None
    except (OSError, Image.DecompressionBombError):
        raise HTTPException(422, "The image could not be decoded.") from None
    if min(img.size) < 32:
        raise HTTPException(422, "Image is too small (minimum 32 x 32 pixels).")
    return img


@app.post("/v1/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File larger than {MAX_BYTES // 2**20} MB.")
    if not raw:
        raise HTTPException(422, "Empty file.")
    img = read_image(raw)
    t0 = time.perf_counter()
    result = state["model"].predict([img])[0]
    infer_ms = (time.perf_counter() - t0) * 1000

    PREDICTIONS.labels(result["label"]).inc()
    CONFIDENCE.observe(result["probability"])
    BRIGHTNESS.observe(float(np.asarray(img.convert("L").resize((64, 64))).mean()))
    state["drift"].observe(result["label"])
    return {"model_version": state["model"].version, **result, "inference_ms": round(infer_ms, 1),
            "disclaimer": DISCLAIMER}


@app.get("/v1/drift")
def drift():
    return state["drift"].report()


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
