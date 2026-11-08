# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""What the service measures about itself.

Prometheus metrics (GET /metrics):
  lesion_requests_total{endpoint,status}   every request
  lesion_request_seconds{endpoint}         latency histogram
  lesion_predictions_total{label}          predicted class counts
  lesion_confidence                        top-class probability histogram
  lesion_input_brightness                  mean pixel brightness of uploads (0-255)
  lesion_prediction_psi                    drift score (below)

Drift: a model can't see its own accuracy in production (no labels), but it
can notice that what it is shown has changed. The monitor keeps the last N
predictions and compares their class mix with the mix the model produced on
the official test set, using the population stability index:
    PSI = sum over classes of (p - q) * ln(p / q)
Common rule of thumb: below 0.1 stable, 0.1 to 0.25 worth a look, above 0.25
the inputs have shifted. A shift doesn't prove the model is wrong, only that
it is being used on data unlike its test set.
"""
from collections import deque
from threading import Lock

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("lesion_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("lesion_request_seconds", "Request latency", ["endpoint"],
                    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("lesion_predictions_total", "Predicted class", ["label"])
CONFIDENCE = Histogram("lesion_confidence", "Top-class probability", buckets=[i / 10 for i in range(1, 11)])
BRIGHTNESS = Histogram("lesion_input_brightness", "Mean brightness of uploaded images",
                       buckets=(25, 50, 75, 100, 125, 150, 175, 200, 225, 255))
PSI = Gauge("lesion_prediction_psi", "Population stability index of recent predictions vs the test set")

EPS = 1e-4


def psi(p, q) -> float:
    p = np.clip(np.asarray(p, float), EPS, None)
    q = np.clip(np.asarray(q, float), EPS, None)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((p - q) * np.log(p / q)))


class DriftMonitor:
    def __init__(self, classes, reference: dict, window: int = 500, min_samples: int = 50):
        self.classes = list(classes)
        self.reference = np.array([reference[c] for c in self.classes], float)
        self.window, self.min_samples = window, min_samples
        self.recent = deque(maxlen=window)
        self.lock = Lock()

    def observe(self, label: str):
        with self.lock:
            self.recent.append(self.classes.index(label))
            if len(self.recent) >= self.min_samples:
                PSI.set(self._psi())

    def _psi(self):
        counts = np.bincount(np.array(self.recent), minlength=len(self.classes))
        return psi(counts / counts.sum(), self.reference)

    def report(self) -> dict:
        with self.lock:
            n = len(self.recent)
            k = len(self.classes)
            counts = np.bincount(np.array(self.recent, dtype=int), minlength=k) if n else np.zeros(k)
            share = (counts / n).round(4).tolist() if n else [0.0] * len(self.classes)
            value = round(self._psi(), 4) if n >= self.min_samples else None
        status = ("not enough data" if value is None else "stable" if value < 0.1
                  else "worth a look" if value < 0.25 else "shifted")
        return {"window": n, "min_samples": self.min_samples, "psi": value, "status": status,
                "recent_share": dict(zip(self.classes, share, strict=False)),
                "reference_share": dict(zip(self.classes, self.reference.round(4).tolist(), strict=False))}
