# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""The model: an ONNX export of the EfficientNet-B0 trained on HAM10000, plus its card.

model/model.onnx        the network (input: N x 3 x 224 x 224 float32, output: N x 7 logits)
model/model_card.json   classes, the temperature and melanoma threshold chosen on
                        validation, test metrics, a reference prediction mix for
                        drift monitoring, and the SHA-256 of model.onnx

Preprocessing matches the evaluation transform used in training exactly:
resize to 224 x 224 (bilinear), scale to [0, 1], normalise with the ImageNet
mean and std. tests/test_model.py checks this against torchvision when it is
installed.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ModelError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LesionModel:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
        onnx_path = model_dir / "model.onnx"
        if sha256(onnx_path) != self.card["sha256"]:
            raise ModelError("model.onnx does not match the SHA-256 in model_card.json")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(self.card.get("threads", 0)) or 0
        self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.classes = self.card["classes"]
        self.names = self.card["class_names"]
        self.size = int(self.card.get("input_size", 224))
        self.temperature = float(self.card["temperature"])
        self.mel_index = self.classes.index("MEL")
        self.mel_threshold = float(self.card["melanoma_threshold"])

    @property
    def version(self) -> str:
        return self.card["version"]

    def preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        return x.transpose(2, 0, 1)

    def predict(self, images: list[Image.Image]) -> list[dict]:
        batch = np.stack([self.preprocess(im) for im in images]).astype(np.float32)
        logits = self.session.run(None, {self.input_name: batch})[0] / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        out = []
        for p in probs:
            top = int(p.argmax())
            mel = float(p[self.mel_index])
            out.append({
                "label": self.classes[top],
                "name": self.names[self.classes[top]],
                "probability": round(float(p[top]), 4),
                "probabilities": {c: round(float(v), 4) for c, v in zip(self.classes, p, strict=False)},
                "melanoma": {"probability": round(mel, 4), "threshold": round(self.mel_threshold, 4),
                             "flag": mel >= self.mel_threshold},
            })
        return out
