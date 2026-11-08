# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Builds a tiny stand-in model with the real model's interface, for tests and CI.

    python scripts/make_test_model.py tests/model

Same input (N x 3 x 224 x 224), same 7 logits, same card fields as the real
export, but the network is just global average pooling and a fixed linear
layer, so tests run in milliseconds and need no trained weights. Its card says
"TEST MODEL" in the version, so it can't be mistaken for the real one.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
NAMES = {"MEL": "Melanoma", "NV": "Melanocytic nevus", "BCC": "Basal cell carcinoma",
         "AKIEC": "Actinic keratosis / Bowen's disease", "BKL": "Benign keratosis", "DF": "Dermatofibroma",
         "VASC": "Vascular lesion"}


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    W = numpy_helper.from_array(rng.normal(0, 2, (3, len(CLASSES))).astype(np.float32), "W")
    B = numpy_helper.from_array(np.zeros(len(CLASSES), np.float32), "B")
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
         helper.make_node("Flatten", ["pooled"], ["flat"]),
         helper.make_node("Gemm", ["flat", "W", "B"], ["logits"])],
        "test_model",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", len(CLASSES)])],
        [W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = out / "model.onnx"
    onnx.save(model, path)
    card = {
        "version": "TEST MODEL (untrained stand-in)",
        "architecture": "global average pool + linear (test stand-in)",
        "classes": CLASSES, "class_names": NAMES, "input_size": 224,
        "temperature": 1.5, "melanoma_threshold": 0.3,
        "test_metrics": {}, "reference_prediction_share": {c: round(1 / len(CLASSES), 4) for c in CLASSES},
        "intended_use": "Tests only.", "limitations": ["Untrained."],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "tests/model")))
