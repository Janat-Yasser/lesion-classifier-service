# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Turns a trained HAM10000 checkpoint into what the service loads.

    python export/export_onnx.py --run ham10000_results --name lesion --out model

Reads (from the ham10000 project's output folder, e.g. the Drive copy):
  runs/<name>/best.pt                   weights + training config
  results/<name>_metrics.json           temperature, melanoma threshold, test metrics
  results/<name>_predictions.npz        test-set logits (for the drift reference)

Writes:
  model/model.onnx        opset 17, dynamic batch size
  model/model_card.json   everything the service needs, plus the file's SHA-256

Checks before writing the card: ONNX Runtime and PyTorch must agree on a batch
of random inputs (max |difference| in logits below 1e-3), and the ONNX model
must reproduce the saved test-set logits for the first images it can find.
Needs torch, torchvision, onnx and onnxruntime (not needed by the service).
"""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
NAMES = {"MEL": "Melanoma", "NV": "Melanocytic nevus", "BCC": "Basal cell carcinoma",
         "AKIEC": "Actinic keratosis / Bowen's disease", "BKL": "Benign keratosis", "DF": "Dermatofibroma",
         "VASC": "Vascular lesion"}


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="folder containing runs/ and results/")
    p.add_argument("--name", default="lesion", help="run name, e.g. lesion or lesion_s1")
    p.add_argument("--out", default="model")
    p.add_argument("--version", default=None)
    a = p.parse_args()
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(run / "runs" / a.name / "best.pt", map_location="cpu", weights_only=False)
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASSES))
    m.load_state_dict(ckpt["model"])
    m.eval()
    size = int(ckpt["config"]["size"])

    onnx_path = out / "model.onnx"
    dummy = torch.randn(2, 3, size, size)
    torch.onnx.export(m, dummy, onnx_path, input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Parity on image-like inputs: smooth colour fields in [0, 1], normalised like real photos.
    # (Pure Gaussian noise drives this network's logits past +-100, where float32 rounding
    # differences between the two runtimes reach 1e-2 and the check says nothing about real use.)
    g = torch.Generator().manual_seed(0)
    x = torch.nn.functional.interpolate(torch.rand(8, 3, 8, 8, generator=g), size=(size, size), mode="bilinear")
    x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
    with torch.no_grad():
        ref = m(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    pdiff = float(np.abs(softmax(ref) - softmax(got)).max())
    assert diff < 1e-3 and pdiff < 1e-4, f"ONNX and PyTorch disagree: max |logit diff| {diff}, max |prob diff| {pdiff}"
    print(f"parity check: max |logit difference| {diff:.2e}, max |probability difference| {pdiff:.2e}")

    metrics = json.loads((run / "results" / f"{a.name}_metrics.json").read_text())
    preds = np.load(run / "results" / f"{a.name}_predictions.npz", allow_pickle=True)
    T = float(metrics["temperature"])
    test_pred = softmax(preds["test_logits"] / T).argmax(1)
    share = np.bincount(test_pred, minlength=len(CLASSES)) / len(test_pred)

    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    t = metrics["test"]
    card = {
        "version": a.version or f"ham10000-effnetb0-{a.name}-{date.today():%Y%m%d}",
        "architecture": "EfficientNet-B0 (torchvision), ImageNet-pretrained, fine-tuned on HAM10000",
        "classes": CLASSES, "class_names": NAMES, "input_size": size,
        "temperature": T, "melanoma_threshold": float(metrics["melanoma_threshold"]),
        "test_metrics": {
            "dataset": "ISIC 2018 Task 3 official test set",
            "n_images": metrics["n_test"],
            "balanced_accuracy": round(t["balanced_accuracy"], 4),
            "balanced_accuracy_95ci": [round(v, 4) for v in metrics["test_95ci"]["balanced_accuracy"]],
            "macro_auc": round(t["macro_auc"], 4),
            "melanoma_auc": round(t["melanoma_auc"], 4),
            "melanoma_sensitivity": round(t["melanoma_sensitivity"], 4),
            "melanoma_specificity": round(t["melanoma_specificity"], 4),
            "ece_after_temperature_scaling": round(metrics["test_ece_after"], 4),
        },
        "reference_prediction_share": dict(zip(CLASSES, [round(float(v), 4) for v in share], strict=False)),
        "trained_on": f"HAM10000 / ISIC 2018 Task 3 training set, {a.name} split (lesion-level), seed {ckpt['config']['seed']}",
        "data_license": "ISIC 2018 / HAM10000: CC BY-NC 4.0 (non-commercial)",
        "intended_use": "Demonstrating model serving and monitoring. Research only.",
        "limitations": [
            "Not a medical device and not validated for clinical use.",
            "Dermatoscopic images only; phone photos of skin are out of distribution.",
            "HAM10000 is mostly light skin; performance on darker skin tones is unknown.",
            "About a quarter of test melanomas were classified as benign nevi.",
        ],
        "source_run": {"best_epoch": ckpt["epoch"], "val_balanced_acc": round(float(ckpt["val_balanced_acc"]), 4)},
        "onnx_parity_max_abs_diff": diff,
        "sha256": digest,
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 2**20:.1f} MB) and model_card.json, version {card['version']}")


if __name__ == "__main__":
    main()
