# Copyright (c) 2026 Janat Yasser. All rights reserved.
import json

import numpy as np
import pytest
from PIL import Image

from app.model import LesionModel, ModelError


def test_rejects_tampered_model(model_dir, tmp_path):
    import shutil
    d = tmp_path / "m"
    shutil.copytree(model_dir, d)
    card = json.loads((d / "model_card.json").read_text())
    card["sha256"] = "0" * 64
    (d / "model_card.json").write_text(json.dumps(card))
    with pytest.raises(ModelError):
        LesionModel(d)


def test_batch_equals_single(model_dir):
    m = LesionModel(model_dir)
    rng = np.random.default_rng(1)
    imgs = [Image.fromarray(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)) for _ in range(3)]
    batch = m.predict(imgs)
    for img, b in zip(imgs, batch, strict=False):
        assert m.predict([img])[0]["probabilities"] == b["probabilities"]


def test_temperature_softens(model_dir):
    m = LesionModel(model_dir)
    img = Image.fromarray(np.random.default_rng(2).integers(0, 255, (224, 224, 3), dtype=np.uint8))
    m.temperature = 1.0
    sharp = m.predict([img])[0]["probability"]
    m.temperature = 4.0
    soft = m.predict([img])[0]["probability"]
    assert soft <= sharp


def test_preprocessing_matches_training_transform(model_dir):
    """Same numbers as the evaluation transform in ham10000/step3_train.py."""
    transforms = pytest.importorskip("torchvision.transforms")
    m = LesionModel(model_dir)
    img = Image.fromarray(np.random.default_rng(3).integers(0, 255, (450, 600, 3), dtype=np.uint8))
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    assert np.abs(m.preprocess(img) - t(img).numpy()).max() < 1e-5
