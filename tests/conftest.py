# Copyright (c) 2026 Janat Yasser. All rights reserved.
import importlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    from make_test_model import build
    return build(tmp_path_factory.mktemp("model"))


@pytest.fixture
def client(model_dir, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    import app.main as main
    main = importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def image_bytes(fmt="JPEG", size=(300, 225), color=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (size[1], size[0], 3)
    arr = rng.integers(0, 255, shape, dtype=np.uint8) if color is None else np.full(shape, color, np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()
