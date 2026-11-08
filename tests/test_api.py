# Copyright (c) 2026 Janat Yasser. All rights reserved.
from conftest import image_bytes


def post(client, data, name="lesion.jpg", ctype="image/jpeg"):
    return client.post("/v1/predict", files={"file": (name, data, ctype)})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True}
    assert r.headers["x-request-id"]


def test_request_id_is_echoed(client):
    assert client.get("/health", headers={"X-Request-ID": "abc123"}).headers["x-request-id"] == "abc123"


def test_version_exposes_card(client):
    v = client.get("/version").json()
    assert v["version"].startswith("TEST MODEL") and v["classes"][0] == "MEL" and "limitations" in v


def test_predict_jpeg(client):
    r = post(client, image_bytes())
    assert r.status_code == 200, r.text
    body = r.json()
    probs = body["probabilities"]
    assert set(probs) == {"MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"}
    assert abs(sum(probs.values()) - 1) < 1e-3
    assert body["label"] == max(probs, key=probs.get) and body["probability"] == probs[body["label"]]
    assert body["melanoma"]["flag"] == (body["melanoma"]["probability"] >= body["melanoma"]["threshold"])
    assert "Not a medical device" in body["disclaimer"]


def test_predict_png_and_grayscale(client):
    assert post(client, image_bytes("PNG"), "x.png", "image/png").status_code == 200
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (100, 100), 120).save(buf, "PNG")
    assert post(client, buf.getvalue(), "g.png", "image/png").status_code == 200


def test_same_image_same_answer(client):
    a, b = post(client, image_bytes(seed=3)).json(), post(client, image_bytes(seed=3)).json()
    assert a["probabilities"] == b["probabilities"]


def test_rejects_non_image(client):
    assert post(client, b"hello, not an image", "a.txt", "text/plain").status_code == 415


def test_rejects_other_formats(client):
    assert post(client, image_bytes("GIF"), "a.gif", "image/gif").status_code == 415


def test_rejects_empty_tiny_and_huge(client):
    assert post(client, b"").status_code == 422
    assert post(client, image_bytes(size=(16, 16))).status_code == 422
    big = image_bytes("PNG", size=(1400, 1400))   # random noise PNG, well over 1 MB
    assert len(big) > 2**20 and post(client, big, "big.png", "image/png").status_code == 413


def test_metrics_count_predictions(client):
    post(client, image_bytes())
    text = client.get("/metrics").text
    for name in ["lesion_requests_total", "lesion_request_seconds_bucket", "lesion_predictions_total",
                 "lesion_confidence_bucket", "lesion_input_brightness_bucket", "lesion_prediction_psi"]:
        assert name in text, name


def test_drift_report(client):
    r = client.get("/v1/drift").json()
    assert r["status"] in {"not enough data", "stable", "worth a look", "shifted"}
    assert set(r["reference_share"]) == set(r["recent_share"])
