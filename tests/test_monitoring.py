# Copyright (c) 2026 Janat Yasser. All rights reserved.
import pytest

from app.monitoring import DriftMonitor, psi

C = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
REF = dict(zip(C, [0.11, 0.6, 0.06, 0.03, 0.14, 0.03, 0.03], strict=False))


def test_psi_zero_for_identical():
    assert psi([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0)


def test_psi_known_value():
    # (0.5-0.2)ln(0.5/0.2) + (0.5-0.8)ln(0.5/0.8)
    assert psi([0.5, 0.5], [0.2, 0.8]) == pytest.approx(0.3 * 0.916291 + 0.3 * 0.470004, rel=1e-4)


def test_monitor_waits_for_enough_data():
    m = DriftMonitor(C, REF, window=100, min_samples=50)
    for _ in range(10):
        m.observe("NV")
    assert m.report()["status"] == "not enough data"


def test_monitor_stable_when_mix_matches():
    m = DriftMonitor(C, REF, window=1000, min_samples=50)
    for c, share in REF.items():
        for _ in range(round(share * 1000)):
            m.observe(c)
    r = m.report()
    assert r["psi"] < 0.01 and r["status"] == "stable"


def test_monitor_flags_shift():
    m = DriftMonitor(C, REF, window=200, min_samples=50)
    for _ in range(200):
        m.observe("MEL")     # everything suddenly looks like melanoma
    assert m.report()["status"] == "shifted"


def test_window_forgets_old_predictions():
    m = DriftMonitor(C, REF, window=60, min_samples=50)
    for _ in range(60):
        m.observe("MEL")
    for _ in range(60):
        m.observe("NV")
    assert m.report()["recent_share"]["MEL"] == 0
