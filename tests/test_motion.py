"""격한 움직임 (app/detectors/motion.py, FD-017): 20Hz 가속도·자이로 합성 신호로 규칙 확인."""

import numpy as np
import pytest

from app.detectors.motion import G, analyze_motion

RATE = 20


def make(acc_g, gyro=None, t0=1763280000000):
    """acc_g: 샘플별 가속도 크기(g) 또는 (n, 3) g 벡터. gyro: rad/s 크기."""
    acc_g = np.asarray(acc_g, dtype=float)
    vec = acc_g if acc_g.ndim == 2 else np.stack([np.zeros_like(acc_g), acc_g, np.zeros_like(acc_g)], axis=1)
    n = len(vec)
    gy = np.zeros(n) if gyro is None else np.broadcast_to(np.asarray(gyro, dtype=float), (n,))
    samples = [[t0 + i * 50, *(vec[i] * G), 0.0, 0.0, float(gy[i])] for i in range(n)]
    return {"rate_hz": RATE, "samples": samples}


def still(sec):
    return [1.0] * int(sec * RATE)


def test_standing_still_is_nothing():
    r = analyze_motion(make(still(3)))
    assert r["status"] == "ok" and r["score"] == 0.0 and r["type"] is None and r["samples"] == 60


def test_fall_free_fall_then_impact_then_still():
    r = analyze_motion(make(still(1) + [0.2] * 5 + [4.0] + still(1.75)))
    assert r["type"] == "fall" and r["score"] >= 0.8
    assert r["free_fall_sec"] == pytest.approx(0.25, abs=0.06) and r["peak_g"] == 4.0


def test_fall_at_window_end_is_borderline_danger():
    # 충격 뒤 정지 여부는 아직 모름 → 기준 충격이면 위험(약 0.6)
    r = analyze_motion(make(still(2.4) + [0.2] * 5 + [2.2] + still(0.25)))
    assert r["type"] == "fall" and 0.5 <= r["score"] < 0.8


def test_jump_with_soft_landing_is_not_danger():
    r = analyze_motion(make(still(1) + [0.3] * 4 + [1.6] + [1.2, 0.9] * 10 + still(0.9)))
    assert r["scores"]["fall"] < 0.5 and r["is_danger"] is False


def test_hard_hit_without_fall_is_impact():
    r = analyze_motion(make(still(1.5) + [4.0] + still(1.45)))
    assert r["type"] == "motion_impact" and 0.5 < r["score"] < 0.7


def test_struggle_needs_shaking_and_spinning():
    rnd = np.random.default_rng(0)
    shaking = 1.0 + rnd.normal(0, 2.0, 60)
    r = analyze_motion(make(np.abs(shaking), gyro=6.0))
    assert r["scores"]["struggle"] >= 0.5 and r["is_danger"] is True  # 몸싸움은 큰 충격도 함께 나와 유형은 더 큰 쪽
    assert analyze_motion(make(np.abs(shaking), gyro=0.5))["scores"]["struggle"] < 0.5  # 회전 없음


def test_walking_and_running_are_not_danger():
    t = np.arange(60) / RATE
    walk = 1.0 + 0.4 * np.sin(2 * np.pi * 2 * t)
    run = 1.0 + 0.9 * np.sin(2 * np.pi * 3 * t)
    for m in (make(walk, gyro=2.0), make(run, gyro=4.0)):
        r = analyze_motion(m)
        assert r["is_danger"] is False and max(r["scores"].values()) <= 0.3


def test_missing_or_short_is_no_data():
    assert analyze_motion(None)["status"] == "no_data"
    r = analyze_motion(make(still(0.3)))
    assert r["status"] == "no_data" and r["samples"] == 6
    bad = {"rate_hz": 20, "samples": [["x"]] * 30}
    assert analyze_motion(bad)["status"] == "no_data"


def test_phone_source_and_unsorted_samples():
    m = make(still(1) + [0.2] * 5 + [4.0] + still(1.75))
    m["samples"] = m["samples"][::-1]
    m["source"] = "phone"
    r = analyze_motion(m)
    assert r["type"] == "fall" and r["source"] == "phone"
