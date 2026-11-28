"""경로 이탈 (POST /api/ai/route-status, app/route_state.py): 백엔드가 이탈·재진입을 알려 주면 세션별로 들고 있다."""

import os

os.environ["AI_LOAD_MODELS"] = "0"
os.environ.pop("AI_API_KEY", None)
os.environ.pop("AI_BACKEND_URL", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import config  # noqa: E402
from app.detectors.audio_detector import empty_sound_result, voice_result  # noqa: E402
from app.main import app  # noqa: E402
from app.risk import combine  # noqa: E402
from app.route_state import RouteStateStore, store  # noqa: E402

SID = "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
PCM_1S = bytes(16000 * 2)


@pytest.fixture(autouse=True)
def clean():
    store.clear()
    yield
    store.clear()


def test_default_is_on_route():
    s = RouteStateStore()
    assert s.get(SID) == {"status": "on_route", "deviated": False, "since": None, "distance_m": None, "applied": 0.0}
    assert s.get(None)["deviated"] is False


def test_deviated_stays_until_reentry():
    s = RouteStateStore()
    s.set(SID, True, at_ms=1000, distance_m=85.0)
    assert s.get(SID)["deviated"] is True and s.get(SID)["since"] == 1000  # 추가 신호가 없으면 계속 이탈
    s.set(SID, True, at_ms=5000, distance_m=120.0)  # 같은 상태 재전송: 시작 시각 유지, 거리만 갱신
    assert s.get(SID)["since"] == 1000 and s.get(SID)["distance_m"] == 120.0
    s.set(SID, False, at_ms=9000)  # 재진입 또는 경로 수정
    assert s.get(SID)["deviated"] is False and s.get(SID)["since"] is None


def test_late_old_signal_is_ignored():
    s = RouteStateStore()
    s.set(SID, False, at_ms=9000)
    assert s.set(SID, True, at_ms=1000)["deviated"] is False  # 늦게 도착한 이전 신호


def test_idle_session_is_forgotten(monkeypatch):
    s = RouteStateStore()
    s.set(SID, True, at_ms=1000)
    monkeypatch.setattr(config, "ROUTE_IDLE_SEC", -1)
    assert s.get(SID)["deviated"] is False


def test_route_boost_only_with_other_signals():
    deviated = {"status": "deviated", "deviated": True, "since": 1, "distance_m": 80.0, "applied": 0.0}
    alone = combine(voice_result("no_speech"), empty_sound_result(), route=dict(deviated))
    assert alone["risk_score"] == 0.0 and alone["is_danger"] is False  # 이탈만으로는 위험 아님

    r = dict(deviated)
    out = combine(voice_result("ok", "도와", 0.42 * config.VOICE_THRESHOLD / 0.5), empty_sound_result(), route=r)
    assert r["applied"] == pytest.approx(config.ROUTE_BOOST) and out["risk_score"] == pytest.approx(0.52, abs=0.01)
    assert out["is_danger"] is True

    on = {**deviated, "status": "on_route", "deviated": False}
    assert combine(voice_result("ok", "도와", 0.42 * config.VOICE_THRESHOLD / 0.5), empty_sound_result(),
                   route=on)["is_danger"] is False


def test_api_sets_state_and_analyze_returns_it():
    with TestClient(app) as c:
        r = c.post("/api/ai/route-status", json={"sessionId": SID, "deviated": True,
                                                  "at": "2026-11-28T12:00:00Z", "distanceM": 85.2})
        assert r.status_code == 200
        assert r.json() == {"sessionId": SID, "deviated": True, "since": 1795867200000, "distanceM": 85.2}
        a = c.post(f"/api/ai/analyze/pcm?session_id={SID}", content=PCM_1S,
                   headers={"Content-Type": "application/octet-stream"}).json()
        assert a["route"]["status"] == "deviated" and a["route"]["distance_m"] == 85.2

        r = c.post("/api/ai/route-status", json={"sessionId": SID, "deviated": False, "at": "2026-11-28T12:01:00Z"})
        assert r.json()["deviated"] is False and r.json()["since"] is None
        a = c.post(f"/api/ai/analyze/pcm?session_id={SID}", content=PCM_1S,
                   headers={"Content-Type": "application/octet-stream"}).json()
        assert a["route"]["deviated"] is False


def test_api_rejects_bad_body():
    with TestClient(app) as c:
        assert c.post("/api/ai/route-status", json={"deviated": True}).status_code == 422
        assert c.post("/api/ai/route-status", json={"sessionId": SID}).status_code == 422
        assert c.post("/api/ai/route-status", json={"sessionId": SID, "deviated": True,
                                                    "distanceM": -1}).status_code == 422
