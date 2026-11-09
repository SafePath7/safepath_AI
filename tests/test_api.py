"""더미 모드(AI_LOAD_MODELS=0) API 테스트. 실행: pytest"""

import os

os.environ["AI_LOAD_MODELS"] = "0"
os.environ.pop("AI_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

PCM_1_5S = bytes(24000 * 2)  # 16kHz × 1.5초 × 16bit


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "mode": "dummy", "models_loaded": []}


def test_analyze_pcm_dummy():
    with TestClient(app) as c:
        r = c.post("/analyze/pcm", content=PCM_1_5S, headers={"Content-Type": "application/octet-stream"})
    body = r.json()
    assert r.status_code == 200
    assert body["mode"] == "dummy"
    assert body["duration_sec"] == 1.5
    assert body["is_danger"] is False


def test_analyze_pcm_other_sample_rate_dummy():
    # 더미 모드에는 librosa가 없으므로 리샘플링 없이 길이만 계산해야 한다 (이전에는 500).
    with TestClient(app) as c:
        r = c.post("/analyze/pcm?sample_rate=44100", content=bytes(44100 * 2))
    assert r.status_code == 200
    assert r.json()["duration_sec"] == 1.0


def test_analyze_pcm_odd_bytes():
    with TestClient(app) as c:
        r = c.post("/analyze/pcm", content=b"\x00\x00\x00")
    assert r.status_code == 400


def test_analyze_pcm_empty():
    with TestClient(app) as c:
        r = c.post("/analyze/pcm", content=b"")
    assert r.status_code == 400


def test_analyze_file_dummy():
    with TestClient(app) as c:
        r = c.post("/analyze", files={"file": ("clip.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["received_bytes"] == 8
    assert body["mode"] == "dummy"
    assert body["duration_sec"] is None
    assert body["is_danger"] is False


def test_analyze_file_empty():
    with TestClient(app) as c:
        r = c.post("/analyze", files={"file": ("clip.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_upload_too_large(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)  # 약 1KB
    with TestClient(app) as c:
        assert c.post("/analyze/pcm", content=bytes(4096)).status_code == 413
        assert c.post("/analyze", files={"file": ("a.wav", bytes(4096), "audio/wav")}).status_code == 413


def test_api_key(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "API_KEY", "secret")
    with TestClient(app) as c:
        assert c.post("/analyze/pcm", content=PCM_1_5S).status_code == 401
        assert c.post("/analyze/pcm", content=PCM_1_5S, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/analyze/pcm", content=PCM_1_5S, headers={"X-API-Key": "secret"}).status_code == 200


def test_detect_danger_phrase_ignores_spaces():
    from app.detectors.audio_detector import detect_danger_phrase

    assert detect_danger_phrase("살려 주세요")
    assert detect_danger_phrase("누가 좀 도와 줘요")
    assert not detect_danger_phrase("오늘 날씨 좋네요")
