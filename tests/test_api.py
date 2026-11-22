"""더미 모드(AI_LOAD_MODELS=0) API 테스트. 실행: pytest"""

import os

os.environ["AI_LOAD_MODELS"] = "0"
os.environ.pop("AI_API_KEY", None)
os.environ.pop("AI_BACKEND_URL", None)
os.environ.pop("AI_DEMO_PROFILE", None)

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
        r = c.post("/api/ai/analyze/pcm", content=PCM_1_5S, headers={"Content-Type": "application/octet-stream"})
    body = r.json()
    assert r.status_code == 200
    assert body["mode"] == "dummy"
    assert body["duration_sec"] == 1.5
    assert body["is_danger"] is False
    # 더미 응답도 모델 모드와 같은 FD-015 형식이어야 백엔드가 그대로 파싱할 수 있다
    assert body["yamnet"] == {
        "score": 0.0, "type": None, "is_danger": False,
        "categories": {"scream": 0.0, "shout": 0.0, "glass_break": 0.0, "impact": 0.0},
        "hits": [],
    }


def test_dummy_yamnet_matches_model_schema():
    from app.detectors.audio_detector import empty_sound_result
    from app.main import DUMMY_RESULT

    assert DUMMY_RESULT["yamnet"] == empty_sound_result()


def test_dummy_risk_matches_model_schema():
    from app.main import DUMMY_RESULT
    from app.risk import combine

    assert {k: DUMMY_RESULT[k] for k in ("risk_score", "risk_type", "is_danger", "signals")} == combine(
        DUMMY_RESULT["whisper"], DUMMY_RESULT["yamnet"])
    from app.detectors.loudness import empty_loudness_result
    from app.detectors.motion import empty_motion_result
    assert DUMMY_RESULT["loudness"].keys() == empty_loudness_result().keys()
    assert DUMMY_RESULT["motion"].keys() == empty_motion_result("x").keys()


def test_analyze_echoes_clip_meta():
    # 폰이 3초 WAV를 1초마다 보낼 때 붙이는 식별 정보를 그대로 돌려줘야 응답을 요청과 짝지을 수 있다
    form = {"session_id": "s-1", "seq": "12", "captured_at": "1763280000000"}
    with TestClient(app) as c:
        r = c.post("/api/ai/analyze", files={"file": ("clip.wav", b"RIFF0000", "audio/wav")}, data=form)
        p = c.post("/api/ai/analyze/pcm", params=form, content=PCM_1_5S)
    for body in (r.json(), p.json()):
        assert body["session_id"] == "s-1" and body["seq"] == 12 and body["captured_at"] == 1763280000000
        assert body["event_id"] is None and body["event_status"] == "off"
        assert body["risk_score"] == 0.0 and body["risk_type"] is None
        assert isinstance(body["processing_ms"], int)


def test_analyze_without_meta_still_works():
    with TestClient(app) as c:
        body = c.post("/api/ai/analyze/pcm", content=PCM_1_5S).json()
    assert body["session_id"] is None and body["seq"] is None


def test_analyze_rejects_bad_seq():
    with TestClient(app) as c:
        assert c.post("/api/ai/analyze/pcm", params={"seq": "-1"}, content=PCM_1_5S).status_code == 422


def test_dummy_speaker_matches_model_schema():
    from app.detectors.speaker import verify_result
    from app.main import DUMMY_RESULT

    assert DUMMY_RESULT["speaker"] == verify_result("off")


def test_dummy_whisper_matches_model_schema():
    from app.detectors.audio_detector import voice_result
    from app.main import DUMMY_RESULT

    assert DUMMY_RESULT["whisper"] == voice_result("skipped")


def test_voice_profile_dummy():
    with TestClient(app) as c:
        r = c.post("/api/ai/voice-profiles", files={"file": ("enroll.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "dummy" and body["model_version"] == "dummy"
    # voice_profiles NOT NULL 컬럼을 모두 채울 수 있어야 백엔드 저장 흐름을 더미로 확인할 수 있다
    assert len(body["embedding"]) == 192
    assert isinstance(body["baseline_volume"], float) and isinstance(body["baseline_pitch"], float)


def test_voice_profile_with_urgent_recording_dummy():
    # 2단계(위급할 때 목소리)는 선택 항목. 보내도 형식은 같다
    files = {"file": ("calm.wav", b"RIFF0000", "audio/wav"), "urgent_file": ("urgent.wav", b"RIFF1111", "audio/wav")}
    with TestClient(app) as c:
        r = c.post("/api/ai/voice-profiles", files=files)
    assert r.status_code == 200 and len(r.json()["embedding"]) == 192


CALM = {"file": ("calm.wav", b"RIFF0000", "audio/wav")}
AUTH = {"Authorization": "Bearer test-token"}


def test_voice_profile_saves_to_backend_with_forwarded_token(fake_backend):
    # 폰 → AI → 백엔드 registerVoiceProfile (토큰 그대로 전달, 처음이면 POST, 이미 있으면 PUT /me)
    with TestClient(app) as c:
        first = c.post("/api/ai/voice-profiles", files=CALM, headers=AUTH)
        again = c.post("/api/ai/voice-profiles", files=CALM, headers=AUTH)
    assert first.status_code == 200 and first.json()["saved_to"] == "backend:created"
    assert again.json()["saved_to"] == "backend:updated"
    post = fake_backend.find("POST", "/api/voice-profiles")[0]
    assert post["token"] == AUTH["Authorization"] and post["key"] == "internal-k"


def test_voice_profile_needs_token_when_backend_set(fake_backend):
    with TestClient(app) as c:
        r = c.post("/api/ai/voice-profiles", files=CALM)
    assert r.status_code == 401 and r.json()["detail"]["code"] == "no_token"
    assert fake_backend.calls == []


def test_voice_profile_backend_errors(fake_backend):
    with TestClient(app) as c:
        fake_backend.status[("POST", "/api/voice-profiles")] = 401  # 만료된 토큰 → 그대로 전달
        assert c.post("/api/ai/voice-profiles", files=CALM, headers=AUTH).status_code == 401
        fake_backend.status[("POST", "/api/voice-profiles")] = 500  # 백엔드 오류 → 502
        r = c.post("/api/ai/voice-profiles", files=CALM, headers=AUTH)
    assert r.status_code == 502 and r.json()["detail"]["code"] == "save_failed"


def test_voice_profile_demo_file_without_backend(monkeypatch, tmp_path):
    from app import config

    monkeypatch.setattr(config, "BACKEND_URL", None)
    monkeypatch.setattr(config, "DEMO_PROFILE", str(tmp_path / "demo.json"))
    with TestClient(app) as c:
        r = c.post("/api/ai/voice-profiles", files=CALM)
    assert r.json()["saved_to"] == "demo" and (tmp_path / "demo.json").is_file()


def test_analyze_accepts_motion_and_location(monkeypatch):
    # 움직임(FD-017)·위치는 지금은 점수에 쓰지 않고, 위험 이벤트의 분석 데이터(sensor)로 넘긴다
    from app import events

    seen = {}
    monkeypatch.setattr(events.reporter, "report",
                        lambda result, meta, token, clip, motion, location: seen.update(
                            motion=motion, location=location, token=token, clip=clip) or (None, "off"))
    form = {"session_id": "s-1", "motion": '{"rate_hz": 20, "samples": [[0, 0.1, 9.8, 0.2, 0, 0.1, 0]]}',
            "lat": "37.5", "lng": "127.0", "accuracy_m": "12"}
    with TestClient(app) as c:
        r = c.post("/api/ai/analyze", files={"file": ("voice_1.wav", b"RIFF0000", "audio/wav")}, data=form, headers=AUTH)
        bad = c.post("/api/ai/analyze", files={"file": ("voice_1.wav", b"RIFF0000", "audio/wav")}, data={"motion": "{"})
    assert r.status_code == 200 and bad.status_code == 400
    assert seen["motion"]["samples"][0][2] == 9.8 and seen["location"]["lat"] == 37.5
    assert seen["token"] == AUTH["Authorization"] and seen["clip"] == ("voice_1.wav", b"RIFF0000")


def test_old_paths_are_gone():
    with TestClient(app) as c:
        assert c.post("/analyze", files=CALM).status_code == 404
        assert c.post("/voice-profile", files=CALM).status_code == 404


def test_voice_profile_empty_file():
    with TestClient(app) as c:
        r = c.post("/api/ai/voice-profiles", files={"file": ("enroll.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_analyze_pcm_other_sample_rate_dummy():
    # 더미 모드에는 librosa가 없으므로 리샘플링 없이 길이만 계산해야 한다 (이전에는 500).
    with TestClient(app) as c:
        r = c.post("/api/ai/analyze/pcm?sample_rate=44100", content=bytes(44100 * 2))
    assert r.status_code == 200
    assert r.json()["duration_sec"] == 1.0


def test_analyze_pcm_odd_bytes():
    with TestClient(app) as c:
        r = c.post("/api/ai/analyze/pcm", content=b"\x00\x00\x00")
    assert r.status_code == 400


def test_analyze_pcm_empty():
    with TestClient(app) as c:
        r = c.post("/api/ai/analyze/pcm", content=b"")
    assert r.status_code == 400


def test_analyze_file_dummy():
    with TestClient(app) as c:
        r = c.post("/api/ai/analyze", files={"file": ("clip.wav", b"RIFF0000", "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["received_bytes"] == 8
    assert body["mode"] == "dummy"
    assert body["duration_sec"] is None
    assert body["is_danger"] is False


def test_analyze_file_empty():
    with TestClient(app) as c:
        r = c.post("/api/ai/analyze", files={"file": ("clip.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_upload_too_large(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)  # 약 1KB
    with TestClient(app) as c:
        assert c.post("/api/ai/analyze/pcm", content=bytes(4096)).status_code == 413
        assert c.post("/api/ai/analyze", files={"file": ("a.wav", bytes(4096), "audio/wav")}).status_code == 413


def test_api_key(monkeypatch):
    from app import config

    monkeypatch.setattr(config, "API_KEY", "secret")
    with TestClient(app) as c:
        assert c.post("/api/ai/analyze/pcm", content=PCM_1_5S).status_code == 401
        assert c.post("/api/ai/analyze/pcm", content=PCM_1_5S, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/api/ai/analyze/pcm", content=PCM_1_5S, headers={"X-API-Key": "secret"}).status_code == 200
