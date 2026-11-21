"""음성 프로필 조회·보관 (app/profiles.py)과 백엔드 저장 호출 (app/backend.py). 가짜 백엔드로 확인한다."""

import base64
import json
import time

import pytest

from app import backend, config
from app.profiles import ProfileStore, normalize, save_demo, token_subject
from conftest import BACKEND_PROFILE, PROFILE, parse_multipart


def bearer(uid: str) -> str:
    """서명 없는 가짜 Firebase ID 토큰 (sub만 의미 있음)."""
    payload = base64.urlsafe_b64encode(json.dumps({"sub": uid}).encode()).decode().rstrip("=")
    return f"Bearer h.{payload}.s"


def wait_done(store, session_id, token, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        status, p = store.get(session_id, token)
        if status != "loading":
            return status, p
        time.sleep(0.02)
    raise AssertionError("프로필 조회가 끝나지 않음")


def test_normalize_accepts_extract_result_and_backend_camel_case():
    a = normalize(PROFILE)  # POST /api/ai/voice-profiles 응답을 그대로 저장한 시연 파일
    b = normalize({**BACKEND_PROFILE, "baselineDetails": json.dumps({"pitch_p90": 193.0})})
    assert a == b and a["baseline_details"]["pitch_p90"] == 193.0
    with pytest.raises(ValueError):
        normalize({"embedding": [0.1]})


def test_token_subject():
    assert token_subject(bearer("uid-7")) == "uid-7"
    assert token_subject("Bearer not-a-jwt") is None and token_subject(None) is None


def test_off_without_source_token_or_session(monkeypatch, fake_backend):
    store = ProfileStore()
    assert store.get("s1", None) == ("off", None)          # 토큰 없음
    assert store.get(None, bearer("u")) == ("off", None)   # 세션 없음
    monkeypatch.setattr(config, "BACKEND_URL", None)
    assert store.get("s1", bearer("u")) == ("off", None)   # 백엔드 미설정
    assert fake_backend.calls == []


def test_backend_fetch_once_per_session_with_forwarded_token(fake_backend):
    tok = bearer("u7")
    fake_backend.profiles[tok] = BACKEND_PROFILE
    store = ProfileStore()
    assert store.get("s1", tok)[0] == "loading"  # 첫 요청은 기다리지 않고 검증 없이 분석한다
    status, p = wait_done(store, "s1", tok)
    assert status == "found" and p["baseline_pitch"] == 152.8
    for _ in range(5):  # 같은 세션의 1초마다 오는 요청은 백엔드를 다시 부르지 않는다
        assert store.get("s1", tok)[0] == "found"
    gets = fake_backend.find("GET", "/api/voice-profiles/me/features")
    assert len(gets) == 1 and gets[0]["token"] == tok and gets[0]["key"] == "internal-k"
    assert wait_done(store, "s2", tok)[0] == "found"  # 새 귀가 세션이면 다시 받는다
    assert len(fake_backend.find("GET", "/api/voice-profiles")) == 2


def test_not_registered_and_backend_error(fake_backend):
    store = ProfileStore()
    assert wait_done(store, "s1", bearer("nobody")) == ("no_profile", None)  # 404 = 등록 안 함
    fake_backend.status[("GET", "/api/voice-profiles/me/features")] = 500
    assert wait_done(store, "s2", bearer("u")) == ("no_profile", None)  # 오류 = 잠시 뒤 재시도, 그동안 검증 없이 분석


def test_kept_for_whole_session_and_dropped_when_idle(monkeypatch, tmp_path):
    # 귀가 세션은 10분을 넘길 수 있다: 요청이 계속 오면 시간이 지나도 유지, 요청이 끊기면 지운다
    demo = tmp_path / "demo.json"
    demo.write_text(json.dumps(PROFILE), encoding="utf-8")
    monkeypatch.setattr(config, "BACKEND_URL", None)
    monkeypatch.setattr(config, "DEMO_PROFILE", str(demo))
    monkeypatch.setattr(config, "PROFILE_IDLE_SEC", 600)
    clock = [1000.0]
    monkeypatch.setattr("app.profiles.time.monotonic", lambda: clock[0])
    store = ProfileStore()
    store.get("s1", None)
    end = time.time() + 3
    while store.get("s1", None)[0] == "loading" and time.time() < end:
        time.sleep(0.02)
    for _ in range(60):  # 1분마다 요청이 오며 60분 귀가
        clock[0] += 60
        assert store.get("s1", None)[0] == "found"
    clock[0] += 601  # 세션 요청이 10분 넘게 없음 → 끝난 것으로 보고 지움
    store.get("other", None)
    assert len(store) == 1


def test_save_profile_posts_then_puts_when_already_registered(fake_backend):
    tok = bearer("u7")
    recs = [("file", "calm.m4a", b"CALM", "audio/mp4"), ("urgentFile", "urgent.m4a", b"URGENT", "audio/mp4")]
    assert backend.save_profile(tok, PROFILE, recs) == "created"
    post = fake_backend.find("POST", "/api/voice-profiles")[0]
    assert post["token"] == tok and post["key"] == "internal-k"
    parts = parse_multipart(post["ctype"], post["body"])
    assert json.loads(parts["profile"][1]) == BACKEND_PROFILE
    assert parts["file"] == ("calm.m4a", b"CALM") and parts["urgentFile"] == ("urgent.m4a", b"URGENT")
    assert backend.save_profile(tok, PROFILE, recs) == "updated"  # 두 번째는 409 → PUT /me
    assert len(fake_backend.find("PUT", "/api/voice-profiles/me")) == 1


def test_resave_applies_to_running_session(fake_backend):
    tok = bearer("u7")
    store = ProfileStore()
    assert wait_done(store, "s1", tok) == ("no_profile", None)
    fake_backend.profiles[tok] = BACKEND_PROFILE  # 등록 후
    store.forget(tok)  # /api/ai/voice-profiles 저장 성공 시 호출
    assert wait_done(store, "s1", tok)[0] == "found"


def test_demo_profile_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BACKEND_URL", None)
    monkeypatch.setattr(config, "DEMO_PROFILE", str(tmp_path / "demo.json"))
    store = ProfileStore()
    assert wait_done(store, None, None) == ("no_profile", None)  # 파일 없음
    save_demo(PROFILE)
    store.forget(None)
    status, p = wait_done(store, None, None)
    assert status == "found" and p["baseline_pitch"] == 152.8
