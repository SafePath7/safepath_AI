"""위험 이벤트 보고 (app/events.py): 위험이면 createEvent → uploadEventMedia, 30초 안의 반복 감지는 묶는다."""

import json
import time

from app import config
from app.events import EventReporter, describe, iso_utc
from conftest import parse_multipart

TOKEN = "Bearer test-token"
META = {"session_id": "11111111-2222-3333-4444-555555555555", "seq": 41, "captured_at": 1763280003000}


def danger(kind="danger_phrase", score=0.82, text="살려주세요"):
    return {"risk_score": score, "risk_type": kind, "is_danger": True,
            "whisper": {"text": text, "score": 0.9, "matches": ["살려주세요"]},
            "yamnet": {"score": 0.41, "categories": {"scream": 0.41}},
            "speaker": {"status": "ok", "is_user": True, "similarity": 0.6, "agitation": 0.7, "applied": 0.1}}


SAFE = {"risk_score": 0.1, "risk_type": None, "is_danger": False}


def wait_uploads(fb, n, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        ups = fb.find("POST", "/api/risk-events/")
        if len(ups) >= n:
            return ups
        time.sleep(0.02)
    raise AssertionError(f"미디어 업로드 {n}건이 오지 않음: {len(fb.find('POST', '/api/risk-events/'))}")


def test_iso_and_description():
    assert iso_utc(1763280003000) == "2025-11-16T08:00:03.000Z"
    assert describe(danger()) == "위험 문구 감지: '살려주세요' (위험도 0.82)"
    assert describe(danger("scream", 0.7)) == "비명 감지 (위험도 0.70)"


def test_off_without_backend_token_or_session(monkeypatch, fake_backend):
    r = EventReporter()
    assert r.report(danger(), META, None, None) == (None, "off")
    assert r.report(danger(), {**META, "session_id": None}, TOKEN, None) == (None, "off")
    monkeypatch.setattr(config, "BACKEND_URL", None)
    assert r.report(danger(), META, TOKEN, None) == (None, "off")
    assert fake_backend.calls == []


def test_safe_result_does_not_call_backend(fake_backend):
    assert EventReporter().report(SAFE, META, TOKEN, None) == (None, "none")
    assert fake_backend.calls == []


def test_danger_creates_event_and_uploads_audio_and_sensor(fake_backend):
    motion = {"rate_hz": 20, "samples": [[0, 0.1, 9.8, 0.2, 0.0, 0.1, 0.0]]}
    location = {"lat": 37.5, "lng": 127.0, "accuracy_m": 12.0, "location_at": 1763280002500}
    event_id, status = EventReporter().report(danger(), META, TOKEN, ("voice_1763280003000.wav", b"RIFFWAV"),
                                              motion, location)
    assert status == "created" and event_id == 101
    create = fake_backend.find("POST", "/api/risk-events")[0]
    assert create["path"] == "/api/risk-events" and create["token"] == TOKEN and create["key"] == "internal-k"
    assert json.loads(create["body"]) == {
        "sessionId": META["session_id"], "eventType": "danger_phrase",
        "description": "위험 문구 감지: '살려주세요' (위험도 0.82)", "detectedAt": "2025-11-16T08:00:03.000Z"}

    ups = wait_uploads(fake_backend, 2)
    assert all(u["path"] == "/api/risk-events/101/medias" and u["token"] == TOKEN for u in ups)
    parts = [parse_multipart(u["ctype"], u["body"]) for u in ups]
    by_type = {p["mediaType"][1].decode(): p["file"] for p in parts}
    assert by_type["audio"] == ("voice_1763280003000.wav", b"RIFFWAV")
    report = json.loads(by_type["sensor"][1])
    assert report["riskScore"] == 0.82 and report["voice"]["text"] == "살려주세요"
    assert report["location"] == location and report["motion"] == motion and report["seq"] == 41


def test_repeated_danger_in_same_session_is_grouped(monkeypatch, fake_backend):
    # 1초마다 겹치는 3초 창이라 한 번의 외침이 여러 창에서 위험으로 나온다 → 확인 질문(이벤트)은 하나
    clock = [1000.0]
    monkeypatch.setattr("app.events.time.monotonic", lambda: clock[0])
    r = EventReporter()
    first = r.report(danger(), META, TOKEN, None)
    clock[0] += 1
    assert r.report(danger("scream"), META, TOKEN, None) == (first[0], "grouped")
    other = r.report(danger(), {**META, "session_id": "other"}, TOKEN, None)  # 다른 세션은 따로
    assert other[1] == "created" and other[0] != first[0]
    clock[0] += config.EVENT_GROUP_SEC + 1  # 확인 대기 시간이 지나면 새 이벤트
    assert r.report(danger(), META, TOKEN, None)[1] == "created"
    assert len(fake_backend.find("POST", "/api/risk-events")) - len(fake_backend.find("POST", "/api/risk-events/")) == 3


def test_create_failure_returns_failed(fake_backend):
    fake_backend.status[("POST", "/api/risk-events")] = 500
    r = EventReporter()
    assert r.report(danger(), META, TOKEN, None) == (None, "failed")
    del fake_backend.status[("POST", "/api/risk-events")]
    assert r.report(danger(), META, TOKEN, None)[1] == "created"  # 실패는 묶지 않고 다음 감지 때 다시 만든다


def test_suspicion_period_keeps_collecting_every_three_seconds(monkeypatch, fake_backend):
    # 의심 단계(확인 대기 중)에는 위험이 아닌 창도 3초마다 겹치지 않게 올린다. 실제 사고의 근거 자료
    clock = [1000.0]
    monkeypatch.setattr("app.events.time.monotonic", lambda: clock[0])
    r = EventReporter()
    t0 = META["captured_at"]
    assert r.report(danger(), META, TOKEN, (f"voice_{t0}.wav", b"A0"))[1] == "created"
    sent = [(SAFE, 1000), (danger("scream"), 2000), (SAFE, 3010), (SAFE, 4000), (SAFE, 5000), (SAFE, 5980)]
    results = []
    for res, dt in sent:
        clock[0] += 1
        meta = {**META, "captured_at": t0 + dt}
        results.append(r.report(res, meta, TOKEN, (f"voice_{t0 + dt}.wav", b"A"), {"rate_hz": 20, "samples": []}))
    # 응답은 그대로: 위험 창만 이벤트 id, 안전한 창은 none (폰이 확인 질문을 다시 띄우지 않게)
    assert results[0] == (None, "none") and results[1] == (101, "grouped") and results[2] == (None, "none")
    ups = wait_uploads(fake_backend, 6)
    time.sleep(0.2)
    audio = sorted(parse_multipart(u["ctype"], u["body"])["file"][0]
                   for u in fake_backend.find("POST", "/api/risk-events/101/medias")
                   if parse_multipart(u["ctype"], u["body"])["mediaType"][1] == b"audio")
    assert audio == [f"voice_{t0}.wav", f"voice_{t0 + 3010}.wav", f"voice_{t0 + 5980}.wav"]
    sensors = [parse_multipart(u["ctype"], u["body"])["file"][0] for u in ups
               if parse_multipart(u["ctype"], u["body"])["mediaType"][1] == b"sensor"]
    assert f"event_101_{t0 + 3010}.json" in sensors
    clock[0] += config.EVENT_GROUP_SEC  # 확인 대기 시간이 지나면 안전한 창은 더 올리지 않는다
    n = len(fake_backend.calls)
    assert r.report(SAFE, {**META, "captured_at": t0 + 40000}, TOKEN, ("x.wav", b"A")) == (None, "none")
    time.sleep(0.2)
    assert len(fake_backend.calls) == n


# ---- 원본 이어 붙이기: 폰처럼 들쭉날쭉한 조각을 이어 붙인 '최근 3초' 창을 보낸다 ----

import io as _io
import random
import wave as _wave

from app.events import StreamStitcher, overlap_bytes

SR = 16000


def wav_bytes(pcm: bytes) -> bytes:
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR), w.writeframes(pcm)
    return buf.getvalue()


def wav_pcm(data: bytes) -> bytes:
    with _wave.open(_io.BytesIO(data), "rb") as w:
        return w.readframes(w.getnframes())


def phone_windows(seconds: float, seed=1):
    """(seq, captured_at, 창 WAV, 지금까지 녹음 길이 바이트). 조각 0.5~1.4초, 창 = 이어 붙인 소리의 최근 3초."""
    rnd = random.Random(seed)
    stream = bytes(rnd.getrandbits(8) for _ in range(int(seconds * SR) * 2))
    out, pos, seq, t = [], 0, 0, 1763280000000
    while pos < len(stream):
        n = int(rnd.uniform(0.5, 1.4) * SR) * 2
        pos = min(pos + n, len(stream))
        t += n // 32
        window = stream[max(0, pos - 3 * SR * 2):pos]
        out.append((seq, t, wav_bytes(window), pos))
        seq += 1
    return stream, out


def uploaded_files(fb, last="event_101_audio.json", timeout=10.0):
    """이벤트 101에 올라온 파일 {이름: 내용}. 마지막 파일(녹음 정보)이 올 때까지 기다린다."""
    end = time.time() + timeout
    while True:
        files = {}
        for u in fb.find("POST", "/api/risk-events/101/medias"):
            p = parse_multipart(u["ctype"], u["body"])
            files[p["file"][0]] = p["file"][1]
        if last in files or time.time() > end:
            time.sleep(0.3)  # 같은 시점의 다른 업로드 스레드도 마무리
            return {parse_multipart(u["ctype"], u["body"])["file"][0]: parse_multipart(u["ctype"], u["body"])["file"][1]
                    for u in fb.find("POST", "/api/risk-events/101/medias")}
        time.sleep(0.05)


def test_overlap_and_stitcher_rebuild_original():
    stream, wins = phone_windows(20)
    st = StreamStitcher()
    got = bytearray()
    for seq, _, w, _ in wins:
        new, gap = st.feed(seq, wav_pcm(w), SR)
        assert not gap
        got += new
    assert bytes(got) == stream
    assert overlap_bytes(b"", b"x" * 4000) == 0
    assert st.feed(wins[-2][0], wav_pcm(wins[-2][2]), SR) is None  # 늦게 온 지난 창은 무시


def test_event_uploads_preroll_and_continuous_audio(monkeypatch, fake_backend):
    monkeypatch.setattr(config, "EVENT_PREROLL_SEC", 10)
    stream, wins = phone_windows(45, seed=7)
    r = EventReporter()
    trigger = next(i for i, w in enumerate(wins) if w[3] >= 20 * SR * 2)  # 약 20초에 위험 감지
    order = list(range(len(wins)))
    order[trigger + 3], order[trigger + 4] = order[trigger + 4], order[trigger + 3]  # 요청 순서 뒤바뀜
    del order[trigger + 8]  # 요청 하나 빠짐 (다음 창에 그 소리가 들어 있음)
    last_pos = 0
    for i in order:
        seq, cap, w, pos = wins[i]
        res = danger() if i == trigger else SAFE
        r.report(res, {**META, "seq": seq, "captured_at": cap}, TOKEN, (f"voice_{cap}.wav", w))
        last_pos = max(last_pos, pos)
        if i == trigger:
            assert fake_backend.find("POST", "/api/risk-events")  # 위험 전에는 백엔드를 부르지 않음
            start = wins[trigger][3] - 10 * SR * 2
    r.finish(META["session_id"], 101)
    files = uploaded_files(fake_backend)
    full = wav_pcm(files["event_101_full.wav"])
    assert full == stream[start:last_pos]  # 직전 10초부터 끝까지 원본과 같음
    info = json.loads(files["event_101_audio.json"])
    assert info["gapsAtSec"] == [] and info["prerollSec"] == 10
    assert b"".join(wav_pcm(files[n]) for n in info["parts"]) == full  # 조각을 순서대로 붙이면 전체
    assert len(info["parts"]) >= 5 and all(n in files for n in info["parts"])
    assert sum(1 for n in files if n.endswith(".json") and n != "event_101_audio.json") == len(info["parts"]) - 1


def test_long_dropout_is_recorded_as_gap(monkeypatch, fake_backend):
    stream, wins = phone_windows(30, seed=3)
    r = EventReporter()
    for i, (seq, cap, w, pos) in enumerate(wins):
        if 5 <= i <= 9:
            continue  # 3초 넘게 요청이 안 옴
        r.report(danger() if i == 3 else SAFE, {**META, "seq": seq, "captured_at": cap}, TOKEN, ("v.wav", w))
    r.finish(META["session_id"], 101)
    files = uploaded_files(fake_backend)
    assert len(json.loads(files["event_101_audio.json"])["gapsAtSec"]) == 1
