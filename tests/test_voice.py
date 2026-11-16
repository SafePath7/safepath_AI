"""FD-014 위험 음성 감지 테스트. 실제 Whisper 없이 문구 매칭과 처리 흐름(무음·인식 실패)을 검증한다.

한국어 실제 음성(TTS 4명·소음·청크 조건) 평가는 docs/models/README.md의 FD-014 평가 결과 참고.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from app import config
from app.detectors import audio_detector as det
from app.detectors.phrase import compact_hangul, match_phrase, score_danger_phrase, to_jamo

SR = 16000


# ── 문구 매칭 ───────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "살려주세요", "살려 주세요!", "아저씨 왜 이러세요, 살려주세요.", "누가 좀 도와주세요",
    "놔주세요 이거 놔주세요", "살려줘 제발", "놔줘",
])
def test_exact_danger_phrases_score_1(text):
    score, matches = score_danger_phrase(text)
    assert score == 1.0
    assert matches[0]["exact"] is True


@pytest.mark.parametrize("heard, phrase", [
    # Whisper base가 실제로 이렇게 받아 적은 사례 (한국어 TTS 평가에서 수집)
    ("나 주세요, 이거 나 주세요", "놔주세요"),
    ("도와주세여", "도와주세요"),
    ("살료주세요", "살려주세요"),
])
def test_one_jamo_misrecognition_is_matched(heard, phrase):
    score, matches = score_danger_phrase(heard)
    assert matches[0]["phrase"] == phrase
    assert matches[0]["exact"] is False
    assert score == pytest.approx(0.9)  # 정확히 일치하지 않으면 0.9배
    assert score >= config.VOICE_THRESHOLD


@pytest.mark.parametrize("text", [
    "안녕하세요 오늘 저녁에 친구랑 밥 먹고 집에 가는 길이에요",
    "엄마 나 지금 버스에서 내렸어 금방 갈게",
    "메뉴판 좀 보여주세요",          # 도와주세요와 2개 차이
    "한 번만 봐주세요",              # 놔주세요와 자음(초성) 차이
    "여기로 와주세요",               # 놔주세요와 자음(초성) 차이
    "내가 도와줄게 걱정하지 마",      # 3음절 문구는 정확히 일치할 때만
    "이제 그만 집에 가자",
    "도와주셔서 정말 감사합니다",
    "살려고 매일 운동하는 중이야",
    "그만두고 싶다는 생각이 들어",
    "택시 좀 잡아 주세요",
    "",
])
def test_everyday_speech_is_not_danger(text):
    score, matches = score_danger_phrase(text)
    assert score == 0.0, matches


def test_everyday_uses_of_danger_words_get_lower_weight():
    # 알려진 한계: 키워드가 그대로 들어간 일상 표현은 위험으로 판정된다 (FD-018 종합·FD-019 확인 질문에서 거른다)
    assert score_danger_phrase("이 숙제 좀 도와줘")[0] == pytest.approx(0.8)
    assert score_danger_phrase("장난 그만해 진짜")[0] == pytest.approx(0.7)
    assert score_danger_phrase("살려주세요 그만해")[0] == 1.0  # 여러 개면 최댓값


def test_matches_sorted_by_score():
    _, matches = score_danger_phrase("그만해 도와주세요")
    assert [m["phrase"] for m in matches] == ["도와주세요", "그만해"]


def test_jamo_helpers():
    assert compact_hangul("살려 주세요! Help 119") == "살려주세요"
    assert to_jamo("한") == (18, 0, 4) and to_jamo("놔")[2] == 0  # ㅎ·ㅏ·ㄴ / 받침 없음
    assert match_phrase("나주세요", "놔주세요") == (1, "나주세요")
    assert match_phrase("봐주세요", "놔주세요") is None
    assert match_phrase("보여주세요", "도와주세요") is None


# ── 처리 흐름: 무음 생략·인식 실패 ──────────────────────────────
class FakeWhisper:
    def __init__(self, text="", error=None):
        self.text, self.error, self.calls = text, error, []

    def transcribe(self, waveform, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return iter([SimpleNamespace(text=f" {self.text} ")]), None


def speech_like(seconds=2.0):
    t = np.arange(int(SR * seconds)) / SR
    return (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


@pytest.fixture
def speech_present(monkeypatch):
    monkeypatch.setattr(det, "has_speech", lambda w: True)


def test_silence_skips_whisper():
    model = FakeWhisper("살려주세요")
    r = det.analyze_voice(model, np.zeros(SR * 5, np.float32))
    assert r["status"] == "silence" and r["score"] == 0.0 and r["is_danger"] is False
    assert model.calls == []  # 무음 → 분석 생략


def test_very_quiet_noise_is_silence():
    quiet = (np.random.default_rng(0).normal(0, 1e-4, SR * 5)).astype(np.float32)  # 약 -80 dBFS
    assert det.analyze_voice(FakeWhisper(), quiet)["status"] == "silence"


def test_no_speech_skips_whisper(monkeypatch):
    monkeypatch.setattr(det, "has_speech", lambda w: False)
    model = FakeWhisper("감사합니다")  # 잡음에서 Whisper가 지어내는 대표적인 문장
    r = det.analyze_voice(model, speech_like())
    assert r["status"] == "no_speech" and r["text"] == "" and r["is_danger"] is False
    assert model.calls == []


def test_recognition_failure_scores_zero(speech_present):
    r = det.analyze_voice(FakeWhisper(error=RuntimeError("decode failed")), speech_like())
    assert r == {"status": "error", "text": "", "score": 0.0, "is_danger": False, "matches": []}


def test_vad_failure_also_scores_zero(monkeypatch):
    def boom(w):
        raise RuntimeError("vad failed")

    monkeypatch.setattr(det, "has_speech", boom)
    assert det.analyze_voice(FakeWhisper(), speech_like())["status"] == "error"


def test_danger_speech(speech_present):
    r = det.analyze_voice(FakeWhisper("아저씨 왜 이러세요 살려 주세요"), speech_like())
    assert r["status"] == "ok" and r["score"] == 1.0 and r["is_danger"] is True
    assert r["text"] == "아저씨 왜 이러세요 살려 주세요"
    assert r["matches"][0]["phrase"] == "살려주세요"


def test_normal_speech(speech_present):
    r = det.analyze_voice(FakeWhisper("엄마 나 지금 버스에서 내렸어"), speech_like())
    assert r["status"] == "ok" and r["score"] == 0.0 and r["is_danger"] is False


def test_transcribe_options(speech_present, monkeypatch):
    model = FakeWhisper("안녕하세요")
    det.analyze_voice(model, speech_like())
    assert model.calls[0]["language"] == "ko"
    assert model.calls[0]["condition_on_previous_text"] is False
    assert model.calls[0]["hotwords"] is None

    monkeypatch.setattr(config, "WHISPER_HOTWORDS", True)
    det.analyze_voice(model, speech_like())
    assert "살려주세요" in model.calls[1]["hotwords"]


def test_voice_threshold_is_configurable(speech_present, monkeypatch):
    monkeypatch.setattr(config, "VOICE_THRESHOLD", 0.75)
    assert det.analyze_voice(FakeWhisper("장난 그만해"), speech_like())["is_danger"] is False  # 0.7 < 0.75
    assert det.analyze_voice(FakeWhisper("이것 좀 도와줘"), speech_like())["is_danger"] is True  # 0.8
