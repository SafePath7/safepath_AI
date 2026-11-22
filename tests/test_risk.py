"""FD-018 1단계 위험도 종합 테스트."""

import pytest

from app import config
from app.detectors.audio_detector import empty_sound_result, voice_result
from app.risk import DANGER_PHRASE, combine, rescale


def sound(**cats):
    r = empty_sound_result()
    r["categories"].update(cats)
    return r


def test_rescale_puts_threshold_at_half():
    assert rescale(0.3, 0.3) == pytest.approx(0.5)
    assert rescale(0.0, 0.3) == 0.0 and rescale(1.0, 0.3) == pytest.approx(1.0)
    assert rescale(0.15, 0.3) == pytest.approx(0.25)


@pytest.mark.parametrize("voice_score,sound_score", [(0.0, 0.0), (0.49, 0.0), (0.5, 0.0), (0.0, 0.29), (0.0, 0.3), (1.0, 0.9)])
def test_single_signal_same_decision_as_per_model_thresholds(voice_score, sound_score):
    # 신호가 하나뿐이면 RISK_THRESHOLD 기본값(0.5)에서 모델별 임계값으로 판정한 것과 결과가 같아야 한다
    r = combine(voice_result("ok", "x", voice_score), sound(scream=sound_score))
    want = voice_score >= config.VOICE_THRESHOLD or sound_score >= config.YAMNET_THRESHOLD
    assert r["is_danger"] is want
    assert 0.0 <= r["risk_score"] <= 1.0


def test_two_borderline_signals_together_are_danger():
    # FR-AI-05: 따로는 위험이 아닌 두 신호가 함께 나오면 위험도가 오른다 (noisy-OR)
    r = combine(voice_result("ok", "x", 0.42), sound(scream=0.24))  # 0.42 + 0.4 → 0.65
    assert r["is_danger"] is True and r["risk_type"] == DANGER_PHRASE
    assert r["risk_score"] == pytest.approx(1 - 0.58 * 0.6, abs=1e-4)
    assert [s["type"] for s in r["signals"]] == [DANGER_PHRASE, "scream"]


def test_small_signals_do_not_pile_up():
    # COMBINE_MIN(0.25) 미만의 작은 신호는 가장 큰 신호가 아니면 더하지 않는다
    r = combine(voice_result("ok", "x", 0.2), sound(scream=0.12))  # 0.2, 0.2
    assert r["risk_score"] == pytest.approx(0.2) and r["is_danger"] is False


def test_type_is_strongest_signal():
    r = combine(voice_result("ok", "살려주세요", 1.0), sound(glass_break=0.35))
    assert r["risk_type"] == DANGER_PHRASE
    r = combine(voice_result("ok", "도와줘", 0.8), sound(glass_break=0.95))
    assert r["risk_type"] == "glass_break"


def test_type_is_none_when_safe():
    r = combine(voice_result("ok", "안녕", 0.0), sound(shout=0.2))
    assert r["is_danger"] is False and r["risk_type"] is None and r["risk_score"] > 0


def test_missing_results_use_remaining():
    # SDS FD-018: 일부 결과 누락 → 남은 결과로 계산. 인식 실패(status=error)는 빼고 음향만 본다
    r = combine(voice_result("error"), sound(impact=0.6))
    assert r["risk_type"] == "impact"
    assert combine(None, None) == {"risk_score": 0.0, "risk_type": None, "is_danger": False, "signals": []}


def test_risk_type_fits_event_type_column():
    from app.risk import LOUD_SOUND, MOTION_TYPES, SOUND_TYPES
    assert all(len(t) <= 20 for t in (DANGER_PHRASE, LOUD_SOUND, *SOUND_TYPES, *MOTION_TYPES))  # risk_events.event_type VARCHAR(20)


# ── 화자 검증 가산 ───────────────────────────────────────────────
def speaker(boost, status="ok"):
    from app.detectors.speaker import verify_result
    return verify_result(status, boost=boost)


def test_boost_pushes_borderline_sound_over_threshold():
    # 고함 점수가 임계값 바로 아래(위험도 0.42)인데 본인 목소리가 평소보다 크게 올라갔으면 위험
    sp = speaker(0.15)
    r = combine(voice_result("no_speech"), sound(shout=0.25), sp)
    assert r["is_danger"] is True and r["risk_type"] == "shout"
    assert sp["applied"] == pytest.approx(0.15)


def test_no_boost_without_existing_signal():
    # 평소보다 크게 말하기만 한 것은 위험이 아니다 (VERIFY_BOOST_MIN_BASE 미만에는 더하지 않음)
    sp = speaker(0.15)
    r = combine(voice_result("ok", "야 여기야", 0.0), sound(shout=0.05), sp)
    assert r["is_danger"] is False and sp["applied"] == 0.0


def test_boost_never_lowers_or_exceeds_one():
    sp = speaker(0.15)
    assert combine(voice_result("ok", "살려주세요", 1.0), sound(), sp)["risk_score"] == 1.0
    base = combine(voice_result("ok", "도와줘", 0.8), sound())
    for status in ("off", "loading", "no_profile", "version_mismatch", "no_speech", "error"):
        assert combine(voice_result("ok", "도와줘", 0.8), sound(), speaker(0.0, status)) == base


# ── 갑자기 큰 소리·움직임과 함께 종합 (SOUND_MODE=loudness) ───────────────
def loud(score):
    from app.detectors.loudness import empty_loudness_result
    return {**empty_loudness_result(), "status": "ok", "score": score, "is_danger": score >= 0.5}


def moving(score, kind="fall"):
    from app.detectors.motion import empty_motion_result
    r = empty_motion_result("ok")
    r["scores"][kind] = score
    return {**r, "score": score, "type": kind if score >= 0.5 else None, "is_danger": score >= 0.5}


def test_loudness_decides_sound_and_yamnet_only_names_it():
    # YAMNet 점수가 낮아도 갑자기 큰 소리면 위험, 이름은 YAMNet이 가장 높게 본 소리
    r = combine(voice_result("no_speech"), sound(scream=0.2, impact=0.05), loudness=loud(0.7))
    assert r["is_danger"] is True and r["risk_type"] == "scream"
    r = combine(voice_result("no_speech"), sound(scream=0.05), loudness=loud(0.7))
    assert r["risk_type"] == "loud_sound"
    # YAMNet이 비명이라고 해도 크지 않으면(예: 멀리서 들린 TV) 위험 아님
    r = combine(voice_result("no_speech"), sound(scream=0.9), loudness=loud(0.1))
    assert r["is_danger"] is False


def test_yamnet_mode_keeps_old_behaviour(monkeypatch):
    monkeypatch.setattr(config, "SOUND_MODE", "yamnet")
    r = combine(voice_result("no_speech"), sound(scream=0.9), loudness=loud(0.1))
    assert r["is_danger"] is True and r["risk_type"] == "scream"


def test_loud_sound_and_fall_together():
    r = combine(voice_result("no_speech"), sound(), loudness=loud(0.35), motion=moving(0.35))
    assert r["is_danger"] is True and r["risk_score"] == pytest.approx(1 - 0.65 * 0.65)
    assert {s["type"] for s in r["signals"]} == {"loud_sound", "fall"}


def test_motion_alone_and_missing_motion():
    assert combine(None, None, motion=moving(0.8, "struggle"))["risk_type"] == "struggle"
    from app.detectors.motion import empty_motion_result
    r = combine(voice_result("ok", "살려주세요", 1.0), sound(), loudness=loud(0.1), motion=empty_motion_result("no_data"))
    assert r["risk_type"] == DANGER_PHRASE and "fall" not in {s["type"] for s in r["signals"]}
