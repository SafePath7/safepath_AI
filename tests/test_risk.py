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


@pytest.mark.parametrize("voice_score,sound_score", [(0.0, 0.0), (0.49, 0.29), (0.5, 0.0), (0.0, 0.3), (1.0, 0.9)])
def test_same_decision_as_per_model_thresholds(voice_score, sound_score):
    # RISK_THRESHOLD 기본값(0.5)에서는 모델별 임계값으로 판정한 것과 결과가 같아야 한다
    r = combine(voice_result("ok", "x", voice_score), sound(scream=sound_score))
    want = voice_score >= config.VOICE_THRESHOLD or sound_score >= config.YAMNET_THRESHOLD
    assert r["is_danger"] is want
    assert 0.0 <= r["risk_score"] <= 1.0


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
    assert combine(None, None) == {"risk_score": 0.0, "risk_type": None, "is_danger": False}


def test_risk_type_fits_event_type_column():
    from app.risk import SOUND_TYPES
    assert all(len(t) <= 20 for t in (DANGER_PHRASE, *SOUND_TYPES))  # risk_events.event_type VARCHAR(20)
