"""FD-016 음성 프로필 테스트. 실제 화자 모델 없이 품질 검사·저장 형식을 검증한다.

TTS 화자 4명으로 한 실측 평가는 docs/models/README.md의 FD-016 평가 결과 참고.
"""

import numpy as np
import pytest

from app import config
from app.detectors import speaker as sp

SR = 16000


def tone(freq=200.0, seconds=10.0, amp=0.1):
    """기본 주파수 freq의 배음 신호 (사람 목소리처럼 F0 측정이 되는 소리)."""
    t = np.arange(int(SR * seconds)) / SR
    w = sum(amp / k * np.sin(2 * np.pi * freq * k * t) for k in range(1, 6))
    return w.astype(np.float32)


class FakeSpeakerModel:
    pass


@pytest.fixture
def fake_pipeline(monkeypatch):
    """VAD는 입력을 그대로 말소리로 보고, 임베딩은 고정 벡터를 돌려준다."""
    monkeypatch.setattr(sp, "speech_only", lambda w: w)
    vec = np.random.default_rng(0).normal(size=sp.EMBEDDING_DIM).astype(np.float32)
    monkeypatch.setattr(sp, "embed", lambda m, s: vec / np.linalg.norm(vec))


# ── 순수 계산 ───────────────────────────────────────────────────
def test_frame_dbfs_of_known_level():
    t = np.arange(SR) / SR
    sine = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # RMS 0.0707 → -23.0 dBFS
    assert np.median(sp.frame_dbfs(sine)) == pytest.approx(-23.0, abs=0.2)


def test_model_version_fits_db_column():
    assert len(sp.MODEL_VERSION) <= 50  # voice_profiles.model_version VARCHAR(50)


# ── 재녹음 안내 (SDS: 녹음이 짧거나 작음) ───────────────────────
def test_no_speech(monkeypatch):
    monkeypatch.setattr(sp, "speech_only", lambda w: np.zeros(0, np.float32))
    with pytest.raises(sp.ProfileError) as e:
        sp.extract_profile(FakeSpeakerModel(), np.zeros(SR * 20, np.float32))
    assert e.value.code == "no_speech"


def test_too_short(fake_pipeline):
    with pytest.raises(sp.ProfileError) as e:
        sp.extract_profile(FakeSpeakerModel(), tone(seconds=config.ENROLL_MIN_SPEECH_SEC - 3))
    assert e.value.code == "too_short"
    assert e.value.info["min_sec"] == config.ENROLL_MIN_SPEECH_SEC
    assert "짧습니다" in str(e.value)


def test_too_quiet(fake_pipeline):
    pytest.importorskip("librosa")
    with pytest.raises(sp.ProfileError) as e:
        sp.extract_profile(FakeSpeakerModel(), tone(seconds=10, amp=0.003))  # 약 -51 dBFS
    assert e.value.code == "too_quiet"


def test_no_pitch_for_unvoiced_noise(fake_pipeline):
    pytest.importorskip("librosa")
    noise = np.random.default_rng(1).normal(0, 0.1, SR * 10).astype(np.float32)  # 속삭임처럼 F0 없는 소리
    with pytest.raises(sp.ProfileError) as e:
        sp.extract_profile(FakeSpeakerModel(), noise)
    assert e.value.code == "no_pitch"


# ── 정상 등록: voice_profiles 저장값 ────────────────────────────
@pytest.mark.parametrize("f0", [120.0, 220.0])  # 남성·여성 평균 F0 근처
def test_profile_matches_voice_profiles_columns(fake_pipeline, f0):
    pytest.importorskip("librosa")
    p = sp.extract_profile(FakeSpeakerModel(), tone(freq=f0, seconds=12))
    assert p["model_version"] == sp.MODEL_VERSION
    assert len(p["embedding"]) == sp.EMBEDDING_DIM
    assert np.linalg.norm(p["embedding"]) == pytest.approx(1.0, abs=1e-4)  # L2 정규화
    assert p["baseline_pitch"] == pytest.approx(f0, rel=0.03)
    assert -40 < p["baseline_volume"] < 0
    assert p["details"]["speech_sec"] == pytest.approx(12.0, abs=0.1)
    assert p["details"]["pitch_p10"] <= p["baseline_pitch"] <= p["details"]["pitch_p90"]


def test_measure_voice_detects_raised_pitch_and_volume():
    # '평소 대비 변화' 계산의 기초: 높고 큰 목소리는 기준값보다 피치·음량이 올라가야 한다
    pytest.importorskip("librosa")
    calm, urgent = sp.measure_voice(tone(150, 5, 0.05)), sp.measure_voice(tone(200, 5, 0.15))
    assert urgent["pitch_hz"] / calm["pitch_hz"] == pytest.approx(200 / 150, rel=0.03)
    assert urgent["volume_dbfs"] - calm["volume_dbfs"] == pytest.approx(9.5, abs=0.5)  # 3배 = +9.5 dB
