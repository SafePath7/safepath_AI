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


# ── 2단계 등록: 위급할 때 목소리 ────────────────────────────────
@pytest.fixture
def two_voices(monkeypatch):
    """평소 목소리(낮은 피치)와 다급한 목소리(높은 피치)에 서로 다른 임베딩을 준다."""
    rng = np.random.default_rng(5)
    calm_vec, urgent_vec = (v / np.linalg.norm(v) for v in rng.normal(size=(2, sp.EMBEDDING_DIM)))
    monkeypatch.setattr(sp, "speech_only", lambda w: w)
    monkeypatch.setattr(sp, "embed", lambda m, s: urgent_vec if np.abs(s).max() > 0.2 else calm_vec)
    return calm_vec, urgent_vec


def test_urgent_recording_keeps_baseline_from_calm_only(two_voices):
    pytest.importorskip("librosa")
    calm_vec, _ = two_voices
    calm, urgent = tone(150, 12, 0.05), tone(220, 3, 0.4)
    p = sp.extract_profile(FakeSpeakerModel(), calm, urgent, mix_urgent=False)
    assert p["baseline_pitch"] == pytest.approx(150, rel=0.03)  # 다급한 목소리가 기준값을 끌어올리면 안 된다
    assert np.allclose(p["embedding"], calm_vec, atol=1e-5) and p["details"]["embedding_source"] == "calm"
    u = p["details"]["urgent"]
    assert u["pitch_ratio"] == pytest.approx(220 / 150, rel=0.03) and u["volume_delta_db"] > 15
    assert p["model_version"] == sp.MODEL_VERSION  # 같은 ECAPA 공간이라 버전은 그대로


def test_urgent_recording_mixed_into_embedding(two_voices):
    pytest.importorskip("librosa")
    calm_vec, urgent_vec = two_voices
    p = sp.extract_profile(FakeSpeakerModel(), tone(150, 12, 0.05), tone(220, 3, 0.4), mix_urgent=True)
    e = np.array(p["embedding"])
    assert np.linalg.norm(e) == pytest.approx(1.0, abs=1e-4)
    assert sp.cosine(e, calm_vec) > 0.6 and sp.cosine(e, urgent_vec) > 0.6  # 두 목소리 모두에 가까워진다
    assert p["details"]["embedding_source"] == "calm+urgent"


def test_urgent_recording_too_short(two_voices):
    with pytest.raises(sp.ProfileError) as e:
        sp.extract_profile(FakeSpeakerModel(), tone(150, 12, 0.05), tone(220, 0.5, 0.4))
    assert e.value.code == "urgent_too_short" and "다급하게" in str(e.value)


# ── 화자 검증 (FD-016 결합) ──────────────────────────────────────
def enrolled(f0=150.0, vec=None):
    vec = np.random.default_rng(0).normal(size=sp.EMBEDDING_DIM) if vec is None else vec
    vec = vec / np.linalg.norm(vec)
    return {"model_version": sp.MODEL_VERSION, "speaker_embedding": vec.tolist(),
            "baseline_volume": -26.0, "baseline_pitch": f0,
            "baseline_details": {"volume_p90": -22.0, "pitch_p90": f0 * 1.1}}


def test_verify_same_speaker_raised_voice_gives_boost(fake_pipeline):
    pytest.importorskip("librosa")
    r = sp.verify_voice(FakeSpeakerModel(), tone(freq=210, seconds=3, amp=0.3), enrolled(150))  # 피치 x1.4, 크게
    assert r["status"] == "ok" and r["is_user"] is True and r["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert r["pitch_ratio"] == pytest.approx(1.4, rel=0.03)
    assert r["agitation"] == 1.0 and r["boost"] == pytest.approx(config.VERIFY_BOOST_MAX)


def test_verify_calm_voice_gives_no_boost(fake_pipeline):
    pytest.importorskip("librosa")
    r = sp.verify_voice(FakeSpeakerModel(), tone(freq=150, seconds=3, amp=0.05), enrolled(150))
    assert r["is_user"] is True and r["agitation"] == 0.0 and r["boost"] == 0.0


def test_verify_other_speaker_gives_no_boost(fake_pipeline):
    pytest.importorskip("librosa")
    other = np.random.default_rng(99).normal(size=sp.EMBEDDING_DIM)  # 무작위 벡터끼리는 코사인 약 0
    r = sp.verify_voice(FakeSpeakerModel(), tone(freq=210, seconds=3, amp=0.3), enrolled(150, other))
    assert r["is_user"] is False and r["boost"] == 0.0 and r["agitation"] == 1.0


def test_verify_skips_on_version_mismatch_and_short_speech(fake_pipeline):
    old = {**enrolled(), "model_version": "fd016-v0/old"}
    assert sp.verify_voice(FakeSpeakerModel(), tone(seconds=3), old)["status"] == "version_mismatch"
    assert sp.verify_voice(FakeSpeakerModel(), tone(seconds=0.5), enrolled())["status"] == "no_speech"


def test_agitation_without_details_uses_fixed_margin():
    p = {**enrolled(150), "baseline_details": None}
    _, _, calm = sp.agitation_level({"pitch_hz": 160.0, "volume_dbfs": -24.0}, p)   # x1.07, +2dB
    _, _, loud = sp.agitation_level({"pitch_hz": 150.0, "volume_dbfs": -14.0}, p)   # +12dB
    assert calm == 0.0 and loud == 1.0
