"""
사용자 음성 개인화 (FD-016, M-06): 등록 녹음에서 화자 임베딩과 개인 기준값(평소 음량·피치)을 추출한다.

결과는 백엔드 voice_profiles 테이블 컬럼에 그대로 들어간다.
    speaker_embedding REAL[]       ← embedding (192차원, L2 정규화)
    baseline_volume   DOUBLE       ← 말소리 구간 음량 중앙값 (dBFS)
    baseline_pitch    DOUBLE       ← 말소리 구간 피치(F0) 중앙값 (Hz)
    model_version     VARCHAR(50)  ← MODEL_VERSION
model_version이 바뀌면 이전 임베딩과 비교할 수 없으므로, 백엔드는 보관한 원본(audio_file_url)으로 다시 추출한다.

이 모듈은 아무것도 저장하지 않는다. 녹음은 요청 처리 중 메모리에만 있다.
"""

import logging

import numpy as np

from app import config

log = logging.getLogger("safepath.ai.speaker")

SR = 16000
EMBEDDING_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
EMBEDDING_DIM = 192
# 임베딩 모델이나 기준값 계산 방식이 바뀌면 올린다 (VARCHAR(50) 이내)
MODEL_VERSION = "fd016-v1/ecapa-voxceleb-192"

PITCH_FMIN, PITCH_FMAX = 65.0, 500.0  # 성인 말소리 F0 범위 (비명은 이보다 높을 수 있으나 기준값은 평소 목소리)
FRAME = 1024                           # 64ms
HOP = 256                              # 16ms


class ProfileError(ValueError):
    """재녹음이 필요한 등록 녹음 (SDS FD-016 예외 처리: 녹음이 짧거나 작음 → 재녹음 안내)."""

    MESSAGES = {
        "no_speech": "말소리가 들리지 않습니다. 조용한 곳에서 다시 녹음해 주세요.",
        "too_short": "말소리가 너무 짧습니다. 안내 문장을 끝까지 읽어 주세요.",
        "too_quiet": "목소리가 너무 작습니다. 휴대폰을 입에 가까이 대고 다시 녹음해 주세요.",
        "no_pitch": "목소리 높낮이를 측정하지 못했습니다. 속삭이지 말고 평소 목소리로 다시 녹음해 주세요.",
        "urgent_too_short": "위급할 때 목소리가 거의 들리지 않습니다. 실제 상황처럼 크고 다급하게 다시 외쳐 주세요.",
    }

    def __init__(self, code: str, **info):
        super().__init__(self.MESSAGES[code])
        self.code, self.info = code, info


# ── 모델 ────────────────────────────────────────────────────────
def load_speaker_model():
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    # Windows는 관리자 권한 없이 심볼릭 링크를 만들 수 없어 COPY로 받는다.
    return EncoderClassifier.from_hparams(
        source=EMBEDDING_SOURCE,
        savedir=str(config.MODEL_CACHE_DIR / "speechbrain" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )


def embed(speaker_model, speech: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        e = speaker_model.encode_batch(torch.from_numpy(np.ascontiguousarray(speech, np.float32))[None])
    e = e.reshape(-1).cpu().numpy().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-12)


# ── 말소리 구간과 기준값 ────────────────────────────────────────
def speech_only(waveform: np.ndarray) -> np.ndarray:
    """Silero VAD로 말소리 구간만 이어 붙인다. 쉼·무음이 기준값을 끌어내리지 않게 한다."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    spans = get_speech_timestamps(waveform, VadOptions(min_speech_duration_ms=250, min_silence_duration_ms=300))
    if not spans:
        return np.zeros(0, np.float32)
    return np.concatenate([waveform[s["start"]:s["end"]] for s in spans]).astype(np.float32)


def frame_dbfs(speech: np.ndarray) -> np.ndarray:
    n = 1 + max(0, len(speech) - FRAME) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(speech, FRAME)[::HOP][:n] if len(speech) >= FRAME else speech[None]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10))


def voiced_pitch(speech: np.ndarray) -> np.ndarray:
    """pYIN으로 유성음 프레임의 F0(Hz)만 돌려준다."""
    import librosa

    f0, voiced, _ = librosa.pyin(speech, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=SR,
                                 frame_length=FRAME * 2, hop_length=HOP * 2)
    return f0[voiced & np.isfinite(f0)]


def measure_voice(speech: np.ndarray) -> dict:
    """말소리 구간의 음량·피치 분포. 기준값 등록과 이후 '평소 대비 변화' 계산에 같은 함수를 쓴다."""
    db = frame_dbfs(speech)
    db = db[db > db.max() - 40]  # 단어 사이 숨소리 같은 아주 작은 프레임 제외
    f0 = voiced_pitch(speech)
    return {
        "volume_dbfs": float(np.median(db)),
        "volume_p10": float(np.percentile(db, 10)),
        "volume_p90": float(np.percentile(db, 90)),
        "pitch_hz": float(np.median(f0)) if f0.size else None,
        "pitch_p10": float(np.percentile(f0, 10)) if f0.size else None,
        "pitch_p90": float(np.percentile(f0, 90)) if f0.size else None,
        "voiced_frames": int(f0.size),
    }


# ── 등록 ────────────────────────────────────────────────────────
def extract_profile(speaker_model, waveform: np.ndarray, urgent: np.ndarray | None = None,
                    mix_urgent: bool | None = None) -> dict:
    """등록 녹음 → voice_profiles 저장값. 재녹음이 필요하면 ProfileError.

    waveform: 1단계 평소 목소리(20~30초). 평소 음량·피치 기준값은 이것으로만 잰다.
    urgent:   2단계 위급할 때 목소리(선택, "살려주세요" 등을 크고 다급하게 3~5번).
              다급한 목소리는 본인이어도 평소 목소리와 임베딩 유사도가 떨어진다(FD-016 평가 0.77 → 0.53).
              mix_urgent(기본 config.ENROLL_MIX_URGENT)가 켜져 있으면 두 임베딩을 합쳐 저장해 이 하락을 줄인다.
              같은 ECAPA 공간의 벡터라 model_version은 그대로다. 어떤 녹음으로 만들었는지는 details.embedding_source.
              꺼져 있어도 details.urgent에 평소 대비 변화를 기록한다(사람마다 다른 '다급할 때 변화폭' 참고값).
    """
    if mix_urgent is None:
        mix_urgent = config.ENROLL_MIX_URGENT
    speech = speech_only(waveform)
    speech_sec = len(speech) / SR
    if speech_sec == 0:
        raise ProfileError("no_speech")
    if speech_sec < config.ENROLL_MIN_SPEECH_SEC:
        raise ProfileError("too_short", speech_sec=round(speech_sec, 1), min_sec=config.ENROLL_MIN_SPEECH_SEC)

    v = measure_voice(speech)
    if v["volume_dbfs"] < config.ENROLL_MIN_DBFS:
        raise ProfileError("too_quiet", volume_dbfs=round(v["volume_dbfs"], 1), min_dbfs=config.ENROLL_MIN_DBFS)
    if v["pitch_hz"] is None or v["voiced_frames"] < 50:  # 유성음 약 1.6초 미만
        raise ProfileError("no_pitch")

    embedding = embed(speaker_model, speech)
    details = {
        "speech_sec": round(speech_sec, 1),
        "total_sec": round(len(waveform) / SR, 1),
        "volume_p10": round(v["volume_p10"], 2),
        "volume_p90": round(v["volume_p90"], 2),
        "pitch_p10": round(v["pitch_p10"], 1),
        "pitch_p90": round(v["pitch_p90"], 1),
        "embedding_source": "calm",
    }

    if urgent is not None:
        u_speech = speech_only(urgent)
        u_sec = len(u_speech) / SR
        if u_sec < config.ENROLL_URGENT_MIN_SPEECH_SEC:
            raise ProfileError("urgent_too_short", speech_sec=round(u_sec, 1),
                               min_sec=config.ENROLL_URGENT_MIN_SPEECH_SEC)
        u_emb = embed(speaker_model, u_speech)
        u = measure_voice(u_speech)
        details["urgent"] = {
            "speech_sec": round(u_sec, 1),
            "similarity_to_calm": round(cosine(embedding, u_emb), 4),
            "pitch_ratio": round(u["pitch_hz"] / v["pitch_hz"], 3) if u["pitch_hz"] else None,
            "volume_delta_db": round(u["volume_dbfs"] - v["volume_dbfs"], 2),
        }
        if mix_urgent:
            mixed = embedding + u_emb
            embedding = mixed / (np.linalg.norm(mixed) + 1e-12)
            details["embedding_source"] = "calm+urgent"

    return {
        "model_version": MODEL_VERSION,
        "embedding": [round(float(x), 6) for x in embedding],
        "baseline_volume": round(v["volume_dbfs"], 2),
        "baseline_pitch": round(v["pitch_hz"], 1),
        # DB 컬럼 밖의 참고값: 평소 변동 폭과 2단계 녹음 결과. baseline_details(JSONB)에 저장을 제안.
        "details": details,
    }


# ── 화자 검증 (분석 중, 등록 프로필이 있을 때) ───────────────────
def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def verify_result(status: str, **values) -> dict:
    """화자 검증 결과 형식. status가 ok가 아니면 점수에 반영하지 않는다(boost 0)."""
    base = {"status": status, "similarity": None, "is_user": None, "pitch_ratio": None,
            "volume_delta_db": None, "agitation": None, "boost": 0.0, "applied": 0.0}
    return {**base, **values}


def agitation_level(m: dict, profile: dict) -> tuple[float | None, float | None, float]:
    """평소 목소리 대비 높아진 정도 0~1. (피치 비율, 음량 차이 dB, 흥분도)

    피치: 평소 중앙값 대비 비율. 평소 변동 상단(pitch_p90)까지는 0, 그 위로 VERIFY_PITCH_SPAN만큼 오르면 1.
    음량: 평소 중앙값 대비 dB. 평소 변동 상단(volume_p90)까지는 0, 그 위로 VERIFY_VOLUME_SPAN_DB만큼 커지면 1.
    마이크 거리에 따라 음량은 쉽게 변하므로 둘 중 큰 값을 쓰되 각각 상한 1로 자른다.
    details(평소 변동 폭)가 없으면 고정 여유(피치 x1.1, 음량 +3dB)를 쓴다.
    """
    details = profile.get("baseline_details") or {}
    base_pitch, base_vol = profile["baseline_pitch"], profile["baseline_volume"]
    ratio = m["pitch_hz"] / base_pitch if m["pitch_hz"] and base_pitch else None
    delta = m["volume_dbfs"] - base_vol
    pitch_top = (details.get("pitch_p90") or base_pitch * 1.1) / base_pitch
    vol_top = (details.get("volume_p90") or base_vol + 3.0) - base_vol
    a_pitch = 0.0 if ratio is None else (ratio - pitch_top) / config.VERIFY_PITCH_SPAN
    a_vol = (delta - vol_top) / config.VERIFY_VOLUME_SPAN_DB
    level = float(np.clip(max(a_pitch, a_vol), 0.0, 1.0))
    return ratio, delta, level


def verify_voice(speaker_model, waveform: np.ndarray, profile: dict) -> dict:
    """등록한 본인 목소리인지, 평소보다 높고 큰지 본다. 점수는 더하기만 한다(boost).

    본인 판정이 틀려도 위험 감지가 막히지 않게, 결과는 위험도를 올리는 데만 쓴다.
    (FD-016 평가: 다급한 목소리는 본인이어도 유사도가 0.5 안팎으로 떨어져 타인과 겹친다.)
    """
    if profile.get("model_version") != MODEL_VERSION:
        return verify_result("version_mismatch")
    speech = speech_only(waveform)
    if len(speech) < config.VERIFY_MIN_SPEECH_SEC * SR:
        return verify_result("no_speech")
    sim = cosine(embed(speaker_model, speech), profile["speaker_embedding"])
    is_user = sim >= config.VERIFY_SIM_THRESHOLD
    m = measure_voice(speech)
    ratio, delta, level = agitation_level(m, profile)
    boost = config.VERIFY_BOOST_MAX * level if is_user else 0.0
    return verify_result(
        "ok",
        similarity=round(sim, 4),
        is_user=is_user,
        pitch_ratio=None if ratio is None else round(ratio, 3),
        volume_delta_db=round(delta, 2),
        agitation=round(level, 3),
        boost=round(boost, 4),
    )
