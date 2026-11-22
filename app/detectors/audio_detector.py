"""
Whisper(위험 문구, FD-014) + YAMNet(소리 이름, FD-015) 병렬 탐지 모듈
같은 오디오 클립을 두 모델에 동시에 넣고, 갑자기 큰 소리(loudness)·움직임(FD-017)·화자 검증과 함께 종합한다.
(cap_ai/audio_detector.py에서 옮김)
"""

import hashlib
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app import config
from app.risk import combine
from app.detectors.phrase import DANGER_PHRASES, score_danger_phrase

log = logging.getLogger("safepath.ai.detector")

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

# ── 위험 판단 기준 ──────────────────────────────────────────────
# 위험 문구(FD-014)는 app/detectors/phrase.py의 DANGER_PHRASES

# SDS FD-015: 비명·고함·유리 파손·충격음. 클래스는 인덱스가 아니라 YAMNet class map의
# display_name으로 찾는다(모델 버전이 바뀌어도 엉뚱한 클래스를 잡지 않도록).
# 유형 코드는 이후 백엔드 risk_events.event_type(VARCHAR 20)에 매핑하기 쉽게 짧은 영문으로 둔다.
DANGER_CATEGORIES: dict[str, list[str]] = {
    "scream": ["Screaming"],                       # 비명
    "shout": ["Shout", "Yell", "Bellow"],          # 고함 ("Battle cry"는 YAMNet 521개 클래스에 없어 Bellow로 교체)
    "glass_break": ["Glass", "Shatter", "Breaking"],  # 유리 파손
    "impact": ["Smash, crash", "Thump, thud", "Gunshot, gunfire", "Explosion", "Boom"],  # 충격음
}
DANGER_CLASSES = [c for classes in DANGER_CATEGORIES.values() for c in classes]


# ── 모델 로드 (서버 시작 시 한 번) ─────────────────────────────
def load_models() -> dict:
    from faster_whisper import WhisperModel

    t = time.perf_counter()
    whisper = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    log.info("Whisper(%s) 로드 %.1fs", config.WHISPER_MODEL, time.perf_counter() - t)

    t = time.perf_counter()
    yamnet = load_yamnet()
    class_names = load_yamnet_class_names(yamnet)
    danger_index = resolve_danger_index(class_names)
    log.info("YAMNet 로드 %.1fs (클래스 %d개, 위험 클래스 %d개)",
             time.perf_counter() - t, len(class_names), sum(len(v) for v in danger_index.values()))
    return {
        "whisper": whisper,
        "yamnet": yamnet,
        "yamnet_danger_index": danger_index,
    }


def load_yamnet():
    """TF Hub에서 YAMNet을 받아 프로젝트 캐시(models/tfhub)에 두고 적재한다.

    캐시 폴더가 비어 있으면(다운로드 중단, 디스크 정리) 그 모델 폴더만 지우고 한 번 다시 받는다.
    """
    cache_dir = config.MODEL_CACHE_DIR / "tfhub"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TFHUB_CACHE_DIR", str(cache_dir))

    import tensorflow_hub as hub

    try:
        return hub.load(YAMNET_URL)
    except ValueError as exc:
        if "saved_model.pb" not in str(exc):
            raise
        # TF Hub는 sha1(URL) 이름의 폴더에 모델을 둔다. 그 폴더가 깨져 있으면 지우고 재다운로드한다.
        # TFHUB_CACHE_DIR을 외부에서 지정했다면 그 경로를 따른다.
        broken = Path(os.environ["TFHUB_CACHE_DIR"]) / hashlib.sha1(YAMNET_URL.encode("utf8")).hexdigest()
        log.warning("YAMNet 캐시가 손상돼 다시 받습니다: %s", broken)
        shutil.rmtree(broken, ignore_errors=True)
        return hub.load(YAMNET_URL)


# ── Whisper: 위험 음성 탐지 (FD-014) ────────────────────────────
# status: ok(분석함) / silence(음량이 무음 기준 미만) / no_speech(소리는 있지만 말소리 없음) / error(인식 실패)
def voice_result(status: str, text: str = "", score: float = 0.0, matches: list | None = None) -> dict:
    return {
        "status": status,
        "text": text,
        "score": round(score, 3),
        "is_danger": score >= config.VOICE_THRESHOLD,
        "matches": matches or [],
    }


def rms_dbfs(waveform: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))) if waveform.size else 0.0
    return 20 * np.log10(rms) if rms > 0 else -np.inf


def has_speech(waveform: np.ndarray) -> bool:
    """Silero VAD(faster-whisper 내장)로 말소리 구간이 있는지 본다. 모델은 첫 호출 후 캐시된다."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    return bool(get_speech_timestamps(waveform, VadOptions(min_speech_duration_ms=250)))


def analyze_voice(whisper_model, waveform: np.ndarray) -> dict:
    """waveform: 16kHz mono float32. SDS FD-014 처리 흐름과 예외 처리를 따른다.

    1. 무음 → 분석 생략 (음량 기준, 그다음 VAD 기준). Whisper는 무음·잡음에서 없는 말을 지어내므로
       말소리가 없으면 아예 돌리지 않는다.
    2. Whisper 변환 후 위험 문구 점수(0~1)
    3. 인식 실패(예외) → 점수 0. 요청은 실패시키지 않고 다음 청크 분석을 이어가게 한다.
    """
    if waveform.size == 0 or rms_dbfs(waveform) < config.SILENCE_DBFS:
        return voice_result("silence")
    try:
        if not has_speech(waveform):
            return voice_result("no_speech")
        segments, _ = whisper_model.transcribe(
            waveform,
            language="ko",
            vad_filter=True,                    # 말소리 사이 무음 구간을 잘라 환각을 줄인다
            condition_on_previous_text=False,   # 짧은 청크에서 같은 문장을 반복 출력하는 현상 방지
            hotwords=" ".join(DANGER_PHRASES) if config.WHISPER_HOTWORDS else None,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception:  # noqa: BLE001 - 어떤 이유로든 인식 실패면 점수 0 (FD-014 예외 처리)
        log.exception("Whisper 인식 실패: 점수 0으로 처리")
        return voice_result("error")
    score, matches = score_danger_phrase(text)
    return voice_result("ok", text, score, matches)


# ── YAMNet: 위험 음향 탐지 (FD-015) ─────────────────────────────
def load_yamnet_class_names(yamnet_model) -> list[str]:
    import pandas as pd

    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    return list(pd.read_csv(class_map_path)["display_name"])


def resolve_danger_index(class_names: list[str]) -> dict[str, list[tuple[str, int]]]:
    """유형별 위험 클래스 이름 → class map 인덱스. 없는 이름은 경고 후 제외한다."""
    missing = [c for c in DANGER_CLASSES if c not in class_names]
    if missing:
        log.warning("YAMNet 클래스 목록에 없는 위험 클래스(탐지 제외): %s", missing)
    return {
        category: [(name, class_names.index(name)) for name in names if name in class_names]
        for category, names in DANGER_CATEGORIES.items()
    }


def aggregate_frames(frame_scores: np.ndarray, top_k: int) -> np.ndarray:
    """(프레임 수, 클래스 수) 점수를 클래스별 상위 k개 프레임 평균으로 합친다 → (클래스 수,)"""
    if frame_scores.ndim != 2 or frame_scores.shape[0] == 0:
        return np.zeros(frame_scores.shape[-1] if frame_scores.ndim == 2 else 0, dtype=np.float32)
    k = min(top_k, frame_scores.shape[0])
    top = np.sort(frame_scores, axis=0)[-k:]
    return top.mean(axis=0)


def empty_sound_result() -> dict:
    return {
        "score": 0.0,
        "type": None,
        "is_danger": False,
        "categories": {category: 0.0 for category in DANGER_CATEGORIES},
        "hits": [],
    }


def score_danger_sound(frame_scores: np.ndarray, danger_index: dict, threshold: float,
                       top_k: int) -> dict:
    """프레임별 YAMNet 점수로 위험 음향 점수(0~1)와 유형을 낸다.

    - 클래스 점수: 상위 k개 프레임 평균
    - 유형 점수: 그 유형에 속한 클래스 점수 중 최댓값
    - 음향 점수(score): 유형 점수 중 최댓값, type은 그 유형
    - 모든 유형이 임계값 미만이면 위험 없음(type=None) — SDS FD-015 예외 처리
    """
    clip = aggregate_frames(np.asarray(frame_scores, dtype=np.float32), top_k)
    if clip.size == 0:
        return empty_sound_result()

    categories: dict[str, float] = {}
    hits = []
    for category, members in danger_index.items():
        best = 0.0
        for name, idx in members:
            s = float(clip[idx])
            best = max(best, s)
            if s >= threshold:
                hits.append({"class": name, "category": category, "score": round(s, 4)})
        categories[category] = round(best, 4)

    top_type = max(categories, key=categories.get) if categories else None
    score = categories.get(top_type, 0.0) if top_type else 0.0
    is_danger = score >= threshold
    return {
        "score": score,
        "type": top_type if is_danger else None,
        "is_danger": is_danger,
        "categories": categories,
        "hits": sorted(hits, key=lambda h: h["score"], reverse=True),
    }


def run_yamnet(yamnet_model, danger_index: dict, waveform: np.ndarray) -> dict:
    """waveform: 16kHz mono float32 -1~1. returns score_danger_sound() 결과"""
    if waveform.size == 0:
        return empty_sound_result()
    scores, _embeddings, _spectrogram = yamnet_model(waveform)
    return score_danger_sound(scores.numpy(), danger_index, config.YAMNET_THRESHOLD, config.YAMNET_TOP_K)


# ── 병렬 실행 ───────────────────────────────────────────────────
def analyze_clip(models: dict, waveform: np.ndarray, profile: dict | None = None,
                 profile_status: str = "off", session_id: str | None = None, motion: dict | None = None) -> dict:
    """위험 문구·소리·큰 소리·움직임(·화자 검증)을 돌려 종합 위험도를 낸다 (FD-018).

    profile: 사용자 음성 프로필(app.profiles). 없으면 화자 검증은 profile_status 상태로 건너뛴다.
    session_id: 큰 소리의 기준이 되는 세션 배경 음량을 이어서 잰다. motion: 같은 3초 구간 움직임(FD-017).
    """
    from app.detectors.loudness import analyze_loudness
    from app.detectors.motion import analyze_motion
    from app.detectors.speaker import verify_result, verify_voice

    with ThreadPoolExecutor(max_workers=3) as pool:
        f_whisper = pool.submit(analyze_voice, models["whisper"], waveform)
        f_yamnet = pool.submit(run_yamnet, models["yamnet"], models["yamnet_danger_index"], waveform)
        f_speaker = None
        if profile is not None and models.get("speaker") is not None:
            f_speaker = pool.submit(verify_voice, models["speaker"], waveform, profile)
    speech_dbfs = (profile or {}).get("baseline_volume")  # 그 사람의 평소 말소리 음량
    loudness_result = analyze_loudness(waveform, session_id, speech_dbfs=speech_dbfs)  # 계산이 가벼워 따로 안 돌림
    motion_result = analyze_motion(motion)
    whisper_result = f_whisper.result()
    yamnet_result = f_yamnet.result()
    if f_speaker is None:
        speaker_result = verify_result(profile_status)
    else:
        try:
            speaker_result = f_speaker.result()
        except Exception:  # 화자 검증 실패가 위험 감지를 막지 않게
            log.exception("화자 검증 실패")
            speaker_result = verify_result("error")
    risk = combine(whisper_result, yamnet_result, speaker_result, loudness_result, motion_result)
    return {**risk, "whisper": whisper_result, "yamnet": yamnet_result, "loudness": loudness_result,
            "motion": motion_result, "speaker": speaker_result}
