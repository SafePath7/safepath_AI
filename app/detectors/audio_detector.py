"""
Whisper(위험 문구) + YAMNet(위험 음향) 병렬 탐지 모듈
같은 오디오 클립을 두 모델에 동시에 넣고 결과를 합친다.
(cap_ai/audio_detector.py에서 옮김)
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app import config

log = logging.getLogger("safepath.ai.detector")

# ── 위험 판단 기준 ──────────────────────────────────────────────
DANGER_KEYWORDS = [
    "살려주세요", "살려줘", "도와주세요", "도와줘", "놔주세요", "그만해",
]

DANGER_CLASSES = [
    "Screaming", "Shout", "Yell", "Battle cry",
    "Glass", "Shatter", "Breaking",
    "Gunshot, gunfire", "Explosion", "Boom",
    "Smash, crash", "Thump, thud",
]


# ── 모델 로드 (서버 시작 시 한 번) ─────────────────────────────
def load_models() -> dict:
    import tensorflow_hub as hub
    from faster_whisper import WhisperModel

    whisper = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    class_names = load_yamnet_class_names(yamnet)
    missing = [c for c in DANGER_CLASSES if c not in class_names]
    if missing:
        # 모델 버전에 따라 이름이 달라지면 해당 클래스는 탐지되지 않으므로 시작 시 알린다.
        log.warning("YAMNet 클래스 목록에 없는 DANGER_CLASSES: %s", missing)
    return {
        "whisper": whisper,
        "yamnet": yamnet,
        "yamnet_classes": class_names,
    }


# ── Whisper: 위험 문구 탐지 ─────────────────────────────────────
_KEYWORDS_NO_SPACE = [re.sub(r"\s+", "", kw) for kw in DANGER_KEYWORDS]


def detect_danger_phrase(text: str) -> bool:
    # Whisper가 "살려 주세요"처럼 띄어 쓸 수 있으므로 공백을 지우고 비교한다.
    compact = re.sub(r"\s+", "", text)
    return any(kw in compact for kw in _KEYWORDS_NO_SPACE)


def run_whisper(whisper_model, waveform: np.ndarray) -> dict:
    """waveform: 16kHz mono float32. returns {"text", "is_danger"}"""
    # vad_filter: 무음 구간을 건너뛰어 "감사합니다" 같은 환각 출력을 줄인다.
    segments, _ = whisper_model.transcribe(waveform, language="ko", vad_filter=True)
    text = " ".join(seg.text for seg in segments).strip()
    return {"text": text, "is_danger": detect_danger_phrase(text)}


# ── YAMNet: 위험 음향 탐지 ──────────────────────────────────────
def load_yamnet_class_names(yamnet_model) -> list[str]:
    import pandas as pd

    class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
    return list(pd.read_csv(class_map_path)["display_name"])


def detect_danger_sound(scores, class_names: list[str], threshold: float) -> list[tuple[str, float]]:
    mean_scores = scores.numpy().mean(axis=0)
    hits = []
    for cls in DANGER_CLASSES:
        if cls in class_names:
            idx = class_names.index(cls)
            if mean_scores[idx] > threshold:
                hits.append((cls, float(mean_scores[idx])))
    return hits


def run_yamnet(yamnet_model, class_names: list[str], waveform: np.ndarray) -> dict:
    """waveform: 16kHz mono float32 -1~1. returns {"hits", "is_danger"}"""
    scores, _embeddings, _spectrogram = yamnet_model(waveform)
    hits = detect_danger_sound(scores, class_names, config.YAMNET_THRESHOLD)
    return {
        "hits": [{"class": c, "score": round(s, 4)} for c, s in hits],
        "is_danger": len(hits) > 0,
    }


# ── 병렬 실행 ───────────────────────────────────────────────────
def analyze_clip(models: dict, waveform: np.ndarray) -> dict:
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_whisper = pool.submit(run_whisper, models["whisper"], waveform)
        f_yamnet = pool.submit(run_yamnet, models["yamnet"], models["yamnet_classes"], waveform)
    whisper_result = f_whisper.result()
    yamnet_result = f_yamnet.result()
    return {
        "whisper": whisper_result,
        "yamnet": yamnet_result,
        "is_danger": whisper_result["is_danger"] or yamnet_result["is_danger"],
    }
