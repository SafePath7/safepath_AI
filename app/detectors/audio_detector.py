"""
Whisper(위험 문구) + YAMNet(위험 음향) 병렬 탐지 모듈
같은 오디오 클립을 두 모델에 동시에 넣고 결과를 합친다.
(cap_ai/audio_detector.py에서 옮김)
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from app import config

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
    return {
        "whisper": whisper,
        "yamnet": yamnet,
        "yamnet_classes": load_yamnet_class_names(yamnet),
    }


# ── Whisper: 위험 문구 탐지 ─────────────────────────────────────
def detect_danger_phrase(text: str) -> bool:
    return any(kw in text for kw in DANGER_KEYWORDS)


def run_whisper(whisper_model, waveform: np.ndarray) -> dict:
    """waveform: 16kHz mono float32. returns {"text", "is_danger"}"""
    segments, _ = whisper_model.transcribe(waveform, language="ko")
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
