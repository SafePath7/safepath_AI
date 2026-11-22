"""
위험 음향: 갑자기 큰 소리 (FD-015 변경, TBD-5).

소리 종류를 맞히는 대신 "갑자기, 평소 말소리보다 훨씬 크게" 났는지로 판정한다. 모델 없이 파형에서 계산한다.
    1) 3초 창을 FRAME_SEC(0.1초) 프레임으로 나눠 프레임별 음량(dBFS)을 잰다.
       완전 무음 프레임(SILENT_DBFS 미만, 워치 녹음 시작·소음 차단 구간)은 없는 것으로 본다.
    2) 배경 음량: 그 세션 최근 창들의 조용한 쪽 음량(창별 하위 20% 프레임 값)의 중앙값.
       세션 기록이 적으면 이 창의 값을 쓴다. 시끄러운 길에서는 배경도 같이 올라간다.
    3) 두 조건을 모두 봐서 점수를 낸다 (둘 중 작은 쪽):
       - 갑자기: 가장 큰 프레임 - 배경 >= LOUD_JUMP_DB (0.5점), +LOUD_SPAN_DB면 1점
       - 크게:  가장 큰 프레임 - 평소 말소리 음량 >= LOUD_OVER_SPEECH_DB (0.5점)
         평소 말소리 음량 = 음성 프로필의 baseline_volume(말소리 구간 음량 중앙값), 없으면 LOUD_SPEECH_REF_DBFS
       말소리 기준과 비교하므로 기기마다 마이크 증폭이 달라도 기준이 같이 움직인다.

기준값 근거(초안):
    - 충격성 소음은 배경보다 10~20dB 이상 높은 순간 상승으로 보는 경우가 많다(ISO 1996-1 충격음 보정,
      음향 이벤트 검출의 에너지 기준) → 15dB.
    - 발성 강도: 평소 대화 약 60dB SPL, 고함 약 80~90dB SPL(1m) → 평소 말소리보다 +10~20dB → 16dB.
    - 평소 말소리 기본값 -22dBFS: 실제 워치 녹음(평소 말투)의 말소리 구간 중앙값 -24·-18·-21dBFS.
      이 워치는 평소 말소리의 가장 큰 순간이 이미 -9dBFS라, 절대 음량 기준(dBFS)은 쓰지 않는다.
워치로 고함·박수·문 쾅·길거리 소리를 녹음하면 다시 맞춘다(docs/models).
"""

import threading
import time

import numpy as np

from app import config

FRAME_SEC = 0.1
QUIET_PERCENTILE = 20  # 창의 '조용한 쪽' 음량
HISTORY = 30  # 세션 배경 계산에 쓰는 최근 창 수 (약 30초)
MIN_HISTORY = 3  # 이보다 적으면 그 창 안에서 배경을 잰다
FLOOR_DBFS = -90.0  # 완전 무음 프레임의 음량
SILENT_DBFS = -70.0  # 이보다 작은 프레임은 소리가 없는 것으로 본다
SESSION_IDLE_SEC = 600


def frame_dbfs(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """프레임별 RMS 음량(dBFS). 마지막 자투리 프레임은 버린다."""
    n = int(sr * FRAME_SEC)
    usable = (waveform.size // n) * n
    if usable == 0:
        return np.zeros(0)
    frames = waveform[:usable].astype(np.float64).reshape(-1, n)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(rms)
    return np.maximum(db, FLOOR_DBFS)


class BackgroundTracker:
    """세션마다 최근 창들의 조용한 쪽 음량을 들고 있다 (메모리만)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._levels: dict[str, tuple[float, list[float]]] = {}

    def background(self, session_id: str | None, quiet_now: float) -> float:
        if not session_id:
            return quiet_now
        with self._lock:
            entry = self._levels.get(session_id)
            history = entry[1] if entry else []
        if len(history) < MIN_HISTORY:
            return quiet_now
        return float(np.median(history))

    def add(self, session_id: str | None, quiet_now: float, loud: bool) -> None:
        """이번 창의 조용한 쪽 음량을 기록한다. 큰 소리가 난 창은 배경을 끌어올리지 않게 넣지 않는다."""
        if not session_id or loud:
            return
        now = time.monotonic()
        with self._lock:
            for k in [k for k, (seen, _) in self._levels.items() if now - seen > SESSION_IDLE_SEC]:
                del self._levels[k]
            history = self._levels.get(session_id, (now, []))[1]
            history.append(quiet_now)
            del history[:-HISTORY]
            self._levels[session_id] = (now, history)

    def clear(self) -> None:
        with self._lock:
            self._levels.clear()


tracker = BackgroundTracker()


def _ramp(x: float, zero: float, half: float, full: float) -> float:
    if x <= zero:
        return 0.0
    if x < half:
        return 0.5 * (x - zero) / (half - zero)
    return 0.5 + 0.5 * min(1.0, (x - half) / max(full - half, 1e-6))


def loud_score(jump_db: float, over_speech_db: float) -> float:
    """갑자기(jump)와 크게(평소 말소리보다) 두 점수 중 작은 쪽."""
    sudden = _ramp(jump_db, 0.0, config.LOUD_JUMP_DB, config.LOUD_JUMP_DB + config.LOUD_SPAN_DB)
    over = config.LOUD_OVER_SPEECH_DB
    loud = _ramp(over_speech_db, over - 4, over, over + 6)  # 평소보다 12dB 이하는 0
    return round(min(sudden, loud), 4)


def empty_loudness_result(status: str = "silence") -> dict:
    return {"status": status, "score": 0.0, "is_danger": False, "peak_dbfs": None, "background_dbfs": None,
            "jump_db": None, "speech_dbfs": None, "over_speech_db": None, "peak_at_sec": None}


def analyze_loudness(waveform: np.ndarray, session_id: str | None = None, sr: int = 16000,
                     background: BackgroundTracker = tracker, speech_dbfs: float | None = None) -> dict:
    """갑자기 큰 소리 점수. status: ok / silence(소리가 없거나 너무 짧음).

    speech_dbfs: 그 사람의 평소 말소리 음량(음성 프로필 baseline_volume). 없으면 LOUD_SPEECH_REF_DBFS.
    """
    db = frame_dbfs(waveform, sr)
    active = db[db >= SILENT_DBFS]
    if active.size < 3:
        return empty_loudness_result()
    quiet_now = float(np.percentile(active, QUIET_PERCENTILE))
    bg = background.background(session_id, quiet_now)
    peak_i = int(np.argmax(db))
    peak = float(db[peak_i])
    speech = config.LOUD_SPEECH_REF_DBFS if speech_dbfs is None else float(speech_dbfs)
    jump, over_speech = peak - bg, peak - speech
    score = loud_score(jump, over_speech)
    is_danger = score >= 0.5
    background.add(session_id, quiet_now, is_danger)
    return {
        "status": "ok",
        "score": score,
        "is_danger": is_danger,
        "peak_dbfs": round(peak, 1),
        "background_dbfs": round(bg, 1),
        "jump_db": round(jump, 1),
        "speech_dbfs": round(speech, 1),
        "over_speech_db": round(over_speech, 1),
        "peak_at_sec": round(peak_i * FRAME_SEC, 1),
    }
