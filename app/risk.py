"""
위험도 종합 (FD-018 1단계): 모델별 점수를 하나의 위험도(0~1)와 위험 유형으로 합친다.

모델마다 점수 눈금이 달라(위험 문구는 0.5, 위험 음향은 0.3에서 위험) 그대로 최댓값을 내면 안 된다.
각 점수를 '자기 임계값이 0.5가 되도록' 0~1로 다시 맞춘 뒤 최댓값을 위험도로 쓴다.
    점수 < 임계값  : 0 ~ 0.5   (위험 아님)
    점수 >= 임계값 : 0.5 ~ 1   (위험)
그래서 RISK_THRESHOLD(SDS TBD-5, 기본 0.5)는 모델별 임계값과 같은 판정을 낸다.
움직임(FD-017)과 화자 검증이 붙으면 여기에 항목을 더한다. 결과가 빠진 모델은 계산에서 뺀다(SDS: 일부 결과 누락 → 남은 결과로 계산).
"""

from app import config

# risk_type 값. 백엔드 risk_events.event_type(VARCHAR(20))에 그대로 넣을 수 있게 짧은 영문 코드로 둔다.
DANGER_PHRASE = "danger_phrase"  # 위험 문구 (FD-014)
SOUND_TYPES = ("scream", "shout", "glass_break", "impact")  # 위험 음향 (FD-015)


def rescale(score: float, threshold: float) -> float:
    """임계값이 0.5에 오도록 0~1 점수를 다시 맞춘다."""
    score = min(max(float(score), 0.0), 1.0)
    threshold = min(max(threshold, 1e-6), 1 - 1e-6)
    if score < threshold:
        return 0.5 * score / threshold
    return 0.5 + 0.5 * (score - threshold) / (1 - threshold)


def combine(whisper: dict | None, yamnet: dict | None) -> dict:
    """모델별 결과 → {risk_score, risk_type, is_danger}."""
    candidates = []  # (위험도, 유형)
    if whisper and whisper.get("status") == "ok":
        candidates.append((rescale(whisper["score"], config.VOICE_THRESHOLD), DANGER_PHRASE))
    if yamnet:
        cats = yamnet.get("categories") or {}
        if cats:
            top = max(cats, key=cats.get)
            candidates.append((rescale(cats[top], config.YAMNET_THRESHOLD), top))
    if not candidates:
        return {"risk_score": 0.0, "risk_type": None, "is_danger": False}
    score, kind = max(candidates)
    is_danger = score >= config.RISK_THRESHOLD
    return {"risk_score": round(score, 4), "risk_type": kind if is_danger else None, "is_danger": is_danger}
