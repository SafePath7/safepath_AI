"""
위험도 종합 (FD-018 1단계): 모델별 점수를 하나의 위험도(0~1)와 위험 유형으로 합친다.

모델마다 점수 눈금이 달라(위험 문구는 0.5, 위험 음향은 0.3에서 위험) 그대로 최댓값을 내면 안 된다.
각 점수를 '자기 임계값이 0.5가 되도록' 0~1로 다시 맞춘 뒤 최댓값을 위험도로 쓴다.
    점수 < 임계값  : 0 ~ 0.5   (위험 아님)
    점수 >= 임계값 : 0.5 ~ 1   (위험)
그래서 RISK_THRESHOLD(SDS TBD-5, 기본 0.5)는 모델별 임계값과 같은 판정을 낸다.
화자 검증(음성 프로필이 있을 때): 등록한 본인 목소리가 평소보다 높고 크면 위험도를 최대 VERIFY_BOOST_MAX만큼 올린다.
    위험 신호가 이미 어느 정도(VERIFY_BOOST_MIN_BASE) 있을 때만 더하므로, 크게 말하는 것만으로는 위험이 되지 않는다.
    점수를 깎지는 않는다(본인 판정이 틀려도 감지가 막히지 않게).
움직임(FD-017)이 붙으면 여기에 항목을 더한다. 결과가 빠진 모델은 계산에서 뺀다(SDS: 일부 결과 누락 → 남은 결과로 계산).
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


def combine(whisper: dict | None, yamnet: dict | None, speaker: dict | None = None) -> dict:
    """모델별 결과 → {risk_score, risk_type, is_danger}. speaker가 있으면 실제로 더한 값을 speaker["applied"]에 적는다."""
    candidates = []  # (위험도, 유형)
    if whisper and whisper.get("status") == "ok":
        candidates.append((rescale(whisper["score"], config.VOICE_THRESHOLD), DANGER_PHRASE))
    if yamnet:
        cats = yamnet.get("categories") or {}
        if cats:
            top = max(cats, key=cats.get)
            candidates.append((rescale(cats[top], config.YAMNET_THRESHOLD), top))
    if speaker is not None:
        speaker["applied"] = 0.0
    if not candidates:
        return {"risk_score": 0.0, "risk_type": None, "is_danger": False}
    score, kind = max(candidates)
    if speaker and speaker.get("status") == "ok" and speaker.get("boost", 0) > 0 and score >= config.VERIFY_BOOST_MIN_BASE:
        boosted = min(1.0, score + speaker["boost"])
        speaker["applied"] = round(boosted - score, 4)
        score = boosted
    is_danger = score >= config.RISK_THRESHOLD
    return {"risk_score": round(score, 4), "risk_type": kind if is_danger else None, "is_danger": is_danger}
