"""
위험도 종합 (FD-018): 신호별 점수를 하나의 위험도(0~1)와 위험 유형으로 합친다.

1) 눈금 맞추기: 신호마다 점수 눈금이 달라(위험 문구 0.5, YAMNet 0.3에서 위험) 각 점수를 '자기 임계값 = 0.5'로 맞춘다.
       점수 < 임계값 : 0 ~ 0.5 (위험 아님) / 점수 >= 임계값 : 0.5 ~ 1 (위험)
   갑자기 큰 소리(loudness)와 움직임(motion)은 처음부터 이 눈금으로 낸다.
2) 여러 신호 종합 (FR-AI-05 "여러 신호 동시 발생 시 위험도 상승"): noisy-OR
       위험도 = 1 - (1 - s1)(1 - s2)...
   신호 하나면 그 점수 그대로라 단독 판정은 모델별 임계값과 같다. 애매한 신호 둘(예: 큰 소리 0.35 + 움직임 0.35)이
   함께 나오면 0.58로 위험이 된다. COMBINE_MIN 미만의 작은 신호는 가장 큰 신호가 아니면 빼서 잡음이 쌓이지 않게 한다.
   결과가 빠진 신호(인식 실패, 움직임 없음)는 계산에서 뺀다(SDS: 일부 결과 누락 → 남은 결과로 계산).
3) 화자 검증(음성 프로필이 있을 때): 등록한 본인 목소리가 평소보다 높고 크면 위험도를 최대 VERIFY_BOOST_MAX만큼 올린다.
   위험 신호가 이미 어느 정도(VERIFY_BOOST_MIN_BASE) 있을 때만 더하고, 점수를 깎지는 않는다.

위험 음향 판정(SOUND_MODE): loudness(기본) = 갑자기 큰 소리 점수로 판정하고 YAMNet은 소리 이름만 붙인다.
yamnet = 예전 방식(YAMNet 위험 클래스 점수로 판정).
위치는 쓸 판단 근거(위험 지역 등)가 아직 없어 넣지 않는다.
"""

from app import config

# risk_type 값. 백엔드 risk_events.event_type(VARCHAR(20))에 그대로 넣을 수 있게 짧은 영문 코드로 둔다.
DANGER_PHRASE = "danger_phrase"  # 위험 문구 (FD-014)
SOUND_TYPES = ("scream", "shout", "glass_break", "impact")  # 위험 음향 이름 (YAMNet, FD-015)
LOUD_SOUND = "loud_sound"  # 갑자기 큰 소리인데 YAMNet 이름이 없을 때
MOTION_TYPES = ("fall", "motion_impact", "struggle")  # 격한 움직임 (FD-017)


def rescale(score: float, threshold: float) -> float:
    """임계값이 0.5에 오도록 0~1 점수를 다시 맞춘다."""
    score = min(max(float(score), 0.0), 1.0)
    threshold = min(max(threshold, 1e-6), 1 - 1e-6)
    if score < threshold:
        return 0.5 * score / threshold
    return 0.5 + 0.5 * (score - threshold) / (1 - threshold)


def sound_label(yamnet: dict | None) -> str:
    """큰 소리에 붙일 이름: YAMNet이 가장 높게 본 위험 소리(SOUND_LABEL_MIN 이상), 없으면 loud_sound."""
    cats = (yamnet or {}).get("categories") or {}
    if cats:
        top = max(cats, key=cats.get)
        if cats[top] >= config.SOUND_LABEL_MIN:
            return top
    return LOUD_SOUND


def signals_of(whisper: dict | None, yamnet: dict | None, loudness: dict | None = None,
               motion: dict | None = None) -> list[tuple[float, str]]:
    """(위험도 눈금 점수, 유형) 목록. 결과가 없는 신호는 뺀다."""
    out = []
    if whisper and whisper.get("status") == "ok":
        out.append((rescale(whisper["score"], config.VOICE_THRESHOLD), DANGER_PHRASE))
    if config.SOUND_MODE == "yamnet" or loudness is None:
        cats = (yamnet or {}).get("categories") or {}
        if cats:
            top = max(cats, key=cats.get)
            out.append((rescale(cats[top], config.YAMNET_THRESHOLD), top))
    elif loudness.get("status") == "ok":
        out.append((float(loudness["score"]), sound_label(yamnet)))
    if motion and motion.get("status") == "ok":
        scores = motion.get("scores") or {}
        kind = motion.get("type") or (max(scores, key=scores.get) if scores else "struggle")
        out.append((float(motion["score"]), kind))
    return out


def combine(whisper: dict | None, yamnet: dict | None, speaker: dict | None = None,
            loudness: dict | None = None, motion: dict | None = None) -> dict:
    """신호별 결과 → {risk_score, risk_type, is_danger, signals}. speaker가 있으면 실제로 더한 값을 speaker["applied"]에 적는다."""
    if speaker is not None:
        speaker["applied"] = 0.0
    found = sorted(signals_of(whisper, yamnet, loudness, motion), reverse=True)
    if not found:
        return {"risk_score": 0.0, "risk_type": None, "is_danger": False, "signals": []}
    used = [found[0]] + [x for x in found[1:] if x[0] >= config.COMBINE_MIN]
    keep = 1.0
    for value, _ in used:
        keep *= 1.0 - min(max(value, 0.0), 1.0)
    score, kind = 1.0 - keep, found[0][1]
    if speaker and speaker.get("status") == "ok" and speaker.get("boost", 0) > 0 and score >= config.VERIFY_BOOST_MIN_BASE:
        boosted = min(1.0, score + speaker["boost"])
        speaker["applied"] = round(boosted - score, 4)
        score = boosted
    is_danger = score >= config.RISK_THRESHOLD
    return {
        "risk_score": round(score, 4),
        "risk_type": kind if is_danger else None,
        "is_danger": is_danger,
        # 종합에 쓴 신호 (큰 순서). 동시에 무엇이 나왔는지 백엔드 설명·근거 데이터에 쓴다
        "signals": [{"type": k, "score": round(v, 4)} for v, k in used if v > 0],
    }
