"""
위험 문구 매칭 (FD-014). Whisper 인식 텍스트에서 위험 문구를 찾아 0~1 점수를 낸다.

Whisper는 한국어 짧은 외침을 자주 틀리게 받아 적는다 (실측: "놔주세요" → "나 주세요").
그래서 정확히 일치하지 않아도 같은 음절 수에서 모음(중성) 1개만 다르면 같은 문구로 본다.
실측 오인식은 모음 차이였고(놔→나, 려→료, 요→여), 자음 차이를 허용하면 일상 표현과 겹친다.
오탐을 막기 위해 4음절 이상 문구에만 적용한다.
  - "나주세요" ↔ "놔주세요" : 모음 1개 차이(ㅏ/ㅘ) → 일치
  - "봐주세요"·"와주세요" ↔ "놔주세요" : 자음 차이 → 불일치 ("한 번만 봐주세요")
  - "보여주세요" ↔ "도와주세요" : 2개 차이 → 불일치 ("메뉴판 보여주세요")
  - "도와줄게" ↔ "도와줘" : 3음절 → 정확히 일치할 때만
"""

import re

# 문구 → 가중치. 일상 대화에도 흔히 나오는 표현은 낮게 둔다(위험 판정은 되지만 FD-018 종합 때 덜 반영).
#   "이 숙제 좀 도와줘", "장난 그만해", "그렇게 하지 마세요" 같은 일상 사용이 있어 0.7~0.8
DANGER_PHRASES: dict[str, float] = {
    "살려주세요": 1.0,
    "살려줘": 1.0,
    "도와주세요": 1.0,
    "놔주세요": 1.0,
    "놔줘": 1.0,       # 놔주세요의 반말형 (살려주세요/살려줘와 같은 짝)
    "도와줘": 0.8,
    "그만해": 0.7,
    "하지마세요": 0.7,  # 실측에서 "하지 마세요 놔주세요" 중 뒷부분만 오인식되는 경우를 보완
}

FUZZY_MIN_SYLLABLES = 4   # 이 길이 이상 문구만 모음 1개 차이 허용
FUZZY_PENALTY = 0.9       # 정확히 일치하지 않은 경우 점수에 곱한다

_HANGUL = re.compile(r"[가-힣]")


def to_jamo(ch: str) -> tuple[int, int, int]:
    """한글 음절 1개 → (초성, 중성, 종성) 번호. 종성이 없으면 0."""
    code = ord(ch) - 0xAC00
    return code // 588, (code % 588) // 28, code % 28


def vowel_only_diff(a: str, b: str) -> int | None:
    """같은 길이 음절열에서 초성·종성이 모두 같으면 다른 모음 개수, 자음이 하나라도 다르면 None."""
    diff = 0
    for x, y in zip(a, b):
        (cx, vx, jx), (cy, vy, jy) = to_jamo(x), to_jamo(y)
        if cx != cy or jx != jy:
            return None
        diff += vx != vy
    return diff


def compact_hangul(text: str) -> str:
    """띄어쓰기·문장부호·영문을 지우고 한글 음절만 남긴다 ("살려 주세요!" → "살려주세요")."""
    return "".join(_HANGUL.findall(text))


def match_phrase(compact: str, phrase: str) -> tuple[int, str] | None:
    """compact 안에서 phrase를 찾는다. 정확히 있으면 (0, phrase), 모음 1개 차이면 (1, 들린 구간)."""
    if phrase in compact:
        return 0, phrase
    n = len(phrase)
    if n < FUZZY_MIN_SYLLABLES:
        return None
    for start in range(0, len(compact) - n + 1):
        window = compact[start:start + n]
        if vowel_only_diff(window, phrase) == 1:
            return 1, window
    return None


def score_danger_phrase(text: str) -> tuple[float, list[dict]]:
    """텍스트의 위험 문구 점수(0~1)와 찾은 문구 목록. 점수는 찾은 문구 중 최댓값."""
    compact = compact_hangul(text)
    matches = []
    for phrase, weight in DANGER_PHRASES.items():
        found = match_phrase(compact, phrase)
        if found is None:
            continue
        distance, heard = found
        score = weight * (1.0 if distance == 0 else FUZZY_PENALTY)
        matches.append({"phrase": phrase, "heard": heard, "exact": distance == 0, "score": round(score, 3)})
    matches.sort(key=lambda m: m["score"], reverse=True)
    return (matches[0]["score"] if matches else 0.0), matches
