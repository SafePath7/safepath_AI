"""
격한 움직임 감지 (FD-017): 폰·워치 가속도·자이로 시계열 → 움직임 점수(0~1)와 유형.

입력 (/api/ai/analyze의 motion): {"rate_hz": 20, "source": "watch"|"phone", "samples": [[t_ms, ax, ay, az, gx, gy, gz], ...]}
    가속도 m/s²(중력 포함, 안드로이드 TYPE_ACCELEROMETER), 자이로 rad/s. 같은 3초 구간.

학습 데이터가 없어 손목 착용 낙상 검출 문헌의 규칙(임계값) 방식으로 시작한다.
    fall          낙상: 자유낙하(가속도 크기 < FREE_FALL_G, FREE_FALL_SEC 이상) 뒤 1초 안에 충격, 그 뒤 거의 정지
    motion_impact 충격: 자유낙하 없이 큰 충격 (부딪힘·맞음)
    motion_impact 충격: 자유낙하 없이, 조용하던 중에 혼자 튀는 큰 충격 (부딪힘·맞음)
    struggle      격한 움직임: STRUGGLE_SEC 동안 팔에 실린 힘(가속도 크기 중앙값)과 회전 속도가 계속 큼 (몸싸움·휘두르기)
기준값 근거:
    - 자유낙하: 0.5~0.6g 미만 0.1~0.3초 (문헌). 낙상 충격 2g (문헌, 20Hz는 최댓값을 놓치므로 아래쪽).
    - 정지: 충격 뒤 1초 가속도 흔들림(표준편차) 0.2g 미만 (문헌).
    - 워치 실측(2026-11-22, 갤럭시 워치 20Hz, 걷기·계단·뛰기·팔 휘두르기·손목 털기 각 10초):
        뛰기도 손목 최댓값이 7.7g까지 나와(센서 최대 8g에서 잘림) 최댓값만으로는 충격과 못 가른다
        → 충격은 '나머지 구간이 조용할 때(중앙값 1.3g 이하)'만 센다. 뛰기의 나머지 중앙값 약 2g.
        가속도 흔들림(표준편차)은 뛰기 1.1~1.8g, 휘두르기 1.0~1.6g로 겹쳐 못 쓴다
        → 가속도 크기 중앙값(뛰기 2.1g, 휘두르기 3.8g, 털기 4.3g)과 평균 회전 속도(뛰기 최대 5.7, 휘두르기 7.3~8.9,
          털기 17~22rad/s)를 함께 본다. 걷기·계단은 중앙값 1.0g, 회전 2rad/s 안팎.
    넘어지기 녹음과 여러 사람의 뛰기(전력 질주)가 오면 다시 맞춘다(docs/models).

점수는 각 유형의 기준을 딱 넘을 때 0.5가 되도록 맞춘다(app/risk.py와 같은 눈금).
"""

import numpy as np

G = 9.80665

FREE_FALL_G = 0.6
FREE_FALL_SEC = 0.15
FALL_IMPACT_G = 2.0  # 자유낙하 뒤 이만큼이면 낙상 충격
IMPACT_G = 3.0  # 자유낙하 없이 이만큼이면 충격 (0.5점)
IMPACT_FULL_G = 6.0  # 1점
STILL_STD_G = 0.2
STILL_SEC = 1.0
CALM_G = 1.3  # 충격 앞뒤를 뺀 나머지 가속도 중앙값이 이 이하면 '조용하던 중' (뛰기 약 2g)
BUSY_G = 1.8  # 이 이상이면 충격으로 안 본다
STRUGGLE_SEC = 1.5
STRUGGLE_ACC_G = 3.0  # 가속도 크기 중앙값 (0.5점). 뛰기 2.1g, 휘두르기 3.8g
STRUGGLE_GYRO = 7.0  # rad/s, 평균 회전 속도 (0.5점). 뛰기 최대 5.7, 휘두르기 7.3 이상
MIN_SAMPLES = 10

TYPES = ("fall", "motion_impact", "struggle")


def empty_motion_result(status: str) -> dict:
    return {"status": status, "score": 0.0, "type": None, "is_danger": False, "source": None,
            "samples": 0, "scores": {t: 0.0 for t in TYPES}, "peak_g": None, "free_fall_sec": None}


def _ramp(x: float, zero: float, half: float, full: float) -> float:
    """x가 zero면 0, half면 0.5, full이면 1 (구간별 직선)."""
    if x <= zero:
        return 0.0
    if x < half:
        return 0.5 * (x - zero) / (half - zero)
    return min(1.0, 0.5 + 0.5 * (x - half) / (full - half))


def parse_samples(motion: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(시각 초, 가속도 크기 g, 회전 속도 크기 rad/s). 형식이 틀린 샘플은 버린다."""
    rows = []
    for s in motion.get("samples") or []:
        if isinstance(s, (list, tuple)) and len(s) >= 7:
            try:
                rows.append([float(v) for v in s[:7]])
            except (TypeError, ValueError):
                continue
    if len(rows) < MIN_SAMPLES:
        return None
    a = np.asarray(rows)
    a = a[np.all(np.isfinite(a), axis=1)]
    if len(a) < MIN_SAMPLES:
        return None
    a = a[np.argsort(a[:, 0], kind="stable")]
    t = (a[:, 0] - a[0, 0]) / 1000.0
    if t[-1] <= 0:  # 시각이 없거나 모두 같으면 rate_hz로 만든다
        rate = float(motion.get("rate_hz") or 20)
        t = np.arange(len(a)) / rate
    acc = np.linalg.norm(a[:, 1:4], axis=1) / G
    gyro = np.linalg.norm(a[:, 4:7], axis=1)
    return t, acc, gyro


def _runs(mask: np.ndarray, t: np.ndarray) -> list[tuple[int, int, float]]:
    """연속으로 True인 구간 (시작, 끝, 길이 초)."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            step = (t[-1] - t[0]) / max(len(t) - 1, 1)
            out.append((start, end, t[end] - t[start] + step))
            start = None
    return out


def score_fall(t, acc) -> tuple[float, float | None]:
    """자유낙하 → 충격 → 정지. (점수, 자유낙하 길이)"""
    best, ff_len = 0.0, None
    for start, end, dur in _runs(acc < FREE_FALL_G, t):
        if dur < FREE_FALL_SEC:
            continue
        after = (t > t[end]) & (t <= t[end] + 1.0)
        if not after.any():
            s = 0.25  # 창 끝의 자유낙하: 충격은 다음 창에서 본다
        else:
            idx = np.where(after)[0]
            peak_i = idx[np.argmax(acc[idx])]
            impact = _ramp(acc[peak_i], 1.2, FALL_IMPACT_G, IMPACT_FULL_G)
            rest = (t > t[peak_i] + 0.2) & (t <= t[peak_i] + 0.2 + STILL_SEC)
            if rest.sum() >= 5:
                still = 1.0 if acc[rest].std() < STILL_STD_G else max(0.0, 1 - (acc[rest].std() - STILL_STD_G) / 0.5)
            else:
                still = 0.5  # 정지 여부는 아직 모름 (창 끝)
            # 자유낙하 0.2 + 충격 최대 0.4 + 정지 최대 0.4. 자유낙하 + 기준 충격(0.5) + 모름(0.5) ≈ 0.6
            s = 0.2 + 0.4 * impact + 0.4 * still
            if acc[peak_i] < FALL_IMPACT_G:
                s = min(s, 0.3)  # 충격이 약하면(뛰기·점프 착지) 다른 신호와 합쳐도 쉽게 넘지 않게 낮게 둔다
        if s > best:
            best, ff_len = s, round(dur, 2)
    return round(float(min(best, 1.0)), 4), ff_len


def score_impact(t, acc) -> float:
    """조용하던 중에 혼자 튀는 큰 충격. 뛰기·휘두르기처럼 계속 큰 움직임의 최댓값은 세지 않는다."""
    peak_i = int(np.argmax(acc))
    level = _ramp(float(acc[peak_i]), 1.5, IMPACT_G, IMPACT_FULL_G)
    rest = np.abs(t - t[peak_i]) > 0.3
    if rest.sum() < 5:
        return round(level, 4)
    busy = float(np.median(acc[rest]))
    calm = 1.0 if busy <= CALM_G else max(0.0, 1 - (busy - CALM_G) / (BUSY_G - CALM_G))
    return round(level * calm, 4)


def score_struggle(t, acc, gyro) -> float:
    """STRUGGLE_SEC 창을 밀어 가며 팔에 실린 힘(가속도 중앙값)과 회전 속도가 둘 다 큰 구간."""
    best = 0.0
    for i in range(len(t)):
        w = (t >= t[i]) & (t < t[i] + STRUGGLE_SEC)
        if w.sum() < 10 or t[w][-1] - t[i] < STRUGGLE_SEC * 0.8:
            break
        force = _ramp(float(np.median(acc[w])), 1.5, STRUGGLE_ACC_G, STRUGGLE_ACC_G * 1.5)
        spin = _ramp(float(gyro[w].mean()), 5.5, STRUGGLE_GYRO, STRUGGLE_GYRO * 1.7)
        best = max(best, min(force, spin))  # 둘 다 커야 한다 (뛰기는 둘 다 기준 아래, 손목 돌리기는 회전만 큼)
    return round(best, 4)


def analyze_motion(motion: dict | None) -> dict:
    """status: ok / no_data(움직임 없음·샘플 부족 → 종합에서 뺀다)."""
    if not motion:
        return empty_motion_result("no_data")
    parsed = parse_samples(motion)
    if parsed is None:
        r = empty_motion_result("no_data")
        r["samples"] = len(motion.get("samples") or [])
        return r
    t, acc, gyro = parsed
    fall, ff_len = score_fall(t, acc)
    peak = float(acc.max())
    scores = {"fall": fall, "motion_impact": score_impact(t, acc), "struggle": score_struggle(t, acc, gyro)}
    kind = max(scores, key=scores.get)
    score = float(scores[kind])
    return {
        "status": "ok",
        "score": score,
        "type": kind if score >= 0.5 else None,
        "is_danger": score >= 0.5,
        "source": motion.get("source") or "watch",
        "samples": len(t),
        "scores": scores,
        "peak_g": round(peak, 2),
        "free_fall_sec": ff_len,
    }
