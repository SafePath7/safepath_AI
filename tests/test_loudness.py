"""갑자기 큰 소리 (app/detectors/loudness.py): 배경보다 갑자기 커진 정도로 판정."""

import numpy as np
import pytest

from app.detectors.loudness import BackgroundTracker, analyze_loudness, frame_dbfs, loud_score

SR = 16000


def noise(seconds, dbfs, seed=0):
    rnd = np.random.default_rng(seed)
    x = rnd.standard_normal(int(SR * seconds))
    return (x / np.sqrt(np.mean(x * x)) * 10 ** (dbfs / 20)).astype(np.float32)


def window(bg_db, burst_db=None, burst_sec=0.3, at=1.5):
    x = noise(3.0, bg_db)
    if burst_db is not None:
        i = int(at * SR)
        x[i:i + int(burst_sec * SR)] = noise(burst_sec, burst_db, seed=1)
    return x


def test_frame_levels():
    db = frame_dbfs(noise(1.0, -30))
    assert len(db) == 10 and np.allclose(db, -30, atol=0.5)
    assert frame_dbfs(np.zeros(SR, dtype=np.float32)).max() == -90.0


def test_sudden_loud_bang_is_danger():
    # 평소 말소리 기준 -22dBFS보다 22dB 큰 -0dBFS 근처 → 1점
    r = analyze_loudness(window(-55, -0.5), background=BackgroundTracker())
    assert r["is_danger"] is True and r["score"] == pytest.approx(1.0, abs=0.05)
    assert r["jump_db"] == pytest.approx(54.5, abs=1.5) and r["peak_at_sec"] == pytest.approx(1.5, abs=0.11)


def test_normal_speech_level_is_not_loud():
    # 워치는 평소 말소리의 큰 순간도 -9dBFS 근처라, 조용한 곳에서 말을 시작하면 배경보다는 크게 튄다
    r = analyze_loudness(window(-50, -9, burst_sec=1.0), background=BackgroundTracker())
    assert r["jump_db"] > 30 and r["is_danger"] is False and r["score"] < 0.25


def test_profile_speech_level_moves_the_bar():
    loud_for_quiet_person = analyze_loudness(window(-55, -8), background=BackgroundTracker(), speech_dbfs=-30)
    same_for_loud_person = analyze_loudness(window(-55, -8), background=BackgroundTracker(), speech_dbfs=-18)
    assert loud_for_quiet_person["is_danger"] is True and same_for_loud_person["is_danger"] is False


def test_steady_noise_and_quiet_talk_are_not_danger():
    assert analyze_loudness(window(-25), background=BackgroundTracker())["is_danger"] is False  # 시끄러운 길 그대로
    # 조용한 곳에서 평소 말소리: 배경보다 15dB 이상 튀어도 절대 음량이 작으면 아님
    r = analyze_loudness(window(-65, -45, burst_sec=1.0), background=BackgroundTracker())
    assert r["jump_db"] > 15 and r["score"] == 0.0


def test_score_scale():
    # 갑자기(배경 대비)와 크게(평소 말소리 대비) 중 작은 쪽
    assert loud_score(15, 30) == 0.5 and loud_score(25, 30) == 1.0 and loud_score(7.5, 30) == 0.25
    assert loud_score(40, 16) == 0.5 and loud_score(40, 22) == 1.0 and loud_score(40, 12) == 0.0
    assert loud_score(-3, 30) == 0.0


def test_session_background_catches_sustained_scream():
    # 3초 내내 큰 소리면 창 안에서는 배경을 알 수 없다 → 세션의 평소 음량과 비교
    tr = BackgroundTracker()
    assert analyze_loudness(window(-3), background=tr)["is_danger"] is False  # 세션 기록 없음
    tr.clear()
    for i in range(3):
        analyze_loudness(window(-55), "s1", background=tr)
    r = analyze_loudness(window(-3), "s1", background=tr)
    assert r["is_danger"] is True and r["background_dbfs"] == pytest.approx(-55, abs=1.5)
    # 큰 소리가 난 창은 배경에 넣지 않으니 다음 창도 같은 기준
    assert analyze_loudness(window(-3), "s1", background=tr)["is_danger"] is True
    assert analyze_loudness(window(-3), "s2", background=tr)["is_danger"] is False  # 다른 세션은 따로


def test_digital_silence_is_ignored_for_background():
    # 워치 녹음 앞부분·말 사이의 완전 무음(0)을 배경으로 쓰면 모든 말소리가 '80dB 튐'이 된다
    x = window(-40, -9, burst_sec=1.0)
    x[: int(1.4 * SR)] = 0.0
    r = analyze_loudness(x, background=BackgroundTracker())
    assert r["background_dbfs"] > -70 and r["is_danger"] is False


def test_too_short_is_silence():
    assert analyze_loudness(np.zeros(1000, dtype=np.float32))["status"] == "silence"
    assert analyze_loudness(np.zeros(SR * 3, dtype=np.float32), background=BackgroundTracker())["status"] == "silence"
