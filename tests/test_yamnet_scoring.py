"""FD-015 위험 음향 점수 계산 테스트. YAMNet 없이 프레임 점수 행렬을 직접 만들어 검증한다."""

import logging

import numpy as np
import pytest

from app.detectors.audio_detector import (
    DANGER_CATEGORIES,
    DANGER_CLASSES,
    aggregate_frames,
    resolve_danger_index,
    score_danger_sound,
)

THRESHOLD = 0.3
TOP_K = 2

# 실제 YAMNet처럼 위험 클래스 사이에 다른 클래스가 섞인 class map
CLASS_NAMES = ["Speech", "Music"] + DANGER_CLASSES + ["Silence"]
INDEX = resolve_danger_index(CLASS_NAMES)


def frames(n: int, events: dict[str, list[float]] | None = None) -> np.ndarray:
    """n프레임짜리 점수 행렬. events={"클래스명": [프레임별 점수...]}는 앞 프레임부터 채운다."""
    m = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)
    m[:, CLASS_NAMES.index("Speech")] = 0.8  # 위험과 무관한 배경 클래스
    for name, values in (events or {}).items():
        m[: len(values), CLASS_NAMES.index(name)] = values
    return m


def score(m: np.ndarray, threshold: float = THRESHOLD, top_k: int = TOP_K) -> dict:
    return score_danger_sound(m, INDEX, threshold, top_k)


def test_all_below_threshold_is_not_danger():
    r = score(frames(20, {"Screaming": [0.1] * 20, "Shatter": [0.2] * 20}))
    assert r["is_danger"] is False
    assert r["type"] is None  # SDS: 모든 위험 클래스 임계값 미만 → 위험 없음
    assert r["hits"] == []
    assert r["score"] == pytest.approx(0.2)


def test_short_event_in_long_chunk_is_detected():
    # 10초 청크(20프레임) 중 2프레임(약 1초)만 비명. 전체 평균이면 0.07로 놓친다.
    m = frames(20, {"Screaming": [0.7, 0.7]})
    assert m[:, CLASS_NAMES.index("Screaming")].mean() < THRESHOLD
    r = score(m)
    assert r["is_danger"] is True
    assert r["type"] == "scream"
    assert r["score"] == pytest.approx(0.7)
    assert r["hits"] == [{"class": "Screaming", "category": "scream", "score": 0.7}]


def test_single_frame_spike_is_ignored():
    # 한 프레임(0.96초 창)짜리 순간 잡음은 상위 2개 평균에서 절반으로 줄어 임계값 미만
    r = score(frames(20, {"Thump, thud": [0.5]}))
    assert r["is_danger"] is False
    assert r["categories"]["impact"] == pytest.approx(0.25)


def test_threshold_is_inclusive():
    r = score(frames(10, {"Shout": [0.3, 0.3]}))
    assert r["is_danger"] is True
    assert r["type"] == "shout"


def test_category_score_is_max_of_its_classes():
    r = score(frames(10, {"Glass": [0.35, 0.35], "Shatter": [0.6, 0.6], "Breaking": [0.4, 0.4]}))
    assert r["categories"]["glass_break"] == pytest.approx(0.6)
    assert r["type"] == "glass_break"
    assert [h["class"] for h in r["hits"]] == ["Shatter", "Breaking", "Glass"]  # 점수 내림차순


def test_type_is_highest_category_when_several_fire():
    r = score(frames(10, {"Screaming": [0.4, 0.4], "Smash, crash": [0.9, 0.9]}))
    assert r["type"] == "impact"
    assert r["score"] == pytest.approx(0.9)
    assert {h["category"] for h in r["hits"]} == {"scream", "impact"}


def test_categories_always_have_all_four_types():
    r = score(frames(5))
    assert set(r["categories"]) == set(DANGER_CATEGORIES) == {"scream", "shout", "glass_break", "impact"}


def test_short_clip_with_fewer_frames_than_top_k():
    # 0.5초 클립은 YAMNet 프레임이 1개뿐 → 그 프레임 점수를 그대로 쓴다
    r = score(frames(1, {"Screaming": [0.5]}))
    assert r["is_danger"] is True
    assert r["score"] == pytest.approx(0.5)


def test_empty_frames():
    r = score(np.zeros((0, len(CLASS_NAMES)), dtype=np.float32))
    assert r["is_danger"] is False and r["score"] == 0.0


def test_aggregate_frames_top_k_mean():
    m = np.array([[0.1], [0.9], [0.5], [0.3]], dtype=np.float32)
    assert aggregate_frames(m, 1)[0] == pytest.approx(0.9)
    assert aggregate_frames(m, 2)[0] == pytest.approx(0.7)
    assert aggregate_frames(m, 10)[0] == pytest.approx(0.45)  # k > 프레임 수 → 전체 평균


def test_missing_class_names_are_skipped_with_warning(caplog):
    names = [n for n in CLASS_NAMES if n != "Bellow"]
    with caplog.at_level(logging.WARNING):
        index = resolve_danger_index(names)
    assert "Bellow" in caplog.text
    assert [n for n, _ in index["shout"]] == ["Shout", "Yell"]
    assert all(names[i] == n for members in index.values() for n, i in members)
