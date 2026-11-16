"""실제 YAMNet을 적재하는 통합 테스트 (FD-015).

모델 의존성(requirements-models.txt)과 첫 실행 시 다운로드(약 14MB)가 필요하므로 기본으로는 건너뛴다.
실행: AI_RUN_MODEL_TESTS=1 pytest tests/test_yamnet_model.py
"""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("AI_RUN_MODEL_TESTS") != "1", reason="AI_RUN_MODEL_TESTS=1일 때만 실행 (모델 다운로드 필요)"
)


@pytest.fixture(scope="module")
def yamnet():
    pytest.importorskip("tensorflow_hub")
    from app.detectors.audio_detector import load_yamnet, load_yamnet_class_names, resolve_danger_index

    model = load_yamnet()
    names = load_yamnet_class_names(model)
    return model, names, resolve_danger_index(names)


def test_all_danger_classes_exist_in_class_map(yamnet):
    from app.detectors.audio_detector import DANGER_CLASSES

    _, names, index = yamnet
    assert len(names) == 521
    assert [c for c in DANGER_CLASSES if c not in names] == []
    assert sum(len(v) for v in index.values()) == len(DANGER_CLASSES)


@pytest.mark.parametrize("seconds", [0.5, 1.5, 5, 10])
def test_silence_is_not_danger(yamnet, seconds):
    from app.detectors.audio_detector import run_yamnet

    model, _, index = yamnet
    r = run_yamnet(model, index, np.zeros(int(16000 * seconds), dtype=np.float32))
    assert r["is_danger"] is False
    assert r["type"] is None
    assert 0.0 <= r["score"] < 0.3


def test_soft_tone_is_not_danger(yamnet):
    from app.detectors.audio_detector import run_yamnet

    model, _, index = yamnet
    t = np.arange(16000 * 5) / 16000
    r = run_yamnet(model, index, (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
    assert r["is_danger"] is False
