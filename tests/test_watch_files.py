"""scripts/watch_files_check.py: 폰 디버그 WAV(voice_<ms>.wav) 정렬·구간 나누기·이어 붙이기."""

import numpy as np

from scripts.watch_files_check import SR, find_files, phone_windows, split_recordings, stitch, write_wav


def jittered_windows(audio):
    """폰 창 + 녹음 시각이 몇 ms씩 흔들리는 실제 상황."""
    windows, stamps = phone_windows(audio, start_ms=1794491370983)
    return windows, [t + i % 3 for i, t in enumerate(stamps)]


def test_stitch_restores_continuous_audio():
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, 7 * SR).astype(np.float32)
    windows, stamps = jittered_windows(audio)
    restored = stitch(windows, stamps)
    assert abs(len(restored) - len(audio)) <= 0.01 * SR
    assert np.allclose(restored[:3 * SR], audio[:3 * SR])  # 3초가 차기 전은 정확히 복원


def test_files_sorted_by_name_number_and_split_on_gap(tmp_path):
    for t in (1794491371980, 1794491370983, 1794491380000, 1794491372981):  # 순서 뒤섞임, 7초 끊김 1번
        write_wav(tmp_path / f"voice_{t}.wav", np.zeros(SR, np.float32))
    (tmp_path / "other.wav").write_bytes(b"")
    files = find_files([str(tmp_path)])
    stamps = [t for t, _ in files]
    assert stamps == sorted(stamps) and len(stamps) == 4
    assert split_recordings(stamps) == [[0, 1, 2], [3]]
