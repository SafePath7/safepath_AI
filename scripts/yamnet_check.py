"""
YAMNet 위험 음향 점검 도구 (FD-015 임계값 튜닝용). 서버 없이 오디오 파일을 바로 분석한다.

사용 (모델 의존성 필요: pip install -r requirements-models.txt):
    python scripts/yamnet_check.py 비명.wav 유리.mp3 ...
    python scripts/yamnet_check.py samples/ --chunk 10      # 폴더 전체, 10초 청크로 잘라서
    python scripts/yamnet_check.py a.wav --threshold 0.25 --top-k 3

출력: 파일(청크)별 위험 점수·유형, 유형별 점수, 상위 5개 YAMNet 클래스, 위험 점수가 가장 높은 시각.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app import config  # noqa: E402
from app.audio import TARGET_SR, load_audio_file  # noqa: E402
from app.detectors.audio_detector import (  # noqa: E402
    load_yamnet,
    load_yamnet_class_names,
    resolve_danger_index,
    score_danger_sound,
)

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
FRAME_HOP_SEC = 0.48  # YAMNet 프레임 간격


def iter_files(paths: list[str]):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*") if f.suffix.lower() in AUDIO_EXT)
        else:
            yield p


def chunks(waveform: np.ndarray, chunk_sec: float | None, overlap_sec: float):
    if not chunk_sec:
        yield 0.0, waveform
        return
    size, step = int(chunk_sec * TARGET_SR), int((chunk_sec - overlap_sec) * TARGET_SR)
    for start in range(0, max(1, len(waveform) - int(overlap_sec * TARGET_SR)), step):
        yield start / TARGET_SR, waveform[start:start + size]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    p = argparse.ArgumentParser(description="YAMNet 위험 음향 점수 확인")
    p.add_argument("paths", nargs="+", help="오디오 파일 또는 폴더")
    p.add_argument("--threshold", type=float, default=config.YAMNET_THRESHOLD)
    p.add_argument("--top-k", type=int, default=config.YAMNET_TOP_K)
    p.add_argument("--chunk", type=float, help="이 길이(초)로 잘라 청크별로 분석 (SDS: 5~10초)")
    p.add_argument("--overlap", type=float, default=1.0, help="청크 겹침(초), 기본 1")
    a = p.parse_args()

    print("YAMNet 로드 중...")
    model = load_yamnet()
    names = load_yamnet_class_names(model)
    index = resolve_danger_index(names)
    danger_idx = [i for members in index.values() for _, i in members]
    print(f"threshold={a.threshold}  top_k={a.top_k}  chunk={a.chunk or '전체'}\n")

    for path in iter_files(a.paths):
        try:
            waveform = load_audio_file(path.read_bytes(), path.name)
        except Exception as exc:  # noqa: BLE001
            print(f"[읽기 실패] {path}: {exc}\n")
            continue
        print(f"■ {path.name}  ({len(waveform) / TARGET_SR:.1f}초)")
        for offset, clip in chunks(waveform, a.chunk, a.overlap):
            frame_scores = model(clip)[0].numpy()
            r = score_danger_sound(frame_scores, index, a.threshold, a.top_k)
            mark = "위험" if r["is_danger"] else "정상"
            label = f"  [{offset:5.1f}s~] " if a.chunk else "  "
            print(f"{label}{mark}  score={r['score']:.3f}  type={r['type']}  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in r["categories"].items()))
            peak = int(frame_scores[:, danger_idx].max(axis=1).argmax())
            top5 = frame_scores.mean(axis=0).argsort()[-5:][::-1]
            print(f"      위험 최고 시각 ≈ {offset + peak * FRAME_HOP_SEC:.1f}s | 상위 클래스(평균): "
                  + ", ".join(f"{names[i]} {frame_scores[:, i].mean():.2f}" for i in top5))
        print()


if __name__ == "__main__":
    main()
