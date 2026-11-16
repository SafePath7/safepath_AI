"""
내 녹음을 워치 → 폰 경로와 같은 형식으로 바꾼다 (실제 워치 녹음을 받기 전 점검·시연용).

하는 일
    1) 입력 녹음(m4a, mp3, wav 등)을 16kHz 모노 16bit WAV로 변환해 저장
    2) 파일마다 Whisper 인식 텍스트와 위험 문구 판정을 출력 (서버와 같은 analyze_voice)
    3) 녹음들을 1초 무음을 사이에 두고 이어 붙인 뒤, 워치처럼 AAC(16kHz 모노 32kbps)로 압축했다 풀고,
       폰처럼 1초마다 최근 3초를 voice_<녹음시각ms>.wav로 저장
    → 3)의 폴더를 scripts/watch_files_check.py로 분석한다.

사용 (모델 모드 의존성과 ffmpeg 필요):
    python scripts/simulate_watch.py samples/test
    python scripts/simulate_watch.py a.m4a b.m4a --out samples/demo
    python scripts/simulate_watch.py samples/test --no-aac        # 압축 없이 (음질 손실 비교용)
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from scripts.watch_files_check import SR, phone_windows, write_wav  # noqa: E402

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


def find_inputs(paths: list[str]) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXT)
        else:
            files.append(p)
    return files


def aac_roundtrip(x: np.ndarray, ffmpeg: str) -> np.ndarray:
    """워치 VoiceAacEncoder와 같은 설정(AAC-LC 16kHz 모노 32kbps, ADTS)으로 압축했다 푼다."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run([ffmpeg, "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "pipe:0",
                          "-c:a", "aac", "-b:a", "32k", "-f", "adts", "pipe:1"],
                         input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run([ffmpeg, "-loglevel", "error", "-f", "aac", "-i", "pipe:0",
                          "-f", "s16le", "-ar", str(SR), "-ac", "1", "pipe:1"],
                         input=enc, capture_output=True, check=True).stdout
    y = np.frombuffer(dec, "<i2").astype(np.float32) / 32768
    return y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="녹음 파일 또는 폴더")
    ap.add_argument("--out", help="결과 폴더 (기본: 첫 입력 폴더). 아래에 wav/, voice_debug/를 만든다")
    ap.add_argument("--no-aac", action="store_true", help="AAC 압축을 건너뛴다")
    ap.add_argument("--no-model", action="store_true", help="Whisper 인식을 건너뛰고 파일만 만든다")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    inputs = find_inputs(args.paths)
    if not inputs:
        sys.exit("녹음 파일을 찾지 못했습니다.")
    first = Path(args.paths[0])
    out = Path(args.out) if args.out else (first if first.is_dir() else first.parent)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg and not args.no_aac:
        sys.exit("ffmpeg를 찾지 못했습니다. ffmpeg를 PATH에 추가하거나 --no-aac로 실행하세요.")

    from app.audio import load_audio_file

    clips = []
    print(f"## 1) 16kHz 모노 WAV 변환 → {out / 'wav'}")
    for f in inputs:
        x = load_audio_file(f.read_bytes(), f.name)
        write_wav(out / "wav" / f"{f.stem}.wav", x)
        clips.append((f.name, x))
        print(f"   {f.name}: {len(x) / SR:.2f}초")

    if not args.no_model:
        from faster_whisper import WhisperModel

        from app import config
        from app.detectors.audio_detector import analyze_voice, rms_dbfs

        print(f"\n## 2) Whisper({config.WHISPER_MODEL}) 인식·위험 문구 판정 (서버와 같은 처리)")
        model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        for name, x in clips:
            t = time.perf_counter()
            r = analyze_voice(model, x)
            ms = (time.perf_counter() - t) * 1000
            mark = "⚠ 위험" if r["is_danger"] else "  안전"
            print(f"   {mark}  점수 {r['score']:.2f}  '{r['text'] or '(' + r['status'] + ')'}'"
                  f"   [{name}, {rms_dbfs(x):.1f} dBFS, {ms:.0f}ms]")

    gap = np.zeros(SR, np.float32)
    audio = np.concatenate([gap] + [p for _, x in clips for p in (x, gap)])
    if not args.no_aac:
        audio = aac_roundtrip(audio, ffmpeg)
    windows, stamps = phone_windows(audio, start_ms=int(time.time() * 1000))
    debug_dir = out / "voice_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for old in debug_dir.glob("voice_*.wav"):  # 이전 실행 결과와 섞이지 않게
        old.unlink()
    for x, t in zip(windows, stamps):
        write_wav(debug_dir / f"voice_{t}.wav", x)
    how = "AAC 32kbps 압축 후 " if not args.no_aac else ""
    print(f"\n## 3) 워치 형식: {len(audio) / SR:.1f}초 녹음을 {how}3초 창 {len(windows)}개로 → {debug_dir}")
    print(f"\n다음 단계:\n   python scripts/watch_files_check.py \"{debug_dir}\"")


if __name__ == "__main__":
    main()
