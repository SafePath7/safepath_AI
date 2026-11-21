"""
워치 실녹음 점검 도구. 폰 디버그 빌드가 저장한 voice_<녹음시각ms>.wav 파일들을 시간순으로 분석한다.

폰(VoiceDebugRecorder)은 AI 서버로 보낼 것과 똑같은 3초 WAV를 약 1초마다 저장한다.
    - 파일 이름 숫자 = 그 창의 마지막 1초 조각 녹음 시각(Unix epoch ms). 작을수록 먼저 녹음된 것.
    - 이웃한 파일은 2초씩 겹친다. 그대로 이어 붙이면 같은 소리가 3번 나온다.
    - 폰은 녹음이 2.5초 넘게 끊기면 창을 비운다. 이 도구도 같은 기준으로 녹음 구간을 나눈다.

사용:
    python scripts/watch_files_check.py voice_debug/                 # 시간순 분석표 (모델 필요)
    python scripts/watch_files_check.py voice_debug/ --no-model      # 순서·간격·길이만 확인
    python scripts/watch_files_check.py voice_debug/ --stitch out/   # 구간별 연속 녹음 wav도 만든다
    python scripts/watch_files_check.py voice_debug/ --csv result.csv
    python scripts/watch_files_check.py voice_debug/ --profile profiles/7.json   # 화자 검증 포함
"""

import argparse
import csv
import datetime as dt
import os
import re
import sys
import wave
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

SR = 16000
WINDOW_SEC = 3.0
RESET_GAP_MS = 2500  # 폰 VoiceWindowBuffer.maxGapMs와 같은 값
NAME = re.compile(r"voice_(\d{10,})\.wav$", re.IGNORECASE)


def find_files(paths: list[str]) -> list[tuple[int, Path]]:
    """voice_<ms>.wav 파일을 (녹음시각ms, 경로)로 모아 시간순으로 정렬한다. 같은 시각은 하나만 남긴다."""
    found = {}
    for p in map(Path, paths):
        for f in (sorted(p.rglob("*.wav")) if p.is_dir() else [p]):
            m = NAME.search(f.name)
            if m:
                found.setdefault(int(m.group(1)), f)
    return sorted(found.items())


def split_recordings(stamps: list[int], reset_gap_ms: int = RESET_GAP_MS) -> list[list[int]]:
    """시각 목록을 녹음 구간으로 나눈다. 이전 파일과 reset_gap_ms 넘게 벌어지면 새 구간."""
    groups: list[list[int]] = []
    for i, t in enumerate(stamps):
        if i == 0 or t - stamps[i - 1] > reset_gap_ms:
            groups.append([])
        groups[-1].append(i)
    return groups


def read_wav(path: Path) -> np.ndarray:
    """폰이 만든 16kHz 모노 16bit WAV를 float32로 읽는다. 다른 형식이면 librosa로 변환한다."""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SR, 1, 2):
            return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    from app.audio import load_audio_file
    return load_audio_file(path.read_bytes(), path.name)


def stitch(windows: list[np.ndarray], stamps: list[int]) -> np.ndarray:
    """겹치는 3초 창들을 하나의 연속 녹음으로 되돌린다 (한 녹음 구간 안의 창만 넣는다).

    폰은 '이전 창 + 새 조각'의 뒤 3초를 저장하므로, 창이 3초가 차기 전에는 길이 차이가 곧 새 조각이다.
    3초가 찬 뒤에는 새 조각 길이를 알 수 없어 녹음 시각 차이로 어림한다(AAC 때문에 조각 길이가 조금씩 다름).
    """
    if not windows:
        return np.zeros(0, np.float32)
    parts = [windows[0]]
    full = int(WINDOW_SEC * SR)
    for i in range(1, len(windows)):
        prev, cur = windows[i - 1], windows[i]
        if len(prev) < full and len(cur) > len(prev):  # 이전 창이 3초가 안 찼으면 늘어난 만큼이 새 조각
            added = len(cur) - len(prev)
        else:
            added = round((stamps[i] - stamps[i - 1]) * SR / 1000)
        added = max(0, min(added, len(cur)))
        if added:
            parts.append(cur[-added:])
    return np.concatenate(parts).astype(np.float32)


def phone_windows(audio: np.ndarray, start_ms: int, chunk_sec: float = 1.0):
    """폰 VoiceWindowBuffer와 같은 방식: 1초 조각을 이어 붙이고 뒤 3초만 남긴 창 목록과 각 창의 녹음 시각(ms)."""
    n = int(chunk_sec * SR)
    win = np.zeros(0, np.float32)
    windows, stamps = [], []
    for i, k in enumerate(range(0, len(audio), n)):
        win = np.concatenate([win, audio[k:k + n]])[-int(WINDOW_SEC * SR):]
        windows.append(win.copy())
        stamps.append(start_ms + i * int(chunk_sec * 1000))
    return windows, stamps


def write_wav(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def clock(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S.%f")[:-3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="voice_*.wav 파일 또는 폴더")
    ap.add_argument("--no-model", action="store_true", help="모델 없이 순서·간격·길이만 확인")
    ap.add_argument("--stitch", metavar="DIR", help="녹음 구간별 연속 wav를 이 폴더에 저장")
    ap.add_argument("--csv", metavar="FILE", help="분석표를 CSV로 저장")
    ap.add_argument("--profile", metavar="JSON", help="음성 프로필(scripts/enroll_profile.py 결과)로 화자 검증도 함께")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    files = find_files(args.paths)
    if not files:
        sys.exit("voice_<숫자>.wav 파일을 찾지 못했습니다.")
    stamps = [t for t, _ in files]
    groups = split_recordings(stamps)
    first = dt.datetime.fromtimestamp(stamps[0] / 1000)
    print(f"파일 {len(files)}개, {first:%Y-%m-%d}, 녹음 구간 {len(groups)}개\n")

    models = None
    if not args.no_model:
        from app.detectors.audio_detector import analyze_clip, load_models
        print("모델 로드 중...", flush=True)
        models = load_models()
    profile = None
    if args.profile and models is not None:
        import json

        from app.detectors.speaker import load_speaker_model
        from app.profiles import normalize
        profile = normalize(json.loads(Path(args.profile).read_text(encoding="utf-8")))
        models["speaker"] = load_speaker_model()
        print(f"화자 검증: {args.profile} (평소 피치 {profile['baseline_pitch']} Hz, 음량 {profile['baseline_volume']} dBFS)")

    rows = []
    for g, idx in enumerate(groups, 1):
        span = (stamps[idx[-1]] - stamps[idx[0]]) / 1000
        print(f"── 구간 {g}: {clock(stamps[idx[0]])} ~ {clock(stamps[idx[-1]])} ({len(idx)}개, 약 {span + 1:.0f}초)")
        print(f"   {'녹음 시각':12s} {'간격ms':>6s} {'길이s':>5s}" + ("" if models is None else
              f" {'위험도':>6s} {'유형':13s} {'처리ms':>6s}" + ("" if profile is None else
              f" {'유사도':>5s} {'흥분':>4s} {'가산':>5s}") + "  인식 텍스트"))
        windows = []
        for k, i in enumerate(idx):
            t, path = files[i]
            x = read_wav(path)
            windows.append(x)
            gap = "" if k == 0 else str(t - stamps[i - 1])
            row = {"group": g, "file": path.name, "captured_at": t, "time": clock(t), "gap_ms": gap,
                   "duration_sec": round(len(x) / SR, 2)}
            line = f"   {clock(t):12s} {gap:>6s} {len(x) / SR:>5.2f}"
            if models is not None:
                import time
                t0 = time.perf_counter()
                r = analyze_clip(models, x, profile, "found" if profile else "off")
                ms = round((time.perf_counter() - t0) * 1000)
                mark = "⚠" if r["is_danger"] else " "
                text = r["whisper"]["text"] or f"({r['whisper']['status']})"
                line += f" {r['risk_score']:>6.2f} {str(r['risk_type'] or '-'):13s} {ms:>6d}"
                if profile is not None:
                    v = r["speaker"]
                    if v["status"] == "ok":
                        user = "본인" if v["is_user"] else "타인"
                        line += f" {v['similarity']:>5.2f} {v['agitation']:>4.2f} {v['applied']:>+5.2f}"
                        text = f"[{user}] {text}"
                    else:
                        line += f" {'-':>5s} {'-':>4s} {'-':>5s}"
                    row.update(speaker_status=v["status"], similarity=v["similarity"], is_user=v["is_user"],
                               pitch_ratio=v["pitch_ratio"], volume_delta_db=v["volume_delta_db"],
                               agitation=v["agitation"], boost_applied=v["applied"])
                line += f" {mark} {text}"
                row.update(risk_score=r["risk_score"], risk_type=r["risk_type"], is_danger=r["is_danger"],
                           voice_score=r["whisper"]["score"], sound_score=r["yamnet"]["score"],
                           sound_type=r["yamnet"]["type"], text=r["whisper"]["text"], processing_ms=ms)
            print(line)
            rows.append(row)
        if args.stitch:
            out = Path(args.stitch) / f"recording_{g:02d}_{stamps[idx[0]]}.wav"
            audio = stitch(windows, [stamps[i] for i in idx])
            write_wav(out, audio)
            print(f"   → 연속 녹음 {len(audio) / SR:.1f}초: {out}")
        print()

    if models is not None:
        danger = [r for r in rows if r["is_danger"]]
        print(f"위험 판정 {len(danger)}/{len(rows)}개 창")
    if args.csv:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"CSV 저장: {args.csv}")


if __name__ == "__main__":
    main()
