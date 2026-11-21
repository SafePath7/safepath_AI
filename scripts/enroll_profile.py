"""
음성 등록 (FD-016) 오프라인 도구. 등록 녹음에서 음성 프로필 JSON을 만든다 (서버의 POST /voice-profile과 같은 계산).

등록은 두 단계다.
    1단계 평소 목소리: 조용한 곳에서 평소 말하듯 문장 읽기 20~30초 → 평소 음량·피치 기준값 + 목소리 특징
    2단계 위급할 때 목소리(선택): 실제 상황처럼 "살려주세요"·"도와주세요"·"놔주세요"를 크고 다급하게 3~5번
          → 다급할 때도 본인 목소리를 알아보도록 목소리 특징에 더한다 (기준값에는 쓰지 않는다)

--urgent를 주면 두 프로필을 만든다. 실제 녹음으로 어느 쪽이 나은지 비교하는 용도다.
    <out>.json      1단계만으로 만든 목소리 특징 (서버 기본값 AI_ENROLL_MIX_URGENT=0과 같음)
    <out>_mix.json  1단계 + 2단계를 합친 목소리 특징

만든 파일은 그대로 화자 검증에 쓸 수 있다.
    - 점검: python scripts/watch_files_check.py <voice_debug 폴더> --profile profiles/7.json
    - 서버: AI_DEMO_PROFILE=<이 JSON 경로> 로 시작하면 모든 세션에 이 프로필을 쓴다 (백엔드 없이 시연용)

사용 (모델 모드 의존성 필요):
    python scripts/enroll_profile.py 평소.m4a --out profiles/7.json
    python scripts/enroll_profile.py 평소.m4a --urgent 살려주세요1.m4a 살려주세요2.m4a --out profiles/7.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402


def load_many(files: list[str]) -> np.ndarray:
    """여러 녹음을 0.5초 무음을 사이에 두고 이어 붙인다."""
    from app.audio import load_audio_file

    gap = np.zeros(8000, np.float32)
    parts = []
    for f in map(Path, files):
        parts += [load_audio_file(f.read_bytes(), f.name), gap]
    return np.concatenate(parts)


def save(profile: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="1단계 평소 목소리 녹음 (여러 개면 이어 붙인다)")
    ap.add_argument("--urgent", nargs="+", metavar="FILE", help="2단계 위급할 때 목소리 녹음 (선택)")
    ap.add_argument("--out", required=True, help="저장할 JSON 경로 (예: profiles/me.json)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from app.detectors.speaker import ProfileError, extract_profile, load_speaker_model

    calm = load_many(args.files)
    urgent = load_many(args.urgent) if args.urgent else None
    print(f"1단계 {len(args.files)}개 {len(calm) / 16000:.1f}초"
          + (f", 2단계 {len(args.urgent)}개 {len(urgent) / 16000:.1f}초" if urgent is not None else "")
          + ". 화자 모델 로드 중...", flush=True)
    model = load_speaker_model()
    try:
        profile = extract_profile(model, calm, urgent, mix_urgent=False)
        mixed = extract_profile(model, calm, urgent, mix_urgent=True) if urgent is not None else None
    except ProfileError as exc:
        sys.exit(f"재녹음 필요 ({exc.code}): {exc} {exc.info}")

    d = profile["details"]
    print(f"평소 목소리: 말소리 {d['speech_sec']}초, 음량 {profile['baseline_volume']} dBFS "
          f"({d['volume_p10']}~{d['volume_p90']}), 피치 {profile['baseline_pitch']} Hz ({d['pitch_p10']}~{d['pitch_p90']})")
    out = Path(args.out)
    save(profile, out)
    print(f"저장: {out}  (1단계만, model_version {profile['model_version']})")
    if mixed is not None:
        u = d["urgent"]
        ratio = f"x{u['pitch_ratio']}" if u["pitch_ratio"] else "측정 불가"
        print(f"위급할 때 목소리: 말소리 {u['speech_sec']}초, 평소 대비 피치 {ratio}, 음량 {u['volume_delta_db']:+} dB, "
              f"평소 목소리와 유사도 {u['similarity_to_calm']}")
        mix_path = out.with_name(f"{out.stem}_mix{out.suffix}")
        save(mixed, mix_path)
        print(f"저장: {mix_path}  (1단계 + 2단계)")
        print("\n비교: 같은 voice_debug 폴더를 두 프로필로 돌려 '유사도' 열을 비교하세요.")
        print(f"   python scripts/watch_files_check.py <폴더> --profile {out}")
        print(f"   python scripts/watch_files_check.py <폴더> --profile {mix_path}")


if __name__ == "__main__":
    main()
