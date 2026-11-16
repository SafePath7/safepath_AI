"""환경변수 기반 설정. 값은 .env.example을 참고해 설정한다."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str) -> str:
    """값이 없거나 빈 문자열이면 기본값을 쓴다 (.env의 `KEY=` 같은 빈 줄 대비)."""
    value = os.getenv(name, "").strip()
    return value or default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


# 모델 적재 여부. 0이면 Whisper·YAMNet을 올리지 않고 더미 응답만 돌려준다(연동 확인용).
LOAD_MODELS: bool = _bool("AI_LOAD_MODELS", False)

# faster-whisper 모델 크기 (tiny / base / small ...)
WHISPER_MODEL: str = _str("AI_WHISPER_MODEL", "base")

# 음성 위험 판정 임계값 (FD-014). 위험 문구 점수가 이 값 이상이면 위험.
VOICE_THRESHOLD: float = float(_str("AI_VOICE_THRESHOLD", "0.5"))

# 무음 판정 음량(dBFS). 클립 전체 RMS가 이보다 작으면 Whisper를 돌리지 않는다 (FD-014 무음 → 분석 생략).
SILENCE_DBFS: float = float(_str("AI_SILENCE_DBFS", "-50"))

# Whisper hotwords: 위험 문구 쪽으로 디코딩을 유도한다. 한국어 평가에서 탐지율은 올랐지만 짧은 청크의 일상 대화 오탐(최대 29%)이 생겨 기본으로 끈다.
WHISPER_HOTWORDS: bool = _bool("AI_WHISPER_HOTWORDS", False)

# 음성 프로필 등록(FD-016) 녹음 품질 기준. 미달이면 422로 재녹음 안내.
#   말소리 최소 길이(초): 화자 임베딩은 3초 이상이면 동작하지만 평소 피치·음량을 안정적으로 재려면 더 필요하다.
ENROLL_MIN_SPEECH_SEC: float = float(_str("AI_ENROLL_MIN_SPEECH_SEC", "8"))
#   말소리 음량 중앙값 최솟값(dBFS): 휴대폰을 입에서 멀리 둔 녹음은 대략 -40 dBFS 아래로 떨어진다.
ENROLL_MIN_DBFS: float = float(_str("AI_ENROLL_MIN_DBFS", "-40"))

# YAMNet 위험 클래스 판정 임계값 (SDS FD-015 초기값 0.3). 점수가 이 값 이상이면 위험.
YAMNET_THRESHOLD: float = float(_str("AI_YAMNET_THRESHOLD", "0.3"))

# YAMNet 프레임(0.96초 창, 0.48초 간격) 점수를 클립 점수로 합칠 때 상위 몇 프레임을 평균할지.
# 전체 평균은 10초 청크(20프레임)에서 0.5초짜리 유리 파손음을 1/10 이하로 희석시키므로 상위 k개만 쓴다.
# k=2(약 1초)는 한 프레임짜리 순간 잡음에 반응하지 않으면서 짧은 사건은 잡는 절충값이다.
YAMNET_TOP_K: int = max(1, int(_str("AI_YAMNET_TOP_K", "2")))

# 모델 가중치 캐시 폴더 (Git 제외). Windows 임시 폴더에 두면 디스크 정리 때 파일만 지워지고
# 빈 폴더가 남아 TF Hub가 "saved_model.pb 없음" 오류로 실패하므로 프로젝트 안에 둔다.
MODEL_CACHE_DIR: Path = Path(_str("AI_MODEL_CACHE_DIR", str(REPO_ROOT / "models")))

# 원시 PCM 입력 기본 샘플레이트 (워치 WatchVoiceRecorder: 16kHz mono 16bit)
PCM_SAMPLE_RATE: int = int(_str("AI_PCM_SAMPLE_RATE", "16000"))

# 요청 본문 최대 크기(MB). 백엔드 미디어 업로드 제한(10MB)과 맞춘다.
MAX_UPLOAD_MB: float = float(_str("AI_MAX_UPLOAD_MB", "10"))

# 설정하면 /analyze 요청에 X-API-Key 헤더가 필요하다 (SDS NFR-SEC-03: 백엔드에서만 호출)
API_KEY: str | None = os.getenv("AI_API_KEY", "").strip() or None
