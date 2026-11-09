"""환경변수 기반 설정. 값은 .env.example을 참고해 설정한다."""

import os


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

# YAMNet 위험 클래스 판정 임계값 (SDS FD-015 초기값 0.3)
YAMNET_THRESHOLD: float = float(_str("AI_YAMNET_THRESHOLD", "0.3"))

# 원시 PCM 입력 기본 샘플레이트 (워치 WatchVoiceRecorder: 16kHz mono 16bit)
PCM_SAMPLE_RATE: int = int(_str("AI_PCM_SAMPLE_RATE", "16000"))

# 요청 본문 최대 크기(MB). 백엔드 미디어 업로드 제한(10MB)과 맞춘다.
MAX_UPLOAD_MB: float = float(_str("AI_MAX_UPLOAD_MB", "10"))

# 설정하면 /analyze 요청에 X-API-Key 헤더가 필요하다 (SDS NFR-SEC-03: 백엔드에서만 호출)
API_KEY: str | None = os.getenv("AI_API_KEY", "").strip() or None
