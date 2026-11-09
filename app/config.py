"""환경변수 기반 설정. 값은 .env.example을 참고해 설정한다."""

import os


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# 모델 적재 여부. 0이면 Whisper·YAMNet을 올리지 않고 더미 응답만 돌려준다(연동 확인용).
LOAD_MODELS: bool = _bool("AI_LOAD_MODELS", False)

# faster-whisper 모델 크기 (tiny / base / small ...)
WHISPER_MODEL: str = os.getenv("AI_WHISPER_MODEL", "base")

# YAMNet 위험 클래스 판정 임계값 (SDS FD-015 초기값 0.3)
YAMNET_THRESHOLD: float = float(os.getenv("AI_YAMNET_THRESHOLD", "0.3"))

# 원시 PCM 입력 기본 샘플레이트 (워치 WatchVoiceRecorder: 16kHz mono 16bit)
PCM_SAMPLE_RATE: int = int(os.getenv("AI_PCM_SAMPLE_RATE", "16000"))

# 설정하면 /analyze 요청에 X-API-Key 헤더가 필요하다 (SDS NFR-SEC-03: 백엔드에서만 호출)
API_KEY: str | None = os.getenv("AI_API_KEY") or None
