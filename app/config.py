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
#   2단계 '위급할 때 목소리' 녹음(선택)의 최소 말소리 길이(초). "살려주세요" 3~5번이면 2~4초.
ENROLL_URGENT_MIN_SPEECH_SEC: float = float(_str("AI_ENROLL_URGENT_MIN_SPEECH_SEC", "1.5"))
#   2단계 녹음 임베딩을 평소 목소리 임베딩과 합쳐 저장할지. 실제 사람 녹음으로 효과를 확인하기 전까지 끈다.
ENROLL_MIX_URGENT: bool = _bool("AI_ENROLL_MIX_URGENT", False)

# YAMNet 위험 클래스 판정 임계값 (SDS FD-015 초기값 0.3). 점수가 이 값 이상이면 위험.
YAMNET_THRESHOLD: float = float(_str("AI_YAMNET_THRESHOLD", "0.3"))

# YAMNet 프레임(0.96초 창, 0.48초 간격) 점수를 클립 점수로 합칠 때 상위 몇 프레임을 평균할지.
# 전체 평균은 10초 청크(20프레임)에서 0.5초짜리 유리 파손음을 1/10 이하로 희석시키므로 상위 k개만 쓴다.
# k=2(약 1초)는 한 프레임짜리 순간 잡음에 반응하지 않으면서 짧은 사건은 잡는 절충값이다.
YAMNET_TOP_K: int = max(1, int(_str("AI_YAMNET_TOP_K", "2")))
# 위험 음향 판정 방식 (TBD-5): loudness = 갑자기 큰 소리(app/detectors/loudness.py), YAMNet은 소리 이름만 붙임
#                               yamnet = 예전 방식(YAMNet 위험 클래스 점수로 판정)
SOUND_MODE: str = _str("AI_SOUND_MODE", "loudness").lower()
# 갑자기 큰 소리: 배경보다 LOUD_JUMP_DB(+LOUD_SPAN_DB면 1점) 그리고 평소 말소리보다 LOUD_OVER_SPEECH_DB 커야 위험
# 평소 말소리 = 음성 프로필 baseline_volume, 없으면 LOUD_SPEECH_REF_DBFS (워치 실측 말소리 구간 중앙값 근처)
LOUD_JUMP_DB: float = float(_str("AI_LOUD_JUMP_DB", "15"))
LOUD_SPAN_DB: float = float(_str("AI_LOUD_SPAN_DB", "10"))
LOUD_OVER_SPEECH_DB: float = float(_str("AI_LOUD_OVER_SPEECH_DB", "16"))
LOUD_SPEECH_REF_DBFS: float = float(_str("AI_LOUD_SPEECH_REF_DBFS", "-22"))
# 큰 소리에 YAMNet 이름(scream 등)을 붙이는 최소 점수. 못 넘으면 loud_sound
SOUND_LABEL_MIN: float = float(_str("AI_SOUND_LABEL_MIN", "0.15"))

# 화자 검증 결합 (음성 프로필이 있을 때). 본인 목소리가 평소보다 높고 크면 위험도를 최대 VERIFY_BOOST_MAX만큼 올린다.
#   점수를 깎거나 판정을 막지는 않는다. 다급한 목소리는 본인이어도 유사도가 크게 떨어지기 때문이다(FD-016 평가).
VERIFY_SIM_THRESHOLD: float = float(_str("AI_VERIFY_SIM_THRESHOLD", "0.4"))   # 이 이상이면 본인
VERIFY_BOOST_MAX: float = float(_str("AI_VERIFY_BOOST_MAX", "0.15"))         # 위험도에 더하는 최댓값
VERIFY_BOOST_MIN_BASE: float = float(_str("AI_VERIFY_BOOST_MIN_BASE", "0.3"))  # 위험도가 이 이상일 때만 더한다
VERIFY_MIN_SPEECH_SEC: float = float(_str("AI_VERIFY_MIN_SPEECH_SEC", "1.0"))  # 말소리가 이보다 짧으면 검증 생략
VERIFY_PITCH_SPAN: float = float(_str("AI_VERIFY_PITCH_SPAN", "0.3"))          # 평소 상단보다 피치 x0.3 더 오르면 흥분도 1
VERIFY_VOLUME_SPAN_DB: float = float(_str("AI_VERIFY_VOLUME_SPAN_DB", "6"))    # 평소 상단보다 6dB 더 크면 흥분도 1

# 백엔드 연동 (app/backend.py). 폰이 보낸 Firebase 토큰을 그대로 넘긴다. BACKEND_KEY는 선택(설정하면 X-Internal-Key 헤더, 개발 중에는 비워 둠).
#   BACKEND_URL을 비우면 백엔드를 부르지 않는다(프로필 조회·위험 이벤트 생성·프로필 저장 모두 생략).
BACKEND_URL: str | None = os.getenv("AI_BACKEND_URL", "").strip().rstrip("/") or None
BACKEND_KEY: str | None = os.getenv("AI_BACKEND_KEY", "").strip() or None
BACKEND_TIMEOUT_SEC: float = float(_str("AI_BACKEND_TIMEOUT_SEC", "3"))            # 조회·이벤트 생성
BACKEND_UPLOAD_TIMEOUT_SEC: float = float(_str("AI_BACKEND_UPLOAD_TIMEOUT_SEC", "10"))  # 녹음·미디어 업로드
# 시연용 음성 프로필 파일. 설정하면 백엔드 대신 모든 세션에 이 프로필을 쓰고, 등록 결과도 여기에 저장한다.
DEMO_PROFILE: str | None = os.getenv("AI_DEMO_PROFILE", "").strip() or None
# 프로필은 귀가 세션(session_id) 동안 보관한다. AI 서버는 세션 종료를 모르므로, 그 세션 요청이 이 시간(초) 동안 없으면 지운다.
PROFILE_IDLE_SEC: float = float(_str("AI_PROFILE_IDLE_SEC", "600"))

# 종합 위험도 임계값 (SDS FD-018 TBD-5). risk_score가 이 값 이상이면 is_danger.
# risk_score는 모델별 점수를 '각자의 임계값 = 0.5'로 맞춘 값이라, 0.5면 모델별 임계값과 같은 판정이다 (app/risk.py).
RISK_THRESHOLD: float = float(_str("AI_RISK_THRESHOLD", "0.5"))
# 여러 신호 종합(FD-018): 이 값 이상인 신호끼리만 noisy-OR로 합친다(작은 잡음 신호가 쌓여 위험이 되지 않게)
COMBINE_MIN: float = float(_str("AI_COMBINE_MIN", "0.25"))
# 위험 이벤트를 만든 뒤 이 시간(초) 안에 같은 세션에서 다시 위험이 나오면 같은 이벤트로 묶는다 (FD-019, TBD-6 확인 대기 시간).
EVENT_GROUP_SEC: float = float(_str("AI_EVENT_GROUP_SEC", "30"))
# 위험 이벤트를 만들 때 함께 올리는 직전 소리 길이(초). 평소에는 세션마다 이만큼만 메모리에 들고 있다.
EVENT_PREROLL_SEC: float = float(_str("AI_EVENT_PREROLL_SEC", "10"))

# 모델 가중치 캐시 폴더 (Git 제외). Windows 임시 폴더에 두면 디스크 정리 때 파일만 지워지고
# 빈 폴더가 남아 TF Hub가 "saved_model.pb 없음" 오류로 실패하므로 프로젝트 안에 둔다.
MODEL_CACHE_DIR: Path = Path(_str("AI_MODEL_CACHE_DIR", str(REPO_ROOT / "models")))

# 원시 PCM 입력 기본 샘플레이트 (워치 WatchVoiceRecorder: 16kHz mono 16bit)
PCM_SAMPLE_RATE: int = int(_str("AI_PCM_SAMPLE_RATE", "16000"))

# 요청 본문 최대 크기(MB). 백엔드 미디어 업로드 제한(10MB)과 맞춘다.
MAX_UPLOAD_MB: float = float(_str("AI_MAX_UPLOAD_MB", "10"))

# 설정하면 /api/ai/** 요청에 X-API-Key 헤더가 필요하다 (개발용 고정 키).
# 사용자 인증은 폰이 보낸 Firebase 토큰을 백엔드가 확인한다 (AI 서버는 그대로 전달).
API_KEY: str | None = os.getenv("AI_API_KEY", "").strip() or None
