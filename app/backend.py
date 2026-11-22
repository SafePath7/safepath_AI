"""
AI 서버 → 백엔드 호출 (팀 API 명세 "AI 위험 분석" 섹션).

인증: 폰이 AI 서버에 보낸 Authorization(Firebase ID 토큰)을 그대로 넘긴다. AI_BACKEND_KEY를 설정했을 때만
X-Internal-Key를 붙인다(개발 중에는 안 씀, 배포 때 AI 서버 전용 API를 막을 용도). 사용자는 백엔드가 토큰으로 정한다(본문에 userId 없음).

    getVoiceProfileFeatures  GET  /api/voice-profiles/me/features       화자 검증용 프로필 (404 = 미등록)
    registerVoiceProfile     POST /api/voice-profiles                   프로필 + 원본 녹음 (409면 아래로)
    updateVoiceProfile       PUT  /api/voice-profiles/me                재등록
    createEvent              POST /api/risk-events                      위험 이벤트 생성 → id
    uploadEventMedia         POST /api/risk-events/{eventId}/medias     위험 순간 음성(audio)·분석 데이터(sensor)
"""

import json
import logging
import secrets
import urllib.error
import urllib.request

from app import config

log = logging.getLogger("safepath.ai.backend")


class BackendError(Exception):
    """백엔드 호출 실패. status: HTTP 상태 코드 (연결 실패·시간 초과면 None)."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def enabled() -> bool:
    return bool(config.BACKEND_URL)


def multipart(fields: list[tuple[str, str | None, bytes, str]]) -> tuple[bytes, str]:
    """(이름, 파일명 또는 None, 내용, Content-Type) 목록 → (본문, Content-Type 헤더)."""
    boundary = "----safepath" + secrets.token_hex(12)
    out = bytearray()
    for name, filename, data, ctype in fields:
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        out += f"--{boundary}\r\nContent-Disposition: {disp}\r\nContent-Type: {ctype}\r\n\r\n".encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def call(method: str, path: str, token: str, *, json_body=None, fields=None,
         timeout: float | None = None) -> bytes:
    """백엔드를 호출하고 응답 본문을 돌려준다. 2xx가 아니거나 연결에 실패하면 BackendError."""
    if not config.BACKEND_URL:
        raise BackendError("AI_BACKEND_URL이 설정되지 않았습니다.")
    headers = {"Authorization": token, "Accept": "application/json"}
    if config.BACKEND_KEY:
        headers["X-Internal-Key"] = config.BACKEND_KEY
    data = None
    if json_body is not None:
        data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif fields is not None:
        data, headers["Content-Type"] = multipart(fields)
    req = urllib.request.Request(config.BACKEND_URL + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout or config.BACKEND_TIMEOUT_SEC) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise BackendError(f"{method} {path} → {exc.code}", exc.code) from exc
    except (urllib.error.URLError, OSError) as exc:  # 꺼짐·시간 초과
        raise BackendError(f"{method} {path} 연결 실패: {exc}") from exc


# ── 음성 프로필 ─────────────────────────────────────────────────
def fetch_profile(token: str) -> dict | None:
    """getVoiceProfileFeatures. 미등록(404)이면 None."""
    try:
        body = call("GET", "/api/voice-profiles/me/features", token)
    except BackendError as exc:
        if exc.status == 404:
            return None
        raise
    return json.loads(body.decode("utf-8"))


def profile_json(profile: dict) -> dict:
    """POST /api/ai/voice-profiles 추출 결과 → 백엔드 저장 API의 profile 파트 (camelCase)."""
    return {
        "speakerEmbedding": profile["embedding"],
        "baselineVolume": profile["baseline_volume"],
        "baselinePitch": profile["baseline_pitch"],
        "modelVersion": profile["model_version"],
        "baselineDetails": profile.get("details"),
    }


def save_profile(token: str, profile: dict, recordings: list[tuple[str, str, bytes, str]]) -> str:
    """registerVoiceProfile, 이미 있으면(409) updateVoiceProfile. "created" 또는 "updated"를 돌려준다.

    recordings: (파트 이름 file/urgentFile, 파일명, 내용, Content-Type) 원본 녹음.
    """
    body = json.dumps(profile_json(profile)).encode("utf-8")
    fields = [("profile", None, body, "application/json"), *recordings]
    timeout = config.BACKEND_UPLOAD_TIMEOUT_SEC
    try:
        call("POST", "/api/voice-profiles", token, fields=fields, timeout=timeout)
        return "created"
    except BackendError as exc:
        if exc.status != 409:
            raise
    call("PUT", "/api/voice-profiles/me", token, fields=fields, timeout=timeout)
    return "updated"


# ── 위험 이벤트 ─────────────────────────────────────────────────
def create_event(token: str, session_id: str, event_type: str, description: str, detected_at: str) -> int:
    """createEvent. 만든 이벤트 id를 돌려준다."""
    body = call("POST", "/api/risk-events", token, json_body={
        "sessionId": session_id, "eventType": event_type,
        "description": description[:500], "detectedAt": detected_at,
    })
    return int(json.loads(body.decode("utf-8"))["id"])


def upload_media(token: str, event_id: int, media_type: str, filename: str, data: bytes, content_type: str) -> None:
    """uploadEventMedia. media_type: audio(위험 순간 음성) / sensor(위치·움직임·분석 결과 JSON)."""
    call("POST", f"/api/risk-events/{event_id}/medias", token,
         fields=[("file", filename, data, content_type), ("mediaType", None, media_type.encode(), "text/plain")],
         timeout=config.BACKEND_UPLOAD_TIMEOUT_SEC)
