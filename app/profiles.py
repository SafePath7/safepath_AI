"""
음성 프로필 조회·보관 (화자 검증용).

폰은 1초마다 요청하므로 매번 백엔드를 부르지 않는다. 귀가 세션마다 한 번 받아 세션이 이어지는 동안 메모리에 둔다.
    - 보관 단위: session_id. 새 세션이면 다시 받는다.
    - AI 서버는 세션 종료를 모른다. 그 세션 요청이 PROFILE_IDLE_SEC(기본 10분) 동안 없으면 끝난 것으로 보고 지운다.
    - AI 서버를 거쳐 재등록하면(save) 그 사용자의 받아 둔 값을 지워 진행 중인 세션에도 바로 반영한다.
조회는 백그라운드에서 하고, 받아 오기 전 요청은 화자 검증 없이 분석한다(분석이 늦어지지 않게).

조회 순서
    1) AI_DEMO_PROFILE 파일 (시연·테스트용. 모든 세션에 같은 프로필, scripts/enroll_profile.py 결과)
    2) 백엔드 getVoiceProfileFeatures (폰이 보낸 Firebase 토큰 + X-Internal-Key, app/backend.py)
       200 → 프로필, 404 → 프로필 없음, 그 밖의 오류 → RETRY_AFTER_ERROR_SEC 뒤 다시 시도

받는 형식. snake_case·camelCase 모두 받는다.
    speaker_embedding(또는 embedding) [192], baseline_volume, baseline_pitch, baseline_details(또는 details), model_version
"""

import base64
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app import backend, config

log = logging.getLogger("safepath.ai.profiles")

RETRY_AFTER_ERROR_SEC = 30.0

_ALIASES = {
    "speaker_embedding": ("speaker_embedding", "speakerEmbedding", "embedding"),
    "baseline_volume": ("baseline_volume", "baselineVolume"),
    "baseline_pitch": ("baseline_pitch", "baselinePitch"),
    "baseline_details": ("baseline_details", "baselineDetails", "details"),
    "model_version": ("model_version", "modelVersion"),
}


def normalize(raw: dict) -> dict:
    """백엔드·파일 형식을 하나로 맞춘다. 필수 값이 없으면 ValueError."""
    out = {}
    for key, names in _ALIASES.items():
        out[key] = next((raw[n] for n in names if raw.get(n) is not None), None)
    if isinstance(out["baseline_details"], str):  # JSONB를 문자열로 줄 때
        out["baseline_details"] = json.loads(out["baseline_details"])
    missing = [k for k in ("speaker_embedding", "baseline_volume", "baseline_pitch", "model_version") if out[k] is None]
    if missing:
        raise ValueError(f"프로필에 값이 없습니다: {missing}")
    out["speaker_embedding"] = [float(x) for x in out["speaker_embedding"]]
    out["baseline_volume"], out["baseline_pitch"] = float(out["baseline_volume"]), float(out["baseline_pitch"])
    return out


def token_subject(token: str | None) -> str | None:
    """Firebase ID 토큰(JWT)의 sub(=Firebase UID). 서명은 확인하지 않는다.

    캐시를 사용자별로 묶는 데만 쓴다. 신원 확인은 토큰을 받은 백엔드가 한다.
    """
    if not token:
        return None
    try:
        payload = token.split()[-1].split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return str(claims.get("sub") or claims.get("user_id") or "") or None
    except (IndexError, ValueError):
        return None


@dataclass
class _Entry:
    token: str | None
    subject: str | None
    status: str = "loading"         # loading / found / no_profile
    profile: dict | None = None
    last_used: float = 0.0          # 마지막 요청 시각 (세션이 끝났는지 판단)
    retry_at: float | None = None   # 조회 오류 후 다시 시도할 시각


class ProfileStore:
    def __init__(self):
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    @staticmethod
    def source() -> str | None:
        if config.DEMO_PROFILE:
            return "demo"
        if backend.enabled():
            return "backend"
        return None

    def get(self, session_id: str | None, token: str | None) -> tuple[str, dict | None]:
        """(상태, 프로필). 상태: off(조회 안 함) / loading / no_profile / found."""
        src = self.source()
        if src is None or (src == "backend" and not (token and session_id)):
            return "off", None
        key = session_id or "-"  # 시연 프로필은 세션 없이도 쓴다
        now = time.monotonic()
        with self._lock:
            self._evict_idle(now)
            e = self._entries.get(key)
            if e is None:
                e = self._entries[key] = _Entry(token, token_subject(token))
                self._start_load(e)
            else:
                if token:
                    e.token = token  # 토큰은 1시간마다 바뀐다. 재시도 때 최신 토큰을 쓴다
                if e.retry_at is not None and e.retry_at <= now:  # 오류 뒤 재시도
                    e.retry_at = None
                    self._start_load(e)
            e.last_used = now
            return e.status, e.profile

    def forget(self, token: str | None) -> None:
        """재등록한 사용자의 받아 둔 프로필을 지워 진행 중인 세션에도 새 값이 쓰이게 한다."""
        subject = token_subject(token)
        with self._lock:
            for k in [k for k, e in self._entries.items()
                      if config.DEMO_PROFILE or (subject and e.subject == subject)]:
                del self._entries[k]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def _evict_idle(self, now: float) -> None:
        idle = [k for k, e in self._entries.items() if e.last_used and now - e.last_used > config.PROFILE_IDLE_SEC]
        for k in idle:
            del self._entries[k]

    def _start_load(self, e: _Entry) -> None:
        threading.Thread(target=self._load, args=(e,), daemon=True).start()

    def _load(self, e: _Entry) -> None:
        try:
            profile = self.fetch(e.token)
            status, retry_at = ("found" if profile else "no_profile"), None
            log.info("voice profile %s", "loaded" if profile else "not registered")
        except Exception as exc:  # 백엔드 꺼짐·형식 오류: 화자 검증 없이 계속, 잠시 뒤 재시도
            log.warning("voice profile 조회 실패: %s", exc)
            profile, status, retry_at = None, "no_profile", time.monotonic() + RETRY_AFTER_ERROR_SEC
        with self._lock:  # 조회는 새 세션이거나 오류 뒤에만 하므로 덮어써도 잃는 값이 없다
            e.profile, e.status, e.retry_at = profile, status, retry_at

    @staticmethod
    def fetch(token: str | None) -> dict | None:
        if config.DEMO_PROFILE:
            path = Path(config.DEMO_PROFILE)
            return normalize(json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else None
        raw = backend.fetch_profile(token)
        return normalize(raw) if raw else None


def save_demo(profile: dict) -> None:
    """백엔드 없이 시연할 때: 추출 결과를 AI_DEMO_PROFILE 파일에 저장한다(원본 녹음은 저장하지 않음)."""
    path = Path(config.DEMO_PROFILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")


store = ProfileStore()
