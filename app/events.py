"""
위험 이벤트 보고 (AI 서버 → 백엔드, FD-018·FD-019).

평소 (메모리만):
    폰은 1초마다 "최근 3초" 창을 보낸다. 창끼리 겹치는 부분을 찾아 새로 붙은 소리만 이어 붙이면
    워치가 녹음한 원본과 같은 연속 녹음이 된다(StreamStitcher). 세션마다 직전 EVENT_PREROLL_SEC(기본 10초)만
    메모리에 들고 있고 백엔드에는 보내지 않는다.

의심 시작 (/api/ai/analyze 결과가 위험):
    1) createEvent로 위험 이벤트를 만든다(백엔드에서 PENDING). 만든 id를 폰에 돌려준다 → 폰이 워치 확인 질문.
    2) 직전 소리 + 지금 창을 첫 음성 조각으로, 분석 결과·위치·움직임을 sensor JSON으로 올린다(uploadEventMedia).

의심 단계 (생성 후 EVENT_GROUP_SEC, 기본 30초 = 사용자 확인 대기):
    - 위험 여부와 상관없이 이어 붙인 소리를 약 3초 쌓일 때마다 다음 조각으로, 그때의 sensor JSON과 함께 올린다.
      조각들은 빈틈·겹침 없이 이어지므로 순서대로 붙이면 원본이다.
    - 같은 세션에서 다시 위험이 나오면 새 이벤트를 만들지 않고 같은 id를 돌려준다
      (SDS FD-019 "대기 중 추가 위험 감지 → 같은 질문으로 묶음").
    - 끝나면 직전 소리부터 끝까지 한 파일(event_<id>_full.wav)과 녹음 정보(event_<id>_audio.json)를 올린다.
    보호자 알림은 백엔드가 확인(CONFIRMED) 때 보낸다. AI 서버는 알림을 보내지 않는다.

이어 붙일 수 없는 입력(WAV가 아닌 파일 등)이면 받은 창을 3초마다 그대로 올린다.

event_status (응답): off(백엔드·토큰·세션 없음) / none(위험 아님) / created / grouped / failed(생성 실패)
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app import backend, config
from app.audio import pcm16_to_wav, wav_to_pcm16

log = logging.getLogger("safepath.ai.events")

CLIP_SEC = 3.0  # 폰이 보내는 창 길이. 의심 단계에는 이 간격으로 올린다
CLIP_SLACK_SEC = 0.5  # 조각 길이가 흔들려도 약 3초마다 올리도록 둔 여유
PROBE_BYTES = 3200  # 겹침을 찾을 때 쓰는 새 창 앞부분(16kHz 기준 0.1초)
MAX_RECORDING_BYTES = 9 * 1024 * 1024  # 백엔드 파일 제한(10MB) 안쪽
SESSION_IDLE_SEC = 600  # 이 시간 요청이 없는 세션의 메모리 소리는 지운다

DESCRIPTIONS = {
    "danger_phrase": "위험 문구 감지",
    "scream": "비명 감지",
    "shout": "고함 감지",
    "glass_break": "유리 깨지는 소리 감지",
    "impact": "충격음 감지",
}


def iso_utc(ms: int | None) -> str:
    t = datetime.fromtimestamp(ms / 1000, timezone.utc) if ms else datetime.now(timezone.utc)
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def describe(result: dict) -> str:
    kind = result["risk_type"]
    text = DESCRIPTIONS.get(kind, "위험 감지")
    if kind == "danger_phrase" and result.get("whisper", {}).get("text"):
        text += f": '{result['whisper']['text'][:200]}'"
    return f"{text} (위험도 {result['risk_score']:.2f})"


def sensor_report(result: dict, meta: dict, motion: dict | None, location: dict | None) -> dict:
    """uploadEventMedia(sensor)로 올리는 분석 데이터. 위험 순간의 근거를 보호자·관리자가 볼 수 있게 남긴다."""
    whisper, yamnet, speaker = result.get("whisper") or {}, result.get("yamnet") or {}, result.get("speaker") or {}
    return {
        "sessionId": meta["session_id"],
        "seq": meta["seq"],
        "capturedAt": meta["captured_at"],
        "riskScore": result["risk_score"],
        "riskType": result["risk_type"],
        "voice": {"text": whisper.get("text"), "score": whisper.get("score"), "matches": whisper.get("matches")},
        "sound": {"score": yamnet.get("score"), "categories": yamnet.get("categories")},
        "speaker": {k: speaker.get(k) for k in ("status", "is_user", "similarity", "agitation", "applied")},
        "location": location,
        "motion": motion,
    }


def overlap_bytes(prev: bytes, new: bytes) -> int:
    """prev 끝과 new 앞이 겹치는 길이(바이트). 폰은 이어 붙인 소리의 최근 3초를 그대로 잘라 보내므로 겹침은 바이트까지 같다."""
    probe = new[:PROBE_BYTES]
    if not prev or len(probe) < PROBE_BYTES:
        return 0
    p = prev.find(probe)
    while p != -1:
        m = len(prev) - p
        if p % 2 == 0 and m <= len(new) and new[:m] == prev[p:]:
            return m  # 처음 찾은 자리 = 가장 긴 겹침
        p = prev.find(probe, p + 1)
    return 0


class StreamStitcher:
    """한 세션의 겹치는 3초 창들을 원본 연속 녹음으로 잇는다. 직전 preroll_sec만 들고 있다."""

    def __init__(self):
        self.sr: int | None = None
        self.prev = b""
        self.last_seq: int | None = None
        self.tail = bytearray()  # 직전 소리 (평소에는 메모리에만)

    def feed(self, seq: int | None, pcm: bytes, sr: int) -> tuple[bytes, bool] | None:
        """(새로 붙은 소리, 앞과 끊겼는지). 이미 지난 창(늦게 온 요청)이면 None. 그 소리는 다음 창에 이미 들어 있다."""
        if self.last_seq is not None and seq is not None and seq <= self.last_seq and self.last_seq - seq <= 5:
            return None
        gap = False
        if sr != self.sr:
            if self.sr is not None:
                gap = True
            self.sr, self.prev, self.tail = sr, b"", bytearray()
        m = overlap_bytes(self.prev, pcm)
        if self.prev and m == 0:
            gap = True  # 3초 넘게 빠졌거나 폰이 창을 새로 시작함
        new = pcm[m:]
        self.prev, self.last_seq = pcm, seq
        self.tail += new
        keep = int(config.EVENT_PREROLL_SEC * sr) * 2
        if len(self.tail) > keep:
            del self.tail[:len(self.tail) - keep]
        return new, gap


@dataclass
class _Session:
    lock: threading.Lock = field(default_factory=threading.Lock)
    seen: float = 0.0
    stitch: StreamStitcher = field(default_factory=StreamStitcher)
    # 열려 있는(의심 단계) 이벤트
    event_id: int | None = None
    created: float = 0.0
    token: str | None = None
    rec: bytearray | None = None  # 직전 소리부터 이어 붙인 녹음
    rec_sr: int = 0
    rec_sent: int = 0  # 조각으로 이미 올린 바이트
    rec_start_ms: int | None = None
    parts: list = field(default_factory=list)
    gaps: list = field(default_factory=list)  # 끊긴 자리(녹음 시작부터 초)
    last_clip_ms: int | None = None  # 이어 붙이지 못할 때: 마지막으로 올린 창의 captured_at
    last_clip_mono: float = 0.0
    timer: threading.Timer | None = None
    # 업로드는 세션마다 한 줄로 차례대로 보낸다 (조각 번호 순서 유지, 백엔드에 동시 연결이 몰리지 않게)
    uploader: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(1, thread_name_prefix="event-upload"))


class EventReporter:
    def __init__(self):
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            sessions, self._sessions = list(self._sessions.values()), {}
        for s in sessions:
            if s.timer:
                s.timer.cancel()
            s.uploader.shutdown(wait=False)

    def report(self, result: dict, meta: dict, token: str | None, clip: tuple[str, bytes] | None,
               motion: dict | None = None, location: dict | None = None) -> tuple[int | None, str]:
        """(event_id, event_status). 스레드풀에서 호출한다. 백엔드는 위험이거나 의심 단계일 때만 부른다."""
        session_id = meta["session_id"]
        if not (backend.enabled() and token and session_id):
            return None, "off"
        danger = result["is_danger"]
        with self._lock:
            now = time.monotonic()
            for k in [k for k, s in self._sessions.items()
                      if now - s.seen > SESSION_IDLE_SEC and s.event_id is None]:
                self._sessions.pop(k).uploader.shutdown(wait=False)  # 오래 요청이 없는 세션의 메모리 소리 정리
            sess = self._sessions.setdefault(session_id, _Session())
            sess.seen = now
        jobs: list = []
        with sess.lock:  # 같은 세션의 동시 요청이 이벤트를 두 번 만들거나 소리 순서를 섞지 않게
            now = time.monotonic()
            decoded = wav_to_pcm16(clip[1]) if clip else None
            fed = sess.stitch.feed(meta.get("seq"), *decoded) if decoded else None
            if sess.event_id is not None and now - sess.created > config.EVENT_GROUP_SEC:
                jobs += self._close(sess)  # 의심 단계가 끝남 (타이머보다 요청이 먼저 온 경우)
            if sess.event_id is not None:
                event_id = sess.event_id
                status = "grouped" if danger else "none"
                if fed is not None:
                    jobs += self._extend(sess, fed, meta, result, motion, location)
                elif decoded is None and self._clip_due(sess, meta.get("captured_at"), now):
                    jobs.append(self._raw_job(sess, clip, meta, result, motion, location))
            elif not danger:
                self._run(sess, jobs)
                return None, "none"
            else:
                try:
                    event_id = backend.create_event(token, session_id, result["risk_type"], describe(result),
                                                    iso_utc(meta["captured_at"]))
                except (backend.BackendError, KeyError, ValueError) as exc:
                    log.warning("위험 이벤트 생성 실패 session=%s: %s", session_id, exc)
                    self._run(sess, jobs)
                    return None, "failed"
                status = "created"
                log.info("위험 이벤트 생성 session=%s event=%s type=%s risk=%.2f",
                         session_id, event_id, result["risk_type"], result["risk_score"])
                jobs += self._open(sess, event_id, token, now, decoded, meta, clip, result, motion, location)
        self._run(sess, jobs)
        return (event_id if danger else None), status

    def finish(self, session_id: str, event_id: int) -> None:
        """의심 단계 끝: 남은 조각과 전체 녹음을 올린다. 타이머가 부른다."""
        with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            return
        with sess.lock:
            jobs = self._close(sess) if sess.event_id == event_id else []
        self._run(sess, jobs)

    # ---- 내부: 잠금(sess.lock) 안에서 올릴 일을 만들고, 업로드는 잠금 밖에서 세션 업로드 줄에 넣는다 ----

    def _open(self, sess, event_id, token, now, decoded, meta, clip, result, motion, location) -> list:
        sess.event_id, sess.created, sess.token = event_id, now, token
        sess.parts, sess.gaps = [], []
        sess.last_clip_ms, sess.last_clip_mono = meta.get("captured_at"), now
        session_id = meta["session_id"]
        sess.timer = threading.Timer(config.EVENT_GROUP_SEC, self.finish, args=(session_id, event_id))
        sess.timer.daemon = True
        sess.timer.start()
        if decoded is None:  # 이어 붙이지 못하는 입력: 받은 창 그대로
            sess.rec = None
            return [self._raw_job(sess, clip, meta, result, motion, location)]
        sess.rec, sess.rec_sr, sess.rec_sent = bytearray(sess.stitch.tail), sess.stitch.sr, 0
        if meta.get("captured_at"):
            sess.rec_start_ms = meta["captured_at"] - round(len(sess.rec) / 2 / sess.rec_sr * 1000)
        else:
            sess.rec_start_ms = None
        return [self._part_job(sess, meta, result, motion, location)]

    def _extend(self, sess, fed, meta, result, motion, location) -> list:
        if sess.rec is None:
            return []
        new, gap = fed
        if gap:
            sess.gaps.append(round(len(sess.rec) / 2 / sess.rec_sr, 2))
        if len(sess.rec) + len(new) > MAX_RECORDING_BYTES:
            return []
        sess.rec += new
        if len(sess.rec) - sess.rec_sent >= (CLIP_SEC - CLIP_SLACK_SEC) * sess.rec_sr * 2:
            return [self._part_job(sess, meta, result, motion, location)]
        return []

    def _part_job(self, sess, meta, result, motion, location):
        """녹음에서 아직 안 올린 부분을 다음 조각으로 + 그때의 sensor JSON."""
        n = len(sess.parts) + 1
        name = f"event_{sess.event_id}_audio_{n:02d}.wav"
        sess.parts.append(name)
        audio = pcm16_to_wav(bytes(sess.rec[sess.rec_sent:]), sess.rec_sr)
        sess.rec_sent = len(sess.rec)
        report = json.dumps(sensor_report(result, meta, motion, location), ensure_ascii=False).encode("utf-8")
        sensor = f"event_{sess.event_id}_{meta.get('captured_at') or n}.json"
        return (sess.token, sess.event_id, [("audio", name, audio, "audio/wav"),
                                            ("sensor", sensor, report, "application/json")])

    def _raw_job(self, sess, clip, meta, result, motion, location):
        report = json.dumps(sensor_report(result, meta, motion, location), ensure_ascii=False).encode("utf-8")
        uploads = [("sensor", f"event_{sess.event_id}_{meta.get('captured_at') or 'analysis'}.json", report,
                    "application/json")]
        if clip:
            uploads.insert(0, ("audio", clip[0], clip[1], "audio/wav"))
        return (sess.token, sess.event_id, uploads)

    def _close(self, sess) -> list:
        if sess.timer:
            sess.timer.cancel()
        jobs = []
        if sess.rec is not None and sess.event_id is not None:
            uploads = []
            if len(sess.rec) > sess.rec_sent:  # 마지막 조각
                name = f"event_{sess.event_id}_audio_{len(sess.parts) + 1:02d}.wav"
                sess.parts.append(name)
                uploads.append(("audio", name, pcm16_to_wav(bytes(sess.rec[sess.rec_sent:]), sess.rec_sr), "audio/wav"))
            uploads.append(("audio", f"event_{sess.event_id}_full.wav", pcm16_to_wav(bytes(sess.rec), sess.rec_sr),
                            "audio/wav"))
            info = {
                "eventId": sess.event_id,
                "sampleRate": sess.rec_sr,
                "startedAt": sess.rec_start_ms,
                "durationSec": round(len(sess.rec) / 2 / sess.rec_sr, 2),
                "prerollSec": config.EVENT_PREROLL_SEC,
                "parts": sess.parts,
                "full": f"event_{sess.event_id}_full.wav",
                "gapsAtSec": sess.gaps,
            }
            uploads.append(("sensor", f"event_{sess.event_id}_audio.json",
                            json.dumps(info, ensure_ascii=False).encode("utf-8"), "application/json"))
            jobs.append((sess.token, sess.event_id, uploads))
            log.info("위험 이벤트 %s 녹음 마감: %.1f초, 조각 %d개, 끊김 %d곳",
                     sess.event_id, info["durationSec"], len(sess.parts), len(sess.gaps))
        sess.event_id, sess.timer, sess.rec, sess.token = None, None, None, None
        return jobs

    @staticmethod
    def _clip_due(sess: _Session, captured_at: int | None, now: float) -> bool:
        """이어 붙이지 못할 때: 지난번에 올린 창과 겹치지 않으면(약 3초 뒤) 올리고 기록한다."""
        gap = CLIP_SEC - CLIP_SLACK_SEC
        if captured_at is not None and sess.last_clip_ms is not None:
            due = captured_at - sess.last_clip_ms >= gap * 1000
        else:
            due = now - sess.last_clip_mono >= gap
        if due:
            sess.last_clip_ms, sess.last_clip_mono = captured_at, now
        return due

    def _run(self, sess: _Session, jobs: list) -> None:
        for job in jobs:
            try:
                sess.uploader.submit(self._upload, *job)
            except RuntimeError:  # 세션 정리 직후 (드묾): 따로 보낸다
                threading.Thread(target=self._upload, args=job, daemon=True).start()

    @staticmethod
    def _upload(token: str, event_id: int, uploads: list) -> None:
        for media_type, name, data, ctype in uploads:  # 순서대로 (조각 번호 순서 유지)
            try:
                backend.upload_media(token, event_id, media_type, name, data, ctype)
            except backend.BackendError as exc:  # 이벤트는 이미 있으므로 첨부 실패는 기록만 한다
                log.warning("위험 이벤트 %s 미디어(%s %s) 업로드 실패: %s", event_id, media_type, name, exc)


reporter = EventReporter()
