"""공용 픽스처: 팀 API 명세의 백엔드 API를 흉내 내는 가짜 백엔드."""

import json
import threading
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from app import config

PROFILE = {"model_version": "fd016-v1/ecapa-voxceleb-192", "embedding": [0.1] * 192,
           "baseline_volume": -20.5, "baseline_pitch": 152.8, "details": {"pitch_p90": 193.0}}
BACKEND_PROFILE = {"speakerEmbedding": [0.1] * 192, "baselineVolume": -20.5, "baselinePitch": 152.8,
                   "baselineDetails": {"pitch_p90": 193.0}, "modelVersion": "fd016-v1/ecapa-voxceleb-192"}


def parse_multipart(ctype: str, body: bytes) -> dict:
    msg = BytesParser(policy=HTTP).parsebytes(f"Content-Type: {ctype}\r\n\r\n".encode() + body)
    return {part.get_param("name", header="content-disposition"): (part.get_filename(), part.get_payload(decode=True))
            for part in msg.iter_parts()}


class FakeBackend:
    """요청을 기록하고, 토큰별로 정한 상태를 돌려준다.

    profiles: 토큰 → 프로필(없으면 404). status: (method, path) → 강제 상태 코드. registered: 이미 등록한 토큰(POST 409).
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.profiles: dict[str, dict] = {}
        self.registered: set[str] = set()
        self.status: dict[tuple[str, str], int] = {}
        self.next_event_id = 100
        self.url = ""

    def handle(self, h: BaseHTTPRequestHandler, method: str):
        body = h.rfile.read(int(h.headers.get("Content-Length") or 0))
        token = h.headers.get("Authorization")
        call = {"method": method, "path": h.path, "token": token, "key": h.headers.get("X-Internal-Key"),
                "ctype": h.headers.get("Content-Type"), "body": body}
        self.calls.append(call)
        code, out = self.respond(method, h.path, token)
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.end_headers()
        if out is not None:
            h.wfile.write(json.dumps(out).encode())

    def respond(self, method, path, token):
        if (method, path) in self.status:
            return self.status[(method, path)], None
        if method == "GET" and path == "/api/voice-profiles/me/features":
            return (200, self.profiles[token]) if token in self.profiles else (404, None)
        if method == "POST" and path == "/api/voice-profiles":
            if token in self.registered:
                return 409, None
            self.registered.add(token)
            return 201, {"registered": True}
        if method == "PUT" and path == "/api/voice-profiles/me":
            return 200, {"registered": True}
        if method == "POST" and path == "/api/risk-events":
            self.next_event_id += 1
            return 201, {"id": self.next_event_id, "status": "PENDING"}
        if method == "POST" and path.startswith("/api/risk-events/") and path.endswith("/medias"):
            return 201, {"id": 1}
        return 404, None

    def find(self, method, path_prefix):
        return [c for c in self.calls if c["method"] == method and c["path"].startswith(path_prefix)]


@pytest.fixture
def fake_backend(monkeypatch):
    fb = FakeBackend()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            fb.handle(self, "GET")

        def do_POST(self):
            fb.handle(self, "POST")

        def do_PUT(self):
            fb.handle(self, "PUT")

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    fb.url = f"http://127.0.0.1:{srv.server_port}"
    monkeypatch.setattr(config, "BACKEND_URL", fb.url)
    monkeypatch.setattr(config, "BACKEND_KEY", "internal-k")
    monkeypatch.setattr(config, "DEMO_PROFILE", None)
    yield fb
    srv.shutdown()
