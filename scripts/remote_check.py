"""
다른 PC(노트북·백엔드 PC)에서 AI 서버로 요청을 보내 연결과 더미 응답을 확인한다.

표준 라이브러리만 사용하므로 Python 3.7 이상만 있으면 pip 설치 없이 실행된다.
이 파일 하나만 노트북에 복사해서 실행해도 된다.

사용:
    python remote_check.py --url http://172.21.3.30:8000
    python remote_check.py --url http://172.21.3.30:8000 --wav 내녹음.wav
    python remote_check.py --url http://172.21.3.30:8000 --repeat 5
    python remote_check.py --url http://172.21.3.30:8000 --api-key <키>   # 서버에 AI_API_KEY를 설정한 경우

순서:
    1) TCP 연결   : IP·포트까지 닿는지 (방화벽·네트워크 문제를 여기서 구분)
    2) GET  /health
    3) POST /analyze      : 백엔드가 보낼 형식(multipart, 필드 이름 file)으로 wav 업로드
    4) POST /analyze/pcm  : 워치 형식(16kHz mono 16bit) 원시 PCM
각 단계의 상태 코드·응답 형식을 검사하고 마지막에 PASS/FAIL 요약을 출력한다.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import socket
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave

SAMPLE_RATE = 16000
CHUNK_SEC = 1.5  # 워치 WatchVoiceRecorder 청크 길이

RESPONSE_KEYS = {"mode", "received_bytes", "duration_sec", "whisper", "yamnet", "is_danger"}

results: list[tuple[str, bool, str]] = []


# ── 출력 도우미 ────────────────────────────────────────────────
def setup_console() -> None:
    # Windows 콘솔에서 한글이 깨지지 않게 UTF-8로 맞춘다 (Python 3.7+).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass


def title(text: str) -> None:
    print(f"\n=== {text} " + "=" * max(0, 60 - len(text)))


def record(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


# ── 테스트 오디오 생성 ─────────────────────────────────────────
def make_pcm(seconds: float, freq: float = 440.0) -> bytes:
    """16kHz mono 16bit PCM 사인파. 무음이 아닌 실제 소리 데이터로 보낸다."""
    n = int(seconds * SAMPLE_RATE)
    amp = 0.3 * 32767
    return b"".join(
        struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))) for i in range(n)
    )


def make_wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(make_pcm(seconds))
    return buf.getvalue()


def multipart(field: str, filename: str, data: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    return head + data + tail, f"multipart/form-data; boundary={boundary}"


# ── HTTP ───────────────────────────────────────────────────────
def request(method: str, url: str, timeout: float, data: bytes | None = None,
            headers: dict | None = None) -> tuple[int | None, object, float, str]:
    """(상태 코드, 본문(JSON이면 dict), 소요 ms, 오류 설명)을 돌려준다."""
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:  # 4xx/5xx도 응답 본문을 보여준다
        status, raw = e.code, e.read()
    except (urllib.error.URLError, socket.timeout, ConnectionError) as e:
        return None, None, (time.perf_counter() - start) * 1000, str(getattr(e, "reason", e))
    ms = (time.perf_counter() - start) * 1000
    try:
        body = json.loads(raw.decode("utf-8"))
    except ValueError:
        body = raw.decode("utf-8", errors="replace")
    return status, body, ms, ""


def show(status, body, ms) -> None:
    print(f"  -> HTTP {status} ({ms:.0f} ms)")
    text = json.dumps(body, ensure_ascii=False, indent=2) if isinstance(body, dict) else str(body)
    print("  " + text.replace("\n", "\n  "))


def check_analyze_body(name: str, status, body, expect_bytes: int) -> bool:
    if status == 401:
        return record(name, False, "401: 서버에 AI_API_KEY가 설정돼 있습니다. --api-key 로 키를 넘기세요.")
    if status == 413:
        return record(name, False, "413: 서버의 AI_MAX_UPLOAD_MB보다 큰 파일입니다.")
    if status != 200:
        return record(name, False, f"HTTP {status}")
    if not isinstance(body, dict):
        return record(name, False, "JSON 응답이 아닙니다.")
    missing = RESPONSE_KEYS - body.keys()
    if missing:
        return record(name, False, f"응답에 필드가 없습니다: {sorted(missing)}")
    if body["received_bytes"] != expect_bytes:
        return record(name, False, f"보낸 {expect_bytes}바이트 ≠ 서버가 받은 {body['received_bytes']}바이트")
    return record(name, True, f"mode={body['mode']}, received_bytes={body['received_bytes']}, "
                              f"is_danger={body['is_danger']}")


# ── 단계 ───────────────────────────────────────────────────────
def step_tcp(host: str, port: int, timeout: float) -> bool:
    title(f"1) TCP 연결 {host}:{port}")
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            local_ip = s.getsockname()[0]
        print(f"  이 노트북 IP(서버 로그에 찍힐 주소): {local_ip}")
        return record("TCP 연결", True)
    except socket.timeout:
        hint = ("시간 초과: 패킷이 막히고 있습니다. (a) 서버 PC 방화벽에서 8000 포트 인바운드 허용, "
                "(b) 두 기기가 같은 네트워크인지(학교 와이파이는 기기 간 통신을 막는 경우가 많음) 확인하세요.")
    except ConnectionRefusedError:
        hint = ("연결 거부: 서버 PC에는 닿았지만 8000 포트에서 듣는 프로그램이 없습니다. "
                "서버가 켜져 있는지, --host 0.0.0.0 으로 실행했는지 확인하세요 (127.0.0.1이면 외부 접속 불가).")
    except socket.gaierror:
        hint = "주소를 해석할 수 없습니다. --url 의 IP/호스트명을 확인하세요."
    except OSError as e:
        hint = f"{e} (네트워크 경로가 없음: IP 대역이 다르거나 네트워크가 끊겼을 수 있습니다)"
    return record("TCP 연결", False, hint)


def step_health(base: str, timeout: float) -> bool:
    title("2) GET /health")
    status, body, ms, err = request("GET", f"{base}/health", timeout)
    if status is None:
        return record("GET /health", False, err)
    show(status, body, ms)
    ok = status == 200 and isinstance(body, dict) and body.get("status") == "ok"
    return record("GET /health", ok, f"mode={body.get('mode')}" if ok else f"HTTP {status}")


def step_analyze(base: str, timeout: float, auth: dict, wav_path: str | None, repeat: int) -> None:
    title("3) POST /analyze (multipart, 백엔드 형식)")
    if wav_path:
        with open(wav_path, "rb") as f:
            data = f.read()
        filename = os.path.basename(wav_path)
        print(f"  파일: {wav_path} ({len(data)} bytes)")
    else:
        data, filename = make_wav(CHUNK_SEC), "remote_check_tone.wav"
        print(f"  --wav 미지정: {CHUNK_SEC}초 440Hz 사인파 wav를 만들어 보냅니다 ({len(data)} bytes)")
    body, ctype = multipart("file", filename, data, "audio/wav")

    times = []
    for i in range(repeat):
        name = "POST /analyze" + (f" #{i + 1}" if repeat > 1 else "")
        status, resp, ms, err = request("POST", f"{base}/analyze", timeout, body,
                                        {"Content-Type": ctype, **auth})
        if status is None:
            record(name, False, err)
            return
        if i == 0:
            show(status, resp, ms)
        if check_analyze_body(name, status, resp, len(data)):
            times.append(ms)
    if len(times) > 1:
        print(f"  응답 시간: 평균 {sum(times) / len(times):.0f} ms, 최대 {max(times):.0f} ms")


def step_pcm(base: str, timeout: float, auth: dict) -> None:
    title("4) POST /analyze/pcm (워치 형식 16kHz mono 16bit)")
    pcm = make_pcm(CHUNK_SEC)
    status, resp, ms, err = request("POST", f"{base}/analyze/pcm", timeout, pcm,
                                    {"Content-Type": "application/octet-stream", **auth})
    if status is None:
        record("POST /analyze/pcm", False, err)
        return
    show(status, resp, ms)
    if check_analyze_body("POST /analyze/pcm", status, resp, len(pcm)) and resp.get("duration_sec") != CHUNK_SEC:
        record("PCM 길이 계산", False, f"duration_sec={resp.get('duration_sec')} (기대값 {CHUNK_SEC})")


def main() -> int:
    setup_console()
    p = argparse.ArgumentParser(description="원격 PC에서 SafePath AI 서버 연결·더미 응답 확인")
    p.add_argument("--url", required=True, help="AI 서버 주소, 예: http://172.21.3.30:8000")
    p.add_argument("--wav", help="업로드할 wav 파일 (생략하면 테스트 음을 만들어 보냄)")
    p.add_argument("--api-key", default=os.getenv("AI_API_KEY"), help="서버에 AI_API_KEY를 설정한 경우")
    p.add_argument("--repeat", type=int, default=1, help="/analyze 반복 횟수 (응답 시간 측정)")
    p.add_argument("--timeout", type=float, default=10.0, help="요청 타임아웃(초)")
    a = p.parse_args()

    base = a.url.rstrip("/")
    if "://" not in base:
        base = "http://" + base
    parsed = urllib.parse.urlparse(base)
    host, port = parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)
    auth = {"X-API-Key": a.api_key} if a.api_key else {}

    print(f"대상: {base}   (이 PC: {socket.gethostname()}, Python {sys.version.split()[0]})")

    # ngrok 등 https 주소는 TCP 단계를 건너뛰어도 된다. 그 외에는 TCP부터 확인한다.
    if parsed.scheme == "http" and not step_tcp(host, port, a.timeout):
        print("\nTCP 연결이 안 되면 이후 단계도 모두 실패하므로 여기서 멈춥니다.")
    elif step_health(base, a.timeout):
        step_analyze(base, a.timeout, auth, a.wav, max(1, a.repeat))
        step_pcm(base, a.timeout, auth)

    title("요약")
    for name, ok, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    passed = all(ok for _, ok, _ in results) and bool(results)
    print("\n결과: " + ("모두 통과. 서버 창에 이 노트북 IP로 POST /analyze 200 로그가 찍혔는지 확인하세요."
                        if passed else "실패한 단계의 안내 문구를 확인하세요."))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
