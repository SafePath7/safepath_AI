"""
AI 서버 수신 확인 스크립트 (표준 라이브러리만 사용)

사용:
    python scripts/send_test.py                      # http://localhost:8000
    python scripts/send_test.py --url http://<IP>:8000 --wav sample.wav

1) GET /health
2) POST /analyze/pcm  : 워치 형식(16kHz mono 16bit) 1.5초 무음 청크
3) POST /analyze      : --wav 파일을 지정한 경우 파일 업로드
"""

import argparse
import json
import os
import urllib.request
import uuid

CHUNK_SEC = 1.5
SAMPLE_RATE = 16000


def call(req: urllib.request.Request) -> None:
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        print(f"{req.get_method()} {req.full_url} -> {resp.status}")
        print(json.dumps(body, ensure_ascii=False, indent=2))


def headers(api_key: str | None, extra: dict) -> dict:
    h = dict(extra)
    if api_key:
        h["X-API-Key"] = api_key
    return h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--wav", help="업로드할 오디오 파일 경로")
    p.add_argument("--api-key", default=os.getenv("AI_API_KEY"))
    a = p.parse_args()
    base = a.url.rstrip("/")

    call(urllib.request.Request(f"{base}/health"))

    pcm = bytes(int(CHUNK_SEC * SAMPLE_RATE) * 2)  # 16bit 무음
    call(urllib.request.Request(f"{base}/analyze/pcm", data=pcm, method="POST",
                                headers=headers(a.api_key, {"Content-Type": "application/octet-stream"})))

    if a.wav:
        boundary = uuid.uuid4().hex
        with open(a.wav, "rb") as f:
            data = f.read()
        name = os.path.basename(a.wav)
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
        call(urllib.request.Request(f"{base}/analyze", data=body, method="POST",
                                    headers=headers(a.api_key, {"Content-Type": f"multipart/form-data; boundary={boundary}"})))


if __name__ == "__main__":
    main()
