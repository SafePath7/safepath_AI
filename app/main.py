"""
SafePath AI 분석 서버 (SDS M-10)

GET  /health        : 서버·모델 상태
POST /analyze       : 오디오 파일 업로드(wav, mp3, m4a 등) → 위험 판단
POST /analyze/pcm   : 원시 PCM(16bit LE, mono) 바이트 → 위험 판단 (워치 청크 형식)

AI_LOAD_MODELS=0이면 모델을 올리지 않고 더미 응답을 돌려준다(연동 확인용).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile

from app import config
from app.audio import duration_sec, load_audio_file, load_pcm16

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("safepath.ai")

models: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.LOAD_MODELS:
        from app.detectors.audio_detector import load_models

        log.info("모델 로드 중 (Whisper %s, YAMNet)...", config.WHISPER_MODEL)
        models.update(load_models())
        log.info("모델 로드 완료")
    else:
        log.info("AI_LOAD_MODELS=0: 모델 없이 더미 모드로 시작")
    yield
    models.clear()


app = FastAPI(title="SafePath AI Server", version="0.1.0", lifespan=lifespan)


def verify_api_key(x_api_key: str | None = Header(default=None)):
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="API 키가 올바르지 않습니다.")


def mode() -> str:
    return "model" if models else "dummy"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": mode(),
        "models_loaded": [k for k in models if k != "yamnet_classes"],
    }


def run_analysis(waveform, source: str, received_bytes: int) -> dict:
    dur = duration_sec(waveform) if waveform is not None else None
    if models and waveform is not None:
        from app.detectors.audio_detector import analyze_clip

        result = analyze_clip(models, waveform)
    else:
        result = {
            "whisper": {"text": "", "is_danger": False},
            "yamnet": {"hits": [], "is_danger": False},
            "is_danger": False,
        }
    log.info("analyze source=%s bytes=%d duration=%s mode=%s is_danger=%s",
             source, received_bytes, dur, mode(), result["is_danger"])
    return {"mode": mode(), "received_bytes": received_bytes, "duration_sec": dur, **result}


@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    waveform = None  # 더미 모드는 디코딩 없이 수신만 확인한다
    if models:
        try:
            waveform = load_audio_file(data, file.filename)
        except Exception as exc:  # 디코딩 실패
            raise HTTPException(status_code=400, detail=f"오디오를 읽을 수 없습니다: {exc}") from exc
    return run_analysis(waveform, source=f"file:{file.filename}", received_bytes=len(data))


@app.post("/analyze/pcm", dependencies=[Depends(verify_api_key)])
async def analyze_pcm(request: Request, sample_rate: int = Query(default=None, ge=8000, le=48000)):
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail="PCM 데이터가 없습니다.")
    try:
        waveform = load_pcm16(data, sample_rate or config.PCM_SAMPLE_RATE)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return run_analysis(waveform, source="pcm", received_bytes=len(data))
