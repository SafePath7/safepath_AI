"""
SafePath AI 분석 서버 (SDS M-10)

GET  /health        : 서버·모델 상태
POST /analyze       : 오디오 파일 업로드(폰이 보내는 3초 WAV 등) → 위험도(0~1)·위험 유형
POST /analyze/pcm   : 원시 PCM(16bit LE, mono) 바이트 → 위험도(0~1)·위험 유형
POST /voice-profile : 음성 등록 녹음 → 화자 임베딩·평소 음량·피치 (FD-016, voice_profiles 저장값)

AI_LOAD_MODELS=0이면 모델을 올리지 않고 더미 응답을 돌려준다(연동 확인용).
"""

import logging
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import config
from app.audio import check_pcm16, duration_sec, load_audio_file, load_pcm16, pcm16_duration_sec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("safepath.ai")

models: dict = {}

DUMMY_RESULT = {
    "risk_score": 0.0,
    "risk_type": None,
    "is_danger": False,
    "whisper": {  # 모델 모드와 같은 형식 (audio_detector.voice_result)
        "status": "skipped",  # 더미 모드: 분석하지 않음
        "text": "",
        "score": 0.0,
        "is_danger": False,
        "matches": [],
    },
    "yamnet": {  # 모델 모드와 같은 형식 (audio_detector.empty_sound_result)
        "score": 0.0,
        "type": None,
        "is_danger": False,
        "categories": {"scream": 0.0, "shout": 0.0, "glass_break": 0.0, "impact": 0.0},
        "hits": [],
    },
}

# 더미 모드 음성 프로필: voice_profiles NOT NULL 컬럼을 채울 수 있는 고정값 (연동 확인용, 실제 비교에 쓰면 안 됨)
DUMMY_PROFILE = {
    "model_version": "dummy",
    "embedding": [0.0] * 192,
    "baseline_volume": -25.0,
    "baseline_pitch": 150.0,
    "details": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.LOAD_MODELS:
        from app.detectors.audio_detector import load_models
        from app.detectors.speaker import load_speaker_model

        log.info("모델 로드 중 (Whisper %s, YAMNet, ECAPA 화자 임베딩)...", config.WHISPER_MODEL)
        models.update(load_models())
        models["speaker"] = load_speaker_model()
        log.info("모델 로드 완료")
    else:
        log.info("AI_LOAD_MODELS=0: 모델 없이 더미 모드로 시작")
    yield
    models.clear()


app = FastAPI(title="SafePath AI Server", version="0.1.0", lifespan=lifespan)


def verify_api_key(x_api_key: str | None = Header(default=None)):
    if config.API_KEY and not secrets.compare_digest(
        (x_api_key or "").encode(), config.API_KEY.encode()
    ):
        raise HTTPException(status_code=401, detail="API 키가 올바르지 않습니다.")


def mode() -> str:
    return "model" if models else "dummy"


def max_bytes() -> int:
    return int(config.MAX_UPLOAD_MB * 1024 * 1024)


def too_large() -> HTTPException:
    return HTTPException(status_code=413, detail=f"요청이 {config.MAX_UPLOAD_MB:g}MB를 넘습니다.")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mode": mode(),
        "models_loaded": [k for k in models if k in ("whisper", "yamnet", "speaker")],
    }


def clip_meta(session_id, user_id, seq, captured_at) -> dict:
    """폰이 청크마다 붙이는 식별 정보. 모두 선택값이며 응답에 그대로 돌려준다(user_id는 로그에만)."""
    return {"session_id": session_id, "user_id": user_id, "seq": seq, "captured_at": captured_at}


async def run_analysis(waveform, dur: float | None, source: str, received_bytes: int,
                       meta: dict, started: float) -> dict:
    if models and waveform is not None:
        from app.detectors.audio_detector import analyze_clip

        # 추론은 동기 작업이므로 이벤트 루프를 막지 않게 스레드풀에서 돌린다.
        result = await run_in_threadpool(analyze_clip, models, waveform)
    else:
        result = DUMMY_RESULT
    processing_ms = round((time.perf_counter() - started) * 1000)
    # lag: 녹음 시각 → 분석 완료. 워치·서버 시계 차이가 섞여 있어 참고용(늘어나는지만 본다).
    lag = int(time.time() * 1000) - meta["captured_at"] if meta["captured_at"] else None
    log.info("analyze source=%s session=%s user=%s seq=%s bytes=%d duration=%s mode=%s "
             "risk=%.2f type=%s processing_ms=%d lag_ms=%s",
             source, meta["session_id"], meta["user_id"], meta["seq"], received_bytes, dur, mode(),
             result["risk_score"], result["risk_type"], processing_ms, lag)
    return {
        "session_id": meta["session_id"],
        "seq": meta["seq"],
        "captured_at": meta["captured_at"],
        "mode": mode(),
        "received_bytes": received_bytes,
        "duration_sec": dur,
        "processing_ms": processing_ms,
        **result,
    }


@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None, max_length=64),
    user_id: str | None = Form(default=None, max_length=64),
    seq: int | None = Form(default=None, ge=0),
    captured_at: int | None = Form(default=None, ge=0),
):
    started = time.perf_counter()
    meta = clip_meta(session_id, user_id, seq, captured_at)
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")
    data = await file.read(max_bytes() + 1)
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(data) > max_bytes():
        raise too_large()

    waveform, dur = None, None  # 더미 모드는 디코딩 없이 수신만 확인한다
    if models:
        try:
            waveform = await run_in_threadpool(load_audio_file, data, file.filename)
        except Exception as exc:  # 디코딩 실패
            raise HTTPException(status_code=400, detail=f"오디오를 읽을 수 없습니다: {exc}") from exc
        dur = duration_sec(waveform)
    return await run_analysis(waveform, dur, f"file:{file.filename}", len(data), meta, started)


@app.post("/analyze/pcm", dependencies=[Depends(verify_api_key)])
async def analyze_pcm(
    request: Request,
    sample_rate: int | None = Query(default=None, ge=8000, le=48000),
    session_id: str | None = Query(default=None, max_length=64),
    user_id: str | None = Query(default=None, max_length=64),
    seq: int | None = Query(default=None, ge=0),
    captured_at: int | None = Query(default=None, ge=0),
):
    started = time.perf_counter()
    meta = clip_meta(session_id, user_id, seq, captured_at)
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > max_bytes():
            raise too_large()
    data = bytes(data)
    if not data:
        raise HTTPException(status_code=400, detail="PCM 데이터가 없습니다.")
    sr = sample_rate or config.PCM_SAMPLE_RATE

    try:
        check_pcm16(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    waveform = None  # 더미 모드는 변환(리샘플링) 없이 길이만 계산한다
    if models:
        waveform = await run_in_threadpool(load_pcm16, data, sr)
    dur = pcm16_duration_sec(data, sr)
    return await run_analysis(waveform, dur, "pcm", len(data), meta, started)


@app.post("/voice-profile", dependencies=[Depends(verify_api_key)])
async def voice_profile(file: UploadFile = File(...)):
    """음성 등록 녹음에서 voice_profiles 저장값을 추출한다 (FD-016). 녹음은 저장하지 않는다.

    녹음이 짧거나 작으면 422와 재녹음 사유 코드를 돌려준다 (SDS: 재녹음 안내).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")
    data = await file.read(max_bytes() + 1)
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    if len(data) > max_bytes():
        raise too_large()

    if "speaker" not in models:
        log.info("voice-profile bytes=%d mode=dummy", len(data))
        return {"mode": "dummy", "received_bytes": len(data), **DUMMY_PROFILE}

    from app.detectors.speaker import ProfileError, extract_profile

    try:
        waveform = await run_in_threadpool(load_audio_file, data, file.filename)
    except Exception as exc:  # 디코딩 실패
        raise HTTPException(status_code=400, detail=f"오디오를 읽을 수 없습니다: {exc}") from exc
    try:
        profile = await run_in_threadpool(extract_profile, models["speaker"], waveform)
    except ProfileError as exc:
        log.info("voice-profile 재녹음 필요: %s %s", exc.code, exc.info)
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc), **exc.info}) from exc
    log.info("voice-profile bytes=%d speech=%ss volume=%s pitch=%s", len(data),
             profile["details"]["speech_sec"], profile["baseline_volume"], profile["baseline_pitch"])
    return {"mode": "model", "received_bytes": len(data), **profile}
