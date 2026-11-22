"""
SafePath AI 분석 서버 (SDS M-10). 팀 API 명세 "AI 위험 분석 (석현)" 참고.

GET  /health                : 서버·모델 상태
POST /api/ai/analyze        : analyzeRisk. 폰이 보내는 3초 WAV + 움직임·위치 → 위험도(0~1)·위험 유형.
                              위험이면 백엔드 createEvent·uploadEventMedia 호출 후 eventId 반환
POST /api/ai/analyze/pcm    : 위와 같고 원시 PCM(16bit LE, mono)을 받는다 (개발·테스트용)
POST /api/ai/voice-profiles : extractVoiceProfile. 음성 등록 녹음 → 화자 임베딩·평소 음량·피치 (FD-016)
                              → 백엔드 registerVoiceProfile(이미 있으면 updateVoiceProfile)로 저장

인증: 폰이 보낸 Authorization(Firebase ID 토큰)을 백엔드 호출에 그대로 넘긴다 (app/backend.py).
AI_LOAD_MODELS=0이면 모델을 올리지 않고 더미 응답을 돌려준다(연동 확인용).
"""

import json
import logging
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import backend, config
from app.audio import check_pcm16, duration_sec, load_audio_file, load_pcm16, pcm16_duration_sec, pcm16_to_wav
from app.events import reporter as event_reporter
from app.profiles import save_demo
from app.profiles import store as profile_store

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
    "signals": [],
    "loudness": {  # 모델 모드와 같은 형식 (loudness.empty_loudness_result)
        "status": "skipped", "score": 0.0, "is_danger": False,
        "peak_dbfs": None, "background_dbfs": None, "jump_db": None, "speech_dbfs": None, "over_speech_db": None,
        "peak_at_sec": None,
    },
    "motion": {  # 모델 모드와 같은 형식 (motion.empty_motion_result)
        "status": "skipped", "score": 0.0, "type": None, "is_danger": False, "source": None, "samples": 0,
        "scores": {"fall": 0.0, "motion_impact": 0.0, "struggle": 0.0}, "peak_g": None, "free_fall_sec": None,
    },
    "speaker": {  # 모델 모드와 같은 형식 (speaker.verify_result)
        "status": "off",
        "similarity": None,
        "is_user": None,
        "pitch_ratio": None,
        "volume_delta_db": None,
        "agitation": None,
        "boost": 0.0,
        "applied": 0.0,
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
        log.info("백엔드 연동: %s, 시연 프로필: %s", config.BACKEND_URL or "-", config.DEMO_PROFILE or "-")
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


def clip_meta(session_id, seq, captured_at) -> dict:
    """폰이 청크마다 붙이는 식별 정보. 모두 선택값이며 응답에 그대로 돌려준다."""
    return {"session_id": session_id, "seq": seq, "captured_at": captured_at}


def parse_motion(raw: str | None) -> dict | None:
    """motion: JSON {"rate_hz": 20, "source": "watch"|"phone", "samples": [[t_ms, ax, ay, az, gx, gy, gz], ...]} (FD-017)."""
    if not raw:
        return None
    try:
        motion = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="motion이 올바른 JSON이 아닙니다.") from exc
    if not isinstance(motion, dict) or not isinstance(motion.get("samples", []), list):
        raise HTTPException(status_code=400, detail='motion은 {"rate_hz", "samples": [...]} 형식이어야 합니다.')
    return motion


def location_of(lat, lng, accuracy_m, location_at) -> dict | None:
    if lat is None or lng is None:
        return None
    return {"lat": lat, "lng": lng, "accuracy_m": accuracy_m, "location_at": location_at}


def loud_log(r: dict | None) -> str:
    if not r or r.get("status") != "ok":
        return "-"
    return f"{r['score']:.2f}(+{r['jump_db']}dB,{r['peak_dbfs']}dBFS)"


def motion_log(r: dict | None, raw: dict | None) -> str:
    if not r or r.get("status") != "ok":
        return f"-({len(raw.get('samples', []))})" if raw else "-"
    return f"{r['score']:.2f}({r['samples']},{r['peak_g']}g)"


def speaker_log(sp: dict) -> str:
    if sp["status"] != "ok":
        return sp["status"]
    return f"sim={sp['similarity']:.2f},agit={sp['agitation']:.2f},+{sp['applied']:.2f}"


async def run_analysis(waveform, dur: float | None, source: str, received_bytes: int, meta: dict, started: float,
                       token: str | None, clip: tuple[str, bytes] | None,
                       motion: dict | None = None, location: dict | None = None) -> dict:
    if models and waveform is not None:
        from app.detectors.audio_detector import analyze_clip

        # 화자 검증용 프로필: 귀가 세션마다 한 번 받아 둔다. 받아 오는 중이면 검증 없이 분석한다.
        profile_status, profile = profile_store.get(meta["session_id"], token)
        # 추론은 동기 작업이므로 이벤트 루프를 막지 않게 스레드풀에서 돌린다.
        result = await run_in_threadpool(analyze_clip, models, waveform, profile, profile_status,
                                         meta["session_id"], motion)
    else:
        result = DUMMY_RESULT
    # 위험이면 백엔드에 위험 이벤트를 만들고(같은 세션 30초 안이면 묶음) id를 돌려준다
    event_id, event_status = await run_in_threadpool(
        event_reporter.report, result, meta, token, clip, motion, location)
    processing_ms = round((time.perf_counter() - started) * 1000)
    # lag: 녹음 시각 → 분석 완료. 워치·서버 시계 차이가 섞여 있어 참고용(늘어나는지만 본다).
    lag = int(time.time() * 1000) - meta["captured_at"] if meta["captured_at"] else None
    log.info("analyze source=%s session=%s seq=%s bytes=%d duration=%s mode=%s "
             "risk=%.2f type=%s loud=%s motion=%s speaker=%s event=%s/%s location=%s processing_ms=%d lag_ms=%s",
             source, meta["session_id"], meta["seq"], received_bytes, dur, mode(),
             result["risk_score"], result["risk_type"], loud_log(result.get("loudness")),
             motion_log(result.get("motion"), motion), speaker_log(result["speaker"]), event_status, event_id,
             "yes" if location else "-", processing_ms, lag)
    return {
        "session_id": meta["session_id"],
        "seq": meta["seq"],
        "captured_at": meta["captured_at"],
        "mode": mode(),
        "received_bytes": received_bytes,
        "duration_sec": dur,
        "processing_ms": processing_ms,
        "event_id": event_id,
        "event_status": event_status,
        **result,
    }


@app.post("/api/ai/analyze", dependencies=[Depends(verify_api_key)])
async def analyze(
    file: UploadFile = File(...),
    session_id: str | None = Form(default=None, max_length=64),
    seq: int | None = Form(default=None, ge=0),
    captured_at: int | None = Form(default=None, ge=0),
    motion: str | None = Form(default=None),
    lat: float | None = Form(default=None, ge=-90, le=90),
    lng: float | None = Form(default=None, ge=-180, le=180),
    accuracy_m: float | None = Form(default=None, ge=0),
    location_at: int | None = Form(default=None, ge=0),
    authorization: str | None = Header(default=None),
):
    """analyzeRisk: 폰이 1초마다 보내는 최근 3초 음성 + 움직임·위치 → 위험도·위험 유형 (+ 위험 이벤트 id)."""
    started = time.perf_counter()
    meta = clip_meta(session_id, seq, captured_at)
    motion_data = parse_motion(motion)
    location = location_of(lat, lng, accuracy_m, location_at)
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")
    data = await read_upload(file)
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    waveform, dur = None, None  # 더미 모드는 디코딩 없이 수신만 확인한다
    if models:
        try:
            waveform = await run_in_threadpool(load_audio_file, data, file.filename)
        except Exception as exc:  # 디코딩 실패
            raise HTTPException(status_code=400, detail=f"오디오를 읽을 수 없습니다: {exc}") from exc
        dur = duration_sec(waveform)
    return await run_analysis(waveform, dur, f"file:{file.filename}", len(data), meta, started,
                              authorization, (file.filename, data), motion_data, location)


@app.post("/api/ai/analyze/pcm", dependencies=[Depends(verify_api_key)])
async def analyze_pcm(
    request: Request,
    sample_rate: int | None = Query(default=None, ge=8000, le=48000),
    session_id: str | None = Query(default=None, max_length=64),
    seq: int | None = Query(default=None, ge=0),
    captured_at: int | None = Query(default=None, ge=0),
    authorization: str | None = Header(default=None),
):
    """analyzeRisk와 같고 원시 PCM(16bit LE, mono)을 받는다. 개발·테스트용."""
    started = time.perf_counter()
    meta = clip_meta(session_id, seq, captured_at)
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
    clip = (f"voice_{captured_at or 'pcm'}.wav", pcm16_to_wav(data, sr))
    return await run_analysis(waveform, dur, "pcm", len(data), meta, started, authorization, clip)


async def read_upload(f: UploadFile) -> bytes:
    data = await f.read(max_bytes() + 1)
    if len(data) > max_bytes():
        raise too_large()
    return data


@app.post("/api/ai/voice-profiles", dependencies=[Depends(verify_api_key)])
async def voice_profile(
    file: UploadFile = File(...),
    urgent_file: UploadFile | None = File(default=None),
    authorization: str | None = Header(default=None),
):
    """extractVoiceProfile: 음성 등록 녹음 → 화자 임베딩·평소 음량·피치 (FD-016) → 백엔드에 저장.

    file: 1단계 평소 목소리(20~30초). urgent_file: 2단계 위급할 때 목소리(선택, "살려주세요" 등을 크고 다급하게).
    저장: 백엔드 registerVoiceProfile(이미 있으면 updateVoiceProfile)에 추출 결과와 원본 녹음을 보낸다.
          Authorization(Firebase 토큰)이 필요하다. 백엔드 미설정이면 AI_DEMO_PROFILE 파일, 그것도 없으면 저장 안 함.
    녹음이 짧거나 작으면 422와 재녹음 사유 코드를 돌려준다 (SDS: 재녹음 안내). 이때는 저장하지 않는다.
    AI 서버는 녹음을 보관하지 않는다.
    """
    if backend.enabled() and not authorization:
        raise HTTPException(status_code=401, detail={
            "code": "no_token", "message": "로그인 토큰(Authorization)이 필요합니다."})
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")
    data = await read_upload(file)
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")
    urgent_data = None
    if urgent_file is not None and urgent_file.filename:
        urgent_data = await read_upload(urgent_file) or None

    if "speaker" not in models:
        log.info("voice-profile bytes=%d mode=dummy", len(data))
        profile, result_mode = dict(DUMMY_PROFILE), "dummy"
    else:
        from app.detectors.speaker import ProfileError, extract_profile

        try:
            waveform = await run_in_threadpool(load_audio_file, data, file.filename)
            urgent = None
            if urgent_data:
                urgent = await run_in_threadpool(load_audio_file, urgent_data, urgent_file.filename)
        except Exception as exc:  # 디코딩 실패
            raise HTTPException(status_code=400, detail=f"오디오를 읽을 수 없습니다: {exc}") from exc
        try:
            profile = await run_in_threadpool(extract_profile, models["speaker"], waveform, urgent)
        except ProfileError as exc:
            log.info("voice-profile 재녹음 필요: %s %s", exc.code, exc.info)
            raise HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc), **exc.info}) from exc
        log.info("voice-profile bytes=%d speech=%ss volume=%s pitch=%s", len(data),
                 profile["details"]["speech_sec"], profile["baseline_volume"], profile["baseline_pitch"])
        result_mode = "model"

    saved_to = None
    if backend.enabled():
        recordings = [("file", file.filename, data, file.content_type or "application/octet-stream")]
        if urgent_data:
            recordings.append(("urgentFile", urgent_file.filename, urgent_data,
                               urgent_file.content_type or "application/octet-stream"))
        try:
            action = await run_in_threadpool(backend.save_profile, authorization, profile, recordings)
        except backend.BackendError as exc:
            log.warning("voice-profile 백엔드 저장 실패: %s", exc)
            if exc.status in (401, 403, 404):  # 토큰 오류·미가입: 앱이 로그인 처리를 하도록 그대로 전달
                raise HTTPException(status_code=exc.status, detail={
                    "code": "backend_rejected", "message": "로그인 정보를 확인할 수 없습니다. 다시 로그인해 주세요."}) from exc
            raise HTTPException(status_code=502, detail={
                "code": "save_failed", "message": "음성 프로필을 저장하지 못했습니다. 잠시 후 다시 시도해 주세요."}) from exc
        profile_store.forget(authorization)
        saved_to = f"backend:{action}"
    elif config.DEMO_PROFILE:
        await run_in_threadpool(save_demo, profile)
        profile_store.forget(None)
        saved_to = "demo"
    log.info("voice-profile saved_to=%s", saved_to)
    return {"mode": result_mode, "received_bytes": len(data), "saved_to": saved_to, **profile}
