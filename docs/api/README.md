# AI 서버 API

현재 형식은 10주차 연동 확인용입니다. 위험도(0~1)·위험 유형을 포함한 최종 형식은 11주차에 백엔드와 합의한 뒤 이 문서를 갱신합니다.

## 공통

- 기본 주소: `http://localhost:8000`
- 호출 주체: 백엔드(SDS M-03 세션 모듈). 앱·워치는 AI 서버를 직접 호출하지 않습니다.
- `AI_API_KEY`가 설정된 경우 `/analyze`, `/analyze/pcm`에 `X-API-Key` 헤더가 필요합니다. 없거나 다르면 401입니다.
- 요청 본문이 `AI_MAX_UPLOAD_MB`(기본 10MB, 백엔드 미디어 업로드 제한과 동일)를 넘으면 413입니다.

## GET /health

```json
{"status": "ok", "mode": "dummy", "models_loaded": []}
```

모델 모드에서는 `"mode": "model"`, `"models_loaded": ["whisper", "yamnet"]`입니다.

## POST /analyze/pcm

워치·폰이 수집한 원시 오디오 청크를 그대로 보냅니다. 워치(`WatchVoiceRecorder`)의 현재 형식은 16kHz, mono, 16bit PCM, 1.5초 청크입니다.

- 본문: `application/octet-stream`, 16bit little-endian PCM 바이트
- 쿼리: `sample_rate`(선택, 8000~48000, 기본 16000). 16kHz가 아니면 모델 모드에서 리샘플링합니다. 더미 모드는 변환 없이 바이트 길이로 `duration_sec`만 계산합니다.
- 바이트 길이가 홀수이거나 본문이 비면 400입니다.

```sh
curl -X POST "http://localhost:8000/analyze/pcm" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @chunk.pcm
```

## POST /analyze

- 본문: `multipart/form-data`, 필드 이름 `file` (wav, mp3, m4a 등)
- 더미 모드에서는 디코딩하지 않고 수신 바이트만 확인하므로 `duration_sec`이 `null`입니다.

```sh
curl -X POST "http://localhost:8000/analyze" -F "file=@clip.wav"
```

## 응답 (두 엔드포인트 공통)

```json
{
  "mode": "model",
  "received_bytes": 48000,
  "duration_sec": 1.5,
  "whisper": {"text": "살려주세요", "is_danger": true},
  "yamnet": {
    "score": 0.4495,
    "type": "glass_break",
    "is_danger": true,
    "categories": {"scream": 0.003, "shout": 0.001, "glass_break": 0.4495, "impact": 0.0385},
    "hits": [
      {"class": "Glass", "category": "glass_break", "score": 0.4495},
      {"class": "Breaking", "category": "glass_break", "score": 0.3008}
    ]
  },
  "is_danger": true
}
```

| 필드 | 설명 |
|---|---|
| `mode` | `dummy`면 모델 없이 고정값(위험 없음)을 돌려준 것. 형식은 모델 모드와 같습니다. |
| `received_bytes` | 받은 바이트 수 |
| `duration_sec` | 16kHz 기준 오디오 길이(초) |
| `whisper.text` / `whisper.is_danger` | 인식 텍스트, 위험 문구 포함 여부 |
| `yamnet.score` | 위험 음향 점수 0~1 (FD-015). 4개 유형 점수 중 최댓값 |
| `yamnet.type` | 위험 음향 유형. 위험이 아니면 `null` |
| `yamnet.is_danger` | `score`가 임계값(기본 0.3) 이상이면 `true` |
| `yamnet.categories` | 유형별 점수. 항상 4개 키가 모두 있습니다. |
| `yamnet.hits` | 임계값 이상인 YAMNet 클래스, 점수 내림차순 |
| `is_danger` | 두 모델 중 하나라도 위험이면 `true` (FD-018 위험도 종합 전 임시 규칙) |

### 위험 음향 유형 (FD-015)

| `type` | SDS 표현 | YAMNet 클래스 |
|---|---|---|
| `scream` | 비명 | Screaming |
| `shout` | 고함 | Shout, Yell, Bellow |
| `glass_break` | 유리 파손 | Glass, Shatter, Breaking |
| `impact` | 충격음 | Smash, crash / Thump, thud / Gunshot, gunfire / Explosion / Boom |

점수 계산: YAMNet은 0.48초 간격 프레임마다 점수를 냅니다. 클래스마다 상위 2개 프레임을 평균해 클립 점수로 쓰고(`AI_YAMNET_TOP_K`), 유형 점수는 소속 클래스 중 최댓값입니다. 전체 프레임 평균은 5~10초 청크에서 짧은 사건을 희석시켜 쓰지 않습니다.

## POST /voice-profile (FD-016 음성 등록)

백엔드 `POST /api/voice-profiles`(registerVoiceProfile)가 사용자 등록 녹음을 이 엔드포인트로 넘기고, 응답 값을 `voice_profiles`에 저장합니다. AI 서버는 녹음을 저장하지 않습니다(원본 암호화 보관은 백엔드, SDS 3.6).

- 본문: `multipart/form-data`, 필드 이름 `file` (wav, mp3, m4a 등)
- 권장 녹음: 평소 목소리로 안내 문장 20~30초. 말소리가 8초(`AI_ENROLL_MIN_SPEECH_SEC`) 미만이거나 음량이 -40 dBFS(`AI_ENROLL_MIN_DBFS`) 미만이면 422입니다.

```json
{
  "mode": "model",
  "received_bytes": 158112,
  "model_version": "fd016-v1/ecapa-voxceleb-192",
  "embedding": [0.0123, -0.0456, "... 192개"],
  "baseline_volume": -20.5,
  "baseline_pitch": 152.8,
  "details": {"speech_sec": 22.1, "total_sec": 24.0, "volume_p10": -30.3, "volume_p90": -16.0, "pitch_p10": 76.0, "pitch_p90": 193.0}
}
```

| 응답 필드 | `voice_profiles` 컬럼 | 설명 |
|---|---|---|
| `embedding` | `speaker_embedding REAL[]` | ECAPA-TDNN 화자 임베딩 192차원, L2 정규화 (코사인 = 내적) |
| `baseline_volume` | `baseline_volume` | 말소리 구간 음량 중앙값 (dBFS) |
| `baseline_pitch` | `baseline_pitch` | 말소리 구간 피치(F0) 중앙값 (Hz) |
| `model_version` | `model_version VARCHAR(50)` | 바뀌면 보관한 원본(`audio_file_url`)으로 다시 추출해야 합니다 |
| `details` | (컬럼 없음) | 평소 변동 폭. '평소 대비 변화' 판단에 쓸 수 있어 저장을 제안합니다 |

재녹음 안내 (422):

```json
{"detail": {"code": "too_short", "message": "말소리가 너무 짧습니다. 안내 문장을 끝까지 읽어 주세요.", "speech_sec": 4.2, "min_sec": 8.0}}
```

| `code` | 의미 |
|---|---|
| `no_speech` | 말소리가 없음 |
| `too_short` | 말소리가 너무 짧음 |
| `too_quiet` | 목소리가 너무 작음 |
| `no_pitch` | 피치 측정 불가 (속삭임 등) |

더미 모드는 `model_version: "dummy"`와 고정값(0 벡터 192개)을 돌려줍니다. 백엔드 저장 흐름 확인용이며 실제 비교에 쓰면 안 됩니다.

## 11주차에 정할 것

- 위험도(0~1)·위험 유형 필드와 백엔드 `risk_events.event_type` 값 매핑
- 오디오·센서 업로드 경로(앱 → 백엔드 → AI 서버)와 청크 길이: 워치 현재 1.5초, SDS는 5~10초·1초 겹침
- 세션 식별자·타임스탬프를 요청에 포함할지 여부
