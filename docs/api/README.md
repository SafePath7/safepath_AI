# AI 서버 API

11주차 합의안입니다. 폰 앱이 워치 소리를 모아 AI 서버로 직접 보내고, AI 서버는 위험도(0~1)와 위험 유형을 돌려줍니다.

## 흐름

```
워치 (16kHz 모노 16bit, 1초 조각 → AAC 압축)
  → 폰 (AAC 풀기, 최근 3초를 WAV로, 약 1초마다)
  → AI 서버 POST /analyze  → risk_score·risk_type 응답
  → 폰: 위험이면 워치에 확인 질문(FD-019), 확정되면 백엔드 POST /api/risk-events
```

- AI 서버는 판정 결과를 저장하지 않습니다. 위험 이벤트 기록·보호자 알림은 폰 → 백엔드가 맡습니다.
- 3초 창이 1초씩 밀리며 2초가 겹치므로, 같은 위험이 연속 2~3번 판정될 수 있습니다. 폰은 확인 질문을 띄운 동안 들어온 판정을 같은 질문으로 묶습니다(SDS FD-019 "대기 중 추가 위험 감지 → 같은 질문으로 묶음").
- 위치는 보내지 않습니다. 경로 이탈 판단은 폰·백엔드가 합니다.
- 측정(이 PC, 16코어 CPU): 말소리가 있는 3초 창 분석 평균 0.52초(95%가 0.6초 이내), 말소리 없는 창 0.01초. 녹음부터 폰이 결과를 받기까지 보통 약 3초이며 대부분은 워치 → 폰 전송(1.5~2초)입니다.

## 공통

- 기본 주소: `http://<AI 서버 IP>:8000`. 폰과 AI 서버가 같은 네트워크에 있어야 합니다(외부 접속 주소는 추후 결정).
- `AI_API_KEY`가 설정된 경우 `/analyze`, `/analyze/pcm`, `/voice-profile`에 `X-API-Key` 헤더가 필요합니다. 없거나 다르면 401입니다.
- 요청 본문이 `AI_MAX_UPLOAD_MB`(기본 10MB)를 넘으면 413입니다. 3초 WAV는 약 96KB입니다.

## GET /health

```json
{"status": "ok", "mode": "model", "models_loaded": ["whisper", "yamnet", "speaker"]}
```

더미 모드에서는 `"mode": "dummy"`, `"models_loaded": []`입니다.

## POST /analyze (폰 → AI, 기본 경로)

`multipart/form-data`

| 필드 | 필수 | 형식 | 설명 |
|---|---|---|---|
| `file` | 예 | WAV 파일 | 16kHz 모노 16bit, 최근 3초. 파일 이름은 `voice_<captured_at>.wav`(폰 디버그 저장 파일과 같은 이름)로 보내 주세요. 다른 샘플레이트·mp3·m4a도 받지만 변환 시간이 듭니다. |
| `session_id` | 아니오 | 문자열 ≤64 | 귀가 세션 ID (`safe_return_sessions`의 UUID) |
| `user_id` | 아니오 | 문자열 ≤64 | 사용자 ID. 화자 검증(등록 음성 비교)에 쓸 예정. 응답에는 돌려주지 않습니다. |
| `seq` | 아니오 | 정수 ≥0 | 3초 창의 마지막 조각 순번 (`WatchVoiceSender`의 `seq`). 빠진 조각 확인용 |
| `captured_at` | 아니오 | 정수 ≥0 | 3초 창의 마지막 조각 녹음 시각, Unix epoch 밀리초 (`capturedAt`). 디버그 파일 이름의 숫자와 같은 값 |

식별 필드는 모두 선택이지만 폰은 매번 보내 주세요. HTTP 요청은 도착 순서가 보장되지 않으므로 순서·시각은 파일 이름이 아니라 `captured_at`으로 판단합니다. `seq`는 조각이 빠졌는지(41 다음 43) 확인하는 데 씁니다. 응답에 `seq`·`captured_at`을 그대로 돌려주니 어느 요청의 결과인지 이 값으로 맞추면 됩니다.

```sh
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@voice_1763280000000.wav" -F "session_id=7f3c...-uuid" -F "user_id=12" \
  -F "seq=42" -F "captured_at=1763280000000"
```

Android(OkHttp) 예:

```kotlin
val body = MultipartBody.Builder().setType(MultipartBody.FORM)
    .addFormDataPart("file", "voice_$capturedAt.wav", wav.toRequestBody("audio/wav".toMediaType()))
    .addFormDataPart("session_id", sessionId)
    .addFormDataPart("user_id", userId.toString())
    .addFormDataPart("seq", seq.toString())
    .addFormDataPart("captured_at", capturedAt.toString())
    .build()
```

## POST /analyze/pcm

WAV 머리말 없이 PCM 바이트만 보낼 때 씁니다. 응답은 `/analyze`와 같습니다.

- 본문: `application/octet-stream`, 16bit little-endian 모노 PCM
- 쿼리: `sample_rate`(선택, 기본 16000), `session_id`, `user_id`, `seq`, `captured_at` (위 표와 같음)
- 바이트 길이가 홀수이거나 본문이 비면 400입니다.

```sh
curl -X POST "http://localhost:8000/analyze/pcm?session_id=abc&seq=42&captured_at=1763280000000" \
  -H "Content-Type: application/octet-stream" --data-binary @chunk.pcm
```

## 응답 (두 엔드포인트 공통)

```json
{
  "session_id": "7f3c...-uuid",
  "seq": 42,
  "captured_at": 1763280000000,
  "mode": "model",
  "received_bytes": 96044,
  "duration_sec": 3.0,
  "processing_ms": 540,
  "risk_score": 1.0,
  "risk_type": "danger_phrase",
  "is_danger": true,
  "whisper": {
    "status": "ok", "text": "살려주세요", "score": 1.0, "is_danger": true,
    "matches": [{"phrase": "살려주세요", "heard": "살려주세요", "exact": true, "score": 1.0}]
  },
  "yamnet": {
    "score": 0.0123, "type": null, "is_danger": false,
    "categories": {"scream": 0.0123, "shout": 0.004, "glass_break": 0.0, "impact": 0.001},
    "hits": []
  }
}
```

폰이 쓰는 값은 맨 위 세 개(`risk_score`, `risk_type`, `is_danger`)입니다. `whisper`·`yamnet`은 모델별 근거(로그·디버깅·보호자 화면 설명용)입니다.

| 필드 | 설명 |
|---|---|
| `session_id`, `seq`, `captured_at` | 요청 값을 그대로 돌려줍니다(안 보냈으면 `null`). |
| `mode` | `dummy`면 모델 없이 고정값(위험 없음)을 돌려준 것. 형식은 모델 모드와 같습니다. |
| `received_bytes` / `duration_sec` | 받은 바이트 수 / 오디오 길이(초). 더미 모드 `/analyze`는 디코딩하지 않아 `null`입니다. |
| `processing_ms` | AI 서버가 요청을 받아 응답을 만들기까지 걸린 시간 |
| **`risk_score`** | 종합 위험도 0~1 (FD-018). 0.5 이상이면 위험 |
| **`risk_type`** | 위험 유형 (아래 표). 위험이 아니면 `null` |
| **`is_danger`** | `risk_score`가 `AI_RISK_THRESHOLD`(SDS TBD-5, 기본 0.5) 이상이면 `true` |
| `whisper.status` | `ok` 인식함 / `silence` 무음 / `no_speech` 말소리 없음 / `error` 인식 실패 / `skipped` 더미 모드. `ok`가 아니면 점수 0 (FD-014) |
| `whisper.text` / `score` / `matches` | 인식 텍스트, 위험 문구 점수 0~1, 찾은 위험 문구 |
| `yamnet.score` / `type` / `categories` / `hits` | 위험 음향 점수 0~1과 유형, 유형별 점수, 임계값 이상인 YAMNet 클래스 (FD-015) |

### 위험 유형 (`risk_type`)

백엔드 `risk_events.event_type`(VARCHAR(20))에 그대로 넣을 수 있는 값입니다.

| 값 | 의미 | 근거 |
|---|---|---|
| `danger_phrase` | 위험 문구 ("살려주세요", "도와주세요", "놔줘" 등) | Whisper (FD-014) |
| `scream` | 비명 | YAMNet (FD-015) |
| `shout` | 고함 | YAMNet |
| `glass_break` | 유리 파손 | YAMNet |
| `impact` | 충격음 (부딪힘·쿵·총성·폭발) | YAMNet |

움직임(FD-017)이 붙으면 `fall` 등 유형이 추가됩니다.

### 위험도 계산 (FD-018 1단계)

모델마다 위험 기준이 다릅니다(위험 문구 0.5, 위험 음향 0.3). 그래서 각 점수를 **자기 기준이 0.5가 되도록** 0~1로 다시 맞춘 뒤 가장 큰 값을 `risk_score`로, 그 모델의 유형을 `risk_type`으로 씁니다. 인식 실패처럼 결과가 없는 모델은 빼고 계산합니다(SDS: 일부 결과 누락 → 남은 결과로 계산). 움직임 점수와 화자 검증(등록한 본인 목소리면 가산)은 이후 같은 방식으로 더합니다.

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

## 아직 정할 것

- 폰이 밖(LTE)에서 AI 서버에 접속할 주소와 인증 방식(지금은 같은 네트워크 + `X-API-Key`)
- 움직임 센서(FD-017) 전송 형식: 워치 가속도·자이로를 같은 요청에 넣을지, 별도 엔드포인트로 할지
- 화자 검증용 등록 음성 정보(`voice_profiles`)를 AI 서버가 받는 경로: AI 서버가 세션 시작 때 백엔드에서 한 번 받아 두는 방식을 제안
- 음성 등록 녹음도 워치 마이크·AAC 경로로 받을지(평소 음량 기준값은 마이크마다 달라 감지 때와 같은 경로가 필요)
