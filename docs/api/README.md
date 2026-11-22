# AI 서버 API

팀 API 명세 "AI 위험 분석 (석현)" 섹션의 상세 문서입니다. 폰 앱이 워치에서 모은 음성·움직임·위치를 AI 서버로 직접 보내고, AI 서버는 위험도를 돌려주며, 위험하면 백엔드에 위험 이벤트와 근거 데이터를 넘깁니다.

| API 이름 | API | Method | Endpoint | 제공 | 호출 |
|---|---|---|---|---|---|
| 실시간 위험도 분석 | analyzeRisk | POST | `/api/ai/analyze` | AI 서버 | 폰 |
| 위험 이벤트 생성 | createEvent | POST | `/api/risk-events` | 백엔드 (기존) | AI 서버 |
| 위험 데이터 전송 | uploadEventMedia | POST | `/api/risk-events/{eventId}/medias` | 백엔드 (기존) | AI 서버 |
| 음성 프로필 분석 | extractVoiceProfile | POST | `/api/ai/voice-profiles` | AI 서버 | 폰 |
| 음성 프로필 저장 | registerVoiceProfile (재등록: updateVoiceProfile) | POST (PUT) | `/api/voice-profiles` (`/api/voice-profiles/me`) | 백엔드 (기존) | AI 서버 |
| 화자 검증용 음성 정보 조회 | getVoiceProfileFeatures | GET | `/api/voice-profiles/me/features` | 백엔드 (신규 요청) | AI 서버 |

명세에 없는 개발용: `GET /health`(상태 확인), `POST /api/ai/analyze/pcm`(원시 PCM 입력).

## 흐름

```
워치 (16kHz 모노 16bit 1초 조각 → AAC, 가속도·자이로 20Hz)
  → 폰 (최근 3초 WAV + 같은 구간 움직임 + 위치, 약 1초마다)
  → AI POST /api/ai/analyze ──────────────→ 폰: risk_score·risk_type·event_id
        │ 위험이면                                  │ event_id가 있으면 워치 확인 질문 (FD-019)
        ├ createEvent (PENDING)                     └ 응답 → 백엔드 PATCH /api/risk-events/{eventId}/status
        └ uploadEventMedia: 3초 음성(audio) + 위치·움직임·분석 결과(sensor)

폰 (등록 녹음) → AI POST /api/ai/voice-profiles → 백엔드 registerVoiceProfile (원본 녹음 포함)
AI (귀가 세션 첫 요청 때) → 백엔드 getVoiceProfileFeatures → 화자 검증에 사용
```

- AI 서버는 아무것도 저장하지 않습니다. 위험 이벤트·근거 데이터·음성 프로필은 모두 백엔드에 저장됩니다.
- 3초 창이 1초씩 밀리며 2초가 겹치므로 같은 위험이 연속으로 판정됩니다. AI 서버는 한 세션에서 이벤트를 만든 뒤 30초(`AI_EVENT_GROUP_SEC`) 안의 위험은 **같은 이벤트로 묶어** 같은 `event_id`를 돌려줍니다(SDS FD-019 "대기 중 추가 위험 감지 → 같은 질문으로 묶음"). 폰은 `event_id`가 바뀔 때만 새 질문을 띄우면 됩니다.
- 측정(이 PC, 16코어 CPU): 말소리가 있는 3초 창 분석 평균 0.52초(95%가 0.6초 이내), 말소리 없는 창 0.01초. 녹음부터 폰이 결과를 받기까지 보통 약 3초이며 대부분은 워치 → 폰 전송(1.5~2초)입니다. 위험 이벤트 생성은 응답 전에 하고(백엔드 응답 시간만큼 늘어남), 미디어 업로드는 응답 뒤 백그라운드로 합니다.

## 공통

- 기본 주소: `http://<AI 서버 IP>:8000`. 지금은 폰과 AI 서버가 같은 네트워크에 있어야 합니다(외부 HTTPS 주소는 AWS 배포 때 결정).
- **인증**: 폰은 백엔드를 부를 때와 똑같이 `Authorization: Bearer <Firebase ID 토큰>`을 붙입니다. AI 서버는 토큰을 직접 검사하지 않고, 백엔드를 부를 때 **그 헤더를 그대로 넘깁니다**. 백엔드는 앱 요청과 똑같이 Firebase 토큰만 확인하면 됩니다. AI 서버 전용 키(`X-Internal-Key`)는 개발 중에는 쓰지 않습니다(배포 때 필요하면 `AI_BACKEND_KEY`를 설정하면 헤더가 붙습니다). 사용자는 백엔드가 토큰으로 정하므로 요청에 `userId`를 넣지 않습니다.
- 토큰이 없거나 `AI_BACKEND_URL`이 비어 있으면 분석만 하고 백엔드는 부르지 않습니다(화자 검증·이벤트 생성 생략). 개발·스크립트용입니다.
- `AI_API_KEY`가 설정된 경우 `/api/ai/**`에 `X-API-Key` 헤더가 필요합니다(개발용 고정 키). 없거나 다르면 401입니다.
- 요청 본문이 `AI_MAX_UPLOAD_MB`(기본 10MB, 파일마다)를 넘으면 413입니다. 3초 WAV는 약 96KB입니다.

## GET /health

```json
{"status": "ok", "mode": "model", "models_loaded": ["whisper", "yamnet", "speaker"]}
```

더미 모드(`AI_LOAD_MODELS=0`)에서는 `"mode": "dummy"`, `"models_loaded": []`입니다.

## POST /api/ai/analyze (analyzeRisk)

`multipart/form-data`, 헤더 `Authorization: Bearer <Firebase ID 토큰>`

| 필드 | 필수 | 형식 | 설명 |
|---|---|---|---|
| `file` | 예 | WAV 파일 | 16kHz 모노 16bit, 최근 3초. 파일 이름은 `voice_<captured_at>.wav`. 위험이면 이 파일이 그대로 증거 음성으로 저장됩니다. |
| `session_id` | 예* | 문자열 ≤64 | 귀가 세션 ID (`startSession` 응답의 UUID). 없으면 화자 검증·이벤트 생성 안 함 |
| `seq` | 예* | 정수 ≥0 | 3초 창의 마지막 조각 순번 (`WatchVoiceSender`의 `seq`) |
| `captured_at` | 예* | 정수 ≥0 | 3초 창의 마지막 조각 녹음 시각, Unix epoch 밀리초. 이벤트의 `detectedAt`이 됩니다 |
| `motion` | 아니오 | JSON 문자열 | 같은 3초 구간의 움직임. `{"rate_hz": 20, "source": "watch", "samples": [[t_ms, ax, ay, az, gx, gy, gz], ...]}` (가속도 m/s² 중력 포함, 자이로 rad/s). `source`는 `watch`(기본) 또는 `phone`(워치 미연결 시 폰 센서, FD-017). `t_ms`는 센서가 측정한 시각을 Unix epoch 밀리초로 바꾼 값이고, `captured_at`과 같은 워치 시계를 씁니다. 구간은 `captured_at - 3000 < t_ms ≤ captured_at`이고 시각 순으로 보냅니다. 샘플이 없으면 필드를 빼 주세요. |
| `lat`, `lng` | 아니오 | 실수 | 폰의 현재 위치 (위도 -90~90, 경도 -180~180) |
| `accuracy_m` | 아니오 | 실수 ≥0 | 위치 정확도(m) |
| `location_at` | 아니오 | 정수 ≥0 | 위치 측정 시각, epoch 밀리초 |

\* 형식상 선택이지만 폰은 매번 보내 주세요. HTTP 요청은 도착 순서가 보장되지 않으므로 순서는 `captured_at`으로, 빠진 조각은 `seq`로 판단합니다. 응답에 그대로 돌려주니 어느 요청의 결과인지 이 값으로 맞추면 됩니다.

움직임은 낙상·충격·격한 움직임 점수(FD-017)로 위험도에 들어갑니다. 위치는 아직 위험도에 쓰지 않고 위험 이벤트의 근거 데이터(`sensor`)로만 넘깁니다. 형식이 틀린 `motion`은 400, 샘플이 10개보다 적으면 움직임은 빼고 계산합니다.

```sh
curl -X POST "http://localhost:8000/api/ai/analyze" -H "Authorization: Bearer $ID_TOKEN" \
  -F "file=@voice_1763280000000.wav" -F "session_id=7f3c...-uuid" -F "seq=42" -F "captured_at=1763280000000" \
  -F 'motion={"rate_hz":20,"samples":[[1763279997050,0.1,9.8,0.2,0.0,0.1,0.0]]}' \
  -F "lat=37.5665" -F "lng=126.9780" -F "accuracy_m=12"
```

Android(OkHttp) 예:

```kotlin
val body = MultipartBody.Builder().setType(MultipartBody.FORM)
    .addFormDataPart("file", "voice_$capturedAt.wav", wav.toRequestBody("audio/wav".toMediaType()))
    .addFormDataPart("session_id", sessionId)
    .addFormDataPart("seq", seq.toString())
    .addFormDataPart("captured_at", capturedAt.toString())
    .addFormDataPart("motion", motionJson)            // 선택
    .addFormDataPart("lat", lat.toString())            // 선택
    .addFormDataPart("lng", lng.toString())
    .build()
val request = Request.Builder().url("$aiBaseUrl/api/ai/analyze")
    .header("Authorization", "Bearer $idToken")       // 백엔드 호출과 같은 Firebase ID 토큰
    .post(body).build()
```

### 응답

```json
{
  "session_id": "7f3c...-uuid",
  "seq": 42,
  "captured_at": 1763280000000,
  "mode": "model",
  "received_bytes": 96044,
  "duration_sec": 3.0,
  "processing_ms": 540,
  "event_id": 101,
  "event_status": "created",
  "risk_score": 1.0,
  "risk_type": "danger_phrase",
  "is_danger": true,
  "signals": [{"type": "danger_phrase", "score": 1.0}, {"type": "fall", "score": 0.62}],
  "whisper": {
    "status": "ok", "text": "살려주세요", "score": 1.0, "is_danger": true,
    "matches": [{"phrase": "살려주세요", "heard": "살려주세요", "exact": true, "score": 1.0}]
  },
  "yamnet": {
    "score": 0.0123, "type": null, "is_danger": false,
    "categories": {"scream": 0.0123, "shout": 0.004, "glass_break": 0.0, "impact": 0.001},
    "hits": []
  },
  "loudness": {"status": "ok", "score": 0.04, "is_danger": false, "peak_dbfs": -9.7, "background_dbfs": -46.0,
               "jump_db": 36.3, "speech_dbfs": -22.0, "over_speech_db": 12.3, "peak_at_sec": 1.2},
  "motion": {"status": "ok", "score": 0.62, "type": "fall", "is_danger": true, "source": "watch", "samples": 60,
             "scores": {"fall": 0.62, "motion_impact": 0.25, "struggle": 0.0}, "peak_g": 2.3, "free_fall_sec": 0.25},
  "speaker": {"status": "no_profile", "similarity": null, "is_user": null, "pitch_ratio": null,
              "volume_delta_db": null, "agitation": null, "boost": 0.0, "applied": 0.0}
}
```

폰이 쓰는 값은 `risk_score`, `risk_type`, `is_danger`, `event_id`입니다. `signals`·`whisper`·`yamnet`·`loudness`·`motion`·`speaker`는 신호별 근거(로그·디버깅용)입니다.

| 필드 | 설명 |
|---|---|
| `session_id`, `seq`, `captured_at` | 요청 값을 그대로 돌려줍니다(안 보냈으면 `null`). |
| `mode` | `dummy`면 모델 없이 고정값(위험 없음)을 돌려준 것. 형식은 모델 모드와 같습니다. |
| `received_bytes` / `duration_sec` | 받은 바이트 수 / 오디오 길이(초). 더미 모드는 디코딩하지 않아 `null`입니다. |
| `processing_ms` | AI 서버가 요청을 받아 응답을 만들기까지 걸린 시간 |
| **`risk_score`** | 종합 위험도 0~1 (FD-018). 0.5 이상이면 위험 |
| **`risk_type`** | 위험 유형 (아래 표). 위험이 아니면 `null` |
| **`is_danger`** | `risk_score`가 `AI_RISK_THRESHOLD`(SDS TBD-5, 기본 0.5) 이상이면 `true` |
| `signals` | 종합에 쓴 신호와 점수(큰 순서). 여러 신호가 함께 나왔는지 볼 때 씁니다 |
| **`event_id`** | 백엔드 위험 이벤트 id. 이 창에서 만들었거나(`created`) 확인 대기 중인 이벤트에 묶였으면(`grouped`) 값이 있고, 아니면 `null`. 폰은 이 id로 워치 확인 질문을 띄우고 응답을 `PATCH /api/risk-events/{eventId}/status`로 보냅니다. |
| `event_status` | `created` 새로 만듦 / `grouped` 대기 중 이벤트에 묶음 / `none` 위험 아님 / `failed` 백엔드 생성 실패 / `off` 토큰·세션·백엔드 설정 없음 |
| `whisper.status` | `ok` 인식함 / `silence` 무음 / `no_speech` 말소리 없음 / `error` 인식 실패 / `skipped` 더미 모드. `ok`가 아니면 점수 0 (FD-014) |
| `whisper.text` / `score` / `matches` | 인식 텍스트, 위험 문구 점수 0~1, 찾은 위험 문구 |
| `yamnet.categories` | 소리 종류별 YAMNet 점수. 지금은 판정이 아니라 큰 소리의 이름(`scream` 등)을 붙이는 데만 씁니다 |
| `loudness` | 갑자기 큰 소리 점수와 근거: 가장 큰 순간 음량, 배경 음량, 배경 대비 상승(`jump_db`), 평소 말소리 대비(`over_speech_db`) |
| `motion` | 움직임 점수와 유형(`fall`·`motion_impact`·`struggle`), 최대 가속도(g), 자유낙하 길이. 움직임이 없으면 `status: no_data` |
| `speaker` | 화자 검증 결과 ([아래](#응답의-speaker)). 음성 프로필이 있을 때만 `status: ok` |

`event_status: failed`(백엔드 꺼짐 등)여도 `is_danger`는 그대로 옵니다. 이때 폰은 위험 표시를 하고, 다음 창에서 다시 위험이면 AI 서버가 이벤트 생성을 다시 시도합니다.

### 위험 유형 (`risk_type`)

백엔드 `risk_events.event_type`(VARCHAR(20))에 그대로 들어가는 값입니다.

| 값 | 의미 | 근거 |
|---|---|---|
| `danger_phrase` | 위험 문구 ("살려주세요", "도와주세요", "놔줘" 등) | Whisper (FD-014) |
| `scream` | 갑자기 큰 소리 + 비명으로 들림 | 큰 소리 + YAMNet 이름 (FD-015) |
| `shout` | 갑자기 큰 소리 + 고함 | 〃 |
| `glass_break` | 갑자기 큰 소리 + 유리 파손 | 〃 |
| `impact` | 갑자기 큰 소리 + 충격음 (부딪힘·쿵·총성·폭발) | 〃 |
| `loud_sound` | 갑자기 큰 소리 (종류 모름) | 큰 소리 |
| `fall` | 낙상 (자유낙하 → 충격 → 정지) | 움직임 (FD-017) |
| `motion_impact` | 강한 충격 (넘어짐 없이 부딪힘·맞음) | 움직임 |
| `struggle` | 격한 움직임 (흔들림과 회전이 계속 큼: 몸싸움·뿌리치기) | 움직임 |

### 위험도 계산 (FD-018)

1. **눈금 맞추기**: 신호마다 위험 기준이 달라서(위험 문구 0.5 등) 각 점수를 **자기 기준이 0.5가 되도록** 0~1로 맞춥니다.
2. **여러 신호 종합 (noisy-OR)**: `risk_score = 1 − (1 − 음성)(1 − 소리)(1 − 움직임)`.
   - 신호 하나면 그 점수 그대로라 단독 판정은 전과 같습니다.
   - 따로는 애매한 신호가 함께 나오면 올라갑니다(FR-AI-05). 예: 큰 소리 0.35 + 낙상 0.35 → 0.58 위험.
   - 0.25(`AI_COMBINE_MIN`) 미만의 작은 신호는 가장 큰 신호가 아니면 더하지 않아 잡음이 쌓이지 않습니다.
   - 결과가 없는 신호(인식 실패, 움직임 없음)는 빼고 계산합니다(SDS: 일부 결과 누락 → 남은 결과로 계산).
3. **화자 검증**: 본인 목소리가 평소보다 높고 클 때 위험도를 최대 0.15 올립니다([화자 검증](#화자-검증-getvoiceprofilefeatures)).

`risk_type`은 가장 큰 신호의 유형입니다. 위치는 판단 근거(위험 지역 등)가 아직 없어 넣지 않습니다.

### 갑자기 큰 소리 (FD-015 변경, TBD-5)

소리 종류를 맞히는 대신 **갑자기, 평소 말소리보다 훨씬 크게** 났는지로 판정합니다(`AI_SOUND_MODE=loudness`, 기본). 모델 없이 파형에서 0.1초마다 음량(dBFS)을 잽니다. 두 조건을 모두 넘어야 위험(0.5점)입니다.

| 조건 | 기준 (초안) | 근거 |
|---|---|---|
| 갑자기: 가장 큰 순간 − 배경 음량 | +15dB (`AI_LOUD_JUMP_DB`), +25dB면 1점 | 충격성 소음은 배경보다 10~20dB 이상 순간 상승 (ISO 1996-1, 음향 이벤트 검출) |
| 크게: 가장 큰 순간 − 평소 말소리 음량 | +16dB (`AI_LOUD_OVER_SPEECH_DB`) | 평소 대화 약 60dB SPL, 고함 80~90dB SPL → +10~20dB |

- 배경 음량은 그 세션 최근 30초의 조용한 쪽 음량이라 시끄러운 길에서는 기준도 올라갑니다. 완전 무음(워치 녹음 시작 등)은 뺍니다.
- 평소 말소리 음량은 음성 프로필의 `baseline_volume`이고, 프로필이 없으면 −22dBFS(`AI_LOUD_SPEECH_REF_DBFS`, 워치 실측 말소리 중앙값 근처)입니다.
- 이 워치는 평소 말소리의 큰 순간이 이미 −9dBFS라서 절대 음량 기준은 쓰지 않습니다. 실제 워치 평소 말투 녹음 11개 창은 모두 0.14점 이하였습니다.
- YAMNet은 큰 소리가 났을 때 이름(아래 표, 점수 0.15 이상)을 붙이는 데만 씁니다. `AI_SOUND_MODE=yamnet`이면 예전처럼 YAMNet 점수로 판정합니다.
- 기준값은 문헌 초안이라 워치로 고함·박수·문 쾅·길거리 소리를 녹음하면 다시 맞춥니다.

### 격한 움직임 (FD-017)

학습 데이터가 없어 손목 착용 낙상 검출 문헌의 규칙 방식으로 시작합니다. 가속도 크기는 g(중력 = 1g) 기준입니다.

| 유형 | 판정 (초안) |
|---|---|
| `fall` | 자유낙하(0.6g 미만, 0.15초 이상) → 1초 안에 충격(2g 이상) → 1초 정지(흔들림 0.2g 미만). 자유낙하 + 기준 충격이면 약 0.6, 정지까지 확인되면 0.8 이상 |
| `motion_impact` | 자유낙하 없이 큰 충격 3g에서 0.5, 6g에서 1 |
| `struggle` | 1.5초 동안 가속도 흔들림(표준편차) 1.2g 이상 **그리고** 평균 회전 속도 4rad/s 이상 (뛰기는 흔들림만, 손목 돌리기는 회전만 커서 제외) |

- 20Hz는 50ms 간격이라 짧은 충격의 최댓값을 실제보다 작게 잽니다. 그래서 문헌 값의 아래쪽을 씁니다.
- 걷기·뛰기 합성 신호는 0.3점 이하였습니다. 실제 넘어지기·휘두르기·걷기·뛰기 녹음으로 다시 맞춥니다.

### 소리 이름 (YAMNet)

| `type` | SDS 표현 | YAMNet 클래스 |
|---|---|---|
| `scream` | 비명 | Screaming |
| `shout` | 고함 | Shout, Yell, Bellow |
| `glass_break` | 유리 파손 | Glass, Shatter, Breaking |
| `impact` | 충격음 | Smash, crash / Thump, thud / Gunshot, gunfire / Explosion / Boom |

점수 계산: YAMNet은 0.48초 간격 프레임마다 점수를 냅니다. 클래스마다 상위 2개 프레임을 평균해 클립 점수로 쓰고(`AI_YAMNET_TOP_K`), 유형 점수는 소속 클래스 중 최댓값입니다. 전체 프레임 평균은 긴 청크에서 짧은 사건을 희석시켜 쓰지 않습니다.

## POST /api/ai/analyze/pcm (개발용)

WAV 머리말 없이 PCM 바이트만 보낼 때 씁니다. 응답은 `/api/ai/analyze`와 같고, 위험이면 PCM을 WAV로 감싸 증거 음성으로 올립니다.

- 본문: `application/octet-stream`, 16bit little-endian 모노 PCM
- 쿼리: `sample_rate`(선택, 기본 16000), `session_id`, `seq`, `captured_at`. 움직임·위치는 받지 않습니다.
- 바이트 길이가 홀수이거나 본문이 비면 400입니다.

## 위험 이벤트 보고 (AI 서버 → 백엔드)

`/api/ai/analyze` 결과가 위험이면 AI 서버가 폰의 토큰을 붙여 백엔드 기존 API를 부릅니다.

**1. createEvent `POST /api/risk-events`** (응답 전에, 제한 시간 3초)

```json
{"sessionId": "7f3c...-uuid", "eventType": "danger_phrase",
 "description": "위험 문구 감지: '살려주세요' (위험도 0.82)", "detectedAt": "2026-11-21T09:00:03.000Z"}
```

- `detectedAt` = 요청의 `captured_at`(없으면 현재 시각), UTC ISO-8601.
- 응답 본문의 `id`를 `event_id`로 폰에 돌려줍니다.

**2. uploadEventMedia `POST /api/risk-events/{eventId}/medias`** (응답 뒤 백그라운드)

| `mediaType` | `file` |
|---|---|
| `audio` | 이어 붙인 원본 녹음 조각 `event_<eventId>_audio_<NN>.wav`, 끝나면 전체 `event_<eventId>_full.wav` (16kHz mono 16bit WAV) |
| `sensor` | 조각마다 분석 결과 `event_<eventId>_<captured_at>.json` (아래), 끝나면 녹음 정보 `event_<eventId>_audio.json` |

```json
{
  "sessionId": "7f3c...-uuid", "seq": 42, "capturedAt": 1763280000000,
  "riskScore": 0.82, "riskType": "danger_phrase",
  "voice": {"text": "살려주세요", "score": 0.9, "matches": [...]},
  "sound": {"score": 0.41, "categories": {"scream": 0.41, "shout": 0.1, "glass_break": 0.0, "impact": 0.0}},
  "loudness": {"score": 0.71, "peak_dbfs": -3.5, "background_dbfs": -44.0, "jump_db": 40.5, "over_speech_db": 18.5, ...},
  "motionScore": {"status": "ok", "score": 0.62, "type": "fall", "scores": {...}, "peak_g": 2.3, "free_fall_sec": 0.25, "source": "watch"},
  "signals": [{"type": "scream", "score": 0.71}, {"type": "fall", "score": 0.62}],
  "speaker": {"status": "ok", "is_user": true, "similarity": 0.62, "agitation": 0.7, "applied": 0.1},
  "location": {"lat": 37.5665, "lng": 126.978, "accuracy_m": 12.0, "location_at": 1763279999500},
  "motion": {"rate_hz": 20, "samples": [[...]]}
}
```

업로드가 실패해도 이벤트는 이미 만들어졌으므로 기록만 하고 넘어갑니다.

**3. 원본 녹음과 의심 단계 수집**

폰은 1초마다 "최근 3초" 창을 보내서 창끼리 2초 정도 겹칩니다. 폰은 이어 붙인 소리의 최근 3초를 그대로 잘라 보내므로 겹치는 부분은 바이트까지 같습니다. AI 서버는 겹침을 찾아 새로 붙은 소리만 이어서 **워치가 녹음한 원본과 같은 연속 녹음**을 만듭니다. 조각 길이가 들쭉날쭉하거나, 요청 순서가 바뀌거나, 요청이 한두 개 빠져도 원본과 같습니다(테스트로 확인).

| 단계 | AI 서버가 하는 일 |
|---|---|
| 평소 | 세션마다 직전 `AI_EVENT_PREROLL_SEC`(기본 10초)만 **메모리에** 들고 있음. 백엔드에 보내지 않음 |
| 의심 시작(위험 감지) | createEvent → 직전 10초 + 지금 창을 첫 조각(`audio_01`)으로, 분석 결과를 `sensor`로 올림 |
| 의심 단계(`AI_EVENT_GROUP_SEC`, 기본 30초) | 위험 여부와 상관없이 새 소리가 약 3초 쌓일 때마다 다음 조각 + 그때의 `sensor`를 올림 |
| 의심 단계 끝 | 남은 조각, 직전 10초부터 끝까지 한 파일(`full.wav`), 녹음 정보(`audio.json`)를 올림 |

- 조각(`audio_01`, `audio_02`, ...)은 빈틈이나 겹침 없이 이어지므로 번호 순서대로 붙이면 `full.wav`와 같습니다. 조각은 확인 전에도 바로 볼 수 있게, 전체 파일은 증거로 쓰기 쉽게 올립니다.
- 한 이벤트의 음성은 약 40초, 1.3MB 정도입니다(조각과 전체 파일을 합하면 그 두 배).
- 녹음 정보 `audio.json`: `{"eventId", "sampleRate", "startedAt"(녹음 시작 epoch ms, 추정), "durationSec", "prerollSec", "parts", "full", "gapsAtSec"}`. 요청이 3초 넘게 끊기면 그 자리를 `gapsAtSec`에 초 단위로 적습니다(그 사이 소리는 없음).
- 응답은 바뀌지 않습니다. 위험한 창만 `event_id`(`grouped`)가 오고, 안전한 창은 `none`이라 폰이 확인 질문을 다시 띄우지 않습니다.
- WAV가 아닌 파일 등 이어 붙일 수 없는 입력이면 받은 창을 3초마다 그대로(`voice_<captured_at>.wav`) 올립니다.
- 보호자 알림은 AI 서버가 보내지 않습니다. 백엔드가 확인 결과(`CONFIRMED`)에 따라 보냅니다.

**백엔드에 요청한 변경**

- 사용자 토큰으로 `createEvent`, `uploadEventMedia`를 부를 수 있게 허용 (`uploadEventMedia`는 지금 관리자만 가능)
- AI가 만든 PENDING 이벤트는 **보호자 알림을 보내지 않고**, 폰이 확인 질문 응답으로 상태를 확정할 때 알림 (지금은 생성 즉시 알림이라 "괜찮아요" 응답과 맞지 않음)

## POST /api/ai/voice-profiles (extractVoiceProfile, FD-016 음성 등록)

폰이 등록 녹음을 AI 서버로 보내면, AI 서버가 프로필을 뽑아 **백엔드 registerVoiceProfile로 결과와 원본 녹음을 넘기고**, 폰에는 결과를 돌려줍니다. AI 서버는 녹음을 보관하지 않습니다(원본 암호화 보관은 백엔드, SDS 3.6).

```
폰 → AI POST /api/ai/voice-profiles (file, urgent_file, Authorization)
      ├ 재녹음 필요 → 422를 폰에 바로 (저장 안 함)
      └ 성공 → 백엔드 POST /api/voice-profiles (이미 있어 409면 PUT /api/voice-profiles/me) → 폰에 200
```

- 본문: `multipart/form-data`, 헤더 `Authorization: Bearer <Firebase ID 토큰>` (백엔드가 설정돼 있으면 필수, 없으면 401 `no_token`)
  - `file` (필수): **1단계 평소 목소리** 20~30초 (wav, mp3, m4a 등)
  - `urgent_file` (선택): **2단계 위급할 때 목소리**. "살려주세요"·"도와주세요"·"놔주세요"를 크고 다급하게 3~5번
- 처리는 동기입니다. CPU 기준 25초 녹음 약 2.7초(2단계 포함 약 3초) + 백엔드 저장 시간.
- 1단계는 말소리가 8초(`AI_ENROLL_MIN_SPEECH_SEC`) 미만이거나 음량이 -40 dBFS(`AI_ENROLL_MIN_DBFS`) 미만이면 422, 2단계는 말소리가 1.5초 미만이면 422입니다.
- 가능하면 감지 때와 같은 **워치 마이크**로 녹음합니다. 평소 음량 기준값은 마이크마다 다르게 나옵니다.

### 등록 화면 안내 문구 (앱용)

> **1단계: 평소 목소리 (20~30초)**
> 조용한 곳에서 평소 말하듯 편하게 아래 문장을 읽어 주세요. 일부러 크게 하거나 꾸미지 않아도 됩니다.
> 이 녹음은 평소 목소리 크기와 높이를 알아 두는 데 쓰입니다.
>
> **2단계: 위급할 때 목소리 (3~5번)**
> 실제로 위험한 상황이라고 생각하고 "살려주세요", "도와주세요", "놔주세요"를 **크고 다급하게** 외쳐 주세요. 실제 상황처럼 할수록 위급할 때 본인 목소리를 더 잘 알아봅니다.
> 주변 사람이 놀랄 수 있으니 혼자 있는 곳에서 녹음해 주세요.

두 단계로 나누는 이유: 평소 음량·피치 기준값은 **평소 목소리로만** 재야 "평소보다 크고 높다"를 알 수 있습니다. 반대로 목소리 특징(임베딩)은 다급한 목소리가 들어 있어야 위급할 때도 본인을 알아봅니다(평가: 다급한 목소리는 본인이어도 유사도가 0.77에서 0.53으로 떨어짐). 그래서 2단계 녹음은 기준값에는 쓰지 않고 목소리 특징에만 더합니다. 이 합치기는 실제 사람 녹음으로 효과를 확인하기 전까지 꺼 두며(`AI_ENROLL_MIX_URGENT=0`), 꺼져 있어도 2단계 녹음의 평소 대비 변화는 `details.urgent`에 기록합니다.

### 응답

```json
{
  "mode": "model",
  "received_bytes": 158112,
  "saved_to": "backend:created",
  "model_version": "fd016-v1/ecapa-voxceleb-192",
  "embedding": [0.0123, -0.0456, "... 192개"],
  "baseline_volume": -20.5,
  "baseline_pitch": 152.8,
  "details": {
    "speech_sec": 22.1, "total_sec": 24.0,
    "volume_p10": -30.3, "volume_p90": -16.0, "pitch_p10": 76.0, "pitch_p90": 193.0,
    "embedding_source": "calm",
    "urgent": {"speech_sec": 3.4, "similarity_to_calm": 0.55, "pitch_ratio": 1.32, "volume_delta_db": 9.8}
  }
}
```

`saved_to`: `backend:created`(처음 등록) / `backend:updated`(재등록) / `demo`(백엔드 미설정, 시연용 `AI_DEMO_PROFILE` 파일) / `null`(저장할 곳 미설정).

### 백엔드로 보내는 값 (registerVoiceProfile / updateVoiceProfile)

`multipart/form-data`, 헤더: 폰의 `Authorization` 그대로

| 파트 | 내용 |
|---|---|
| `profile` | JSON `{"speakerEmbedding": [...192], "baselineVolume": -20.5, "baselinePitch": 152.8, "modelVersion": "...", "baselineDetails": {...}}` |
| `file` | 1단계 원본 녹음 (폰이 보낸 파일 그대로, 파일명·Content-Type 유지) |
| `urgentFile` | 2단계 원본 녹음 (있을 때만) |

| `profile` 필드 | `voice_profiles` 컬럼 | 설명 |
|---|---|---|
| `speakerEmbedding` | `speaker_embedding REAL[]` | ECAPA-TDNN 화자 임베딩 192차원, L2 정규화 (코사인 = 내적) |
| `baselineVolume` | `baseline_volume` | 1단계 말소리 구간 음량 중앙값 (dBFS, 0이 최대) |
| `baselinePitch` | `baseline_pitch` | 1단계 말소리 구간 피치(F0) 중앙값 (Hz) |
| `modelVersion` | `model_version VARCHAR(50)` | 임베딩을 뽑은 모델·계산 방식. 아래 참고 |
| `baselineDetails` | **`baseline_details JSONB` (추가 요청)** | 평소 변동 폭(`*_p10`, `*_p90`), 임베딩을 만든 녹음(`embedding_source`), 2단계 결과(`urgent`, 없으면 키 없음). 1KB 미만, nullable. 화자 검증이 '평소보다 크고 높다'를 사람마다 다르게 판단하는 데 씁니다. 그대로 저장했다가 getVoiceProfileFeatures로 돌려주세요. |

- 원본은 암호화 보관하고 위치를 `audio_file_url`에, 저장 시각을 `consented_at`에 넣습니다(앱에서 동의 후에만 녹음).
- 재등록(PUT)은 덮어쓰고 이전 원본(1·2단계 모두)은 삭제합니다. 등록 한 번이 한 세트입니다.
- 백엔드 401·403·404(토큰 오류·미가입)는 그대로 폰에 전달하고(`code: backend_rejected`), 그 밖의 오류나 10초 시간 초과는 **502** `{"detail": {"code": "save_failed", ...}}`입니다. AI 서버는 재시도하지 않습니다. 앱이 다시 보내면 처음부터 다시 분석·저장합니다(덮어쓰기라 중복 문제 없음).
- 저장에 성공하면 AI 서버는 그 사용자의 받아 둔 프로필을 지워, 진행 중인 귀가 세션에도 새 값이 바로 쓰이게 합니다.

`model_version`: 임베딩 숫자의 의미는 뽑은 모델에 따라 완전히 달라집니다. 모델을 바꾸면 예전 값과 비교할 수 없으므로 버전을 올립니다. AI 서버는 저장된 버전이 지금 버전과 다르면 화자 검증을 건너뜁니다(위험 감지는 그대로 동작). 필요해지면 백엔드가 보관한 원본을 다시 보내 새 값으로 덮어씁니다. 2단계 녹음을 합쳐도 같은 모델의 벡터라 버전은 바뀌지 않습니다.

### 재녹음 안내 (422)

```json
{"detail": {"code": "too_short", "message": "말소리가 너무 짧습니다. 안내 문장을 끝까지 읽어 주세요.", "speech_sec": 4.2, "min_sec": 8.0}}
```

| `code` | 의미 |
|---|---|
| `no_speech` | 1단계에 말소리가 없음 |
| `too_short` | 1단계 말소리가 너무 짧음 |
| `too_quiet` | 1단계 목소리가 너무 작음 |
| `no_pitch` | 1단계 피치 측정 불가 (속삭임 등) |
| `urgent_too_short` | 2단계 말소리가 너무 짧음 |

앱은 `message`를 그대로 보여 주고 해당 단계를 다시 녹음하게 하면 됩니다. 그 밖의 오류: 400 빈 파일·읽을 수 없는 오디오, 401 토큰 없음·거절, 413 10MB 초과, 502 백엔드 저장 실패.

더미 모드는 `model_version: "dummy"`와 고정값(0 벡터 192개)을 돌려주고 저장도 똑같이 합니다. 백엔드 저장 흐름 확인용이며 실제 비교에 쓰면 안 됩니다(버전이 달라 화자 검증에서 자동으로 건너뜀).

## 화자 검증 (getVoiceProfileFeatures)

`/api/ai/analyze` 요청에 토큰과 `session_id`가 있으면 AI 서버가 그 사용자의 음성 프로필로 **본인 목소리인지, 평소보다 높고 큰지**를 봅니다. 결과는 위험도를 **올리는 데만** 씁니다.

- 프로필은 **귀가 세션(`session_id`)마다 한 번** 백엔드에서 받아 세션 동안 메모리에 둡니다. 그 세션 요청이 10분(`AI_PROFILE_IDLE_SEC`) 동안 없으면 세션이 끝난 것으로 보고 지웁니다.
- 받아 오는 동안 들어온 요청은 기다리지 않고 화자 검증 없이 분석합니다. 백엔드가 응답하지 않아도 위험 감지는 그대로 동작합니다.

```
GET /api/voice-profiles/me/features
Authorization: <폰이 보낸 Firebase ID 토큰 그대로>
```

| 응답 | 의미 |
|---|---|
| 200 | 프로필 (아래) |
| 404 | 음성 등록을 안 한 사용자 → 화자 검증 없이 분석 |
| 그 밖 | 30초 뒤 다시 시도, 그동안 화자 검증 없이 분석 |

```json
{
  "speakerEmbedding": [0.0123, -0.0456, "... 192개"],
  "baselineVolume": -20.5,
  "baselinePitch": 152.8,
  "baselineDetails": {"volume_p90": -16.0, "pitch_p90": 193.0, "...": "등록 때 보낸 baselineDetails 그대로"},
  "modelVersion": "fd016-v1/ecapa-voxceleb-192"
}
```

`getVoiceProfile`(`/api/voice-profiles/me`)과 달리 임베딩까지 돌려주는 AI 서버용 API입니다(배포 때는 AI 서버 전용 키로 막을 예정). camelCase와 snake_case 모두 받습니다. AI 서버 설정: `AI_BACKEND_URL`(예: `http://<백엔드>:8080`).

백엔드 없이 시연할 때는 `AI_DEMO_PROFILE`에 프로필 JSON 파일(`scripts/enroll_profile.py` 결과나 `/api/ai/voice-profiles` 응답) 경로를 넣으면 모든 세션에 그 프로필을 씁니다.

### 응답의 `speaker`

```json
"speaker": {"status": "ok", "similarity": 0.62, "is_user": true, "pitch_ratio": 1.35,
            "volume_delta_db": 8.1, "agitation": 0.71, "boost": 0.1065, "applied": 0.1065}
```

| 필드 | 설명 |
|---|---|
| `status` | `ok` 검증함 / `off` 토큰·세션·조회 설정 없음 / `loading` 프로필 받는 중 / `no_profile` 등록 안 함 / `version_mismatch` 모델 버전 다름 / `no_speech` 말소리 1초 미만 / `error` 검증 실패. `ok`가 아니면 점수에 반영하지 않습니다. |
| `similarity` / `is_user` | 등록 목소리와의 코사인 유사도, `AI_VERIFY_SIM_THRESHOLD`(기본 0.4) 이상이면 본인 |
| `pitch_ratio` / `volume_delta_db` | 평소 피치 대비 비율, 평소 음량 대비 차이(dB) |
| `agitation` | 평소보다 높고 큰 정도 0~1. 평소 변동 상단(`pitch_p90`, `volume_p90`)을 넘은 만큼 오릅니다. |
| `boost` | 본인이면 `0.15 × agitation`, 아니면 0 |
| `applied` | 실제로 위험도에 더한 값. 위험 신호가 이미 0.3(`AI_VERIFY_BOOST_MIN_BASE`) 이상일 때만 더하므로, 크게 말하기만 해서는 위험이 되지 않습니다. |

기준값(유사도 0.4, 흥분도 범위)은 TTS로 정한 임시값입니다. 실제 사람 녹음으로 다시 조정합니다.

## 다른 팀에 요청한 것

**백엔드**
- `getVoiceProfileFeatures` `GET /api/voice-profiles/me/features` 신규 (사용자 토큰)
- 사용자 토큰으로 AI 서버가 `registerVoiceProfile`·`updateVoiceProfile`·`createEvent`·`uploadEventMedia`를 부를 수 있게 허용
- `voice_profiles.baseline_details JSONB` 컬럼
- AI가 만든 PENDING 이벤트는 확정 때 보호자 알림, 확인 질문 응답용 상태 값(예: `CONFIRMED`·`DISMISSED`)
- 파일 크기: 등록 요청은 파일마다 10MB까지 올 수 있음(`spring.servlet.multipart.max-file-size=10MB`, `max-request-size=25MB`)

**프론트**
- 폰에서 `/api/ai/analyze` 호출 (Firebase ID 토큰, 3초 WAV, 같은 구간 움직임, 위치)
- 워치 움직임 샘플을 묶어 보내기: 지금은 20Hz 샘플을 DataItem 하나로 덮어써 중간 샘플이 빠질 수 있음
- `event_id`가 새로 오면 워치 확인 질문, 응답을 `PATCH /api/risk-events/{eventId}/status`로 전송

## 아직 정할 것

- 폰이 밖(LTE)에서 AI 서버에 접속할 주소(HTTPS). AWS 배포 때 결정
- AI 서버가 토큰을 직접 검증하지 않으므로, 백엔드를 부르지 않는 분석 요청 자체는 누구나 보낼 수 있음. 배포 때 AI 서버에서도 Firebase 토큰 검증을 할지 결정
- 확인 질문 대기 시간(SDS TBD-6). 지금 이벤트 묶음 시간 30초(`AI_EVENT_GROUP_SEC`)를 같은 값으로 맞출 예정
- 움직임 점수(FD-017) 계산: 형식은 받고 있으며, 실제 워치 데이터로 기준을 정한 뒤 추가
