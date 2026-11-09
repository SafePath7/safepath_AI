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
  "yamnet": {"hits": [{"class": "Screaming", "score": 0.4123}], "is_danger": true},
  "is_danger": true
}
```

| 필드 | 설명 |
|---|---|
| `mode` | `dummy`면 모델 없이 고정값(위험 없음)을 돌려준 것 |
| `received_bytes` | 받은 바이트 수 |
| `duration_sec` | 16kHz 기준 오디오 길이(초) |
| `whisper.text` / `whisper.is_danger` | 인식 텍스트, 위험 문구 포함 여부 |
| `yamnet.hits` | 임계값을 넘은 위험 클래스와 점수 |
| `is_danger` | 두 모델 중 하나라도 위험이면 `true` |

## 11주차에 정할 것

- 위험도(0~1)·위험 유형 필드와 백엔드 `risk_events.event_type` 값 매핑
- 오디오·센서 업로드 경로(앱 → 백엔드 → AI 서버)와 청크 길이: 워치 현재 1.5초, SDS는 5~10초·1초 겹침
- 세션 식별자·타임스탬프를 요청에 포함할지 여부
