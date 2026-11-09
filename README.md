# SafePath AI 서버

SafePath는 안전귀가를 지원하기 위한 캡스톤 프로젝트입니다. 이 저장소는 오디오를 분석해 위험 상황을 판단하는 FastAPI 기반 AI 분석 서버(SDS M-10)를 관리합니다.

## 기술 스택

- Python 3.11, FastAPI, Uvicorn
- faster-whisper(위험 문구), TensorFlow Hub YAMNet(위험 음향), librosa

## 현재 상태

- `GET /health`, `POST /analyze`(오디오 파일), `POST /analyze/pcm`(원시 PCM) 엔드포인트가 있습니다.
- `AI_LOAD_MODELS=0`이면 모델 없이 더미 응답을 돌려줍니다. 백엔드·앱과의 요청 수신 확인에 사용합니다.
- `AI_LOAD_MODELS=1`이면 서버 시작 시 Whisper(base)와 YAMNet을 한 번 적재하고 같은 클립을 두 모델에 병렬로 넣습니다.
- 위험도(0~1)와 위험 유형을 포함한 최종 응답 형식, 움직임 감지, 화자 검증은 아직 구현되지 않았습니다. 일정은 팀 마일스톤 11~13주차를 따릅니다.

## 로컬 개발 시작

1. Python 3.11 환경을 준비합니다. Windows는 [개발 환경 문서](docs/setup/README.md)를 참고하세요.
2. pip을 최신으로 올린 뒤 더미 모드 의존성을 설치합니다. 오래된 pip은 한글 Windows에서 requirements 파일을 읽다가 실패할 수 있습니다.

   ```sh
   python -m pip install --upgrade pip
   pip install -r requirements-dev.txt
   ```

3. `.env.example`을 `.env`로 복사하고 필요하면 값을 바꿉니다. 기본값은 더미 모드입니다. 서버는 `.env`를 직접 읽지 않으므로 실행할 때 `--env-file .env`를 붙입니다.
4. 서버를 실행합니다. 기본 주소는 `http://localhost:8000`입니다(백엔드는 8080).

   ```sh
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
   ```

5. 다른 터미널에서 수신을 확인합니다.

   ```sh
   python scripts/send_test.py
   ```

모델 모드로 실행하려면 `pip install -r requirements-models.txt` 후 `AI_LOAD_MODELS=1`로 시작합니다. 첫 실행 때 Whisper와 YAMNet 가중치를 내려받습니다.

테스트는 저장소 루트에서 `pytest`로 실행합니다(`pytest.ini`가 경로를 잡아 줍니다). 테스트는 더미 모드에서 동작하므로 모델 설치가 필요하지 않습니다.

## API

| 메서드·경로 | 입력 | 설명 |
|---|---|---|
| GET `/health` | 없음 | `status`, `mode`(`dummy`/`model`), `models_loaded` |
| POST `/analyze` | `multipart/form-data`의 `file` | wav, mp3, m4a 등 오디오 파일 |
| POST `/analyze/pcm` | `application/octet-stream` 본문 | 16bit little-endian mono PCM. `?sample_rate=` 생략 시 16000 |

`AI_API_KEY`를 설정하면 `/analyze` 계열 요청에 `X-API-Key` 헤더가 필요합니다. 요청 본문이 `AI_MAX_UPLOAD_MB`(기본 10MB)를 넘으면 413입니다. 요청·응답 형식은 [API 문서](docs/api/README.md)를 참고하세요.

## 프로젝트 구조

- `app/main.py` — FastAPI 앱, 엔드포인트
- `app/config.py` — 환경변수 설정
- `app/audio.py` — 오디오 입력을 16kHz mono float32로 변환
- `app/detectors/audio_detector.py` — Whisper·YAMNet 위험 탐지
- `scripts/send_test.py` — 수신 확인 스크립트(표준 라이브러리만 사용)
- `tests/` — 더미 모드 API 테스트
- `docs/` — API, 개발 환경, 모델 문서

## GitHub 협업

- `main`은 안정적인 통합 코드를 유지하고 `feature/*` 브랜치에서 기능별로 작업합니다.
- Pull Request를 통해 코드를 검토하고 병합합니다.
- `.env`, API 키, 녹음 파일은 Git에 업로드하지 않습니다.

## 세부 문서

- [API](docs/api/README.md)
- [개발 환경(Windows)](docs/setup/README.md)
- [모델 정리(Whisper·YAMNet)](docs/models/README.md)
