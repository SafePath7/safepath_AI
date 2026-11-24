# SafePath AI 서버

SafePath는 안전귀가를 지원하기 위한 캡스톤 프로젝트입니다. 이 저장소는 오디오를 분석해 위험 상황을 판단하는 FastAPI 기반 AI 분석 서버(SDS M-10)를 관리합니다.

## 기술 스택

- Python 3.11, FastAPI, Uvicorn
- faster-whisper(위험 문구), TensorFlow Hub YAMNet(위험 음향), librosa

## 현재 상태

- `GET /health`, `POST /api/ai/analyze`(오디오 파일), `POST /api/ai/analyze/pcm`(원시 PCM) 엔드포인트가 있습니다.
- `AI_LOAD_MODELS=0`이면 모델 없이 더미 응답을 돌려줍니다. 백엔드·앱과의 요청 수신 확인에 사용합니다.
- `AI_LOAD_MODELS=1`이면 서버 시작 시 Whisper(base)와 YAMNet을 한 번 적재하고 같은 클립을 두 모델에 병렬로 넣습니다.
- FD-015 위험 음향 감지(YAMNet): 비명·고함·유리 파손·충격음 유형별 점수(0~1)와 유형을 돌려줍니다. 초기 임계값 0.3.
- 폰이 3초 WAV를 약 1초마다 `POST /api/ai/analyze`로 보내면 종합 위험도 `risk_score`(0~1)와 위험 유형 `risk_type`을 돌려줍니다(FD-018 1단계: 위험 문구·위험 음향).
- 움직임 감지(FD-017)와 화자 검증 결합은 아직 구현되지 않았습니다. 일정은 팀 마일스톤 11~13주차를 따릅니다.

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

모델 모드로 실행하려면 `pip install -r requirements-models.txt` 후 `AI_LOAD_MODELS=1`로 시작합니다. 첫 실행 때 Whisper·YAMNet·화자 임베딩(ECAPA) 가중치를 내려받습니다.

> **저장소를 새로 받은 팀원(git pull)**: 모델 모드를 쓴다면 `pip install -r requirements-models.txt`를 다시 실행하세요. 11주차에 PyTorch·SpeechBrain이 추가되고 `setuptools<81`이 고정됐습니다. 자세한 내용은 [개발 환경 문서의 "저장소를 새로 받은 뒤"](docs/setup/README.md#저장소를-새로-받은-뒤git-pull-할-일)를 참고하세요. 더미 모드만 쓴다면 할 일이 없습니다.

테스트는 저장소 루트에서 `pytest`로 실행합니다(`pytest.ini`가 경로를 잡아 줍니다). 테스트는 더미 모드에서 동작하므로 모델 설치가 필요하지 않습니다.

## API

| 메서드·경로 | 입력 | 설명 |
|---|---|---|
| GET `/health` | 없음 | `status`, `mode`(`dummy`/`model`), `models_loaded` |
| POST `/api/ai/analyze` | `multipart/form-data`의 `file` + `session_id`·`seq`·`captured_at`·`motion`·위치(선택), 헤더 `Authorization` | analyzeRisk. 폰이 보내는 3초 WAV 등 → `risk_score`·`risk_type`, 위험이면 백엔드 위험 이벤트 생성 후 `event_id` |
| POST `/api/ai/analyze/pcm` | `application/octet-stream` 본문 | 개발용. 16bit little-endian mono PCM. `?sample_rate=` 생략 시 16000 |
| POST `/api/ai/voice-profiles` | `multipart/form-data`의 `file`·`urgent_file`, 헤더 `Authorization` | extractVoiceProfile. 음성 등록 → 화자 임베딩·평소 음량·피치 → 백엔드 registerVoiceProfile |

`AI_API_KEY`를 설정하면 `/api/ai/**` 요청에 `X-API-Key` 헤더가 필요합니다. 요청 본문이 `AI_MAX_UPLOAD_MB`(기본 10MB)를 넘으면 413입니다. 요청·응답 형식은 [API 문서](docs/api/README.md)를 참고하세요.

## 프로젝트 구조

- `app/main.py` — FastAPI 앱, 엔드포인트
- `app/config.py` — 환경변수 설정
- `app/audio.py` — 오디오 입력을 16kHz mono float32로 변환
- `app/detectors/audio_detector.py` — Whisper·YAMNet 위험 탐지
- `scripts/send_test.py` — 수신 확인 스크립트(표준 라이브러리만 사용)
- `scripts/simulate_watch.py` — 내 녹음을 워치 → 폰 형식(AAC 압축, 1초마다 3초 창)으로 변환하고 Whisper 결과 출력
- `scripts/watch_files_check.py` — 폰이 저장한 워치 실녹음(`voice_<ms>.wav`) 시간순 분석·연속 녹음 복원
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

## Docker로 실행

Docker Desktop 등에서 Docker 데몬을 실행한 뒤 이 저장소 루트에서 빌드합니다. 기존 Python 로컬 실행 방법은 그대로 사용할 수 있습니다.

```sh
docker build -t safepath-ai .
docker run --rm --name safepath-ai -p 8000:8000 safepath-ai
```

다른 터미널에서 `curl http://localhost:8000/health`를 실행하면 `mode`가 `dummy`, `models_loaded`가 빈 배열인 응답을 확인할 수 있습니다. 종료는 서버 터미널에서 Ctrl+C를 누릅니다. 컨테이너는 `0.0.0.0:8000`에서 수신하며 `-p`가 호스트 포트와 연결합니다.

설정은 실행 시 전달합니다. 예를 들어 `docker run --rm -p 8000:8000 -e AI_BACKEND_URL=http://host.docker.internal:8080 safepath-ai`처럼 설정할 수 있습니다(Docker Desktop 기준). 또는 `.env.example`을 복사한 로컬 `.env`를 `docker run --rm -p 8000:8000 --env-file .env safepath-ai`로 전달합니다. 컨테이너의 `localhost`는 백엔드가 아니라 AI 컨테이너 자신입니다. 비밀값은 이미지에 넣지 않고 실행 환경에서 전달하세요. 백엔드 주소를 비우면 백엔드 호출을 생략합니다.

### 실제 모델 사용

기본 이미지는 모델 의존성을 포함하지 않으므로 모델용 이미지를 별도로 빌드하고 실행할 때 모델을 켭니다.

```sh
docker build --build-arg INSTALL_MODELS=1 -t safepath-ai-model .
docker run --rm --name safepath-ai-model -p 8000:8000 \
  -e AI_LOAD_MODELS=1 -e AI_MODEL_CACHE_DIR=/cache/models \
  -e HF_HOME=/cache/huggingface -v safepath-model-cache:/cache safepath-ai-model
```

이 이미지는 CPU용 PyTorch와 `requirements-models.txt`의 의존성을 설치합니다. 빌드 크기와 시간, CPU·메모리 사용량이 더 크며 플랫폼에 맞는 패키지가 필요합니다. 최초 실행 시 인터넷으로 Whisper, YAMNet, ECAPA 가중치를 내려받아 시작이 오래 걸립니다. 볼륨은 YAMNet·ECAPA와 Hugging Face의 Whisper 캐시를 컨테이너 재생성 후에도 보존합니다. `/health`에서 `mode`가 `model`이고 `models_loaded`에 `whisper`, `yamnet`, `speaker`가 모두 있는지 확인하세요. 모델 다운로드 또는 로딩이 실패하면 서버가 정상 시작되지 않을 수 있으므로 `docker logs safepath-ai-model`을 확인하세요. 로컬 `models/`는 빌드에 포함하지 않으며, 별도 캐시를 쓰려면 컨테이너에 마운트해야 합니다.

### 나중에 백엔드 Compose와 연결

두 저장소가 같은 부모 폴더에 있고 AI 폴더가 `safepath_AI`라면 백엔드의 `compose.yaml`에서 AI 서비스의 `build.context`를 예를 들어 `../safepath_AI`로 지정합니다(경로는 Compose 파일 위치 기준). 컨테이너 포트는 `8000`이며, 호스트에서도 접근하려면 `ports: ["8000:8000"]`을 설정합니다. 같은 Compose 네트워크에서 AI가 백엔드를 호출할 때는 `AI_BACKEND_URL=http://<백엔드 서비스명>:<백엔드 컨테이너 포트>`를 사용하고, 백엔드가 AI를 호출할 때는 `http://<AI 서비스명>:8000`을 사용합니다. 기본값은 `AI_LOAD_MODELS=0`이고, 필요하면 `AI_BACKEND_KEY`, `AI_API_KEY` 등은 실행 환경이나 별도 환경변수 파일로 전달합니다. 실제 모델용 빌드는 `INSTALL_MODELS=1` 빌드 인자와 `AI_LOAD_MODELS=1` 실행 환경변수가 모두 필요합니다. 서비스명과 포트는 백엔드 프로젝트의 실제 설정에 맞춰 정해야 합니다.
