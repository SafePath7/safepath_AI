# 개발 환경 (Windows)

## Anaconda로 환경 만들기

```powershell
conda create -n safepath-ai python=3.11 -y
conda activate safepath-ai
cd <저장소 경로>\safepath_AI
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pytest
```

- pip을 먼저 올리는 이유: 오래된 pip(21.x)은 한글 Windows(cp949)에서 UTF-8 requirements 파일을 읽지 못하고 `UnicodeDecodeError`로 멈춥니다.
- `pytest`가 모두 통과하면 환경 준비가 끝난 것입니다.

## 더미 모드 실행

```powershell
copy .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env
```

`.env` 없이 실행하면 기본값(더미 모드)으로 시작합니다. 환경변수를 직접 지정해도 됩니다(`$env:AI_LOAD_MODELS = "0"`).

다른 PowerShell 창에서:

```powershell
conda activate safepath-ai
python scripts\send_test.py
```

`/health` 응답과 `/analyze/pcm` 응답이 출력되고, 서버 창에 `analyze source=pcm bytes=48000 duration=1.5 mode=dummy` 로그가 남으면 수신 확인이 끝난 것입니다.

`POST /analyze`(파일 업로드, 백엔드가 보낼 형식)까지 확인하려면 wav 파일을 지정합니다.

```powershell
python scripts\send_test.py --wav <wav 파일 경로>
```

서버 창에 `analyze source=file:<파일명> ... mode=dummy` 로그가 남으면 됩니다.

## 모델 모드 실행

```powershell
pip install -r requirements-models.txt
$env:AI_LOAD_MODELS = "1"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 첫 실행 때 Whisper(base, 약 145MB), YAMNet(약 14MB), 화자 임베딩 ECAPA(약 80MB) 가중치를 내려받아 몇 분 걸립니다. 이후 시작에는 약 20~25초 걸립니다.
- 시작 로그에 `모델 로드 완료`가 찍히고 `/health`의 `models_loaded`가 `["whisper", "yamnet", "speaker"]`이면 준비된 것입니다.
- YAMNet과 ECAPA는 저장소의 `models/`에 저장됩니다(Git 제외). Whisper는 Hugging Face 캐시(`%USERPROFILE%\.cache\huggingface`)에 저장됩니다.
- YAMNet은 `models/tfhub`에 둡니다. 예전처럼 Windows 임시 폴더에 두면 디스크 정리 후 빈 폴더만 남아 `saved_model.pb` 오류가 납니다. 캐시가 깨져 있으면 서버가 자동으로 지우고 다시 받습니다.
- 실제 모델 테스트: `$env:AI_RUN_MODEL_TESTS = "1"; pytest` (기본 `pytest`는 모델 테스트를 건너뜁니다)
- 위험 음향 점수 확인·임계값 튜닝: `python scripts\yamnet_check.py <파일 또는 폴더> --chunk 10`
- 설치가 끝나면 `pip freeze`로 확인한 버전을 `requirements-models.txt`에 고정합니다.
- mp3·m4a 파일 디코딩에는 ffmpeg가 필요할 수 있습니다. wav와 `/analyze/pcm`은 필요하지 않습니다.

## 저장소를 새로 받은 뒤(git pull) 할 일

| 사용 모드 | 할 일 |
|---|---|
| 더미 모드만 사용 | 없음. `pytest`로 확인만 합니다. |
| 모델 모드 사용 | `pip install -r requirements-models.txt`를 **다시 실행**합니다. |

11주차(FD-014·015·016)에서 바뀐 점:

- **PyTorch·SpeechBrain이 추가**됐습니다(FD-016 화자 임베딩). 설치에 수백 MB가 필요합니다. Windows의 PyPI `torch`는 CPU 전용입니다. Linux에서는 `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu`를 먼저 실행해야 수 GB짜리 CUDA 버전을 피할 수 있습니다.
- **`setuptools<81`이 고정**됐습니다. 최신 setuptools에서는 `tensorflow_hub` import가 `No module named 'pkg_resources'`로 실패합니다. 이 오류가 나면 `pip install "setuptools<81"`을 실행합니다.
- **처음 서버를 켤 때 화자 모델(약 80MB)을 추가로 내려받습니다.** Hugging Face 경고(`HF_TOKEN`)가 보여도 무시해도 됩니다.
- 예전에 YAMNet을 써 본 PC는 `%LOCALAPPDATA%\Temp\tfhub_modules`에 깨진 캐시가 남아 있을 수 있습니다. 이제 쓰지 않는 폴더라 지워도 됩니다.
- 응답 형식이 바뀌었습니다. `whisper`에 `status`·`score`·`matches`, `yamnet`에 `score`·`type`·`categories`가 추가됐습니다([API 문서](../api/README.md)). 더미 모드도 같은 형식을 돌려줍니다.
- 새 엔드포인트 `POST /voice-profile`(음성 등록, FD-016)이 생겼습니다.
- `.env.example`에 설정이 추가됐습니다. 기존 `.env`에 없어도 기본값으로 동작하지만, 바꾸려면 `.env.example`을 참고해 추가합니다.

설치 확인:

```powershell
$env:AI_RUN_MODEL_TESTS = "1"; pytest
```

모델 테스트까지 모두 통과하면 됩니다(처음에는 YAMNet을 내려받아 시간이 걸립니다).

## 같은 네트워크의 백엔드에서 호출할 때

- 서버를 `--host 0.0.0.0`으로 실행하고, 백엔드 쪽에는 `http://<이 PC의 IP>:8000`을 알려줍니다.
- Windows 방화벽에서 8000 포트 인바운드를 허용해야 할 수 있습니다.
- 외부에 노출할 때는 `AI_API_KEY`를 설정합니다.

### 다른 PC에서 연결 확인

[scripts/remote_check.py](../../scripts/remote_check.py)는 표준 라이브러리만 쓰므로 파일 하나만 복사해 Python 3.7 이상에서 바로 실행할 수 있습니다.

```powershell
python remote_check.py --url http://<AI 서버 PC IP>:8000
```

TCP 연결, `/health`, `/analyze`(multipart), `/analyze/pcm`을 차례로 확인하고 PASS/FAIL을 출력합니다. 연결이 안 되면 원인(서버 꺼짐·방화벽·다른 네트워크)을 구분해 안내합니다. 성공하면 서버 창에 보낸 PC의 IP로 `POST /analyze ... 200 OK` 로그가 찍힙니다.
