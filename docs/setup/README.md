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

- 첫 실행 때 Whisper(base)와 YAMNet 가중치를 내려받으므로 시간이 걸립니다.
- 설치가 끝나면 `pip freeze`로 확인한 버전을 `requirements-models.txt`에 고정합니다.
- mp3·m4a 파일 디코딩에는 ffmpeg가 필요할 수 있습니다. wav와 `/analyze/pcm`은 필요하지 않습니다.

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
