# 개발 환경 (Windows)

## Anaconda로 환경 만들기

```powershell
conda create -n safepath-ai python=3.11 -y
conda activate safepath-ai
cd <저장소 경로>\safepath_AI
pip install -r requirements-dev.txt
```

## 더미 모드 실행

```powershell
$env:AI_LOAD_MODELS = "0"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

다른 PowerShell 창에서:

```powershell
conda activate safepath-ai
python scripts\send_test.py
```

`/health` 응답과 `/analyze/pcm` 응답이 출력되고, 서버 창에 `analyze source=pcm bytes=48000 duration=1.5 mode=dummy` 로그가 남으면 수신 확인이 끝난 것입니다.

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
