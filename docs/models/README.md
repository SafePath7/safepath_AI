# 모델 정리: Whisper·YAMNet

> 9월 로컬 테스트 때 정리한 문서(`Whisper_YAMNet_세팅정리.txt`)를 옮긴 것입니다. 코드 위치는 `app/detectors/audio_detector.py`입니다.

```text
Safe Path - AI 파트 : Whisper(음성 탐지) + YAMNet(음향 분류) 세팅 정리
====================================================================

[개념 구분]
- Whisper  : 음성을 텍스트로 변환. "살려주세요" 같은 위험 문구 자체를 탐지.
- YAMNet   : 소리의 종류를 분류. 비명, 유리 깨지는 소리, 충격음 등 비언어 음향 이벤트 탐지.
- 두 모델은 담당 영역이 다르므로 하나의 오디오 클립에 대해 병렬로 돌리는 구조가 맞다.


====================================================================
1. Whisper 세팅
====================================================================

1) 설치
   pip install -U openai-whisper

   배포 단계에서 CPU로 돌릴 경우, faster-whisper(CTranslate2 기반, 동일 정확도에서
   최대 4배 빠름)를 추천.
   pip install faster-whisper


2) 모델 크기 선택

   모델    파라미터   속도(상대)   VRAM/메모리   용도
   tiny    39M        ~32x        ~1GB          실시간성 중요, 정확도 희생
   base    74M        ~16x        ~1GB          캡스톤 기본 추천
   small   244M       ~6x         ~2GB          정확도 더 필요할 때
   medium  769M       ~2x         ~5GB           CPU에선 부담 큼

   -> base로 시작, 위험 문구 인식률 부족하면 small로 상향.


3) 기본 사용 (openai-whisper)

   import whisper

   model = whisper.load_model("base")
   result = model.transcribe("clip.wav", language="ko")  # 한국어 명시 중요
   print(result["text"])


4) faster-whisper 버전 (배포용, 더 빠름)

   from faster_whisper import WhisperModel

   model = WhisperModel("base", device="cpu", compute_type="int8")  # int8이 CPU에서 빠름
   segments, info = model.transcribe("clip.wav", language="ko")
   text = " ".join(seg.text for seg in segments)


5) 위험 문구 탐지 로직

   DANGER_KEYWORDS = ["살려주세요", "살려줘", "도와주세요", "도와줘", "놔주세요", "그만해"]

   def detect_danger_phrase(text: str) -> bool:
       return any(kw in text for kw in DANGER_KEYWORDS)

   * Whisper가 오타/유사 발음으로 인식하는 경우가 있어, 추후 유사도 매칭
     (difflib 또는 자모 단위 비교)으로 보강하는 것을 고려.


6) 주의할 점
   - Whisper는 내부적으로 16kHz로 리샘플링해서 처리하므로 입력 샘플레이트는
     신경 쓰지 않아도 됨.
   - 짧은 클립(2~5초)일수록 문맥 부족으로 오인식이 늘어남. 5~10초 단위로 테스트 권장.


====================================================================
2. YAMNet 세팅
====================================================================

1) 설치
   pip install tensorflow tensorflow_hub numpy pandas


2) 모델 로드

   import tensorflow_hub as hub

   yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')


3) 오디오 입력 요구사항 (중요)
   YAMNet은 16kHz, mono, -1~1 범위의 float32 waveform만 받음. 반드시 변환 필요.

   import librosa

   waveform, sr = librosa.load("clip.wav", sr=16000, mono=True)  # sr=16000이 핵심


4) 추론

   import pandas as pd

   scores, embeddings, spectrogram = yamnet_model(waveform)

   class_map_path = yamnet_model.class_map_path().numpy()
   class_names = list(pd.read_csv(class_map_path)['display_name'])

   mean_scores = scores.numpy().mean(axis=0)
   top5_idx = mean_scores.argsort()[-5:][::-1]
   for i in top5_idx:
       print(class_names[i], mean_scores[i])


5) 위험 관련 클래스 필터링
   YAMNet은 AudioSet 기반 다수의 클래스를 출력. 클래스 개수나 이름은 모델
   버전에 따라 달라질 수 있으므로, 숫자를 하드코딩하지 말고 항상
   class_map_path()로 불러온 class_names에서 문자열 이름으로 검색해서
   사용할 것 (인덱스를 직접 믿고 접근하면 버전이 바뀔 때 조용히 틀린
   클래스를 잡는 버그가 생길 수 있음). 위험 감지에 쓸 만한 클래스만 추려서 체크.

   DANGER_CLASSES = [
       "Screaming", "Shout", "Yell", "Battle cry",
       "Glass", "Shatter", "Breaking",
       "Gunshot, gunfire", "Explosion", "Boom",
       "Smash, crash", "Thump, thud",
   ]

   def detect_danger_sound(scores, class_names, threshold=0.3):
       mean_scores = scores.numpy().mean(axis=0)
       hits = []
       for cls in DANGER_CLASSES:
           if cls in class_names:
               idx = class_names.index(cls)
               if mean_scores[idx] > threshold:
                   hits.append((cls, float(mean_scores[idx])))
       return hits

   * threshold(0.3)는 시작점. 실측 데이터로 조정 필요.


====================================================================
3. 둘을 나란히 돌릴 때 체크포인트
====================================================================
- 샘플레이트 통일 : 녹음/전송 단계에서 아예 16kHz mono로 통일해두면 Whisper,
  YAMNet 양쪽에 그대로 입력 가능 (전처리 단순화).
- 같은 클립을 병렬로 : 하나의 오디오 클립을 Whisper(문구)와 YAMNet(음향)에
  동시에 넣고, 두 결과를 이후 합치는 구조.
- YAMNet이 Whisper보다 훨씬 가벼움 (TF Hub 모델) -> 리소스는 Whisper 쪽에
  더 배분.


====================================================================
4. 입력 데이터 형식
====================================================================

[Whisper]
- 받을 수 있는 파일 : wav, mp3, m4a, flac, ogg 등 ffmpeg가 지원하는 거의 모든
  오디오 포맷 (내부적으로 ffmpeg 호출해서 디코딩).
- 내부적으로 실제 처리하는 형태 : 16kHz, mono, float32 waveform으로 강제 변환
  (원본이 44.1kHz 스테레오 mp3여도 알아서 리샘플링).
- 코드로 넣는 방법 3가지 :
    model.transcribe("clip.wav")            # 1. 파일 경로
    model.transcribe(audio_array)           # 2. numpy 배열 (float32, shape (n_samples,))
    # 3. bytes로 받은 경우 -> 임시파일로 저장 후 경로로 넘기는 게 무난함

[YAMNet]
- 딱 하나만 받음 : float32 numpy 배열, 16kHz, mono, -1.0~1.0 범위로 정규화된
  waveform. shape은 (n_samples,) 1차원.
- Whisper처럼 자동 변환 안 해주므로 직접 맞춰야 함.
    import librosa
    waveform, sr = librosa.load("clip.wav", sr=16000, mono=True)
    # librosa.load는 기본적으로 -1~1 정규화까지 같이 해줌

    # 원본이 int16 PCM(스마트폰 마이크에서 흔한 형식)인 경우
    waveform = pcm_int16.astype(np.float32) / 32768.0  # -1~1로 정규화

[실제 서비스 흐름 참고]
1. 폰 -> FastAPI : 오디오는 보통 wav 또는 m4a 파일(혹은 base64 인코딩)로 전송
2. FastAPI에서 받은 후 : 원본 오디오를 16kHz mono로 한 번 변환해두고
   - Whisper에는 파일 경로 또는 그 배열 그대로
   - YAMNet에는 float32 정규화까지 마친 배열로
   -> 입구에서 한 번만 16kHz mono로 통일해두면 두 모델에 개별 변환이 필요 없어
      코드가 깔끔해짐 (앞의 "샘플레이트 통일" 체크포인트와 동일한 이유).


====================================================================
5. 실시간으로 받을 때 문제되는 지점 (나중에 참고)
====================================================================
Whisper/YAMNet은 "한 문장을 다 듣고 해석하는" 모델이라 순수 실시간 스트림을
바로 흘려넣을 수 없음. 짧은 조각을 반복 처리하는 "주기적 업로드" 방식으로
갈 수밖에 없는데, 이 방식에서 특히 먼저 부딪히는 지점:

1) 청크 경계에서 소리/단어가 잘림 (가장 흔한 문제)
   - 5초 단위로 뚝뚝 자르면 "살려주세요"가 "살려" + "주세요"로 나뉘어
     둘 다 위험 문구로 안 잡힐 수 있음.
   - 해결 : 청크를 겹치게 자르기 (overlap)
       CHUNK_SEC = 5
       OVERLAP_SEC = 1  # 앞 청크와 1초 겹치게
     겹친 만큼 같은 소리를 두 번 탐지할 수 있으므로 타임스탬프 기준 중복 제거 필요.

2) 처리 속도가 실시간을 못 따라가면 백로그가 쌓임
   - 5초 오디오 분석에 7초가 걸리면 계속 밀려서 뒤늦게 알림이 갈 수 있음
     (안전 서비스에서 치명적).
   - 체크할 것 : 모델의 real-time factor. whisper-base가 CPU에서 5초 오디오를
     처리하는 데 몇 초 걸리는지 실측 후, 5초 미만이 아니면 tiny로 낮추거나
     GPU 검토.

3) 모델을 요청마다 다시 로드하면 느려짐 (흔한 실수)
   - FastAPI 엔드포인트 안에서 whisper.load_model()을 매번 호출하면 로딩
     시간(수 초)이 매 요청마다 추가됨.
   - 해결 : 서버 시작 시 한 번만 로드해서 메모리에 계속 들고 있기.
       from contextlib import asynccontextmanager
       from fastapi import FastAPI

       models = {}

       @asynccontextmanager
       async def lifespan(app: FastAPI):
           models["whisper"] = whisper.load_model("base")
           models["yamnet"] = hub.load('https://tfhub.dev/google/yamnet/1')
           yield

       app = FastAPI(lifespan=lifespan)

4) 무음 구간까지 다 분석하면 리소스 낭비
   - 사용자가 조용히 걷기만 하는 시간이 대부분인데 그 구간까지 계속 돌리면
     서버 자원 낭비.
   - 해결 : 가벼운 VAD(Voice Activity Detection)를 앞단에 두고, 소리가
     임계값 이상일 때만 본 모델로 넘기기 (webrtcvad 같은 초경량 라이브러리로 충분).

5) 동시 사용자 여러 명이면 한 대의 서버가 병목
   - Whisper/YAMNet 인스턴스 하나가 순차 처리하면 사용자가 늘수록 대기열이
     길어짐. 캡스톤 데모(팀원 몇 명 테스트) 수준에선 문제없음.
   - 참고만 해둘 것 : 나중에 필요해지면 워커 프로세스를 늘리거나(Gunicorn
     multi-worker), 큐(Celery/Redis)를 붙이는 식으로 확장 가능. 지금 단계
     에서 구현할 필요는 없음.

6) 네트워크 불안정 (모바일 환경 특성)
   - 걸어가면서 업로드하면 패킷 유실, 순서 뒤바뀜, 업로드 실패가 흔함.
     청크 하나가 유실되면 그 구간은 통째로 분석을 못 받음.
   - 해결 방향 : 업로드 실패 시 재시도 로직 + 청크에 타임스탬프를 찍어서
     순서 보장.

* 지금 단계(로컬에서 모델 테스트)에서는 신경 안 써도 됨. 나중에 FastAPI에
  실제로 붙일 때 1번(경계 잘림)과 3번(모델 재로딩)을 제일 먼저 마주칠
  가능성이 높음. 4, 5, 6번은 서비스가 커질 때 고려할 부분.


====================================================================
6. 참고만 (추후 연결 과정 - 지금 단계에서는 몰라도 됨)
====================================================================
나중에 FastAPI에 얹을 때, detect_danger_phrase()와 detect_danger_sound()의
출력을 그대로 멀티모달 위험도 점수(AI-04) 산출 로직의 입력으로 합치면 됨.
지금은 Whisper, YAMNet 각각이 로컬에서 잘 동작하는지에만 집중.

```
