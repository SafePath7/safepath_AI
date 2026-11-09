"""오디오 입력을 16kHz mono float32(-1~1) waveform으로 통일한다."""

import os
import tempfile

import numpy as np

TARGET_SR = 16000


def load_audio_file(data: bytes, filename: str) -> np.ndarray:
    """업로드된 오디오 파일(wav, mp3, m4a 등)을 16kHz mono float32로 변환한다."""
    import librosa  # 모델 모드에서만 필요하므로 지연 import

    suffix = os.path.splitext(filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        waveform, _ = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
    finally:
        os.unlink(tmp_path)
    return waveform.astype(np.float32)


def check_pcm16(data: bytes) -> None:
    """16bit PCM 바이트 길이를 검사한다. 더미 모드에서도 같은 기준으로 400을 돌려주기 위해 분리했다."""
    if len(data) % 2:
        raise ValueError("PCM 바이트 길이가 홀수입니다. 16bit PCM인지 확인하세요.")


def pcm16_duration_sec(data: bytes, sample_rate: int) -> float:
    """디코딩 없이 16bit mono PCM 바이트 길이로 재생 시간을 계산한다."""
    return round(len(data) / 2 / sample_rate, 3)


def load_pcm16(data: bytes, sample_rate: int = TARGET_SR) -> np.ndarray:
    """원시 PCM(16bit little-endian, mono) 바이트를 16kHz float32로 변환한다.

    워치(WatchVoiceRecorder)가 보내는 형식: 16kHz, mono, ENCODING_PCM_16BIT.
    16kHz가 아니면 librosa로 리샘플링하므로 모델 모드 의존성이 필요하다.
    """
    check_pcm16(data)
    waveform = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if sample_rate != TARGET_SR:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
    return waveform


def duration_sec(waveform: np.ndarray) -> float:
    return round(len(waveform) / TARGET_SR, 3)
