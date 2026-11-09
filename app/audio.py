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


def load_pcm16(data: bytes, sample_rate: int = TARGET_SR) -> np.ndarray:
    """원시 PCM(16bit little-endian, mono) 바이트를 16kHz float32로 변환한다.

    워치(WatchVoiceRecorder)가 보내는 형식: 16kHz, mono, ENCODING_PCM_16BIT.
    """
    if len(data) % 2:
        raise ValueError("PCM 바이트 길이가 홀수입니다. 16bit PCM인지 확인하세요.")
    waveform = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if sample_rate != TARGET_SR:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=TARGET_SR)
    return waveform


def duration_sec(waveform: np.ndarray) -> float:
    return round(len(waveform) / TARGET_SR, 3)
