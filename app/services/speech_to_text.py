from fastapi import UploadFile, File, HTTPException
from app.services.model_registry import ModelRegistry
import io
import numpy as np
import soundfile as sf
import librosa

SAMPLE_RATE = 16000

def load_audio_from_upload(file: UploadFile,) -> np.ndarray:
    """
    Decode uploaded audio into a mono float32 waveform @ 16 kHz
    """
    try:
        audio_bytes = file.file.read()
        data, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
            always_2d=False
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

    # Convert stereo → mono if needed
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)

    return data


def convert_speech_to_text(audio: UploadFile = File(...)):
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Audio required."
        )
    waveform = load_audio_from_upload(audio)
    asr_model = ModelRegistry.get("asr")
    transcript = asr_model.transcribe(waveform)
    return transcript