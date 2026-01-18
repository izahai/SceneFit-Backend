from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import uuid

from app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/v1/asr", tags=["asr"])

TMP_DIR = Path("/tmp/asr_feedback")
TMP_DIR.mkdir(parents=True, exist_ok=True)

def transcribe_audio(tmp_path: str):
    asr_model = ModelRegistry.get("asr")
    transcript = asr_model.transcribe(str(tmp_path))
    return transcript

@router.post("/audio_fb")
def asr_relevance_feedback(
    audio: UploadFile = File(...)
):
    if not audio.content_type or not audio.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Audio required."
        )
    tmp_path = TMP_DIR / f"{uuid.uuid4()}_{audio.filename}"
    try:
        with tmp_path.open("wb") as f:
            f.write(audio.file.read())
        transcript = transcribe_audio(tmp_path)
        return {
            "transcript": transcript,
            "signal_type": "voice_feedback"
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
