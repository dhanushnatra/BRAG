from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse


router = APIRouter(prefix="/output", tags=["audio"])

@router.get("/{file_name}")
def play_audio(file_name: str):
    file_path = Path(f"output/{file_name}")
    return FileResponse(file_path)

@router.get("/list/")
def list_audio_files():
    audio_files = [f for f in Path("output").glob("*.wav")]
    return {"audio_files": [str(f.name) for f in audio_files]}

