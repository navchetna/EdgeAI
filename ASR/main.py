import os
import time
import uvicorn
from loguru import logger
from whisper import Whisper

from models import TranscriptionRequest, TranscriptionResponse

from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, Form, UploadFile, FastAPI, HTTPException


app = FastAPI(root_path="/v1")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

asr = Whisper(
    model_name=os.getenv("MODEL_NAME", "whisper_large_v3"),
    device=os.getenv("DEVICE", "GPU")
)


@app.post("/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Source audio file"),
    model: str = Form(..., description="Model Name"),
    response_format: str = Form("text", description="Format response")
):
    logger.info("Request received")
    st = time.perf_counter()
    audio_bytes = await file.read()
    audio_array, sr = asr.convert_bytes_to_array(audio_bytes)
    transcription = asr.generate(audio_array)
    response_time = time.perf_counter() - st
    logger.info("Request processed")
    return TranscriptionResponse(text=str(transcription), latency=response_time)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)
    