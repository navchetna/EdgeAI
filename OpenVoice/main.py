import os
import time
import uvicorn
from loguru import logger 
from models import SpeechRequest

from tts import OpenVoiceTTS, numpy_to_wav_bytes
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(root_path="/v1")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = OpenVoiceTTS()


@app.post("/audio/speech")
async def generate_audio(request: SpeechRequest):
    logger.info("Request received")
    st = time.perf_counter()
    try:
        language = request.instructions
        audio, latency = pipeline.generate_cloned_voice(request.input)
        audio_bytes = numpy_to_wav_bytes(audio)
        latency = time.perf_counter() - st
        logger.info(f"Request processed: {latency}")
        return StreamingResponse(audio_bytes, media_type="audio/mpeg")

    except Exception as e:
        logger.info(f"Audio generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run("main:app",host="0.0.0.0", port=8000)