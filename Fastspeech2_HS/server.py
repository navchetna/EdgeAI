import asyncio
import base64
import io
import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.io.wavfile import write as wav_write

from main_ov import Text2SpeechApp
from utilities import SAMPLING_RATE, SUPPORTED_OUTPUT_LANGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Language code mapping (Bhashini 2-letter <-> full name) ---

LANG_CODE_TO_NAME = {
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "pa": "punjabi",
}
LANG_NAME_TO_CODE = {v: k for k, v in LANG_CODE_TO_NAME.items()}

# --- Pydantic models for Bhashini pipeline request/response ---


class LanguageConfig(BaseModel):
    sourceLanguage: str
    sourceScriptCode: str | None = None
    targetLanguage: str | None = None


class TaskConfig(BaseModel):
    language: LanguageConfig
    serviceId: str | None = None
    gender: str = "female"
    samplingRate: int = 48000


class PipelineTask(BaseModel):
    taskType: str
    config: TaskConfig


class InputItem(BaseModel):
    source: str


class InputData(BaseModel):
    input: list[InputItem] | None = None


class PipelineRequest(BaseModel):
    pipelineTasks: list[PipelineTask]
    inputData: InputData


class AudioItem(BaseModel):
    audioContent: str | None = None
    audioUri: str | None = None


class ResponseConfig(BaseModel):
    audioFormat: str = "wav"
    language: LanguageConfig
    encoding: str = "base64"
    samplingRate: int = 48000


class PipelineResponseItem(BaseModel):
    taskType: str
    config: ResponseConfig
    output: list | None = None
    audio: list[AudioItem] | None = None


class PipelineResponse(BaseModel):
    pipelineResponse: list[PipelineResponseItem]


# --- App setup ---

app = FastAPI(title="FastSpeech2 TTS API (Bhashini-compatible)")

# Dict of language_name -> Text2SpeechApp instance
tts_engines: dict[str, Text2SpeechApp] = {}


@app.on_event("startup")
def load_models():
    """Load TTS models for all configured languages at startup."""
    for lang_name in SUPPORTED_OUTPUT_LANGS:
        lang_name = lang_name.strip().lower()
        if lang_name not in LANG_NAME_TO_CODE:
            logger.warning(f"Unknown language '{lang_name}' in LANGUAGES env var, skipping.")
            continue
        logger.info(f"Loading TTS models for '{lang_name}'...")
        try:
            tts_engines[lang_name] = Text2SpeechApp(language=lang_name, dtype=os.getenv("TTS_DTYPE", "float32"))
            logger.info(f"Successfully loaded '{lang_name}' with genders: {tts_engines[lang_name].supported_genders}")
        except Exception:
            logger.exception(f"Failed to load models for '{lang_name}'")


def _synthesize(tts_app: Text2SpeechApp, text: str, gender: str, requested_sr: int) -> str:
    """Run TTS inference and return base64-encoded WAV string."""
    audio_tensor = tts_app.generate_audio_bytes(text=text, speaker_gender=gender)

    # Convert to int16 numpy
    if hasattr(audio_tensor, "numpy"):
        audio_np = audio_tensor.numpy().astype(np.int16)
    else:
        audio_np = np.array(audio_tensor, dtype=np.int16)

    # Resample if requested rate differs from native rate
    output_sr = SAMPLING_RATE
    if requested_sr != SAMPLING_RATE:
        import librosa
        audio_float = audio_np.astype(np.float32) / 32768.0
        audio_float = librosa.resample(audio_float, orig_sr=SAMPLING_RATE, target_sr=requested_sr)
        audio_np = (audio_float * 32768.0).astype(np.int16)
        output_sr = requested_sr

    # Write WAV to in-memory buffer
    buf = io.BytesIO()
    wav_write(buf, output_sr, audio_np)
    wav_bytes = buf.getvalue()

    return base64.b64encode(wav_bytes).decode("ascii")


@app.post("/services/inference/pipeline", response_model=PipelineResponse)
async def inference_pipeline(request: PipelineRequest):
    if not request.pipelineTasks:
        raise HTTPException(status_code=400, detail="pipelineTasks is empty")

    task = request.pipelineTasks[0]

    if task.taskType != "tts":
        raise HTTPException(status_code=400, detail=f"Unsupported taskType: '{task.taskType}'. Only 'tts' is supported.")

    # Resolve language
    lang_code = task.config.language.sourceLanguage.lower()
    lang_name = LANG_CODE_TO_NAME.get(lang_code)
    if not lang_name:
        raise HTTPException(status_code=400, detail=f"Unsupported language code: '{lang_code}'")

    if lang_name not in tts_engines:
        raise HTTPException(status_code=400, detail=f"Language '{lang_name}' not loaded. Available: {list(tts_engines.keys())}")

    tts_app = tts_engines[lang_name]

    # Resolve gender
    gender = task.config.gender.lower()
    if gender not in tts_app.supported_genders:
        raise HTTPException(status_code=400, detail=f"Gender '{gender}' not available for '{lang_name}'. Available: {tts_app.supported_genders}")

    requested_sr = task.config.samplingRate

    # Validate input
    if not request.inputData.input:
        raise HTTPException(status_code=400, detail="inputData.input is empty")

    # Process all input texts and collect audio
    audio_items = []
    for item in request.inputData.input:
        b64_audio = await asyncio.to_thread(_synthesize, tts_app, item.source, gender, requested_sr)
        audio_items.append(AudioItem(audioContent=b64_audio, audioUri=None))

    response = PipelineResponse(
        pipelineResponse=[
            PipelineResponseItem(
                taskType="tts",
                config=ResponseConfig(
                    audioFormat="wav",
                    language=LanguageConfig(sourceLanguage=lang_code, sourceScriptCode=""),
                    encoding="base64",
                    samplingRate=requested_sr,
                ),
                output=None,
                audio=audio_items,
            )
        ]
    )
    return response


@app.get("/health")
def health():
    return {
        "status": "ok",
        "languages": {
            lang: engine.supported_genders for lang, engine in tts_engines.items()
        },
    }
