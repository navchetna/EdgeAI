import asyncio
import base64
import logging
import os
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openvino_inference_optimized import IndicASROpenVINOOptimized

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic models for Bhashini pipeline request/response ---


class LanguageConfig(BaseModel):
    sourceLanguage: str
    sourceScriptCode: str | None = None


class TaskConfig(BaseModel):
    language: LanguageConfig
    serviceId: str | None = None
    postProcessors: list[str] | None = None
    preProcessors: list[str] | None = None
    samplingRate: int | None = None


class PipelineTask(BaseModel):
    taskType: str
    config: TaskConfig


class AudioInputItem(BaseModel):
    audioContent: str | None = None
    audioUri: str | None = None


class InputData(BaseModel):
    audio: list[AudioInputItem] | None = None


class PipelineRequest(BaseModel):
    pipelineTasks: list[PipelineTask]
    inputData: InputData


class OutputItem(BaseModel):
    source: str


class PipelineResponseItem(BaseModel):
    taskType: str
    config: dict | None = None
    output: list[OutputItem] | None = None
    audio: list | None = None


class PipelineResponse(BaseModel):
    pipelineResponse: list[PipelineResponseItem]


# --- App setup ---

app = FastAPI(title="IndicConformer ASR API (Bhashini-compatible)")

asr_model: IndicASROpenVINOOptimized | None = None


@app.on_event("startup")
def load_model():
    """Load the IndicConformer model at startup."""
    global asr_model

    device = os.getenv("ASR_DEVICE", "GPU")
    model_dir = os.getenv("ASR_MODEL_DIR", "openvino_models")
    config_path = os.getenv("ASR_CONFIG_PATH", ".")

    logger.info(f"Loading IndicConformer model from '{model_dir}' on {device}...")
    try:
        asr_model = IndicASROpenVINOOptimized(
            model_dir=model_dir,
            config_path=config_path,
            device=device,
        )
        langs = asr_model.get_supported_languages()
        logger.info(f"Successfully loaded IndicConformer with {len(langs)} languages")
    except Exception:
        logger.exception("Failed to load IndicConformer model")


def _transcribe(audio_bytes: bytes, lang: str) -> str:
    """Run ASR inference on audio bytes and return transcribed text."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        text = asr_model.transcribe(tmp_path, lang=lang, decoding="ctc")
        return text
    finally:
        os.unlink(tmp_path)


@app.post("/services/inference/pipeline", response_model=PipelineResponse)
async def inference_pipeline(request: PipelineRequest):
    if not request.pipelineTasks:
        raise HTTPException(status_code=400, detail="pipelineTasks is empty")

    task = request.pipelineTasks[0]

    if task.taskType != "asr":
        raise HTTPException(status_code=400, detail=f"Unsupported taskType: '{task.taskType}'. Only 'asr' is supported.")

    if asr_model is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded")

    # Resolve language
    lang_code = task.config.language.sourceLanguage.lower()
    supported = asr_model.get_supported_languages()
    if lang_code not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported language: '{lang_code}'. Supported: {supported}")

    if not request.inputData.audio:
        raise HTTPException(status_code=400, detail="inputData.audio is empty")

    outputs = []
    for audio_input in request.inputData.audio:
        if audio_input.audioContent:
            audio_bytes = base64.b64decode(audio_input.audioContent)
        elif audio_input.audioUri:
            import urllib.request
            with urllib.request.urlopen(audio_input.audioUri) as resp:
                audio_bytes = resp.read()
        else:
            raise HTTPException(status_code=400, detail="Each audio item must have audioContent or audioUri")

        text = await asyncio.to_thread(_transcribe, audio_bytes, lang_code)
        outputs.append(OutputItem(source=text))

    return PipelineResponse(
        pipelineResponse=[
            PipelineResponseItem(
                taskType="asr",
                config=None,
                output=outputs,
                audio=None,
            )
        ]
    )


@app.get("/health")
def health():
    if asr_model is None:
        return {"status": "error", "detail": "Model not loaded"}
    info = asr_model.get_model_info()
    return {
        "status": "ok",
        "device": info["device"],
        "languages": info["languages"],
        "models_loaded": info["models_loaded"],
    }
