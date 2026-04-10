"""
Bhashini-compatible NMT API server for EN -> Indic translation.
Uses IndicTrans2 OpenVINO inference engine.
Port: 8003
"""

import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bhashini 2-letter code -> FLORES code mapping
LANG_CODE_TO_FLORES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    "sa": "san_Deva",
    "sd": "snd_Arab",
    "mai": "mai_Deva",
    "doi": "doi_Deva",
    "ks": "kas_Arab",
    "mni": "mni_Beng",
    "sat": "sat_Olck",
    "brx": "brx_Deva",
    "gom": "gom_Deva",
    "lus": "lus_Latn",
}
FLORES_TO_LANG_CODE = {v: k for k, v in LANG_CODE_TO_FLORES.items()}


# --- Pydantic models for Bhashini pipeline request/response ---

class LanguageConfig(BaseModel):
    sourceLanguage: str
    targetLanguage: str | None = None
    sourceScriptCode: str | None = None
    targetScriptCode: str | None = None


class TaskConfig(BaseModel):
    language: LanguageConfig
    serviceId: str | None = None


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


class OutputItem(BaseModel):
    source: str
    target: str


class PipelineResponseItem(BaseModel):
    taskType: str
    config: dict | None = None
    output: list[OutputItem] | None = None


class PipelineResponse(BaseModel):
    pipelineResponse: list[PipelineResponseItem]


# --- App setup ---

app = FastAPI(title="IndicTrans2 EN->Indic NMT API (Bhashini-compatible)")

translator = None


@app.on_event("startup")
def load_model():
    global translator

    import importlib
    module = importlib.import_module("run_indictrans2-en-indic_ov")
    IndicTrans2OpenVINO = module.IndicTrans2OpenVINO

    device = os.getenv("NMT_DEVICE", "GPU")
    model_dir = os.getenv("NMT_MODEL_DIR", "./openvino_models/indictrans2-en-indic-1B-fp32/optimum")
    model_name = os.getenv("NMT_MODEL_NAME", "ai4bharat/indictrans2-en-indic-1B")
    warmup_iters = int(os.getenv("NMT_WARMUP", "3"))

    logger.info(f"Loading EN->Indic NMT model from '{model_dir}' on {device}...")
    translator = IndicTrans2OpenVINO(
        model_dir=model_dir,
        device=device,
        model_name=model_name,
    )

    if warmup_iters > 0:
        translator.warmup(warmup_iters)

    logger.info("EN->Indic NMT model ready")


def _translate(sentences: list[str], src_lang: str, tgt_lang: str) -> list[str]:
    return translator.translate(sentences, src_lang=src_lang, tgt_lang=tgt_lang)


@app.post("/services/inference/pipeline", response_model=PipelineResponse)
async def inference_pipeline(request: PipelineRequest):
    if not request.pipelineTasks:
        raise HTTPException(status_code=400, detail="pipelineTasks is empty")

    task = request.pipelineTasks[0]

    if task.taskType != "translation":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported taskType: '{task.taskType}'. Only 'translation' is supported.",
        )

    if translator is None:
        raise HTTPException(status_code=503, detail="NMT model not loaded")

    # Resolve language codes
    src_code = task.config.language.sourceLanguage.lower()
    tgt_code = (task.config.language.targetLanguage or "").lower()

    if src_code != "en":
        raise HTTPException(
            status_code=400,
            detail=f"This server only supports English as source language. Got: '{src_code}'",
        )

    if not tgt_code:
        raise HTTPException(status_code=400, detail="targetLanguage is required")

    src_flores = LANG_CODE_TO_FLORES.get(src_code)
    tgt_flores = LANG_CODE_TO_FLORES.get(tgt_code)

    if src_flores is None:
        raise HTTPException(status_code=400, detail=f"Unsupported source language: '{src_code}'")
    if tgt_flores is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported target language: '{tgt_code}'. Supported: {list(LANG_CODE_TO_FLORES.keys())}",
        )

    if not request.inputData.input:
        raise HTTPException(status_code=400, detail="inputData.input is empty")

    source_texts = [item.source for item in request.inputData.input]

    translations = await asyncio.to_thread(_translate, source_texts, src_flores, tgt_flores)

    outputs = [
        OutputItem(source=src, target=tgt)
        for src, tgt in zip(source_texts, translations)
    ]

    return PipelineResponse(
        pipelineResponse=[
            PipelineResponseItem(
                taskType="translation",
                config={
                    "language": {
                        "sourceLanguage": src_code,
                        "targetLanguage": tgt_code,
                    }
                },
                output=outputs,
            )
        ]
    )


@app.get("/health")
def health():
    if translator is None:
        return {"status": "error", "detail": "Model not loaded"}
    return {
        "status": "ok",
        "direction": "en-indic",
        "supported_targets": [k for k in LANG_CODE_TO_FLORES if k != "en"],
    }
