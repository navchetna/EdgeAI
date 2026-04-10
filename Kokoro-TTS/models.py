from typing import Literal
from pydantic import BaseModel

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "af_alloy"
    instructions: str = None
    response_format: str = None
    speed: float = 1.0
    stream_format: Literal['sse', 'audio'] = 'sse'