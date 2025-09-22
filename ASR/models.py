from pydantic import BaseModel

class TranscriptionRequest(BaseModel):
    model: str
    file: bytes
    response_type: str = "text"


class TranscriptionResponse(BaseModel):
    text: str
    latency: float