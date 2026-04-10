from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="None")
speech_file_path = Path(__file__).parent / "tts_audio.wav"

with client.audio.speech.with_streaming_response.create(
    model="fastspeech2",
    input="Once upon a time there was a boy who used to live in a dark forest.",
    voice="af_alloy",
) as response:
    response.stream_to_file(speech_file_path)