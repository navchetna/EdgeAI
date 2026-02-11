from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="None")
audio_file= open("sample.wav", "rb")

transcription = client.audio.transcriptions.create(
    model="wav2vec", 
    file=audio_file
)

print(transcription.text)