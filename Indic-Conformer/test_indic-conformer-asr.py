import os
import requests
import base64

url = "http://localhost:8002/services/inference/pipeline"

# Use a sample audio file - adjust path as needed
audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", "hindi.wav")
if not os.path.exists(audio_file):
    # Fallback: try any .wav file in current directory
    for f in os.listdir(os.path.dirname(os.path.abspath(__file__))):
        if f.endswith(".wav"):
            audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
            break

print(f"Using audio file: {audio_file}")

with open(audio_file, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("ascii")

payload = {
    "pipelineTasks": [
        {
            "taskType": "asr",
            "config": {
                "language": {"sourceLanguage": "hi"},
                "postProcessors": ["itn", "punctuation"],
            },
        }
    ],
    "inputData": {
        "audio": [
            {"audioContent": audio_b64}
        ]
    },
}

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    for item in data["pipelineResponse"][0]["output"]:
        print(f"Transcription: {item['source']}")
else:
    print(f"Error: {response.text}")
