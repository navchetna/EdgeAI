import requests
import json
import base64

url = "http://localhost:8000/services/inference/pipeline"

payload = {
    "pipelineTasks": [
        {
            "taskType": "tts",
            "config": {
                "language": {"sourceLanguage": "hi"},
                "gender": "female",
                "samplingRate": 48000,
            },
        }
    ],
    "inputData": {
        "input": [
            {"source": "मेरा नाम कशा है"}
        ]
    },
}

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    audio_b64 = data["pipelineResponse"][0]["audio"][0]["audioContent"]
    audio_bytes = base64.b64decode(audio_b64)
    with open("test_output.wav", "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved to test_output.wav ({len(audio_bytes)} bytes)")
else:
    print(f"Error: {response.text}")
