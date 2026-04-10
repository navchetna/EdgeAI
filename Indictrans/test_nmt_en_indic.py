"""Test script for EN -> Indic NMT server on port 8003."""

import json
import requests

url = "http://localhost:8003/services/inference/pipeline"

payload = {
    "pipelineTasks": [
        {
            "taskType": "translation",
            "config": {
                "language": {
                    "sourceLanguage": "en",
                    "targetLanguage": "hi",
                },
                "serviceId": "indictrans2-en-indic",
            },
        }
    ],
    "inputData": {
        "input": [
            {"source": "My name is Ritik and I live in Bangalore."},
            {"source": "The weather is very nice today."},
            {"source": "India is a beautiful country with diverse cultures."},
        ]
    },
}

print("Testing EN -> Hindi translation...")
print(f"URL: {url}")
print()

response = requests.post(url, json=payload)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    for item in data["pipelineResponse"][0]["output"]:
        print(f"  SRC: {item['source']}")
        print(f"  TGT: {item['target']}")
        print()
else:
    print(f"Error: {response.text}")
