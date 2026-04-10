"""Test script for Indic -> Indic NMT server on port 8004."""

import json
import requests

url = "http://localhost:8004/services/inference/pipeline"

payload = {
    "pipelineTasks": [
        {
            "taskType": "translation",
            "config": {
                "language": {
                    "sourceLanguage": "hi",
                    "targetLanguage": "bn",
                },
                "serviceId": "indictrans2-indic-indic",
            },
        }
    ],
    "inputData": {
        "input": [
            {"source": "\u091c\u092c \u092e\u0948\u0902 \u091b\u094b\u091f\u093e \u0925\u093e, \u0924\u094b \u092e\u0948\u0902 \u0939\u0930 \u0926\u093f\u0928 \u092a\u093e\u0930\u094d\u0915 \u092e\u0947\u0902 \u091c\u093e\u0924\u093e \u0925\u093e\u0964"},
            {"source": "\u092d\u093e\u0930\u0924 \u090f\u0915 \u0938\u0941\u0902\u0926\u0930 \u0926\u0947\u0936 \u0939\u0948\u0964"},
            {"source": "\u0906\u091c \u092e\u094c\u0938\u092e \u092c\u0939\u0941\u0924 \u0905\u091a\u094d\u091b\u093e \u0939\u0948\u0964"},
        ]
    },
}

print("Testing Hindi -> Bengali translation...")
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
