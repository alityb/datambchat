from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/chat"

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    # Forward the message to Ollama's API
    payload = {
        "model": "llama3",
        "messages": req.history + [{"role": "user", "content": req.message}],
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        return {"response": data.get("message", {}).get("content", "")}
    else:
        return {"error": "Ollama API error", "details": response.text}

@app.get("/stat-definitions")
def stat_definitions():
    # Placeholder for stat definitions endpoint
    return {"definitions": []} 