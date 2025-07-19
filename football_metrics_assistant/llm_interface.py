import requests
from typing import List, Dict

OLLAMA_API_URL = "http://localhost:11434/api/chat"

def ask_llama(message: str, history: List[Dict] = None, model: str = "llama3") -> str:
    if history is None:
        history = []
    payload = {
        "model": model,
        "messages": history + [{"role": "user", "content": message}],
        "stream": False
    }
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("message", {}).get("content", "")
    else:
        return f"[Ollama API error] {response.text}" 