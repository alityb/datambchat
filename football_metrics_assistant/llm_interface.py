import requests
from typing import List, Dict

OLLAMA_API_URL = "http://localhost:11434/api/chat"

def ask_llama(message: str, history: List[Dict] = None, model: str = "llama3.2:1b") -> str:
    if history is None:
        history = []
    
    # System message for detailed but concise responses
    messages = [{"role": "system", "content": "You are a football analytics expert. Provide clear, informative explanations of football statistics and metrics. Be conversational but concise - aim for 2-3 paragraphs maximum."}] + history + [{"role": "user", "content": message}]
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)  # Longer timeout for detailed responses
        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "")
        else:
            return f"[Ollama API error] {response.text}"
    except requests.exceptions.Timeout:
        return "[Timeout] Response took too long. Please try a simpler query."
    except Exception as e:
        return f"[Error] {str(e)}" 