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

def classify_query_type(query: str, model: str = "llama3.2:1b") -> str:
    # Simple rule-based classification first
    query_lower = query.lower()
    if any(word in query_lower for word in ['top', 'best', 'highest', 'most']):
        return "TOP_N"
    if any(word in query_lower for word in ['how many', 'count', 'number of']):
        return "COUNT"
    prompt = f"""
Classify this football analytics question as one of:
- COUNT: User wants a count of items (e.g., 'How many players in Serie A under 23?')
- TOP_N: User wants the top N by a stat (e.g., 'Top 5 in Premier League by assists')
- LIST: User wants a list of items (e.g., 'List all defenders in Bundesliga')
- FILTER: User wants to filter by a condition (e.g., 'Players in La Liga with more than 10 goals')
- OTHER: Anything else

Query: "{query}"
Type (just the type, nothing else):
"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that classifies football analytics questions by type. Only output the type label (COUNT, TOP_N, LIST, FILTER, OTHER)."},
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            content = data.get("message", {}).get("content", "").strip().upper()
            # Only keep the first word (in case LLM adds explanation)
            return content.split()[0]
        else:
            return "OTHER"
    except Exception as e:
        return "OTHER" 