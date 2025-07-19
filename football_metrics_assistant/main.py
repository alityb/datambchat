from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
from football_metrics_assistant.preprocessor import preprocess_query
from football_metrics_assistant.retriever import HybridRetriever
from football_metrics_assistant.llm_interface import ask_llama
# from football_metrics_assistant.tools import filter_players, generate_chart  # For future use

app = FastAPI()

OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Initialize retriever (stub)
retriever = HybridRetriever()

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    # 1. Preprocess the query for structured hints
    preprocessed = preprocess_query(req.message)

    # 2. Retrieve stat definitions/context using preprocessed hints
    retrieval = retriever.retrieve(req.message, preprocessed_hints=preprocessed)
    stat_context = retrieval.get("stat_definitions", [])
    position_context = retrieval.get("position_info", [])
    analysis_context = retrieval.get("analysis_guides", [])

    # 3. (Stub) Use tools for filtering/sorting if needed
    # In the future: filter_players, generate_chart, etc.

    # 4. Compose context for LLM
    context_parts = []
    
    # Add stat definitions
    if stat_context:
        context_parts.append("Stat Definitions:")
        for stat_def in stat_context:
            context_parts.append(f"- {stat_def['text']}")
    
    # Add position information
    if position_context:
        context_parts.append("Position Information:")
        for pos_info in position_context:
            context_parts.append(f"- {pos_info['text']}")
    
    # Add analysis guides
    if analysis_context:
        context_parts.append("Analysis Guidelines:")
        for guide in analysis_context:
            context_parts.append(f"- {guide['text']}")
    
    # Add preprocessed hints
    context_parts.append(f"Query Analysis: {preprocessed}")
    
    context = "\n\n".join(context_parts)

    # 5. Send everything to Llama 3 (Ollama)
    llm_prompt = f"User query: {req.message}\n\nContext:\n{context}"
    llm_response = ask_llama(llm_prompt, req.history)

    # 6. Return a clean, conversational answer
    return {
        "response": llm_response, 
        "preprocessed": preprocessed, 
        "retrieval": {
            "stat_definitions": len(stat_context),
            "position_info": len(position_context),
            "analysis_guides": len(analysis_context)
        }
    }

@app.get("/stat-definitions")
def stat_definitions():
    # Placeholder for stat definitions endpoint
    return {"definitions": []} 