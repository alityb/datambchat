from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
from preprocessor import preprocess_query
from retriever import HybridRetriever
from llm_interface import ask_llama
from tools import analyze_query  # Uncomment this to use real data
import time

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

    # 3. Use tools for real data filtering/sorting
    data_analysis = None
    if preprocessed.get('stat') and (preprocessed.get('top_n') or preprocessed.get('team') or preprocessed.get('position') or preprocessed.get('league')):
        try:
            data_analysis = analyze_query(preprocessed)
        except Exception as e:
            data_analysis = {"error": f"Data analysis failed: {str(e)}"}

    # 4. Compose context for LLM
    context_parts = []
    
    # Add relevant context
    if stat_context:
        context_parts.append("Stat Definitions:")
        for stat_def in stat_context[:2]:  # Limit to 2 most relevant
            context_parts.append(f"- {stat_def['text']}")
    
    if position_context:
        context_parts.append("Position Information:")
        for pos_info in position_context[:1]:  # Limit to 1 most relevant
            context_parts.append(f"- {pos_info['text']}")
    
    if analysis_context:
        context_parts.append("Analysis Guidelines:")
        for guide in analysis_context[:1]:  # Limit to 1 most relevant
            context_parts.append(f"- {guide['text']}")
    
    # Add preprocessed hints
    context_parts.append(f"Query Analysis: {preprocessed}")
    
    # Add real data analysis if available
    if data_analysis and not data_analysis.get('error'):
        if data_analysis.get('success') and data_analysis.get('top_players'):
            context_parts.append("Real Data Results:")
            context_parts.append(f"Top {len(data_analysis['top_players'])} players found:")
            for i, player in enumerate(data_analysis['top_players'][:5], 1):  # Limit to top 5
                context_parts.append(f"{i}. {player['Player']} ({player['Team within selected timeframe']}) - {player.get('Position', 'N/A')} - {player.get(data_analysis.get('stat', ''), 'N/A')}")
        elif data_analysis.get('filtered_data'):
            context_parts.append("Filtered Data:")
            for player in data_analysis['filtered_data'][:3]:  # Limit to 3
                context_parts.append(f"- {player['Player']} ({player['Team within selected timeframe']}) - {player.get('Position', 'N/A')}")
    
    context = "\n\n".join(context_parts)

    # 5. Send everything to Llama 3 (Ollama)
    llm_start = time.time()
    llm_prompt = f"User query: {req.message}\n\nContext:\n{context}"
    llm_response = ask_llama(llm_prompt, req.history)
    llm_time = time.time() - llm_start

    # 6. Return a clean, conversational answer
    if data_analysis and not data_analysis.get('error') and data_analysis.get('success'):
        # Return real data directly in table format
        response = f"Here are the top {len(data_analysis['top_players'])} players by {data_analysis.get('stat', 'the requested stat')}:\n\n"
        response += "| Rank | Player | Team | Position | Value |\n"
        response += "|------|--------|------|----------|-------|\n"
        
        for i, player in enumerate(data_analysis['top_players'], 1):
            player_name = player['Player']
            team = player['Team within selected timeframe']
            position = player.get('Position', 'N/A')
            stat_value = player.get(data_analysis.get('stat', ''), 'N/A')
            response += f"| {i} | {player_name} | {team} | {position} | {stat_value} |\n"
        
        response += f"\nData based on {data_analysis.get('count', 0)} players matching your criteria."
    else:
        # For non-data queries, use LLM response
        response = llm_response

    return {
        "response": response, 
        "preprocessed": preprocessed, 
        "retrieval": {
            "stat_definitions": len(stat_context),
            "position_info": len(position_context),
            "analysis_guides": len(analysis_context)
        },
        "data_analysis": data_analysis if data_analysis else None
    }

@app.get("/stat-definitions")
def stat_definitions():
    # Placeholder for stat definitions endpoint
    return {"definitions": []} 