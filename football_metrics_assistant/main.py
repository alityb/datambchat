from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
from football_metrics_assistant.preprocessor import preprocess_query
from football_metrics_assistant.retriever import HybridRetriever
from football_metrics_assistant.llm_interface import ask_llama
from football_metrics_assistant.tools import analyze_query  # Uncomment this to use real data
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
    # Always analyze if there are any filters or if it's a COUNT/LIST query
    if (
        preprocessed.get('stat')
        or preprocessed.get('query_type', '').startswith('COUNT')
        or preprocessed.get('query_type', '').startswith('LIST')
        or preprocessed.get('top_n')
        or preprocessed.get('team')
        or preprocessed.get('position')
        or preprocessed.get('league')
        or preprocessed.get('age_filter')
    ):
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
    if data_analysis and not data_analysis.get('error') and data_analysis.get('success'):
        # NEW: Simplified multi-league logic - now we have overall top players
        if data_analysis.get('multi_league', False):
            # Get the overall top players (already sorted)
            top_players = data_analysis.get('top_players', [])
            stat_col = data_analysis.get('stat')
            overall_summary = data_analysis.get('summary', {})
            
            # Build combined table showing overall top N
            table = None
            if top_players:
                table = f"| Rank | Player | Team | League | Position | Value |\n"
                table += "|------|--------|------|--------|----------|-------|\n"
                for i, player in enumerate(top_players, 1):
                    player_name = player['Player']
                    team = player['Team within selected timeframe']
                    league = player.get('League', 'N/A')
                    position = player.get('Position', 'N/A')
                    stat_value = player.get(stat_col, 'N/A')
                    if isinstance(stat_value, (int, float)):
                        stat_value = f"{stat_value:.2f}"
                    table += f"| {i} | {player_name} | {team} | {league} | {position} | {stat_value} |\n"
                
                total_players = data_analysis.get('count', 0)
                leagues_list = preprocessed.get('league', [])
                leagues_str = ', '.join(leagues_list) if isinstance(leagues_list, list) else str(leagues_list)
                table += f"\nOverall top {len(top_players)} from {total_players} players across {leagues_str}."
            
            # Build summary for overall top player
            summary = "No players found matching your criteria."
            if top_players and stat_col:
                top_player = top_players[0]
                player_name = top_player['Player']
                team = top_player['Team within selected timeframe']
                league = top_player.get('League', 'N/A')
                stat_value = top_player.get(stat_col, 'N/A')
                if isinstance(stat_value, (int, float)):
                    stat_value = f"{stat_value:.2f}"
                
                leagues_list = preprocessed.get('league', [])
                leagues_str = ', '.join(leagues_list) if isinstance(leagues_list, list) else str(leagues_list)
                
                # Add age filter info if present
                age_info = ""
                if preprocessed.get('age_filter'):
                    age_op = preprocessed['age_filter']['op']
                    age_val = preprocessed['age_filter']['value']
                    age_text = f"under {age_val}" if age_op == "<" else f"over {age_val}" if age_op == ">" else f"age {age_val}"
                    age_info = f" ({age_text})"
                
                summary = f"{player_name} ({team}, {league}) has the highest {stat_col} among players{age_info} in {leagues_str} with {stat_value}."
            
            return {
                "summary": summary,
                "table": table,
                "preprocessed": preprocessed,
                "retrieval": {
                    "stat_definitions": len(stat_context),
                    "position_info": len(position_context),
                    "analysis_guides": len(analysis_context)
                },
                "data_analysis": data_analysis
            }
        # Determine the correct stat column name for table and summary
        stat_col = None
        if 'stat_formula' in preprocessed:
            # Use display_expr if present
            stat_col = preprocessed['stat_formula'].get('display_expr')
        elif 'stat' in data_analysis:
            stat_col = data_analysis['stat']
        elif 'stat' in preprocessed:
            stat_col = preprocessed['stat']
        else:
            stat_col = ''
        filtered_data = []
        # Prefer top_players if present, else filtered_data
        if data_analysis.get('top_players'):
            filtered_data = data_analysis['top_players']
        elif data_analysis.get('filtered_data'):
            filtered_data = data_analysis['filtered_data']
        # Build the table
        table = None
        if filtered_data:
            table = f"| Rank | Player | Team | Position | Value |\n"
            table += "|------|--------|------|----------|-------|\n"
            for i, player in enumerate(filtered_data, 1):
                player_name = player['Player']
                team = player['Team within selected timeframe']
                position = player.get('Position', 'N/A')
                stat_value = player.get(stat_col, 'N/A')
                table += f"| {i} | {player_name} | {team} | {position} | {stat_value} |\n"
            table += f"\nData based on {data_analysis.get('count', 0)} players matching your criteria."
        # Unified summary logic
        if stat_col and filtered_data:
            # Sort by stat_col descending and get the top player
            try:
                top_player = max(filtered_data, key=lambda p: p.get(stat_col, float('-inf')) if isinstance(p.get(stat_col), (int, float)) else float('-inf'))
                player_name = top_player['Player']
                team = top_player['Team within selected timeframe']
                stat_value = top_player.get(stat_col, 'N/A')
                league = preprocessed.get('league', '')
                league_str = f" in {league}" if league else ""
                summary = f"{player_name} ({team}) had the highest {stat_col}{league_str} with {stat_value}."
            except Exception as e:
                summary = f"Top player summary unavailable due to error: {e}"
        elif filtered_data:
            league = preprocessed.get('league', '')
            value = preprocessed.get('stat_value')
            op = preprocessed.get('stat_op')
            op_str = {
                '>=': 'at least',
                '>': 'more than',
                '<=': 'at most',
                '<': 'less than',
                '==': 'exactly'
            }.get(op, op)
            summary = f"There are {data_analysis['count']} players in {league} with {op_str} {value} {stat_col}."
        else:
            summary = "No players found matching your criteria."
        return {
            "summary": summary,
            "table": table,
            "preprocessed": preprocessed,
            "retrieval": {
                "stat_definitions": len(stat_context),
                "position_info": len(position_context),
                "analysis_guides": len(analysis_context)
            },
            "data_analysis": data_analysis if data_analysis else None
        }
    else:
        # For non-data queries or no results, return a clear message
        return {
            "summary": "No players found matching your criteria.",
            "table": None,
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