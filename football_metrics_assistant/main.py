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
        # Multi-league summary logic
        if data_analysis.get('multi_league', False):
            leagues = data_analysis.get('top_players_by_league', {})
            all_players = []
            stat_col = None
            summary_by_league = data_analysis.get('summary_by_league', {})
            # Collect all players from all leagues
            for league, players in leagues.items():
                if players:
                    if not stat_col and summary_by_league and league in summary_by_league:
                        stat_col = summary_by_league[league].get('stat')
                    for player in players:
                        player = dict(player)  # Copy to avoid mutating original
                        player['League'] = league
                        all_players.append(player)
            # Fallback to stat in data_analysis if not found
            if not stat_col and 'stat' in data_analysis:
                stat_col = data_analysis['stat']
            # Sort all players by stat_col descending
            all_players = [p for p in all_players if stat_col in p and p[stat_col] is not None]
            all_players.sort(key=lambda p: p.get(stat_col, float('-inf')), reverse=True)
            # Build combined table (top 5)
            table = None
            if all_players:
                table = f"| Rank | Player | Team | League | Position | Value |\n"
                table += "|------|--------|------|--------|----------|-------|\n"
                for i, player in enumerate(all_players[:5], 1):
                    player_name = player['Player']
                    team = player['Team within selected timeframe']
                    league = player.get('League', 'N/A')
                    position = player.get('Position', 'N/A')
                    stat_value = player.get(stat_col, 'N/A')
                    table += f"| {i} | {player_name} | {team} | {league} | {position} | {stat_value} |\n"
                table += f"\nData based on {len(all_players)} players across all selected leagues."
            # Build summary for overall top player
            if all_players:
                top_player = all_players[0]
                player_name = top_player['Player']
                team = top_player['Team within selected timeframe']
                league = top_player.get('League', 'N/A')
                stat_value = top_player.get(stat_col, 'N/A')
                summary = f"{player_name} ({team}, {league}) had the highest {stat_col} across all selected leagues with {stat_value}."
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