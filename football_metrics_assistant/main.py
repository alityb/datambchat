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
        
        query_type = preprocessed.get('query_type', '')
        
        # Handle COUNT queries specially
        if query_type == 'COUNT':
            count = data_analysis.get('count', 0)
            applied_filters = data_analysis.get('applied_filters', [])
            
            # Build a descriptive summary
            filter_descriptions = []
            
            # League info
            league = preprocessed.get('league')
            if league:
                if isinstance(league, list):
                    if len(league) > 1:
                        filter_descriptions.append(f"in {', '.join(league)}")
                    else:
                        filter_descriptions.append(f"in {league[0]}")
                else:
                    filter_descriptions.append(f"in {league}")
            
            # Age filter info
            age_filter = preprocessed.get('age_filter')
            if age_filter:
                op = age_filter['op']
                value = age_filter['value']
                if op == '<':
                    filter_descriptions.append(f"under {value} years old")
                elif op == '>':
                    filter_descriptions.append(f"over {value} years old")
                elif op == '==':
                    filter_descriptions.append(f"aged {value}")
            
            # Minutes filter info
            minutes_value = preprocessed.get('minutes_value')
            minutes_op = preprocessed.get('minutes_op')
            if minutes_value is not None:
                if minutes_op == '>=':
                    filter_descriptions.append(f"with {minutes_value}+ minutes")
                elif minutes_op == '>':
                    filter_descriptions.append(f"with more than {minutes_value} minutes")
                elif minutes_op == '<':
                    filter_descriptions.append(f"with less than {minutes_value} minutes")
                elif minutes_op == '<=':
                    filter_descriptions.append(f"with at most {minutes_value} minutes")
            
            # Stat value filter info
            stat = preprocessed.get('stat')
            stat_value = preprocessed.get('stat_value')
            stat_op = preprocessed.get('stat_op')
            if stat and stat_value is not None:
                op_text = {
                    '>=': 'at least',
                    '>': 'more than',
                    '<=': 'at most',
                    '<': 'less than',
                    '==': 'exactly'
                }.get(stat_op, stat_op)
                filter_descriptions.append(f"with {op_text} {stat_value} {stat}")
            
            # Position filter
            position = preprocessed.get('position')
            if position:
                if isinstance(position, list):
                    filter_descriptions.append(f"playing as {', '.join(position)}")
                else:
                    filter_descriptions.append(f"playing as {position}")
            
            # Team filter
            team = preprocessed.get('team')
            if team:
                filter_descriptions.append(f"from {team}")
            
            # Build the final summary
            if filter_descriptions:
                filters_text = ' '.join(filter_descriptions)
                summary = f"There are {count} players {filters_text}."
            else:
                summary = f"There are {count} players total."
            
            return {
                "summary": summary,
                "table": None,  # No table needed for count queries
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