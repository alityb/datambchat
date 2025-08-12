from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
from football_metrics_assistant.preprocessor import preprocess_query
from football_metrics_assistant.retriever import HybridRetriever
from football_metrics_assistant.llm_interface import ask_gemini
from football_metrics_assistant.tools import analyze_query, generate_player_report  # Add generate_player_report
import time
import numpy as np

app = FastAPI()

# Initialize retriever (stub)
retriever = HybridRetriever()

class ChatRequest(BaseModel):
    message: str
    history: list = []

def to_python_type(val):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
    return val

def clean_dict_for_json(obj):
    """Recursively clean a dict/list structure to ensure JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_dict_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict_for_json(v) for v in obj]
    else:
        return to_python_type(obj)

def format_stat_definition_response(definition_data: dict) -> dict:
    """Format stat definition data for the frontend."""
    
    # Clean the definition data first
    definition_data = clean_dict_for_json(definition_data)
    
    stat_name = definition_data['stat_name']
    definition = definition_data['definition']
    comparison_text = definition_data['comparison_text']
    top_players = definition_data['top_players']
    basic_stats = definition_data['basic_stats']
    position_insights = definition_data['position_insights']
    
    # Build comprehensive summary
    summary_parts = [
        f"**{stat_name}**\n",
        f"{definition}\n",
        f"**Dataset Overview:**",
        f"• Total players analyzed: {basic_stats['total_players']}",
        f"• Average value: {basic_stats['mean']:.2f}",
        f"• Range: {basic_stats['min']:.2f} to {basic_stats['max']:.2f}\n"
    ]
    
    # Position insights
    if position_insights:
        summary_parts.append("**Best Positions for this Stat:**")
        for i, pos_data in enumerate(position_insights, 1):
            summary_parts.append(f"{i}. {pos_data['position']}: {pos_data['average']:.2f} average ({pos_data['player_count']} players)")
        summary_parts.append("")
    
    # Top performers
    if top_players:
        summary_parts.append(f"**Top performers ({comparison_text} values):**")
        for i, player in enumerate(top_players[:3], 1):
            summary_parts.append(f"{i}. {player['name']} ({player['team']}) - {player['stat_value']:.2f}")
        summary_parts.append("")
        
    # Build detailed table
    table_rows = ["| Rank | Player | Team | League | Position | Age | Minutes | Value |"]
    table_rows.append("|------|--------|------|--------|----------|-----|---------|-------|")
    
    for i, player in enumerate(top_players, 1):
        name = player['name']
        team = player['team']
        league = player['league']
        position = player['position']
        age = player['age']
        minutes = player['minutes']
        value = f"{player['stat_value']:.2f}" if isinstance(player['stat_value'], (int, float)) else str(player['stat_value'])
        
        table_rows.append(f"| {i} | {name} | {team} | {league} | {position} | {age} | {minutes} | {value} |")
    
    return {
        "summary": "\n".join(summary_parts),
        "table": "\n".join(table_rows),
        "preprocessed": {"query_type": "STAT_DEFINITION", "stat": stat_name},
        "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
        "data_analysis": definition_data
    }

def generate_summary(data_analysis, preprocessed, filtered_data, stat_col):
    """Generate a proper summary without duplicate stat mentions."""
    
    if not data_analysis or not data_analysis.get('success'):
        return "No players found matching your criteria."
    
    count = data_analysis.get('count', 0)
    applied_filters = data_analysis.get('applied_filters', [])
    
    # FIXED: Clean up duplicate stat filters in description
    filter_descriptions = []
    stat_filter_added = False
    
    for f in applied_filters:
        if not f:  # Skip None or empty filters
            continue
        if f.startswith("Position:"):
            pos = f.replace("Position: ", "")
            filter_descriptions.append(pos.lower() + "s")
        elif f.startswith("League:"):
            league = f.replace("League: ", "")
            filter_descriptions.append(f"in {league}")
        elif "per 90 >=" in f and not stat_filter_added:
            # Only add the first stat filter to avoid duplication
            parts = f.split(" >= ")
            if len(parts) == 2:
                stat_name = parts[0]
                value = parts[1]
                filter_descriptions.append(f"with at least {value} {stat_name.lower()}")
                stat_filter_added = True
        elif f.startswith("Minutes played") and ">=" in f:
            parts = f.split(" >= ")
            if len(parts) == 2:
                value = parts[1]
                filter_descriptions.append(f"with {value}+ minutes")
    
    base_desc = " ".join(filter_descriptions) if filter_descriptions else ""
    
    # Get top performer if available
    if filtered_data and stat_col:
        try:
            top_player = filtered_data[0]  # First in list (highest value)
            player_name = top_player.get('Player', 'Unknown Player')
            team = top_player.get('Team within selected timeframe', 'Unknown Team')
            stat_value = top_player.get(stat_col, 'N/A')
            
            if base_desc:
                summary = f"Found {count} {base_desc}. Top: {player_name} ({team}) with {stat_value}."
            else:
                summary = f"Top {stat_col}: {player_name} ({team}) with {stat_value}."
        except Exception as e:
            print(f"[ERROR] Summary generation failed: {e}")
            summary = f"Found {count} players {base_desc}." if base_desc else f"Found {count} players."
    else:
        summary = f"Found {count} players {base_desc}." if base_desc else f"Found {count} players."
    
    return summary

def format_player_report_response(report_data: dict) -> dict:
    """Format player report data for the frontend."""
    
    # Clean the report data first
    report_data = clean_dict_for_json(report_data)
    
    basic = report_data['basic_info']
    league_comp = report_data['league_comparison']
    
    # Build comprehensive summary
    summary_parts = [
        f"**{basic['name']}** - {basic['position']} at {basic['team']} ({basic['league']})",
        f"Age: {basic['age']} | Minutes: {basic['minutes_played']} | Matches: {basic['matches_played']}\n"
    ]
    
    # League comparison
    summary_parts.append(f"**League Performance ({basic['league']}):**")
    summary_parts.append(f"Compared to {league_comp['position_peers']} {basic['position']}s in {basic['league']}\n")
    
    # Top 5 leagues comparison if available
    if report_data.get('top5_comparison'):
        top5_comp = report_data['top5_comparison']
        summary_parts.append(f"**Top 5 Leagues Performance:**")
        summary_parts.append(f"Compared to {top5_comp['total_players']} {basic['position']}s across {', '.join(top5_comp['leagues'])}\n")
    
    # Strengths
    if report_data['strengths']:
        summary_parts.append("**Key Strengths:**")
        for strength in report_data['strengths'][:3]:
            summary_parts.append(f"• {strength['stat']}: {strength['description']} ({strength['percentile']}th percentile)")
        summary_parts.append("")
    
    # Weaknesses
    if report_data['weaknesses']:
        summary_parts.append("**Areas for Improvement:**")
        for weakness in report_data['weaknesses'][:3]:
            summary_parts.append(f"• {weakness['stat']}: {weakness['description']} ({weakness['percentile']}th percentile)")
        summary_parts.append("")
    
    # Similar players
    if report_data['similar_players']:
        summary_parts.append("**Similar Players:**")
        for player in report_data['similar_players'][:3]:
            summary_parts.append(f"• {player['name']} ({player['team']}) - Age {player['age']}")
    
    # Build detailed table
    table_rows = ["| Stat | Player Value | League Percentile | T5 Percentile |"]
    table_rows.append("|------|--------------|------------------|---------------|")
    
    for stat in report_data['key_stats_analyzed']:
        player_value = report_data['player_stats'].get(stat, 'N/A')
        league_pct = report_data['league_comparison']['percentiles'].get(stat, 'N/A')
        top5_pct = report_data.get('top5_percentiles', {}).get(stat, 'N/A')
        
        # Format values
        if isinstance(player_value, (int, float)) and player_value != 'N/A':
            player_value = f"{player_value:.2f}"
        if isinstance(league_pct, (int, float)) and league_pct != 'N/A':
            league_pct = f"{league_pct}%"
        if isinstance(top5_pct, (int, float)) and top5_pct != 'N/A':
            top5_pct = f"{top5_pct}%"
            
        table_rows.append(f"| {stat} | {player_value} | {league_pct} | {top5_pct} |")
    
    return {
        "summary": "\n".join(summary_parts),
        "table": "\n".join(table_rows),
        "preprocessed": {"query_type": "PLAYER_REPORT", "player": basic['name']},
        "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
        "data_analysis": report_data
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # 1. Preprocess the query for structured hints
        preprocessed = preprocess_query(req.message)

        # Handle PLAYER_REPORT queries early
        if preprocessed.get('query_type') == 'PLAYER_REPORT':
            player_name = preprocessed.get('player')
            if not player_name:
                return {
                    "summary": "No player specified for report. Please specify a player name.",
                    "table": None,
                    "preprocessed": preprocessed,
                    "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
                    "data_analysis": {"error": "No player specified"}
                }
            
            try:
                report_data = generate_player_report(player_name)
                if report_data.get('success'):
                    return format_player_report_response(report_data)
                else:
                    return {
                        "summary": report_data.get('error', 'Failed to generate player report'),
                        "table": None,
                        "preprocessed": preprocessed,
                        "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
                        "data_analysis": report_data
                    }
            except Exception as e:
                print(f"[ERROR] Player report failed: {str(e)}")
                return {
                    "summary": f"Error generating player report: {str(e)}",
                    "table": None,
                    "preprocessed": preprocessed,
                    "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
                    "data_analysis": {"error": str(e)}
                }

        # Handle STAT_DEFINITION queries early
        if preprocessed.get('query_type') == 'STAT_DEFINITION':
            stat_name = preprocessed.get('stat')
            if not stat_name:
                return {
                    "summary": "No statistic specified for definition. Please specify a statistic name.",
                    "table": None,
                    "preprocessed": preprocessed,
                    "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
                    "data_analysis": {"error": "No statistic specified"}
                }
            
            try:
                from football_metrics_assistant.tools import generate_stat_definition_report
                definition_data = generate_stat_definition_report(stat_name)
                if definition_data.get('success'):
                    return format_stat_definition_response(definition_data)
                else:
                    error_msg = definition_data.get('error', 'Failed to generate stat definition')
                    suggestions = definition_data.get('suggestions', [])
                    if suggestions:
                        error_msg += f"\n\nDid you mean: {', '.join(suggestions)}?"
                    return {
                        "summary": error_msg,
                        "table": None,
                        "preprocessed": preprocessed,
                        "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
                        "data_analysis": definition_data
                    }
            except Exception as e:
                print(f"[ERROR] Stat definition failed: {str(e)}")
                return {
                    "summary": f"Error generating stat definition: {str(e)}",
                    "table": None,
                    "preprocessed": preprocessed,
                    "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
                    "data_analysis": {"error": str(e)}
                }

        # 2. Retrieve stat definitions/context using preprocessed hints
        try:
            retrieval = retriever.retrieve(req.message, preprocessed_hints=preprocessed)
            stat_context = retrieval.get("stat_definitions", [])
            position_context = retrieval.get("position_info", [])
            analysis_context = retrieval.get("analysis_guides", [])
        except Exception as e:
            print(f"[ERROR] Retrieval failed: {str(e)}")
            stat_context = []
            position_context = []
            analysis_context = []

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
            or preprocessed.get('minutes_op')
            or preprocessed.get('stat_op')
        ):
            try:
                data_analysis = analyze_query(preprocessed)
            except Exception as e:
                print(f"[ERROR] Data analysis failed: {str(e)}")
                data_analysis = {"error": f"Data analysis failed: {str(e)}"}
            
    except Exception as e:
        print(f"[ERROR] Chat endpoint failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "summary": f"An error occurred while processing your query: {str(e)}",
            "table": None,
            "preprocessed": {"error": str(e)},
            "retrieval": {"stat_definitions": 0, "position_info": 0, "analysis_guides": 0},
            "data_analysis": {"error": str(e)}
        }

    
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
                "data_analysis": clean_dict_for_json(data_analysis)
            }
        
        # Initialize variables for the main processing
        stat_col = None
        filtered_data = []
        
        # Determine the correct stat column name for table and summary
        if 'stat_formula' in preprocessed:
            # Use display_expr if present
            stat_col = preprocessed['stat_formula'].get('display_expr')
        elif 'stat' in data_analysis:
            stat_col = data_analysis['stat']
        elif 'stat' in preprocessed:
            stat_col = preprocessed['stat']
        else:
            stat_col = ''
        
        # Prefer top_players if present, else filtered_data
        if data_analysis.get('top_players'):
            filtered_data = data_analysis['top_players']
        elif data_analysis.get('filtered_data'):
            filtered_data = data_analysis['filtered_data']
        
        # SORT FIX: Ensure filtered_data is sorted by stat_col descending
        if filtered_data and stat_col:
            try:
                # Sort by stat_col in descending order (highest first)
                filtered_data = sorted(filtered_data, 
                                    key=lambda x: x.get(stat_col, float('-inf')) if isinstance(x.get(stat_col), (int, float)) else float('-inf'), 
                                    reverse=True)
                print(f"[DEBUG] After sorting, top 3:")
                for i, player in enumerate(filtered_data[:3]):
                    print(f"  {i+1}. {player['Player']}: {player.get(stat_col)}")
            except Exception as e:
                print(f"[DEBUG] Sorting failed: {e}")
        
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
        summary = generate_summary(data_analysis, preprocessed, filtered_data, stat_col)
        
        return {
            "summary": summary,
            "table": table,
            "preprocessed": preprocessed,
            "retrieval": {
                "stat_definitions": len(stat_context),
                "position_info": len(position_context),
                "analysis_guides": len(analysis_context)
            },
            "data_analysis": clean_dict_for_json(data_analysis) if data_analysis else None
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
            "data_analysis": clean_dict_for_json(data_analysis) if data_analysis else None
        }
    

@app.get("/stat-definitions")
def stat_definitions():
    # Placeholder for stat definitions endpoint 
    return {"definitions": []}