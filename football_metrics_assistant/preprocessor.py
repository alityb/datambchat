import re
from typing import Dict, Any, List
from football_metrics_assistant.data_utils import (
    get_all_teams, get_all_players, get_all_positions, get_all_leagues,
    get_alias_to_column_map, normalize_colname
)

def fuzzy_find(query: str, candidates: List[str], threshold: float = 0.8) -> List[str]:
    """
    Returns a list of candidates that are close matches to the query (case-insensitive, normalized).
    Uses simple substring and ratio matching for now.
    """
    import difflib
    norm_query = normalize_colname(query)
    matches = []
    for cand in candidates:
        norm_cand = normalize_colname(cand)
        if norm_query in norm_cand or norm_cand in norm_query:
            matches.append(cand)
        else:
            ratio = difflib.SequenceMatcher(None, norm_query, norm_cand).ratio()
            if ratio >= threshold:
                matches.append(cand)
    return matches

def extract_stats(query: str) -> List[str]:
    """
    More precise stat extraction with scoring and filtering.
    """
    alias_map = get_alias_to_column_map()
    lowered = query.lower()
    
    # Common stat patterns to look for
    stat_patterns = [
        r'poss\+/-?',  # Poss+/-
        r'\bxg\b',         # xG
        r'\bxa\b',         # xA
        r'\bnpxg\b',       # npxG
        r'goals\s*\+\s*assists',  # Goals + Assists
        r'exit\s*line',  # Exit Line
        r'clean\s*sheets',  # Clean sheets
        r'passes?\s*per\s*90',  # Passes per 90
        r'shots?\s*per\s*90',   # Shots per 90
        r'assists?\s*per\s*90', # Assists per 90
        r'goals?\s*per\s*90',   # Goals per 90
    ]
    
    found_stats = []
    for pattern in stat_patterns:
        matches = re.findall(pattern, lowered)
        for match in matches:
            # Try to find the best matching column
            best_match = None
            best_score = 0
            
            for alias, column in alias_map.items():
                norm_alias = normalize_colname(alias)
                norm_match = normalize_colname(match)
                
                # Exact match gets highest score
                if norm_match == norm_alias:
                    score = 1.0
                # Substring match gets medium score
                elif norm_match in norm_alias or norm_alias in norm_match:
                    score = 0.8
                # Fuzzy match gets lower score
                else:
                    import difflib
                    score = difflib.SequenceMatcher(None, norm_match, norm_alias).ratio()
                
                if score > best_score and score > 0.7:  # Stricter threshold
                    best_score = score
                    best_match = column
            
            if best_match and best_match not in found_stats:
                found_stats.append(best_match)
    
    return found_stats

def preprocess_query(query: str) -> Dict[str, Any]:
    """
    Preprocesses the user query to extract structured hints for downstream logic.
    Uses real data for teams, players, positions, leagues, and stat aliases.
    """
    result = {"original": query}
    lowered = query.lower()

    # 1. Top N / Best N
    top_match = re.search(r"(?:top|best)\s*(\d+)", lowered)
    if top_match:
        result["top_n"] = int(top_match.group(1))

    # 2. Age filter extraction
    age_match = re.search(r"under\s*(\d+)", lowered)
    if age_match:
        result["age_filter"] = {"op": "<", "value": int(age_match.group(1))}
    else:
        age_match = re.search(r"over\s*(\d+)", lowered)
        if age_match:
            result["age_filter"] = {"op": ">", "value": int(age_match.group(1))}
        else:
            age_match = re.search(r"age\s*(\d+)", lowered)
            if age_match:
                result["age_filter"] = {"op": "==", "value": int(age_match.group(1))}

    # 3. Season/timeframe extraction
    season_match = re.search(r"(\d{4}/\d{2}|this season|last season)", lowered)
    if season_match:
        result["season"] = season_match.group(1)

    # 4. Stat/metric extraction (improved precision)
    found_stats = extract_stats(query)
    if found_stats:
        result["stat"] = found_stats[0] if len(found_stats) == 1 else found_stats

    # 5. Team extraction (using fuzzy matching)
    teams = get_all_teams()
    found_teams = fuzzy_find(query, teams, threshold=0.8)
    if found_teams:
        result["team"] = found_teams[0] if len(found_teams) == 1 else found_teams

    # 6. Player extraction (using fuzzy matching)
    players = get_all_players()
    found_players = fuzzy_find(query, players, threshold=0.8)
    if found_players:
        result["player"] = found_players[0] if len(found_players) == 1 else found_players

    # 7. Position extraction (using fuzzy matching)
    positions = get_all_positions()
    found_positions = fuzzy_find(query, positions, threshold=0.8)
    if found_positions:
        result["position"] = found_positions[0] if len(found_positions) == 1 else found_positions

    # 8. League extraction (using fuzzy matching)
    leagues = get_all_leagues()
    found_leagues = fuzzy_find(query, leagues, threshold=0.8)
    if found_leagues:
        result["league"] = found_leagues[0] if len(found_leagues) == 1 else found_leagues

    return result 