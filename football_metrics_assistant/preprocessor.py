import re
from typing import Dict, Any, List
from football_metrics_assistant.data_utils import (
    get_all_teams, get_all_players, get_all_positions, get_all_leagues,
    get_alias_to_column_map, normalize_colname
)
from spellchecker import SpellChecker
import phonetics

LEAGUE_PRIORITY = [
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
    "Eredivisie",
    "Championship",
    "Brazil Serie A",
    "Serie B",
    # ...add more as needed
]

def correction_dict():
    # Add more as you see common typos
    return {
        'serei a': 'Serie A',
        'premier legaue': 'Premier League',
        'ligue 1': 'Ligue 1',
        'bundesliga': 'Bundesliga',
        'laliga': 'La Liga',
        'champions leage': 'Champions League',
        'clean sheats': 'Clean sheets',
        'assits': 'Assists per 90',
        'goals+assists': 'Goals + Assists per 90',
        # ...add more as needed
    }

def robust_correction(phrase: str, candidates: List[str]) -> str:
    corrections = correction_dict()
    norm_phrase = phrase.lower().strip()
    print(f"[DEBUG] Robust correction input: '{phrase}' (normalized: '{norm_phrase}')")
    # 1. Correction dict (substring match)
    for typo, correct in corrections.items():
        if typo in norm_phrase:
            print(f"[DEBUG] Correction dict: '{norm_phrase}' contains '{typo}' -> '{correct}'")
            return correct
    # 2. Spellchecker
    spell = SpellChecker()
    corrected = ' '.join([spell.correction(w) for w in norm_phrase.split()])
    if corrected != norm_phrase:
        print(f"[DEBUG] Spellchecker: '{norm_phrase}' -> '{corrected}'")
        norm_phrase = corrected
    # 3. Phonetic matching
    input_phon = phonetics.metaphone(norm_phrase)
    for cand in candidates:
        if phonetics.metaphone(cand.lower()) == input_phon:
            print(f"[DEBUG] Phonetic match: '{norm_phrase}' -> '{cand}'")
            return cand
    # 4. Fuzzy matching
    from difflib import get_close_matches
    match = get_close_matches(norm_phrase, candidates, n=1, cutoff=0.6)
    if match:
        print(f"[DEBUG] Fuzzy match: '{norm_phrase}' -> '{match[0]}'")
        return match[0]
    print(f"[DEBUG] No robust match found for '{phrase}', returning as is.")
    return phrase

def fuzzy_find(query: str, candidates: List[str], threshold: float = 0.8) -> List[str]:
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

def extract_stat_phrase(query: str) -> str:
    # Always use the phrase after the last 'by' for stat extraction
    lowered = query.lower()
    if ' by ' in lowered:
        stat_phrase = lowered.split(' by ')[-1].strip()
        print(f"[DEBUG] Extracted stat phrase after last 'by': '{stat_phrase}'")
        return stat_phrase
    # Fallback: after last 'in' or 'for'
    for kw in [' in ', ' for ']:
        if kw in lowered:
            stat_phrase = lowered.split(kw)[-1].strip()
            print(f"[DEBUG] Extracted stat phrase after last '{kw.strip()}': '{stat_phrase}'")
            return stat_phrase
    print(f"[DEBUG] No explicit stat phrase found, using full query.")
    return query

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
        # Add defensive stats
        r'sliding\s*tackles?\s*per\s*90',  # Sliding tackles per 90
        r'tackles?\s*per\s*90',  # Tackles per 90
        r'interceptions?\s*per\s*90',  # Interceptions per 90
        r'clearances?\s*per\s*90',  # Clearances per 90
        r'blocks?\s*per\s*90',  # Blocks per 90
        r'aerial\s*duels?\s*per\s*90',  # Aerial duels per 90
        r'defensive\s*duels?\s*per\s*90',  # Defensive duels per 90
        r'saves?\s*per\s*90',  # Saves per 90
        r'crosses?\s*per\s*90',  # Crosses per 90
        r'dribbles?\s*per\s*90',  # Dribbles per 90
        r'progressive\s*actions?\s*per\s*90',  # Progressive actions per 90
        r'progressive\s*passes?\s*per\s*90',  # Progressive passes per 90
        r'progressive\s*carries?\s*per\s*90',  # Progressive carries per 90
        # Add percentage stats
        r'pass\s*completion\s*%',  # Pass completion %
        r'shot\s*accuracy\s*%',  # Shot accuracy %
        r'tackle\s*success\s*%',  # Tackle success %
        r'aerial\s*duel\s*success\s*%',  # Aerial duel success %
        # Add per 100 stats
        r'xg\s*per\s*100\s*touches',  # xG per 100 touches
        r'xa\s*per\s*100\s*passes',  # xA per 100 passes
        r'goals?\s*per\s*100\s*touches',  # Goals per 100 touches
        r'assists?\s*per\s*100\s*passes',  # Assists per 100 passes
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
    
    # Fallback: if no stats found by patterns, try direct alias matching
    if not found_stats:
        # Look for any stat name in the query that matches an alias
        query_words = lowered.split()
        for word in query_words:
            # Skip common words that aren't stats
            skip_words = {'top', 'best', 'players', 'by', 'in', 'from', 'of', 'league', 'under', 'over', 'age', 'defenders', 'midfielders', 'forwards', 'strikers', 'goalkeepers'}
            if word in skip_words:
                continue
            
            # Try to match with aliases
            for alias, column in alias_map.items():
                norm_alias = normalize_colname(alias)
                norm_word = normalize_colname(word)
                
                if norm_word == norm_alias or norm_word in norm_alias or norm_alias in norm_word:
                    if column not in found_stats:
                        found_stats.append(column)
                        break
    
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
    # Add robust mapping for common football position groups
    lowered_query = query.lower()
    if 'defender' in lowered_query:
        found_positions = ['Centre-back', 'Full-back']
    elif 'midfielder' in lowered_query:
        found_positions = ['Midfielder', 'Winger']
    elif 'forward' in lowered_query or 'striker' in lowered_query:
        found_positions = ['Striker', 'Winger']
    if found_positions:
        result["position"] = found_positions[0] if len(found_positions) == 1 else found_positions

    # 8. League extraction (using fuzzy matching)
    leagues = get_all_leagues()
    
    # Look for league keywords in the query
    league_keywords = ['in', 'from', 'of', 'league', 'premier', 'bundesliga', 'laliga', 'serie', 'ligue', 'eredivisie', 'championship']
    found_leagues = []
    
    for keyword in league_keywords:
        if keyword in lowered:
            # Extract the part after the keyword
            parts = lowered.split(keyword)
            if len(parts) > 1:
                potential_league = parts[1].strip()
                # Clean up the potential league name
                potential_league = re.sub(r'\b(top|best|players?|by|and|or|the)\b', '', potential_league).strip()
                if potential_league:
                    found_leagues = fuzzy_find(potential_league, leagues, threshold=0.85)
                    if found_leagues:
                        break
    
    # If no league found with keywords, try the whole query
    if not found_leagues:
        found_leagues = fuzzy_find(query, leagues, threshold=0.85)
    
    if found_leagues:
        result["league"] = found_leagues[0] if len(found_leagues) == 1 else found_leagues
    else:
        result["league"] = None # Indicate no league found

    return result 