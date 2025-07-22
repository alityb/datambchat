import re
from typing import Dict, Any, List
from football_metrics_assistant.data_utils import (
    get_all_teams, get_all_players, get_all_positions, get_all_leagues,
    get_alias_to_column_map, normalize_colname
)
from spellchecker import SpellChecker
import phonetics
from football_metrics_assistant.llm_interface import classify_query_type

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
    corrected = ' '.join([spell.correction(w) or w for w in norm_phrase.split()])
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
    alias_map = get_alias_to_column_map()
    stat_phrase = extract_stat_phrase(query)
    # Robust correction pipeline for stat phrase
    all_aliases = list(alias_map.keys())
    corrected_stat = robust_correction(stat_phrase, all_aliases)
    from difflib import get_close_matches
    matches = get_close_matches(corrected_stat, all_aliases, n=3, cutoff=0.6)
    print(f"[DEBUG] Fuzzy stat matches for phrase '{corrected_stat}': {matches}")
    if matches:
        chosen_stats = [alias_map[m] for m in matches]
        print(f"[DEBUG] Chosen stat columns from fuzzy matches: {chosen_stats}")
        return chosen_stats
    # Substring and lowercased matching
    norm_corrected = corrected_stat.lower().replace(' ', '')
    substring_matches = [col for alias, col in alias_map.items() if norm_corrected in alias.lower().replace(' ', '')]
    if substring_matches:
        print(f"[DEBUG] Substring stat matches for phrase '{corrected_stat}': {substring_matches}")
        return substring_matches
    # Fallback: try default mapping for generic terms
    for alias, col in alias_map.items():
        if alias in corrected_stat:
            print(f"[DEBUG] Fallback: matched alias '{alias}' in stat phrase '{corrected_stat}' to column '{col}'")
            return [col]
    print(f"[DEBUG] No stat match found for phrase '{corrected_stat}'")
    return []

def extract_stat_value_filter(query: str):
    # Pattern: 'at least' or 'atleast'
    match = re.search(r"at ?least ([0-9.]+) ([\w\s+/-]+)", query.lower())
    if match:
        value = float(match.group(1))
        stat_phrase = match.group(2).strip()
        return {"stat": stat_phrase, "stat_op": ">=", "stat_value": value}
    # Pattern: 'more than 10 assists'
    match = re.search(r"more than ([0-9.]+) ([\w\s+/-]+)", query.lower())
    if match:
        value = float(match.group(1))
        stat_phrase = match.group(2).strip()
        return {"stat": stat_phrase, "stat_op": ">", "stat_value": value}
    # Pattern: 'less than 5 clean sheets'
    match = re.search(r"less than ([0-9.]+) ([\w\s+/-]+)", query.lower())
    if match:
        value = float(match.group(1))
        stat_phrase = match.group(2).strip()
        return {"stat": stat_phrase, "stat_op": "<", "stat_value": value}
    # Pattern: 'at most' or 'atmost'
    match = re.search(r"at ?most ([0-9.]+) ([\w\s+/-]+)", query.lower())
    if match:
        value = float(match.group(1))
        stat_phrase = match.group(2).strip()
        return {"stat": stat_phrase, "stat_op": "<=", "stat_value": value}
    # Pattern: 'exactly 2 goals'
    match = re.search(r"exactly ([0-9.]+) ([\w\s+/-]+)", query.lower())
    if match:
        value = float(match.group(1))
        stat_phrase = match.group(2).strip()
        return {"stat": stat_phrase, "stat_op": "==", "stat_value": value}
    return None

def prioritize_leagues(matched_leagues):
    priority = {name: i for i, name in enumerate(LEAGUE_PRIORITY)}
    return sorted(
        matched_leagues,
        key=lambda x: priority.get(x, len(priority))
    )

def preprocess_query(query: str) -> Dict[str, Any]:
    result = {"original": query}
    lowered = query.lower()

    # LLM-powered query type classification
    query_type = classify_query_type(query)
    query_type = query_type.strip().replace(':', '').upper()
    result["query_type"] = query_type

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

    # 4. Stat/metric extraction (only for TOP_N or FILTER)
    if query_type in ("TOP_N", "FILTER"):
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

    # 8. League extraction (robust pipeline)
    leagues = get_all_leagues()
    league_keywords = ['in', 'from', 'of', 'league', 'premier', 'bundesliga', 'laliga', 'serie', 'ligue', 'eredivisie', 'championship']
    found_leagues = []
    for keyword in league_keywords:
        if keyword in lowered:
            parts = lowered.split(keyword)
            if len(parts) > 1:
                potential_league = parts[1].strip()
                potential_league = re.sub(r'\b(top|best|players?|by|and|or|the)\b', '', potential_league).strip()
                if potential_league:
                    corrected_league = robust_correction(potential_league, leagues)
                    found_leagues = fuzzy_find(corrected_league, leagues, threshold=0.6)
                    print(f"[DEBUG] League extraction: keyword='{keyword}', potential_league='{potential_league}', corrected='{corrected_league}', matches={found_leagues}")
                    if found_leagues:
                        break
    if not found_leagues:
        corrected_league = robust_correction(query, leagues)
        found_leagues = fuzzy_find(corrected_league, leagues, threshold=0.6)
        print(f"[DEBUG] League extraction: fallback to whole query, corrected='{corrected_league}', matches={found_leagues}")
    if found_leagues:
        found_leagues = prioritize_leagues(found_leagues)
        result["league"] = found_leagues[0]  # Always use the top-priority league only
    else:
        result["league"] = None

    # Stat value filter extraction
    stat_value_filter = extract_stat_value_filter(query)
    if stat_value_filter:
        # Use stat aliasing for the stat phrase
        stat_aliases = get_alias_to_column_map()
        stat_phrase = stat_value_filter["stat"]
        # Fuzzy match stat phrase to column
        from difflib import get_close_matches
        all_aliases = list(stat_aliases.keys())
        match = get_close_matches(stat_phrase, all_aliases, n=1, cutoff=0.6)
        if match:
            stat_col = stat_aliases[match[0]]
            result["stat"] = stat_col
            result["stat_op"] = stat_value_filter["stat_op"]
            result["stat_value"] = stat_value_filter["stat_value"]
        else:
            # If no match, just use the phrase
            result["stat"] = stat_phrase
            result["stat_op"] = stat_value_filter["stat_op"]
            result["stat_value"] = stat_value_filter["stat_value"]

    # LLM fallback for stat/league extraction if missing or ambiguous
    ambiguous_league = not result.get("league") or (isinstance(result.get("league"), list) and len(result["league"]) > 3)
    ambiguous_stat = not result.get("stat") or (isinstance(result.get("stat"), list) and len(result["stat"]) > 3)
    if ambiguous_league or ambiguous_stat:
        from football_metrics_assistant.llm_interface import ask_llama
        llm_prompt = f"Extract the main stat and league(s) from this football query.\nQuery: '{query}'\nRespond in JSON: {{'stat': ..., 'league': ...}}. Only include fields you are confident about."
        try:
            llm_response = ask_llama(llm_prompt)
            import json
            llm_json = json.loads(llm_response)
            alias_map = get_alias_to_column_map()
            # Normalize stat
            if ambiguous_stat and llm_json.get('stat'):
                llm_stat = llm_json['stat']
                # Try direct, substring, and lowercased match
                stat_col = None
                if llm_stat in alias_map:
                    stat_col = alias_map[llm_stat]
                else:
                    norm_llm_stat = llm_stat.lower().replace(' ', '')
                    for alias, col in alias_map.items():
                        if norm_llm_stat in alias.lower().replace(' ', ''):
                            stat_col = col
                            break
                if stat_col:
                    result['stat'] = stat_col
                else:
                    result['stat'] = llm_stat
            # Normalize league
            if ambiguous_league and llm_json.get('league'):
                leagues = get_all_leagues()
                llm_league = llm_json['league']
                if isinstance(llm_league, list):
                    league_matches = [l for l in leagues if l in llm_league or l.lower() in [x.lower() for x in llm_league]]
                    result['league'] = league_matches if league_matches else llm_league
                else:
                    league_match = next((l for l in leagues if l.lower() == llm_league.lower()), None)
                    result['league'] = league_match if league_match else llm_league
            print(f"[DEBUG] LLM fallback extraction: {llm_json}")
        except Exception as e:
            print(f"[DEBUG] LLM fallback extraction failed: {e}")

    print(f"[DEBUG] Preprocessed query result: {result}")
    return result 