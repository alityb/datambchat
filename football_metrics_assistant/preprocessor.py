import re
from typing import Dict, Any, List
from football_metrics_assistant.data_utils import (
    get_all_teams, get_all_players, get_all_positions, get_all_leagues,
    get_alias_to_column_map, normalize_colname
)
from spellchecker import SpellChecker
import phonetics
from football_metrics_assistant.llm_interface import classify_query_type
from difflib import get_close_matches, SequenceMatcher

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
    # If a math operator is present, extract the phrase containing it (before ' in ' or ' for ')
    lowered = query.lower()
    if any(op in lowered for op in ['+', '-', '/', '*']):
        # Find the first ' in ' or ' for ' after the operator, and take everything before it
        for kw in [' in ', ' for ']:
            idx = lowered.find(kw)
            if idx != -1:
                # Only use if the operator is before the keyword
                op_idx = max(lowered.find(op) for op in ['+', '-', '/', '*'])
                if op_idx != -1 and op_idx < idx:
                    stat_phrase = query[:idx].strip()
                    print(f"[DEBUG] Extracted stat formula phrase before '{kw.strip()}': '{stat_phrase}'")
                    return stat_phrase
        # No 'in' or 'for' after operator, use full query
        print(f"[DEBUG] Extracted stat formula phrase (full query): '{query.strip()}'")
        return query.strip()
    # Otherwise, use the old logic
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

def extract_stat_formula(query: str) -> dict:
    """
    Detects and parses stat formulas like 'xg/xa', 'goals + assists', 'xg - xa', 'xA + xG', 'Sliding tackles per 90 + progressive passes per 90'.
    Returns a dict with the formula expression (safe for eval), mapped columns, display expr, and operators.
    """
    alias_map = get_alias_to_column_map()
    stat_phrase = extract_stat_phrase(query)
    # Only consider the part before ' in ' or ' for ' for stat formulas
    stat_phrase = re.split(r' in | for ', stat_phrase)[0].strip()
    # Look for math operators
    ops = re.findall(r'[+\-/*]', stat_phrase)
    if not ops:
        print(f"[DEBUG] No math operator found in stat phrase: '{stat_phrase}'")
        return None
    # Split by operators, map each part
    parts = re.split(r'[+\-/*]', stat_phrase)
    print(f"[DEBUG] Formula parts: {parts}, ops: {ops}")
    mapped_cols = []
    safe_vars = []
    safe_map = {}
    # Forced mapping for goals/assists (singular/plural)
    forced_map = {
        'goals': 'Goals per 90',
        'goal': 'Goals per 90',
        'goals per 90': 'Goals per 90',
        'goal per 90': 'Goals per 90',
        'assists': 'Assists per 90',
        'assist': 'Assists per 90',
        'assists per 90': 'Assists per 90',
        'assist per 90': 'Assists per 90',
        'xg': 'xG per 90',
        'xa': 'xA per 90',
        'npxg': 'npxG per 90',
    }
    prefixes = ["most ", "top ", "highest ", "best "]
    for part in parts:
        part = part.strip()
        part_lc = part.lower().strip()
        # Remove common prefixes
        orig_part_lc = part_lc
        for prefix in prefixes:
            if part_lc.startswith(prefix):
                part_lc = part_lc[len(prefix):].strip()
                print(f"[DEBUG] Removed prefix '{prefix}' from '{orig_part_lc}' -> '{part_lc}'")
                break
        # PATCH: Remove leading numbers (e.g., '10 xg' -> 'xg')
        part_lc = re.sub(r'^\d+\s*', '', part_lc)
        # Forced mapping check
        if part_lc in forced_map:
            mapped_col = forced_map[part_lc]
            print(f"[DEBUG] Formula part '{part}' normalized to '{part_lc}' forced-mapped to '{mapped_col}' (forced map)")
        elif part_lc == 'goals per 90':
            mapped_col = 'Goals per 90'
            print(f"[DEBUG] Formula part '{part}' direct-mapped to 'Goals per 90'")
        elif part_lc == 'assists per 90':
            mapped_col = 'Assists per 90'
            print(f"[DEBUG] Formula part '{part}' direct-mapped to 'Assists per 90'")
        else:
            # 1. Try exact alias match (case-insensitive)
            norm_part = part_lc.replace(' ', '').replace('-', '').replace('+', 'plus').replace('/', '').replace('(', '').replace(')', '')
            if norm_part in alias_map:
                mapped_col = alias_map[norm_part]
                print(f"[DEBUG] Formula part '{part}' normalized to '{part_lc}' mapped to '{mapped_col}' via exact alias match '{norm_part}'")
            else:
                # 2. Try robust_correction and fuzzy match
                all_aliases = list(alias_map.keys())
                corrected = robust_correction(part_lc, all_aliases)
                match = get_close_matches(corrected, all_aliases, n=1, cutoff=0.8)  # stricter cutoff
                if match and alias_map[match[0]].lower().startswith('x'):
                    mapped_col = alias_map[match[0]]
                    print(f"[DEBUG] Formula part '{part}' normalized to '{part_lc}' mapped to '{mapped_col}' via fuzzy match '{match[0]}'")
                else:
                    # 3. General substring match (all stats)
                    sub_matches = [col for alias, col in alias_map.items() if norm_part in alias.lower()]
                    if sub_matches:
                        mapped_col = sub_matches[0]
                        print(f"[DEBUG] Formula part '{part}' normalized to '{part_lc}' mapped to '{mapped_col}' via general substring match")
                    else:
                        # 4. LLM fallback
                        try:
                            from football_metrics_assistant.llm_interface import ask_llama
                            llm_prompt = f"Given the following stat columns: {list(set(alias_map.values()))}\nWhich column best matches the user phrase: '{part}'? Respond with the exact column name only."
                            llm_col = ask_llama(llm_prompt).strip()
                            if llm_col in alias_map.values():
                                mapped_col = llm_col
                                print(f"[DEBUG] Formula part '{part}' normalized to '{part_lc}' mapped to '{mapped_col}' via LLM fallback")
                            else:
                                mapped_col = part
                                print(f"[DEBUG] Formula part '{part}' normalized to '{part_lc}' could not be mapped, using as is (even after LLM fallback)")
                        except Exception as e:
                            mapped_col = part
                            print(f"[DEBUG] LLM fallback failed for '{part}': {e}")
        mapped_cols.append(mapped_col)
        # Build safe variable name
        safe_var = mapped_col.replace(' ', '_').replace('.', '_').replace('%', 'pct').replace('(', '').replace(')', '').replace('-', '_').replace('+', 'plus').replace('/', '_')
        safe_vars.append(safe_var)
        safe_map[safe_var] = mapped_col
    # Build safe expr for eval
    safe_expr = ''
    display_expr = ''
    for i, safe_var in enumerate(safe_vars):
        safe_expr += f'safe_map["{safe_var}"]'
        display_expr += f'({mapped_cols[i]})'
        if i < len(ops):
            safe_expr += ops[i]
            display_expr += ops[i]
    print(f"[DEBUG] Formula safe_expr: {safe_expr}, display_expr: {display_expr}, mapped_cols: {mapped_cols}")
    return {'expr': safe_expr, 'display_expr': display_expr, 'columns': mapped_cols, 'ops': ops, 'safe_map': safe_map}

def extract_stats(query: str) -> List[str]:
    alias_map = get_alias_to_column_map()
    stat_phrase = extract_stat_phrase(query)
    # Robust correction pipeline for stat phrase
    all_aliases = list(alias_map.keys())
    corrected_stat = robust_correction(stat_phrase, all_aliases)
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

def extract_minutes_filter(query: str):
    """Extract minutes played filters like '500+ minutes', 'more than 1000 minutes', etc."""
    # Pattern: '500+ minutes' or '500 + minutes'
    match = re.search(r"(\d+)\s*\+\s*minutes?", query.lower())
    if match:
        value = int(match.group(1))
        return {"minutes_op": ">=", "minutes_value": value}
    
    # Pattern: 'more than 500 minutes'
    match = re.search(r"more than (\d+) minutes?", query.lower())
    if match:
        value = int(match.group(1))
        return {"minutes_op": ">", "minutes_value": value}
    
    # Pattern: 'at least 500 minutes'
    match = re.search(r"at ?least (\d+) minutes?", query.lower())
    if match:
        value = int(match.group(1))
        return {"minutes_op": ">=", "minutes_value": value}
    
    # Pattern: 'less than 500 minutes'
    match = re.search(r"less than (\d+) minutes?", query.lower())
    if match:
        value = int(match.group(1))
        return {"minutes_op": "<", "minutes_value": value}
    
    # Pattern: 'under 500 minutes'
    match = re.search(r"under (\d+) minutes?", query.lower())
    if match:
        value = int(match.group(1))
        return {"minutes_op": "<", "minutes_value": value}
    
    # Pattern: 'over 500 minutes'
    match = re.search(r"over (\d+) minutes?", query.lower())
    if match:
        value = int(match.group(1))
        return {"minutes_op": ">", "minutes_value": value}
    
    return None


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
            else:
                # NEW: Handle U25, U-25, U21, etc. patterns
                u_age_match = re.search(r"u-?(\d+)", lowered)
                if u_age_match:
                    age_value = int(u_age_match.group(1))
                    result["age_filter"] = {"op": "<", "value": age_value}
                    print(f"[DEBUG] Detected 'U{age_value}' pattern, setting age_filter to < {age_value}")

    # Add this after the query type classification
    if re.search(r'\b(\w+(?:\s+\w+)*)\s+report\b', query.lower()):
        player_match = re.search(r'\b(\w+(?:\s+\w+)*)\s+report\b', query.lower())
        if player_match:
            result["query_type"] = "PLAYER_REPORT"
            player_name = player_match.group(1).strip()
            # Find the actual player name using fuzzy matching
            players = get_all_players()
            found_players = fuzzy_find(player_name, players, threshold=0.7)
            if found_players:
                result["player"] = found_players[0]
            else:
                result["player"] = player_name

    # 3. Season/timeframe extraction
    season_match = re.search(r"(\d{4}/\d{2}|this season|last season)", lowered)
    if season_match:
        result["season"] = season_match.group(1)

    # 4. Stat/metric extraction (only for TOP_N or FILTER)
    stat_formula = extract_stat_formula(query)
    if stat_formula:
        result["stat_formula"] = stat_formula
        result["stat"] = stat_formula['expr']
        # Force query_type to TOP_N for formula queries
        result["query_type"] = "TOP_N"
    elif query_type in ("TOP_N", "FILTER"):
        found_stats = extract_stats(query)
        if found_stats:
            result["stat"] = found_stats[0] if len(found_stats) == 1 else found_stats
    # --- PATCH: Single-stat forced mapping fallback ---
    if (query_type in ("TOP_N", "FILTER")) and not result.get("stat"):
        # Forced mapping for common stats
        forced_map = {
            'goals': 'Goals per 90',
            'goal': 'Goals per 90',
            'goals per 90': 'Goals per 90',
            'goal per 90': 'Goals per 90',
            'assists': 'Assists per 90',
            'assist': 'Assists per 90',
            'assists per 90': 'Assists per 90',
            'assist per 90': 'Assists per 90',
            'xg': 'xG per 90',
            'xa': 'xA per 90',
            'npxg': 'npxG per 90',
        }
        prefixes = ["most ", "top ", "highest ", "best "]
        lowered_query = query.lower().strip()
        for prefix in prefixes:
            if lowered_query.startswith(prefix):
                lowered_query = lowered_query[len(prefix):].strip()
                break
        # Try forced map
        for forced, mapped in forced_map.items():
            if re.search(rf"\b{re.escape(forced)}\b", lowered_query):
                result["stat"] = mapped
                print(f"[DEBUG] [PATCH] Single-stat fallback: '{forced}' mapped to '{mapped}'")
                break
        # If still not found, try alias map
        if not result.get("stat"):
            alias_map = get_alias_to_column_map()
            # PATCH: Prefer the longest matching alias
            best_alias = None
            for alias in sorted(alias_map.keys(), key=len, reverse=True):
                if re.search(rf"\b{re.escape(alias)}\\b", lowered_query):
                    best_alias = alias
                    break
            if best_alias:
                result["stat"] = alias_map[best_alias]
                print(f"[DEBUG] [PATCH] Single-stat alias fallback: '{best_alias}' mapped to '{alias_map[best_alias]}'")

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
    lowered_query = query.lower()
    # Only consider the part after ' in ' or ' for ' for league extraction
    league_part = re.split(r' in | for ', lowered_query)
    if len(league_part) > 1:
        league_str = league_part[1]
    else:
        league_str = lowered_query
    league_phrases = re.split(r'\band\b|,|&', league_str)
    found_leagues = []
    def norm_league(s):
        return s.lower().strip().replace('.', '').replace('-', ' ').replace(' ', '')
    norm_league_map = {norm_league(lg): lg for lg in leagues}
    # --- Remove age/irrelevant words from league phrases ---
    age_patterns = [r'under ?\d+', r'u-?\d+', r'over ?\d+', r'age ?\d+']
    def clean_league_phrase(phrase):
        # Remove age patterns more broadly
        age_patterns = [r'under ?\d+', r'u-?\d+', r'over ?\d+', r'age ?\d+']
        for pat in age_patterns:
            phrase = re.sub(pat, '', phrase, flags=re.IGNORECASE)
        
        # Remove other irrelevant words that might interfere
        phrase = re.sub(r'\b(players?|best|top|most|highest)\b', '', phrase, flags=re.IGNORECASE)
        
        return phrase.strip()
    
    for phrase in league_phrases:
        orig_phrase = phrase
        phrase = clean_league_phrase(phrase)
        phrase = phrase.strip().replace('.', '').replace('-', ' ')
        norm_phrase = phrase.lower().replace(' ', '')
        print(f"[DEBUG] Trying to match league phrase: '{orig_phrase}' cleaned to '{phrase}' normalized to '{norm_phrase}'")
        # 1. Try exact normalized match
        if norm_phrase in norm_league_map:
            matched_league = norm_league_map[norm_phrase]
            print(f"[DEBUG] Normalized match: '{norm_phrase}' -> '{matched_league}'")
            found_leagues.append(matched_league)
            continue
        # PATCH: Try matching last 3, 2, or 1 words as league
        words = phrase.split()
        matched = False
        for n in range(3, 0, -1):
            if len(words) >= n:
                candidate = ' '.join(words[-n:])
                norm_candidate = candidate.lower().replace(' ', '')
                if norm_candidate in norm_league_map:
                    matched_league = norm_league_map[norm_candidate]
                    print(f"[DEBUG] Suffix match: '{norm_candidate}' -> '{matched_league}'")
                    found_leagues.append(matched_league)
                    matched = True
                    break
        if matched:
            continue
        # 2. Fuzzy match on normalized forms (PATCH: use higher cutoff 0.9)
        norm_leagues = list(norm_league_map.keys())
        match = get_close_matches(norm_phrase, norm_leagues, n=1, cutoff=0.9)
        if match:
            matched_league = norm_league_map[match[0]];
            print(f"[DEBUG] Fuzzy normalized match: '{norm_phrase}' -> '{matched_league}'")
            found_leagues.append(matched_league)
            continue
        # 3. Fallback: print similarity for debugging
        for norm_lg, orig_lg in norm_league_map.items():
            sim = SequenceMatcher(None, norm_phrase, norm_lg).ratio()
            print(f"[DEBUG] Similarity between '{norm_phrase}' and '{norm_lg}': {sim}")
    found_leagues = list(dict.fromkeys(found_leagues))  # Remove duplicates, preserve order
    if found_leagues:
        found_leagues = prioritize_leagues(found_leagues)
        if len(found_leagues) == 1:
            result["league"] = found_leagues[0]
        else:
            result["league"] = found_leagues
        print(f"[DEBUG] All matched leagues: {found_leagues}")
    else:
        # fallback to old logic if nothing found
        league_keywords = ['in', 'from', 'of', 'league', 'premier', 'bundesliga', 'laliga', 'serie', 'ligue', 'eredivisie', 'championship']
        for keyword in league_keywords:
            if keyword in lowered_query:
                parts = lowered_query.split(keyword)
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
            if len(found_leagues) == 1:
                result["league"] = found_leagues[0]
            else:
                result["league"] = found_leagues
            print(f"[DEBUG] All matched leagues: {found_leagues}")
        else:
            result["league"] = None

    # --- Custom league aliases for top 5/7 leagues ---
    top5_leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
    top7_leagues = top5_leagues + ["Eredivisie", "Liga Portugal"]
    if re.search(r"top ?5 leagues?", lowered):
        result["league"] = top5_leagues
        result["_league_is_alias"] = True
        print(f"[DEBUG] Detected 'top 5 leagues' in query, using: {top5_leagues}")
    elif re.search(r"top ?7 leagues?", lowered):
        result["league"] = top7_leagues
        result["_league_is_alias"] = True
        print(f"[DEBUG] Detected 'top 7 leagues' in query, using: {top7_leagues}")

    # Stat value filter extraction
    stat_value_filter = extract_stat_value_filter(query)
    if stat_value_filter:
        # Use stat aliasing for the stat phrase
        stat_aliases = get_alias_to_column_map()
        stat_phrase = stat_value_filter["stat"]
        # Fuzzy match stat phrase to column
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
    
    minutes_filter = extract_minutes_filter(query)
    if minutes_filter:
        result["minutes_op"] = minutes_filter["minutes_op"]
        result["minutes_value"] = minutes_filter["minutes_value"]
        print(f"[DEBUG] Extracted minutes filter: {minutes_filter['minutes_op']} {minutes_filter['minutes_value']}")


    # LLM fallback for stat/league extraction if missing or ambiguous
    ambiguous_league = not result.get("league") or (isinstance(result.get("league"), list) and len(result["league"]) > 3)
    ambiguous_stat = not result.get("stat") or (isinstance(result.get("stat"), list) and len(result["stat"]) > 3)
    # Do not run LLM fallback if league is set by alias
    if (ambiguous_league and not result.get("_league_is_alias")) or ambiguous_stat:
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
            # Normalize league (only if not set by alias)
            if ambiguous_league and not result.get("_league_is_alias") and llm_json.get('league'):
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