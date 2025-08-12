import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple
from football_metrics_assistant.data_utils import (
    get_all_teams, get_all_players, get_all_positions, get_all_leagues,
    get_alias_to_column_map, normalize_colname
)
from football_metrics_assistant.llm_interface import classify_query_type
from difflib import get_close_matches


class SimplifiedPreprocessor:
    """
    Robust preprocessor with comprehensive error handling and improved stat matching.
    """
    
    def __init__(self):
        try:
            self.alias_map = get_alias_to_column_map()
            self.teams = get_all_teams()
            self.players = get_all_players()
            self.positions = get_all_positions()
            self.leagues = get_all_leagues()
        except Exception as e:
            print(f"[ERROR] Failed to initialize data: {e}")
            # Provide fallback empty collections
            self.alias_map = {}
            self.teams = []
            self.players = []
            self.positions = []
            self.leagues = []
        
        # Enhanced stat mappings with common variations
        self.enhanced_stat_mappings = {
            # Goalkeeper stats
            'saves': 'Saves per 90',
            'save percentage': 'Save percentage %.1',
            'saves per 90': 'Saves per 90',
            'clean sheets': 'Clean sheets',
            
            # Defensive stats  
            'tackles': 'Sliding tackles per 90',
            'tackles per 90': 'Sliding tackles per 90',
            'sliding tackles': 'Sliding tackles per 90',
            'sliding tackles per 90': 'Sliding tackles per 90',
            'interceptions': 'Interceptions per 90',
            'interceptions per 90': 'Interceptions per 90',
            'clearances': 'Clearances per 90',
            'clearances per 90': 'Clearances per 90',
            'blocks': 'Blocks per 90',
            'blocks per 90': 'Blocks per 90',
            'duels won': 'Duels won %',
            'duels won %': 'Duels won %',
            'duels won per 90': 'Duels won %',
            'aerial duels': 'Aerial duels per 90',
            'aerial duels per 90': 'Aerial duels per 90',
            'aerial duels won': 'Aerial duels won per 90',
            'aerial duels won per 90': 'Aerial duels won per 90',
            
            # Attacking stats
            'goals': 'Goals per 90',
            'goals per 90': 'Goals per 90',
            'goals per game': 'Goals per 90',
            'assists': 'Assists per 90',
            'assists per 90': 'Assists per 90',
            'assists per game': 'Assists per 90',
            'xg': 'xG per 90',
            'xg per 90': 'xG per 90',
            'expected goals': 'xG per 90',
            'expected goals per 90': 'xG per 90',
            'xa': 'xA per 90',
            'xa per 90': 'xA per 90',
            'expected assists': 'xA per 90',
            'expected assists per 90': 'xA per 90',
            'npxg': 'npxG per 90',
            'npxg per 90': 'npxG per 90',
            'non penalty xg': 'npxG per 90',
            'goals and assists': 'Goals + Assists per 90',
            'goals + assists': 'Goals + Assists per 90',
            'goals plus assists': 'Goals + Assists per 90',
            'shots': 'Shots per 90',
            'shots per 90': 'Shots per 90',
            'shots on target': 'Shots on target %.1',
            
            # Passing stats
            'passes': 'Passes per 90',
            'passes per 90': 'Passes per 90',
            'pass completion': 'Pass completion %.1',
            'pass accuracy': 'Pass completion %.1',
            'key passes': 'Key passes per 90',
            'key passes per 90': 'Key passes per 90',
            'progressive passes': 'Progressive passes per 90',
            'progressive passes per 90': 'Progressive passes per 90',
            'crosses': 'Crosses per 90',
            'crosses per 90': 'Crosses per 90',
            'cross accuracy': 'Cross accuracy %.1',
            
            # Physical stats
            'touches': 'Touches per 90',
            'touches per 90': 'Touches per 90',
            'dribbles': 'Dribbles attempted per 90',
            'dribbles attempted': 'Dribbles attempted per 90',
            'dribbles per 90': 'Dribbles attempted per 90',
            'dribble success': 'Dribble success rate %.1',
            'dribble success rate': 'Dribble success rate %.1',
            'progressive carries': 'Progressive carries per 90',
            'progressive carries per 90': 'Progressive carries per 90',
        }
        
        # Core league mappings
        self.league_keywords = {
            'premier league': 'Premier League',
            'epl': 'Premier League',
            'pl': 'Premier League',
            'la liga': 'La Liga',
            'laliga': 'La Liga',
            'spain': 'La Liga',
            'bundesliga': 'Bundesliga',
            'germany': 'Bundesliga',
            'serie a': 'Serie A',
            'seriea': 'Serie A',
            'italy': 'Serie A',
            'ligue 1': 'Ligue 1',
            'ligue1': 'Ligue 1',
            'france': 'Ligue 1',
            'eredivisie': 'Eredivisie',
            'netherlands': 'Eredivisie',
            'championship': 'Championship',
            'mls': 'MLS'
        }
        
        # League aliases for multi-league queries
        self.league_aliases = {
            'top 5 leagues': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
            'big 5': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
            'top five': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
            'big five': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1'],
            'top 7 leagues': ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1', 'Eredivisie', 'Liga Portugal']
        }
        
        # Position mappings
        self.position_keywords = {
            'striker': 'Striker',
            'strikers': 'Striker',
            'forward': 'Striker', 
            'forwards': 'Striker',
            'cf': 'Striker',
            'st': 'Striker',
            'defender': ['Centre-back', 'Full-back'],
            'defenders': ['Centre-back', 'Full-back'],
            'cb': 'Centre-back',
            'centreback': 'Centre-back',
            'centre-back': 'Centre-back',
            'centerback': 'Centre-back',
            'center-back': 'Centre-back',
            'center back': 'Centre-back',
            'center backs': 'Centre-back',
            'centre back': 'Centre-back',
            'centre backs': 'Centre-back',
            'fullback': 'Full-back',
            'full-back': 'Full-back',
            'full back': 'Full-back',
            'full backs': 'Full-back',
            'lb': 'Full-back',
            'rb': 'Full-back',
            'left back': 'Full-back',
            'right back': 'Full-back',
            'leftback': 'Full-back',
            'rightback': 'Full-back',
            'midfielder': 'Midfielder',
            'midfielders': 'Midfielder',
            'cm': 'Midfielder',
            'dm': 'Midfielder',
            'am': 'Midfielder',
            'central midfielder': 'Midfielder',
            'defensive midfielder': 'Midfielder',
            'attacking midfielder': 'Midfielder',
            'winger': 'Winger',
            'wingers': 'Winger',
            'lw': 'Winger',
            'rw': 'Winger',
            'left winger': 'Winger',
            'right winger': 'Winger',
            'goalkeeper': 'Goalkeeper',
            'goalkeepers': 'Goalkeeper',
            'keeper': 'Goalkeeper',
            'keepers': 'Goalkeeper',
            'gk': 'Goalkeeper'
        }

    def _safe_regex_search(self, pattern: str, text: str, flags=0) -> Optional[re.Match]:
        """
        Safe regex search with error handling.
        """
        try:
            return re.search(pattern, text, flags)
        except re.error as e:
            print(f"[ERROR] Regex error with pattern '{pattern}': {e}")
            return None
        except Exception as e:
            print(f"[ERROR] Unexpected error in regex search: {e}")
            return None

    def _safe_regex_findall(self, pattern: str, text: str, flags=0) -> List[str]:
        """
        Safe regex findall with error handling.
        """
        try:
            return re.findall(pattern, text, flags)
        except re.error as e:
            print(f"[ERROR] Regex error with pattern '{pattern}': {e}")
            return []
        except Exception as e:
            print(f"[ERROR] Unexpected error in regex findall: {e}")
            return []

    def _normalize_name(self, name: str) -> str:
        """
        Normalize a name safely.
        """
        if not name or not isinstance(name, str):
            return ""
        
        try:
            # Convert to lowercase
            name = name.lower().strip()
            
            # Remove accents and diacritics
            name = unicodedata.normalize('NFKD', name)
            name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
            
            # Remove dots, hyphens, and common punctuation (but keep spaces)
            name = re.sub(r'[.\-_\'"]', '', name)
            
            # Normalize multiple spaces to single spaces
            name = re.sub(r'\s+', ' ', name).strip()
            
            return name
        except Exception as e:
            print(f"[ERROR] Failed to normalize name '{name}': {e}")
            return str(name).lower().strip()

    def _extract_name_variations(self, full_name: str) -> List[str]:
        """
        Extract common variations of a player name safely.
        """
        if not full_name:
            return []
        
        try:
            normalized = self._normalize_name(full_name)
            parts = normalized.split()
            
            if not parts:
                return [normalized] if normalized else []
            
            variations = [normalized]  # Full normalized name
            
            # Add individual parts (first name, last name)
            for part in parts:
                if len(part) > 1:  # Skip single letters unless they're the only part
                    variations.append(part)
            
            # Add first name + last name combinations
            if len(parts) >= 2:
                # First + Last
                variations.append(f"{parts[0]} {parts[-1]}")
                
                # First initial + Last name (e.g., "m salah")
                if len(parts[0]) > 0:
                    variations.append(f"{parts[0][0]} {parts[-1]}")
                
                # Last name only (most common search)
                variations.append(parts[-1])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_variations = []
            for var in variations:
                if var and var not in seen:
                    seen.add(var)
                    unique_variations.append(var)
            
            return unique_variations
            
        except Exception as e:
            print(f"[ERROR] Failed to extract name variations for '{full_name}': {e}")
            return [str(full_name).lower().strip()] if full_name else []

    def _find_player_name(self, query_name: str) -> str:
        """
        Find player name - simplified to avoid preliminary matching conflicts.
        The main lookup functions will handle the comprehensive search.
        """
        if not query_name:
            return ""
        
        try:
            query_name = str(query_name).strip()
            print(f"[DEBUG] _find_player_name called with: '{query_name}'")
            
            # For player report queries, skip preliminary search and let the main
            # lookup functions handle it comprehensively
            # This prevents conflicts where preliminary search finds wrong matches
            
            print(f"[DEBUG] Returning original query unchanged: '{query_name}'")
            return query_name
            
        except Exception as e:
            print(f"[ERROR] _find_player_name failed: {e}")
            return query_name

    def _calculate_name_similarity(self, query_name: str, player_name: str) -> float:
        """
        Calculate similarity score between query and player name with error handling.
        """
        try:
            query_norm = self._normalize_name(query_name)
            player_norm = self._normalize_name(player_name)
            
            if not query_norm or not player_norm:
                return 0.0
            
            # Exact match gets highest score
            if query_norm == player_norm:
                return 1.0
            
            # Check if query is contained in player name or vice versa
            if query_norm in player_norm:
                return 0.9 * (len(query_norm) / len(player_norm))
            
            if player_norm in query_norm:
                return 0.9 * (len(player_norm) / len(query_norm))
            
            # Check variations
            query_variations = self._extract_name_variations(query_name)
            player_variations = self._extract_name_variations(player_name)
            
            best_score = 0.0
            for q_var in query_variations:
                for p_var in player_variations:
                    if q_var == p_var:
                        # Weight by length of match
                        score = 0.8 * (len(q_var) / max(len(query_norm), len(player_norm)))
                        best_score = max(best_score, score)
            
            # Fuzzy string matching as fallback
            try:
                fuzzy_matches = get_close_matches(query_norm, [player_norm], n=1, cutoff=0.6)
                if fuzzy_matches:
                    fuzzy_score = 0.7  # Lower than exact matches
                    best_score = max(best_score, fuzzy_score)
            except Exception as e:
                print(f"[ERROR] Fuzzy matching failed: {e}")
            
            return best_score
            
        except Exception as e:
            print(f"[ERROR] Failed to calculate name similarity: {e}")
            return 0.0

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Main preprocessing function with comprehensive error handling.
        """
        # Initialize with safe defaults
        result = {"original": query if query else "", "query_type": "LIST"}
        
        try:
            if not query or not isinstance(query, str):
                return result
            
            lowered = query.lower().strip()
            if not lowered:
                return result
            
            # Step 1: Query type classification with error handling
            try:
                query_type = classify_query_type(query)
                if query_type and isinstance(query_type, str):
                    result["query_type"] = query_type.strip().replace(':', '').upper()
                else:
                    result["query_type"] = self._fallback_query_classification(lowered)
            except Exception as e:
                print(f"[DEBUG] LLM classification failed: {e}, using fallback")
                result["query_type"] = self._fallback_query_classification(lowered)
            
            # Step 2: Extract basic numeric filters
            self._extract_top_n(lowered, result)
            self._extract_age_filters(lowered, result)
            self._extract_minutes_filters(lowered, result)
            
            # Step 3: Extract entities (order matters - most specific first)
            self._extract_player_reports(lowered, result)
            self._extract_stat_definitions(lowered, result)
            
            # Step 4: Extract filters and stats (if not already handled)
            if result["query_type"] not in ["PLAYER_REPORT", "STAT_DEFINITION"]:
                self._extract_leagues(lowered, result)
                self._extract_positions(lowered, result)
                self._extract_teams(lowered, result)
                
                # Handle stat value filters BEFORE general stat extraction
                self._extract_stat_value_filters(lowered, result)
                
                # Only extract general stats if no specific stat was found
                if not result.get('stat'):
                    self._extract_stats_and_formulas(lowered, result)
            
            # Step 5: Post-processing and validation
            self._validate_and_cleanup(result)
            
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {e}")
            # Return safe fallback result
            result = {
                "original": query if query else "",
                "query_type": "LIST",
                "error": f"Preprocessing failed: {str(e)}"
            }
        
        return result

    def _fallback_query_classification(self, lowered: str) -> str:
        """Fallback query classification when LLM fails."""
        try:
            if any(word in lowered for word in ['report', 'profile', 'analysis of']):
                return "PLAYER_REPORT"
            elif any(word in lowered for word in ['define', 'what is', 'explain', 'meaning']):
                return "STAT_DEFINITION"
            elif any(word in lowered for word in ['top', 'best', 'highest']):
                return "TOP_N"
            elif any(word in lowered for word in ['how many', 'count', 'number of']):
                return "COUNT"
            elif any(word in lowered for word in ['list', 'show me', 'who are']):
                return "LIST"
            else:
                return "FILTER"
        except Exception as e:
            print(f"[ERROR] Fallback classification failed: {e}")
            return "LIST"

    def _extract_top_n(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract top N with safe regex patterns."""
        try:
            patterns = [
                r"(?:top|best|highest)\s*(\d+)",
                r"(\d+)\s*(?:best|top)"
            ]
            
            for pattern in patterns:
                match = self._safe_regex_search(pattern, lowered)
                if match:
                    result["top_n"] = int(match.group(1))
                    break
        except Exception as e:
            print(f"[ERROR] Failed to extract top_n: {e}")

    def _extract_age_filters(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract age filters with safe regex patterns."""
        try:
            # U21, U-25, U23 patterns - use safer regex
            u_match = self._safe_regex_search(r"u-?(\d+)", lowered)
            if u_match:
                result["age_filter"] = {"op": "<", "value": int(u_match.group(1))}
                return
            
            # Standard age patterns with safer regex
            age_patterns = [
                (r"under\s+(\d+)", "<"),
                (r"over\s+(\d+)", ">"),
                (r"above\s+(\d+)", ">"),
                (r"below\s+(\d+)", "<"),
                (r"age\s+(\d+)", "=="),
                (r"older\s+than\s+(\d+)\s+years?", ">"),
                (r"(\d+)\+?\s+years?\s+old", ">"),
                (r"more\s+than\s+(\d+)\s+years?\s+old", ">"),
                (r"at\s+least\s+(\d+)\s+years?\s+old", ">="),
                (r"younger\s+than\s+(\d+)\s+years?", "<"),
                (r"less\s+than\s+(\d+)\s+years?\s+old", "<")
            ]
            
            for pattern, op in age_patterns:
                match = self._safe_regex_search(pattern, lowered)
                if match:
                    result["age_filter"] = {"op": op, "value": int(match.group(1))}
                    break
        except Exception as e:
            print(f"[ERROR] Failed to extract age filters: {e}")

    def _extract_minutes_filters(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract minutes played filters with safe regex."""
        try:
            minutes_patterns = [
                (r"(\d+)\+\s*minutes?", ">="),
                (r"more\s+than\s+(\d+)\s+minutes?", ">"),
                (r"at\s+least\s+(\d+)\s+minutes?", ">="),
                (r"over\s+(\d+)\s+minutes?", ">"),
                (r"less\s+than\s+(\d+)\s+minutes?", "<"),
                (r"under\s+(\d+)\s+minutes?", "<"),
                (r"exactly\s+(\d+)\s+minutes?", "=="),
                (r"(\d+)\s+minutes?\s+exactly", "=="),
                (r"(\d+)\s+minutes?\s+played", "=="),
                (r"with\s+(\d+)\s+minutes?", "=="),
                (r"(\d+)\s+minutes?\s+only", "==")
            ]
            
            for pattern, op in minutes_patterns:
                match = self._safe_regex_search(pattern, lowered)
                if match:
                    result["minutes_op"] = op
                    result["minutes_value"] = int(match.group(1))
                    break
            
            # Special case for "0 minutes" or similar patterns
            if not result.get("minutes_op"):
                zero_minutes_patterns = [
                    r"0\s+minutes?",
                    r"zero\s+minutes?",
                    r"no\s+minutes?",
                    r"without\s+minutes?"
                ]
                for pattern in zero_minutes_patterns:
                    if self._safe_regex_search(pattern, lowered):
                        result["minutes_op"] = "=="
                        result["minutes_value"] = 0
                        break
                        
        except Exception as e:
            print(f"[ERROR] Failed to extract minutes filters: {e}")

    def _extract_player_reports(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract player report requests with safe regex."""
        try:
            # Pattern 1: "player_name report"
            report_match = self._safe_regex_search(r'(.+?)\s+report\b', lowered)
            if report_match:
                result["query_type"] = "PLAYER_REPORT"
                player_name = report_match.group(1).strip()
                matched_player = self._find_player_name(player_name)
                result["player"] = matched_player
                return
            
            # Pattern 2: "report on player_name" or "report for player_name"
            report_match2 = self._safe_regex_search(r'report\s+(?:on|for)\s+(.+)', lowered)
            if report_match2:
                result["query_type"] = "PLAYER_REPORT"
                player_name = report_match2.group(1).strip()
                matched_player = self._find_player_name(player_name)
                result["player"] = matched_player
                return
            
            # Pattern 3: Just a player name (if it's a short query)
            words = lowered.split()
            if len(words) <= 2 and len(' '.join(words)) > 3:
                # Try to match as a player name
                matched_player = self._find_player_name(' '.join(words))
                # Only assume it's a player report if we found a confident match
                if matched_player != ' '.join(words):
                    result["query_type"] = "PLAYER_REPORT"
                    result["player"] = matched_player
                    return
        except Exception as e:
            print(f"[ERROR] Failed to extract player reports: {e}")

    def _extract_stat_definitions(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract stat definition requests with safe regex."""
        try:
            definition_patterns = [
                r'define\s+(.+)',
                r'what\s+is\s+(.+)',
                r'explain\s+(.+)',
                r'(.+)\s+definition',
                r'(.+)\s+meaning',
                r'how\s+is\s+(.+)\s+calculated',
                r'what\s+does\s+(.+)\s+mean'
            ]
            
            for pattern in definition_patterns:
                match = self._safe_regex_search(pattern, lowered)
                if match:
                    result["query_type"] = "STAT_DEFINITION"
                    stat_phrase = match.group(1).strip()
                    
                    # Clean up the stat phrase
                    clean_words = ['statistic', 'stat', 'metric', 'the', 'a', 'an']
                    cleaned_phrase = stat_phrase
                    for word in clean_words:
                        cleaned_phrase = re.sub(rf'\b{re.escape(word)}\b', '', cleaned_phrase, flags=re.IGNORECASE).strip()
                    
                    # Try to map to actual stat
                    mapped_stat = self._map_stat_phrase(cleaned_phrase)
                    if mapped_stat:
                        result["stat"] = mapped_stat
                    else:
                        result["stat"] = stat_phrase
                    break
        except Exception as e:
            print(f"[ERROR] Failed to extract stat definitions: {e}")

    def _extract_leagues(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract leagues with safe processing."""
        try:
            found_leagues = []
            
            # Check for league aliases first
            for alias, league_list in self.league_aliases.items():
                if alias in lowered:
                    result["league"] = league_list
                    result["_league_is_alias"] = True
                    return
            
            # Check for individual leagues
            for keyword, league_name in self.league_keywords.items():
                try:
                    if len(keyword) <= 3:  # Short forms like 'epl', 'pl'
                        pattern = rf'\b{re.escape(keyword)}\b'
                    else:  # Longer names
                        pattern = rf'{re.escape(keyword)}'
                    
                    if self._safe_regex_search(pattern, lowered) and league_name not in found_leagues:
                        found_leagues.append(league_name)
                except Exception as e:
                    print(f"[ERROR] Failed to process league keyword '{keyword}': {e}")
                    continue
            
            if found_leagues:
                result["league"] = found_leagues[0] if len(found_leagues) == 1 else found_leagues
        except Exception as e:
            print(f"[ERROR] Failed to extract leagues: {e}")

    def _extract_positions(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract positions with safe keyword matching."""
        try:
            for keyword, position in self.position_keywords.items():
                pattern = rf'\b{re.escape(keyword)}\b'
                if self._safe_regex_search(pattern, lowered):
                    result["position"] = position
                    break
        except Exception as e:
            print(f"[ERROR] Failed to extract positions: {e}")

    def _extract_teams(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract teams with safe processing."""
        try:
            # Only extract teams if query is clearly team-focused
            team_indicators = ['from', 'at', 'playing for']
            if any(indicator in lowered for indicator in team_indicators):
                try:
                    matches = get_close_matches(lowered, [team.lower() for team in self.teams], n=1, cutoff=0.8)
                    if matches:
                        # Find original case team name
                        for team in self.teams:
                            if team.lower() == matches[0]:
                                result["team"] = team
                                break
                except Exception as e:
                    print(f"[ERROR] Failed team matching: {e}")
        except Exception as e:
            print(f"[ERROR] Failed to extract teams: {e}")

    def _extract_stat_value_filters(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract stat value filters with safe regex patterns."""
        try:
            print(f"[DEBUG] Extracting stat value filters from: '{lowered}'")
            
            # Use safer regex patterns with proper escaping
            patterns = [
                (r"at\s+least\s+([\d.]+)\s+(\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">="),
                (r"atleast\s+([\d.]+)\s+(\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">="),
                (r"more\s+than\s+([\d.]+)\s+(\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">"),
                (r"over\s+([\d.]+)\s+(\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">"),
                (r"under\s+([\d.]+)\s+(\w+)(?:\s+per\s+\w+)?(?=\s|$)", "<"),
                (r"less\s+than\s+([\d.]+)\s+(\w+)(?:\s+per\s+\w+)?(?=\s|$)", "<"),
                (r"exactly\s+([\d.]+)\s+(\w+)(?:\s+per\s+\w+)?(?=\s|$)", "=="),
                (r"with\s+([\d.]+)\+\s*(\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">="),
            ]
            
            for pattern, op in patterns:
                match = self._safe_regex_search(pattern, lowered)
                if match:
                    try:
                        value = float(match.group(1))
                        stat_word = match.group(2).strip()
                        
                        print(f"[DEBUG] Found stat filter: '{stat_word}' {op} {value}")
                        
                        # Try to map the stat word to actual column
                        mapped_stat = self._map_stat_phrase(stat_word)
                        
                        # If that fails, try with "per 90" suffix for common stats
                        if not mapped_stat and stat_word in ['goals', 'assists', 'passes', 'tackles', 'saves']:
                            mapped_stat = self._map_stat_phrase(f"{stat_word} per 90")
                        
                        if mapped_stat:
                            result["stat"] = mapped_stat
                            result["stat_op"] = op
                            result["stat_value"] = value
                            result["query_type"] = "FILTER"
                            print(f"[DEBUG] Successfully mapped: {mapped_stat} {op} {value}")
                            return
                        else:
                            print(f"[DEBUG] Could not map stat: '{stat_word}'")
                    except (ValueError, IndexError) as e:
                        print(f"[ERROR] Failed to parse stat filter values: {e}")
                        continue
            
            print(f"[DEBUG] No stat value filters found")
        except Exception as e:
            print(f"[ERROR] Failed to extract stat value filters: {e}")

    def _extract_stats_and_formulas(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract stats and handle formula detection with safe processing."""
        try:
            # Skip if we already have a stat from stat value filter
            if result.get("stat"):
                print(f"[DEBUG] Stat already extracted: {result.get('stat')}, skipping general stat extraction")
                return
            
            # Check for math operators first (formula detection)
            math_operators = ['+', '-', '/', '*']
            has_math = any(op in lowered for op in math_operators)
            is_report_query = any(word in lowered for word in ['report', 'define', 'what is'])
            is_minutes_plus = bool(self._safe_regex_search(r'\d+\+\s*minutes', lowered))
            
            if has_math and not is_report_query and not is_minutes_plus:
                formula_result = self._extract_stat_formula(lowered)
                if formula_result:
                    result["stat_formula"] = formula_result
                    result["stat"] = formula_result['expr']
                    return
            
            # Extract regular stats only if no stat found yet
            stat = self._extract_single_stat(lowered)
            if stat:
                result["stat"] = stat
        except Exception as e:
            print(f"[ERROR] Failed to extract stats and formulas: {e}")

    def _extract_single_stat(self, lowered: str) -> Optional[str]:
        """Extract a single stat from the query with safe processing."""
        try:
            print(f"[DEBUG] Extracting single stat from: '{lowered}'")
            
            # Handle "with" filter clauses - we want the stat before "with"
            main_part = lowered
            
            # Better handling of "with" clauses - only split if it's a filter
            if ' with ' in lowered:
                parts = lowered.split(' with ')
                if len(parts) > 1:
                    # Check if part after "with" is a filter (contains numbers or filter words)
                    filter_indicators = ['more than', 'less than', 'at least', 'over', 'under', 'minutes', 'than', r'\+', r'\d']
                    after_with = parts[1]
                    is_filter = any(self._safe_regex_search(indicator, after_with) for indicator in filter_indicators)
                    if is_filter:
                        main_part = parts[0].strip()
                        print(f"[DEBUG] Detected filter after 'with', using: '{main_part}'")
            
            # Handle location delimiters
            for delimiter in [' in ', ' for ', ' from ']:
                if delimiter in main_part:
                    main_part = main_part.split(delimiter)[0].strip()
                    break
            
            # Remove prefixes
            prefixes = ['top ', 'best ', 'highest ', 'most ', 'players by ']
            for prefix in prefixes:
                if main_part.startswith(prefix):
                    main_part = main_part[len(prefix):].strip()
                    break
            
            # Remove numbers at the start
            main_part = re.sub(r'^\d+\s*', '', main_part)
            
            print(f"[DEBUG] Cleaned phrase: '{main_part}'")
            
            # Try enhanced mappings first
            mapped_stat = self._map_stat_phrase(main_part)
            if mapped_stat:
                print(f"[DEBUG] Mapped to: '{mapped_stat}'")
                return mapped_stat
            
            # Fallback to alias map
            best_match = None
            best_score = 0
            
            try:
                for alias, column in self.alias_map.items():
                    if alias in main_part:
                        score = len(alias)
                        if score > best_score:
                            best_match = column
                            best_score = score
            except Exception as e:
                print(f"[ERROR] Failed alias map lookup: {e}")
            
            print(f"[DEBUG] Best match: '{best_match}'")
            return best_match
            
        except Exception as e:
            print(f"[ERROR] Failed to extract single stat: {e}")
            return None

    def _extract_stat_formula(self, lowered: str) -> Optional[Dict[str, Any]]:
        """Extract and parse stat formulas with safe processing."""
        try:
            print(f"[DEBUG] Extracting formula from: {lowered}")
            
            # Remove location/filter phrases safely
            location_patterns = [
                r'\s+in\s+\w+(?:\s+\w+)*(?:\s+league)?',
                r'\s+from\s+\w+(?:\s+\w+)*',
                r'\s+for\s+\w+(?:\s+\w+)*',
                r'\s+playing\s+for\s+\w+(?:\s+\w+)*',
                r'\s+with\s+\d+\+?\s*\w+',
                r'\s+with\s+(?:more|less|at least|over|under)\s+\d+',
            ]
            
            cleaned_query = lowered
            for pattern in location_patterns:
                match = self._safe_regex_search(pattern, cleaned_query)
                if match and match.end() >= len(cleaned_query) - 10:
                    cleaned_query = cleaned_query[:match.start()].strip()
                    print(f"[DEBUG] Removed location phrase, cleaned to: '{cleaned_query}'")
                    break
            
            # Look for division patterns first
            division_patterns = [
                (r'\s+divided\s+by\s+', '/'),
                (r'\s*/\s*', '/'),
                (r'\s+/\s+', '/'),
                (r'/', '/'),
            ]
            
            division_found = False
            division_pos = None
            division_op = None
            
            for pattern, op in division_patterns:
                match = self._safe_regex_search(pattern, cleaned_query)
                if match:
                    division_found = True
                    division_pos = (match.start(), match.end())
                    division_op = op
                    break
            
            if division_found:
                start, end = division_pos
                left_part = cleaned_query[:start].strip()
                right_part = cleaned_query[end:].strip()
                
                if not left_part or not right_part:
                    return None
                    
                parts = [left_part, right_part]
                ops = [division_op]
            else:
                # Handle other operators
                ops_pattern = r'[+\-*]'
                ops_matches = list(re.finditer(ops_pattern, cleaned_query))
                
                if not ops_matches:
                    return None
                
                parts = []
                ops = []
                last_pos = 0
                
                for match in ops_matches:
                    part = cleaned_query[last_pos:match.start()].strip()
                    if part:
                        parts.append(part)
                    ops.append(match.group())
                    last_pos = match.end()
                
                final_part = cleaned_query[last_pos:].strip()
                if final_part:
                    parts.append(final_part)
            
            print(f"[DEBUG] Formula parts: {parts}")
            print(f"[DEBUG] Formula ops: {ops}")
            
            if len(parts) != len(ops) + 1:
                return None
            
            # Clean and map parts
            mapped_cols = []
            safe_vars = []
            safe_map = {}
            
            for part in parts:
                # Clean part
                clean_part = part
                
                # Remove prefixes
                prefixes = ["most ", "top ", "highest ", "best ", "players by ", "strikers by "]
                for prefix in prefixes:
                    if clean_part.startswith(prefix):
                        clean_part = clean_part[len(prefix):].strip()
                        break
                
                # Remove numbers
                clean_part = re.sub(r'^\d+\s*', '', clean_part)
                clean_part = clean_part.strip()
                
                print(f"[DEBUG] Cleaning part: '{part}' -> '{clean_part}'")
                
                # Map to column
                mapped_col = self._map_stat_phrase(clean_part)
                if not mapped_col:
                    print(f"[DEBUG] Could not map: '{clean_part}'")
                    return None
                
                print(f"[DEBUG] Mapped '{clean_part}' to '{mapped_col}'")
                mapped_cols.append(mapped_col)
                safe_var = self._make_safe_var(mapped_col)
                safe_vars.append(safe_var)
                safe_map[safe_var] = mapped_col
            
            # Build expressions
            safe_expr = ''
            display_expr = ''
            
            for i, (safe_var, mapped_col) in enumerate(zip(safe_vars, mapped_cols)):
                safe_expr += f'safe_map["{safe_var}"]'
                display_expr += mapped_col
                
                if i < len(ops):
                    safe_expr += f' {ops[i]} '
                    display_expr += f' {ops[i]} '
            
            print(f"[DEBUG] Built formula: {display_expr}")
            
            return {
                'expr': safe_expr,
                'display_expr': display_expr,
                'columns': mapped_cols,
                'ops': ops,
                'safe_map': safe_map
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to extract stat formula: {e}")
            return None

    def _map_stat_phrase(self, phrase: str) -> Optional[str]:
        """Map a stat phrase to an actual column name with enhanced mappings."""
        try:
            if not phrase or not isinstance(phrase, str):
                return None
            
            phrase = phrase.strip().lower()
            
            # Try enhanced mappings first
            if phrase in self.enhanced_stat_mappings:
                return self.enhanced_stat_mappings[phrase]
            
            # Try alias map
            if phrase in self.alias_map:
                return self.alias_map[phrase]
            
            # Normalized match
            try:
                norm_phrase = normalize_colname(phrase)
                if norm_phrase in self.alias_map:
                    return self.alias_map[norm_phrase]
                
                if norm_phrase in self.enhanced_stat_mappings:
                    return self.enhanced_stat_mappings[norm_phrase]
            except Exception as e:
                print(f"[ERROR] Failed to normalize phrase '{phrase}': {e}")
            
            # Fuzzy matching with error handling
            try:
                # First try enhanced mappings
                matches = get_close_matches(phrase, list(self.enhanced_stat_mappings.keys()), n=1, cutoff=0.85)
                if matches:
                    return self.enhanced_stat_mappings[matches[0]]
                
                # Then try alias map
                matches = get_close_matches(phrase, list(self.alias_map.keys()), n=1, cutoff=0.85)
                if matches:
                    return self.alias_map[matches[0]]
            except Exception as e:
                print(f"[ERROR] Fuzzy matching failed for phrase '{phrase}': {e}")
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to map stat phrase '{phrase}': {e}")
            return None

    def _make_safe_var(self, col_name: str) -> str:
        """Convert column name to safe variable name."""
        try:
            if not col_name:
                return "unknown"
            
            safe_name = (str(col_name).replace(' ', '_')
                                     .replace('.', '_')
                                     .replace('%', 'pct')
                                     .replace('(', '')
                                     .replace(')', '')
                                     .replace('-', '_')
                                     .replace('+', 'plus')
                                     .replace('/', '_'))
            return safe_name if safe_name else "unknown"
        except Exception as e:
            print(f"[ERROR] Failed to make safe var from '{col_name}': {e}")
            return "unknown"

    def _validate_and_cleanup(self, result: Dict[str, Any]) -> None:
        """Final validation and cleanup with error handling."""
        try:
            # Set default top_n if missing for TOP_N queries
            if result.get("query_type") == "TOP_N" and "top_n" not in result:
                result["top_n"] = 5
            
            # Better fallback for ambiguous queries
            query_type = result.get("query_type")
            if query_type in ["OTHER", None, ""]:
                # If we have filters but no clear intent, default to FILTER
                filter_keys = ["position", "league", "team", "age_filter", "minutes_op", "stat_op"]
                if any(key in result for key in filter_keys):
                    result["query_type"] = "FILTER"
                # If we have a stat, default to TOP_N
                elif result.get("stat"):
                    result["query_type"] = "TOP_N"
                else:
                    result["query_type"] = "LIST"
            
            # Ensure we have a stat for stat-based queries
            if query_type in ["TOP_N", "FILTER"] and not result.get("stat") and not result.get("stat_formula"):
                # Try one more time with a simpler approach
                simple_stat = self._extract_simple_stat_fallback(result.get("original", ""))
                if simple_stat:
                    result["stat"] = simple_stat
                    
        except Exception as e:
            print(f"[ERROR] Validation and cleanup failed: {e}")

    def _extract_simple_stat_fallback(self, query: str) -> Optional[str]:
        """Simple fallback stat extraction for common cases with error handling."""
        try:
            if not query:
                return None
            
            lowered = query.lower()
            
            # Simple keyword mapping with enhanced mappings
            for keyword, stat in self.enhanced_stat_mappings.items():
                if keyword in lowered:
                    return stat
            
            return None
            
        except Exception as e:
            print(f"[ERROR] Simple stat fallback failed: {e}")
            return None


# Main function to maintain compatibility
def preprocess_query(query: str) -> Dict[str, Any]:
    """
    Main preprocessing function using the robust approach.
    """
    try:
        preprocessor = SimplifiedPreprocessor()
        return preprocessor.preprocess_query(query)
    except Exception as e:
        print(f"[ERROR] Preprocessing completely failed: {e}")
        # Return absolute fallback
        return {
            "original": query if query else "",
            "query_type": "LIST",
            "error": f"Complete preprocessing failure: {str(e)}"
        }