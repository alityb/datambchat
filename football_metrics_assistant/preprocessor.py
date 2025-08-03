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
    Simplified preprocessor that handles core cases correctly.
    Uses clear, debuggable extraction with minimal interdependencies.
    Now includes improved player name matching.
    """
    
    def __init__(self):
        self.alias_map = get_alias_to_column_map()
        self.teams = get_all_teams()
        self.players = get_all_players()
        self.positions = get_all_positions()
        self.leagues = get_all_leagues()
        
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
            'fullback': 'Full-back',
            'full-back': 'Full-back',
            'lb': 'Full-back',
            'rb': 'Full-back',
            'midfielder': 'Midfielder',
            'midfielders': 'Midfielder',
            'cm': 'Midfielder',
            'dm': 'Midfielder',
            'am': 'Midfielder',
            'winger': 'Winger',
            'wingers': 'Winger',
            'lw': 'Winger',
            'rw': 'Winger',
            'goalkeeper': 'Goalkeeper',
            'goalkeepers': 'Goalkeeper',
            'keeper': 'Goalkeeper',
            'keepers': 'Goalkeeper',
            'gk': 'Goalkeeper'
        }

    # ============ IMPROVED PLAYER MATCHING METHODS ============
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize a name by:
        1. Converting to lowercase
        2. Removing accents/diacritics
        3. Removing extra spaces
        4. Removing dots and common punctuation
        """
        if not name:
            return ""
        
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

    def _extract_name_variations(self, full_name: str) -> List[str]:
        """
        Extract common variations of a player name.
        For example: "Mohamed Salah" -> ["mohamed", "salah", "mohamed salah", "m salah"]
        """
        if not full_name:
            return []
        
        normalized = self._normalize_name(full_name)
        parts = normalized.split()
        
        if not parts:
            return [normalized]
        
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
        
        # Add middle name combinations if present
        if len(parts) >= 3:
            # First + Middle + Last
            variations.append(f"{parts[0]} {parts[1]} {parts[-1]}")
            
            # First + Middle initial + Last
            if len(parts[1]) > 0:
                variations.append(f"{parts[0]} {parts[1][0]} {parts[-1]}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var and var not in seen:
                seen.add(var)
                unique_variations.append(var)
        
        return unique_variations

    def _calculate_name_similarity(self, query_name: str, player_name: str) -> float:
        """
        Calculate similarity score between query and player name.
        Returns a score between 0 and 1, where 1 is perfect match.
        """
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
        fuzzy_matches = get_close_matches(query_norm, [player_norm], n=1, cutoff=0.6)
        if fuzzy_matches:
            fuzzy_score = 0.7  # Lower than exact matches
            best_score = max(best_score, fuzzy_score)
        
        return best_score

    def _find_player_name(self, query_name: str) -> str:
        """
        Find player name using improved matching algorithm.
        """
        from football_metrics_assistant.data_utils import load_data
        
        query_name = query_name.strip()
        print(f"[DEBUG] Searching for player: '{query_name}'")
        
        # Load actual player data
        df = load_data()
        if df.empty or 'Player' not in df.columns:
            print("[DEBUG] No player data available")
            return query_name  # Return original if no data
        
        actual_players = df['Player'].unique().tolist()
        print(f"[DEBUG] Total players in database: {len(actual_players)}")
        
        best_match = None
        best_score = 0.0
        min_score = 0.5  # Minimum confidence threshold
        
        # Calculate similarity scores for all players
        for player in actual_players:
            score = self._calculate_name_similarity(query_name, player)
            if score > best_score:
                best_score = score
                best_match = player
        
        # Only return match if it meets minimum threshold
        if best_score >= min_score:
            print(f"[DEBUG] Best match: '{best_match}' (confidence: {best_score:.3f})")
            return best_match
        else:
            print(f"[DEBUG] No confident match found. Best was: '{best_match}' (confidence: {best_score:.3f})")
            
            # Show some suggestions for debugging
            suggestions = []
            for player in actual_players:
                score = self._calculate_name_similarity(query_name, player)
                if score > 0.3:  # Lower threshold for suggestions
                    suggestions.append((player, score))
            
            suggestions.sort(key=lambda x: x[1], reverse=True)
            if suggestions:
                print(f"[DEBUG] Possible suggestions:")
                for player, score in suggestions[:5]:
                    print(f"  - {player} (confidence: {score:.3f})")
            
            return query_name  # Return original if no good match found

    # ============ REST OF THE EXISTING METHODS (unchanged) ============

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Main preprocessing function with clear, sequential steps.
        """
        result = {"original": query}
        lowered = query.lower().strip()
        
        # Step 1: Query type classification (LLM-based) with error handling
        try:
            query_type = classify_query_type(query).strip().replace(':', '').upper()
        except Exception as e:
            print(f"[DEBUG] LLM classification failed: {e}, using fallback")
            query_type = self._fallback_query_classification(lowered)
        
        result["query_type"] = query_type
        
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
        
        return result

    def _fallback_query_classification(self, lowered: str) -> str:
        """Fallback query classification when LLM fails."""
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

    def _extract_top_n(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract top N with clear patterns."""
        patterns = [
            r"(?:top|best|highest)\s*(\d+)",
            r"(\d+)\s*(?:best|top)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                result["top_n"] = int(match.group(1))
                break

    def _extract_age_filters(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract age filters with clear patterns."""
        # U21, U-25, U23 patterns
        u_match = re.search(r"u-?(\d+)", lowered)
        if u_match:
            result["age_filter"] = {"op": "<", "value": int(u_match.group(1))}
            return
        
        # Standard age patterns
        age_patterns = [
            (r"under\s*(\d+)", "<"),
            (r"over\s*(\d+)", ">"),
            (r"above\s*(\d+)", ">"),
            (r"below\s*(\d+)", "<"),
            (r"age\s*(\d+)", "==")
        ]
        
        for pattern, op in age_patterns:
            match = re.search(pattern, lowered)
            if match:
                result["age_filter"] = {"op": op, "value": int(match.group(1))}
                break

    def _extract_minutes_filters(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract minutes played filters."""
        minutes_patterns = [
            (r"(\d+)\+\s*minutes?", ">="),
            (r"more than (\d+) minutes?", ">"),
            (r"at least (\d+) minutes?", ">="),
            (r"over (\d+) minutes?", ">"),
            (r"less than (\d+) minutes?", "<"),
            (r"under (\d+) minutes?", "<")
        ]
        
        for pattern, op in minutes_patterns:
            match = re.search(pattern, lowered)
            if match:
                result["minutes_op"] = op
                result["minutes_value"] = int(match.group(1))
                break

    def _extract_player_reports(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract player report requests with improved pattern matching."""
        
        # Pattern 1: "player_name report"
        report_match = re.search(r'(.+?)\s+report\b', lowered)
        if report_match:
            result["query_type"] = "PLAYER_REPORT"
            player_name = report_match.group(1).strip()
            matched_player = self._find_player_name(player_name)
            result["player"] = matched_player
            return
        
        # Pattern 2: "report on player_name" or "report for player_name"
        report_match2 = re.search(r'report\s+(?:on|for)\s+(.+)', lowered)
        if report_match2:
            result["query_type"] = "PLAYER_REPORT"
            player_name = report_match2.group(1).strip()
            matched_player = self._find_player_name(player_name)
            result["player"] = matched_player
            return
        
        # Pattern 3: Just a player name (if it's a short query and clearly a name)
        words = lowered.split()
        if len(words) <= 2 and len(' '.join(words)) > 3:  # Allow up to 2 words for names like "e can"
            # Try to match as a player name
            matched_player = self._find_player_name(' '.join(words))
            # Only assume it's a player report if we found a confident match
            if matched_player != ' '.join(words):  # If _find_player_name returned something different, it found a match
                result["query_type"] = "PLAYER_REPORT"
                result["player"] = matched_player
                return

    def _extract_stat_definitions(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract stat definition requests."""
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
            match = re.search(pattern, lowered)
            if match:
                result["query_type"] = "STAT_DEFINITION"
                stat_phrase = match.group(1).strip()
                
                # Clean up the stat phrase but preserve it if mapping fails
                clean_words = ['statistic', 'stat', 'metric', 'the', 'a', 'an']
                cleaned_phrase = stat_phrase
                for word in clean_words:
                    cleaned_phrase = re.sub(rf'\b{word}\b', '', cleaned_phrase, flags=re.IGNORECASE).strip()
                
                # Try to map to actual stat
                mapped_stat = self._map_stat_phrase(cleaned_phrase)
                if mapped_stat:
                    result["stat"] = mapped_stat
                else:
                    # Keep the original phrase so error messages are meaningful
                    result["stat"] = stat_phrase
                break

    def _extract_leagues(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract leagues with multi-league support."""
        found_leagues = []
        
        # Check for league aliases first (these should override individual leagues)
        for alias, league_list in self.league_aliases.items():
            if alias in lowered:
                result["league"] = league_list
                result["_league_is_alias"] = True
                return  # IMPORTANT: Return early to avoid mixing with individual leagues
        
        # Check for individual leagues - but be more precise with word boundaries
        for keyword, league_name in self.league_keywords.items():
            # Use word boundaries to avoid partial matches, except for short forms
            if len(keyword) <= 3:  # Short forms like 'epl', 'pl'
                pattern = rf'\b{re.escape(keyword)}\b'
            else:  # Longer names
                pattern = rf'{re.escape(keyword)}'
            
            if re.search(pattern, lowered) and league_name not in found_leagues:
                found_leagues.append(league_name)
        
        # Handle "and" patterns for multi-league
        if " and " in lowered:
            parts = [part.strip() for part in lowered.split(" and ")]
            for part in parts:
                for keyword, league_name in self.league_keywords.items():
                    if len(keyword) <= 3:
                        pattern = rf'\b{re.escape(keyword)}\b'
                    else:
                        pattern = rf'{re.escape(keyword)}'
                    
                    if re.search(pattern, part) and league_name not in found_leagues:
                        found_leagues.append(league_name)
        
        if found_leagues:
            result["league"] = found_leagues[0] if len(found_leagues) == 1 else found_leagues

    def _extract_positions(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract positions with keyword matching."""
        for keyword, position in self.position_keywords.items():
            if re.search(rf'\b{keyword}\b', lowered):
                result["position"] = position
                break

    def _extract_teams(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract teams (conservative approach)."""
        # Only extract teams if query is clearly team-focused
        team_indicators = ['from', 'at', 'playing for']
        if any(indicator in lowered for indicator in team_indicators):
            matches = get_close_matches(lowered, [team.lower() for team in self.teams], n=1, cutoff=0.8)
            if matches:
                # Find original case team name
                for team in self.teams:
                    if team.lower() == matches[0]:
                        result["team"] = team
                        break

    def _extract_stat_value_filters(self, lowered: str, result: Dict[str, Any]) -> None:
        """
        Extract stat value filters with precise pattern matching.
        """
        print(f"[DEBUG] Extracting stat value filters from: '{lowered}'")
        
        # Use word boundaries and non-greedy matching to avoid capturing "and 1000"
        patterns = [
            (r"at least ([\d.]+) (\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">="),          # "at least 0.5 goals"
            (r"atleast ([\d.]+) (\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">="),           # "atleast 0.5 goals"
            (r"more than ([\d.]+) (\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">"),          # "more than 0.5 goals"
            (r"over ([\d.]+) (\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">"),               # "over 0.5 goals"
            (r"under ([\d.]+) (\w+)(?:\s+per\s+\w+)?(?=\s|$)", "<"),              # "under 0.5 goals"
            (r"less than ([\d.]+) (\w+)(?:\s+per\s+\w+)?(?=\s|$)", "<"),          # "less than 0.5 goals"
            (r"exactly ([\d.]+) (\w+)(?:\s+per\s+\w+)?(?=\s|$)", "=="),           # "exactly 0.5 goals"
            (r"with ([\d.]+)\+ (\w+)(?:\s+per\s+\w+)?(?=\s|$)", ">="),            # "with 0.5+ goals"
        ]
        
        for pattern, op in patterns:
            match = re.search(pattern, lowered)
            if match:
                value = float(match.group(1))
                stat_word = match.group(2).strip()
                
                print(f"[DEBUG] Found stat filter: '{stat_word}' {op} {value}")
                
                # Try to map the stat word to actual column
                # First try the word alone
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
    
    print(f"[DEBUG] No stat value filters found")
    def _extract_stats_and_formulas(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract stats and handle formula detection."""
        # Check for math operators first (formula detection)
        if any(op in lowered for op in ['+', '-', '/', '*']) and not any(word in lowered for word in ['report', 'define', 'what is']):
            # FIXED: Don't treat "1000+ minutes" as a formula
            if not re.search(r'\d+\+\s*minutes', lowered):
                formula_result = self._extract_stat_formula(lowered)
                if formula_result:
                    result["stat_formula"] = formula_result
                    result["stat"] = formula_result['expr']
                    return
        
        # Extract regular stats
        stat = self._extract_single_stat(lowered)
        if stat:
            result["stat"] = stat

    def _extract_single_stat(self, lowered: str) -> Optional[str]:
        """Extract a single stat from the query."""
        print(f"[DEBUG] Extracting single stat from: '{lowered}'")
        
        # Handle "with" filter clauses - we want the stat before "with"
        main_part = lowered
        
        # FIXED: Better handling of "with" clauses - only split if it's a filter
        if ' with ' in lowered:
            parts = lowered.split(' with ')
            if len(parts) > 1:
                # Check if part after "with" is a filter (contains numbers or filter words)
                filter_indicators = ['more than', 'less than', 'at least', 'over', 'under', 'minutes', 'than', '+', r'\d']
                after_with = parts[1]
                if any(re.search(indicator, after_with) for indicator in filter_indicators):
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
        
        # Remove numbers
        main_part = re.sub(r'^\d+\s*', '', main_part)
        
        print(f"[DEBUG] Cleaned phrase: '{main_part}'")
        
        # Find best match
        best_match = None
        best_score = 0
        
        for alias, column in self.alias_map.items():
            if alias in main_part:
                score = len(alias)
                if score > best_score:
                    best_match = column
                    best_score = score
        
        print(f"[DEBUG] Best match: '{best_match}'")
        return best_match

    def _extract_stat_formula(self, lowered: str) -> Optional[Dict[str, Any]]:
        """Extract and parse stat formulas with improved division handling and location filtering."""
        print(f"[DEBUG] Extracting formula from: {lowered}")
        
        # First, remove location/filter phrases that might interfere with formula parsing
        # Common location indicators
        location_patterns = [
            r'\s+in\s+\w+(?:\s+\w+)*(?:\s+league)?',  # "in premier league", "in la liga"
            r'\s+from\s+\w+(?:\s+\w+)*',              # "from manchester united"
            r'\s+for\s+\w+(?:\s+\w+)*',               # "for strikers"
            r'\s+playing\s+for\s+\w+(?:\s+\w+)*',     # "playing for arsenal"
            r'\s+with\s+\d+\+?\s*\w+',                # "with 1000+ minutes"
            r'\s+with\s+(?:more|less|at least|over|under)\s+\d+',  # "with more than 500"
        ]
        
        # Clean the query by removing location phrases
        cleaned_query = lowered
        for pattern in location_patterns:
            match = re.search(pattern, cleaned_query)
            if match:
                # Only remove if it's at the end or followed by more location info
                if match.end() >= len(cleaned_query) - 10:  # Near the end
                    cleaned_query = cleaned_query[:match.start()].strip()
                    print(f"[DEBUG] Removed location phrase, cleaned to: '{cleaned_query}'")
                    break
        
        # Look for division patterns first (more specific)
        division_found = False
        division_pos = None
        division_op = None
        
        division_patterns = [
            (r'\s+divided\s+by\s+', '/'),
            (r'\s*/\s*', '/'),
            (r'\s+/\s+', '/'),
            (r'/', '/'),
        ]
        
        for pattern, op in division_patterns:
            match = re.search(pattern, cleaned_query)
            if match:
                division_found = True
                division_pos = (match.start(), match.end())
                division_op = op
                break
        
        if division_found:
            # Handle division case
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
            
            # Extract parts and operators
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
            
            # Additional cleaning for stat names
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

    def _map_stat_phrase(self, phrase: str) -> Optional[str]:
        """Map a stat phrase to an actual column name."""
        phrase = phrase.strip().lower()
        
        # Direct alias match
        if phrase in self.alias_map:
            return self.alias_map[phrase]
        
        # Normalized match
        norm_phrase = normalize_colname(phrase)
        if norm_phrase in self.alias_map:
            return self.alias_map[norm_phrase]
        
        # FIXED: More consistent fuzzy matching
        # First try high confidence fuzzy match on original phrase
        matches = get_close_matches(phrase, list(self.alias_map.keys()), n=1, cutoff=0.85)
        if matches:
            return self.alias_map[matches[0]]
        
        # Then try fuzzy match on normalized phrase
        matches = get_close_matches(norm_phrase, list(self.alias_map.keys()), n=1, cutoff=0.85)
        if matches:
            return self.alias_map[matches[0]]
        
        return None

    def _find_player_name(self, query_name: str) -> str:
        """Find player name with simple matching."""
        query_name = query_name.strip().lower()
        
        # Exact match
        for player in self.players:
            if player.lower() == query_name:
                return player
        
        # Famous player mappings
        famous_players = {
            'haaland': 'E. Haaland',
            'messi': 'L. Messi',
            'ronaldo': 'Cristiano Ronaldo',
            'mbappe': 'K. Mbappé',
            'kane': 'Harry Kane',
            'salah': 'Mohamed Salah'
        }
        
        if query_name in famous_players:
            mapped_name = famous_players[query_name]
            if mapped_name in self.players:
                return mapped_name
        
        # High-confidence fuzzy match
        matches = get_close_matches(query_name, [p.lower() for p in self.players], n=1, cutoff=0.9)
        if matches:
            for player in self.players:
                if player.lower() == matches[0]:
                    return player
        
        return query_name  # Return original if no match

    def _make_safe_var(self, col_name: str) -> str:
        """Convert column name to safe variable name."""
        return (col_name.replace(' ', '_')
                       .replace('.', '_')
                       .replace('%', 'pct')
                       .replace('(', '')
                       .replace(')', '')
                       .replace('-', '_')
                       .replace('+', 'plus')
                       .replace('/', '_'))

    def _validate_and_cleanup(self, result: Dict[str, Any]) -> None:
        """Final validation and cleanup."""
        # Set default top_n if missing for TOP_N queries
        if result.get("query_type") == "TOP_N" and "top_n" not in result:
            result["top_n"] = 5
        
        # FIXED: Better fallback for ambiguous queries
        query_type = result.get("query_type")
        if query_type in ["OTHER", None]:
            # If we have filters but no clear intent, default to FILTER
            if any(key in result for key in ["position", "league", "team", "age_filter", "minutes_op", "stat_op"]):
                result["query_type"] = "FILTER"
            # If we have a stat, default to TOP_N
            elif result.get("stat"):
                result["query_type"] = "TOP_N"
            else:
                result["query_type"] = "LIST"
        
        # Ensure we have a stat for stat-based queries
        if query_type in ["TOP_N", "FILTER"] and not result.get("stat") and not result.get("stat_formula"):
            # Try one more time with a simpler approach
            simple_stat = self._extract_simple_stat_fallback(result["original"])
            if simple_stat:
                result["stat"] = simple_stat

    def _extract_simple_stat_fallback(self, query: str) -> Optional[str]:
        """Simple fallback stat extraction for common cases."""
        lowered = query.lower()
        
        # Simple keyword mapping
        simple_mappings = {
            'goals': 'Goals per 90',
            'assists': 'Assists per 90', 
            'xg': 'xG per 90',
            'xa': 'xA per 90',
            'passes': 'Passes per 90',
            'tackles': 'Sliding tackles per 90',
            'saves': 'Saves per 90',
            'save percentage': 'Save percentage %.1'
        }
        
        for keyword, stat in simple_mappings.items():
            if keyword in lowered:
                return stat
        
        return None


# Main function to maintain compatibility
def preprocess_query(query: str) -> Dict[str, Any]:
    """
    Main preprocessing function using the simplified approach.
    """
    preprocessor = SimplifiedPreprocessor()
    return preprocessor.preprocess_query(query)