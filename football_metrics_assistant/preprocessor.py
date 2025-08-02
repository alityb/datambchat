import re
from typing import Dict, Any, List, Optional
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
            'gk': 'Goalkeeper',
            'keeper': 'Goalkeeper'
        }

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Main preprocessing function with clear, sequential steps.
        """
        result = {"original": query}
        lowered = query.lower().strip()
        
        # Step 1: Query type classification (LLM-based)
        query_type = classify_query_type(query).strip().replace(':', '').upper()
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
            self._extract_stats_and_formulas(lowered, result)
            self._extract_stat_value_filters(lowered, result)
        
        # Step 5: Post-processing and validation
        self._validate_and_cleanup(result)
        
        return result

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
        """Extract player report requests."""
        report_match = re.search(r'(\w+(?:\s+\w+)*)\s+report\b', lowered)
        if report_match:
            result["query_type"] = "PLAYER_REPORT"
            player_name = report_match.group(1).strip()
            
            # Simple player name matching
            matched_player = self._find_player_name(player_name)
            result["player"] = matched_player

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

    def _extract_stats_and_formulas(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract stats and handle formula detection."""
        # Check for math operators first (formula detection)
        if any(op in lowered for op in ['+', '-', '/', '*']) and not any(word in lowered for word in ['report', 'define', 'what is']):
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
        # Get the main part of the query (before location/filters)
        main_part = lowered
        for delimiter in [' in ', ' for ', ' from ', ' with ']:
            if delimiter in lowered:
                parts = lowered.split(delimiter)
                if len(parts) > 1:
                    # Take the part that's most likely to contain the stat
                    if delimiter == ' with ' and any(word in parts[1] for word in ['than', 'least', 'most']):
                        # "with more than X goals" - stat is after 'with'
                        continue
                    else:
                        main_part = parts[0]
                        break
        
        # Look for stat keywords in the main part
        best_match = None
        best_score = 0
        
        for alias, column in self.alias_map.items():
            if alias in main_part:
                # Prefer longer, more specific matches
                score = len(alias)
                if score > best_score:
                    best_match = column
                    best_score = score
        
        return best_match

    def _extract_stat_formula(self, lowered: str) -> Optional[Dict[str, Any]]:
        """Extract and parse stat formulas."""
        # Find operators - FIXED: Include division and handle spaces
        ops_pattern = r'[+\-*/]|\s+/\s+|\s+divided\s+by\s+'
        ops_matches = list(re.finditer(ops_pattern, lowered))
        
        if not ops_matches:
            return None
        
        # Extract operators and their positions
        ops = []
        split_positions = [0]
        
        for match in ops_matches:
            op_text = match.group().strip()
            if 'divided by' in op_text or '/' in op_text:
                ops.append('/')
            elif '+' in op_text:
                ops.append('+')
            elif '-' in op_text:
                ops.append('-')
            elif '*' in op_text:
                ops.append('*')
            
            split_positions.append(match.start())
            split_positions.append(match.end())
        
        split_positions.append(len(lowered))
        
        # Extract parts between operators
        parts = []
        for i in range(0, len(split_positions)-1, 2):
            part = lowered[split_positions[i]:split_positions[i+1]].strip()
            if part:
                parts.append(part)
        
        # Clean each part and map to columns
        mapped_cols = []
        safe_vars = []
        safe_map = {}
        
        for part in parts:
            # Remove common prefixes
            for prefix in ["most ", "top ", "highest ", "best ", "players by ", "strikers by "]:
                if part.startswith(prefix):
                    part = part[len(prefix):].strip()
                    break
            
            # Remove numbers at start
            part = re.sub(r'^\d+\s*', '', part)
            
            # Map the part to a column
            mapped_col = self._map_stat_phrase(part)
            if not mapped_col:
                return None  # If we can't map a part, fail the formula
            
            mapped_cols.append(mapped_col)
            safe_var = self._make_safe_var(mapped_col)
            safe_vars.append(safe_var)
            safe_map[safe_var] = mapped_col
        
        # Build expressions
        safe_expr = ''
        display_expr = ''
        for i, (safe_var, mapped_col) in enumerate(zip(safe_vars, mapped_cols)):
            safe_expr += f'safe_map["{safe_var}"]'
            display_expr += f'({mapped_col})'
            if i < len(ops):
                safe_expr += ops[i]
                display_expr += ops[i]
        
        return {
            'expr': safe_expr,
            'display_expr': display_expr,
            'columns': mapped_cols,
            'ops': ops,
            'safe_map': safe_map
        }

    def _extract_stat_value_filters(self, lowered: str, result: Dict[str, Any]) -> None:
        """Extract stat value filters (e.g., 'more than 0.5 goals')."""
        patterns = [
            (r"at least ([\d.]+) (\w+(?:\s+\w+)*)", ">="),
            (r"more than ([\d.]+) (\w+(?:\s+\w+)*)", ">"),
            (r"less than ([\d.]+) (\w+(?:\s+\w+)*)", "<"),
            (r"under ([\d.]+) (\w+(?:\s+\w+)*)", "<"),
            (r"over ([\d.]+) (\w+(?:\s+\w+)*)", ">"),
            (r"exactly ([\d.]+) (\w+(?:\s+\w+)*)", "=="),
            (r"with ([\d.]+)\+ (\w+(?:\s+\w+)*)", ">=")  # "with 0.5+ goals"
        ]
        
        for pattern, op in patterns:
            match = re.search(pattern, lowered)
            if match:
                value = float(match.group(1))
                stat_phrase = match.group(2).strip()
                
                # Map stat phrase to actual column
                mapped_stat = self._map_stat_phrase(stat_phrase)
                if mapped_stat:
                    result["stat"] = mapped_stat
                    result["stat_op"] = op
                    result["stat_value"] = value
                break

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
        
        # FIXED: Add specific league name fuzzy matching
        if any(word in phrase for word in ['serie', 'seria', 'italy']):
            if 'serie a' in self.league_keywords.values() or 'Serie A' in [v for v in self.league_keywords.values()]:
                return None  # This will be handled by league extraction
        
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
        
        # Ensure we have a stat for stat-based queries
        query_type = result.get("query_type")
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
            'tackles': 'Sliding tackles per 90'
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