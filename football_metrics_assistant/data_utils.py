import pandas as pd
import os
from functools import lru_cache
from typing import List, Dict, Tuple

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data.csv')

# Meta columns that are not stats
META_COLUMNS = [
    'Player', 'Team within selected timeframe', 'League', 'Position', 'Age',
    'Minutes played', 'Matches played', 'Team', 'Season', 'Performance Index'
]

@lru_cache(maxsize=1)
def load_data():
    """
    Loads the data.csv file efficiently, infers dtypes, and handles large files.
    Returns a pandas DataFrame.
    """
    df = pd.read_csv(DATA_PATH, low_memory=False)
    # Normalize column names
    df.columns = [col.strip() for col in df.columns]
    return df

@lru_cache(maxsize=1)
def get_all_columns() -> Tuple[List[str], List[str]]:
    """
    Returns all columns as (raw, normalized) lists.
    """
    df = load_data()
    raw = list(df.columns)
    normalized = [normalize_colname(col) for col in raw]
    return raw, normalized

@lru_cache(maxsize=1)
def get_all_players() -> List[str]:
    df = load_data()
    return sorted(df['Player'].dropna().unique())

@lru_cache(maxsize=1)
def get_all_teams() -> List[str]:
    df = load_data()
    col = 'Team within selected timeframe' if 'Team within selected timeframe' in df.columns else 'Team'
    return sorted(df[col].dropna().unique())

@lru_cache(maxsize=1)
def get_all_positions() -> List[str]:
    df = load_data()
    return sorted(df['Position'].dropna().unique())

@lru_cache(maxsize=1)
def get_all_leagues() -> List[str]:
    df = load_data()
    return sorted(df['League'].dropna().unique())

@lru_cache(maxsize=1)
def get_stat_columns() -> List[str]:
    df = load_data()
    return [col for col in df.columns if col not in META_COLUMNS]

@lru_cache(maxsize=1)
def get_categorical_columns(threshold: int = 30) -> List[str]:
    """
    Returns columns with a small number of unique values (likely categorical).
    """
    df = load_data()
    return [col for col in df.columns if df[col].nunique(dropna=True) <= threshold]

@lru_cache(maxsize=1)
def get_numeric_columns() -> List[str]:
    df = load_data()
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]

@lru_cache(maxsize=1)
def get_percent_columns() -> List[str]:
    """
    Returns columns that are likely percentages (by name or value range).
    """
    df = load_data()
    percent_cols = [col for col in df.columns if '%' in col or 'percent' in col.lower()]
    # Also check value range (0-1 or 0-100)
    for col in get_numeric_columns():
        if col not in percent_cols:
            vals = df[col].dropna()
            if not vals.empty and ((vals.between(0, 1).mean() > 0.8) or (vals.between(0, 100).mean() > 0.8)):
                percent_cols.append(col)
    return sorted(set(percent_cols))

@lru_cache(maxsize=1)
def get_per_90_columns() -> List[str]:
    return [col for col in get_stat_columns() if 'per 90' in col.lower()]

@lru_cache(maxsize=1)
def get_per_100_columns() -> List[str]:
    return [col for col in get_stat_columns() if 'per 100' in col.lower()]

def normalize_colname(col: str) -> str:
    return col.strip().lower().replace(' ', '').replace('-', '').replace('+', 'plus').replace('/', '').replace('(', '').replace(')', '')

@lru_cache(maxsize=1)
def get_alias_to_column_map() -> Dict[str, str]:
    """
    Returns a mapping from all possible user aliases (normalized) to real column names, including fuzzy/partial matches and auto-generated forms.
    """
    stat_cols = get_stat_columns()
    alias_map = {}
    for col in stat_cols:
        norm_col = normalize_colname(col)
        # Add the normalized column name
        alias_map[norm_col] = col
        # Add the original column name (lowercase, stripped)
        alias_map[col.strip().lower()] = col
        # Add underscores
        alias_map[col.strip().lower().replace(' ', '_')] = col
        # Add no spaces
        alias_map[col.strip().lower().replace(' ', '')] = col
        # Add dashes/plus/minus replaced
        alias_map[col.strip().lower().replace(' ', '').replace('-', '').replace('+', 'plus').replace('/', '').replace('(', '').replace(')', '')] = col
        # Add per90/per100 abbreviations
        if 'per 90' in col.lower():
            base = col.lower().replace('per 90', '').strip().replace(' ', '')
            alias_map[base + '90'] = col
            alias_map[base + 'p90'] = col
            # e.g., assists per 90 -> a90
            if 'assist' in base:
                alias_map['a90'] = col
            if 'goal' in base:
                alias_map['g90'] = col
            if 'xg' in base:
                alias_map['xg90'] = col
            if 'xa' in base:
                alias_map['xa90'] = col
        if 'per 100' in col.lower():
            base = col.lower().replace('per 100', '').strip().replace(' ', '')
            alias_map[base + '100'] = col
            alias_map[base + 'p100'] = col
        # Add short forms for common stats
        if 'xg' in norm_col:
            alias_map['xg'] = col
        if 'xa' in norm_col:
            alias_map['xa'] = col
        if 'npxg' in norm_col:
            alias_map['npxg'] = col
        if 'goalsassists' in norm_col or 'goals+assists' in norm_col:
            alias_map['goals+assists'] = col
            alias_map['goalsassists'] = col
        if 'possession+/-' in norm_col or 'possessionplusminus' in norm_col or 'poss+/-' in norm_col or 'possplusminus' in norm_col:
            alias_map['poss+/-'] = col
            alias_map['possession+/-'] = col
            alias_map['possplusminus'] = col
            alias_map['possessionplusminus'] = col
            alias_map['poss'] = col
        if 'exitline' in norm_col:
            alias_map['exit line'] = col
            alias_map['exitline'] = col
    # Add default mappings for common football stats (if present)
    default_stat_map = {
        'tackles': 'Tackles per 90',
        'sliding tackles': 'Sliding tackles per 90',
        'assists': 'Assists per 90',
        'duels': 'Duels per 90',
        'interceptions': 'Interceptions per 90',
        'saves': 'Saves per 90',
        'goals': 'Goals per 90',
        'shots': 'Shots per 90',
        'passes': 'Passes per 90',
        'crosses': 'Crosses per 90',
        'dribbles': 'Dribbles attempted per 90',
        'clean sheets': 'Clean sheets',
        'progressive passes': 'Progressive passes per 90',
        'progressive carries': 'Progressive carries per 90',
        'progressive actions': 'Progressive actions per 90',
        'key passes': 'Key passes per 90',
        'possession +/-': 'Possession +/-',
        'poss+/-': 'Possession +/-',
        'goals + assists': 'Goals + Assists per 90',
        'goals+assists': 'Goals + Assists per 90',
        'npxg': 'npxG per 90',
        'xg': 'xG per 90',
        'xa': 'xA per 90',
        'shot assists': 'Shot assists per 90',
        'deep completions': 'Deep completions per 90',
        'through passes': 'Through passes per 90',
        'forward passes': 'Forward passes per 90',
        'long passes': 'Long passes per 90',
        'short passes': 'Short passes per 90',
        'aerial duels': 'Aerial duels per 90',
        'defensive duels': 'Defensive duels per 90',
        'blocks': 'Blocks per 90',
        'clearances': 'Clearances per 90',
        'touches': 'Touches per 90',
        'fouls': 'Fouls suffered per 90',
        'penalties': 'Penalties attempted',
        'penalty success': 'Penalty success rate %',
        'save percentage': 'Save percentage %.1',
        'shot accuracy': 'Shots on target %.1',
        'pass completion': 'Pass completion %.1',
        'dribble success': 'Dribble success rate %.1',
        'cross accuracy': 'Cross accuracy %.1',
        'duels won': 'Duels won %',
        'offensive duels': 'Offensive duels per 90',
        'defensive duels won': 'Defensive duels won per 90',
        'aerial duels won': 'Aerial duels won per 90',
        'progressive passes completed': 'Progressive passes completed per 90',
        'progressive pass accuracy': 'Progressive pass accuracy %.1',
        'progressive action rate': 'Progressive action rate',
        'ball-carrying frequency': 'Ball-carrying frequency',
        'xg/shot': 'xG/Shot',
        'npxg/shot': 'npxG/Shot',
        'xg+xA': 'xG+xA per 90',
        'npxg+xa': 'npxG+xA per 90',
        'goals per xg': 'Goals per xG',
        'assists per xa': 'Assists per xA',
        'goals - xg': 'Goals - xG per 90',
        'assists - xa': 'Assists - xA per 90',
        'matches': 'Matches played',
        'minutes': 'Minutes played',
        'age': 'Age',
    }
    for user_alias, colname in default_stat_map.items():
        if colname in stat_cols:
            alias_map[user_alias] = colname
    # Debug: print all aliases for inspection
    print("[DEBUG] Stat alias map (sample):", dict(list(alias_map.items())[:20]))
    return alias_map

@lru_cache(maxsize=1)
def get_column_to_aliases_map() -> Dict[str, List[str]]:
    """
    Returns a mapping from real column names to all their possible aliases (normalized).
    """
    alias_map = get_alias_to_column_map()
    col_to_aliases = {}
    for alias, col in alias_map.items():
        col_to_aliases.setdefault(col, []).append(alias)
    return col_to_aliases

@lru_cache(maxsize=1)
def get_normalized_column_list() -> List[str]:
    """
    Returns all column names normalized for fuzzy matching.
    """
    df = load_data()
    return [normalize_colname(col) for col in df.columns]

@lru_cache(maxsize=1)
def get_column_type_map() -> Dict[str, List[str]]:
    """
    Returns a dict mapping each column to its type(s): meta, stat, categorical, numeric, percent, per_90, per_100, etc.
    """
    df = load_data()
    type_map = {}
    cat_cols = set(get_categorical_columns())
    num_cols = set(get_numeric_columns())
    percent_cols = set(get_percent_columns())
    per_90_cols = set(get_per_90_columns())
    per_100_cols = set(get_per_100_columns())
    for col in df.columns:
        types = []
        if col in META_COLUMNS:
            types.append('meta')
        if col in cat_cols:
            types.append('categorical')
        if col in num_cols:
            types.append('numeric')
        if col in percent_cols:
            types.append('percent')
        if col in per_90_cols:
            types.append('per_90')
        if col in per_100_cols:
            types.append('per_100')
        if col not in META_COLUMNS:
            types.append('stat')
        type_map[col] = types
    return type_map 