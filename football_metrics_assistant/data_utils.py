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
    Returns a mapping from all possible user aliases (normalized) to real column names, including fuzzy/partial matches.
    """
    stat_cols = get_stat_columns()
    alias_map = {}
    for col in stat_cols:
        norm_col = normalize_colname(col)
        alias_map[norm_col] = col
        # Add common football stat aliases and partials
        if 'possession+/-' in norm_col or 'possessionplusminus' in norm_col:
            alias_map['poss+/-'] = col
            alias_map['possession+/-'] = col
            alias_map['possplusminus'] = col
        if 'xg' in norm_col:
            alias_map['xg'] = col
        if 'xa' in norm_col:
            alias_map['xa'] = col
        if 'goalsassists' in norm_col:
            alias_map['goals+assists'] = col
            alias_map['goalsassists'] = col
        if 'npxg' in norm_col:
            alias_map['npxg'] = col
        if 'exitline' in norm_col:
            alias_map['exit line'] = col
            alias_map['exitline'] = col
        # Add per90/per100 short aliases
        if 'per90' in norm_col:
            alias_map[norm_col.replace('per90', '') + 'p90'] = col
        if 'per100' in norm_col:
            alias_map[norm_col.replace('per100', '') + 'p100'] = col
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