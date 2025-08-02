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
    return col.strip().lower().replace(' ', '').replace('-', '').replace('+', 'plus').replace('/', '').replace('(', '').replace(')', '').replace('%', 'pct').replace('.', '')

@lru_cache(maxsize=1)
def get_alias_to_column_map() -> Dict[str, str]:
    """
    Returns a mapping from all possible user aliases (normalized) to real column names.
    This is now based on your ACTUAL dataset columns.
    """
    stat_cols = get_stat_columns()
    alias_map = {}
    
    # First, add all existing columns with their variations
    for col in stat_cols:
        norm_col = normalize_colname(col)
        
        # Add normalized versions
        alias_map[norm_col] = col
        alias_map[col.strip().lower()] = col
        alias_map[col.strip().lower().replace(' ', '_')] = col
        alias_map[col.strip().lower().replace(' ', '')] = col
        
        # Add per90/per100 abbreviations
        if 'per 90' in col.lower():
            base = col.lower().replace('per 90', '').strip().replace(' ', '')
            alias_map[base + '90'] = col
            alias_map[base + 'p90'] = col
            
        if 'per 100' in col.lower():
            base = col.lower().replace('per 100', '').strip().replace(' ', '')
            alias_map[base + '100'] = col
            alias_map[base + 'p100'] = col
    
    # Now add specific mappings based on your ACTUAL columns
    column_mappings = {
        # Goals
        'goals': 'Goals per 90',
        'goal': 'Goals per 90',
        'goals per 90': 'Goals per 90',
        'g90': 'Goals per 90',
        'goles': 'Goals per 90',  # Handle typo from test
        
        # Non-penalty goals
        'non penalty goals': 'Non-penalty goals per 90',
        'npg': 'Non-penalty goals per 90',
        'non-penalty goals': 'Non-penalty goals per 90',
        
        # Assists
        'assists': 'Assists per 90',
        'assist': 'Assists per 90',
        'assists per 90': 'Assists per 90',
        'a90': 'Assists per 90',
        'assits': 'Assists per 90',  # Handle typo from test
        
        # xG (Expected Goals)
        'xg': 'xG per 90',
        'xg per 90': 'xG per 90',
        'expected goals': 'xG per 90',
        'xg90': 'xG per 90',
        'highest xg': 'xG per 90',
        
        # npxG (Non-penalty Expected Goals)
        'npxg': 'npxG per 90',
        'npxg per 90': 'npxG per 90',
        'non penalty xg': 'npxG per 90',
        'non-penalty xg': 'npxG per 90',
        'non penalty expected goals': 'npxG per 90',
        
        # xA (Expected Assists)
        'xa': 'xA per 90',
        'xa per 90': 'xA per 90',
        'expected assists': 'xA per 90',
        'xa90': 'xA per 90',
        'highest xa': 'xA per 90',
        
        # Goals + Assists
        'goals+assists': 'Goals + Assists per 90',
        'goals + assists': 'Goals + Assists per 90',
        'g+a': 'Goals + Assists per 90',
        'goal contributions': 'Goals + Assists per 90',
        'goals and assists': 'Goals + Assists per 90',
        'goal+assist': 'Goals + Assists per 90',
        'contributions': 'Goals + Assists per 90',
        'goals plus assists': 'Goals + Assists per 90',
        
        # NPG + A
        'npg+a': 'NPG+A per 90',
        'non penalty goals + assists': 'NPG+A per 90',
        
        # xG + xA
        'xg+xa': 'xG+xA per 90',
        'xg + xa': 'xG+xA per 90',
        'expected goals + assists': 'xG+xA per 90',
        
        # npxG + xA
        'npxg+xa': 'npxG+xA per 90',
        'npxg + xa': 'npxG+xA per 90',
        
        # Other attacking stats
        'shots': 'Shots per 90',
        'shots per 90': 'Shots per 90',
        'shots on target': 'Shots on target per 90',
        'headed goals': 'Headed goals per 90',
        'touches in box': 'Touches in box per 90',
        
        # Passing stats
        'passes': 'Passes per 90',
        'passes per 90': 'Passes per 90',
        'key passes': 'Key passes per 90',
        'progressive passes': 'Progressive passes per 90',
        'forward passes': 'Forward passes per 90',
        'long passes': 'Long passes per 90',
        'short passes': 'Short passes per 90',
        'through passes': 'Through passes per 90',
        'crosses': 'Crosses per 90',
        'deep completions': 'Deep completions per 90',
        'shot assists': 'Shot assists per 90',
        
        # Defensive stats
        'tackles': 'Sliding tackles per 90',
        'sliding tackles': 'Sliding tackles per 90',
        'interceptions': 'Interceptions per 90',
        'blocks': 'Shots blocked per 90',
        'defensive duels': 'Defensive duels per 90',
        'aerial duels': 'Aerial duels per 90',
        'duels': 'Duels per 90',
        
        # Possession stats
        'possession +/-': 'Possession +/-',
        'poss+/-': 'Possession +/-',
        'possessions won': 'Possessions won per 90',
        'touches': 'Touches per 90',
        'dribbles': 'Dribbles attempted per 90',
        'progressive carries': 'Progressive carries per 90',
        'progressive actions': 'Progressive actions per 90',
        
        # Goalkeeper stats
        'clean sheets': 'Clean sheets',
        'saves': 'Saves per 90',
        'xg conceded': 'xG conceded per 90',
        'prevented goals': 'Prevented goals per 90',
        
        # Percentages and ratios
        'shot accuracy': 'Shots on target %.1',
        'pass completion': 'Pass completion %.1',
        'dribble success': 'Dribble success rate %.1',
        'cross accuracy': 'Cross accuracy %.1',
        'save percentage': 'Save percentage %.1',
        'duels won': 'Duels won %',
        'aerial duels won': 'Aerial duels won %.1',
        'defensive duels won': 'Defensive duels won %.1',
        'goals per xg': 'Goals per xG',
        'assists per xa': 'Assists per xA',
        
        # Meta columns
        'age': 'Age',
        'minutes': 'Minutes played',
        'matches': 'Matches played',
        'minutes played': 'Minutes played',
        'matches played': 'Matches played',
    }
    
    # Only add mappings for columns that actually exist in your dataset
    for alias, target_col in column_mappings.items():
        if target_col in stat_cols or target_col in META_COLUMNS:
            alias_map[alias] = target_col
            # Also add the normalized version
            norm_alias = normalize_colname(alias)
            alias_map[norm_alias] = target_col
    
    print(f"[DEBUG] Created {len(alias_map)} alias mappings")
    print(f"[DEBUG] Sample mappings: xg -> {alias_map.get('xg', 'NOT FOUND')}")
    print(f"[DEBUG] Sample mappings: npxg -> {alias_map.get('npxg', 'NOT FOUND')}")
    print(f"[DEBUG] Sample mappings: assists -> {alias_map.get('assists', 'NOT FOUND')}")
    
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

def debug_data_info():
    """Print detailed information about the dataset structure"""
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Total columns: {len(df.columns)}")
    
    print("\n=== META COLUMNS ===")
    meta_found = [col for col in df.columns if col in META_COLUMNS]
    meta_missing = [col for col in META_COLUMNS if col not in df.columns]
    print(f"Found: {meta_found}")
    print(f"Missing: {meta_missing}")
    
    print("\n=== KEY STAT COLUMNS ===")
    key_stats = [
        'Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90', 
        'npxG per 90', 'Goals + Assists per 90', 'Clean sheets'
    ]
    for stat in key_stats:
        if stat in df.columns:
            non_null = df[stat].count()
            total = len(df)
            print(f"✓ {stat} ({non_null}/{total} non-null)")
        else:
            print(f"✗ {stat} - MISSING")
    
    print("\n=== ALIAS MAPPING TEST ===")
    alias_map = get_alias_to_column_map()
    test_aliases = ['xg', 'npxg', 'xa', 'assists', 'goals', 'goals+assists']
    for alias in test_aliases:
        mapped = alias_map.get(alias, 'NOT FOUND')
        exists = mapped in df.columns if mapped != 'NOT FOUND' else False
        status = "✓" if exists else "✗"
        print(f"{status} '{alias}' -> '{mapped}' (exists: {exists})")