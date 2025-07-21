import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from football_metrics_assistant.data_utils import load_data, get_stat_columns

# Set style for better-looking charts
plt.style.use('default')
sns.set_palette("husl")

print("[DEBUG] filter_players called")

def filter_players(preprocessed_hints: Dict[str, Any]) -> pd.DataFrame:
    """
    Filters the DataFrame based on preprocessed hints (team, position, age, league, etc.).
    Returns filtered DataFrame and applied filters summary.
    """
    print("[DEBUG] filter_players called (start)")
    df = load_data()
    print("[DEBUG] DataFrame shape after load:", df.shape)
    print("[DEBUG] Unique leagues in data:", df['League'].unique())
    original_count = len(df)
    applied_filters = []
    print(f"[DEBUG] Initial player count: {len(df)}")
    if df.empty:
        print("[DEBUG] DataFrame is EMPTY after load!")
    # Apply filters based on preprocessed hints
    if preprocessed_hints.get('team'):
        team = preprocessed_hints['team']
        if isinstance(team, list):
            team = team[0]  # Take first team if multiple
        df = df[df['Team within selected timeframe'] == team]
        applied_filters.append(f"Team: {team}")
        print(f"[DEBUG] After team filter: {len(df)} players, shape: {df.shape}")
        if df.empty:
            print("[DEBUG] DataFrame is EMPTY after team filter!")
    if preprocessed_hints.get('position'):
        position = preprocessed_hints['position']
        print(f"[DEBUG] Position filter value: {position}")
        print(f"[DEBUG] Unique positions before filter: {df['Position'].unique()}")
        if isinstance(position, list):
            df = df[df['Position'].isin(position)]
        else:
            df = df[df['Position'] == position]
        print(f"[DEBUG] Unique positions after filter: {df['Position'].unique()}")
        applied_filters.append(f"Position: {position}")
        print(f"[DEBUG] After position filter: {len(df)} players, shape: {df.shape}")
        if df.empty:
            print("[DEBUG] DataFrame is EMPTY after position filter!")
    if preprocessed_hints.get('league'):
        league = preprocessed_hints['league']
        if isinstance(league, list):
            league = league[0]  # Take first league if multiple (highest priority)
        df = df[df['League'] == league]
        applied_filters.append(f"League: {league}")
        print(f"[DEBUG] After league filter: {len(df)} players, shape: {df.shape}")
        if df.empty:
            print("[DEBUG] DataFrame is EMPTY after league filter!")
    if preprocessed_hints.get('age_filter'):
        age_filter = preprocessed_hints['age_filter']
        op = age_filter['op']
        value = age_filter['value']
        if op == '<':
            df = df[df['Age'] < value]
        elif op == '>':
            df = df[df['Age'] > value]
        elif op == '==':
            df = df[df['Age'] == value]
        applied_filters.append(f"Age {op} {value}")
        print(f"[DEBUG] After age filter: {len(df)} players, shape: {df.shape}")
        if df.empty:
            print("[DEBUG] DataFrame is EMPTY after age filter!")
    if preprocessed_hints.get('player'):
        player = preprocessed_hints['player']
        if isinstance(player, list):
            player = player[0]  # Take first player if multiple
        df = df[df['Player'] == player]
        applied_filters.append(f"Player: {player}")
        print(f"[DEBUG] After player filter: {len(df)} players, shape: {df.shape}")
        if df.empty:
            print("[DEBUG] DataFrame is EMPTY after player filter!")
    # Filter out players with very few minutes (less than 270 minutes = 3 full games)
    df = df[df['Minutes played'] >= 270]
    applied_filters.append("Minimum 270 minutes played")
    print(f"[DEBUG] After minutes filter: {len(df)} players, shape: {df.shape}")
    if df.empty:
        print("[DEBUG] DataFrame is EMPTY after minutes filter!")
    # Special case: if stat is goalkeeper-specific, filter to goalkeepers
    stat = preprocessed_hints.get('stat')
    if isinstance(stat, list):
        stat = stat[0]
    stat = stat.lower() if stat else ''
    gk_stats = [
        'clean sheets',
        'saves per 90',
        'save percentage %.1',
        'xg conceded per 90',
    ]
    if stat in [s.lower() for s in gk_stats]:
        if 'Position' in df.columns:
            df = df[df['Position'] == 'Goalkeeper']
            applied_filters.append('Position: Goalkeeper (auto for goalkeeper stat)')
            print(f"[DEBUG] After auto-goalkeeper filter: {len(df)} players")
    
    return df, applied_filters, original_count

def sort_and_limit(df: pd.DataFrame, stat: str, top_n: int = 5) -> pd.DataFrame:
    """
    Sorts DataFrame by a stat and returns top N players.
    """
    if stat not in df.columns:
        raise ValueError(f"Stat '{stat}' not found in data. Available stats: {list(df.columns)}")
    
    if not pd.api.types.is_numeric_dtype(df[stat]):
        raise ValueError(f"Stat '{stat}' is not numeric. Column dtype: {df[stat].dtype}")
    
    if df[stat].dropna().empty:
        raise ValueError(f"Stat '{stat}' has no valid (non-NaN) values for the filtered players.")

    # Sort by the stat (descending for most stats, ascending for some)
    ascending_stats = ['Age', 'Minutes played', 'Matches played']  # Lower is better
    ascending = stat in ascending_stats
    
    return df.nlargest(top_n, stat) if not ascending else df.nsmallest(top_n, stat)

def generate_chart(df: pd.DataFrame, stat: str, top_n: int = 5, chart_type: str = 'bar') -> Tuple[plt.Figure, str]:
    """
    Generates a chart for the top N players in a given stat.
    Returns the figure and a description of the chart.
    """
    if df.empty:
        raise ValueError("No data to plot")
    
    # Sort and get top N
    top_players = sort_and_limit(df, stat, top_n)
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if chart_type == 'bar':
        bars = ax.bar(range(len(top_players)), top_players[stat])
        ax.set_xticks(range(len(top_players)))
        ax.set_xticklabels(top_players['Player'], rotation=45, ha='right')
        ax.set_ylabel(stat)
        ax.set_title(f"Top {top_n} Players by {stat}")
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Add team information
        for i, (_, player) in enumerate(top_players.iterrows()):
            team = player.get('Team within selected timeframe', 'Unknown')
            ax.text(i, ax.get_ylim()[0], team, ha='center', va='top', 
                   fontsize=8, alpha=0.7)
    
    elif chart_type == 'horizontal_bar':
        bars = ax.barh(range(len(top_players)), top_players[stat])
        ax.set_yticks(range(len(top_players)))
        ax.set_yticklabels(top_players['Player'])
        ax.set_xlabel(stat)
        ax.set_title(f"Top {top_n} Players by {stat}")
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.2f}', ha='left', va='center')
    
    plt.tight_layout()
    
    # Generate description
    description = f"Chart showing top {top_n} players by {stat}. "
    if not top_players.empty:
        best_player = top_players.iloc[0]
        description += f"Best performer: {best_player['Player']} ({best_player['Team within selected timeframe']}) with {best_player[stat]:.2f} {stat}."
    
    return fig, description

def get_player_summary(df: pd.DataFrame, stat: str) -> Dict[str, Any]:
    """
    Get summary statistics for a given stat.
    """
    if df.empty:
        return {"error": "No data available"}
    
    if stat not in df.columns:
        return {"error": f"Stat '{stat}' not found"}
    
    summary = {
        "stat": stat,
        "count": len(df),
        "mean": df[stat].mean(),
        "median": df[stat].median(),
        "std": df[stat].std(),
        "min": df[stat].min(),
        "max": df[stat].max(),
        "top_5": df.nlargest(5, stat)[['Player', 'Team within selected timeframe', stat]].to_dict('records')
    }
    
    return summary

def analyze_query(preprocessed_hints: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to analyze a query and return results.
    """
    try:
        # Filter players
        df, applied_filters, original_count = filter_players(preprocessed_hints)
        print(f"[DEBUG] After all filters: {len(df)} players")
        
        if df.empty:
            return {
                "error": "No players found matching the criteria",
                "applied_filters": applied_filters,
                "original_count": original_count
            }
        
        # Get stat for analysis
        stat = preprocessed_hints.get('stat')
        if not stat:
            return {
                "error": "No stat specified for analysis",
                "filtered_data": df[['Player', 'Team within selected timeframe', 'Position', 'Age', 'League']].head(10).to_dict('records'),
                "applied_filters": applied_filters,
                "count": len(df)
            }
        
        # Handle case where stat is a list
        if isinstance(stat, list):
            stat = stat[0]  # Take the first stat if multiple
        
        print(f"[DEBUG] Stat column: {stat}")
        print(f"[DEBUG] Stat values: {df[stat].head(10).to_list() if stat in df.columns else 'N/A'}")
        
        # Get top N
        top_n = preprocessed_hints.get('top_n', 5)
        
        # Sort and get top players
        try:
            top_players_df = sort_and_limit(df, stat, top_n)
        except Exception as e:
            return {
                "error": f"Stat error: {str(e)}",
                "applied_filters": applied_filters,
                "count": len(df)
            }
        
        # Generate chart
        fig, chart_description = generate_chart(df, stat, top_n)
        
        # Get summary
        summary = get_player_summary(df, stat)
        # Convert all summary values to Python types
        summary = {k: to_python_type(v) for k, v in summary.items()}
        return {
            "success": True,
            "top_players": [
                {k: to_python_type(v) for k, v in player.items()}
                for player in top_players_df.to_dict('records')
            ],
            "summary": summary,
            "chart_description": chart_description,
            "applied_filters": applied_filters,
            "original_count": original_count,
            "filtered_count": len(df),
            "stat": stat,
            "top_n": top_n,
            "count": len(df)
        }
        
    except Exception as e:
        print(f"[DEBUG] Analysis failed: {str(e)}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "preprocessed_hints": preprocessed_hints
        } 

def to_python_type(val):
    import numpy as np
    if isinstance(val, np.generic):
        val = val.item()
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
    return val 