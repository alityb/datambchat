import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
from football_metrics_assistant.data_utils import load_data, get_stat_columns

# Set style for better-looking charts
plt.style.use('default')
sns.set_palette("husl")

def filter_players(preprocessed_hints: Dict[str, Any]) -> pd.DataFrame:
    """
    Filters the DataFrame based on preprocessed hints (team, position, age, league, etc.).
    Returns filtered DataFrame and applied filters summary.
    """
    df = load_data()
    original_count = len(df)
    applied_filters = []
    
    # Apply filters based on preprocessed hints
    if preprocessed_hints.get('team'):
        team = preprocessed_hints['team']
        if isinstance(team, list):
            team = team[0]  # Take first team if multiple
        df = df[df['Team within selected timeframe'] == team]
        applied_filters.append(f"Team: {team}")
    
    if preprocessed_hints.get('position'):
        position = preprocessed_hints['position']
        if isinstance(position, list):
            position = position[0]  # Take first position if multiple
        df = df[df['Position'] == position]
        applied_filters.append(f"Position: {position}")
    
    if preprocessed_hints.get('league'):
        league = preprocessed_hints['league']
        if isinstance(league, list):
            league = league[0]  # Take first league if multiple
        df = df[df['League'] == league]
        applied_filters.append(f"League: {league}")
    
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
    
    if preprocessed_hints.get('player'):
        player = preprocessed_hints['player']
        if isinstance(player, list):
            player = player[0]  # Take first player if multiple
        df = df[df['Player'] == player]
        applied_filters.append(f"Player: {player}")
    
    # Filter out players with very few minutes (less than 270 minutes = 3 full games)
    df = df[df['Minutes played'] >= 270]
    applied_filters.append("Minimum 270 minutes played")
    
    # Special case: if stat is goalkeeper-specific, filter to goalkeepers
    stat = preprocessed_hints.get('stat', '').lower() if preprocessed_hints.get('stat') else ''
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
    
    return df, applied_filters, original_count

def sort_and_limit(df: pd.DataFrame, stat: str, top_n: int = 5) -> pd.DataFrame:
    """
    Sorts DataFrame by a stat and returns top N players.
    """
    if stat not in df.columns:
        raise ValueError(f"Stat '{stat}' not found in data. Available stats: {list(df.columns)}")
    
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
        
        # Get top N
        top_n = preprocessed_hints.get('top_n', 5)
        
        # Sort and get top players
        top_players = sort_and_limit(df, stat, top_n)
        
        # Generate chart
        fig, chart_description = generate_chart(df, stat, top_n)
        
        # Get summary
        summary = get_player_summary(df, stat)
        
        return {
            "success": True,
            "top_players": top_players[['Player', 'Team within selected timeframe', 'Position', 'Age', stat]].to_dict('records'),
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
        return {
            "error": f"Analysis failed: {str(e)}",
            "preprocessed_hints": preprocessed_hints
        } 