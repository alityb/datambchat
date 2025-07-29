import re
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
            df = df[df['League'].isin(league)]
            applied_filters.append(f"Leagues: {league}")
            print(f"[DEBUG] After multi-league filter: {len(df)} players, shape: {df.shape}, leagues: {league}")
        else:
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

    # NEW: Minutes filter (apply BEFORE the default 270 minutes filter)
    if preprocessed_hints.get('minutes_value') is not None:
        minutes_op = preprocessed_hints.get('minutes_op', '>=')
        minutes_value = preprocessed_hints['minutes_value']
        
        if minutes_op == '>=':
            df = df[df['Minutes played'] >= minutes_value]
        elif minutes_op == '>':
            df = df[df['Minutes played'] > minutes_value]
        elif minutes_op == '<=':
            df = df[df['Minutes played'] <= minutes_value]
        elif minutes_op == '<':
            df = df[df['Minutes played'] < minutes_value]
        elif minutes_op == '==':
            df = df[df['Minutes played'] == minutes_value]
        
        applied_filters.append(f"Minutes played {minutes_op} {minutes_value}")
        print(f"[DEBUG] After custom minutes filter: {len(df)} players, shape: {df.shape}")
        if df.empty:
            print("[DEBUG] DataFrame is EMPTY after custom minutes filter!")
    else:
        # Apply default minimum 270 minutes only if no custom minutes filter
        df = df[df['Minutes played'] >= 270]
        applied_filters.append("Minimum 270 minutes played")
        print(f"[DEBUG] After default minutes filter: {len(df)} players, shape: {df.shape}")
        if df.empty:
            print("[DEBUG] DataFrame is EMPTY after default minutes filter!")

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
            print(f"[DEBUG] After auto-goalkeeper filter: {len(df)} players, shape: {df.shape}")
            if df.empty:
                print("[DEBUG] DataFrame is EMPTY after auto-goalkeeper filter!")

    # Stat value filter (e.g., Goals per 90 >= 0.5)
    if preprocessed_hints.get('stat') and preprocessed_hints.get('stat_value') is not None:
        stat = preprocessed_hints['stat']
        op = preprocessed_hints.get('stat_op', '>=')
        value = preprocessed_hints['stat_value']
        if stat in df.columns:
            if op == '>=':
                df = df[df[stat] >= value]
            elif op == '>':
                df = df[df[stat] > value]
            elif op == '<=':
                df = df[df[stat] <= value]
            elif op == '<':
                df = df[df[stat] < value]
            elif op == '==':
                df = df[df[stat] == value]
            applied_filters.append(f"{stat} {op} {value}")
            print(f"[DEBUG] After stat value filter: {len(df)} players, shape: {df.shape}")
            if df.empty:
                print("[DEBUG] DataFrame is EMPTY after stat value filter!")

    # After all filters:
    print("[DEBUG] Filtered DataFrame after all filters:")
    print(df)
    if df.empty:
        print("[DEBUG] DataFrame is EMPTY after all filters!")
    
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

def get_stat_definition_text(stat_name: str) -> str:
    """
    Get a simple explanation of what a stat means.
    """
    # Comprehensive stat definitions
    stat_definitions = {
        # Attacking stats
        'Goals per 90': 'The average number of goals a player scores per 90 minutes of play. This is the primary measure of a striker\'s finishing ability.',
        'Assists per 90': 'The average number of assists a player provides per 90 minutes. An assist is recorded when a player makes the final pass before a teammate scores.',
        'xG per 90': 'Expected Goals per 90 minutes. This measures the quality of scoring chances a player gets, based on factors like shot distance, angle, and defensive pressure. Higher values indicate better positioning and chance creation.',
        'xA per 90': 'Expected Assists per 90 minutes. This measures the quality of passes a player makes that lead to shots, regardless of whether the shot results in a goal.',
        'npxG per 90': 'Non-Penalty Expected Goals per 90 minutes. This is xG excluding penalty kicks, giving a better measure of open-play scoring ability.',
        'Goals + Assists per 90': 'Combined goals and assists per 90 minutes. This measures a player\'s total direct contribution to their team\'s scoring.',
        'Shots per 90': 'The average number of shots a player takes per 90 minutes, including both on-target and off-target attempts.',
        'Shots on target %.1': 'The percentage of a player\'s shots that are on target (would go in the goal if not saved). Higher percentages indicate better shot accuracy.',
        'Goals per xG': 'The ratio of actual goals scored to expected goals. Values above 1.0 suggest clinical finishing, while below 1.0 may indicate poor finishing or bad luck.',
        'npxG/Shot': 'Non-penalty expected goals per shot. This measures the average quality of a player\'s shooting chances.',
        'xG/Shot': 'Expected goals per shot. This indicates how good the average quality of a player\'s shots are.',
        
        # Passing stats
        'Passes per 90': 'The average number of passes a player completes per 90 minutes. Higher values often indicate involvement in team build-up play.',
        'Pass completion %.1': 'The percentage of passes that reach a teammate successfully. Higher percentages indicate more accurate passing.',
        'Key passes per 90': 'Passes that directly lead to a shot attempt by a teammate per 90 minutes. This measures chance creation ability.',
        'Progressive passes per 90': 'Passes that move the ball significantly closer to the opponent\'s goal (typically 10+ meters forward) per 90 minutes.',
        'Long passes per 90': 'Long-distance passes (typically 25+ meters) attempted per 90 minutes, often used to switch play or find forwards.',
        'Through passes per 90': 'Passes played behind the defensive line to a teammate per 90 minutes. These are particularly dangerous attacking passes.',
        'Cross accuracy %.1': 'The percentage of crosses that reach a teammate. Higher values indicate better crossing ability.',
        'Crosses per 90': 'The average number of crosses a player attempts per 90 minutes, typically from wide positions.',
        
        # Defensive stats
        'Tackles per 90': 'The average number of tackles a player makes per 90 minutes. This includes both successful challenges for the ball.',
        'Sliding tackles per 90': 'The average number of sliding tackles attempted per 90 minutes. This indicates aggressive defensive play.',
        'Interceptions per 90': 'Times per 90 minutes a player intercepts an opponent\'s pass. This shows defensive awareness and positioning.',
        'Clearances per 90': 'Defensive actions per 90 minutes where a player kicks, heads, or throws the ball away from their defensive area.',
        'Blocks per 90': 'Times per 90 minutes a player blocks an opponent\'s shot or pass with their body.',
        'Aerial duels per 90': 'Aerial challenges per 90 minutes, important for defenders and strikers in winning headers.',
        'Aerial duels won per 90': 'Aerial challenges won per 90 minutes. Shows dominance in the air.',
        'Defensive duels per 90': 'One-on-one defensive challenges per 90 minutes, including tackles and physical contests.',
        'Duels won %': 'Percentage of all duels (aerial, defensive, offensive) that a player wins. Higher values indicate physical dominance.',
        
        # Possession stats
        'Touches per 90': 'The total number of times a player touches the ball per 90 minutes. Higher values indicate more involvement in play.',
        'Poss+/-': 'Possession Plus/Minus. The difference between possessions won and lost per 90 minutes. Positive values help team keep the ball.',
        'Ball-carrying frequency': 'How often a player carries the ball forward during their possessions.',
        'Progressive carries per 90': 'Times per 90 minutes a player carries the ball significantly forward (typically 5+ meters toward goal).',
        'Dribbles attempted per 90': 'Number of times per 90 minutes a player attempts to beat an opponent with the ball.',
        'Dribble success rate %.1': 'Percentage of dribble attempts that are successful. Higher values indicate better 1v1 ability.',
        
        # Goalkeeper stats
        'Clean sheets': 'Number of matches where the goalkeeper\'s team doesn\'t concede a goal while they\'re playing.',
        'Saves per 90': 'Average number of saves a goalkeeper makes per 90 minutes.',
        'Save percentage %.1': 'Percentage of shots on target that the goalkeeper saves. Higher values indicate better shot-stopping.',
        'xG conceded per 90': 'Expected goals conceded per 90 minutes based on the quality of shots faced.',
        
        # Physical stats
        'Fouls suffered per 90': 'Number of times per 90 minutes a player is fouled by opponents.',
        'Yellow cards': 'Total number of yellow cards received by the player.',
        'Red cards': 'Total number of red cards received by the player.',
        
        # Match info
        'Minutes played': 'Total minutes the player has been on the field.',
        'Matches played': 'Total number of matches the player has appeared in.',
        'Age': 'The player\'s current age.',
        
        # Advanced stats
        'Performance Index': 'A composite score that evaluates overall player performance across multiple metrics.',
        'Progressive actions per 90': 'Combined progressive passes and carries per 90 minutes. Measures forward ball progression.',
        'Shot assists per 90': 'Passes that lead to a shot (but not necessarily on target) per 90 minutes.',
        'Deep completions per 90': 'Successful passes into the opponent\'s penalty area per 90 minutes.',
        'Exit line': 'A defensive metric measuring how effectively a player prevents opponents from progressing into dangerous areas.',
    }
    
    # Try exact match first
    if stat_name in stat_definitions:
        return stat_definitions[stat_name]
    
    # Try case-insensitive match
    for stat, definition in stat_definitions.items():
        if stat.lower() == stat_name.lower():
            return stat_definitions[stat]
    
    # Try partial match
    for stat, definition in stat_definitions.items():
        if stat_name.lower() in stat.lower() or stat.lower() in stat_name.lower():
            return stat_definitions[stat]
    
    # Default explanation
    return f"This statistic measures a specific aspect of player performance. The exact calculation may vary, but it provides insight into the player's contribution in this area."

def generate_stat_definition_report(stat_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive stat definition report with explanation and top players.
    """
    df = load_data()
    
    # Check if stat exists in the dataset
    if stat_name not in df.columns:
        # Try to find similar stat names
        from difflib import get_close_matches
        similar_stats = get_close_matches(stat_name, df.columns, n=3, cutoff=0.6)
        return {
            "error": f"Statistic '{stat_name}' not found in dataset.",
            "suggestions": similar_stats if similar_stats else []
        }
    
    # Get definition
    definition = get_stat_definition_text(stat_name)
    
    # Filter out players with insufficient minutes and NaN values
    filtered_df = df[
        (df['Minutes played'] >= 270) &  # Minimum minutes filter
        (df[stat_name].notna())  # Remove NaN values
    ].copy()
    
    if filtered_df.empty:
        return {
            "error": f"No players found with valid data for {stat_name}",
            "definition": definition
        }
    
    # Determine if higher or lower values are better
    ascending_stats = ['Age', 'Minutes played', 'Matches played', 'Yellow cards', 'Red cards', 'xG conceded per 90']
    ascending = stat_name in ascending_stats
    
    # Get top 10 players for this stat
    if ascending:
        top_players_df = filtered_df.nsmallest(10, stat_name)
        comparison_text = "lowest"
    else:
        top_players_df = filtered_df.nlargest(10, stat_name)
        comparison_text = "highest"
    
    # Build player list with relevant info
    top_players = []
    for _, player in top_players_df.iterrows():
        player_info = {
            "name": player['Player'],
            "team": player.get('Team within selected timeframe', 'Unknown'),
            "league": player.get('League', 'Unknown'),
            "position": player.get('Position', 'Unknown'),
            "age": to_python_type(player.get('Age', 'Unknown')),
            "minutes": to_python_type(player.get('Minutes played', 0)),
            "matches": to_python_type(player.get('Matches played', 0)),
            "stat_value": to_python_type(player.get(stat_name, 0))
        }
        top_players.append(player_info)
    
    # Calculate some basic stats about this metric
    stat_values = filtered_df[stat_name]
    basic_stats = {
        "total_players": len(filtered_df),
        "mean": to_python_type(stat_values.mean()),
        "median": to_python_type(stat_values.median()),
        "std": to_python_type(stat_values.std()),
        "min": to_python_type(stat_values.min()),
        "max": to_python_type(stat_values.max())
    }
    
    # Position breakdown - which positions excel in this stat
    position_breakdown = filtered_df.groupby('Position')[stat_name].agg(['mean', 'count']).round(2)
    position_breakdown = position_breakdown[position_breakdown['count'] >= 5]  # Only positions with 5+ players
    if not ascending:
        position_breakdown = position_breakdown.sort_values('mean', ascending=False)
    else:
        position_breakdown = position_breakdown.sort_values('mean', ascending=True)
    
    position_insights = []
    for pos, data in position_breakdown.head(3).iterrows():
        position_insights.append({
            "position": pos,
            "average": to_python_type(data['mean']),
            "player_count": to_python_type(data['count'])
        })
    
    return {
        "success": True,
        "stat_name": stat_name,
        "definition": definition,
        "comparison_text": comparison_text,
        "top_players": top_players,
        "basic_stats": basic_stats,
        "position_insights": position_insights
    }

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
        
        query_type = preprocessed_hints.get('query_type', 'TOP_N')

        # --- Formula stat computation ---
        stat_formula = preprocessed_hints.get('stat_formula')
        stat = preprocessed_hints.get('stat')
        computed_stat_col = None
        league = preprocessed_hints.get('league')
        
        # NEW: Multi-league support - but get overall top N, not top N per league
        if isinstance(league, list) and stat:
            top_n = preprocessed_hints.get('top_n', 5)
            
            # Compute the formula stat column ONCE for the entire dataset
            if stat_formula:
                safe_expr = stat_formula['expr']
                display_expr = stat_formula.get('display_expr', safe_expr)
                safe_map = stat_formula.get('safe_map', {})
                columns = stat_formula['columns']
                
                # Check all columns exist
                missing = [col for col in columns if col not in df.columns]
                if missing:
                    print(f"[DEBUG] Formula stat missing columns: {missing}")
                    return {
                        "error": f"Formula stat missing columns: {missing}",
                        "applied_filters": applied_filters,
                        "original_count": original_count
                    }
                
                # Build a safe eval environment for the entire dataset
                local_env = {'safe_map': {}}
                for safe_var, col in safe_map.items():
                    if col in df.columns:
                        local_env['safe_map'][safe_var] = df[col]
                    else:
                        local_env['safe_map'][safe_var] = 0  # or np.nan
                
                try:
                    df[display_expr] = eval(safe_expr, {"__builtins__": None}, local_env)
                    stat_col = display_expr
                    print(f"[DEBUG] Computed formula stat '{display_expr}' for entire multi-league dataset")
                except Exception as e:
                    print(f"[DEBUG] Error computing formula stat: {e}")
                    return {
                        "error": f"Error computing formula stat: {e}",
                        "applied_filters": applied_filters,
                        "original_count": original_count
                    }
            else:
                stat_col = stat
            
            # Now get the overall top N players across ALL leagues
            try:
                # Sort entire dataset and get top N
                top_players_df = sort_and_limit(df, stat_col, top_n)
                
                # Convert to records and add league info
                all_players = []
                for _, player in top_players_df.iterrows():
                    player_dict = player.to_dict()
                    # Clean up the values
                    player_dict = {k: to_python_type(v) for k, v in player_dict.items()}
                    all_players.append(player_dict)
                
                # Build summary stats by league for context
                league_summaries = {}
                for lg in league:
                    df_league = df[df['League'] == lg]
                    if not df_league.empty:
                        league_summaries[lg] = get_player_summary(df_league, stat_col)
                
                # Build overall summary
                overall_summary = get_player_summary(df, stat_col)
                
                def clean_json(obj):
                    if isinstance(obj, dict):
                        return {k: clean_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [clean_json(v) for v in obj]
                    else:
                        return to_python_type(obj)
                
                return {
                    "success": True,
                    "multi_league": True,
                    "top_players": clean_json(all_players),  # Overall top N
                    "summary": clean_json(overall_summary),  # Overall stats
                    "summary_by_league": clean_json(league_summaries),  # League breakdowns for context
                    "applied_filters": applied_filters,
                    "count": len(df),
                    "stat": stat_col
                }
                
            except Exception as e:
                print(f"[DEBUG] Error computing overall top players: {e}")
                return {
                    "error": f"Error computing top players: {e}",
                    "applied_filters": applied_filters,
                    "original_count": original_count
                }
        if stat_formula:
            safe_expr = stat_formula['expr']
            display_expr = stat_formula.get('display_expr', safe_expr)
            columns = stat_formula['columns']
            safe_map = stat_formula.get('safe_map', {})
            ops = stat_formula['ops']
            # Check all columns exist
            missing = [col for col in columns if col not in df.columns]
            if missing:
                print(f"[DEBUG] Formula stat missing columns: {missing}")
                return {
                    "error": f"Formula stat missing columns: {missing}",
                    "applied_filters": applied_filters,
                    "original_count": original_count
                }
            # Compute the formula column
            try:
                # Build a safe eval environment
                local_env = {'safe_map': {}}
                for safe_var, col in safe_map.items():
                    local_env['safe_map'][safe_var] = df[col]
                print(f"[DEBUG] Formula eval local_env keys: {list(local_env['safe_map'].keys())}")
                # Use the display_expr as the new column name
                computed_stat_col = display_expr
                df[computed_stat_col] = eval(safe_expr, {"__builtins__": None}, local_env)
                # Patch: replace inf, -inf, NaN with None
                import numpy as np
                before = df[computed_stat_col].isna().sum()
                df[computed_stat_col] = df[computed_stat_col].replace([np.inf, -np.inf], np.nan)
                after = df[computed_stat_col].isna().sum()
                if after > before:
                    print(f"[DEBUG] Replaced {after - before} inf/-inf values with NaN in computed stat column '{computed_stat_col}'")
                stat = computed_stat_col
            except Exception as e:
                print(f"[DEBUG] Error computing formula stat: {e}")
                return {
                    "error": f"Error computing formula stat: {e}",
                    "applied_filters": applied_filters,
                    "original_count": original_count
                }
        # --- End formula stat computation ---
        

        if query_type == 'COUNT' or query_type == 'FILTER':
            # Always include filtered_data for table rendering
            filtered_data = df[['Player', 'Team within selected timeframe', 'Position', 'Age', 'League']]
            if stat and stat in df.columns:
                filtered_data = filtered_data.copy()
                filtered_data[stat] = df[stat].apply(to_python_type)
            return {
                "success": True,
                "count": len(df),
                "summary": f"There are {len(df)} players matching your criteria.",
                "filtered_data": filtered_data.to_dict('records'),
                "applied_filters": applied_filters
            }
        elif query_type == 'LIST':
            return {
                "success": True,
                "players": df[['Player', 'Team within selected timeframe', 'Position', 'Age', 'League']].to_dict('records'),
                "count": len(df),
                "applied_filters": applied_filters
            }
        elif query_type in ('TOP_N', 'FILTER'):
            # Get stat for analysis
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
        else:
            return {
                "error": "Sorry, I couldn't understand the type of question. Please rephrase.",
                "applied_filters": applied_filters,
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