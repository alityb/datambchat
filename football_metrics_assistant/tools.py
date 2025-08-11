import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher, get_close_matches
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import unicodedata
from football_metrics_assistant.data_utils import load_data, get_stat_columns
import numpy as np

# Set style for better-looking charts
plt.style.use('default')
sns.set_palette("husl")

def to_python_type(val):
    """Convert numpy types to Python native types for JSON serialization."""
    try:
        if val is None:
            return None
        if isinstance(val, np.generic):
            val = val.item()
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                return None
        return val
    except Exception as e:
        print(f"[ERROR] Failed to convert type for value {val}: {e}")
        return None

def filter_players(preprocessed_hints: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str], int]:
    """
    Filters the DataFrame based on preprocessed hints with robust error handling.
    Returns filtered DataFrame, applied filters summary, and original count.
    """
    print("[DEBUG] filter_players called")
    
    try:
        # Add null safety check
        if not preprocessed_hints:
            preprocessed_hints = {}
        
        df = load_data()
        if df is None or df.empty:
            print("[ERROR] Failed to load data or data is empty")
            return pd.DataFrame(), [], 0
        
        print(f"[DEBUG] DataFrame shape after load: {df.shape}")
        original_count = len(df)
        applied_filters = []
        print(f"[DEBUG] Initial player count: {len(df)}")
        
        # Apply team filter
        if preprocessed_hints.get('team'):
            try:
                team = preprocessed_hints['team']
                if isinstance(team, list):
                    team = team[0]
                
                if 'Team within selected timeframe' in df.columns:
                    df = df[df['Team within selected timeframe'] == team]
                    applied_filters.append(f"Team: {team}")
                    print(f"[DEBUG] After team filter: {len(df)} players")
                else:
                    print("[ERROR] 'Team within selected timeframe' column not found")
            except Exception as e:
                print(f"[ERROR] Team filter failed: {e}")

        # Apply position filter
        if preprocessed_hints.get('position'):
            try:
                position = preprocessed_hints['position']
                print(f"[DEBUG] Position filter value: {position}")
                
                if 'Position' in df.columns:
                    if isinstance(position, list):
                        df = df[df['Position'].isin(position)]
                        applied_filters.append(f"Position: {', '.join(position)}")
                    else:
                        df = df[df['Position'] == position]
                        applied_filters.append(f"Position: {position}")
                    print(f"[DEBUG] After position filter: {len(df)} players")
                else:
                    print("[ERROR] 'Position' column not found")
            except Exception as e:
                print(f"[ERROR] Position filter failed: {e}")

        # Apply league filter
        if preprocessed_hints.get('league'):
            try:
                league = preprocessed_hints['league']
                
                if 'League' in df.columns:
                    if isinstance(league, list):
                        df = df[df['League'].isin(league)]
                        applied_filters.append(f"Leagues: {', '.join(league)}")
                    else:
                        df = df[df['League'] == league]
                        applied_filters.append(f"League: {league}")
                    print(f"[DEBUG] After league filter: {len(df)} players")
                else:
                    print("[ERROR] 'League' column not found")
            except Exception as e:
                print(f"[ERROR] League filter failed: {e}")

        # Apply age filter
        if preprocessed_hints.get('age_filter'):
            try:
                age_filter = preprocessed_hints['age_filter']
                op = age_filter['op']
                value = age_filter['value']
                
                if 'Age' in df.columns:
                    if op == '<':
                        df = df[df['Age'] < value]
                    elif op == '>':
                        df = df[df['Age'] > value]
                    elif op == '==':
                        df = df[df['Age'] == value]
                    elif op == '<=':
                        df = df[df['Age'] <= value]
                    elif op == '>=':
                        df = df[df['Age'] >= value]
                    
                    applied_filters.append(f"Age {op} {value}")
                    print(f"[DEBUG] After age filter: {len(df)} players")
                else:
                    print("[ERROR] 'Age' column not found")
            except Exception as e:
                print(f"[ERROR] Age filter failed: {e}")

        # Apply player filter
        if preprocessed_hints.get('player'):
            try:
                player = preprocessed_hints['player']
                if isinstance(player, list):
                    player = player[0]
                
                if 'Player' in df.columns:
                    df = df[df['Player'] == player]
                    applied_filters.append(f"Player: {player}")
                    print(f"[DEBUG] After player filter: {len(df)} players")
                else:
                    print("[ERROR] 'Player' column not found")
            except Exception as e:
                print(f"[ERROR] Player filter failed: {e}")

        # Minutes filter logic
        minutes_filter_applied = False
        
        # Check if custom minutes filter is specified
        if preprocessed_hints.get('minutes_op') and preprocessed_hints.get('minutes_value') is not None:
            try:
                minutes_op = preprocessed_hints['minutes_op']
                minutes_value = preprocessed_hints['minutes_value']
                
                print(f"[DEBUG] Applying custom minutes filter: {minutes_op} {minutes_value}")
                
                if 'Minutes played' in df.columns:
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
                    minutes_filter_applied = True
                    print(f"[DEBUG] After custom minutes filter: {len(df)} players")
                else:
                    print("[ERROR] 'Minutes played' column not found")
            except Exception as e:
                print(f"[ERROR] Custom minutes filter failed: {e}")
        
        # Apply default minimum 270 minutes only if no custom minutes filter was applied
        if not minutes_filter_applied:
            try:
                if 'Minutes played' in df.columns:
                    df = df[df['Minutes played'] >= 270]
                    applied_filters.append("Minimum 270 minutes played")
                    print(f"[DEBUG] After default minutes filter: {len(df)} players")
                else:
                    print("[ERROR] 'Minutes played' column not found for default filter")
            except Exception as e:
                print(f"[ERROR] Default minutes filter failed: {e}")

        # Auto-goalkeeper filter
        try:
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
            
            is_gk_stat = any(gk_stat.lower() == stat for gk_stat in gk_stats)
            
            if is_gk_stat and 'Position' in df.columns:
                df = df[df['Position'] == 'Goalkeeper']
                # applied_filters.append('Position: Goalkeeper')
                print(f"[DEBUG] After auto-goalkeeper filter: {len(df)} players")
        except Exception as e:
            print(f"[ERROR] Auto-goalkeeper filter failed: {e}")

        # Stat value filter
        if (preprocessed_hints.get('stat') and 
            preprocessed_hints.get('stat_value') is not None and
            not any('Minutes played' in f and '>=' in f for f in applied_filters)):
            
            try:
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
                    print(f"[DEBUG] After stat value filter: {len(df)} players")
                else:
                    print(f"[ERROR] Stat column '{stat}' not found in dataframe")
            except Exception as e:
                print(f"[ERROR] Stat value filter failed: {e}")

        print(f"[DEBUG] Final filtered DataFrame: {len(df)} players")
        return df, applied_filters, original_count
        
    except Exception as e:
        print(f"[ERROR] filter_players completely failed: {e}")
        return pd.DataFrame(), [f"Filter error: {str(e)}"], 0

def create_combined_stat(df: pd.DataFrame, stat1: str, stat2: str, operation: str = '+', new_stat_name: str = None) -> pd.DataFrame:
    """
    Create a new statistic by combining two existing stats.
    """
    try:
        if df.empty or stat1 not in df.columns or stat2 not in df.columns:
            raise ValueError(f"Required columns not found: {stat1}, {stat2}")
        
        if new_stat_name is None:
            if operation == '+':
                new_stat_name = f"{stat1} + {stat2}"
            elif operation == '-':
                new_stat_name = f"{stat1} - {stat2}"
            elif operation == '*':
                new_stat_name = f"{stat1} × {stat2}"
            elif operation == '/':
                new_stat_name = f"{stat1} / {stat2}"
            elif operation == 'ratio':
                new_stat_name = f"{stat1} to {stat2} ratio"
        
        df_copy = df.copy()
        
        if operation == '+':
            df_copy[new_stat_name] = df_copy[stat1] + df_copy[stat2]
        elif operation == '-':
            df_copy[new_stat_name] = df_copy[stat1] - df_copy[stat2]
        elif operation == '*':
            df_copy[new_stat_name] = df_copy[stat1] * df_copy[stat2]
        elif operation == '/' or operation == 'ratio':
            df_copy[new_stat_name] = df_copy[stat1] / df_copy[stat2].replace(0, np.nan)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        df_copy[new_stat_name] = df_copy[new_stat_name].replace([np.inf, -np.inf], np.nan)
        
        print(f"[DEBUG] Created combined stat '{new_stat_name}' using {stat1} {operation} {stat2}")
        return df_copy
        
    except Exception as e:
        print(f"[ERROR] Failed to create combined stat: {e}")
        raise

def sort_and_limit(df: pd.DataFrame, stat: str, top_n: int = 5) -> pd.DataFrame:
    """
    Sorts DataFrame by a stat and returns top N players with robust error handling.
    """
    try:
        if df is None or df.empty:
            print("[ERROR] DataFrame is empty for sorting")
            return pd.DataFrame()
        
        if not stat or stat not in df.columns:
            print(f"[ERROR] Stat '{stat}' not found in data. Available stats: {list(df.columns)}")
            return pd.DataFrame()
        
        if not pd.api.types.is_numeric_dtype(df[stat]):
            print(f"[ERROR] Stat '{stat}' is not numeric. Column dtype: {df[stat].dtype}")
            return pd.DataFrame()
        
        valid_data = df[stat].dropna()
        if valid_data.empty:
            print(f"[ERROR] Stat '{stat}' has no valid (non-NaN) values for the filtered players.")
            return pd.DataFrame()

        # Sort by the stat (descending for most stats, ascending for some)
        ascending_stats = ['Age', 'Minutes played', 'Matches played', 'xG conceded per 90', 'Yellow cards', 'Red cards']
        ascending = stat in ascending_stats
        
        if ascending:
            return df.nsmallest(top_n, stat)
        else:
            return df.nlargest(top_n, stat)
            
    except Exception as e:
        print(f"[ERROR] sort_and_limit failed: {e}")
        return pd.DataFrame()

def generate_chart(df: pd.DataFrame, stat: str, top_n: int = 5, chart_type: str = 'bar') -> Tuple[plt.Figure, str]:
    """
    Generates a chart for the top N players in a given stat with error handling.
    """
    try:
        if df.empty:
            raise ValueError("No data to plot")
        
        # Sort and get top N
        top_players = sort_and_limit(df, stat, top_n)
        if top_players.empty:
            raise ValueError("No valid data after sorting")
        
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
            team = best_player.get('Team within selected timeframe', 'Unknown')
            description += f"Best performer: {best_player['Player']} ({team}) with {best_player[stat]:.2f} {stat}."
        
        return fig, description
        
    except Exception as e:
        print(f"[ERROR] Chart generation failed: {e}")
        # Return empty figure with error message
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Chart generation failed:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Chart Generation Error")
        return fig, f"Failed to generate chart: {str(e)}"

def evaluate_stat_formula(df: pd.DataFrame, formula_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Evaluate a stat formula and add the computed column to the dataframe with error handling.
    """
    try:
        if df is None or df.empty:
            print("[ERROR] DataFrame is empty for formula evaluation")
            return pd.DataFrame()
        
        if not formula_data or not isinstance(formula_data, dict):
            raise ValueError("Invalid formula data provided")
        
        safe_map = formula_data.get('safe_map', {})
        expr = formula_data.get('expr', '')
        display_expr = formula_data.get('display_expr', 'Unknown Formula')
        
        if not safe_map or not expr:
            raise ValueError("Formula data missing required fields")
        
        print(f"[DEBUG] Evaluating formula: {display_expr}")
        print(f"[DEBUG] Safe expression: {expr}")
        print(f"[DEBUG] Safe map: {safe_map}")
        
        # Check if all required columns exist
        missing_cols = []
        for safe_var, col_name in safe_map.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)
        
        if missing_cols:
            raise ValueError(f"Missing columns for formula: {missing_cols}")
        
        # Create a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()
        
        # Evaluate the expression safely
        evaluated_expr = expr
        for safe_var, col_name in safe_map.items():
            evaluated_expr = evaluated_expr.replace(f'safe_map["{safe_var}"]', f'df_copy["{col_name}"]')
        
        print(f"[DEBUG] Final expression: {evaluated_expr}")
        
        # Evaluate the formula with error handling
        try:
            df_copy[display_expr] = eval(evaluated_expr)
        except ZeroDivisionError:
            print("[DEBUG] Division by zero detected, handling gracefully")
            df_copy[display_expr] = eval(evaluated_expr)
            df_copy[display_expr] = df_copy[display_expr].replace([float('inf'), float('-inf')], float('nan'))
        except Exception as eval_error:
            raise ValueError(f"Formula evaluation failed: {str(eval_error)}")
        
        # Handle division by zero and infinite values
        df_copy[display_expr] = df_copy[display_expr].replace([float('inf'), float('-inf')], float('nan'))
        
        # Remove rows where the formula result is NaN
        initial_count = len(df_copy)
        df_copy = df_copy.dropna(subset=[display_expr])
        final_count = len(df_copy)
        
        if final_count < initial_count:
            print(f"[DEBUG] Removed {initial_count - final_count} rows with invalid formula results")
        
        print(f"[DEBUG] Formula computed successfully. Sample values:")
        if not df_copy.empty:
            sample_values = df_copy[display_expr].head(5)
            for i, val in enumerate(sample_values):
                player_name = df_copy.iloc[i]['Player'] if 'Player' in df_copy.columns else f"Player {i}"
                print(f"  {player_name}: {val:.3f}")
        
        return df_copy
        
    except Exception as e:
        print(f"[ERROR] Formula evaluation failed: {e}")
        raise ValueError(f"Failed to evaluate formula '{formula_data.get('display_expr', 'unknown')}': {str(e)}")

def get_stat_definition_text(stat_name: str) -> str:
    """
    Get a comprehensive explanation of what a stat means with error handling.
    """
    try:
        if not stat_name or not isinstance(stat_name, str):
            return "Invalid statistic name provided."
        
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
        
    except Exception as e:
        print(f"[ERROR] Failed to get stat definition for '{stat_name}': {e}")
        return f"Unable to provide definition for this statistic due to an error: {str(e)}"

def generate_stat_definition_report(stat_name: str) -> Dict[str, Any]:
    """
    Generate a comprehensive stat definition report with robust error handling.
    """
    try:
        if not stat_name or not isinstance(stat_name, str):
            return {
                "error": "Invalid statistic name provided",
                "suggestions": []
            }
        
        df = load_data()
        if df is None or df.empty:
            return {
                "error": "No player data available",
                "definition": get_stat_definition_text(stat_name)
            }
        
        # Check if stat exists in the dataset
        if stat_name not in df.columns:
            # Try to find similar stat names
            try:
                similar_stats = get_close_matches(stat_name, df.columns, n=3, cutoff=0.6)
                
                error_msg = f"Statistic '{stat_name}' not found in dataset."
                if similar_stats:
                    error_msg += f" Did you mean: {', '.join(similar_stats)}?"
                
                return {
                    "error": error_msg,
                    "suggestions": similar_stats if similar_stats else []
                }
            except Exception as e:
                print(f"[ERROR] Failed to find similar stats: {e}")
                return {
                    "error": f"Statistic '{stat_name}' not found in dataset.",
                    "suggestions": []
                }
        
        # Get definition
        definition = get_stat_definition_text(stat_name)
        
        # Filter out players with insufficient minutes and NaN values
        try:
            filtered_df = df[
                (df['Minutes played'] >= 270) &  # Minimum minutes filter
                (df[stat_name].notna())  # Remove NaN values
            ].copy()
        except KeyError as e:
            return {
                "error": f"Required columns missing: {str(e)}",
                "definition": definition
            }
        
        if filtered_df.empty:
            return {
                "error": f"No players found with valid data for {stat_name}",
                "definition": definition
            }
        
        # Determine if higher or lower values are better
        ascending_stats = ['Age', 'Minutes played', 'Matches played', 'Yellow cards', 'Red cards', 'xG conceded per 90']
        ascending = stat_name in ascending_stats
        
        # Get top 10 players for this stat
        try:
            if ascending:
                top_players_df = filtered_df.nsmallest(10, stat_name)
                comparison_text = "lowest"
            else:
                top_players_df = filtered_df.nlargest(10, stat_name)
                comparison_text = "highest"
        except Exception as e:
            print(f"[ERROR] Failed to sort players by {stat_name}: {e}")
            return {
                "error": f"Failed to analyze {stat_name} data",
                "definition": definition
            }
        
        # Build player list with relevant info
        top_players = []
        for _, player in top_players_df.iterrows():
            try:
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
            except Exception as e:
                print(f"[ERROR] Failed to process player {player.get('Player', 'Unknown')}: {e}")
                continue
        
        # Calculate basic stats about this metric
        try:
            stat_values = filtered_df[stat_name]
            basic_stats = {
                "total_players": len(filtered_df),
                "mean": to_python_type(stat_values.mean()),
                "median": to_python_type(stat_values.median()),
                "std": to_python_type(stat_values.std()),
                "min": to_python_type(stat_values.min()),
                "max": to_python_type(stat_values.max())
            }
        except Exception as e:
            print(f"[ERROR] Failed to calculate basic stats: {e}")
            basic_stats = {
                "total_players": len(filtered_df),
                "mean": 0,
                "median": 0,
                "std": 0,
                "min": 0,
                "max": 0
            }
        
        # Position breakdown - which positions excel in this stat
        position_insights = []
        try:
            position_breakdown = filtered_df.groupby('Position')[stat_name].agg(['mean', 'count']).round(2)
            position_breakdown = position_breakdown[position_breakdown['count'] >= 5]  # Only positions with 5+ players
            
            if not ascending:
                position_breakdown = position_breakdown.sort_values('mean', ascending=False)
            else:
                position_breakdown = position_breakdown.sort_values('mean', ascending=True)
            
            for pos, data in position_breakdown.head(3).iterrows():
                position_insights.append({
                    "position": pos,
                    "average": to_python_type(data['mean']),
                    "player_count": to_python_type(data['count'])
                })
        except Exception as e:
            print(f"[ERROR] Failed to calculate position insights: {e}")
        
        return {
            "success": True,
            "stat_name": stat_name,
            "definition": definition,
            "comparison_text": comparison_text,
            "top_players": top_players,
            "basic_stats": basic_stats,
            "position_insights": position_insights
        }
        
    except Exception as e:
        print(f"[ERROR] generate_stat_definition_report failed: {e}")
        return {
            "error": f"Failed to generate stat definition report: {str(e)}",
            "suggestions": []
        }

def get_player_summary(df: pd.DataFrame, stat: str) -> Dict[str, Any]:
    """
    Get summary statistics for a given stat with error handling.
    """
    try:
        if df is None or df.empty:
            return {"error": "No data available"}
        
        if not stat or stat not in df.columns:
            return {"error": f"Stat '{stat}' not found"}
        
        # Handle NaN values
        valid_data = df[stat].dropna()
        if valid_data.empty:
            return {"error": f"No valid data for stat '{stat}'"}
        
        try:
            top_5_df = df.nlargest(5, stat)
            top_5_records = []
            
            for _, row in top_5_df.iterrows():
                record = {
                    'Player': row['Player'],
                    'Team within selected timeframe': row.get('Team within selected timeframe', 'Unknown'),
                    stat: to_python_type(row[stat])
                }
                top_5_records.append(record)
        except Exception as e:
            print(f"[ERROR] Failed to get top 5 for {stat}: {e}")
            top_5_records = []
        
        summary = {
            "stat": stat,
            "count": len(valid_data),
            "mean": to_python_type(valid_data.mean()),
            "median": to_python_type(valid_data.median()),
            "std": to_python_type(valid_data.std()),
            "min": to_python_type(valid_data.min()),
            "max": to_python_type(valid_data.max()),
            "top_5": top_5_records
        }
        
        return summary
        
    except Exception as e:
        print(f"[ERROR] get_player_summary failed: {e}")
        return {"error": f"Summary generation failed: {str(e)}"}

def get_player_percentiles(player_data: pd.Series, comparison_df: pd.DataFrame, stat_cols: List[str]) -> Dict[str, float]:
    """
    Calculate percentiles for a player compared to a comparison group with error handling.
    """
    try:
        percentiles = {}
        for stat in stat_cols:
            try:
                if stat in player_data.index and stat in comparison_df.columns:
                    player_value = player_data[stat]
                    if pd.notna(player_value):
                        # Calculate percentile (what % of players this player is better than)
                        valid_comparison = comparison_df[stat].dropna()
                        if not valid_comparison.empty:
                            percentile = (valid_comparison < player_value).mean() * 100
                            percentiles[stat] = round(percentile, 1)
            except Exception as e:
                print(f"[ERROR] Failed to calculate percentile for {stat}: {e}")
                continue
        
        return percentiles
        
    except Exception as e:
        print(f"[ERROR] get_player_percentiles failed: {e}")
        return {}

def get_key_stats_for_position(position: str) -> List[str]:
    """
    Return key stats based on player position with error handling.
    """
    try:
        if not position or not isinstance(position, str):
            return []
        
        position_stats = {
            'Goalkeeper': [
                'Clean sheets', 'Saves per 90', 'Save percentage %.1', 
                'xG conceded per 90', 'Passes per 90', 'Long passes per 90'
            ],
            'Centre-back': [
                'Aerial duels won per 90', 'Clearances per 90', 'Blocks per 90',
                'Interceptions per 90', 'Sliding tackles per 90', 'Pass completion %.1',
                'Progressive passes per 90', 'Poss+/-'
            ],
            'Full-back': [
                'Sliding tackles per 90', 'Interceptions per 90', 'Progressive passes per 90',
                'Crosses per 90', 'Cross accuracy %.1', 'Assists per 90',
                'Progressive carries per 90', 'Duels won %'
            ],
            'Midfielder': [
                'Passes per 90', 'Pass completion %.1', 'Progressive passes per 90',
                'Key passes per 90', 'Poss+/-', 'Sliding tackles per 90',
                'Interceptions per 90', 'Goals + Assists per 90'
            ],
            'Winger': [
                'Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90',
                'Dribbles attempted per 90', 'Dribble success rate %.1',
                'Crosses per 90', 'Progressive carries per 90'
            ],
            'Striker': [
                'Goals per 90', 'xG per 90', 'npxG per 90', 'Shots per 90',
                'Shots on target %.1', 'Goals per xG', 'Assists per 90',
                'Aerial duels won per 90'
            ]
        }
        
        # Default stats if position not found
        default_stats = [
            'Goals per 90', 'Assists per 90', 'xG per 90', 'xA per 90',
            'Passes per 90', 'Pass completion %.1', 'Sliding tackles per 90'
        ]
        
        return position_stats.get(position, default_stats)
        
    except Exception as e:
        print(f"[ERROR] get_key_stats_for_position failed: {e}")
        return []

def format_percentile_description(percentile: float) -> str:
    """
    Convert percentile to descriptive text with error handling.
    """
    try:
        if not isinstance(percentile, (int, float)):
            return "Unknown"
        
        if percentile >= 95:
            return "Elite (Top 5%)"
        elif percentile >= 90:
            return "Excellent (Top 10%)"
        elif percentile >= 75:
            return "Above Average (Top 25%)"
        elif percentile >= 50:
            return "Average (Top 50%)"
        elif percentile >= 25:
            return "Below Average (Bottom 50%)"
        else:
            return "Poor (Bottom 25%)"
            
    except Exception as e:
        print(f"[ERROR] format_percentile_description failed: {e}")
        return "Unknown"

def normalize_text(text: str) -> str:
    """
    Normalize text by removing accents, converting to lowercase, and cleaning up.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove accents and normalize unicode
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Convert to lowercase and remove extra spaces
    text = text.lower().strip()
    text = ' '.join(text.split())
    
    return text

def calculate_match_score(query: str, player_name: str) -> float:
    """
    Calculate match score designed for "Initial. LastName" format (e.g., "F. Wirtz", "M. Salah").
    Returns score from 0.0 to 1.0, with higher being better match.
    """
    try:
        # Normalize both inputs
        query_norm = normalize_text(query)
        player_norm = normalize_text(player_name)
        
        # Extract components
        query_parts = query_norm.split()
        player_parts = player_norm.split()
        
        print(f"[DEBUG] Matching '{query_norm}' against '{player_norm}'")
        print(f"[DEBUG] Query parts: {query_parts}, Player parts: {player_parts}")
        
        # EXACT MATCH - highest priority
        if query_norm == player_norm:
            print(f"[DEBUG] Exact match found")
            return 1.0
        
        # CASE 1: Query is just last name (e.g., "wirtz" should match "F. Wirtz")
        if len(query_parts) == 1:
            query_word = query_parts[0]
            
            # Check if it matches the last name in "F. LastName" format
            if (len(player_parts) == 2 and 
                len(player_parts[0]) == 2 and 
                player_parts[0].endswith('.') and
                query_word == player_parts[1]):
                print(f"[DEBUG] Last name match in 'Initial. LastName' format: {query_word} == {player_parts[1]}")
                return 0.92
            
            # Check if it matches last name in "FirstName LastName" format  
            if len(player_parts) >= 2 and query_word == player_parts[-1]:
                print(f"[DEBUG] Last name match in multi-part name")
                return 0.90
            
            # Check if it matches first name/mononym
            if len(player_parts) == 1 and query_word == player_parts[0]:
                print(f"[DEBUG] Mononym match")
                return 0.88
            
            # Check if it matches first name in multi-part name
            if len(player_parts) >= 2 and query_word == player_parts[0]:
                print(f"[DEBUG] First name match")
                return 0.85
            
            # Check for partial matches in last name (for typos)
            if (len(player_parts) >= 2 and 
                len(query_word) >= 3 and 
                len(player_parts[-1]) >= 3):
                # Use sequence matcher for partial similarity
                similarity = SequenceMatcher(None, query_word, player_parts[-1]).ratio()
                if similarity >= 0.8:
                    print(f"[DEBUG] Partial last name match: {similarity:.2f}")
                    return similarity * 0.8  # Reduce score for partial matches
        
        # CASE 2: Query is "initial lastname" (e.g., "f wirtz" -> "F. Wirtz")
        elif (len(query_parts) == 2 and 
              len(query_parts[0]) == 1 and
              len(player_parts) == 2 and
              len(player_parts[0]) == 2 and
              player_parts[0].endswith('.')):
            
            if (query_parts[0] == player_parts[0][0] and  # Initial matches
                query_parts[1] == player_parts[1]):        # Last name matches
                print(f"[DEBUG] 'initial lastname' match")
                return 0.95
        
        # CASE 3: Query is "initial. lastname" (e.g., "f. wirtz" -> "F. Wirtz")  
        elif (len(query_parts) == 2 and
              query_parts[0].endswith('.') and
              len(player_parts) == 2 and
              player_parts[0].endswith('.')):
            
            if (query_parts[0] == player_parts[0] and     # "f." == "f."
                query_parts[1] == player_parts[1]):       # Last name matches
                print(f"[DEBUG] 'initial. lastname' match")
                return 0.97
        
        # CASE 4: Query is "firstname lastname" (e.g., "florian wirtz" -> "F. Wirtz")
        elif (len(query_parts) == 2 and
              len(player_parts) == 2 and
              player_parts[0].endswith('.')):
            
            if (query_parts[0][0] == player_parts[0][0] and  # First letter matches initial
                query_parts[1] == player_parts[1]):           # Last name matches
                print(f"[DEBUG] 'firstname lastname' to 'Initial. LastName' match")
                return 0.93
        
        # CASE 5: Full name matching full name
        elif (len(query_parts) >= 2 and 
              len(player_parts) >= 2 and
              not player_parts[0].endswith('.')):
            
            if (query_parts[0] == player_parts[0] and      # First name exact
                query_parts[-1] == player_parts[-1]):      # Last name exact  
                print(f"[DEBUG] Full name match")
                return 0.96
        
        print(f"[DEBUG] No match pattern found")
        return 0.0
        
    except Exception as e:
        print(f"[ERROR] Match score calculation failed: {e}")
        return 0.0

def find_all_matching_players(df: pd.DataFrame, query: str) -> List[Dict[str, Any]]:
    """
    Scan the ENTIRE dataset and find ALL players that match the query.
    This prevents premature stopping at partial matches.
    """
    if df.empty or 'Player' not in df.columns:
        return []
    
    query_norm = normalize_text(query)
    query_parts = query_norm.split()
    
    print(f"[DEBUG] Comprehensive search for: '{query}' (normalized: '{query_norm}')")
    print(f"[DEBUG] Scanning {len(df)} total players...")
    
    all_matches = []
    exact_lastname_matches = []
    partial_matches = []
    
    # STEP 1: Scan entire dataset and categorize matches
    for idx, row in df.iterrows():
        try:
            player_name = str(row['Player'])
            player_norm = normalize_text(player_name)
            player_parts = player_norm.split()
            
            # Calculate match score
            score = calculate_match_score(query, player_name)
            
            if score > 0:
                match_info = {
                    'player_name': player_name,
                    'match_score': score,
                    'team': row.get('Team within selected timeframe', 'Unknown'),
                    'position': row.get('Position', 'Unknown'),
                    'league': row.get('League', 'Unknown'),
                    'minutes': row.get('Minutes played', 0),
                    'index': idx
                }
                
                all_matches.append(match_info)
                
                # Special categorization for single-word queries (most common)
                if len(query_parts) == 1:
                    query_word = query_parts[0]
                    
                    # Check if this is an exact last name match
                    if (len(player_parts) >= 2 and 
                        query_word == player_parts[-1]):  # Last name exact match
                        exact_lastname_matches.append(match_info)
                    
                    # Check for substring matches in last name
                    elif (len(player_parts) >= 2 and 
                          query_word in player_parts[-1] and 
                          len(query_word) >= 3):
                        partial_matches.append(match_info)
        
        except Exception as e:
            print(f"[ERROR] Failed to process player at index {idx}: {e}")
            continue
    
    # STEP 2: Prioritize results
    print(f"[DEBUG] Found {len(all_matches)} total matches")
    print(f"[DEBUG] Found {len(exact_lastname_matches)} exact lastname matches")
    print(f"[DEBUG] Found {len(partial_matches)} partial matches")
    
    # Sort all matches by score
    all_matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    # Print top matches for debugging
    print(f"[DEBUG] Top 10 matches by score:")
    for i, match in enumerate(all_matches[:10]):
        marker = "★" if match in exact_lastname_matches else " "
        print(f"  {marker}{i+1}. {match['player_name']}: {match['match_score']:.3f} ({match['team']}, {match['league']})")
    
    return all_matches

def smart_player_lookup_v4(df: pd.DataFrame, query: str) -> Tuple[Optional[pd.Series], List[Dict[str, Any]]]:
    """
    Completely rewritten lookup that scans the entire dataset first.
    """
    try:
        if df.empty or 'Player' not in df.columns:
            return None, []
        
        print(f"[DEBUG] smart_player_lookup_v4: Looking up player: '{query}'")
        
        # Step 1: Check nickname map first
        nickname_map = create_enhanced_nickname_map()
        query_normalized = normalize_text(query)
        
        if query_normalized in nickname_map:
            target_name = nickname_map[query_normalized]
            print(f"[DEBUG] Found in nickname map: '{query}' -> '{target_name}'")
            
            # Look for exact match in dataset
            exact_matches = df[df['Player'] == target_name]
            if not exact_matches.empty:
                print(f"[DEBUG] Successfully found via nickname map: {target_name}")
                return exact_matches.iloc[0], []
            else:
                print(f"[DEBUG] Nickname target '{target_name}' not found in dataset")
        
        # Step 2: Comprehensive search of entire dataset
        all_matches = find_all_matching_players(df, query)
        
        if not all_matches:
            print(f"[DEBUG] No matches found for '{query}'")
            return None, []
        
        # Step 3: Apply additional scoring for prominence
        for match in all_matches:
            prominence = 1.0
            
            # Top leagues bonus
            top_leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
            if match['league'] in top_leagues:
                prominence += 0.1
            
            # Minutes played bonus (more playing time = more prominent)
            minutes = match['minutes']
            if minutes >= 2000:
                prominence += 0.2
            elif minutes >= 1000:
                prominence += 0.1
            elif minutes >= 500:
                prominence += 0.05
            
            match['final_score'] = min(match['match_score'] * prominence, 1.0)
        
        # Sort by final score
        all_matches.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"[DEBUG] Top 5 matches after prominence weighting:")
        for i, match in enumerate(all_matches[:5]):
            print(f"  {i+1}. {match['player_name']}: {match['final_score']:.3f} (base: {match['match_score']:.3f}, {match['minutes']} min)")
        
        best_match = all_matches[0]
        
        # Auto-select if score is high enough
        if best_match['final_score'] >= 0.90:
            player_row = df.iloc[best_match['index']]
            print(f"[DEBUG] Auto-selected: {best_match['player_name']} (score: {best_match['final_score']:.3f})")
            return player_row, all_matches[1:4]
        
        # Check if there's a clear winner even with lower score
        if len(all_matches) == 1 or (len(all_matches) > 1 and best_match['final_score'] - all_matches[1]['final_score'] > 0.3):
            player_row = df.iloc[best_match['index']]
            print(f"[DEBUG] Clear winner selected: {best_match['player_name']} (score: {best_match['final_score']:.3f})")
            return player_row, all_matches[1:4]
        
        # Multiple close matches, return alternatives
        print(f"[DEBUG] Multiple close matches, returning alternatives")
        return None, all_matches[:5]
        
    except Exception as e:
        print(f"[ERROR] smart_player_lookup_v4 failed: {e}")
        return None, []

def enhanced_calculate_match_score(query: str, player_name: str) -> float:
    """
    Enhanced matching that prioritizes exact lastname matches.
    """
    try:
        query_norm = normalize_text(query)
        player_norm = normalize_text(player_name)
        
        query_parts = query_norm.split()
        player_parts = player_norm.split()
        
        # EXACT MATCH
        if query_norm == player_norm:
            return 1.0
        
        # Single word query (like "wirtz", "kobel", "adeyemi")
        if len(query_parts) == 1:
            query_word = query_parts[0]
            
            # PRIORITY 1: Exact last name match in "Initial. LastName" format
            if (len(player_parts) == 2 and 
                len(player_parts[0]) == 2 and 
                player_parts[0].endswith('.') and
                query_word == player_parts[1]):
                return 0.95  # Very high score for exact lastname match
            
            # PRIORITY 2: Exact last name match in "FirstName LastName" format
            if len(player_parts) >= 2 and query_word == player_parts[-1]:
                return 0.93
            
            # PRIORITY 3: Single name (mononym) exact match
            if len(player_parts) == 1 and query_word == player_parts[0]:
                return 0.90
            
            # PRIORITY 4: First name exact match (lower priority)
            if len(player_parts) >= 2 and query_word == player_parts[0]:
                return 0.75
            
            # PRIORITY 5: Substring matches (even lower priority)
            if len(query_word) >= 3:
                for player_part in player_parts:
                    clean_part = player_part.rstrip('.')
                    if len(clean_part) >= 3:
                        if query_word in clean_part or clean_part in query_word:
                            similarity = SequenceMatcher(None, query_word, clean_part).ratio()
                            if similarity >= 0.8:
                                return similarity * 0.6  # Much lower for substring
        
        # Two word queries
        elif len(query_parts) == 2:
            # "florian wirtz" -> "F. Wirtz"
            if (len(player_parts) == 2 and 
                player_parts[0].endswith('.') and
                len(player_parts[0]) == 2):
                
                if (query_parts[0][0].lower() == player_parts[0][0].lower() and
                    query_parts[1] == player_parts[1]):
                    return 0.98
            
            # "f wirtz" -> "F. Wirtz"
            elif (len(query_parts[0]) == 1 and
                  len(player_parts) == 2 and
                  player_parts[0].endswith('.')):
                
                if (query_parts[0] == player_parts[0][0] and
                    query_parts[1] == player_parts[1]):
                    return 0.97
            
            # "f. wirtz" -> "F. Wirtz"
            elif (query_parts[0].endswith('.') and
                  len(player_parts) == 2 and
                  player_parts[0].endswith('.')):
                
                if (query_parts[0].lower() == player_parts[0].lower() and
                    query_parts[1] == player_parts[1]):
                    return 0.99
            
            # "firstname lastname" -> "FirstName LastName"
            elif len(player_parts) >= 2:
                if (query_parts[0] == player_parts[0] and
                    query_parts[1] == player_parts[-1]):
                    return 0.96
        
        return 0.0
        
    except Exception as e:
        print(f"[ERROR] enhanced_calculate_match_score failed: {e}")
        return 0.0

def create_enhanced_nickname_map() -> Dict[str, str]:
    """
    Enhanced nickname map with the missing entries that are causing failures.
    """
    return {
        # YOUR FAILING CASES - CRITICAL FIXES
        "kobel": "G. Kobel",
        "gregor kobel": "G. Kobel",
        "g kobel": "G. Kobel",
        "g. kobel": "G. Kobel",
        
        "wirtz": "F. Wirtz", 
        "florian wirtz": "F. Wirtz",
        "f wirtz": "F. Wirtz",
        "f. wirtz": "F. Wirtz",
        
        "adeyemi": "K. Adeyemi",
        "karim adeyemi": "K. Adeyemi", 
        "k adeyemi": "K. Adeyemi",
        "k. adeyemi": "K. Adeyemi",
        
        # Existing mappings...
        "salah": "M. Salah",
        "mo salah": "M. Salah",
        "mohamed salah": "M. Salah",
        
        "messi": "Lionel Messi", 
        "leo messi": "Lionel Messi",
        
        "ronaldo": "Cristiano Ronaldo",
        "cristiano": "Cristiano Ronaldo",
        "cr7": "Cristiano Ronaldo",
        
        "haaland": "Erling Haaland",
        "erling": "Erling Haaland",
        
        "mbappe": "Kylian Mbappé",
        "mbappé": "Kylian Mbappé",
        "kylian": "Kylian Mbappé",
        
        "vinicius": "Vinícius Júnior",
        "vini": "Vinícius Júnior", 
        "vinicius junior": "Vinícius Júnior",
        
        "bellingham": "Jude Bellingham",
        "jude": "Jude Bellingham",
        
        "pedri": "Pedri",
        "gavi": "Gavi",
        
        "benzema": "Karim Benzema",
        "karim": "Karim Benzema",
        
        "lewandowski": "Robert Lewandowski",
        "lewa": "Robert Lewandowski",
        "lewy": "Robert Lewandowski",
        
        "kane": "Harry Kane",
        "harry kane": "Harry Kane",
        
        "son": "Heung-min Son",
        "heung min son": "Heung-min Son",
        
        "de bruyne": "Kevin De Bruyne",
        "kdb": "Kevin De Bruyne",
        "kevin de bruyne": "Kevin De Bruyne",
        
        "modric": "Luka Modrić",
        "luka modric": "Luka Modrić",
        
        "kroos": "Toni Kroos", 
        "toni kroos": "Toni Kroos",
    }

def lastname_focused_search(df: pd.DataFrame, query: str) -> List[Dict[str, Any]]:
    """
    Search that prioritizes lastname matches first, then other matches.
    This addresses the core issue you identified.
    """
    if df.empty or 'Player' not in df.columns:
        return []
    
    query_norm = normalize_text(query)
    query_parts = query_norm.split()
    
    print(f"[DEBUG] Lastname-focused search for: '{query}'")
    
    # If single word query, first find all players with that lastname
    if len(query_parts) == 1:
        query_word = query_parts[0]
        
        # STEP 1: Find all exact lastname matches
        lastname_matches = []
        other_matches = []
        
        print(f"[DEBUG] Looking for players with lastname '{query_word}'...")
        
        for idx, row in df.iterrows():
            try:
                player_name = str(row['Player'])
                player_norm = normalize_text(player_name)
                player_parts = player_norm.split()
                
                # Check for exact lastname match
                is_lastname_match = False
                if len(player_parts) >= 2 and query_word == player_parts[-1]:
                    is_lastname_match = True
                
                match_info = {
                    'player_name': player_name,
                    'team': row.get('Team within selected timeframe', 'Unknown'),
                    'position': row.get('Position', 'Unknown'),
                    'league': row.get('League', 'Unknown'),
                    'minutes': row.get('Minutes played', 0),
                    'age': row.get('Age', 'Unknown'),
                    'index': idx,
                    'is_lastname_match': is_lastname_match
                }
                
                if is_lastname_match:
                    lastname_matches.append(match_info)
                    print(f"[DEBUG] Found lastname match: {player_name}")
                else:
                    # Check other types of matches
                    score = calculate_match_score(query, player_name)
                    if score > 0.5:  # Lower threshold for other matches
                        match_info['match_score'] = score
                        other_matches.append(match_info)
            
            except Exception as e:
                continue
        
        print(f"[DEBUG] Found {len(lastname_matches)} lastname matches")
        print(f"[DEBUG] Found {len(other_matches)} other matches")
        
        # If we have lastname matches, prioritize them
        if lastname_matches:
            # Sort lastname matches by prominence (minutes played, league quality)
            for match in lastname_matches:
                prominence = 1.0
                
                # Top leagues bonus
                top_leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
                if match['league'] in top_leagues:
                    prominence += 0.3
                
                # Minutes bonus
                minutes = match['minutes']
                if minutes >= 2000:
                    prominence += 0.4
                elif minutes >= 1000:
                    prominence += 0.2
                elif minutes >= 500:
                    prominence += 0.1
                
                match['final_score'] = prominence
                match['match_score'] = 0.95  # High base score for lastname matches
            
            lastname_matches.sort(key=lambda x: x['final_score'], reverse=True)
            
            print(f"[DEBUG] Lastname matches ranked by prominence:")
            for i, match in enumerate(lastname_matches):
                print(f"  {i+1}. {match['player_name']}: {match['final_score']:.3f} ({match['league']}, {match['minutes']} min)")
            
            return lastname_matches, other_matches
        
        # If no lastname matches, return other matches
        else:
            other_matches.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            return [], other_matches
    
    else:
        # Multi-word query, use comprehensive search
        all_matches = find_all_matching_players(df, query)
        return all_matches, []
        

def smart_player_lookup_final(df: pd.DataFrame, query: str) -> Tuple[Optional[pd.Series], List[Dict[str, Any]]]:
    """
    Final version that combines nickname lookup with lastname-focused search.
    """
    try:
        if df.empty or 'Player' not in df.columns:
            return None, []
        
        print(f"[DEBUG] FINAL LOOKUP for: '{query}'")
        
        # Step 1: Nickname map (highest priority)
        nickname_map = create_enhanced_nickname_map()
        query_normalized = normalize_text(query)
        
        if query_normalized in nickname_map:
            target_name = nickname_map[query_normalized]
            print(f"[DEBUG] Nickname map hit: '{query}' -> '{target_name}'")
            
            exact_matches = df[df['Player'] == target_name]
            if not exact_matches.empty:
                print(f"[DEBUG] SUCCESS via nickname map")
                return exact_matches.iloc[0], []
        
        # Step 2: Lastname-focused search
        primary_matches, secondary_matches = lastname_focused_search(df, query)
        
        # If we found lastname matches, pick the best one
        if primary_matches:
            best_lastname_match = primary_matches[0]
            player_row = df.iloc[best_lastname_match['index']]
            print(f"[DEBUG] SUCCESS via lastname search: {best_lastname_match['player_name']}")
            return player_row, primary_matches[1:] + secondary_matches[:3]
        
        # If no lastname matches but we have other matches
        elif secondary_matches:
            print(f"[DEBUG] No lastname matches, showing alternatives")
            return None, secondary_matches[:5]
        
        # No matches at all
        else:
            print(f"[DEBUG] No matches found")
            return None, []
            
    except Exception as e:
        print(f"[ERROR] smart_player_lookup_final failed: {e}")
        return None, []

def generate_player_report(player_query: str) -> Dict[str, Any]:
    """
    Enhanced player report generation with new matching system.
    """
    try:
        df = load_data()
        if df is None or df.empty:
            return {"error": "No player data available"}
        
        # Try new smart lookup
        best_match, alternatives = smart_player_lookup_final(df, player_query)
        
        if best_match is not None:
            # We found a good match, generate the report
            player_name = best_match['Player']
            print(f"[DEBUG] Generating report for: {player_name}")
            
            # Generate the report using existing logic
            return generate_player_report_core(best_match, df, alternatives)
        
        else:
            # No clear best match, return alternatives for user to choose from
            if alternatives:
                return {
                    "error": f"Multiple players found matching '{player_query}'. Please be more specific.",
                    "alternatives": alternatives,
                    "suggestion": f"Did you mean {alternatives[0]['player_name']}?",
                    "debug_info": f"Top match score: {alternatives[0]['final_score']:.2f}"
                }
            else:
                # Try a broader search with lower threshold for suggestions
                broad_matches = find_all_matching_players(df, player_query, min_score=0.3)
                if broad_matches:
                    return {
                        "error": f"No exact matches found for '{player_query}'.",
                        "suggestions": broad_matches[:5],
                        "help": "Try using just the last name or check spelling."
                    }
                else:
                    return {"error": f"No players found matching '{player_query}'"}
    
    except Exception as e:
        print(f"[ERROR] generate_player_report failed: {e}")
        return {"error": f"Failed to generate player report: {str(e)}"}

def generate_player_report_core(player_record: pd.Series, df: pd.DataFrame, alternatives: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate player report using the matched player.
    """
    try:
        # Basic info
        basic_info = {
            "name": player_record['Player'],
            "team": player_record.get('Team within selected timeframe', 'Unknown'),
            "league": player_record.get('League', 'Unknown'), 
            "position": player_record.get('Position', 'Unknown'),
            "age": to_python_type(player_record.get('Age', 'Unknown')),
            "minutes_played": to_python_type(player_record.get('Minutes played', 0)),
            "matches_played": to_python_type(player_record.get('Matches played', 0))
        }
        
        # Get key stats for the player's position
        key_stats = get_key_stats_for_position(basic_info['position'])
        available_stats = [stat for stat in key_stats if stat in df.columns]
        
        # Player's key stats
        player_stats = {}
        for stat in available_stats:
            value = player_record.get(stat)
            if pd.notna(value):
                player_stats[stat] = to_python_type(value)
        
        # League comparison (players in same league and position)
        league_comparison_df = df[
            (df['League'] == basic_info['league']) & 
            (df['Position'] == basic_info['position']) &
            (df['Minutes played'] >= 270)
        ]
        
        league_percentiles = get_player_percentiles(player_record, league_comparison_df, available_stats)
        
        # Top 5 leagues comparison (if player is in a top 5 league)
        top5_leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
        top5_comparison = None
        top5_percentiles = {}
        
        if basic_info['league'] in top5_leagues:
            top5_comparison_df = df[
                (df['League'].isin(top5_leagues)) & 
                (df['Position'] == basic_info['position']) &
                (df['Minutes played'] >= 270)
            ]
            top5_percentiles = get_player_percentiles(player_record, top5_comparison_df, available_stats)
            top5_comparison = {
                "total_players": len(top5_comparison_df),
                "leagues": top5_leagues
            }
        
        # Position comparison within league
        position_peers = league_comparison_df[league_comparison_df['Position'] == basic_info['position']]
        
        # Find similar players
        age = basic_info.get('age', 25)
        similar_players_df = df[
            (df['Position'] == basic_info['position']) &
            (df['League'] == basic_info['league']) &
            (df['Age'].between(age - 3, age + 3)) &
            (df['Minutes played'] >= 270) &
            (df['Player'] != player_record['Player'])
        ]
        
        similar_players = []
        if not similar_players_df.empty and len(available_stats) > 0:
            if basic_info['position'] in ['Striker', 'Winger']:
                sort_stat = 'Goals + Assists per 90' if 'Goals + Assists per 90' in available_stats else available_stats[0]
            else:
                sort_stat = 'Poss+/-' if 'Poss+/-' in available_stats else available_stats[0]
            
            top_similar = similar_players_df.nlargest(5, sort_stat) if sort_stat in similar_players_df.columns else similar_players_df.head(5)
            
            for _, similar_player in top_similar.iterrows():
                similar_players.append({
                    "name": similar_player['Player'],
                    "team": similar_player.get('Team within selected timeframe', 'Unknown'),
                    "age": to_python_type(similar_player.get('Age', 'Unknown')),
                    "key_stat_value": to_python_type(similar_player.get(sort_stat, 0))
                })
        
        # Strengths and weaknesses analysis
        strengths = []
        weaknesses = []
        
        for stat, percentile in league_percentiles.items():
            if percentile >= 80:
                strengths.append({
                    "stat": stat,
                    "value": player_stats.get(stat, 0),
                    "percentile": percentile,
                    "description": format_percentile_description(percentile)
                })
            elif percentile <= 20:
                weaknesses.append({
                    "stat": stat,
                    "value": player_stats.get(stat, 0),
                    "percentile": percentile,
                    "description": format_percentile_description(percentile)
                })
        
        # Sort strengths and weaknesses by percentile
        strengths.sort(key=lambda x: x['percentile'], reverse=True)
        weaknesses.sort(key=lambda x: x['percentile'])
        
        result = {
            "success": True,
            "basic_info": basic_info,
            "player_stats": player_stats,
            "league_comparison": {
                "league": basic_info['league'],
                "total_players": len(league_comparison_df),
                "position_peers": len(position_peers),
                "percentiles": league_percentiles
            },
            "top5_comparison": top5_comparison,
            "top5_percentiles": top5_percentiles,
            "strengths": strengths[:5],
            "weaknesses": weaknesses[:5],
            "similar_players": similar_players,
            "key_stats_analyzed": available_stats
        }
        
        # Add alternatives info if provided
        if alternatives:
            result["alternatives_considered"] = alternatives[:3]  # Show top 3 alternatives
        
        return result
        
    except Exception as e:
        print(f"[ERROR] generate_player_report_core failed: {e}")
        return {"error": f"Failed to generate player report: {str(e)}"}

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
        
        # Handle stat formulas
        stat = preprocessed_hints.get('stat')
        stat_formula = preprocessed_hints.get('stat_formula')
        
        if stat_formula:
            print(f"[DEBUG] Processing stat formula: {stat_formula}")
            try:
                df = evaluate_stat_formula(df, stat_formula)
                stat = stat_formula['display_expr']
                print(f"[DEBUG] Formula evaluation successful. Using stat: {stat}")
            except Exception as e:
                return {
                    "error": f"Formula evaluation failed: {str(e)}",
                    "applied_filters": applied_filters,
                    "original_count": original_count
                }
        
        # Handle combined statistics
        combined_stats = preprocessed_hints.get('combined_stats')
        if combined_stats:
            try:
                stat1 = combined_stats.get('stat1')
                stat2 = combined_stats.get('stat2') 
                operation = combined_stats.get('operation', '+')
                new_name = combined_stats.get('name')
                
                df = create_combined_stat(df, stat1, stat2, operation, new_name)
                stat = new_name or f"{stat1} {operation} {stat2}"
                print(f"[DEBUG] Combined stat created: {stat}")
                
            except Exception as e:
                return {
                    "error": f"Failed to create combined statistic: {str(e)}",
                    "applied_filters": applied_filters,
                    "original_count": original_count
                }
        
        query_type = preprocessed_hints.get('query_type', 'FILTER')
        
        # Handle ambiguous queries better
        if query_type in ["OTHER", None] or not query_type:
            if stat or stat_formula or combined_stats:
                query_type = "TOP_N"
            elif any(key in preprocessed_hints for key in ["position", "league", "team", "age_filter"]):
                query_type = "FILTER" 
            else:
                query_type = "LIST"
        
        # Handle stat value filters properly
        if preprocessed_hints.get("stat_op") and preprocessed_hints.get("stat_value") is not None:
            query_type = "FILTER"
        
        if query_type == 'TOP_N':
            if not stat:
                return {
                    "error": "No statistic specified for ranking. Please specify what metric you want to rank by.",
                    "applied_filters": applied_filters,
                    "count": len(df)
                }
            
            top_n = preprocessed_hints.get('top_n', 5)
            
            try:
                top_players_df = sort_and_limit(df, stat, top_n)
                summary = get_player_summary(df, stat)
                
                return {
                    "success": True,
                    "top_players": [
                        {k: to_python_type(v) for k, v in player.items()}
                        for player in top_players_df.to_dict('records')
                    ],
                    "summary": {k: to_python_type(v) for k, v in summary.items()},
                    "applied_filters": applied_filters,
                    "original_count": original_count,
                    "filtered_count": len(df),
                    "stat": stat,
                    "top_n": top_n,
                    "count": len(df)
                }
            except Exception as e:
                return {
                    "error": f"Error ranking by {stat}: {str(e)}",
                    "applied_filters": applied_filters,
                    "count": len(df)
                }
        
        elif query_type == 'FILTER':
            stat = preprocessed_hints.get('stat')
            
            base_columns = ['Player', 'Team within selected timeframe', 'Position', 'Age', 'League']
            filtered_data = df[base_columns].copy()
            
            if stat and stat in df.columns:
                filtered_data[stat] = df[stat]
                
                if preprocessed_hints.get("stat_op") and preprocessed_hints.get("stat_value") is not None:
                    op = preprocessed_hints["stat_op"]
                    value = preprocessed_hints["stat_value"]
                    
                    if op == ">=":
                        stat_filtered_df = df[df[stat] >= value]
                    elif op == ">":
                        stat_filtered_df = df[df[stat] > value]
                    elif op == "<=":
                        stat_filtered_df = df[df[stat] <= value]
                    elif op == "<":
                        stat_filtered_df = df[df[stat] < value]
                    elif op == "==":
                        stat_filtered_df = df[df[stat] == value]
                    else:
                        stat_filtered_df = df
                    
                    if stat_filtered_df.empty:
                        return {
                            "error": f"No players found with {stat} {op} {value}",
                            "applied_filters": applied_filters + [f"{stat} {op} {value}"],
                            "original_count": original_count
                        }
                    
                    result_data = stat_filtered_df[base_columns + [stat]].copy()
                    result_data = result_data.sort_values(by=stat, ascending=False)
                    
                    return {
                        "success": True,
                        "count": len(stat_filtered_df),
                        "summary": f"Found {len(stat_filtered_df)} players with {stat} {op} {value}",
                        "filtered_data": result_data.to_dict('records'),
                        "applied_filters": applied_filters + [f"{stat} {op} {value}"],
                        "stat": stat
                    }            
            
            return {
                "success": True,
                "count": len(df),
                "summary": f"Found {len(df)} players matching your criteria",
                "filtered_data": filtered_data.to_dict('records'),
                "applied_filters": applied_filters
            }
        
        elif query_type in ['LIST', 'COUNT']:
            base_columns = ['Player', 'Team within selected timeframe', 'Position', 'Age', 'League']
            return {
                "success": True,
                "players": df[base_columns].to_dict('records'),
                "count": len(df),
                "applied_filters": applied_filters,
                "summary": f"Found {len(df)} players matching your criteria"
            }
        
        else:
            return {
                "error": f"Query type '{query_type}' not fully supported yet. Please try rephrasing your question.",
                "applied_filters": applied_filters,
                "count": len(df)
            }
        
    except Exception as e:
        print(f"[DEBUG] Analysis failed: {str(e)}")
        return {
            "error": f"Analysis failed: {str(e)}",
            "preprocessed_hints": preprocessed_hints
        }