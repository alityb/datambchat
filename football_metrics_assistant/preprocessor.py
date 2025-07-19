import re
from typing import Dict, Any

def preprocess_query(query: str) -> Dict[str, Any]:
    """
    Preprocesses the user query to extract structured hints for downstream logic.
    Handles terms like 'Top 5', 'Poss+/-', etc.
    """
    result = {"original": query}
    # Handle 'Top N'
    top_match = re.search(r"top (\d+)", query, re.IGNORECASE)
    if top_match:
        result["top_n"] = int(top_match.group(1))
    # Handle 'Poss+/-'
    if "poss+/-" in query.lower():
        result["stat"] = "Poss+/-"
    # Add more rules as needed
    return result 