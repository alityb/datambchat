import google.generativeai as genai
from typing import List, Dict
import re
import os

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("Warning: GOOGLE_API_KEY environment variable not set. Please set it to use Gemini Flash.")

def ask_gemini(message: str, history: List[Dict] = None, model: str = "gemini-1.5-flash") -> str:
    if not GOOGLE_API_KEY:
        return "[Error] Google API key not configured. Please set GOOGLE_API_KEY environment variable."
    
    if history is None:
        history = []
    
    # System message for detailed but concise responses
    system_message = "You are a football analytics expert. Provide clear, informative explanations of football statistics and metrics. Be conversational but concise - aim for 2-3 paragraphs maximum."
    
    # Build conversation history for Gemini
    chat = genai.GenerativeModel(model).start_chat(history=[])
    
    # Add system message as first user message (Gemini doesn't have system messages like OpenAI)
    full_message = f"{system_message}\n\nUser question: {message}"
    
    try:
        response = chat.send_message(full_message)
        return response.text
    except Exception as e:
        return f"[Gemini API error] {str(e)}"

def classify_query_type(query: str, model: str = "gemini-1.5-flash") -> str:
    if not GOOGLE_API_KEY:
        return "OTHER"  # Fallback to rule-based classification
    
    # Enhanced rule-based classification first
    query_lower = query.lower()

    top_n_patterns = [
        r'top\s+\d+',
        r'best\s+\d+',
        r'highest.*by',
        r'most.*by',
        r'top.*players.*by',
        r'best.*players.*by'
    ]
    for pattern in top_n_patterns:
        if re.search(pattern, query_lower):
            return "TOP_N"

    
    # Strong COUNT indicators
    count_patterns = [
        r'how many',
        r'number of',
        r'count of',
        r'total.*players?',
        r'how.*players?.*are',
        r'players?.*with.*\d+\+',  # "players with 500+ minutes"
        r'players?.*more than.*\d+',  # "players more than 500 minutes"
        r'players?.*at least.*\d+',   # "players at least 500 minutes"
        r'players?.*over.*\d+',       # "players over 500 minutes"
        r'players?.*under.*\d+',      # "players under 500 minutes"
        r'players?.*less than.*\d+',  # "players less than 500 minutes"
    ]
    
    for pattern in count_patterns:
        if re.search(pattern, query_lower):
            return "COUNT"
    
    # Add to the count_patterns list in classify_query_type
    report_patterns = [
        r'\w+\s+report',
        r'report\s+on\s+\w+',
        r'profile\s+of\s+\w+',
        r'analysis\s+of\s+\w+'
    ]

    for pattern in report_patterns:
        if re.search(pattern, query_lower):
            return "PLAYER_REPORT"
    # Add to the classify_query_type function, after the report_patterns check:
    define_patterns = [
        r'define\s+',
        r'explain\s+',
        r'\s+definition',
        r'\s+meaning',
        r'what\s+does\s+.*\s+mean',
        r'how\s+is\s+.*\s+calculated'
    ]

    for pattern in define_patterns:
        if re.search(pattern, query_lower):
            return "STAT_DEFINITION"
    # Strong TOP_N indicators
    if any(word in query_lower for word in ['top', 'best', 'highest', 'most']) and not any(word in query_lower for word in ['how many', 'number of']):
        return "TOP_N"
    
    prompt = f"""
Classify this football analytics question as one of:
- COUNT: User wants a count of items (e.g., 'How many players in Serie A under 23?')
- TOP_N: User wants the top N by a stat (e.g., 'Top 5 in Premier League by assists')
- LIST: User wants a list of items (e.g., 'List all defenders in Bundesliga')
- FILTER: User wants to filter by a condition (e.g., 'Players in La Liga with more than 10 goals')
- OTHER: Anything else

Query: "{query}"
Type (just the type, nothing else):
"""
    
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        content = response.text.strip().upper()
        # Only keep the first word (in case LLM adds explanation)
        return content.split()[0]
    except Exception as e:
        return "OTHER" 